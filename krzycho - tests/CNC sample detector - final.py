import cv2
import numpy as np
import serial
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk
import csv
import os
from pypylon import pylon
import math
import re

# === Global variables ===
live_frame = None
live_frame_lock = False  # Lock to prevent GUI updates while processing
running = True
cnc_serial = None
move_step = 0.1   # Default step is 0.1 mm
camera_lock = threading.Lock()
log_file = "focus_log.csv"
calibration_file = "camera_calibration.txt"
status_paused = False
grbl_is_paused = False
btn_pause_resume = None

last_position = "?.?,?,?"
position_lock = threading.Lock()

frame_width = 700
frame_height = 700

manual_buttons = []
step_buttons_refs = []
video_label = None

actual_camera = None
camera = None
microscope = None
view_thread = None

image_width = 1400  # Width of the camera image
image_height = 1400  # Height of the camera image

# Correction matrix for perspective transformation
correction_matrix = np.array([
    [ 1.01692104e+00, -4.50384611e-02, -7.96564445e+01],
    [-5.89281626e-03,  1.00546177e+00, -1.28204110e+02],
    [-5.68544053e-07, -5.31396415e-05,  1.00000000e+00]
    ])

autofocus_steps = [0.1, 0.01, 0.005, 0.001]  # Default autofocus steps

samples = ["A1", "A2", "B1", "B2"]

default_Z_position = -55.9  # Default Z position if not available
last_Z_position = default_Z_position  # Last known Z position

A1abs_position = "X-197.0 Y-210.0 Z"+str(default_Z_position)   # Pozice A1 v MPos
A2abs_position = "X-153.0 Y-210.0 Z"+str(default_Z_position)  # Pozice A2 v MPos
B1abs_position = "X-197.0 Y-165.0 Z"+str(default_Z_position)  # Pozice B1 v MPos
B2abs_position = "X-153.0 Y-165.0 Z"+str(default_Z_position)  # Pozice B2 v MPos

pin_camera_position ="X-215.930 Y-229.450 Z"+str(default_Z_position) # Pin for camera control (if needed, not used in this script)
pin_microscope_position = "X-157.920 Y-174.750 Z-54.660"  # Pin for microscope control (if needed, not used in this script)
# Calibration of the camera for GRBL
# correction factors to mm for GRBL - need to measure by tape
Xmm = 38.0  # camera size x
Ymm = 38.0  # camera size y

microscope_offset_x = -128.433  # microscope offset x in mm
microscope_offset_y = -144.920  # microscope offset y in mm
camera_offset_x = -186.275  # camera offset x in mm
camera_offset_y = -199.863  # camera offset y in mm

# offset center of the camera to microscope center in mm
offXmm = microscope_offset_x - (camera_offset_x) - (Xmm/2)
offYmm = microscope_offset_x - (camera_offset_x) - (Ymm/2)
offXmm = 38.6
offYmm = 34.77

print(f"Offset X: {offXmm:.3f} mm, Offset Y: {offYmm:.3f} mm")

# fXmm and fYmm are the factors to convert pixel coordinates to mm
fXmm = Xmm / image_width # fXmm = Xmm/camera.Width.Value  
fYmm = Ymm / image_height # fYmm = Ymm/camera.Height.Value

# Convert circle to polygon points for linear move in GRBL
def circle_to_polygon_with_segment_length(x, y, r, segment_length):
    """
    Convert a circle to a polygon approximation with roughly equal segment lengths.

    Parameters:
    - x, y: center of the circle
    - r: radius of the circle
    - segment_length: desired length of each polygon edge

    Returns:
    - List of (x, y) tuples representing the polygon vertices
    """
    # Calculate number of segments based on desired length
    circumference = 2 * math.pi * r
    precision = max(3, int(circumference / segment_length))  # at least a triangle

    # Generate points
    return [
        (
            x + r * math.cos(2 * math.pi * i / precision),
            y + r * math.sin(2 * math.pi * i / precision)
        )
        for i in range(precision)
    ]

# Example usage
# polygon = circle_to_polygon_with_segment_length(0, 0, 10, segment_length=2)
# print(polygon)

# === Logging of position and sharpness of the image ===
def log_position_and_sharpness(sharpness):
    with position_lock:
        pos = last_position
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), pos, sharpness])
        print(f"📄 Log: {pos}, Ostrost: {sharpness}")

if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Čas", "Pozice (X,Y,Z)", "Ostrost"])

def init_grbl():
    global cnc_serial

    def extract_mpos(status_line):
        match = re.search(r"MPos:([-\d\.]+),([-\d\.]+),([-\d\.]+)", status_line)
        if match:
            x, y, z = match.groups()
            return float(x), float(y), float(z)
        else:
            return None

    # Open port without resetting GRBL (critical to prevent resetting MPos)
    cnc_serial = serial.Serial()
    # on MacOSX Open Terminal and type:
    # ls /dev/tty.*
    # or ls /dev/cu.usb*
    # e.g /dev/tty.usbserial-1420 or /dev/tty.usbmodem1101
    # cnc_serial.port = '/dev/cu.usbmodem14101'
    cnc_serial.port = 'COM5'
    cnc_serial.baudrate = 115200
    cnc_serial.timeout = 1
    cnc_serial.dtr = False  # prevent soft reset
    cnc_serial.rts = False
    cnc_serial.open()

    time.sleep(2)  # wait for GRBL to be ready
    cnc_serial.reset_input_buffer()

    # Send status query
    cnc_serial.write(b'?\n')
    time.sleep(0.1)

    response = cnc_serial.readline().decode().strip()
    print("Raw GRBL response:", response)

    mpos = extract_mpos(response)
    if mpos:
        print(f"Machine Position (MPos): X={mpos[0]:.3f}, Y={mpos[1]:.3f}, Z={mpos[2]:.3f}")
    else:
        print("MPos not found — make sure $10=1 or $10=3 is set in GRBL.")
    
    return cnc_serial

def send_gcode(command):
    try:
        cnc_serial.write((command.strip() + '\n').encode())
        time.sleep(0.05)
        while cnc_serial.in_waiting:
            cnc_serial.readline()
    except serial.SerialTimeoutException:
        print("⚠️  Serial write timeout. Command lost:", command)
    except Exception as e:
        print("⚠️  Serial error:", e)

def move_axis(axis, distance): # Move axis relative to current position
    send_gcode(f"G21 G91 G1 {axis}{distance:.3f} F250")

def grbl_toggle_pause_resume():
    global grbl_is_paused
    try:
        if grbl_is_paused:
            cnc_serial.write(b'~')
            cnc_serial.flush()
            print("▶️  GRBL Resume sent")
        else:
            cnc_serial.write(b'!')
            cnc_serial.flush()
            print("⏸️  GRBL Pause sent")
        grbl_is_paused = not grbl_is_paused
        # Update the button text
        btn_pause_resume.config(text=("Pokračovat (~)" if grbl_is_paused else "Pozastavit (!)"))
    except Exception as e:
        print("⚠️  Error toggling Pause/Resume:", e)

def grbl_abort():
    try:
        cnc_serial.write(b'\x18')  # Ctrl+X = Soft Reset
        cnc_serial.flush()
        print("🛑 GRBL Abort (Soft Reset) sent")
    except Exception as e:
        print("⚠️  Error sending Abort:", e)

def grbl_clear_alarm():
    send_gcode("$X")
    print("🧹 GRBL Clear Alarm ($X) sent")

def wait_for_grbl_idle():
    global cnc_serial

    if cnc_serial is not None:
        t1 = time.time()
        while time.time() - t1 < 60:           
            # Explicitly request status before checking buffer
            try:
                cnc_serial.write(b'?\n')
                cnc_serial.flush()
            except Exception:
                pass
            t0 = time.time()
            line = b""
            while time.time() - t0 < 0.5:
                if cnc_serial.in_waiting:
                    line = cnc_serial.readline()
                    try:
                        decoded = line.decode(errors='ignore').strip()
                        if "Idle" in decoded:
                            return True
                    except Exception:
                        pass
                time.sleep(0.01)
            time.sleep(0.5)

        print("⚠️  GRBL did not become idle within 60 seconds")


def grbl_home():
    global cnc_serial, last_position
    try:
        send_gcode("$H")  # Send GRBL Home command
        print("🏠 GRBL Home sent")
    except Exception as e:
        print("⚠️  Error sending Home:", e)
    # Wait until homing is finished
    while True:
        try:
            cnc_serial.write(b'?')
            cnc_serial.flush()
            t0 = time.time()
            line = b""
            while time.time() - t0 < 0.5:
                if cnc_serial.in_waiting:
                    line = cnc_serial.readline()
                    break
                time.sleep(0.01)
            decoded = line.decode(errors='ignore').strip()
            if "Idle" in decoded or "Run" in decoded:
                break
        except Exception:
            pass
        time.sleep(0.2)

    #send_gcode("G10 P0 L20 X0 Y0 Z0")  # Set current position as zero
    # Set working position to be the same as MPos (WPos = MPos)
    update_position_blocking()
    with position_lock:
        mpos = last_position
    try:
        x, y, z = [float(val) for val in mpos.split(',')]
        send_gcode(f"G10 L20 P1 X{x:.3f} Y{y:.3f} Z{z:.3f}")
        print(f"Set WPos to MPos: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    except Exception as e:
        print("Failed to set WPos to MPos:", e)



def grbl_init_position():
    try:
        grbl_abort()  # Abort any running job
        grbl_clear_alarm()  # Clear any alarms
        
        send_gcode("G21")  # Set units to mm
        send_gcode("G90")  # Set to absolute positioning
        
        send_gcode("M3 S750")  # Start laser/lamp 75%
        send_gcode("G1 F1000")  # Set feed rate
        
    except Exception as e:
        print("⚠️  Error initializing position:", e)



def init_camera():
    global camera, microscope
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise Exception("Kamera Basler nenalezena!")
    # Initialize both cameras and return them for later switching
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    print("Kamera", camera.GetDeviceInfo().GetModelName())
    camera.Open()
    camera.GainAuto.SetValue("Off")
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTimeAbs.Value = 20000  # in microseconds
    camera.GainRaw.Value = 0
    camera.ReverseX.Value = True
    camera.ReverseY.Value = True
    camera.Width.SetValue(1500)
    camera.Height.SetValue(1500)
    camera.OffsetX.SetValue(282)
    camera.OffsetY.SetValue(22)
    # camera.OffsetX.Value = 0
    # camera.OffsetY.Value = 0
    # camera.Width.Value = camera.Width.Max
    # camera.Height.Value = camera.Height.Max

    print(f"Camera resolution set to {camera.Width.Value}x{camera.Height.Value}")   
    microscope = None
    if len(devices) > 1:
        microscope = pylon.InstantCamera(tl_factory.CreateDevice(devices[1]))
        print("Microskop", microscope.GetDeviceInfo().GetModelName())
        microscope.Open()
        # microscope.GainAuto.SetValue("Continuous")
        # microscope.ExposureAuto.SetValue("Continuous")
        microscope.GainAuto.SetValue("Off")
        microscope.ExposureAuto.SetValue("Off")
        microscope.ExposureTimeAbs.Value = 3000  # in microseconds
        microscope.GainRaw.Value = 0
        # microscope.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    return camera, microscope

def get_image():
    global actual_camera, camera_lock, correction_matrix
    
    with camera_lock:
        if actual_camera is None or not actual_camera.IsGrabbing():
            return None
        try:
            result = actual_camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            if result is not None and result.GrabSucceeded():
                img = result.Array
                # img = cv2.warpPerspective(image, correction_matrix, (width, height))
                result.Release()
                return img
            if result is not None:
                result.Release()
        except Exception as e:
            # Optionally print or log the error
            pass
        return None
    


def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def request_status():
    try:
        cnc_serial.write(b'?')
        cnc_serial.flush()
    except serial.SerialTimeoutException:
        print("⚠️  Serial write timeout (request_status)")
    except Exception as e:
        print("⚠️  Serial error in request_status:", e)

def update_cnc_position_loop():
    global running, last_position, status_paused
    while running:
        if status_paused:
            time.sleep(0.1)
            continue
        request_status()
        t0 = time.time()
        line = b""
        while time.time() - t0 < 0.5:
            if cnc_serial.in_waiting:
                line = cnc_serial.readline()
                break
            time.sleep(0.01)
        try:
            decoded = line.decode(errors='ignore').strip()
            if 'MPos:' in decoded:
                for part in decoded.split('|'):
                    if part.startswith('MPos:'):
                        mpos = part[5:].strip()
                        with position_lock:
                            last_position = mpos
                        break
        except Exception:
            pass
        time.sleep(0.1)

def update_position_blocking():
    global cnc_serial, last_position, position_lock

    try:
        cnc_serial.write(b'?')
        cnc_serial.flush()
    except serial.SerialTimeoutException:
        print("⚠️  Serial write timeout (update_position_blocking)")
        return None
    except Exception as e:
        print("⚠️  Serial error in update_position_blocking:", e)
        return None
    t0 = time.time()
    while time.time() - t0 < 0.5:
        if cnc_serial.in_waiting:
            line = cnc_serial.readline()
            # print("Received:", line.decode().strip())
            try:
                decoded = line.decode(errors='ignore').strip()
                if 'MPos:' in decoded:
                    for part in decoded.split('|'):
                        if part.startswith('MPos:'):
                            mpos = part[5:].strip()
                            with position_lock:
                                last_position = mpos
                            return mpos
            except Exception:
                pass
        time.sleep(0.01)
    return None

def live_view_loop():
    global running, live_frame, live_frame_lock, frame_height, frame_width, image_width, image_height
    while running:
        if actual_camera is not None and actual_camera.IsGrabbing():
            img = get_image()
            if img is not None and live_frame_lock is False:
                if actual_camera == microscope:
                    # For microscope, we don't apply correction matrix
                    live_frame = img
                    frame_height = int(microscope.Height.Max / 2.5)
                    frame_width = int(microscope.Width.Max / 2.5)
                else:
                    live_frame = cv2.warpPerspective(img, correction_matrix, (image_width, image_height))
                    frame_height = int(image_height / 2)
                    frame_width = int(image_width / 2)
                # live_frame = img
        time.sleep(0.01)


def create_gui():
    global video_label, actual_camera, cnc_serial, step_phases

    def run_autofocus():
        # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
        for btn in manual_buttons:
            btn.config(state="disabled")
        threading.Thread(target=autofocus_with_gui_enable).start()

    def run_fine_focus():
        # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
        for btn in manual_buttons:
            btn.config(state="disabled")
        threading.Thread(target=fine_focus_with_gui_enable).start()

    def fine_focus_with_gui_enable():
        try:
            update_position_blocking()
            with position_lock:
                mpos = last_position
            try:
                current_z = float(mpos.split(',')[2])
            except Exception:
                current_z = 0.0
            fine_focus(current_z, step=0.001, search_range=0.01)
        finally:
            # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
            for btn in manual_buttons:
                btn.config(state="normal")

    def autofocus_with_gui_enable():
        global autofocus_steps
        try:
            autofocus(autofocus_steps)
        finally:
            # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
            for btn in manual_buttons:
                btn.config(state="normal")

    def log_current_position():
        img = get_image()
        if img is not None:
            sharpness = round(compute_sharpness(img), 2)
            with position_lock:
                pos = last_position
            log_position_and_sharpness(sharpness)
            print(f"Aktuální pozice: {pos}")

    def run_grbl_home():
        # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
        for btn in manual_buttons:
            btn.config(state="disabled")
        threading.Thread(target=grbl_home_with_gui_enable).start()

    def grbl_home_with_gui_enable():
        try:
            grbl_home()
        finally:
            # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
            for btn in manual_buttons:
                btn.config(state="normal")
    
    def run_find_samples():
        # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
        for btn in manual_buttons:
            btn.config(state="disabled")
        threading.Thread(target=find_samples_with_gui_enable).start()
    
    def find_samples_with_gui_enable():
        try:
            find_samples()
        finally:
            # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
            for btn in manual_buttons:
                btn.config(state="normal")

    def run_get_microscope_images():
        # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
        for btn in manual_buttons:
            btn.config(state="disabled")
        threading.Thread(target=get_microscope_images_with_gui_enable).start()
    
    def get_microscope_images_with_gui_enable():
        try:
            get_microscope_images()
        finally:
            # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
            for btn in manual_buttons:
                btn.config(state="normal")
    def run_calibrate_camera():
        # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
        for btn in manual_buttons:
            btn.config(state="disabled")
        threading.Thread(target=calibrate_camera_with_gui_enable).start()
    def calibrate_camera_with_gui_enable():
        try:
            calibrate_camera()
        finally:
            # for btn in manual_buttons + [b for b, _ in step_buttons_refs]:
            for btn in manual_buttons:
                btn.config(state="normal")    

    root = tk.Tk()
    root.title("CNC Ovládání")

    tk.Label(root, text="Manuální řízení CNC").grid(row=0, column=1, columnspan=2)
    btn_up = tk.Button(root, text="↑", width=5, command=lambda: move_axis('Y', -move_step))
    btn_up.grid(row=2, column=2, padx=5, pady=5)
    btn_left = tk.Button(root, text="←", width=5, command=lambda: move_axis('X', -move_step))
    btn_left.grid(row=3, column=1, padx=5, pady=5)
    btn_right = tk.Button(root, text="→", width=5, command=lambda: move_axis('X', move_step))
    btn_right.grid(row=3, column=3, padx=5, pady=5)
    btn_down = tk.Button(root, text="↓", width=5, command=lambda: move_axis('Y', +move_step))
    btn_down.grid(row=4, column=2, padx=5, pady=5)
    btn_zplus = tk.Button(root, text="Z+", width=5, command=lambda: move_axis('Z', move_step))
    btn_zplus.grid(row=2, column=4, padx=5, pady=5)
    btn_zminus = tk.Button(root, text="Z-", width=5, command=lambda: move_axis('Z', -move_step))
    btn_zminus.grid(row=4, column=4, padx=5, pady=5)
    # manual_buttons.extend([btn_up, btn_left, btn_right, btn_down, btn_zplus, btn_zminus])

    tk.Label(root, text="Krok [mm]:").grid(row=0, column=5, sticky="w")
    step_buttons = [
        (10, "10"),
        (1, "1"),
        (0.1, "0.1"),
        (0.01, "0.01"),
        (0.001, "0.001")
    ]
    def set_step(value):
        global move_step
        move_step = value
        for btn, val in step_buttons_refs:
            btn.config(relief=("sunken" if move_step == val else "raised"))
    step_buttons_refs.clear()
    for idx, (val, text) in enumerate(step_buttons):
        btn = tk.Button(root, text=text, width=5, command=lambda v=val: set_step(v))
        btn.grid(row=1+idx, column=5, padx=2, pady=5)
        step_buttons_refs.append((btn, val))
    for btn, val in step_buttons_refs:
        if val == move_step:
            btn.config(relief="sunken")

    global btn_pause_resume
    
    btn_home = tk.Button(root, text="Domů ($H)", width=14, command=run_grbl_home)
    btn_home.grid(row=7, column=1, padx=2, pady=2)

    btn_pause_resume = tk.Button(root, text="Pause (!)", width=14, command=grbl_toggle_pause_resume)
    btn_pause_resume.grid(row=7, column=2, padx=2, pady=2)


    btn_abort = tk.Button(root, text="Přerušit (Soft Reset)", width=14, command=grbl_abort)
    btn_abort.grid(row=7, column=3, padx=2, pady=2)    

    btn_clear_alarm = tk.Button(root, text="Zrušit Alarm ($X)", width=14, command=grbl_clear_alarm)
    btn_clear_alarm.grid(row=7, column=4, padx=2, pady=2)
 
    manual_buttons.extend([btn_home]) # keep pause_resume, abort, clear_alarm always enabled

    btn_focus = tk.Button(root, text="Zaostřit", width=14, command=run_autofocus)
    btn_focus.grid(row=9, column=1, padx=2, pady=5)
    btn_fine_focus = tk.Button(root, text="Jemné ostření", width=14, command=run_fine_focus)
    btn_fine_focus.grid(row=9, column=2, padx=2, pady=5)
    btn_log = tk.Button(root, text="Ulož pozici", width=14, command=log_current_position)
    btn_log.grid(row=9, column=3, padx=2, pady=5)

    def check_and_switch_camera():
        global actual_camera, camera, microscope
        if actual_camera == microscope:
            switch_camera(camera)
        else:
            switch_camera(microscope)
    
    btn_switch_camera = tk.Button(root, text="Přepnout kameru", width=14, command=lambda: threading.Thread(target=check_and_switch_camera, daemon=True).start())
    btn_switch_camera.grid(row=9, column=4, padx=2, pady=5)

    btn_samples = tk.Button(root, text="Najdi vzorky", width=14, command=run_find_samples)
    btn_samples.grid(row=10, column=1, padx=2, pady=5)
    btn_microscope = tk.Button(root, text="Snímky z mikroskopu", width=14, command=run_get_microscope_images)
    btn_microscope.grid(row=10, column=2, pady=5)
    btn_calibrate_camera = tk.Button(root, text="Kalibrovat kameru", width=14, command=run_calibrate_camera)
    btn_calibrate_camera.grid(row=10, column=3, pady=5)
    btn_end = tk.Button(root, text="Konec", width=14, command=shutdown)
    btn_end.grid(row=10, column=4, pady=5)

    manual_buttons.extend([btn_focus, btn_fine_focus, btn_log, btn_samples, btn_microscope])

    

    video_label = tk.Label(root)
    video_label.grid(row=0, column=6, rowspan=15, padx=10, pady=10)
    return root

def update_tk_live_view():
    global running, live_frame, live_frame_lock, video_label, frame_height, frame_width, last_position
    if running and live_frame is not None:
        frame = cv2.resize(live_frame, (frame_width, frame_height))
        sharpness = round(compute_sharpness(live_frame), 2)
        with position_lock:
            pos = last_position
        # Ensure the image is in color (BGR) for drawing colored shapes/text
        if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, f"Sharpness: {sharpness:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Pos: {pos}", (frame.shape[1]-360, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # Draw a cross in the middle of the frame
        center_x = frame.shape[1] // 2
        center_y = frame.shape[0] // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 1)
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    if running:
        video_label.after(30, update_tk_live_view)

def fine_focus(center_z, step=0.001, search_range=0.02):
    global live_frame, last_Z_position
    steps = int(search_range / step)
    best_sharpness = -1
    best_image = None
    best_z = None

    update_position_blocking()
    with position_lock:
        mpos = last_position
    try:
        current_z = float(mpos.split(',')[2])
        last_current_z = current_z
    except Exception:
        current_z = last_current_z

    start_z = center_z - (steps // 2) * step
    move_axis('Z', start_z - current_z)
    # wait_for_grbl_idle
    time.sleep(1.5)  # Wait for the move to complete
    update_position_blocking()

    for i in range(steps):
        img = get_image()
        if img is None:
            continue
        with position_lock:
            mpos = last_position
        try:
            z_here = float(mpos.split(',')[2])
            last_Z_position = z_here  # Update last known Z position
        except Exception:
            z_here = last_Z_position  # Default Z position if not available

        sharpness = round(compute_sharpness(img), 2)
        print(f"[FINE FOCUS] Krok {i}, Z={z_here:.3f}, MPos={mpos}, Ostrost: {sharpness:.2f}")
        # annotated_img = img.copy()
        # cv2.putText(annotated_img, f"Sharpness: {sharpness:.2f}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # cv2.putText(annotated_img, f"Pos: {mpos}", (annotated_img.shape[1]-360, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # live_frame = annotated_img
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_image = img.copy()
            best_z = z_here
        if i < steps - 1:
            move_axis('Z', step)
            #wait_for_grbl_idle()
            time.sleep(1.0)
            update_position_blocking()
    update_position_blocking()
    with position_lock:
        mpos = last_position
    try:
        current_z = float(mpos.split(',')[2])
    except Exception:
        current_z = last_Z_position
    delta_z = best_z - current_z
    move_axis('Z', delta_z)
    #wait_for_grbl_idle()
    time.sleep(1.5)  # Wait for the move to complete
    update_position_blocking()
    #if best_image is not None:
        # cv2.imwrite("best_focus.png", best_image)
        # print("📷 Uložen nejlepší snímek jako 'best_focus.png'")

    return best_z, best_sharpness

def autofocus(autofocus_steps):
    global status_paused, actual_camera, cnc_serial, last_Z_position
    status_paused = True
    try:
        print("Fáze 1 – Inteligentní ostření...")
        if autofocus_steps is None:
            autofocus_steps = [0.1, 0.01, 0.002]  # Default steps if not provided
        start_direction = 1

        update_position_blocking()
        with position_lock:
            mpos = last_position
        try:
            z_current = float(mpos.split(',')[2])
            last_Z_position = z_current  # Update last known Z position
        except Exception:
            z_current = last_Z_position  # Default Z position if not available
        
        for step_idx, step_size in enumerate(autofocus_steps):
            
            start_direction = 1
            update_position_blocking()

            print(f"--- Fáze {step_idx+1}: krok {step_size} mm ---")
            max_sharpness = -1
            max_z = z_current
            direction = start_direction if step_idx == 0 else -direction
            decreasing_count = 0
            i = 0
            while True:
                move_axis('Z', direction * step_size)
                # Wait for GRBL to become idle
                #wait_for_grbl_idle()
                time.sleep(0.8)  # Wait for the move to complete
                update_position_blocking()
                with position_lock:
                    mpos = last_position
                try:
                    z_here = float(mpos.split(',')[2])
                    last_Z_position = z_here  # Update last known Z position
                except Exception:
                    z_here = last_Z_position  # Default Z position if not available
                img = get_image()
                if img is None:
                    continue
                sharpness = round(compute_sharpness(img), 2)
                print(f"[AUTOFOCUS] Krok {i}, Z={z_here:.3f}, MPos={mpos}, Ostrost={sharpness:.2f}")
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    max_z = z_here
                    decreasing_count = 0
                else:
                    decreasing_count += 1
                if decreasing_count >= 4:
                    print(f"Lokální maximum pro krok {step_size} mm: Ostrost={max_sharpness}, Z={max_z:.3f}")
                    break
                i += 1
                if i > 50:
                    break
            update_position_blocking()
            with position_lock:
                mpos = last_position
            try:
                z_here = float(mpos.split(',')[2])
            except Exception:
                z_here = default_Z_position
            dz = max_z - z_here
            move_axis('Z', dz)
            time.sleep(1.0)  # Wait for the move to complete
            update_position_blocking()
            z_current = max_z
        # print(f"Fáze 2 – Jemné ostření (okolo Z={z_current:.3f})...")
        # best_z_final, final_sharpness = fine_focus(z_current, step=0.001, search_range=0.01)
        update_position_blocking()
        with position_lock:
            pos = last_position
        # print(f"✅ Nejlepší ostrost: {final_sharpness:.2f} na pozici Z = {best_z_final:.3f} mm (MPos={pos})")
        print(f"✅ Nejlepší ostrost: {max_sharpness:.2f} na pozici Z = {max_z:.3f} mm (MPos={pos})")
    finally:
        status_paused = False

def switch_camera(device):
    global camera, microscope, actual_camera, live_frame_lock

    if device is None:
        print("⚠️  Kamera není dostupná!")
        return

    if actual_camera is not device:
        live_frame_lock = True
        try:
            # Stop grabbing on the previous camera
            if actual_camera is not None:
                try:
                    if actual_camera.IsGrabbing():
                        actual_camera.StopGrabbing()
                except Exception:
                    pass
            print(f"Switching to {device.GetDeviceInfo().GetModelName()}")
            actual_camera = device
            # Start grabbing on the new camera
            if not actual_camera.IsGrabbing():
                actual_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            # Wait for at least one frame to be available
            timeout = time.time() + 5
            while not actual_camera.IsGrabbing():
                if time.time() > timeout:
                    print("⚠️  Kamera se nespustila včas.")
                    break
                time.sleep(0.05)
        finally:
            live_frame_lock = False

def calibrate_camera():
    import tkinter.messagebox
    global actual_camera, camera, running, live_frame, live_frame_lock, correction_matrix, image_width, image_height, offXmm, offYmm, Xmm, Ymm, camera_lock, last_Z_position
    switch_camera(camera)  # Ensure we are using the main camera
    cv2.pts_src = []
    send_gcode("G21 G90 G1 "+A1abs_position+" F1000")  # Move to A1 position
    wait_for_grbl_idle()
    time.sleep(1.0)  # Wait for the move to complete

    while actual_camera.IsGrabbing():
        image=get_image()  # Trigger camera to start grabbing
        # Ensure the image is in color (BGR) for drawing colored shapes/text
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # --- Perspective Correction with Mouse Clicks ---
        # Allow user to select four points by clicking on the image window

        

        def select_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(cv2.pts_src) < 4:
                cv2.pts_src.append([x, y])

        # Create window and resize to fit the screen
        cv2.namedWindow("Kliknete mysi na 4 body v rozich", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Kliknete mysi na 4 body v rozich", int(camera.Width.Max/2), int(camera.Height.Max/2))
        cv2.setMouseCallback("Kliknete mysi na 4 body v rozich", select_point)

        img_for_selection = image.copy()
        for pt in getattr(cv2, 'pts_src', []):
            cv2.circle(img_for_selection, tuple(pt), 5, (0, 0, 255), -1)
        cv2.imshow("Kliknete mysi na 4 body v rozich", img_for_selection)

        if len(cv2.pts_src) < 4:
            cv2.waitKey(1)
            continue
        pts_src = np.float32(cv2.pts_src[:4])


        #force width and height to be a fixed size square
        image_width = 1400
        image_height = 1400
        pts_dst = np.float32([
            [0, 0],
            [image_width - 1, 0],
            [image_width - 1, image_height - 1],
            [0, image_height - 1]
        ]) 
        
        # Compute the perspective transform matrix and apply it
        correction_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        # Save calibration parameters to CSV
        with open(calibration_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['width', 'height'])
            writer.writerow([image_width, image_height])
            writer.writerow(['M'])
            for row in correction_matrix:
                writer.writerow(row)
        print(f"Calibration parameters saved to {calibration_file}")
        
        print(f"Calibrating camera: width={image_width}, height={image_height}")
        print("Source points (click on the image):", pts_src)
        print("Destination points (straightened rectangle):", pts_dst)
        print("Transform matrix M:\n", correction_matrix)
    
        
        # if img_for_selection is not None and correction_matrix is not None and width > 0 and height > 0:
        #     warped = cv2.warpPerspective(img_for_selection, correction_matrix, (width, height))
        #     # Show the perspective-corrected image in a separate window
        #     cv2.namedWindow("Corrected Image", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("Corrected Image", int(width/2), int(height/2))
        #     cv2.imshow("Corrected Image", warped)
        #     while True:
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        cv2.destroyAllWindows()
        
        answer = tkinter.messagebox.askquestion("Kalibrace", "Je kalibrace v pořádku?")
        if answer == "no":
            cv2.pts_src = []
            # Repeat the calibration process from line 862 (start of this function's while loop)
            continue
        break

    # Find center of the pin in the camera
    
    send_gcode("G21 G90 G1 "+pin_camera_position+" F1000")  # Move to A1 position
    wait_for_grbl_idle()
    
    while True:
        img_for_selection = live_frame.copy()
        if img_for_selection is not None:

            resized_img = cv2.resize(img_for_selection, (img_for_selection.shape[1] // 2, img_for_selection.shape[0] // 2))
            # Convert grayscale image to BGR color if needed
            if len(resized_img.shape) == 2 or resized_img.shape[2] == 1:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
            # Draw a cross in the middle of the microscope image
            # Draw a cross in the middle of the microscope image
            center_x = resized_img.shape[1] // 2
            center_y = resized_img.shape[0] // 2
            cv2.line(resized_img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 1)
            cv2.line(resized_img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 1)
            cv2.imshow("Najedte krizkem na stred pinu a zmacknete q", resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Get actual X, Y from MPos and save to calibration file
    update_position_blocking()
    with position_lock:
        mpos = last_position
    try:
        mpos_x, mpos_y, mpos_z = [float(val) for val in mpos.split(',')]
    except Exception:
        mpos_x, mpos_y, mpos_z = 0.0, 0.0, 0.0

    # Append actual X, Y to calibration file
    with open(calibration_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['camera_offset_X', 'camera_offset_Y'])
        writer.writerow([mpos_x, mpos_y])
    print(f"Offset camera X, Y from MPos saved to {calibration_file}: X={mpos_x}, Y={mpos_y}")
    cv2.destroyAllWindows()
    camera_offset_x = mpos_x
    camera_offset_y = mpos_y
    print("Střed pinu nalezen. Pokračuji v kalibraci...")
        
    # Find center of the pin in the microscope camera
    print("Nyní přepínám na mikroskop...")
    send_gcode("G21 G90 G1 "+pin_microscope_position+" F1000")  # Move to A1 position
    switch_camera(microscope)  # Switch to microscope camera
    wait_for_grbl_idle()
    switch_camera(microscope)  # Switch to microscope camera
    time.sleep(1.0)  # Wait for the move to complete

    print("Stiskněte q pro ukončení kalibrace...")
    while True:
        img_for_selection = live_frame.copy()
        if img_for_selection is not None:
            resized_img = cv2.resize(img_for_selection, (img_for_selection.shape[1] // 2, img_for_selection.shape[0] // 2))
            # Convert grayscale image to BGR color if needed
            if len(resized_img.shape) == 2 or resized_img.shape[2] == 1:
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
            # Draw a cross in the middle of the microscope image
            center_x = resized_img.shape[1] // 2
            center_y = resized_img.shape[0] // 2
            cv2.line(resized_img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 1)
            cv2.line(resized_img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 1)
            cv2.imshow("Najedte krizkem na stred pinu a zmacknete q", resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Get actual X, Y from MPos and save to calibration file
    update_position_blocking()
    with position_lock:
        mpos = last_position
    try:
        mpos_x, mpos_y, mpos_z = [float(val) for val in mpos.split(',')]
    except Exception:
        mpos_x, mpos_y, mpos_z = 0.0, 0.0, 0.0

    # Append actual X, Y to calibration file
    with open(calibration_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['microscope_offset_X', 'microscope_offset_Y'])
        writer.writerow([mpos_x, mpos_y])
    print(f"Offset microscope X, Y from MPos saved to {calibration_file}: X={mpos_x}, Y={mpos_y}")
    microscope_offset_x = mpos_x
    microscope_offset_y = mpos_y
    cv2.destroyAllWindows()
    offXmm = microscope_offset_x - camera_offset_x - (Xmm / 2)
    offYmm = microscope_offset_y - camera_offset_y - (Ymm / 2)
    print(f"Offset X: {offXmm}, Offset Y: {offYmm}")
    print("Kalibrace ukončena.")
    

def find_samples():
    global view_thread, live_frame, live_frame_lock, camera, cnc_serial, samples, actual_camera, running, A1abs_position, A2abs_position, B1abs_position, B2abs_position, offXmm, offYmm, Xmm, Ymm, image_width, image_height
    
    switch_camera(camera)  # Ensure we are using the main camera for live view
                
    if cnc_serial is not None:
        
        for sample in samples:
            if sample == "A1":
                sample_position = A1abs_position
            elif sample == "A2":
                sample_position = A2abs_position
            elif sample == "B1":
                sample_position = B1abs_position
            elif sample == "B2":
                sample_position = B2abs_position

            print(f"🔍 Hledání vzorků v {sample} na pozici {sample_position}")
            
            send_gcode("G21 G90 G1 "+sample_position+" F1000")  # Move to A1 position
            wait_for_grbl_idle()
            time.sleep(1.0)  # Wait for the move to complete
                
            if live_frame is None:
                print("⚠️  Žádný živý snímek k analýze.")
                return
            
            # Make a copy for drawing
            frame_with_circles = live_frame.copy()

            # Ensure the image is in color (BGR) for drawing colored shapes/text
            if len(frame_with_circles.shape) == 2 or frame_with_circles.shape[2] == 1:
                frame_with_circles = cv2.cvtColor(frame_with_circles, cv2.COLOR_GRAY2BGR)

            gray = cv2.cvtColor(frame_with_circles, cv2.COLOR_BGR2GRAY) if len(frame_with_circles.shape) == 3 else frame_with_circles
            #gray = cv2.medianBlur(gray, 5)  # Apply median blur to reduce noise
            # Detect circles using Hough Transform (HOUGH_GRADIENT_ALT)
            # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, 10, param1=1300, param2=0.90, minRadius=90, maxRadius=600)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # # --- Step 3: Apply adaptive thresholding ---
            # adaptive_thresh = cv2.adaptiveThreshold(
            #     gray,
            #     255,
            #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #     cv2.THRESH_BINARY_INV,
            #     blockSize=5,
            #     C=2
            # )
            adaptive_thresh = cv2.threshold(
                blurred,
                125,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]    
            # --- Step 4: Use watershed to separate touching circles, then find contours ---
            circles = []

            # Noise removal
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)

            # Finding unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Mark the unknown region with zero
            markers[unknown == 255] = 0

            # Watershed
            markers = cv2.watershed(frame_with_circles, markers)

            # Find contours from separated regions
            for marker in np.unique(markers):
                if marker <= 1:
                    continue
                mask = np.zeros(gray.shape, dtype="uint8")
                mask[markers == marker] = 255
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area < 500:  # filter out small contours
                        continue
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if 0.7 < circularity < 1.3:  # adjust as needed for your images
                        (x, y), r = cv2.minEnclosingCircle(cnt)
                        x, y, r = int(x), int(y), int(r)
                        if 30 < r < 650:  # filter by radius
                            circles.append((x, y, r))
   
            if circles is not None:
                fXmm = Xmm / image_width # fXmm = Xmm/camera.Width.Value  
                fYmm = Ymm / image_height # fYmm = Ymm/camera.Height.Value
                
                circle_positions = []
                circle_positions.append(f";Sample {sample}")
                grbl_coordinates = []
                grbl_coordinates.append(f";Sample {sample}")
                # Add current absolute MPos position as a comment
                update_position_blocking()
                with position_lock:
                    mpos = last_position
                grbl_coordinates.append(f";Current MPos: {mpos}")

                for i, (x, y, r) in enumerate(circles):

                    string = str(i)+": "+str(x) + " " + str(y) + " " + str(r)
                    circle_positions.append(f"Circle {i}: center x:{x} y:{y} r:{r}")
                    

                    # Draw circles in image for reference
                    cv2.circle(frame_with_circles, (x,y), r, (0, 255, 200), 2)    # detected circle
                    cv2.circle(frame_with_circles, (x,y), 2, (0, 255, 200), 2)  # center of the circle
                    cv2.putText(frame_with_circles, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                    # Get current MPos X Y Z
                    try:
                        mpos_x, mpos_y, mpos_z = [float(val) for val in mpos.split(',')]
                    except Exception:
                        mpos_x, mpos_y, mpos_z = 0.0, 0.0, 0.0
                    abs_x = x * fXmm + offXmm + mpos_x
                    abs_y = y * fYmm + offYmm + mpos_y
                    abs_z = mpos_z  # Z position remains unchanged
                    # Prepare GRBL commands
                    grbl_coordinates.append(f";Circle {i}")  # move to circle center
                    grbl_coordinates.append(f";Center of the circle: x:{abs_x:.3f} y:{abs_y:.3f}")
                    grbl_coordinates.append(f"G21G90G1X{abs_x:.3f}Y{abs_y:.3f}Z{abs_z:.3f}M3S750F1000")  # move to center point of the circle for autofocus
                    grbl_coordinates.append(f";Commands for circle {i}")
                    
                    # segment_pixels = 100 # Number of pixels per segment for circle approximation
                    segment_pixels = 20

                    polygon = circle_to_polygon_with_segment_length(x, y, r-2, segment_pixels)  # polygon approximating a circle
                    calculated_polygon = np.uint16(np.around(polygon))
                    

                    for i, (x,y) in enumerate(calculated_polygon):
                        cv2.circle(frame_with_circles, (x,y), 20, (0, 100, 255), 2)
                        cv2.putText(frame_with_circles, str(i+1), (x-20, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255))
                        
                        abs_x = x * fXmm + offXmm + mpos_x
                        abs_y = y * fYmm + offYmm + mpos_y
                        abs_z = mpos_z  # Z position remains unchanged
                        # Prepare GRBL commands
                        grbl_coordinates.append(f"G21G90G1X{abs_x:.3f}Y{abs_y:.3f}")  # move to each point of the circle

                with open(sample+"_circle_positions.txt", "w") as f:
                    for line in circle_positions:
                        f.write(line+"\n")
                    f.close()
                with open(sample+"_grbl_circle_positions.gcode", "w") as f:
                    for line in grbl_coordinates:
                        f.write(line+"\n")
                    f.close()
                print(f"✅ Nalezeno {len(circles)} kruhů.")
                
                print(f"📄 Příkazy GRBL uloženy do '{sample}_grbl_circle_positions.gcode'.")
                cv2.imwrite(sample+"detected_circles.png", frame_with_circles)
                print(f"📷 Snímek s detekovanými kruhy uložen jako '{sample}detected_circles.png'.")

                

                # Show the result in the live view frame in the GUI
                t0 = time.time()
                while time.time() - t0 < 3.0:  # Show for 3 seconds
                    live_frame_lock = True
                    live_frame = frame_with_circles.copy()
                live_frame_lock = False

            else: print("No circles detected")


def get_microscope_images():
    global view_thread, live_frame, live_frame_lock, cnc_serial, actual_camera, camera, microscope, samples, Xmm, Ymm, offXmm, offYmm, fXmm, fYmm, A1abs_position, A2abs_position, B1abs_position, B2abs_position
    
    switch_camera(microscope)  # Ensure we are using the microscope camera for live view
    
    if live_frame is None:
        print("⚠️  Žádný snímek z mikroskopu.")
        return
    for sample in samples:
        with open(sample+"_grbl_circle_positions.gcode", "r") as f:
            grbl_commands = f.readlines()
        f.close()

        first_command = True
        circle = "0"  # Default circle number
        
        for command in grbl_commands: 

            command = command.strip()
            
            
            if command.startswith(";Circle "):
                circle = command.lstrip(';Circle ')
                print(f"🔍 Zpracování vzorku {sample}, kruh: {circle}")
                first_command = True
                i = 0

            if command.startswith("G21"): # and i < 3: # test only first 3 commands
                print(f"🔍 Zpracování příkazu: {command}")
                # Move to the position
                send_gcode(command)  # Start laser/lamp at 75% power
                wait_for_grbl_idle()
                time.sleep(1.0)
                if first_command: 
                    move_axis('Z', -0.4)  # go to position for autofocus 
                    time.sleep(0.5)
                    autofocus_steps = [0.025, 0.005, 0.002]  # Default autofocus steps
                    autofocus(autofocus_steps)  # Run autofocus at the first position
                    first_command = False
                # else: # next commands
                #     # move_axis('Z', -0.2)  # go to position for autofocus 
                #     # time.sleep(0.5)
                #     # autofocus_steps = [0.02, 0.01, 0.002]  # Default autofocus steps
                #     # autofocus(autofocus_steps)
                    
                #     update_position_blocking()
                #     with position_lock:
                #         mpos = last_position
                #     try:
                #         current_z = float(mpos.split(',')[2])
                #         last_Z_position = current_z  # Update last known Z position
                #     except Exception:
                #         current_z = last_Z_position  # Default Z position if not available
                    
                #     fine_focus(current_z, step=0.002, search_range=0.05)
                
                img = get_image()
                if img is not None:
                    cv2.imwrite(f"{sample}_C{circle}_{i}_{command.replace('G21G90','').replace('M3S750F1000','')}.png", img)
                    print(f"📷 Snímek z mikroskopu uložen jako '{sample}_C{circle}_{command.replace('G21G90','').replace('M3S750F1000','')}.png'")
                    i += 1
    print("✅ Všechny snímky z mikroskopu uloženy.")
   

def shutdown():
    global running
    running = False
    try:
        import tkinter
        for widget in tkinter._default_root.children.values():
            try:
                widget.destroy()
            except Exception:
                pass
        if tkinter._default_root:
            tkinter._default_root.destroy()
    except Exception:
        pass
    os._exit(0)

def main():
    global cnc_serial, camera, microscope, actual_camera, correction_matrix, image_width, image_height, offXmm, offYmm, Xmm, Ymm, fXmm, fYmm, last_Z_position, last_position, position_lock, status_paused, running
    cnc_serial = init_grbl()
    if os.path.exists(calibration_file):
        with open(calibration_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # print("Calibration file rows:", rows)  # Debug print
            # Accept both 6 or more rows (sometimes only 6 if no trailing newline)
            if len(rows) >= 6 and rows[0][0] == 'width' and rows[2][0] == 'M':
                image_width = int(rows[1][0])
                image_height = int(rows[1][1])
                correction_matrix = np.array([
                    [float(val) for val in rows[3]],
                    [float(val) for val in rows[4]],
                    [float(val) for val in rows[5]]
                ])
                print(f"Loaded calibration: width={image_width}, height={image_height}")
                print("Correction matrix:")
                print(correction_matrix)
                # Read camera_offset_x, camera_offset_y, microscope_offset_x, microscope_offset_y if present
                camera_offset_x = camera_offset_y = microscope_offset_x = microscope_offset_y = None
                for idx, row in enumerate(rows):
                    if row and row[0] == 'camera_offset_X':
                        try:
                            camera_offset_x = float(rows[idx+1][0])
                            camera_offset_y = float(rows[idx+1][1])
                        except Exception:
                            pass
                    if row and row[0] == 'microscope_offset_X':
                        try:
                            microscope_offset_x = float(rows[idx+1][0])
                            microscope_offset_y = float(rows[idx+1][1])
                        except Exception:
                            pass
                print(f"camera_offset_X: {camera_offset_x}, camera_offset_Y: {camera_offset_y}")
                print(f"microscope_offset_X: {microscope_offset_x}, microscope_offset_Y: {microscope_offset_y}")
                # offset center of the camera to microscope center in mm
                offXmm = microscope_offset_x - camera_offset_x - (Xmm/2)
                offYmm = microscope_offset_y - camera_offset_y - (Ymm/2)
                print(f"Offset X: {offXmm}, Offset Y: {offYmm}")
            else:
                print("⚠️  Calibration file format is invalid.")
    
    camera, microscope = init_camera()
    if camera is None:
        print("⚠️  Kamera nenalezena!")
        return
    else:
        actual_camera = camera
        actual_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    pos_thread = threading.Thread(target=update_cnc_position_loop, daemon=True)
    pos_thread.start()
    view_thread = threading.Thread(target=live_view_loop)
    view_thread.start()
    gui = create_gui()
    gui.after(100, update_tk_live_view)

    def home_and_init():
        global status_paused
        # Check if MPos is zero before homing
        update_position_blocking()
        with position_lock:
            mpos = last_position
        try:
            x, y, z = [float(val) for val in mpos.split(',')]
            if x == 0.0 and y == 0.0 and z == 0.0:
                print("Pozice MPos je (0,0,0) – Arduino bylo resetováno.")
                print("Homing a inicializace pozice...")
                grbl_home()
            else:
                print(f"Aktuální MPos: X={x}, Y={y}, Z={z}")
        except Exception:
            print("Nelze zjistit aktuální MPos.")
        status_paused = True
        # grbl_home()
        grbl_init_position()
        status_paused = False

    threading.Thread(target=home_and_init, daemon=True).start()

    gui.mainloop()
    
    running = False
    view_thread.join()
    actual_camera.StopGrabbing()
    actual_camera.Close()
    cnc_serial.close()

if __name__ == "__main__":
    main()
