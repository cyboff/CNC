import serial
import time
import re
import threading
from config import CNC_SERIAL_PORT, CNC_BAUDRATE
from core.logger import logger


cnc_serial = None  # globální proměnná pro sériový port


# === Sdílené proměnné ===

position_lock = threading.Lock()
grbl_status = "Idle"  # Stav GRBL, zda je Idle nebo ne
cnc_last_position = "0.001,0.002,0.003"

def init_grbl():
    global cnc_serial, cnc_last_position, grbl_status, position_lock

    cnc_serial = serial.Serial()
    cnc_serial.port = CNC_SERIAL_PORT
    cnc_serial.baudrate = CNC_BAUDRATE
    cnc_serial.timeout = 1
    cnc_serial.dtr = False
    cnc_serial.rts = False
    cnc_serial.open()

    time.sleep(2)
    cnc_serial.reset_input_buffer()

    def update_position():
        global cnc_last_position, grbl_status, position_lock

        try:
            cnc_last_position, grbl_status = grbl_update_position()
            print(f"[GRBL] Aktuální pozice: {cnc_last_position}, Stav: {grbl_status}")
        except:
            print("Failed to update position")
        threading.Timer(0.5, update_position).start()

    update_position() # spustí periodické aktualizace pozice

    try:

        if cnc_last_position != "0.000,0.000,0.000":
            x, y, z = [float(val) for val in cnc_last_position.split(",")]
            print(f"Machine Position (MPos): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
        else:
            print("MPos not found – trying to home the machine")
            grbl_clear_alarm()
            grbl_home()
    except:
        print("Failed to initialize GRBL")

def send_gcode(command: str):
    """
    Odeslání G-code příkazu do GRBL + vypsání odpovědi.
    """
    global cnc_serial

    if cnc_serial is None or not cnc_serial.is_open:
        print("[GRBL] Port není otevřený – inicializuji GRBL...")
        init_grbl()

    print(f"[GRBL] Posílám: {command}")
    cnc_serial.write((command + "\n").encode())

    while True:
        line = cnc_serial.readline().decode().strip()
        if not line:
            break
        print(f"[GRBL] Odpověď: {line}")
        if line == "ok" or line.startswith("error"):
            break


def move_axis(axis: str, value: float):
    """
    Relativní pohyb v jedné ose.
    """
    gcode = f"G91\nG0 {axis.upper()}{value:.3f}\nG90"
    send_gcode(gcode)

def grbl_home():
    """
    Spustí homing sekvenci ($H)
    """
    global cnc_serial, cnc_last_position, grbl_status, position_lock
    try:
        send_gcode("$H")
        print("🏠 GRBL Home sent")
    except Exception as e:
        print("⚠️  Error sending Home:", e)
        return

        # Počkej na konec homing sekvence
    grbl_wait_for_idle()
    try:
        x, y, z = [float(val) for val in cnc_last_position.split(",")]
        send_gcode(f"G10 L20 P1 X{x:.3f} Y{y:.3f} Z{z:.3f}")
        print(f"Nastaveno WPos na MPos: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    except Exception as e:
        print("Failed to set WPos to MPos:", e)

def grbl_clear_alarm():
    """
    Odblokuje ALARM stav ($X)
    """
    send_gcode("$X")

def grbl_abort():
    """
    Nouzové přerušení (ctrl-x)
    """
    global cnc_serial
    cnc_serial.write(b'\x18')  # Ctrl-X
    print("[GRBL] Abort odeslán (Ctrl-X)")

def grbl_toggle_pause_resume():
    """
    Přepne mezi pauzou a pokračováním (0x13 a 0x11)
    """
    global cnc_serial
    cnc_serial.write(b'\x13')  # Pause (XOFF)
    time.sleep(0.5)
    cnc_serial.write(b'\x11')  # Resume (XON)
    print("[GRBL] Pauza/Resume přepnuto")

def move_to_position(x: float, y: float):
    send_gcode(f"G0 X{x:.2f} Y{y:.2f}")

def move_to_home_position():
    print("[MOTION] Najíždím na výchozí pozici (0, 0)")
    move_to_position(0, 0)

def move_to_coordinates(x: float, y: float):
    move_to_position(x, y)

def move_relative(dx: float, dy: float):
    gcode = f"G91\nG0 X{dx:.3f} Y{dy:.3f}\nG90"
    send_gcode(gcode)


def grbl_update_position():
    """
    Pošle '?' a načte poslední odpověď od GRBL, uloží do last_position
    """
    global cnc_serial, position_lock, grbl_status, cnc_last_position

    cnc_serial.write(b'?')
    cnc_serial.flush()

    t0 = time.time()
    while time.time() - t0 < 0.5:
        if cnc_serial.in_waiting:
            line = cnc_serial.readline()
            # print("Received: (update_position)", line.decode().strip())
            try:
                decoded = line.decode(errors='ignore').strip()
                print("[GRBL] Status:", decoded)
                if "Idle" in decoded:
                    grbl_status = "Idle"
                elif "Run" in decoded:
                    grbl_status = "Run"
                elif "Error" in decoded:
                    grbl_status = "Error"
                elif "Alarm" in decoded:
                    grbl_status = "Alarm"
                else:
                    grbl_status = "Unknown"

                if 'MPos:' in decoded:
                    for part in decoded.split('|'):
                        if part.startswith('MPos:'):
                            mpos = part[5:].strip()
                            with position_lock:
                                cnc_last_position = mpos
                                # print("Aktuální CNC pozice:",last_position)
                                return cnc_last_position, grbl_status
            except Exception:
                pass
        time.sleep(0.01)
    cnc_last_position = "0.000,0.000,0.000"
    grbl_status = "Unknown"
    return cnc_last_position, grbl_status

def grbl_wait_for_idle():
    """
    Čeká, dokud GRBL nepřijde do stavu Idle (stav získá z threadu update_position_timer).
    """
    global grbl_status

    while True:
        if grbl_status == "Idle":
            break
        time.sleep(0.1)
#
#     while True:
#         cnc_serial.write(b'?')
#         cnc_serial.flush()
#         t0 = time.time()
#         while time.time() - t0 < 0.5:
#             if cnc_serial.in_waiting:
#                 line = cnc_serial.readline().decode(errors='ignore').strip()
#                 if "Idle" in line:
#                     print("[GRBL] Stroj je Idle")
#                     grbl_status = "Idle"
#                     return
#                 else:
#                     grbl_idle = False
#         time.sleep(0.1)  # čekej 100 ms před dalším dotazem