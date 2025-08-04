import os
import cv2
import numpy as np
import threading
import time
import json
from pypylon import pylon
from PIL import Image, ImageTk
import config
from core.settings import set_setting, get_setting
from core.motion_controller import grbl_home, grbl_wait_for_idle, move_to_coordinates, grbl_status
import core.camera_manager
import tkinter




camera = None
microscope = None
actual_camera = None
preview_running = False
camera_lock = threading.Lock()
live_frame_lock = threading.Lock()
live_frame = None

def init_cameras():
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

    print(f"Rozlišení kamery nastaveno na {camera.Width.Value}x{camera.Height.Value}")

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

def switch_camera():
    global camera, microscope, actual_camera, live_frame_lock

    with live_frame_lock:
    # Stop grabbing on the previous camera
        if actual_camera is not None:
            if actual_camera.IsGrabbing():
                actual_camera.StopGrabbing()
            if actual_camera is microscope:
                actual_camera = camera
            else:
                actual_camera = microscope

        try:
            print(f"Přepínám kameru na {actual_camera.GetDeviceInfo().GetModelName()}")
            actual_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            # Wait for at least one frame to be available
            timeout = time.time() + 5  # 5 seconds timeout
            while not actual_camera.IsGrabbing():
                if time.time() > timeout:
                    print("⚠️  Kamera se nespustila včas.")
                    break
                time.sleep(0.05)
            if actual_camera.IsGrabbing():
                print(f"Kamera {actual_camera.GetDeviceInfo().GetModelName()} je nyní připravena.")
        except Exception:
            pass

def get_image():
    global actual_camera, camera_lock

    with camera_lock:
        if actual_camera is None or not actual_camera.IsGrabbing():
            return None
        try:
            result = actual_camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
            if result is not None and result.GrabSucceeded():
                img = result.Array
                result.Release()
                # print(f"{actual_camera.GetDeviceInfo().GetModelName()} Snímání úspěšné: {img.shape[1]}x{img.shape[0]} px")
                return img
            if result is not None:
                result.Release()
        except Exception as e:
            # Optionally print or log the error
            pass
        return None

def start_camera_preview(image_label, update_position_callback=None):
    global camera, microscope, actual_camera, preview_running, live_frame_lock

    if preview_running:
        return


    if actual_camera is None or camera is None or microscope is None:
        print("[Camera] Inicializuji kameru...")
        try:
            camera, microscope = init_cameras()
            actual_camera = camera  # Nastavíme aktuální kameru na hlavní kameru
            actual_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            print(f"[Camera] Chyba při inicializaci kamery: {e}")
            return

    preview_running = True

    def preview_loop():
        global camera, microscope, actual_camera, preview_running, live_frame, live_frame_lock

        while preview_running:
                # print("Zachycuji snímek...")
                with live_frame_lock:
                    img = get_image()
                if img is not None:
                    if actual_camera == microscope:
                        # For microscope, we don't apply correction matrix
                        live_frame = img
                    else:
                        live_frame = cv2.warpPerspective(img, config.correction_matrix, (int(config.image_width), int(config.image_height)))
                    # Převod na RGB
                    img_rgb = cv2.cvtColor(live_frame, cv2.COLOR_GRAY2RGB)

                    # Rozměry a střed
                    h, w = img_rgb.shape[:2]
                    cx, cy = w // 2, h // 2

                    # Křížek ve středu
                    cv2.line(img_rgb, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
                    cv2.line(img_rgb, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)

                    # Zmenšit výstup na 640x480 (nebo ponechat originál)
                    img_resized = cv2.resize(img_rgb, (cx, cy))

                    im_pil = Image.fromarray(img_resized)
                    imgtk = ImageTk.PhotoImage(image=im_pil)

                def update():
                    if not image_label.winfo_exists():
                        return  # Widget už neexistuje, ukončíme aktualizaci
                    image_label.imgtk = imgtk
                    image_label.config(image=imgtk)

                image_label.after(0, update)


        # actual_camera.StopGrabbing()
        # actual_camera.Close()

    # spustíme náhled ve vlákně, aby neblokoval GUI
    threading.Thread(target=preview_loop, daemon=True).start()






def stop_camera_preview():
    global preview_running
    preview_running = False
    release_camera()



def release_camera():
    """
    Korektně zavře připojení ke kameře.
    """
    global camera, microscope, actual_camera
    try:
        if actual_camera and actual_camera.IsGrabbing():
            actual_camera.StopGrabbing()
        if actual_camera and actual_camera.IsOpen():
            actual_camera.Close()
        actual_camera = None
        print("[Camera] Kamera uvolněna.")
    except Exception as e:
        print("[Camera] Chyba při uvolňování kamery:", e)

def calibrate_camera(container, image_label):
    """
    Spustí kalibraci kamery.
    """
    global actual_camera, camera, microscope
    # grbl_home() # Domů, aby se zajistilo, že CNC je dobře nastaveno
    # grbl_wait_for_idle()
    if actual_camera is None or actual_camera == microscope:
        switch_camera() # Přepneme na hlavní kameru, pokud je aktuálně mikroskop

    # najedeme na první pozici vzorku
    sample_position = config.sample_positions_mm[0]
    (pos, x, y, z) = sample_position
    print(f"[FIND] Najíždím na pozici {pos}: ({x}, {y}, {z})")
    move_to_coordinates(x, y, z)
    print("[FIND] Snímám fotku hlavní kamerou...")
    # core.camera_manager.preview_running = False  # Zastavíme živý náhled, abychom mohli získat snímek
    # time.sleep(0.2)  # Počkáme, aby se proces náhledu zastavil
    camera_calibration_successful = False
    cv2.pts_src = []  # Proměnná pro ukládání vybraných bodů myši
    # force width and height to be a fixed size square
    image_width = 1400
    image_height = 1400

    print("[Camera] Spouštím kalibraci kamery...")
    while not camera_calibration_successful:
        image = core.camera_manager.get_image()

        if image is not None:
            h, w = image.shape[:2]
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
            cv2.resizeWindow("Kliknete mysi na 4 body v rozich", int(camera.Width.Max / 2), int(camera.Height.Max / 2))
            cv2.setMouseCallback("Kliknete mysi na 4 body v rozich", select_point)

            img_for_selection = image.copy()
            for pt in getattr(cv2, 'pts_src', []):
                cv2.circle(img_for_selection, tuple(pt), 5, (0, 0, 255), -1)
            cv2.imshow("Kliknete mysi na 4 body v rozich", img_for_selection)

            if len(cv2.pts_src) < 4:
                cv2.waitKey(1)
                continue
            pts_src = np.float32(cv2.pts_src[:4])


            pts_dst = np.float32([
                [0, 0],
                [image_width - 1, 0],
                [image_width - 1, image_height - 1],
                [0, image_height - 1]
            ])

            # Compute the perspective transform matrix and apply it
            correction_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            # save the correction matrix to config
            set_setting("correction_matrix", correction_matrix.tolist())
            # Znovu načti korekční matici pro transformaci perspektivy, aby se potvrdilo správné uložení
            config.correction_matrix = np.array(json.loads(get_setting("correction_matrix")))

        else:
            print("[FIND] Chyba při získávání snímku z kamery, obrázek je None.")

        cv2.destroyAllWindows()


        answer = tkinter.messagebox.askquestion("Kalibrace", "Je kalibrace v pořádku?")
        if answer == "no":
            cv2.pts_src = []
            # Repeat the calibration process from line 862 (start of this function's while loop)
            continue
        else:
            print("[FIND] Kalibrace kamery byla úspěšná.")
            camera_calibration_successful = True

    # Po úspěšné kalibraci kamery přejdeme na kalibraci mikroskopu
    print("[Camera] Spouštím kalibraci mikroskopu...")
    # Přepneme na mikroskop, pokud je dostupný

    if actual_camera is None or actual_camera == camera:
        switch_camera()  # Přepneme na mikroskop, pokud je aktuálně hlavní kamera

    microscope_calibration_successful = False
    while not microscope_calibration_successful:
        # Corners of the image in microscope coordinates
        pts_grbl = []
        corners_grbl = [(-157.900, -174.100, -55.300),  # Top-left corner
                        (-119.900, -174.100, -55.300),  # Top-right corner
                        (-119.900, -136.100, -55.300),  # Bottom-right corner
                        (-157.900, -136.100, -55.300)]   # Bottom-left corner

        while len(pts_grbl) < 4:

            for corners in corners_grbl:
                move_to_coordinates(corners[0], corners[1], corners[2])  # Move to the first corner position
                while True:
                    image = core.camera_manager.get_image()
                    if image is not None:
                        h, w = image.shape[:2]
                        # Ensure the image is in color (BGR) for drawing colored shapes/text
                        if len(image.shape) == 2 or image.shape[2] == 1:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                            # Nakreslíme křížek do středu obrazu
                            cv2.line(image, (w / 2 - 15, h / 2), (w / 2 + 15, h / 2), (0, 0, 255), 1)
                            cv2.line(image, (w / 2, h / 2 - 15), (w / 2, h / 2 + 15), (0, 0, 255), 1)
                            # Create window and resize to fit the screen
                            cv2.namedWindow("Najedte na roh krizkem a zmacknete q", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("Najedte na roh krizkem a zmacknete q", int(w / 3), int(h / 3))
                            cv2.imshow("Najedte na roh krizkem a zmacknete q", image)

                    key = cv2.waitKey(30)
                    if key == ord('q'):
                        break

                pos_x, pos_y, pos_z = [float(val) for val in core.motion_controller.grbl_last_position.split(",")]
                # Přepočítáme souřadnice na základě offsetu mikroskopu a pozice vzorku A1
                pts_grbl.append([pos_x-x, pos_y-y])
                print(f"[CALIBRATION] Přidávám bod {len(pts_grbl)}: ({pos_x}, {pos_y})")

        # Po úspěšné kalibraci mikroskopu uložíme korekční matici
        pts_grbl = np.float32(pts_grbl[:4])

        # Compute the perspective transform matrix and apply it
        correction_matrix_microscope = cv2.getPerspectiveTransform(pts_dst, pts_grbl)
        # save the correction matrix to config
        print(f"[CALIBRATION] Korekční matice mikroskopu: {correction_matrix_microscope}")
        set_setting("correction_matrix_microscope", correction_matrix_microscope.tolist())
        # Znovu načti korekční matici pro transformaci perspektivy,
        # aby se potvrdilo správné uložení
        config.correction_matrix_microscope = np.array(json.loads(get_setting("correction_matrix_microscope")))
        # Najedeme mikroskopem na střed čtverce, abychom ověřili správnost kalibrace
        coordinates = cv2.perspectiveTransform(np.array([[[int(image_width // 2)-1, int(image_height//2)-1]]], dtype=np.float32), config.correction_matrix_microscope)[0][0]
        print(f"[CALIBRATION] Střed čtverce v mikroskopu: ({coordinates[0]+x}, {coordinates[1]+y})")
        move_to_coordinates(coordinates[0]+x, coordinates[1]+y, -53.200)
        # Zobrazíme uživateli, že má zkontrolovat střed čtverce
        answer = None
        answer = tkinter.messagebox.askquestion("Kalibrace mikroskopu", "Je kalibrace v pořádku? Klikněte na 'No' pro opakování.")
        if answer == "No":
            # Pokud uživatel odpoví "No", vymažeme body a začneme znovu
            print("[FIND] Kalibrace kamery nebyla úspěšná, opakujeme proces.")
            pts_grbl = []
            continue
        else:
            print("[FIND] Kalibrace kamery byla úspěšná.")
            microscope_calibration_successful = True


