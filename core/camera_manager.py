import cv2
import numpy as np
import threading
import time
import json
from pypylon import pylon
from PIL import Image, ImageTk
from tkinter import messagebox
import config
from core.settings import set_setting, get_setting
from core.motion_controller import grbl_home, move_to_coordinates
import core.camera_manager

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
    camera.Width = camera.Width.Max  # Nastavíme maximální rozlišení
    camera.Height = camera.Height.Max
    camera.PixelFormat.SetValue("Mono8")
    camera.GainAuto.SetValue("Off")
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTimeAbs.Value = 20000  # in microseconds
    camera.GainRaw.Value = 0
    camera.ReverseX.Value = True
    camera.ReverseY.Value = True
    camera.Width.SetValue(min(camera.Width.GetMax(), 1500))
    camera.Height.SetValue(min(camera.Height.GetMax(), 1500))
    try:
        camera.CenterX.SetValue(True)
        camera.CenterY.SetValue(True)
    except Exception:
        pass

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
        print(f"Rozlišení mikroskopu nastaveno na {microscope.Width.Value}x{microscope.Height.Value}")

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
                        live_frame = cv2.resize(live_frame, (live_frame.shape[1] // 4, live_frame.shape[0] // 4))
                    else:
                        live_frame = cv2.warpPerspective(img, config.correction_matrix, (int(config.image_width), int(config.image_height)))
                        live_frame = cv2.resize(live_frame, (live_frame.shape[1] // 2, live_frame.shape[0] // 2))
                    # Převod na RGB
                    img_rgb = cv2.cvtColor(live_frame, cv2.COLOR_GRAY2RGB)

                    # Rozměry a střed
                    h, w = img_rgb.shape[:2]
                    cx, cy = int(w // 2 - 1) , int(h // 2 - 1)

                    # Křížek ve středu
                    cv2.line(img_rgb, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
                    cv2.line(img_rgb, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)


                    im_pil = Image.fromarray(img_rgb)
                    imgtk = ImageTk.PhotoImage(image=im_pil)

                def update():
                    if not image_label.winfo_exists():
                        return  # Widget už neexistuje, ukončíme aktualizaci
                    image_label.imgtk = imgtk
                    image_label.config(image=imgtk)

                image_label.after(100, update)


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

# Kalibrace kamery a mikroskopu
# --- Auto detekce rohových fiducialů (kroužek + křížek) ---

# Parametry vzoru a kamery
SQUARE_MM = 40.0
RADIUS_MM = 1.0         # kroužek ve vzoru má r = 1 mm (SVG)
OFFSET_MM = 2.0         # středy kroužků jsou 2 mm od hrany
ROI_RATIO = 0.20        # rohové ROI = 20 % šířky/výšky

# Stabilizace
ALPHA = 0.25            # vyhlazení EMA (0..1) – menší = klidnější
LOCK_ROI = 100          # lokální ROI kolem poslední pozice (px)
MIN_DELTA = 0.3         # deadband (ignoruj změny < 0.3 px)

def _pix_per_mm(img_w):
    return float(img_w) / SQUARE_MM

def _radius_px(img_w):
    return int(round(RADIUS_MM * _pix_per_mm(img_w)))

def _clamp_roi(x0, y0, x1, y1, w, h):
    x0 = max(0, min(x0, w - 1)); x1 = max(1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(1, min(y1, h))
    if x1 <= x0 + 1: x1 = min(w, x0 + 2)
    if y1 <= y0 + 1: y1 = min(h, y0 + 2)
    return x0, y0, x1, y1

def _smooth(prev, new, alpha=ALPHA):
    if prev is None: return new
    return ((1 - alpha) * prev[0] + alpha * new[0],
            (1 - alpha) * prev[1] + alpha * new[1])

def _corner_rois(w, h, ratio=ROI_RATIO):
    tw, th = int(w * ratio), int(h * ratio)
    # Pořadí: TL, TR, BR, BL
    return [
        (0, 0, tw, th),
        (w - tw, 0, w, th),
        (w - tw, h - th, w, h),
        (0, h - th, tw, h),
    ]

def _detect_circle_in_roi(gray, roi_box, min_r, max_r, seed=None):
    x0, y0, x1, y1 = roi_box
    # lock-on: zmenš ROI kolem poslední pozice
    if seed is not None:
        cx, cy = seed
        rr = LOCK_ROI // 2
        x0, y0, x1, y1 = _clamp_roi(int(cx - rr), int(cy - rr),
                                    int(cx + rr), int(cy + rr),
                                    gray.shape[1], gray.shape[0])
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    # Hough – hledáme vnější kružnici kroužku
    circles = cv2.HoughCircles(
        roi_blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=min(roi.shape)//2,
        param1=120, param2=25,
        minRadius=max(min_r, 8),
        maxRadius=max(max_r, 10)
    )
    if circles is None:
        return None

    circles = np.uint16(np.around(circles[0]))
    if seed is not None:
        sx, sy = seed[0] - x0, seed[1] - y0
        i = int(np.argmin(np.hypot(circles[:,0]-sx, circles[:,1]-sy)))
    else:
        # bez seed: nejblíž očekávanému poloměru
        exp_r = (min_r + max_r) / 2.0
        i = int(np.argmin(np.abs(circles[:,2] - exp_r)))

    c = circles[i]
    cx, cy, r = int(c[0]) + x0, int(c[1]) + y0, int(c[2])
    return float(cx), float(cy), float(r)

def _detect_four_corners(gray, last_centers, img_w):
    """Vrátí [TL, TR, BR, BL] (může obsahovat None). Aplikuje EMA + deadband."""
    R = _radius_px(img_w)
    MIN_R = int(R * 0.5)
    MAX_R = int(R * 1.5)
    rois = _corner_rois(gray.shape[1], gray.shape[0])

    centers = []
    for i, rb in enumerate(rois):
        seed = last_centers[i] if last_centers and last_centers[i] is not None else None
        res = _detect_circle_in_roi(gray, rb, MIN_R, MAX_R, seed=seed)
        if res is None:
            centers.append(last_centers[i] if last_centers else None)
            continue

        new_pt = (res[0], res[1])
        # deadband
        if last_centers and last_centers[i] is not None:
            dx = abs(new_pt[0] - last_centers[i][0])
            dy = abs(new_pt[1] - last_centers[i][1])
            if dx < MIN_DELTA and dy < MIN_DELTA:
                centers.append(last_centers[i]); continue

        centers.append(_smooth(last_centers[i] if last_centers else None, new_pt))
    return centers


# --- hotkey hook pro kalibraci (globální stav) ---
calibration_active = False
_calibration_on_q = None

def register_calibration_hotkeys(on_q):
    global calibration_active, _calibration_on_q
    _calibration_on_q = on_q
    calibration_active = True

def unregister_calibration_hotkeys():
    global calibration_active, _calibration_on_q
    calibration_active = False
    _calibration_on_q = None


def calibrate_camera(container, image_label, move_x, move_y, move_z, step):
    cv2.pts_src = []
    calib_step = "main_camera"
    pts_grbl = []
    current_corner_index = 0
    last_frame = None  # poslední snímek v originálním rozlišení
    # --- auto-detekce rohu: stav ---
    last_centers = [None, None, None, None]  # TL, TR, BR, BL
    auto_ok = False                           # máme 4 stabilní rohy?


    # Parametry
    image_width = config.image_width
    image_height = config.image_height
    sample_position = config.sample_positions_mm[0]
    (pos, base_x, base_y, base_z) = sample_position
    calib_z = config.calib_z
    calib_corners_grbl = config.calib_corners_grbl

    # Přepnutí na hlavní kameru
    if core.camera_manager.actual_camera is None or core.camera_manager.actual_camera == core.camera_manager.microscope:
        switch_camera()

    # Najedeme na startovní pozici
    move_to_coordinates(base_x, base_y, calib_z)
    core.camera_manager.preview_running = False

    def get_frame_with_overlays():
        nonlocal last_frame, auto_ok, last_centers
        img = core.camera_manager.get_image()
        if img is None:
            return None

        # originál pro počítání – grayscale
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # BGR pro overlay
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        last_frame = img_bgr.copy()

        if calib_step == "main_camera":
            # auto detekce rohů
            h, w = gray.shape[:2]
            centers = _detect_four_corners(gray, last_centers, img_w=w)
            last_centers = centers

            # ověř, zda máme všechny 4 body
            auto_ok = all(c is not None for c in centers)
            # Ulož do cv2.pts_src v pořadí TL, TR, BR, BL
            if auto_ok:
                cv2.pts_src = [[int(round(x)), int(round(y))] for (x, y) in centers]

            # overlay ROI a body
            for (x0, y0, x1, y1) in _corner_rois(w, h):
                cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)

            labels = ["TL", "TR", "BR", "BL"]
            for i, c in enumerate(centers):
                if c is None: continue
                x, y = int(round(c[0])), int(round(c[1]))
                cv2.circle(img_bgr, (x, y), 6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img_bgr, f"{labels[i]} {x},{y}", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # textový hint
            status = "Auto: OK (stiskni 'q' pro potvrzení)" if auto_ok else "Hledám rohy..."
            cv2.putText(img_bgr, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        elif calib_step == "microscope":
            h, w = img_bgr.shape[:2]
            cx, cy = int(w // 2 - 1), int(h // 2 - 1)
            cv2.line(img_bgr, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
            cv2.line(img_bgr, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)

        return img_bgr

    def refresh_image():
        nonlocal calib_step
        if calib_step == "done":
            return
        img = get_frame_with_overlays()
        if img is not None:
            if calib_step == "main_camera_done" and len(cv2.pts_src) >= 4 and config.correction_matrix is not None:
                # Pro ověření kalibrace hlavní kamery použijeme korekční matici
                img = cv2.warpPerspective(img, config.correction_matrix, (image_width, image_height))
            # Převod na velikost widgetu
            widget_w = config.frame_width
            widget_h = config.frame_height
            img_resized = cv2.resize(img, (widget_w, widget_h))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
            image_label.imgtk = imgtk
            image_label.config(image=imgtk)
        image_label.after(100, refresh_image)

    def on_q():
        nonlocal calib_step, current_corner_index, pts_grbl
        # potvrzení auto detekce rohů hlavní kamery
        if calib_step == "main_camera":
            if len(cv2.pts_src) == 4:
                finish_main_camera_calib()
            else:
                print("[CALIBRATION] Ještě nemám 4 stabilní body.")
            return

        # původní část pro mikroskop nech beze změny:
        if calib_step == "microscope":
            pos_x, pos_y, pos_z = [float(val) for val in core.motion_controller.grbl_last_position.split(",")]
            pts_grbl.append([pos_x - base_x, pos_y - base_y])
            print(f"[CALIBRATION] Přidávám bod {len(pts_grbl)}: ({pos_x}, {pos_y})")

            current_corner_index += 1
            if current_corner_index < len(calib_corners_grbl):
                threading.Thread(
                    target=move_to_coordinates,
                    args=(calib_corners_grbl[current_corner_index][0] + base_x,
                          calib_corners_grbl[current_corner_index][1] + base_y,
                          calib_z),
                    daemon=True
                ).start()
            else:
                finish_microscope_calib()

    def finish_main_camera_calib():
        nonlocal calib_step

        pts_src = np.float32(cv2.pts_src[:4])
        pts_dst = np.float32([
            [0, 0],
            [image_width-1, 0],
            [image_width-1, image_height-1],
            [0, image_height-1]
        ])
        correction_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        set_setting("correction_matrix", correction_matrix.tolist())
        config.correction_matrix = np.array(json.loads(get_setting("correction_matrix")))
        print("[CALIBRATION] Korekční matice hlavní kamery:", correction_matrix)
        calib_step = "main_camera_done"

        if messagebox.askquestion("Kalibrace", "Je kalibrace hlavní kamery v pořádku?") == "yes":
            calib_step = "microscope"
            start_microscope_calib()
        else:
            cv2.pts_src.clear()

    def start_microscope_calib():
        nonlocal current_corner_index, pts_grbl
        if core.camera_manager.actual_camera is None or core.camera_manager.actual_camera == core.camera_manager.camera:
            switch_camera()
        core.camera_manager.microscope.ExposureTimeAbs.Value = 15000
        pts_grbl = []
        current_corner_index = 0
        threading.Thread(
            target=move_to_coordinates,
            args=(calib_corners_grbl[0][0] + base_x, calib_corners_grbl[0][1] + base_y, calib_z),
            daemon=True
        ).start()

    def finish_microscope_calib():
        pts_dst = np.float32([
            [0, 0],
            [image_width-1, 0],
            [image_width-1, image_height-1],
            [0, image_height-1]
        ])
        pts_grbl_np = np.float32(pts_grbl[:4])
        correction_matrix_grbl = cv2.getPerspectiveTransform(pts_dst, pts_grbl_np)
        set_setting("correction_matrix_grbl", correction_matrix_grbl.tolist())
        set_setting("calib_corners_grbl", pts_grbl_np.tolist())
        config.correction_matrix_grbl = np.array(json.loads(get_setting("correction_matrix_grbl")))
        config.calib_corners_grbl = np.array(json.loads(get_setting("calib_corners_grbl")))
        print("[CALIBRATION] Korekční matice mikroskopu:", correction_matrix_grbl)
        # Ověření kalibrace mikroskopu
        def verify_microscope_calibration():
            nonlocal calib_step
            calib_step = "done"
            core.camera_manager.start_camera_preview(image_label, update_position_callback=None)
            print("[CALIBRATION] Ověřuji kalibraci mikroskopu...")
            test_coordinates = np.float32([
                [0, 0],
                [image_width - 1, 0],
                [image_width - 1, image_height - 1],
                [0, image_height - 1],
                [image_width // 2 - 1, image_height // 2 - 1]
            ])
            for test_point in test_coordinates:
                transformed_point = cv2.perspectiveTransform(np.array([[[test_point[0], test_point[1]]]], dtype=np.float32), config.correction_matrix_grbl)[0][0]
                print(f"[CALIBRATION] Testovací bod {test_point} se transformoval na: {transformed_point}")
                move_to_coordinates(transformed_point[0] + base_x, transformed_point[1] + base_y, calib_z)
                time.sleep(2)
            if messagebox.askquestion("Kalibrace mikroskopu", "Je kalibrace mikroskopu v pořádku?") == "yes":
                core.camera_manager.microscope.ExposureTimeAbs.Value = 3000
                switch_camera()
                move_to_coordinates(base_x, base_y, calib_z)
                messagebox.showinfo("Kalibrace", "Kalibrace dokončena.")
                unregister_calibration_hotkeys()
            else:
                start_microscope_calib()

        threading.Thread(target=verify_microscope_calibration, daemon=True).start()


    # Přidáme callback pro stisknutí klávesy 'q'
    register_calibration_hotkeys(on_q)

    # Start náhledu
    refresh_image()


