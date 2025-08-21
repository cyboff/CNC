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
        messagebox.showerror("Chyba kamery", "Kamera Basler nenalezena!")
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
    camera.ExposureTimeAbs.Value = config.camera_exposure_time  # in microseconds
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
        try:
           microscope.Open()
        except Exception as e:
            messagebox.showerror("Chyba mikroskopu", f"Chyba při připojování se k mikroskopu: {e}")
            raise
        # microscope.GainAuto.SetValue("Continuous")
        # microscope.ExposureAuto.SetValue("Continuous")
        microscope.GainAuto.SetValue("Off")
        microscope.ExposureAuto.SetValue("Off")
        microscope.ExposureTimeAbs.Value = config.microscope_exposure_time  # in microseconds
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
                        sharpness = tenengrad_sharpness(live_frame)
                        live_frame = cv2.resize(live_frame, (live_frame.shape[1] // 4, live_frame.shape[0] // 4))
                    else:
                        live_frame = cv2.warpPerspective(img, config.correction_matrix, (int(config.image_width), int(config.image_height)))
                        live_frame = cv2.resize(live_frame, (live_frame.shape[1] // 2, live_frame.shape[0] // 2))
                    # Převod na RGB
                    img_rgb = cv2.cvtColor(live_frame, cv2.COLOR_GRAY2RGB)
                    if actual_camera == microscope:
                        cv2.putText(img_rgb, f"Ostrost: {sharpness:.2f}", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Rozměry a střed
                    h, w = img_rgb.shape[:2]
                    cx, cy = int(w // 2) , int(h // 2)

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

                image_label.after(250, update) # asi není třeba plynulý pohyb, méně to zatěžuje přenosy a procesor


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

# Převody pixel -> mm
def rectpx_to_mm_offset(u_rect: float, v_rect: float):
    """
    Pixel z WARPnutého (rectified) obrazu -> (dx, dy) v mm
    vůči aktuálnímu base (tj. offset v rovině stolku).
    Vyžaduje config.correction_matrix_grbl.
    """
    H = getattr(config, "correction_matrix_grbl", None)
    if H is None:
        raise RuntimeError("Neexistuje correction_matrix_grbl – dokonči kalibraci mikroskopu.")
    pt = np.array([[[float(u_rect), float(v_rect)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H.astype(np.float32))[0][0]
    return float(out[0]), float(out[1])

def rectpx_to_grbl(u_rect: float, v_rect: float, base_xy=None, z=None):
    """
    Pixel z rectified 38×38mm -> absolutní GRBL (X,Y,Z).
    base_xy je (X0,Y0) v okamžiku pořízení snímku (pokud None, vezme se grbl_last_position).
    Z pokud None, vezme se config.calib_z.
    """
    if base_xy is None:
        x0, y0, z0 = [float(v) for v in core.motion_controller.grbl_last_position.split(",")]
    else:
        x0, y0 = base_xy
        z0 = z if z is not None else getattr(config, "calib_z", 0.0)
    dx, dy = rectpx_to_mm_offset(u_rect, v_rect)
    return x0 + dx, y0 + dy, z0

def rectpx_scaled_to_grbl(u_scaled: float, v_scaled: float, scaled_w: int, scaled_h: int, base_xy=None, z=None):
    """
    Pokud detekuješ v downscalovaném náhledu (např. polovina),
    převeď souřadnice na kanonické rect rozlišení (image_width × image_height)
    a pak na GRBL.
    """
    W = int(config.image_width); H = int(config.image_height)
    u = u_scaled * (W / float(scaled_w))
    v = v_scaled * (H / float(scaled_h))
    return rectpx_to_grbl(u, v, base_xy=base_xy, z=z)

# použití:
#
# # 1) Získej frame a rect warp stejného rozlišení, v jakém proběhla kalibrace:
# img_raw = core.camera_manager.get_image()
# rect = cv2.warpPerspective(img_raw, config.correction_matrix,
#                            (int(config.image_width), int(config.image_height)))  # raw -> rect
# # 2) Ulož si base v okamžiku pořízení snímku (aby mezitím hlava nepopojela):
# base_x, base_y, base_z = [float(v) for v in core.motion_controller.grbl_last_position.split(",")]
#
# # 3) Najdi kruh/křížek v rect (u_rect, v_rect) – tvůj detektor
# u_rect, v_rect = detected_center  # v pixelech rect obrazu
#
# # 4) Převod na GRBL a najeď mikroskopem:
# gx, gy, gz = rectpx_to_grbl(u_rect, v_rect, base_xy=(base_x, base_y), z=config.calib_z)
# core.motion_controller.move_to_position_antibacklash(gx, gy, gz, anti_backlash=True, wait_end=True)


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

def _refine_center(gray, x, y, win=9):
    x_i, y_i = int(round(x)), int(round(y))
    x0 = max(0, x_i - win); y0 = max(0, y_i - win)
    x1 = min(gray.shape[1], x_i + win + 1); y1 = min(gray.shape[0], y_i + win + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return float(x), float(y)
    edges = cv2.Canny(cv2.GaussianBlur(roi, (3,3), 0), 50, 150)
    m = cv2.moments(edges, binaryImage=True)
    if m["m00"] > 1e-6:
        cx = x0 + m["m10"]/m["m00"]
        cy = y0 + m["m01"]/m["m00"]
        return float(cx), float(cy)
    return float(x), float(y)

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
        rx, ry = _refine_center(gray, new_pt[0], new_pt[1], win=9)
        new_pt = (rx, ry)

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
    # Parametry

    image_width = config.image_width
    image_height = config.image_height
    base_x, base_y, base_z = 0, 0, 0
    sample_position = config.sample_positions_mm[0]
    (pos, base_x, base_y, base_z) = sample_position
    calib_z = config.calib_z

    calib_corners_grbl = config.calib_corners_grbl
    prev_correction_matrix_main = getattr(config, "correction_matrix", None)
    pts_src = []
    pts_grbl = []
    pts_dst = np.float32([
        [0, 0],
        [image_width - 1, 0],
        [image_width - 1, image_height - 1],
        [0, image_height - 1]
    ])
    calib_step = "main_camera"

    current_corner_index = 0
    last_frame = None  # poslední snímek v originálním rozlišení
    # --- auto-detekce rohu: stav ---
    last_centers = [None, None, None, None]  # TL, TR, BR, BL
    auto_ok = False  # máme 4 stabilní rohy?

    # # --- PRE-FOCUS: uzamkni správné calib_z přes mikroskop ještě před kalibrací hlavní kamery ---
    # try:
    #     # 1) Vypočti střed terče v rectified prostoru (pixely) a převeď na mm pomocí dřívější mikroskopické vazby
    #     cx_rect = (image_width - 1) / 2.0
    #     cy_rect = (image_height - 1) / 2.0
    #
    #     target_x = base_x  # fallback
    #     target_y = base_y  # fallback
    #
    #     if getattr(config, "correction_matrix_grbl", None) is not None:
    #         center_px = np.array([[[cx_rect, cy_rect]]], dtype=np.float32)
    #         center_mm = cv2.perspectiveTransform(center_px, config.correction_matrix_grbl.astype(np.float32))[0][0]
    #         target_x = base_x + float(center_mm[0])  # correction_matrix_grbl vrací offsety v mm
    #         target_y = base_y + float(center_mm[1])
    #     elif getattr(config, "calib_corners_grbl", None) is not None and len(config.calib_corners_grbl) >= 4:
    #         # fallback: střed = průměr dříve uložených rohů v mm
    #         cc = np.array(config.calib_corners_grbl, dtype=np.float32)
    #         mean_xy = cc.mean(axis=0)
    #         target_x = base_x + float(mean_xy[0])
    #         target_y = base_y + float(mean_xy[1])
    #
    #     # 2) Přepni na mikroskop, najeď na střed a proveď autofocus (Z)
    #     if core.camera_manager.actual_camera is None or core.camera_manager.actual_camera == core.camera_manager.camera:
    #         switch_camera()
    #     core.camera_manager.microscope.ExposureTimeAbs.Value = 15000
    #
    #     move_to_coordinates(target_x, target_y, calib_z)
    #
    #     best_z, best_score = autofocus_z()  # už máš implementováno níže v souboru
    #     calib_z = float(best_z)  # aktualizujeme lokální proměnnou calib_z
    #     config.calib_z = calib_z  # a i globální config, ať ji vidí zbytek kódu
    #     set_setting("calib_z", calib_z)
    #     print(f"[CALIBRATION] PRE-FOCUS: calib_z = {calib_z:.3f} mm (score={best_score:.1f})")
    #
    # except Exception as e:
    #     print(f"[CALIBRATION] PRE-FOCUS přeskočen: {e}")
    #
    # # 3) Přepni zpět na hlavní kameru a pokračuj jako dřív, už s novým calib_z
    # if core.camera_manager.actual_camera is None or core.camera_manager.actual_camera == core.camera_manager.microscope:
    #     switch_camera()

    move_to_coordinates(base_x, base_y, calib_z)

    def get_frame_with_overlays():
        nonlocal last_frame, auto_ok, last_centers, pts_src
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
            # Ulož do pts_src v pořadí TL, TR, BR, BL
            if auto_ok:
                pts_src = [[float(x), float(y)] for (x, y) in centers]

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

        # elif calib_step == "microscope":
        #     h, w = img_bgr.shape[:2]
        #     cx, cy = int(w // 2), int(h // 2)
        #     cv2.line(img_bgr, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
        #     cv2.line(img_bgr, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)

        return img_bgr

    def refresh_image():
        nonlocal calib_step, pts_src
        if calib_step in ("main_camera_done", "microscope", "done"):
            return
        core.camera_manager.preview_running = False
        img = get_frame_with_overlays()
        if img is not None:
            # if calib_step == "main_camera_done" and len(pts_src) >= 4 and config.correction_matrix is not None:
            #     # Pro ověření kalibrace hlavní kamery použijeme korekční matici
            #     img = cv2.warpPerspective(img, config.correction_matrix, (image_width, image_height))
            # Převod na velikost widgetu
            widget_w = config.frame_width
            widget_h = config.frame_height
            img_resized = cv2.resize(img, (widget_w, widget_h))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img_resized))
            image_label.imgtk = imgtk
            image_label.config(image=imgtk)
        image_label.after(100, refresh_image)

    def on_q():
        nonlocal calib_step, current_corner_index, pts_grbl, pts_src, calib_corners_grbl, base_x, base_y, base_z
        # potvrzení auto detekce rohů hlavní kamery
        if calib_step == "main_camera":
            if len(pts_src) == 4:
                # Získáme aktuální pozici GRBL
                base_x, base_y, base_z = [float(val) for val in core.motion_controller.grbl_last_position.split(",")]
                finish_main_camera_calib()
            else:
                print("[CALIBRATION] Ještě nemám 4 stabilní body.")
            return

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
        nonlocal calib_step, pts_src, pts_grbl, current_corner_index, last_frame, auto_ok, last_centers

        pts_src = np.float32(pts_src[:4])
        correction_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        config.correction_matrix = correction_matrix
        print("[CALIBRATION] Korekční matice hlavní kamery:", correction_matrix)
        calib_step = "main_camera_done"
        core.camera_manager.start_camera_preview(image_label, update_position_callback=None)

        if messagebox.askquestion("Kalibrace", "Je kalibrace hlavní kamery v pořádku?") == "yes":
            # Uložení korekční matice do databáze
            set_setting("correction_matrix", correction_matrix.tolist())
            config.correction_matrix = np.array(json.loads(get_setting("correction_matrix")))
            print("[CALIBRATION] Kalibrační matice hlavní kamery uložena:", config.correction_matrix)
            calib_step = "microscope"
            start_microscope_calib()
        else:
            pts_src = []
            pts_grbl = []
            current_corner_index = 0
            last_frame = None  # poslední snímek v originálním rozlišení
            # --- auto-detekce rohu: stav ---
            last_centers = [None, None, None, None]  # TL, TR, BR, BL
            auto_ok = False  # máme 4 stabilní rohy?
            calib_step = "main_camera"
            print("[CALIBRATION] Opakuji kalibraci hlavní kamery...")
            refresh_image()

    def start_microscope_calib():
        nonlocal current_corner_index, pts_grbl, calib_corners_grbl, prev_correction_matrix_main
        if core.camera_manager.actual_camera is None or core.camera_manager.actual_camera == core.camera_manager.camera:
            switch_camera()
        core.camera_manager.microscope.ExposureTimeAbs.Value = config.microscope_exposure_time_calib # zvýšení expozice pro kalibraci
        pts_grbl = []
        current_corner_index = 0
        print(f"[CALIBRATION] Začínám kalibraci mikroskopu.")

        # >>> Klíč: nové rectified -> mm pomocí kombinace starých/nových matic
        if (config.correction_matrix_grbl is not None and
                prev_correction_matrix_main is not None and
                config.correction_matrix is not None):
            H_rectnew_to_mm = (
                    config.correction_matrix_grbl.astype(np.float64) @
                    prev_correction_matrix_main.astype(np.float64) @
                    np.linalg.inv(config.correction_matrix.astype(np.float64))
            )
            # normalizace tak, aby H[2,2] == 1
            H_rectnew_to_mm /= H_rectnew_to_mm[2, 2]
            pts_dst_np = np.array(pts_dst[None, :, :], dtype=np.float32)  # (1,4,2)
            calib_corners_grbl = cv2.perspectiveTransform(pts_dst_np, H_rectnew_to_mm.astype(np.float32))[0]
            print(f"[CALIBRATION] Rohy mikroskopu (aktuální, z kombinované homografie): {calib_corners_grbl}")
        else:
            # Fallback, když není předchozí kalibrace k dispozici
            if config.correction_matrix_grbl is not None:
                calib_corners_grbl = cv2.perspectiveTransform(
                    np.array(pts_dst[None, :, :], dtype=np.float32),
                    config.correction_matrix_grbl
                )[0]
            else:
                calib_corners_grbl = config.calib_corners_grbl

        threading.Thread(
            target=move_to_coordinates,
            args=(calib_corners_grbl[0][0] + base_x, calib_corners_grbl[0][1] + base_y, calib_z),
            daemon=True
        ).start()

    def finish_microscope_calib():
        pts_grbl_np = np.float32(pts_grbl[:4])
        correction_matrix_grbl = cv2.getPerspectiveTransform(pts_dst, pts_grbl_np)
        config.correction_matrix_grbl = correction_matrix_grbl
        config.calib_corners_grbl = pts_grbl_np

        print("[CALIBRATION] Korekční matice mikroskopu:", correction_matrix_grbl)
        print("[CALIBRATION] Korekční matice mikroskopu config:", config.correction_matrix_grbl)
        # Ověření kalibrace mikroskopu
        def verify_microscope_calibration():
            nonlocal calib_step
            calib_step = "done"
            # core.camera_manager.start_camera_preview(image_label, update_position_callback=None)
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
                # uložení korekční matice a rohů
                set_setting("correction_matrix_grbl", correction_matrix_grbl.tolist())
                set_setting("calib_corners_grbl", pts_grbl_np.tolist())
                config.correction_matrix_grbl = np.array(json.loads(get_setting("correction_matrix_grbl")))
                config.calib_corners_grbl = np.array(json.loads(get_setting("calib_corners_grbl")))
                core.camera_manager.microscope.ExposureTimeAbs.Value = config.microscope_exposure_time
                switch_camera()
                move_to_coordinates(base_x, base_y, calib_z)
                messagebox.showinfo("Kalibrace", "Kalibrace dokončena.")
                unregister_calibration_hotkeys()
            else:
                calib_step = "microscope"
                start_microscope_calib()

        threading.Thread(target=verify_microscope_calibration, daemon=True).start()


    # Přidáme callback pro stisknutí klávesy 'q'
    register_calibration_hotkeys(on_q)

    # Start náhledu
    refresh_image()

# --- Autofokus mikroskopu (Tenengrad) ----------------------------------------
# Měření ostrosti obrázku Laplacianem není spolehlivé, Tenengrad metoda je mnohem lepší
def tenengrad_sharpness(gray: np.ndarray) -> float:
    """
    Tenengrad (Sobel) míra ostrosti: průměr z (Gx^2 + Gy^2).
    Očekává grayscale (8bit).
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    fm = gx * gx + gy * gy
    return float(np.mean(fm))

def _grab_focus_frame():
    """
    Bezpečné sejmutí snímku z aktuální kamery (preferujeme mikroskop),
    s malým zpožděním na uklidnění po pohybu osy Z.
    """
    # počkej chvilku na uklidnění vibrací a expozici
    time.sleep(0.08)
    img = get_image()
    # fallback, zkus ještě jednou
    if img is None:
        time.sleep(0.05)
        img = get_image()
    return img

def _ensure_microscope():
    """
    Přepni na mikroskop, pokud ještě není aktivní.
    """
    global actual_camera, microscope, camera
    if microscope is None:
        raise RuntimeError("Mikroskopická kamera není k dispozici.")
    if actual_camera is None or actual_camera is camera:
        switch_camera()

def autofocus_z(
    search_heights_mm=getattr(config, "autofocus_steps", (0.05, 0.01, 0.001)),
    settle_s=0.08,
    max_steps_per_level=200,
    overshoot_backtrack=3, # kolik dalších kroků po sobě s už horší ostrosti
    verbose=True,
):
    """
    Vícestupňový hill-climb autofokus v ose Z pomocí Tenengrad ostrosti.

    Algoritmus:
      1) Na aktuální Z změříme ostrost.
      2) Zkusíme +krok. Pokud se zlepší, pokračujeme stejným směrem.
         Pokud ne, zkusíme opačný směr (–krok).
      3) Pokud přijdou 3 po sobě jdoucí kroky s horší ostrostí,
         vrátíme se na nejlepší Z aktuální úrovně (o N*step zpět)
         a pokračujeme jemnějším krokem.
      4) Po dojetí všech úrovní se nastaví nejlepší nalezené Z.

    Návratová hodnota: (best_z, best_score)
    """
    # _ensure_microscope()

    # Přečti aktuální pozici stroje
    try:
        pos_x, pos_y, pos_z = [float(v) for v in core.motion_controller.grbl_last_position.split(",")]
    except Exception:
        # Pokud není k dispozici, zkus získat přímo snímek a pokračuj
        img0 = _grab_focus_frame()
        if img0 is None:
            raise RuntimeError("Nelze získat snímek pro autofokus.")
        # Bez znalosti Z nemůžeme jezdit -> chyba
        raise RuntimeError("Nelze načíst aktuální GRBL pozici (grbl_last_position).")

    # Pomocná funkce pro přesun na Z
    def goto_z(z_target):
        move_to_coordinates(pos_x, pos_y, z_target)
        # čas na uklidnění
        time.sleep(settle_s)

    # Změř startovní ostrost
    img = _grab_focus_frame()
    if img is None:
        raise RuntimeError("Kamera nevrací snímky (autofokus).")
    best_score = tenengrad_sharpness(img)
    best_z = pos_z
    if verbose:
        print(f"[AF] Start Z={best_z:.6f} mm, score={best_score:.2f}")

    current_z = pos_z

    for level, step in enumerate(search_heights_mm, start=1):
        if verbose:
            print(f"[AF] Level {level}: step={step} mm")

        # Nejprve detekuj výhodnější směr (nahoru vs. dolů)
        trial_scores = []
        # Zkus +step
        goto_z(current_z + step)
        img_p = _grab_focus_frame()
        s_plus = tenengrad_sharpness(img_p) if img_p is not None else -np.inf
        # Zpět
        goto_z(current_z)

        # Zkus -step
        goto_z(current_z - step)
        img_m = _grab_focus_frame()
        s_minus = tenengrad_sharpness(img_m) if img_m is not None else -np.inf
        # Vrať se na výchozí
        goto_z(current_z)

        if s_plus >= s_minus:
            direction = +1
            if verbose:
                print(f"[AF]  Preferuji směr + (s+={s_plus:.2f} >= s-={s_minus:.2f})")
        else:
            direction = -1
            if verbose:
                print(f"[AF]  Preferuji směr - (s-={s_minus:.2f} > s+={s_plus:.2f})")

        worse_in_row = 0
        steps_done = 0
        local_best_z = best_z
        local_best_score = best_score

        while steps_done < max_steps_per_level:
            # Udělej krok ve zvoleném směru
            current_z = current_z + direction * step
            goto_z(current_z)
            img = _grab_focus_frame()
            if img is None:
                if verbose:
                    print("[AF]  Varování: žádný snímek, krok přeskočen.")
                continue
            score = tenengrad_sharpness(img)
            steps_done += 1

            if verbose:
                print(f"[AF]   Z={current_z:.6f} mm -> score={score:.2f}")

            if score > local_best_score:
                local_best_score = score
                local_best_z = current_z
                worse_in_row = 0
                # průběžně aktualizuj i globální best
                if score > best_score:
                    best_score = score
                    best_z = current_z
            else:
                worse_in_row += 1

            # 3× po sobě horší? vrať se a jdi na jemnější krok
            if worse_in_row >= overshoot_backtrack:
                if verbose:
                    print(f"[AF]   {overshoot_backtrack}× horší v řadě -> návrat na nejlepší a jemnější krok.")
                # návrat na doposud nejlepší v úrovni
                goto_z(local_best_z)
                current_z = local_best_z
                break  # přejdeme na další (jemnější) level

        # Po ukončení levelu se drž lokálního maxima
        goto_z(local_best_z)
        current_z = local_best_z
        if verbose:
            print(f"[AF] Level {level} best: Z={local_best_z:.6f} mm, score={local_best_score:.2f}")

    # Na konci nastav globální nejlepší
    goto_z(best_z)
    if verbose:
        print(f"[AF] DONE -> Z={best_z:.6f} mm, score={best_score:.2f}")
    return best_z, best_score


