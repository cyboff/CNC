# realtime_basler_fiducials_stable.py
# Realtime detekce 4 rohových fiducialů (kroužek + křížek) z Basler kamery.
# - Stabilizace: EMA (ALPHA), deadband (MIN_DELTA) a "lock-on" do lokální ROI (LOCK_ROI)
# - Cílový čtverec: 40 x 40 mm, středy kroužků 2 mm od okrajů
# - Klávesy:  q = konec,  s = uložit homografii (px->mm) do homography_px_to_mm.npy
#
# Závislosti: pip install pypylon opencv-python numpy

import cv2
import numpy as np
import time
from pypylon import pylon

# ---------- Parametry vzoru a kamery ----------
IMG_W, IMG_H = 1500, 1500           # cílové rozlišení kamery
SQUARE_MM = 40.0                     # velikost čtverce (strana)
RADIUS_MM = 1.0                      # kroužek ve vzoru má r = 1 mm (SVG)
OFFSET_MM = 2.0                      # středy kroužků jsou 2 mm od hrany
PIX_PER_MM = IMG_W / SQUARE_MM       # 1500 / 40 = 37.5 px/mm (přibližně)
RADIUS_PX = int(round(RADIUS_MM * PIX_PER_MM))            # ~38 px
R_TOL = 0.5                          # ±50 % tolerance poloměru
MIN_R, MAX_R = int(RADIUS_PX*(1-R_TOL)), int(RADIUS_PX*(1+R_TOL))
ROI_RATIO = 0.20                     # velikost rohových ROI (20 % šířky/výšky)
PREVIEW = (750, 750)                 # náhledové okno

# ---------- Stabilizace ----------
ALPHA = 0.25           # síla vyhlazení EMA (0..1); menší = klidnější
LOCK_ROI = 100         # lokální ROI kolem poslední pozice (px) při "lock-on"
MIN_DELTA = 0.3        # minimální změna, jinak držím starou pozici (deadband)

def smooth_point(prev, new, alpha=ALPHA):
    if prev is None:
        return new
    return ((1 - alpha) * prev[0] + alpha * new[0],
            (1 - alpha) * prev[1] + alpha * new[1])

def clamp_roi(x0, y0, x1, y1, w, h):
    x0 = max(0, min(x0, w - 1)); x1 = max(1, min(x1, w))
    y0 = max(0, min(y0, h - 1)); y1 = max(1, min(y1, h))
    if x1 <= x0 + 1: x1 = min(w, x0 + 2)
    if y1 <= y0 + 1: y1 = min(h, y0 + 2)
    return x0, y0, x1, y1

# ---------- Basler ----------
def set_if_exists(node, value):
    try:
        node.SetValue(value)
    except Exception:
        try:
            node(value)
        except Exception:
            pass

def open_basler():
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("Nenalezena žádná Basler kamera.")
    cam = pylon.InstantCamera(tl_factory.CreateFirstDevice())
    cam.Open()

    # Požadovaná nastavení
    set_if_exists(cam.PixelFormat, "Mono8")
    set_if_exists(cam.ExposureAuto, "Off")
    set_if_exists(cam.GainAuto, "Off")
    set_if_exists(cam.GammaEnable, False)
    set_if_exists(cam.GainRaw, 0.0)
    # ExposureTime / ExposureTimeAbs (µs)
    for key in ["ExposureTime", "ExposureTimeAbs"]:
        try:
            getattr(cam, key).SetValue(15000.0)  # cca 15 ms
            break
        except Exception:
            pass

    cam.Width.SetValue(min(cam.Width.GetMax(), IMG_W))
    cam.Height.SetValue(min(cam.Height.GetMax(), IMG_H))
    try:
        cam.CenterX.SetValue(True)
        cam.CenterY.SetValue(True)
    except Exception:
        pass

    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return cam

def grab(cam, timeout_ms=1000):
    res = cam.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
    if res and res.GrabSucceeded():
        arr = res.Array
        res.Release()
        return arr
    if res:
        res.Release()
    return None

# ---------- Geometrie ROI ----------
def get_corner_rois(w, h, ratio=ROI_RATIO):
    tw, th = int(w * ratio), int(h * ratio)
    # Pořadí: TL, TR, BR, BL
    return [
        (0, 0, tw, th),
        (w - tw, 0, w, th),
        (w - tw, h - th, w, h),
        (0, h - th, tw, h),
    ]

# ---------- Detekce kruhu v ROI ----------
def detect_circle_in_roi(gray, roi_box, seed=None):
    x0, y0, x1, y1 = roi_box
    # Pokud máme předchozí pozici, zmenšíme ROI kolem ní
    if seed is not None:
        cx, cy = seed
        rr = LOCK_ROI // 2
        x0, y0, x1, y1 = clamp_roi(int(cx - rr), int(cy - rr),
                                   int(cx + rr), int(cy + rr),
                                   gray.shape[1], gray.shape[0])

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)

    # Hough (čekáme "kroužek"; i když je prstencový, gradient najde vnější kruh)
    circles = cv2.HoughCircles(
        roi_blur, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(roi.shape) // 2,
        param1=120, param2=25,
        minRadius=max(MIN_R, 8),
        maxRadius=max(MAX_R, 10)
    )
    if circles is None:
        return None

    circles = np.uint16(np.around(circles[0]))
    # Preferuj kruh nejblíž seed pozici; bez seed nejblíž oček. poloměru
    if seed is not None:
        sx, sy = seed[0] - x0, seed[1] - y0
        dpos = np.hypot(circles[:, 0] - sx, circles[:, 1] - sy)
        i = int(np.argmin(dpos))
    else:
        diffs = np.abs(circles[:, 2] - RADIUS_PX)
        i = int(np.argmin(diffs))

    c = circles[i]
    cx, cy, r = int(c[0]) + x0, int(c[1]) + y0, int(c[2])
    return float(cx), float(cy), float(r)

# ---------- Detekce všech čtyř rohů ----------
def detect_four_corners(gray, last_centers=None):
    h, w = gray.shape
    rois = get_corner_rois(w, h)
    centers = []
    for i, rb in enumerate(rois):
        seed = None
        if last_centers is not None and last_centers[i] is not None:
            seed = last_centers[i]  # lock-on lokální ROI

        res = detect_circle_in_roi(gray, rb, seed=seed)
        if res is None:
            # když nic nenajdeme, držíme poslední (klidnější dočasně),
            # případně nastavte na None pokud chcete "strictní" chování
            centers.append(last_centers[i] if last_centers else None)
            continue

        new_pt = (res[0], res[1])

        # Deadband – ignoruj miniaturní posun
        if last_centers is not None and last_centers[i] is not None:
            dx = abs(new_pt[0] - last_centers[i][0])
            dy = abs(new_pt[1] - last_centers[i][1])
            if dx < MIN_DELTA and dy < MIN_DELTA:
                centers.append(last_centers[i])
                continue

        # EMA vyhlazení
        centers.append(smooth_point(last_centers[i] if last_centers else None, new_pt))
    return centers  # [TL, TR, BR, BL]

# ---------- Homografie (px -> mm) ----------
def homography_from_centers(centers):
    # Cílové body v mm ve stejném pořadí [TL, TR, BR, BL]
    dst_mm = np.array([[OFFSET_MM, OFFSET_MM],
                       [SQUARE_MM - OFFSET_MM, OFFSET_MM],
                       [SQUARE_MM - OFFSET_MM, SQUARE_MM - OFFSET_MM],
                       [OFFSET_MM, SQUARE_MM - OFFSET_MM]], dtype=np.float32)
    valid = [(i, c) for i, c in enumerate(centers) if c is not None]
    if len(valid) < 4:
        return None, None
    src = np.array([c for _, c in valid], dtype=np.float32)
    dst = dst_mm[[i for i, _ in valid]]
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    return H, mask

# ---------- Vizualizace ----------
def draw_ui(gray, centers, fps=None, H_ok=False):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # rohové ROI
    for (x0, y0, x1, y1) in get_corner_rois(gray.shape[1], gray.shape[0]):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
    # body
    labels = ["TL", "TR", "BR", "BL"]
    for i, c in enumerate(centers):
        if c is None:
            continue
        x, y = int(round(c[0])), int(round(c[1]))
        cv2.circle(vis, (x, y), 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{labels[i]} {x},{y}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # FPS + stav homografie
    txt = f"FPS: {fps:.1f}" if fps else ""
    txt += "  |  H: OK" if H_ok else "  |  H: --"
    cv2.putText(vis, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

    vis_small = cv2.resize(vis, PREVIEW, interpolation=cv2.INTER_AREA)
    return vis_small

# ---------- Hlavní smyčka ----------
def main():
    cam = open_basler()
    print("Běží realtime detekce. Klávesy: 'q' = konec, 's' = uložit homografii (npy).")
    last_centers = [None, None, None, None]
    H_cache = None
    last_t = time.time()
    fps = 0.0

    try:
        while cam.IsGrabbing():
            img = grab(cam, 1000)
            if img is None:
                continue

            # Mono8 -> gray
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            centers = detect_four_corners(gray, last_centers=last_centers)
            last_centers = centers

            H, mask = homography_from_centers(centers)
            if H is not None:
                H_cache = H

            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps = 1.0 / dt
            last_t = now

            vis = draw_ui(gray, centers, fps=fps, H_ok=(H is not None))
            cv2.imshow("Fiducials (preview 750x750)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s') and H_cache is not None:
                np.save("homography_px_to_mm.npy", H_cache)
                print("Uloženo: homography_px_to_mm.npy")
    finally:
        try:
            cam.StopGrabbing()
            cam.Close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
