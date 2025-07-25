# find_process.py
"""
Modul pro najíždění na základní pozici a detekci vzorků pomocí hlavní kamery.
Vrací seznam souřadnic vzorků [(x1, y1), (x2, y2), ...].
"""
import cv2
import numpy as np
from ttkbootstrap.dialogs import Messagebox

import config
import core.camera_manager
from core.camera_manager import switch_camera, get_image, preview_running
from core.logger import logger
from core.motion_controller import move_to_home_position, move_to_coordinates
from core.database import save_project_sample_to_db
from config import camera_offset_x, camera_offset_y, microscope_offset_x, microscope_offset_y
from PIL import Image, ImageTk
import time
from pypylon import pylon

def move_to_sample_center(x: float, y: float):
    """
    Najede mikroskopem na střed vzorku s ohledem na offsety kamery a mikroskopu.
    """
    real_x = x + camera_offset_x + microscope_offset_x
    real_y = y + camera_offset_y + microscope_offset_y
    print(f"[FIND] Najíždím na upravenou pozici: ({real_x}, {real_y})")
    move_to_coordinates(real_x, real_y)


def find_sample_positions(container, image_label, tree, project_id: int, sample_codes: list[str]):
    sample_positions = []
    items = []  # Počet detekovaných drátů pro každý vzorek
    for code in sample_codes:
        sample_position = config.sample_positions_mm[sample_codes.index(code)]
        (pos, x, y, z) = sample_position
        print(f"[FIND] Najíždím na pozici {pos} vzorku {code}: ({x}, {y}, {z})")
        move_to_coordinates(x, y, z)
        print("[FIND] Snímám fotku hlavní kamerou...")
        core.camera_manager.preview_running = False # Zastavíme živý náhled, abychom mohli získat snímek
        time.sleep(0.2)  # Počkáme, aby se proces náhledu zastavil

        img = core.camera_manager.get_image()
        if img is not None:
            # Aplikujeme korekční matici pro perspektivní transformaci
            img = cv2.warpPerspective(img, config.correction_matrix, (int(config.image_width), int(config.image_height)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Převod na RGB pro PIL
            print(f"[FIND] Detekuji počet a pozici drátů ve vzorku {code}")
            # Detekce kruhů na obrázku - výsledek
            items = find_circles(img)

            for i, (x, y, r) in enumerate(items):
                cv2.putText(img, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
                cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
                cv2.circle(img, (x,y), r, (0, 0, 255), 2)
            # Uložíme obrázek do projektu
            core.project_manager.save_image_to_project(project_id, img, f"sample_{code}_position_{pos}.jpg")
            # Zobrazíme náhled obrázku v GUI
            img = cv2.resize(img, (config.frame_width, config.frame_height))  # Změna velikosti na rozměry náhledu
            im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=im_pil)
            if image_label.winfo_exists():
                image_label.imgtk = imgtk  # Uchovat referenci, aby obrázek nezmizel
                image_label.config(image=imgtk)
            else: print("[FIND] Náhled již neexistuje, nemohu zobrazit obrázek.")
        else:
            print("[FIND] Chyba při získávání snímku z kamery, obrázek je None.")

        if not items:
            print(f"[FIND] Žádné dráty nebyly detekovány na pozici {pos} vzorku {code}.")
            image_label.after(0, lambda: Messagebox.show_warning(
                "Žádné dráty nebyly detekovány",
                f"Na pozici {pos} vzorku {code} nebyly detekovány žádné dráty. Zkontrolujte, zda je vzorek správně umístěn."
            ))
        else:
            print(f"[FIND] Detekováno {len(items)} drátů na pozici {pos} vzorku {code}.")
            sample_positions.append((pos,items))  # Přidat pozici a počet detekovaných drátů
            if tree.winfo_exists():
                container.after(0, lambda: tree.insert("", "end", values=(f"{code}", f"{pos}", f"{len(items)}")))
            # Uložit vzorek do databáze
            save_project_sample_to_db(project_id, pos, code)
            #TODO: Přidat detekované a dráty do databáze
        time.sleep(2.0)  # Počkáme, aby se obrázek načetl a zobrazil
        core.camera_manager.start_camera_preview(image_label, update_position_callback=None)  # Restart živého náhledu
        time.sleep(0.5)
    return sample_positions

def find_circles(image):
    """
    Detekuje kruhy na obrázku pomocí Houghovy transformace.
    Vrací seznam středů kruhů [(x1, y1), (x2, y2), ...].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # gray = cv2.medianBlur(gray, 5)  # Apply median blur to reduce noise
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
    kernel = np.ones((3, 3), np.uint8)
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
    markers = cv2.watershed(image, markers)

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
        return circles
    return []