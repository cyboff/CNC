# find_process.py
"""
Modul pro najíždění na základní pozici a detekci vzorků pomocí hlavní kamery.
Vrací seznam souřadnic vzorků [(x1, y1), (x2, y2), ...].
"""
from os import waitpid

import cv2
import numpy as np
from ttkbootstrap.dialogs import Messagebox
import config
import core.camera_manager
import gui.find_samples
from core.logger import logger
from core.motion_controller import move_to_coordinates, grbl_abort, grbl_clear_alarm
from core.database import save_project_sample_to_db, save_sample_items_to_db
from PIL import Image, ImageTk
import time
from core.project_manager import save_image_to_project


# Měření ostrosti obrázku Laplacianem není spolehlivé, Tenengrad metoda je mnohem lepší
def tenengrad_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx ** 2 + gy ** 2
    return np.mean(fm)

def find_sample_positions(container, image_label, tree, project_id: int, sample_codes: list[str]):
    sample_positions = []
    items = []  # Počet detekovaných drátů pro každý vzorek
    if core.camera_manager.actual_camera is None or core.camera_manager.actual_camera == core.camera_manager.microscope:
        core.camera_manager.switch_camera()

    while not gui.find_samples.stop_event.is_set(): # Kontrola, zda není proces přerušen, například při stisku tlačítka Opakovat hledání
        for code in sample_codes:
            sample_position = config.sample_positions_mm[sample_codes.index(code)]
            (pos, x, y, z) = sample_position
            print(f"[FIND] Najíždím na pozici {pos} vzorku {code}: ({x}, {y}, {z})")
            move_to_coordinates(x, y, z)
            print("[FIND] Snímám fotku hlavní kamerou...")
            core.camera_manager.preview_running = False # Zastavíme živý náhled, abychom mohli získat snímek
            time.sleep(0.5)  # Počkáme, aby se proces náhledu zastavil

            img = core.camera_manager.get_image()
            if img is not None:
                # Aplikujeme korekční matici pro perspektivní transformaci
                img = cv2.warpPerspective(img, config.correction_matrix, (int(config.image_width), int(config.image_height)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Převod na RGB pro PIL

                # Detekce kruhů na obrázku
                print(f"[FIND] Detekuji počet a pozici drátů ve vzorku {code}")
                items = find_circles(img)

                for i, (x, y, r) in enumerate(items):
                    cv2.putText(img, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
                    cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
                    cv2.circle(img, (x,y), r, (0, 0, 255), 2)
                # Uložíme obrázek do projektu
                image_path = save_image_to_project(project_id, img, f"sample_{code}_position_{pos}.jpg")
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
                sample_id = save_project_sample_to_db(project_id, pos, code, image_path)  # Uložíme vzorek do databáze
                save_sample_items_to_db(sample_id, items)  # Uložíme detekované dráty do databáze
            time.sleep(2.0)  # Počkáme, aby se obrázek načetl a zobrazil
            core.camera_manager.start_camera_preview(image_label, update_position_callback=None)  # Restart živého náhledu
            time.sleep(0.5)
        return sample_positions
    pass

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
                    circles.append((x, y, r-1))

    if circles is not None:
        return circles
    return []

def get_microscope_images(container, image_label, project_id, position, ean_code, items):
    """
    Převádí detekované kruhy na GRBL příkazy pro pohyb.
    """
    # TODO: Upravit přesnost pohybu podle potřeby (asi na 0.5 mm)
    precision = 2.0 # Přesnost pohybu v mm
    # precision = 0.5  # Přesnost pohybu v mm
    sample_position = next((t for t in config.sample_positions_mm if t[0] == position), None)
    (pos, mpos_x, mpos_y, mpos_z) = sample_position
    logger.info(f"[MICROSCOPE] Získávám mikroskopické obrázky pro pozici {position} vzorku {ean_code} s {len(items)} detekovanými dráty.")
    print(f"[MICROSCOPE] Získávám mikroskopické obrázky pro pozici {sample_position} s {len(items)} detekovanými dráty.")
    for (id, pos_index, x_center, y_center, radius) in items:
        grbl_center = cv2.perspectiveTransform(np.array([[[x_center, y_center]]], dtype=np.float32), config.correction_matrix_grbl)[0][0]
        grbl_radius = cv2.perspectiveTransform(np.array([[[x_center+radius, y_center]]], dtype=np.float32), config.correction_matrix_grbl)[0][0]
        abs_x = grbl_center[0] + mpos_x
        abs_y = grbl_center[1] + mpos_y
        abs_z = mpos_z
        abs_r = grbl_radius[0] - grbl_center[0]  # Vypočítáme absolutní poloměr kruhu
        # Přesuneme mikroskop nad střed drátu (jen pro ladění)
        # core.motion_controller.move_to_position(abs_x, abs_y, abs_z)
        # core.camera_manager.preview_running = False  # Zastavíme živý náhled, abychom mohli získat snímek
        # time.sleep(0.2)  # Počkáme, aby se proces náhledu zastavil
        for step in range(0, int(2 * np.pi * abs_r / precision)):
            angle = (step / (2 * np.pi * abs_r / precision)) * 2 * np.pi
            px = round(abs_x + abs_r * np.cos(angle), 3)
            py = round(abs_y + abs_r * np.sin(angle), 3)
            core.motion_controller.move_to_position(px, py, abs_z-1.0) # Posuneme mikroskop o 1 mm pod vzorek
            time.sleep(0.5)
            print(f"[FIND] Získávám snímek {step} z mikroskopu...")
            # Pomalu posunujeme mikroskop na výšku o 0,5 mm nad vzorek a získáváme obrázky
            max_sharpness = 0
            sharpest_img = None
            # Posunujeme mikroskop o 1 mm nad vzorek
            core.motion_controller.send_gcode(f"G90 G1 Z{abs_z+1} M3 S750 F5")
            time.sleep(0.75) # Počkáme, aby se obnovila odpověď z GRBL
            while core.motion_controller.grbl_status != "Idle":
                img = core.camera_manager.get_image()
                if img is not None:
                    # Aplikujeme korekční matici pro perspektivní transformaci
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Převod na RGB pro PIL
                    (h,w) = img.shape[:2]
                    # Vypočítáme ostrost obrázku pomocí Tenengrad metody
                    sharpness = tenengrad_sharpness(img)
                    print(f"[MICROSCOPE] Ostrost obrázku {step}: {sharpness:.3f}, max:{max_sharpness:.3f}")
                    if sharpness > max_sharpness:
                        max_sharpness = sharpness
                        sharpest_img = img.copy()  # Uložíme nejostřejší obrázek
                    if sharpness < max_sharpness * 0.6:  # Pokud ostrost klesne pod 60% max ostrosti, ukončíme snímání
                        print(f"[MICROSCOPE] Ostrost klesla pod 80% max ostrosti ({sharpness:.3f} < {max_sharpness * 0.8:.3f}), ukončuji snímání.")
                        # TODO: Je to potřeba pořádně ověřit, zda přerušení funguje správně
                        grbl_abort()
                        grbl_clear_alarm()
                else:
                    print("[MICROSCOPE] Chyba při získávání snímku z mikroskopu, obrázek je None.")
            if max_sharpness > 0 and sharpest_img is not None:
                print(f"[MICROSCOPE] Nejostřejší obrázek pro krok {step}: {max_sharpness:.3f}")
                # Uložíme nejostřejší obrázek do projektu
                image_path = save_image_to_project(project_id, sharpest_img, f"microscope_{ean_code}_{pos_index}_{step}.jpg")
                # Uložíme snímek do databáze
                core.database.save_sample_item_positions_to_db(id, step, px, py, image_path)
                # Zobrazíme nejostřejší obrázek v GUI
                img = cv2.resize(sharpest_img, (int(w // 4), int(h // 4)))  # Změna velikosti na rozměry náhledu
                im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=im_pil)
                if image_label.winfo_exists():
                    image_label.imgtk = imgtk  # Uchovat referenci, aby obrázek nezmizel
                    image_label.config(image=imgtk)
                else:
                    print("[MICROSCOPE] Náhled již neexistuje, nemohu zobrazit obrázek.")
    container.after(0, lambda: Messagebox.show_info(f"Snímky z mikroskopu pro vzorek {ean_code} na pozici {position} byly úspěšně získány."))