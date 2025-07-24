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


def find_sample_positions(image_label, tree, project_id: int, sample_codes: list[str]):

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
            print("[FIND] Detekuji pozice vzorků...")
            # TODO: Detekce kruhů nebo kontur pro vzorky
            # DUMMY: Nakresli červený kruh doprostřed obrázku misto detekce
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            cv2.circle(img, center, 40, (0, 0, 255), 4)
            items = find_circles(img)  # Detekce kruhů na obrázku
            for x, y, r in items:
                cv2.circle(img, (x,y), r, (0, 255, 0), 1)
            # DUMMY: Přidáme dummy položky pro detekované dráty
            # items = [(1, -100, -200, 20), (2, -110, -202, 15)]  # Dummy počet detekovaných drátů
            # Převod na RGB pro PIL a zobrazení v náhledu
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
            tree.insert("", "end", values=(f"{code}", f"{pos}", f"{len(items)}"))
            # Uložit vzorek do databáze
            save_project_sample_to_db(project_id, pos, code)
            #TODO: Přidat detekované a dráty do databáze
        time.sleep(2)  # Počkáme, aby se obrázek načetl a zobrazil
        core.camera_manager.start_camera_preview(image_label, update_position_callback=None)  # Restart živého náhledu

    return sample_positions

def find_circles(image):
    """
    Detekuje kruhy na obrázku pomocí Houghovy transformace.
    Vrací seznam středů kruhů [(x1, y1), (x2, y2), ...].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        return [(circle[0], circle[1], circle[2]) for circle in circles[0, :]]
    return []