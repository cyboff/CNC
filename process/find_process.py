# find_process.py
"""
Modul pro najíždění na základní pozici a detekci vzorků pomocí hlavní kamery.
Vrací seznam souřadnic vzorků [(x1, y1), (x2, y2), ...].
"""
import config
from core.logger import logger
from core.motion_controller import move_to_home_position, move_to_coordinates
from core.database import save_project_samples_to_db
from config import camera_offset_x, camera_offset_y, microscope_offset_x, microscope_offset_y

def move_to_sample_center(x: float, y: float):
    """
    Najede mikroskopem na střed vzorku s ohledem na offsety kamery a mikroskopu.
    """
    real_x = x + camera_offset_x + microscope_offset_x
    real_y = y + camera_offset_y + microscope_offset_y
    print(f"[FIND] Najíždím na upravenou pozici: ({real_x}, {real_y})")
    move_to_coordinates(real_x, real_y)


def find_sample_positions(project_id: int, sample_codes: list[str]):
    global grbl_status

    sample_positions = []
    items = []  # Počet detekovaných drátů pro každý vzorek
    for code in sample_codes:
        sample_position = config.sample_positions_mm[sample_codes.index(code)]
        (name, x, y, z) = sample_position
        print(f"[FIND] Najíždím na pozici {name} vzorku {code}: ({x}, {y}, {z})")
        move_to_coordinates(x, y, z)
        print("[FIND] Snímám fotku hlavní kamerou...")
        # TODO: Vyfotit a zpracovat obrázek (zatím dummy)

        print("[FIND] Detekuji pozice vzorků...")
        # TODO: Detekce kontur, výpočet pozic (zatím dummy pozice)
        items = [1,2,3] # Dummy počet detekovaných drátů
        sample_positions.append((name,items))  # Přidat pozici a počet detekovaných drátů

    # Uložit pozice do databáze
    save_project_samples_to_db(project_id, sample_codes, sample_positions)

    return sample_positions