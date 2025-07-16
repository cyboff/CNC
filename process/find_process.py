# find_process.py
"""
Modul pro najíždění na základní pozici a detekci vzorků pomocí hlavní kamery.
Vrací seznam souřadnic vzorků [(x1, y1), (x2, y2), ...].
"""

from core.logger import logger
from core.motion_controller import move_to_home_position, move_to_coordinates
from core.database import save_sample_positions_to_db
from config import camera_offset_x, camera_offset_y


def move_to_home_position():
    """
    Najetí na výchozí souřadnice (např. 0,0)
    """
    logger.info("[FIND] Najíždím na výchozí pozici (0,0)...")
    # TODO: Ovládání motorů – motors.move_to(0, 0)

def move_to_sample_center(x: float, y: float):
    real_x = x + camera_offset_x
    real_y = y + camera_offset_y
    print(f"[FIND] Korekce pozice středu vzorku pomocí offsetu kamery: +{camera_offset_x}, +{camera_offset_y}")
    print(f"[FIND] Najíždím na upravenou pozici: ({real_x}, {real_y})")
    move_to_coordinates(real_x, real_y)


def find_sample_positions(project_id: int, sample_codes: list[str]):
    print("[FIND] Najíždím na výchozí pozici (0,0)...")
    move_to_home_position()

    print("[FIND] Snímám fotku hlavní kamerou...")
    # TODO: Vyfotit a zpracovat obrázek (zatím dummy)

    print("[FIND] Detekuji pozice vzorků...")
    # TODO: Detekce kontur, výpočet pozic (zatím dummy pozice)
    sample_positions = [(100, 200), (300, 200), (500, 200)]
    print(f"[FIND] Nalezeno {len(sample_positions)} vzorků")

    # Pohyb na střed 1. vzorku s offsetem (reálný pohyb)
    if sample_positions:
        first_x, first_y = sample_positions[0]
        print(f"[FIND] Najíždím na STŘED prvního vzorku: ({first_x}, {first_y})")
        move_to_sample_center(first_x, first_y)

    # Pohyb na každou pozici (testovací logika)
    for idx, (x, y) in enumerate(sample_positions):
        print(f"[FIND] Najíždím na pozici vzorku {idx + 1}: ({x}, {y})")
        move_to_coordinates(x, y)

    # Uložit pozice do databáze
    save_sample_positions_to_db(sample_codes, sample_positions)

    return sample_positions