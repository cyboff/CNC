# find_process.py
"""
Modul pro najíždění na základní pozici a detekci vzorků pomocí hlavní kamery.
Vrací seznam souřadnic vzorků [(x1, y1), (x2, y2), ...].
"""
import config
from core.logger import logger
from core.motion_controller import move_to_home_position, move_to_coordinates
from core.database import save_sample_positions_to_db
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

    print("[FIND] Najíždím na výchozí pozici (0,0)...")
    move_to_home_position()
    sample_positions = []
    for code in sample_codes:
        sample_position = config.sample_positions_mm[sample_codes.index(code)]
        print(f"[FIND] Najíždím na pozici vzorku {code}: {sample_position}")
        (x, y, z) = sample_position
        print(f"[FIND] Najíždím na pozici vzorku {code}: ({x}, {y})")
        sample_positions.append((x, y))
        move_to_coordinates(x, y, z)
        print("[FIND] Snímám fotku hlavní kamerou...")
        # TODO: Vyfotit a zpracovat obrázek (zatím dummy)

        print("[FIND] Detekuji pozice vzorků...")
        # TODO: Detekce kontur, výpočet pozic (zatím dummy pozice)

    # Uložit pozice do databáze
    save_sample_positions_to_db(sample_codes, sample_positions)

    return sample_positions