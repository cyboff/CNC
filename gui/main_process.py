# main_process.py
import logging
from core.logger import logger
from process import find_process, images_process, detect_process

def run_full_measurement(project_id, sample_codes):
    logger.info(f"Spouštím celý proces měření pro projekt ID {project_id} se vzorky: {sample_codes}")

    # 1. Najetí na výchozí pozici
    find_process.move_to_home_position()

    # 2. Nalezení pozic všech vzorků pomocí kamery
    sample_positions = find_process.find_sample_positions(sample_codes)

    # 3. Snímkování mikroskopem
    images_process.capture_all_samples(project_id, sample_positions)

    # 4. (Volitelně) detekce vad
    detect_process.detect_defects(project_id)

    logger.info("Proces měření úspěšně dokončen.")

# Volání této funkce by se provádělo z GUI:
# run_full_measurement(project_id, ["code1", "code2", "code3"])
