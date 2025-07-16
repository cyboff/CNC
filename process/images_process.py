from core.logger import logger

def capture_all_samples(project_id: int, positions: list):
    print("[IMAGE] Přepínám na mikroskop...")
    # TODO: přepnout na mikroskop

    for i, (x, y) in enumerate(positions):
        print(f"[IMAGE] ({i+1}/{len(positions)}) Najíždím na pozici {x},{y}...")
        # TODO: ovládání pohybů

        print("[IMAGE] Zaostřuji...")
        # TODO: autofokus nebo algoritmus zaostření

        print("[IMAGE] Snímám obrázek...")
        # TODO: Uložit obrázek do složky projektu

    print("[IMAGE] Všechna snímání dokončena")



def run_autofocus():

    logger.info("run_autofocus")
    # TODO: Ovládání motorů – motors.move_to(0, 0)


def run_fine_focus():

    logger.info("run_fine_focus")
    # TODO: Ovládání motorů – motors.move_to(0, 0)