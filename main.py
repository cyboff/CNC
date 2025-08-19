from core.database import init_db
from gui.main_window import launch_main_window
from core.logger import logger
from core.motion_controller import init_grbl
import threading

if __name__ == "__main__":
    init_db()
    # Není třeba čekat na dokončení inicializace GRBL, homing atd. se provede v samostatném vlákně
    # Spustí se v samostatném vlákně, aby GUI hned otevřelo a uživatel mohl začít pracovat
    threading.Thread(target=init_grbl, daemon=True).start()
    launch_main_window()
    logger.info("Program spuštěn")
    # Spustí hlavní okno aplikace