from core.database import init_db
from gui.main_window import launch_main_window
from core.logger import logger
from core.motion_controller import init_grbl

if __name__ == "__main__":
    init_db()
    init_grbl()
    launch_main_window()
    logger.info("Program spuštěn")