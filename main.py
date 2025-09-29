from core.database import init_db
from gui.main_window import launch_main_window
from core.logger import logger


if __name__ == "__main__":
    init_db()
    # Spustí hlavní okno aplikace
    launch_main_window()
