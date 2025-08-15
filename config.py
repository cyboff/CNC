from core.settings import get_setting, set_setting
import json
import numpy as np

# === Obecné nastavení programu ===
APP_NAME = "CNC Sample Detector"
VERSION = "0.9.5"
COMPANY = "S.S.K. a.s."

# === Cesty ===
PROJECTS_DIR = "projects"
DATABASE_PATH = "data/database.db"

# === Kamera ===
CAMERA_IPS = json.loads(get_setting("CAMERA_IPS"))

# GRBL / CNC nastavení
CNC_SERIAL_PORT = get_setting("CNC_SERIAL_PORT")
CNC_BAUDRATE = int(get_setting("CNC_BAUDRATE"))
CNC_TIMEOUT = 1  # v sekundách
CNC_RTS = get_setting("CNC_RTS") == "False"
CNC_DTR = get_setting("CNC_DTR") == "False"

# === Výchozí rozměry okna ===
def safe_int(value, default=1000):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

WINDOW_WIDTH = safe_int(get_setting("WINDOW_WIDTH"), 1500)
WINDOW_HEIGHT = safe_int(get_setting("WINDOW_HEIGHT"), 1000)

# Rozměry výstupního rámce (např. zobrazení videa)
frame_width = int(get_setting("frame_width"))
frame_height = int(get_setting("frame_height"))

# Rozlišení kamery (v pixelech)
image_width = int(get_setting("image_width"))
image_height = int(get_setting("image_height"))

# Korekční matice pro transformaci perspektivy
correction_matrix = np.array(json.loads(get_setting("correction_matrix")))
correction_matrix_grbl = np.array(json.loads(get_setting("correction_matrix_grbl")))

# Autofocus kroky (od hrubého po jemný)
autofocus_steps = json.loads(get_setting("autofocus_steps"))

# Výchozí pozice osy Z (např. výška mikroskopu)
default_Z_position = float(get_setting("default_Z_position"))
last_Z_position = default_Z_position  # lze měnit runtime

# Předdefinované pozice vzorků (center point)
sample_positions_mm = json.loads(get_setting("sample_positions_mm"))

# Předdefinované pozice rohů kalibračního obrázku (pro kalibraci GRBL)
calib_z = -56.800
calib_corners_grbl = np.array(json.loads(get_setting("calib_corners_grbl")))