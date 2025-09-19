from core.settings import get_setting, set_setting
import json
import numpy as np

# === Obecné nastavení programu ===
APP_NAME = "WDS - Wire Defect Scanner"
VERSION = "0.9.7"
COMPANY = "S.S.K. a.s."

# === Cesty ===
PROJECTS_DIR = get_setting("PROJECTS_DIR") == "projects"
DATABASE_PATH = get_setting("DATABASE_PATH") == "data/database.db"

# === Kamera ===
CAMERA_IPS = json.loads(get_setting("CAMERA_IPS"))
camera_exposure_time = int(get_setting("camera_exposure_time"))  # v milisekundách
microscope_exposure_time = int(get_setting("microscope_exposure_time"))  # v milisekundách
microscope_exposure_time_calib = int(get_setting("microscope_exposure_time_calib"))  # pro kalibrační obraz

# GRBL / CNC nastavení
CNC_SERIAL_PORT = get_setting("CNC_SERIAL_PORT")
CNC_BAUDRATE = int(get_setting("CNC_BAUDRATE"))
CNC_TIMEOUT = 1  # v sekundách
CNC_RTS = "False"
CNC_DTR = "False"

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

# Výchozí pozice osy Z (např. výška mikroskopu)
default_Z_position = float(get_setting("default_Z_position"))
last_Z_position = default_Z_position  # lze měnit runtime

# Předdefinované pozice vzorků (center point)
sample_positions_mm = json.loads(get_setting("sample_positions_mm"))

# Předdefinované pozice rohů kalibračního obrázku (pro kalibraci GRBL)
calib_z = float(get_setting("calib_z"))
calib_corners_grbl = np.array(json.loads(get_setting("calib_corners_grbl")))

anti_backlash_axes = "XYZ"
anti_backlash_mm = 0.02
anti_backlash_final_dir = {'X': +1, 'Y': +1, 'Z': +1}