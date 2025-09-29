from core.settings import get_setting
import json
import numpy as np
import ctypes

# === Obecné nastavení programu ===
APP_NAME = "WDS - Wire Defect Scanner"
VERSION = "0.9.7"
COMPANY = "S.S.K. a.s."

# === Cesty ===
PROJECTS_DIR = get_setting("PROJECTS_DIR")

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

WINDOW_WIDTH = safe_int(get_setting("WINDOW_WIDTH"), 1800)
WINDOW_HEIGHT = safe_int(get_setting("WINDOW_HEIGHT"), 900)

# Rozměry výstupního rámce (např. zobrazení videa)
frame_width = min(safe_int(get_setting("frame_width"), 1500), int(WINDOW_WIDTH * 0.75))
frame_height = min(safe_int(get_setting("frame_height"), 1500), int(WINDOW_HEIGHT * 0.75))

# Rozlišení hlavní kamery (v pixelech)
image_width = int(get_setting("image_width"))
image_height = int(get_setting("image_height"))

# Korekční matice pro transformaci perspektivy
correction_matrix = np.array(json.loads(get_setting("correction_matrix")))
correction_matrix_grbl = np.array(json.loads(get_setting("correction_matrix_grbl")))

# Počet vynechaných bodů pro interpolaci na kontuře drátu pro mikroskopické snímky
# - čím vyšší číslo, tím méně snímků na obvodu drátu
precision = int(get_setting("precision")) # Pro 5x objektiv - 20, pro 10x objektiv - 12

# Výchozí pozice osy Z (např. výška mikroskopu)
default_Z_position = float(get_setting("default_Z_position"))
last_Z_position = default_Z_position  # lze měnit runtime

# Předdefinované pozice vzorků (center point)
sample_positions_mm = json.loads(get_setting("sample_positions_mm"))

# Předdefinované pozice rohů kalibračního obrázku (pro kalibraci GRBL)
calib_z = float(get_setting("calib_z"))
calib_corners_grbl = np.array(json.loads(get_setting("calib_corners_grbl")))

# Anti-backlash nastavení pro pojezdy
anti_backlash_axes = "XYZ"
anti_backlash_mm = 0.02
anti_backlash_final_dir = {'X': +1, 'Y': +1, 'Z': +1}

def reload_settings():
    global PROJECTS_DIR
    global CAMERA_IPS, camera_exposure_time, microscope_exposure_time, microscope_exposure_time_calib
    global CNC_SERIAL_PORT, CNC_BAUDRATE
    global WINDOW_WIDTH, WINDOW_HEIGHT
    global frame_width, frame_height
    global image_width, image_height
    global correction_matrix, correction_matrix_grbl
    global precision
    global default_Z_position
    global sample_positions_mm
    global calib_z, calib_corners_grbl

    PROJECTS_DIR = get_setting("PROJECTS_DIR")
    CAMERA_IPS = json.loads(get_setting("CAMERA_IPS"))
    camera_exposure_time = int(get_setting("camera_exposure_time"))  # v milisekundách
    microscope_exposure_time = int(get_setting("microscope_exposure_time"))  # v milisekundách
    microscope_exposure_time_calib = int(get_setting("microscope_exposure_time_calib"))  # pro kalibrační obraz

    CNC_SERIAL_PORT = get_setting("CNC_SERIAL_PORT")
    CNC_BAUDRATE = int(get_setting("CNC_BAUDRATE"))

    WINDOW_WIDTH = safe_int(get_setting("WINDOW_WIDTH"), 1800)
    WINDOW_HEIGHT = safe_int(get_setting("WINDOW_HEIGHT"), 900)

    frame_width = min(safe_int(get_setting("frame_width"), 1500), int(WINDOW_WIDTH * 0.75))
    frame_height = min(safe_int(get_setting("frame_height"), 1500), int(WINDOW_HEIGHT * 0.75))

    image_width = int(get_setting("image_width"))
    image_height = int(get_setting("image_height"))

    correction_matrix = np.array(json.loads(get_setting("correction_matrix")))
    correction_matrix_grbl = np.array(json.loads(get_setting("correction_matrix_grbl")))

    precision = int(get_setting("precision")) # Pro 5x objektiv - 20, pro 10x objektiv - 12

    default_Z_position = float(get_setting("default_Z_position"))

    sample_positions_mm = json.loads(get_setting("sample_positions_mm"))

    calib_z = float(get_setting("calib_z"))
    calib_corners_grbl = np.array(json.loads(get_setting("calib_corners_grbl")))