from core.settings import get_setting
import json
import numpy as np

# === Obecné nastavení programu ===
APP_NAME = "CNC Sample Detector"
VERSION = "0.9.0"
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

correction_matrix = np.array(json.loads(get_setting("correction_matrix")))

# === Výchozí rozměry okna ===
def safe_int(value, default=1000):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

WINDOW_WIDTH = safe_int(get_setting("WINDOW_WIDTH"), 1500)
WINDOW_HEIGHT = safe_int(get_setting("WINDOW_HEIGHT"), 1000)

# Rozměry výstupního rámce (např. zobrazení videa)
frame_width = float(get_setting("frame_width"))
frame_height = float(get_setting("frame_height"))

# Rozlišení kamery (v pixelech)
image_width = float(get_setting("image_width"))
image_height = float(get_setting("image_height"))

# Velikost zorného pole kamery (v mm)
Xmm = float(get_setting("Xmm"))
Ymm = float(get_setting("Ymm"))

# Přepočtové faktory z pixelů na mm
fXmm = Xmm / image_width
fYmm = Ymm / image_height

# Korekční matice pro transformaci perspektivy
correction_matrix = np.array(json.loads(get_setting("correction_matrix")))

# Autofocus kroky (od hrubého po jemný)
autofocus_steps = get_setting("autofocus_steps")

# Výchozí pozice osy Z (např. výška mikroskopu)
default_Z_position = float(get_setting("default_Z_position"))
last_Z_position = default_Z_position  # lze měnit runtime

# Offset hlavní kamery vůči nástroji
camera_offset_x = float(get_setting("camera_offset_x"))
camera_offset_y = float(get_setting("camera_offset_y"))

# Offset mikroskopu vůči nástroji
microscope_offset_x = float(get_setting("microscope_offset_x"))
microscope_offset_y = float(get_setting("microscope_offset_y"))

# Spočítaný offset mezi středem kamery a mikroskopu (pro kontrolu nebo 2D zarovnání)
offXmm = float(get_setting("offXmm"))
offYmm = float(get_setting("offYmm"))

# Předdefinované pozice vzorků (center point)
sample_positions_mm = [
    ("A1", -197.0, -210.0, default_Z_position),
    ("A2", -153.0, -210.0, default_Z_position),
    ("B1", -197.0, -165.0, default_Z_position),
    ("B2", -153.0, -165.0, default_Z_position),
]

# Speciální pozice (např. pro pin)
pin_camera_position = (-215.930, -229.450, default_Z_position)
pin_microscope_position = (-157.920, -174.750, -54.660)