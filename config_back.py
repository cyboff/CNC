import numpy as np

# === Obecné nastavení programu ===
APP_NAME = "WDS - Wire Defect Scanner"
VERSION = "1.0.0"
COMPANY = "S.S.K. a.s."

# === Cesty ===
PROJECTS_DIR = "projects"
DATABASE_PATH = "data/database.db"

# === Kamera ===
CAMERA_IPS = ["192.168.0.101", "192.168.0.102"]

# === Výchozí rozměry okna ===
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 1000

# GRBL / CNC nastavení
CNC_SERIAL_PORT = '/dev/tty.usbmodem11201'  # nebo COM5 na Windows
CNC_BAUDRATE = 115200
CNC_TIMEOUT = 1  # v sekundách
CNC_DTR = False
CNC_RTS = False


# Rozměry výstupního rámce (např. zobrazení videa)
frame_width = 700
frame_height = 700

# Rozlišení kamery (v pixelech)
image_width = 1400
image_height = 1400

# Velikost zorného pole kamery (v mm)
Xmm = 38.0
Ymm = 38.0

# Přepočtové faktory z pixelů na mm
fXmm = Xmm / image_width
fYmm = Ymm / image_height

# Korekční matice pro transformaci perspektivy
correction_matrix = np.array([
    [ 1.01692104e+00, -4.50384611e-02, -7.96564445e+01],
    [-5.89281626e-03,  1.00546177e+00, -1.28204110e+02],
    [-5.68544053e-07, -5.31396415e-05,  1.00000000e+00]
])

# Autofocus kroky (od hrubého po jemný)
autofocus_steps = [0.1, 0.01, 0.005, 0.001]

# Výchozí pozice osy Z (např. výška mikroskopu)
default_Z_position = -55.9
last_Z_position = default_Z_position  # lze měnit runtime

# Offset hlavní kamery vůči nástroji
camera_offset_x = -186.275
camera_offset_y = -199.863

# Offset mikroskopu vůči nástroji
microscope_offset_x = -128.433
microscope_offset_y = -144.920

# Spočítaný offset mezi středem kamery a mikroskopu
# (pro kontrolu nebo 2D zarovnání)
offXmm = 38.6
offYmm = 34.77


# Předdefinované pozice vzorků (center point)
sample_positions_mm = {
    "A1": (-197.0, 210.0, default_Z_position),
    "A2": (-153.0, 210.0, default_Z_position),
    "B1": (-197.0, 165.0, default_Z_position),
    "B2": (-153.0, 165.0, default_Z_position),
}

# Speciální pozice (např. pro pin)
pin_camera_position = (-215.930, -229.450, default_Z_position)
pin_microscope_position = (-157.920, -174.750, -54.660)