import serial
import time
import re
import threading
from config import CNC_SERIAL_PORT, CNC_BAUDRATE, default_Z_position
from core.logger import logger


cnc_serial = None  # globální proměnná pro sériový port
position_timer = None

# === Sdílené proměnné ===

position_lock = threading.Lock()
grbl_status = "Unknown"  # Stav GRBL, zda je Idle nebo ne
grbl_last_position = "0.001,0.002,0.003"

def init_grbl():
    global cnc_serial, grbl_last_position, grbl_status, position_lock, position_timer

    cnc_serial = serial.Serial()
    cnc_serial.port = CNC_SERIAL_PORT
    cnc_serial.baudrate = CNC_BAUDRATE
    cnc_serial.timeout = 1
    cnc_serial.dtr = False
    cnc_serial.rts = False
    cnc_serial.open()

    time.sleep(2)
    cnc_serial.reset_input_buffer()

    def update_position():
        global grbl_last_position, grbl_status, position_lock

        try:
            grbl_last_position, grbl_status = grbl_update_position()
            # print(f"[GRBL] Aktuální pozice: {grbl_last_position}, Stav: {grbl_status}")
        except:
            print("Failed to update position")
        position_timer = threading.Timer(0.25, update_position)
        position_timer.daemon = True  # aby se ukončil při zavření programu
        position_timer.start()

    update_position() # spustí periodické aktualizace pozice

    try:

        if grbl_last_position != "0.000,0.000,0.000":
            x, y, z = [float(val) for val in grbl_last_position.split(",")]
            print(f"Machine Position (MPos): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
        else:
            print("MPos not found – trying to home the machine")
            grbl_clear_alarm()
            grbl_home()
    except:
        print("Failed to initialize GRBL")

def send_gcode(command: str):
    """
    Odeslání G-code příkazu do GRBL + vypsání odpovědi.
    """
    global cnc_serial

    if cnc_serial is None or not cnc_serial.is_open:
        print("[GRBL] Port není otevřený")

    print(f"[GRBL] Posílám: {command}")
    cnc_serial.write((command + "\n").encode())

    while True:
        line = cnc_serial.readline().decode().strip()
        if not line:
            break
        print(f"[GRBL] Odpověď: {line}")
        if line == "ok" or line.startswith("error"):
            break


def move_axis(axis: str, value: float):
    """
    Relativní pohyb v jedné ose.
    """
    gcode = f"G91 G1 {axis.upper()}{value:.3f} M3 S750 F2000" # M3 S750 je pro spuštění osvětlení, F2000 je rychlost posuvu
    send_gcode(gcode)

def grbl_home():
    """
    Spustí homing sekvenci ($H)
    """
    global cnc_serial, grbl_last_position, grbl_status, position_lock
    try:
        send_gcode("$H")
        print("🏠 GRBL Home sent")
    except Exception as e:
        print("⚠️  Error sending Home:", e)
        return

    # Počkej na konec homing sekvence
    grbl_wait_for_idle()
    # Aktualizuj WPos pozici na MPos, protože po homingu může být WPos jiná než MPos
    # je lepší pracovat s MPos = WPos, protože MPos přichází v odpovědi na '?' častěji
    try:
        x, y, z = [float(val) for val in grbl_last_position.split(",")]
        send_gcode(f"G10 L20 P1 X{x:.3f} Y{y:.3f} Z{z:.3f}")
        print(f"Nastaveno WPos na MPos: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    except Exception as e:
        print("Chyba při nastavování WPos na MPos:", e)

def grbl_clear_alarm():
    """
    Odblokuje ALARM stav ($X)
    """
    send_gcode("$X")

def grbl_abort():
    """
    Nouzové přerušení (ctrl-x)
    """
    global cnc_serial
    cnc_serial.write(b'\x18')  # Ctrl-X
    print("[GRBL] Abort odeslán (Ctrl-X)")

def move_to_position(x: float, y: float, z: float = None):
    global grbl_status
    if z is None:
        z = default_Z_position
    send_gcode(f"G90 G1 X{x:.2f} Y{y:.2f} Z{z:.2f} M3 S750 F2000")  # G90 je absolutní pohyb, F2000 je rychlost posuvu
    grbl_wait_for_idle() # Počkej na dokončení pohybu

def move_to_home_position():
    print("[MOTION] Najíždím na výchozí pozici (-245, -245)")
    move_to_position(-245, -245)  # Použijeme výchozí pozici Mpos pro GRBL, která je -245, -245

def move_to_coordinates(x: float, y: float, z: float = None):
    move_to_position(x, y, z)

def move_relative(dx: float, dy: float):
    gcode = f"G91\nG0 X{dx:.3f} Y{dy:.3f}"
    send_gcode(gcode)


def grbl_update_position():
    """
    Pošle '?' a načte poslední odpověď od GRBL, uloží do last_position
    """
    global cnc_serial, position_lock, grbl_status, grbl_last_position

    cnc_serial.write(b'?')
    cnc_serial.flush()

    t0 = time.time()
    while time.time() - t0 < 0.5:
        if cnc_serial.in_waiting:
            line = cnc_serial.readline()
            # print("Received: (update_position)", line.decode().strip())
            try:
                decoded = line.decode(errors='ignore').strip()
                # print("[GRBL] Status:", decoded)
                if "Idle" in decoded:
                    grbl_status = "Idle"
                elif "Run" in decoded:
                    grbl_status = "Run"
                elif "Error" in decoded:
                    grbl_status = "Error"
                elif "Alarm" in decoded:
                    grbl_status = "Alarm"
                else:
                    grbl_status = "Unknown"

                if 'MPos:' in decoded:
                    for part in decoded.split('|'):
                        if part.startswith('MPos:'):
                            mpos = part[5:].strip()
                            with position_lock:
                                grbl_last_position = mpos
                                # print("Aktuální CNC pozice:",last_position)
                                return grbl_last_position, grbl_status
            except Exception:
                pass
        time.sleep(0.01)
    grbl_last_position = "0.000,0.000,0.000"
    grbl_status = "Unknown"
    return grbl_last_position, grbl_status

def grbl_wait_for_idle():
    global grbl_status
    """
    Čeká, dokud GRBL nepřijde do stavu Idle (stav získá z threadu position_timer v init_grbl()).
    Zamezí se tím opakování dotazů na GRBL stav přes sériovou linku.
    """
    # print("[GRBL] Waiting for Idle:", grbl_status)
    time.sleep(0.3)  # Stav CNC se updatuje každých 0.25s, takže počkáme 0.3s, aby se stihl aktualizovat
    while True:
        if grbl_status == "Idle":
            break
        time.sleep(0.3)