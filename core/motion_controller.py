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
io_lock = threading.RLock()
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
    try:
        cnc_serial.open()
    except Exception as e:
        import tkinter.messagebox as mbox
        mbox.showerror("Chyba při otevírání portu", f"Nepodařilo se otevřít sériový port:\n{e}")
        raise

    time.sleep(2)
    cnc_serial.reset_input_buffer()

    def update_position():
        global grbl_last_position, grbl_status, position_lock

        try:
            grbl_last_position, grbl_status = grbl_update_position()
            # print(f"[GRBL] Aktuální pozice: {grbl_last_position}, Stav: {grbl_status}")
        except:
            print("Chyba při aktualizaci pozice GRBL")
        position_timer = threading.Timer(0.5, update_position) #update pozice každých 0.5s, častěji nestíhá Arduino GRBL odpovídat
        position_timer.daemon = True
        position_timer.start()

    update_position() # spustí periodické aktualizace pozice

    try:

        if grbl_last_position != "0.000,0.000,0.000":
            x, y, z = [float(val) for val in grbl_last_position.split(",")]
            print(f"[GRBL] Stav:{grbl_status} , Pozice (MPos): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
        else:
            print("MPos nenalezena, provedu Homing a nastavím na výchozí hodnoty")
            grbl_abort()
            time.sleep(1)
            grbl_clear_alarm()
            grbl_home()
    except:
        print("Chyba inicializace GRBL")

def send_gcode(command: str):
    global cnc_serial, grbl_status, grbl_last_position

    if cnc_serial is None or not cnc_serial.is_open:
        print("[GRBL] Port není otevřený")

    with io_lock:  # držíme linku exkluzivně
        print(f"[GRBL] Posílám: {command}")
        cnc_serial.flush()
        for attempt in range(3):
            try:
                cnc_serial.write((command + "\n").encode())
                break
            except serial.SerialTimeoutException:
                print(f"[GRBL] Timeout při zápisu (pokus {attempt + 1})")
                time.sleep(0.5)
        else:
            print("[GRBL] Nepodařilo se odeslat po 3 pokusech.")
            return

        # Čti odpovědi až do 'ok' (nebo do vypršení krátkého okna)
        t0 = time.time()
        while time.time() - t0 < 0.8:
            line = cnc_serial.readline().decode(errors='ignore').strip()
            if not line:
                break
            # print(f"[GRBL] Odpověď: {line}")
            if line == "ok":
                break

        # >>> NOVÉ: hned po příkazu udělej rychlé '?' pro čerstvý stav
        try:
            cnc_serial.write(b'?')
            cnc_serial.flush()
            t1 = time.time()
            while time.time() - t1 < 0.4:
                if cnc_serial.in_waiting:
                    decoded = cnc_serial.readline().decode(errors='ignore').strip()
                    # Stav
                    if "Idle" in decoded:
                        grbl_status = "Idle"
                    elif "Run" in decoded:
                        grbl_status = "Run"
                    elif ("Error" in decoded) or ("error" in decoded):
                        grbl_status = "Error"
                    elif ("Alarm" in decoded) or ("alarm" in decoded):
                        grbl_status = "Alarm"
                    # MPos
                    if 'MPos:' in decoded:
                        for part in decoded.split('|'):
                            if part.startswith('MPos:'):
                                grbl_last_position = part[5:].strip()
                                break
                    break
                time.sleep(0.01)
        except Exception:
            pass  # když se to nepovede, nevadí – příští dotaz to dožene

def _get_final_dir(axis: str) -> int:
    """Směr finálního dojezdu pro osu (+1 nebo -1). Lze přepsat v configu jako dict anti_backlash_final_dir."""
    try:
        import config
        d = getattr(config, "anti_backlash_final_dir", None)
        if isinstance(d, dict) and axis in d and d[axis] in (-1, 1):
            return int(d[axis])
    except Exception:
        pass
    return 1  # default: +X, +Y, +Z

def _get_ab_axes() -> set:
    """Které osy řešit anti-backlash (např. 'XY' nebo 'XYZ')."""
    try:
        import config
        s = getattr(config, "anti_backlash_axes", "XY")
        return set(str(s).upper())
    except Exception:
        return set("XY")

def _get_ab_amount() -> float:
    """Velikost předpětí v mm (kolik 'přejedu' proti vůli před finálním dojezdem)."""
    try:
        import config
        v = float(getattr(config, "anti_backlash_mm", 0.02))
        return max(0.0, v)
    except Exception:
        return 0.02


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
        print("🏠 GRBL Home odesláno")
    except Exception as e:
        print("⚠️  Chyba zasílání Home:", e)
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
    time.sleep(0.5)

def grbl_abort():
    """
    Nouzové přerušení (ctrl-x)
    """
    global cnc_serial
    send_gcode("\x18")  # Odeslání Ctrl-X jako G-code příkazu
    time.sleep(0.5)  # Krátká prodleva pro stabilitu
    print("[GRBL] Abort odeslán (Ctrl-X)")

def _send_realtime(byte_cmd: bytes):
    """
    Odeslání real-time příkazu do GRBL (např. b'!', b'~', b'\x85').
    Nepřidává newline a drží io_lock, aby se to nepralo s další I/O.
    """
    global cnc_serial, io_lock
    if cnc_serial is None or not cnc_serial.is_open:
        print("[GRBL] Port není otevřený")
        return
    with io_lock:
        try:
            cnc_serial.write(byte_cmd)
            cnc_serial.flush()
        except Exception as e:
            print(f"[GRBL] Chyba při odesílání real-time příkazu: {e}")

def feed_hold():
    """Okamžitý řízený stop bez ztráty pozice (stav -> Hold)."""
    _send_realtime(b'!')

def resume_motion():
    """Pokračování po feed-holdu (Cycle Start)."""
    _send_realtime(b'~')

def jog_cancel():
    """
    Zruší rozpracovaný pohyb a vyprázdní běžící akci bez resetu.
    V GRBL 1.1 se používá 0x85 jako 'Jog Cancel' i pro ukončení hold stavu bez ztráty pozice.
    """
    _send_realtime(b'\x85')

def cancel_move(timeout: float = 3.0):
    """
    Okamžitě zastaví právě probíhající jogovací pohyb ($J=...).
    -> Pošle realtime Jog Cancel (0x85) a čeká na Idle.
    -> Nepoužívá reset (Ctrl-X), takže se neztratí pozice.
    """
    global grbl_status

    # 1) Pošleme Jog Cancel
    jog_cancel()

    # 2) Počkej na Idle (status aktualizuje timer v init_grbl)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if grbl_status == "Idle":
            print("[GRBL] cancel_move(): jog zrušen, CNC je v Idle.")
            return
        time.sleep(0.05)

    print("[GRBL] cancel_move(): timeout – GRBL není Idle!")


# Absolutní pohyb bez anti-backlashe
def move_to_position(x: float, y: float, z: float = None, feed: float = 2000):
    global grbl_status
    if z is None:
        z = default_Z_position
    if feed is None:
        feed = 2000
    send_gcode(f"G90 G1 X{x:.3f} Y{y:.3f} Z{z:.3f} M3 S750 F{feed}")  # G90 je absolutní pohyb, F500 je rychlost posuvu
    grbl_wait_for_idle() # Počkej na dokončení pohybu

# Absolutní pohyb s anti-backlashem
# (předpětí proti vůli v závitové tyči, které se použije před finálním dojezdem)
# -> pro každou osu vytvoříme 'approach' (předpětí proti vůli) a hned poté finální krátký dojezd
# -> všechny bloky pošleme rychle za sebou (čekáme na ok), na konci případně jednou počkáme na Idle
# -> pokud je anti_backlash=False, tak se použije jen finální dojezd
def move_to_position_antibacklash(x: float, y: float, z: float = None, *, anti_backlash: bool = True,
                     wait_end: bool = True, feed_xy: int = 2000, feed_z: int = 500):
    """
    Absolutní najíždění s volitelným anti-backlashem.
    -> pro každou osu vytvoříme 'approach' (předpětí proti vůli) a hned poté finální krátký dojezd
    -> všechny bloky pošleme rychle za sebou (čekáme na ok), na konci případně jednou počkáme na Idle
    """
    global grbl_status, grbl_last_position
    from config import default_Z_position

    if z is None:
        z = default_Z_position

    # světlo / modalita
    send_gcode("M3 S750")
    send_gcode("G90")  # absolutní režim

    # aktuální pozice (best-effort)
    try:
        with position_lock:
            cx, cy, cz = [float(v) for v in grbl_last_position.split(",")]
    except Exception:
        cx, cy, cz = 0.0, 0.0, 0.0

    AB_AXES = _get_ab_axes()
    AB_MM   = _get_ab_amount()
    eps     = 1e-6

    gcode_queue = []

    def plan_axis(axis: str, current: float, target: float, feed: int):
        """Naplánuje approach + final pro jednu osu bez čekání; vrátí nový 'current'."""
        if abs(target - current) <= eps:
            return current
        if anti_backlash and AB_MM > 0.0 and axis in AB_AXES:
            dirF = _get_final_dir(axis)  # +1 nebo -1
            approach = target - dirF * AB_MM
            # Bezpečný malý clamp: když už jsme 'za' approach ve směru finálního dojezdu, approach přeskoč
            need_approach = (dirF > 0 and approach > current + eps) or (dirF < 0 and approach < current - eps)
            if need_approach:
                if axis == 'X':
                    gcode_queue.append(f"G1 X{approach:.3f} F{feed}")
                elif axis == 'Y':
                    gcode_queue.append(f"G1 Y{approach:.3f} F{feed}")
                else:
                    gcode_queue.append(f"G1 Z{approach:.3f} F{feed}")
        # finální krátký dojezd vždy
        if axis == 'X':
            gcode_queue.append(f"G1 X{target:.3f} F{feed}")
        elif axis == 'Y':
            gcode_queue.append(f"G1 Y{target:.3f} F{feed}")
        else:
            gcode_queue.append(f"G1 Z{target:.3f} F{feed}")
        return target

    # naplánuj osy postupně (per-axis), ale bez průběžného čekání na Idle
    cx = plan_axis('X', cx, x, feed_xy)
    cy = plan_axis('Y', cy, y, feed_xy)
    cz = plan_axis('Z', cz, z, feed_z)

    # odešli všechny bloky do GRBL (send_gcode čeká jen na 'ok', ne na fyzické dojetí)
    for line in gcode_queue:
        send_gcode(line)

    # a teprve teď jednou počkat na dokončení celé sekvence (pokud chceme)
    if wait_end:
        grbl_wait_for_idle()  # jednorázově na závěr celé sady pohybů


def move_to_home_position():
    print("[MOTION] Najíždím na výchozí pozici (-245, -245)")
    move_to_position(-245, -245)  # Použijeme výchozí pozici Mpos pro GRBL, která je -245, -245

def move_to_coordinates(x: float, y: float, z: float = None, feed: float = 2000 ):
    move_to_position(x, y, z, feed)

def move_relative(dx: float, dy: float):
    gcode = f"G91\nG0 X{dx:.3f} Y{dy:.3f}"
    send_gcode(gcode)


def grbl_update_position():
    """
    Pošle '?' a načte odpověď od GRBL (stav + MPos).
    Pokud je linka právě obsazená, zkusí chvilku počkat, jinak vrátí cache.
    """
    global cnc_serial, position_lock, io_lock, grbl_status, grbl_last_position

    # krátký timeout místo úplného non-blockingu (eliminuje zpoždění po send_gcode)
    if not io_lock.acquire(timeout=0.2):
        return grbl_last_position, grbl_status

    try:
        cnc_serial.write(b'?')
        cnc_serial.flush()
        t0 = time.time()
        while time.time() - t0 < 0.6:
            if cnc_serial.in_waiting:
                decoded = cnc_serial.readline().decode(errors='ignore').strip()

                if "Idle" in decoded:
                    grbl_status = "Idle"
                elif "Run" in decoded:
                    grbl_status = "Run"
                elif ("Error" in decoded) or ("error" in decoded):
                    grbl_status = "Error"
                elif ("Alarm" in decoded) or ("alarm" in decoded):
                    grbl_status = "Alarm"

                if 'MPos:' in decoded:
                    for part in decoded.split('|'):
                        if part.startswith('MPos:'):
                            with position_lock:
                                grbl_last_position = part[5:].strip()
                            break
                return grbl_last_position, grbl_status

            time.sleep(0.02)

        # Timeout – cache necháme jak je (žádné "Unknown")
        return grbl_last_position, grbl_status
    finally:
        io_lock.release()

def grbl_wait_for_idle():
    global grbl_status
    """
    Čeká, dokud GRBL nepřijde do stavu Idle (stav získá z threadu position_timer v init_grbl()).
    Zamezí se tím opakování dotazů na GRBL stav přes sériovou linku.
    """
    # print("[GRBL] Waiting for Idle:", grbl_status)
    time.sleep(0.5)  # Stav CNC se updatuje každých 0.5s, takže počkáme 0.5s, aby se stihl aktualizovat
    while True:
        if grbl_status == "Idle":
            break
        time.sleep(0.5)