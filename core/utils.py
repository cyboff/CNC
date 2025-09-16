import socket
import subprocess
import platform
import threading
import ttkbootstrap as ttk
from PIL import Image, ImageTk
from pyexpat.errors import messages

from core.database import get_samples_by_project_id
from core.logger import logger
from config import APP_NAME, VERSION, COMPANY
import config
from tkinter import Label
import cv2
from core.project_manager import get_image_from_project
import core.camera_manager

def create_step_header(parent, text: str):
    """Vytvoří záhlaví kroku s popisným textem."""
    header_frame = ttk.Frame(parent)
    header_frame.pack(fill="x", padx=10, pady=(10, 0))

    label = ttk.Label(header_frame, text=text, style="Step.TLabel")
    label.pack(side="left", padx=10)

    return header_frame


def get_local_ip():
    """Získá lokální IP adresu zařízení."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "N/A"


def ping_ip(ip, callback):
    """Pingne IP adresu asynchronně a zavolá callback(ip, success)."""
    def run_ping():
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", ip]
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL)
            success = result.returncode == 0
        except Exception:
            success = False
        callback(ip, success)

    threading.Thread(target=run_ping, daemon=True).start()


def create_back_button(parent, command, text="⬅️ Zpět"):
    """Vytvoří jednotné tlačítko zpět umístěné vpravo nahoře."""
    top_bar = ttk.Frame(parent)
    top_bar.pack(fill="x", padx=10, pady=5)

    ttk.Button(top_bar,text=text,style="Back.TButton",width=10,command=command).pack(side="right")

    return top_bar


def add_nav_button(parent, text, command):
    button = ttk.Button(parent, text=text, style="Main.TButton", command=command,cursor="hand2")
    button.pack(pady=10, ipadx=20, ipady=10)
    button.configure(width=25)  # nebo .place(width=250), ale width=25 s fontem 16 odpovídá cca 250px


def center_window(window, width=1500, height=1000):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")


def create_header(container, title="CNC Sample Detector", on_back=None):
    header = ttk.Frame(container, style="Header.TFrame")
    header.pack(fill="x", pady=(0, 10))

    logo_path = "assets/logo.png"
    try:
        image = Image.open(logo_path).resize((120, 40))
        logo = ImageTk.PhotoImage(image)
        logo_label = ttk.Label(header, image=logo, background="#2b3c4c")
        logo_label.image = logo
        logo_label.pack(side="left", padx=15, pady=10)
    except Exception as e:
        logger.warning(f"Logo se nepodařilo načíst: {e}")

    ttk.Label(header, text=title, font=("Helvetica", 24, "normal"), foreground="white", background="#2b3c4c").pack(side="left", padx=10)

    if on_back:
        ttk.Button(header, text="← Zpět", command=on_back, style="Back.TButton").pack(side="right", padx=15, pady=10)



def create_footer(container):
    footer = ttk.Frame(container)
    footer.pack(side="bottom", fill="x")
    footer.configure(style="Custom.TFrame")

    # Obsah footeru
    footer_text = f"version: {VERSION}  |  Developed by {COMPANY}"
    ttk.Label(footer, text=footer_text, font=("Arial", 12), foreground="#d8e2f1", background="#48484c").pack(side="left", padx=10, pady=5)

    status_label = ttk.Label(footer, text="Inicializace systému...", font=("Arial", 12), foreground="#d8e2f1", background="#48484c")
    status_label.pack(side="right", padx=12)

    def update_status():
        if status_label.winfo_exists():
            if core.motion_controller.grbl_status is ( "Idle" or "Run" ):
                cnc_status = "Připojeno"
            else:
                cnc_status = f"Stav: {core.motion_controller.grbl_status}"
            if core.camera_manager.camera is not None:
                camera_status = "Připojena"
            else:
                camera_status = "Není připojena"
            if core.camera_manager.microscope is not None:
                microscope_status = "Připojen"
            else:
                microscope_status = "Není připojen"
            message = f"CNC: {cnc_status} | Kamera: {camera_status} | Mikroskop: {microscope_status}"
            status_label.config(text=message)
            footer.after(5000, update_status)

    status_timer = threading.Timer(5, update_status)
    status_timer.daemon = True
    status_timer.start()

def create_camera_preview(parent, frame_width, frame_height, get_position, start_preview):
    preview_frame = ttk.Frame(parent)
    preview_frame.pack(side="right", fill="both", expand=True)

    Label(preview_frame, text="Zobrazení kamery", font=("Helvetica", 14, "bold")).pack()
    image_label = Label(preview_frame, width=config.frame_width, height=config.frame_height)
    image_label.pack()

    position_label = Label(preview_frame, text="Status: ---, X: ---, Y: ---, Z: ---", font=("Helvetica", 10))
    position_label.pack(pady=6)

    def update_position():
        try:
            last_position, grbl_status = get_position()
            x, y, z = map(float, last_position.split(","))
            position_label.config(text=f"Stav: {grbl_status}, X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}")
        except Exception:
            pass
        position_timer = threading.Timer(0.5, update_position)
        position_timer.daemon = True
        position_timer.start()

    start_preview(image_label, update_position_callback=update_position)
    update_position()

    return preview_frame, image_label, position_label

def show_image(image_label, project_id, ean, pos):
    print(f"Zobrazuji náhled vzorku {ean} na pozici {pos}")
    samples_from_db = get_samples_by_project_id(project_id)
    for sample_id, position, ean_code, image_path in samples_from_db:
        if ean == ean_code and pos == position:
            break
    else:
        image_path = None
    img = get_image_from_project(image_path)
    if img is None:
        print(f"Obrázek {image_path} nebyl nalezen v projektu {project_id}.")
        return
    else:
        # Zobrazíme náhled obrázku v GUI
        img = cv2.resize(img, (config.frame_width, config.frame_height))  # Změna velikosti na rozměry náhledu
        im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=im_pil)
        if image_label.winfo_exists():
            core.camera_manager.preview_running = False;  # Zastavíme živý náhled kamery
            image_label.imgtk = imgtk  # Uchovat referenci, aby obrázek nezmizel
            image_label.config(image=imgtk)
        else:
            print("Náhled již neexistuje, nemohu zobrazit obrázek.")