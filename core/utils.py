import socket
import subprocess
import platform
import threading
import ttkbootstrap as ttk
from PIL import Image, ImageTk
from core.logger import logger
from config import APP_NAME, VERSION, COMPANY

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

    status_label = ttk.Label(footer, text="Systém připraven.", font=("Arial", 12), foreground="#d8e2f1", background="#48484c")
    status_label.pack(side="right", padx=12)