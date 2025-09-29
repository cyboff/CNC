import ttkbootstrap as ttk

import core.motion_controller
from core.camera_manager import init_cameras
import threading
from core.motion_controller import cnc_serial, init_grbl
from gui.new_project_wizard import open_new_project_wizard
from gui.log_viewer import show_log_view
from gui.styles import apply_styles
from gui.show_projects import show_projects, open_project_detail
from gui.manual_controller import show_manual_controller
from gui.settings_window import show_settings
from core.logger import logger
from core.database import get_all_projects
from core.utils import center_window, create_header, create_footer, add_nav_button
from config import APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT
import sys

def launch_main_window():
    root = ttk.Window(themename="morph")    # vytvoříme hlavní okno programu
    root.title(APP_NAME)

    apply_styles()                                  # aplikujeme globální styly
    center_window(root,WINDOW_WIDTH,WINDOW_HEIGHT)  # nastaveníme velikost a pozice okna

    container = ttk.Frame(root)                     # vytvoří kontejner pro obsah
    container.pack(fill="both", expand=True)

    logger.info("Aplikace spuštěna")                # logovací funkce
    show_home(container)                            # zobrazíme home stránku

    # Spustíme GUI a teprve potom inicializujeme GRBL a kamery na pozadí
    # (nečeká se na dokončení inicializace bez GUI a informací o stavu zařízení)
    threading.Thread(target=init_grbl, daemon=True).start()
    threading.Thread(target=init_cameras, daemon=True).start()

    root.mainloop()
    # Po zavření hlavního okna ukonči všechny spuštěné procesy
    if core.motion_controller.position_timer is not None:
        logger.info("Zastavuji periodické aktualizace pozice")
        core.motion_controller.position_timer.cancel()
    if core.motion_controller.cnc_serial.is_open:
        logger.info("Zavírám sériový port CNC")
        core.motion_controller.cnc_serial.close()  # zavře sériový port CNC
    sys.exit(0)

def show_home(container):
    for widget in container.winfo_children():
        widget.destroy()

    logger.info("Zobrazení domovské obrazovky")

    create_header(container, APP_NAME)
    create_footer(container)


    # === HLAVNÍ OBSAH ===
    content = ttk.Frame(container)
    content.pack(fill="both", expand=True, padx=20, pady=10)


    # LEFT PANEL – obalený kvůli odsazení odshora
    container_left_wrapper = ttk.Frame(content)
    container_left_wrapper.pack(side="left", fill="y", expand=False, padx=(0, 10))

    left_panel = ttk.Frame(container_left_wrapper)
    left_panel.pack(pady=40)  # posun tlačítek dolů


    add_nav_button(left_panel, "➕ Nové měření",        lambda: (logger.info("Klik: Nové měření"),       open_new_project_wizard(container, lambda: show_home(container))))
    add_nav_button(left_panel, "📂 Otevřít měření",     lambda: (logger.info("Klik: Otevřít měření"),    show_projects(container, lambda: show_home(container))))
    add_nav_button(left_panel, "🛠️ Manuální ovládání",  lambda: (logger.info("Klik: Manuální ovládání"), show_manual_controller(container, lambda: show_home(container))))
    add_nav_button(left_panel, "⚙️ Nastavení",          lambda: (logger.info("Klik: Nastavení"),         show_settings(container, lambda: show_home(container))))
    add_nav_button(left_panel, "🔍 Log akcí",           lambda: (logger.info("Klik: Log akcí"),          show_log_view(container, lambda: show_home(container))))



    # --- Pravá část: tabulka ---

    right_panel = ttk.Frame(content)
    right_panel.pack(side="left", fill="both", expand=True, padx=(10, 0))


    ttk.Label(right_panel, text="Poslední měření:", font=("Helvetica", 16), foreground="#2b3b4c").pack(pady=(10, 5))

    style = ttk.Style()

    # Upravíme záhlaví tabulky
    style.configure("Treeview.Heading", font=("Helvetica", 14, "bold"), background="#2b3b4c", foreground="white")

    # A styl pro samotná data
    style.configure("Treeview", font=("Helvetica", 14), rowheight=28)

    tree_frame = ttk.Frame(right_panel, padding=10)
    tree_frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(tree_frame, columns=("ID", "Název", "Komentář", "Vytvořeno"), show="headings", height=8,
                        style="Treeview")
    for col in ("ID", "Název", "Komentář", "Vytvořeno"):
        tree.heading(col, text=col)

    tree.column("ID", width=50, anchor="center")
    tree.column("Název", width=180)
    tree.column("Komentář", width=280)
    tree.column("Vytvořeno", width=180, anchor="center")
    tree.pack(fill="both", expand=True)

    all_projects = get_all_projects()
    # recent_projects = all_projects[0:25] if len(all_projects) > 25 else all_projects
    for row in all_projects:
        tree.insert("", "end", values=row)

    def on_double_click(event):
        item = tree.selection()
        if item:
            values = tree.item(item[0], "values")
            project_id = values[0]
            logger.info(f"Dvojklik na projekt ID {project_id}")
            open_project_detail(container, project_id, lambda: show_home(container))

    tree.bind("<Double-1>", on_double_click)
