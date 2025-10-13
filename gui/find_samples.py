import threading
import time
from tkinter import ttk
from ttkbootstrap.dialogs import Messagebox
import config
import core.camera_manager
import core.motion_controller
from core.utils import create_header, create_footer, create_camera_preview, show_image
from process.find_process import find_sample_positions
from core.logger import logger
from gui.microscope_images import show_microscope_images
from core.database import delete_sample_items_from_project

positions = []
update_position_timer = None
stop_event = threading.Event()
stop_event.clear()


def show_find_samples(container, project_id, on_back):
    global stop_event
    for widget in container.winfo_children():
        widget.destroy()
    print(f"Krok 3: Spouštím detekci vzorků pro projekt {project_id}")
    create_header(container, "WDS - Wire Defect Scanner - Krok 3: Hledání vzorků", on_back)
    create_footer(container)

    main_frame = ttk.Frame(container)
    main_frame.pack(fill="both", expand=True)

    # VLEVO - Vytvoří tabulku s výsledky
    results_frame = ttk.Frame(main_frame)
    results_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    ttk.Label(results_frame, text="Výsledky hledání vzorků:", font=("Helvetica", 14, "bold")).pack(pady=(10, 5))
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Helvetica", 14, "bold"), background="#2b3b4c", foreground="white")
    style.configure("Treeview", font=("Helvetica", 14), rowheight=28)
    tree = ttk.Treeview(results_frame, columns=("Sample", "Position", "Detected Items"), show="headings", height=8,
                        style="Treeview")
    tree.heading("Sample", text="Vzorek")
    tree.heading("Position", text="Pozice")
    tree.heading("Detected Items", text="Detekované dráty")
    tree.column("Sample", width=100, anchor="center")
    tree.column("Position", width=100, anchor="center")
    tree.column("Detected Items", width=150, anchor="center")
    tree.pack(fill="both", expand=True)

    def on_click(event):
        item = tree.selection()
        if item:
            values = tree.item(item[0], "values")
            ean_code = values[0]
            position = values[1]
            show_image(image_label, project_id, ean_code, position)

    tree.bind("<ButtonRelease-1>", on_click)

    # VPRAVO – kamera
    core.camera_manager.preview_running = False
    preview_frame, image_label, position_label = create_camera_preview(
        main_frame,
        config.frame_width,
        config.frame_height,
        lambda: (core.motion_controller.grbl_last_position, core.motion_controller.grbl_status),
        core.camera_manager.start_camera_preview
    )

    # Spustí hledání vzorků ve vlákně a zobrazí výsledky v tabulce

    def threaded_find_and_show(container, image_label, tree, project_id):
        # Po vložení kazety provedeme pro jistotu homing
        t = threading.Thread(target=core.motion_controller.grbl_home, daemon=True)
        t.start()
        t.join()
        positions = find_sample_positions(container, image_label, tree, project_id)
        container.after(0, lambda: Messagebox.show_info(f"Detekovány {len(positions)} vzorky. Výsledky jsou v tabulce."))
        def stop_camera_preview():
            core.camera_manager.stop_camera_preview()
            print("[FIND] Živý náhled kamery byl zastaven.")
        container.after(0, stop_camera_preview)  # Zastaví živý náhled kamery po dokončení hledání
        def show_image_first_row():
            if tree.get_children():
                first_item = tree.get_children()[0]
                tree.selection_set(first_item)
                values = tree.item(first_item, "values")
                ean_code = values[0]
                position = values[1]
                show_image(image_label, project_id, ean_code, position)
        container.after(2000, show_image_first_row)  # Zobrazí první řádek v tabulce po dokončení hledání

    stop_event.clear()
    t = threading.Thread(target=threaded_find_and_show, args=(container, image_label, tree, project_id), daemon=True)
    t.start()

    # Přidání tlačítek pro opakování detekce a pokračování na mikroskop
    button_frame = ttk.Frame(container)
    button_frame.pack(pady=10)

    def restart_sample_detector(thread, container, image_label, tree, project_id):
        global stop_event
        logger.info(f"[FIND] Opakuji hledání vzorků pro projekt {project_id}")
        stop_event.set()
        while thread.is_alive():
            print("[FIND] Proces hledání vzorků již běží, čekám na dokončení...")
            time.sleep(1)
        stop_event.clear()
        # Smaže všechny samples z databáze pro tento projekt, pokud existují
        if tree.winfo_exists():
            container.after(0, lambda: [tree.delete(item) for item in tree.get_children()])
        delete_sample_items_from_project(project_id)
        # Restart živého náhledu
        core.camera_manager.start_camera_preview(image_label, update_position_callback=None)
        time.sleep(0.5)
        t = threading.Thread(target=threaded_find_and_show, args=(container, image_label, tree, project_id), daemon=True)
        t.start()
        return t

    ttk.Button(button_frame,text="Opakuj hledání",bootstyle="success",command=lambda: restart_sample_detector(t,container, image_label, tree, project_id)).pack(side="left", padx=10)

    def start_show_microscope_images(container, project_id, on_back):
        if not tree.get_children():
            Messagebox.show_error("Musíte detekovat alespoň 1 vzorek.")
            return
        logger.info(f"[FIND] Spouštím proces MICROSCOPE pro projekt {project_id}")
        show_microscope_images(container, project_id, on_back)

    ttk.Button(button_frame, text="Pokračovat na snímání mikroskopem", bootstyle="success", command=lambda: start_show_microscope_images(container, project_id, on_back)).pack(side="left", padx=10)