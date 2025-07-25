import threading
from tkinter import ttk, Label

from ttkbootstrap.dialogs import Messagebox

import config
import core.camera_manager
import core.motion_controller
from core.camera_manager import start_camera_preview
from core.motion_controller import move_axis
from core.utils import create_back_button, create_header, create_footer, create_camera_preview
from process.find_process import find_sample_positions
from PIL import Image, ImageTk
import cv2
from core.logger import logger


update_position_timer = None

def show_microscope_images(container, project_id, samples, on_back):
    for widget in container.winfo_children():
        widget.destroy()
    print(f"Krok 4: Spouštím snímaní mikroskopem pro projekt {project_id} s {len(samples)} vzorky")
    create_header(container, "CNC Sample Detector - Krok 4: Snímání mikroskopem", on_back)
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
    tree.column("Detected Items", width=100, anchor="center")
    tree.pack(fill="both", expand=True)

    def show_image(project_id, image_label, sample_id, position):
        print(f"[FIND] Zobrazuji náhled vzorku {sample_id} na pozici {position}")
        filename = f"sample_{sample_id}_position_{position}.jpg"
        img = core.project_manager.get_image_from_project(project_id, filename)
        if img is None:
            print(f"[FIND] Obrázek {filename} nebyl nalezen v projektu {project_id}.")
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
                print("[FIND] Náhled již neexistuje, nemohu zobrazit obrázek.")

    def on_double_click(event):
        item = tree.selection()
        if item:
            values = tree.item(item[0], "values")
            sample_id = values[0]
            position = values[1]
            show_image(project_id, image_label, sample_id, position)

    tree.bind("<Double-1>", on_double_click)

    # VPRAVO – kamera
    core.camera_manager.preview_running = False
    preview_frame, image_label, position_label = create_camera_preview(
        main_frame,
        config.frame_width,
        config.frame_height,
        lambda: (core.motion_controller.grbl_last_position, core.motion_controller.grbl_status),
        start_camera_preview
    )

    # Spustí hledání vzorků ve vlákně a zobrazí výsledky v tabulce
    def threaded_find_and_show(image_label, tree, project_id, samples):
        positions = find_sample_positions(image_label, tree, project_id, samples)
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
                sample_id = values[0]
                position = values[1]
                show_image(project_id, image_label, sample_id, position)
        container.after(1000, show_image_first_row)  # Zobrazí první řádek v tabulce po dokončení hledání
    threading.Thread(target=threaded_find_and_show, args=(image_label, tree, project_id, samples), daemon=True).start()