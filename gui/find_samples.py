from tkinter import ttk, StringVar, Label, Frame, Button

from ttkbootstrap.dialogs import Messagebox

import config
from core.utils import create_back_button, create_header, create_footer
from core.logger import logger
from core.motion_controller import move_axis, grbl_home, grbl_clear_alarm, grbl_abort
import core.motion_controller
from process.find_process import find_sample_positions
from core.camera_manager import start_camera_preview, switch_camera
import core.camera_manager
import threading

update_position_timer = None

def show_sample_detector(container, project_id, samples, on_back):
    for widget in container.winfo_children():
        widget.destroy()
    print(f"Krok 3: Spouštím detekci vzorků pro projekt {project_id} s {len(samples)} vzorky")
    create_header(container, "CNC Sample Detector - Krok 3: Hledání vzorků", on_back)
    create_footer(container)
    create_back_button(container, on_back)

    main_frame = ttk.Frame(container)
    main_frame.pack(fill="both", expand=True)

    # VPRAVO – kamera
    preview_frame = ttk.Frame(main_frame)
    preview_frame.pack(side="right", fill="both", expand=True)

    Label(preview_frame, text="Zobrazení kamery", font=("Helvetica", 14, "bold")).pack()

    image_label = Label(preview_frame, width=int(config.frame_width), height=int(config.frame_height))
    image_label.pack()

    # === POZICE STROJE ===

    # Live pozice
    position_label = Label(preview_frame, text="Status: ---, X: ---, Y: ---, Z: ---", font=("Helvetica", 10))
    position_label.pack(pady=6)

    def update_position():
        try:
            last_position = core.motion_controller.grbl_last_position
            grbl_status = core.motion_controller.grbl_status
            x, y, z = map(float, last_position.split(","))
            position_label.config(text=f"Stav: {grbl_status}, X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}")
        except Exception:
            pass
        position_timer = threading.Timer(0.5, update_position)
        position_timer.daemon = True
        position_timer.start()

    core.camera_manager.preview_running = False
    start_camera_preview(image_label, update_position_callback=update_position)
    update_position()  # Spustí periodické aktualizace pozice

    # Spustí hledání vzorků ve vlákně a zobrazí výsledky v tabulce
    def threaded_find_and_show():
        positions = find_sample_positions(project_id, samples)
        container.after(0, lambda: Messagebox.show_info(f"Detekovány vzorky: {len(positions)}. Zobrazují se v tabulce níže."))

        results_frame = ttk.Frame(main_frame)
        results_frame.pack(side="left",fill="both", expand=True, padx=10, pady=10)
        ttk.Label(results_frame, text="Výsledky hledání vzorků:", font=("Helvetica", 14, "bold")).pack(pady=(10, 5))
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Helvetica", 14, "bold"), background="#2b3b4c", foreground="white")
        style.configure("Treeview", font=("Helvetica", 14), rowheight=28)
        tree = ttk.Treeview(results_frame, columns=("Sample", "X", "Y"), show="headings", height=8,
                            style="Treeview")
        tree.heading("Sample", text="Vzorek")
        tree.heading("X", text="X")
        tree.heading("Y", text="Y")
        tree.column("Sample", width=100, anchor="center")
        tree.column("X", width=100, anchor="center")
        tree.column("Y", width=100, anchor="center")
        tree.pack(fill="both", expand=True)
        for i, pos in enumerate(positions):
            tree.insert("", "end", values=(f"Vzorek {samples[i]}", f"{pos[0]:.3f}", f"{pos[1]:.3f}"))

    threading.Thread(target=threaded_find_and_show, daemon=True).start()

    # Zobrazí výsledky v tabulce


    # Přepínač kamery


