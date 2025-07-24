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


update_position_timer = None

def show_sample_detector(container, project_id, samples, on_back):
    for widget in container.winfo_children():
        widget.destroy()
    print(f"Krok 3: Spouštím detekci vzorků pro projekt {project_id} s {len(samples)} vzorky")
    create_header(container, "CNC Sample Detector - Krok 3: Hledání vzorků", on_back)
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

    threading.Thread(target=threaded_find_and_show, args=(image_label, tree, project_id, samples), daemon=True).start()