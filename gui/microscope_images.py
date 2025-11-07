import threading
from tkinter import ttk
import config
import time
import core.camera_manager
import core.motion_controller
from core.database import get_samples_by_project_id, get_sample_items_by_sample_id
from core.utils import create_back_button, create_header, create_footer, create_camera_preview, show_image
from process.find_process import get_microscope_images
from core.logger import logger
from ttkbootstrap.dialogs import Messagebox


update_position_timer = None

def show_microscope_images(container, project_id, on_back):
    for widget in container.winfo_children():
        widget.destroy()
    print(f"Krok 4: Spouštím snímaní mikroskopem pro projekt {project_id}")
    create_header(container, "WDS - Wire Defect Scanner - Krok 4: Snímání mikroskopem", on_back)
    create_footer(container)

    main_frame = ttk.Frame(container)
    main_frame.pack(fill="both", expand=True)

    # VLEVO - Vytvoří tabulku s výsledky
    results_frame = ttk.Frame(main_frame)
    results_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    ttk.Label(results_frame, text="Výsledky snímání vzorků:", font=("Helvetica", 14, "bold")).pack(pady=(10, 5))
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Helvetica", 14, "bold"), background="#2b3b4c", foreground="white")
    style.configure("Treeview", font=("Helvetica", 14), rowheight=28)
    tree = ttk.Treeview(results_frame, columns=("Sample", "Position", "Detected Items"), show="headings", height=8,
                        style="Treeview")
    tree.heading("Sample", text="Vzorek")
    tree.heading("Position", text="Pozice")
    tree.heading("Detected Items", text="Nasnímané dráty")
    tree.column("Sample", width=100, anchor="center")
    tree.column("Position", width=80, anchor="center")
    tree.column("Detected Items", width=150, anchor="center")
    tree.pack(fill="both", expand=True)

    # VPRAVO – kamera
    preview_frame, image_label, position_label = create_camera_preview(
        main_frame,
        config.frame_width,
        config.frame_height,
        lambda: (core.motion_controller.grbl_last_position, core.motion_controller.grbl_status),
        core.camera_manager.start_camera_preview
    )

    def on_click(event):
        item = tree.selection()
        if item:
            values = tree.item(item[0], "values")
            ean_code = values[0]
            position = values[1]
            show_image(image_label, project_id, ean_code, position)

    tree.bind("<ButtonRelease-1>", on_click)

    # Přepneme na mikroskop
    core.camera_manager.preview_running = False
    if core.camera_manager.actual_camera == core.camera_manager.camera:
        core.camera_manager.switch_camera()

    # Spustí snímání mikroskopem ve vlákně
    def threaded_get_microscope_images(container, image_label, tree, project_id):
        # Získá vzorky z databáze podle ID projektu
        samples_from_db = get_samples_by_project_id(project_id)
        for sample_id, position, ean_code, image_path in samples_from_db:
            # Získej položky vzorku z databáze
            if image_path is not None:
                items = get_sample_items_by_sample_id(sample_id)
                container.after(0, lambda: tree.insert("", "end", values=(ean_code, position, f"0 z {len(items)}")))
                container.after(0, lambda: show_image(image_label, project_id, ean_code, position))
                time.sleep(0.5)  # Pauza mezi jednotlivými snímky, aby se stihly zobrazit
                get_microscope_images(container, image_label, tree, project_id, position, ean_code, items)
        # Nakonec vyjedeme s kazetou
        threading.Thread(target=core.motion_controller.move_to_home_position(), daemon=True).start()
        container.after(100, lambda: Messagebox.show_info("Všechny snímky byly získány. Můžete vyměnit kazetu."))

    t = threading.Thread(target=threaded_get_microscope_images, args=(container, image_label, tree, project_id), daemon=True)
    t.start()