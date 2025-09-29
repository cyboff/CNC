import ttkbootstrap as ttk
import tkinter.messagebox as messagebox

from core.camera_manager import microscope
from core.database import get_all_projects, get_project_by_id, delete_project, get_samples_by_project_id, \
    get_sample_item_positions_by_item_id, get_sample_items_by_sample_id, delete_sample_items_from_project
from core.logger import logger
from core.utils import create_back_button, create_header, create_footer
from gui.find_samples import show_find_samples
from gui.scan_samples import sample_scanner
from gui.microscope_images import show_microscope_images


def show_projects(container, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    logger.info("Zobrazení seznamu všech projektů")


    # Horní lišta s nadpisem a tlačítkem Zpět
    create_header(container, "WDS - Wire Defect Scanner - projekty",on_back)
    create_footer(container)

    top_bar = ttk.Frame(container)
    top_bar.pack(fill="x", pady=10, padx=10)
    ttk.Label(top_bar, text="Otevřít měření", font=("Helvetica", 20), foreground="#2b3b4c").pack(pady=(10, 5))


    # Definice tabulky se sloupcem navíc "Akce"
    columns = ("ID", "Název", "Komentář", "Vytvořeno", "Akce")
    tree = ttk.Treeview(container, columns=columns, show="headings")

    for col in columns:
        tree.heading(col, text=col)

    tree.column("ID", width=50, anchor="center")
    tree.column("Název", width=200)
    tree.column("Komentář", width=300)
    tree.column("Vytvořeno", width=150, anchor="center")
    tree.column("Akce", width=100, anchor="center")

    tree.pack(expand=True, fill="both", padx=20, pady=10)

    def refresh_table():
        tree.delete(*tree.get_children())
        for row in get_all_projects():
            tree.insert("", "end", values=(row[0], row[1], row[2], row[3], "🗑️ Smazat"))

    def on_click(event):
        item = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if item:  # Akce sloupec
            values = tree.item(item, "values")
            project_id = values[0]
            project_name = values[1]
            if column == "#5":  # Sloupec "Akce" pro smazání
                confirm = messagebox.askyesno("Potvrzení", f"Opravdu smazat měření '{project_name}'?")
                if confirm:
                    delete_project(project_id)
                    refresh_table()
            else:
                open_project_detail(container, project_id, lambda: show_projects(container, on_back))

    tree.bind("<ButtonRelease-1>", on_click)
    refresh_table()


def open_project_detail(container, project_id, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    logger.info(f"Zobrazení detailu měření ID {project_id}")
    create_header(container, "WDS - Wire Defect Scanner - detail projektu", on_back)
    create_footer(container)

    # --- Scrollovací část ---
    canvas = ttk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="n")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    # -------------------------

    top_bar = ttk.Frame(scroll_frame)
    top_bar.pack(fill="x", pady=10, padx=10)

    ttk.Label(top_bar, text=f"Detail měření ID {project_id}", font=("Helvetica", 20)).pack(side="left")

    project = get_project_by_id(project_id)
    if not project:
        ttk.Label(scroll_frame, text="Měření nenalezeno.", font=("Helvetica", 14)).pack(pady=20)
        return

    _, name, comment, created = project

    info_frame = ttk.Frame(scroll_frame, padding=20)
    info_frame.pack(pady=20, fill="x")

    ttk.Label(info_frame, text=f"Název: {name}", font=("Helvetica", 14)).pack(anchor="w", pady=5)
    ttk.Label(info_frame, text=f"Komentář: {comment}", font=("Helvetica", 14)).pack(anchor="w", pady=5)
    ttk.Label(info_frame, text=f"Vytvořeno: {created}", font=("Helvetica", 14)).pack(anchor="w", pady=5)

    samples = get_samples_by_project_id(project_id)
    image_missing = False
    if samples:
        samples_frame = ttk.Frame(scroll_frame)
        samples_frame.pack(pady=10)

        for i, (sample_id, position, ean_code, image_path) in enumerate(samples[:16]):
            row = i // 4
            col = i % 4
            cell = ttk.Frame(samples_frame, borderwidth=1, relief="solid", padding=5)
            cell.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            ttk.Label(cell, text=f"Pozice: {position}").pack(anchor="w")
            ttk.Label(cell, text=f"EAN: {ean_code}").pack(anchor="w")

            if image_path is not None:
                try:
                    from PIL import Image, ImageTk
                    img = Image.open(image_path)
                    img = img.resize((250, 250))
                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(cell, image=photo)
                    img_label.image = photo
                    img_label.pack()
                except Exception:
                    ttk.Label(cell, text="Obrázek nelze načíst").pack()
                    image_missing = True
            else:
                image_missing = True

            items = get_sample_items_by_sample_id(sample_id)
            microscope_images_missing = False
            item_image_count = 0
            ttk.Label(cell, text=f"Detekované dráty: {len(items)}").pack(anchor="w")
            for j, (item_id, _, _, _, _) in enumerate(items):
                positions = get_sample_item_positions_by_item_id(item_id)
                for (_, _, _, _, item_image_path, _) in positions:
                    if item_image_path is not None:
                        try:
                            img = Image.open(item_image_path)
                            item_image_count += 1
                        except Exception:
                            pass
                ttk.Label(cell,text=f"  Pro drát {j+1} získáno {item_image_count} z {len(positions)} snímků").pack(anchor="w")
                if item_image_count < len(positions):
                    microscope_images_missing = True

        if image_missing:
            def delete_and_rescan():
                delete_sample_items_from_project(project_id)
                show_find_samples(container, project_id, on_back)

            ttk.Button(
                scroll_frame, text="EAN kódy vzorků načteny – spustit měření", bootstyle="success",
                command=lambda: delete_and_rescan()
            ).pack(pady=20)
        elif microscope_images_missing:
            ttk.Button(
                scroll_frame, text="Pokračovat na snímání mikroskopem", bootstyle="success",
                command=lambda: show_microscope_images(container, project_id, on_back)
            ).pack(pady=20)

    else:
        ttk.Button(
            scroll_frame, text="Skenovat kódy vzorků", bootstyle="success",
            command=lambda: sample_scanner(container, project_id, on_back)
        ).pack(pady=20)



