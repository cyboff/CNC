import ttkbootstrap as ttk
import tkinter.messagebox as messagebox

from core.camera_manager import microscope
from core.database import get_all_projects, get_project_by_id, delete_project, get_samples_by_project_id, \
    get_sample_item_positions_by_item_id, get_sample_items_by_sample_id
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
    create_header(container, "WDS - Wire Defect Scanner - detail projektu",on_back)
    create_footer(container)

    top_bar = ttk.Frame(container)
    top_bar.pack(fill="x", pady=10, padx=10)

    ttk.Label(top_bar, text=f"Detail měření ID {project_id}", font=("Helvetica", 20)).pack(side="left")


    project = get_project_by_id(project_id)
    if not project:
        ttk.Label(container, text="Měření nenalezeno.", font=("Helvetica", 14)).pack(pady=20)
        return

    _, name, comment, created = project

    info_frame = ttk.Frame(container, padding=20)
    info_frame.pack(pady=20)

    ttk.Label(info_frame, text=f"Název: {name}", font=("Helvetica", 14)).pack(anchor="w", pady=5)
    ttk.Label(info_frame, text=f"Komentář: {comment}", font=("Helvetica", 14)).pack(anchor="w", pady=5)
    ttk.Label(info_frame, text=f"Vytvořeno: {created}", font=("Helvetica", 14)).pack(anchor="w", pady=5)

    # TODO: zde bude v budoucnu detail vzorků, obrázky apod.
    samples = get_samples_by_project_id(project_id)
    image_missing = False
    if samples:
    # Vytvoření tabulky 4x4 pro vzorky
        samples_frame = ttk.Frame(container)
        samples_frame.pack(pady=10)

        for i, (sample_id, position, ean_code, image_path) in enumerate(samples[:16]):  # max 16 vzorků
            row = i // 4
            col = i % 4
            cell = ttk.Frame(samples_frame, borderwidth=1, relief="solid", padding=5)
            cell.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            ttk.Label(cell, text=f"Pozice: {position}").pack(anchor="w")
            ttk.Label(cell, text=f"EAN: {ean_code}").pack(anchor="w")
            # Zobrazíme obrázek vzorku, pokud je k dispozici
            if image_path is not None:
                try:
                    from PIL import Image, ImageTk
                    img = Image.open(image_path)
                    img = img.resize((250, 250))
                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(cell, image=photo)
                    img_label.image = photo
                    img_label.pack()
                except Exception as e:
                    ttk.Label(cell, text="Obrázek nelze načíst").pack()
                    image_missing = True
            else:
                image_missing = True

            # Vypíšeme počet detekovaných drátů a počet snímků z mikroskopu
            items = get_sample_items_by_sample_id(sample_id)

            microscope_images_missing = False
            ttk.Label(cell, text=f"Detekované dráty: {len(items)}").pack(anchor="w")
            for j, (item_id, position_index, x_center, y_center, radius) in enumerate(items):
                positions = get_sample_item_positions_by_item_id(item_id)
                item_image_count = 0
                for k, (item_id, position_index, x_coord, y_coord, item_image_path, defect_detected) in enumerate(
                        positions):
                    if item_image_path is not None:
                        item_image_count += 1
                ttk.Label(cell, text=f"  Drát {j + 1}: Nasnímáno {item_image_count} z {len(positions)} snímků").pack(
                    anchor="w")
                if item_image_count < len(positions):
                    microscope_images_missing = True

        if image_missing is True:
            # Některé vzorky nemají obrázek - nabídneme znovu hledání vzorků
            ttk.Button(
                container, text="Vzorky načteny – spustit měření", bootstyle="success",
                command=lambda: show_find_samples(container, project_id, on_back)
            ).pack(pady=20)

        if microscope_images_missing is True:
            # Některé vzorky nemají obrázky z mikroskopu - nabídneme znovu snímání mikroskopem
            ttk.Button(
                container, text="Pokračovat na snímání mikroskopem", bootstyle="success",
                command=lambda: show_microscope_images(container, project_id, on_back)
            ).pack(pady=20)

    else:
        # Projekt založen, ale bez vzorků - nabídneme skenování EAN kódů
        ttk.Button(
            container, text="Skenovat kódy vzorků", bootstyle="success",
            command=lambda: sample_scanner(container, project_id, on_back)
        ).pack(pady=20)



