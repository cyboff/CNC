import ttkbootstrap as ttk
import tkinter.messagebox as messagebox
from core.database import get_all_projects, get_project_by_id, delete_project
from core.logger import logger
from core.utils import create_back_button, create_header, create_footer


def show_projects(container, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    logger.info("Zobrazení seznamu všech projektů")


    # Horní lišta s nadpisem a tlačítkem Zpět
    create_header(container, "CNC Sample Detector - projekty",on_back)
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
        if item and column == "#5":  # Akce sloupec
            values = tree.item(item, "values")
            project_id = values[0]
            project_name = values[1]
            confirm = messagebox.askyesno("Potvrzení", f"Opravdu smazat měření '{project_name}'?")
            if confirm:
                delete_project(project_id)
                logger.info(f"Smazáno měření ID {project_id} ({project_name})")
                refresh_table()

    tree.bind("<ButtonRelease-1>", on_click)
    refresh_table()



def open_project_detail(container, project_id, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    logger.info(f"Zobrazení detailu měření ID {project_id}")
    create_header(container, "CNC Sample Detector - detail projektu",on_back)
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



