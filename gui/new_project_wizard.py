import ttkbootstrap as ttk
import sqlite3

from core.database import insert_project
from core.project_manager import create_project_folder
from core.utils import create_step_header, create_back_button, create_header, create_footer
from core.logger import logger
from gui.scan_samples import sample_scanner


def open_new_project_wizard(container, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    # Horní lišta s nadpisem a tlačítkem Zpět
    create_header(container, "CNC Sample Detector -  Krok 1: vytvoření nového projektu",on_back)
    create_footer(container)

   # create_step_header(container, "Krok 1: Vytvoření nového projektu")

    ttk.Label(container, text="Nový projekt", style="Header.TLabel").pack(pady=20)

    ttk.Label(container, text="Název měření:").pack()
    entry_name = ttk.Entry(container, width=40)
    entry_name.pack()

    ttk.Label(container, text="Komentář:").pack()
    entry_comment = ttk.Entry(container, width=40)
    entry_comment.pack()

    def save_project():
        name = entry_name.get().strip()
        comment = entry_comment.get().strip()
        if not name:
            from ttkbootstrap.dialogs import Messagebox
            Messagebox.show_error("Musíte zadat název měření.")
            return

        # Vložení projektu do DB
        insert_project(name, comment)

        # Získání ID nového projektu
        conn = sqlite3.connect("data/database.db")
        c = conn.cursor()
        c.execute("SELECT id, created_at FROM projects ORDER BY created_at DESC")
        row = c.fetchone()
        project_id = row[0]
        created_at = row[1]
        # print(f"Do databáze byl vložen projekt s ID: {project_id} vytvořený {created_at}")
        conn.close()

        # Vytvoření složky projektu
        project_path = create_project_folder(project_id)
        logger.info(f"Vytvořena složka projektu: {project_path}")

        # Pokračuj na scan vzorků
        for widget in container.winfo_children():
            widget.destroy()
        print(f"Krok 1: Uložen projekt {name} s ID {project_id} vytvořený {created_at}")
        sample_scanner(container, project_id, on_back)

    ttk.Button(container, text="Uložit a pokračovat", bootstyle="success", command=save_project).pack(pady=20)
