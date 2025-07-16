import os
from tkinter.scrolledtext import ScrolledText
import ttkbootstrap as ttk
from core.utils import create_back_button, create_header, create_footer

def show_log_view(container, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    # Horní lišta s nadpisem a tlačítkem Zpět
    create_header(container, "CNC Sample Detector - log záznamů",on_back)
    create_footer(container)


    ttk.Label(container, text="Záznam logu aplikace", font=("Helvetica", 20)).pack(pady=10)

    log_text = ScrolledText(container, wrap="word", font=("Courier", 10))
    log_text.pack(expand=True, fill="both", padx=20, pady=10)

    log_file_path = os.path.join("data", "app.log")
    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            log_text.insert("1.0", f.read())
    else:
        log_text.insert("1.0", "Logovací soubor neexistuje.")
