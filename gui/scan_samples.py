import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import tkinter as tk

from core.logger import logger
from core.utils import create_header, create_footer, create_back_button
from gui.find_samples import show_sample_detector

def sample_scanner(container, project_id, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    samples = []
    print(f"Krok 2: Skenování EAN kódů pro projekt {project_id}")
    create_header(container, "CNC Sample Detector - Krok 2: Skenování EAN kódů")
    create_footer(container)
    create_back_button(container, on_back)

    ttk.Label(container, text="Přilož čtečku kódů ke vzorku a načti:", style="Header.TLabel").pack(pady=10)

    entry = ttk.Entry(container, font=("Helvetica", 16), width=30)
    entry.pack(pady=5)
    entry.focus()

    listbox = tk.Listbox(container, height=10, font=("Courier", 12))
    listbox.pack(padx=10, pady=10, fill="both", expand=True)

    def on_scan(event=None):
        code = entry.get().strip()
        if code and code not in samples:
            samples.append(code)
            listbox.insert("end", code)
            entry.delete(0, "end")
            logger.info(f"Načten EAN kód: {code}")
        elif code in samples:
            Messagebox.show_info("Kód už byl načten.")
            logger.warning(f"EAN kód {code} byl již načten")
            entry.delete(0, "end")
        if len(samples) > 3:
            Messagebox.show_info("Načteny všechny 4 vzorky. Pokračujte na měření.")
            start_measurement(container, project_id, samples)

    entry.bind("<Return>", on_scan)

    def start_measurement(container, project_id:int, samples: list[str]):
        if len(samples) == 0:
            Messagebox.show_error("Musíte načíst alespoň 1 vzorek.")
            return
        logger.info(f"[SCAN] Spouštím proces FIND pro projekt {project_id}")
        # Pokračovat na měření
        for widget in container.winfo_children():
            widget.destroy()
        show_sample_detector(container, project_id, samples, on_back)


    ttk.Button(
        container, text="Vzorky načteny – spustit měření", bootstyle="success",
        command=lambda: start_measurement(container, project_id, samples)
    ).pack(pady=20)
