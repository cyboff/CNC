import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import tkinter as tk

from core.logger import logger
from core.utils import create_header, create_footer, create_back_button
from process.find_process import find_sample_positions

def sample_scanner(container, project_id, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    samples = []

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
            Messagebox.show_info("Načteny 4 vzorky. Pokračujte na měření.")
            start_measurement()

    entry.bind("<Return>", on_scan)

    def start_measurement():
        if len(samples) == 0:
            Messagebox.show_error("Musíte načíst alespoň 1 vzorek.")
            return
        logger.info(f"[SCAN] Spouštím proces FIND pro projekt {project_id}")
        positions = find_sample_positions(project_id, samples)
        Messagebox.show_info(f"Detekováno {len(positions)} vzorků.")

        logger.info(f"Projekt {project_id} – Načteno {len(samples)} vzorků: {samples}")
        # TODO: pokračovat na měření

    ttk.Button(container, text="Vzorky načteny – spustit měření", bootstyle="success",
               command=start_measurement).pack(pady=20)
