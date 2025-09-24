import ttkbootstrap as ttk
from tkinter import StringVar, messagebox
import config
from core.logger import logger
from core.settings import get_setting, set_setting
from core.utils import create_back_button, create_header, create_footer
from core.settings import get_all_settings  # nová funkce
from core.scrollable_frame import ScrollableFrame  # nebo podle cesty

def show_settings(container, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    logger.info("Kliknutí na nastavení")

    create_header(container, "WDS - Wire Defect Scanner – Nastavení",on_back)
    create_footer(container)

    settings = get_all_settings()

    scrollable = ScrollableFrame(container)
    scrollable.pack(fill="both", expand=True, padx=20, pady=10)
    frame = scrollable.get_frame()
    frame.pack(fill="both", expand=True, padx=20, pady=10)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=20)

    entries = {}
    row = 0

    for key, value in settings.items():
        ttk.Label(frame, text=key.upper(), font=("Helvetica", 10, "bold"), anchor="e").grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        var = ttk.StringVar(value=str(value))
        entry = ttk.Entry(frame, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        entries[key] = var
        row += 1

    def save_settings():
        for key, var in entries.items():
            set_setting(key, var.get())

        try:
            config.reload_settings()
            logger.info("Nastavení uložena a načtena bez restartu.")
            messagebox.showinfo("Uloženo", "Nastavení byla uložena a načtena bez restartu.")
        except Exception as e:
            logger.exception("Nepodařilo se reloadnout config: %s", e)
            messagebox.showwarning("Uloženo s výhradou",
                                   f"Nastavení byla uložena, ale nepodařilo se reloadnout config:\n{e}")

    ttk.Button(container, text="💾 Uložit", command=save_settings, bootstyle="success").pack(pady=15)
