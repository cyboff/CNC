import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import config
from core.database import save_project_sample_to_db
from core.logger import logger
from core.utils import create_header, create_footer, create_back_button
from gui.find_samples import show_find_samples

def sample_scanner(container, project_id, on_back):
    for widget in container.winfo_children():
        widget.destroy()

    samples = []
    print(f"Krok 2: Skenování EAN kódů pro projekt {project_id}")
    create_header(container, "WDS - Wire Defect Scanner - Krok 2: Skenování EAN kódů")
    create_footer(container)
    create_back_button(container, on_back)

    ttk.Label(container, text="Přilož čtečku kódů ke vzorku a načti:", style="Header.TLabel").pack(pady=10)

    entry = ttk.Entry(container, font=("Helvetica", 16), width=30)
    entry.pack(pady=5)
    entry.focus()

    tree = ttk.Treeview(
        container,
        columns=("code", "position"),
        show="headings",
        height=10
    )
    tree.heading("code", text="Vzorek - EAN kód")
    tree.heading("position", text="Vložte na pozici")
    tree.column("code", width=100, anchor="center")
    tree.column("position", width=100, anchor="center")
    tree.pack(padx=200, pady=10, fill="both", expand=True)

    def on_scan(event=None):
        code = entry.get().strip()
        if code and code not in samples:
            samples.append(code)
            pos = config.sample_positions_mm[len(samples)-1][0] if len(samples)-1 < len(config.sample_positions_mm) else None
            tree.insert(
                "",
                "end",
                values=(code, pos)
            )
            entry.delete(0, "end") # vyčistit zadávací pole pole
            # Vložit do databáze
            save_project_sample_to_db(project_id, pos, code, None)
            print(f"Načten vzorek s EAN kódem: {code} pro pozici {pos}")
            logger.info(f"Načten vzorek s EAN kódem: {code} pro pozici {pos}")
        elif code in samples:
            Messagebox.show_info("Kód už byl načten.")
            logger.warning(f"EAN kód {code} byl již načten")
            entry.delete(0, "end")
        if len(samples) > len(config.sample_positions_mm) -1:
            Messagebox.show_info(f"Načteno všech {len(config.sample_positions_mm)} vzorků. Pokračujte na měření.")
            # start_find_samples(container, project_id, samples)

    entry.bind("<Return>", on_scan)

    def start_find_samples(container, project_id:int, samples: list[str]):
        if len(samples) == 0:
            Messagebox.show_error("Musíte načíst alespoň 1 vzorek.")
            return
        logger.info(f"[SCAN] Spouštím proces FIND pro projekt {project_id}")
        # Pokračovat na měření
        for widget in container.winfo_children():
            widget.destroy()
        show_find_samples(container, project_id, on_back)


    ttk.Button(
        container, text="Vzorky načteny – spustit měření", bootstyle="success",
        command=lambda: start_find_samples(container, project_id, samples)
    ).pack(pady=20)
