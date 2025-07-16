import ttkbootstrap as ttk

def apply_styles():
    style = ttk.Style()

    # === Globální nastavení ===
    style.configure("TButton", font=("Helvetica", 20), padding=10)
    style.configure("TLabel", font=("Helvetica", 15), foreground="#2b3b4c")

    # === Speciální tlačítka ===
    style.configure("Green.TButton", font=("Helvetica", 18), padding=12, foreground="white", background="#28a745")
    style.configure("Blue.TButton", font=("Helvetica", 18), padding=12, foreground="white", background="#007bff")

    # === Nadpisy ===
    style.configure("Header.TLabel", font=("Helvetica", 20, "bold"))

    # === Patička ===
    style.configure("Custom.TFrame", background="#48484c", borderwidth=0, relief="solid")

    # === Header ===
    style.configure("Header.TFrame", background="#2b3c4c")


    style.configure("Main.TButton",font=("Helvetica", 16),padding=10,foreground="#ffffff",background="#018c45",borderwidth=0)
    style.map("Main.TButton",background=[("active", "#015A2D")])

    style.configure("MainSmaller.TButton", font=("Helvetica", 12), padding=5, foreground="#ffffff", background="#018c45",
                    borderwidth=0)
    style.map("MainSmaller.TButton", background=[("active", "#015A2D")])

    style.configure("Move.TButton", font=("Helvetica", 16), padding=10, foreground="white", background="#007bff")

    style.configure("Secondary.TButton",font=("Helvetica", 14),padding=8,foreground="#333",background="#ccc")

    style.configure("Back.TButton", font=("Helvetica", 14), padding=6, background="#2b3c4c",borderwidth=0)
    style.configure("Step.TLabel", font=("Helvetica", 16, "bold"), foreground="#444")

    style.configure("Step.TButton", font=("Helvetica", 10), padding=6, width=4)
    style.configure("StepSelected.TButton", font=("Helvetica", 10, "bold"), background="#4CAF50", foreground="white")
    style.map("StepSelected.TButton",
              background=[("active", "#388e3c")],
              foreground=[("active", "white")])