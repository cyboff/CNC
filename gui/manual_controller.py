from tkinter import ttk, StringVar, Label, Frame, Button

import config
from core.utils import create_back_button, create_header, create_footer, create_camera_preview
from core.logger import logger
from core.motion_controller import move_axis, grbl_home, grbl_clear_alarm, grbl_abort
import core.motion_controller
from process.images_process import run_autofocus, run_fine_focus
from process.find_process import find_sample_positions
from core.camera_manager import start_camera_preview, switch_camera, calibrate_camera, autofocus_z
import core.camera_manager
import threading

update_position_timer = None

def show_manual_controller(container, on_back):

    for widget in container.winfo_children():
        widget.destroy()

    logger.info("Kliknutí na manuální řízení")

    create_header(container, "CNC Sample Detector - manuální ovládání", on_back)
    create_footer(container)

    main_frame = ttk.Frame(container)
    main_frame.pack(fill="both", expand=True)

    # VLEVO – ovládání
    control_frame = ttk.Frame(main_frame)
    control_frame.pack(side="left", padx=20, pady=20, fill="y")

    ttk.Label(control_frame, text="Manuální řízení CNC", font=("Helvetica", 14, "bold")).pack(pady=10)

    # Styl tlačítek
    step_values = [0.01, 0.1, 1, 5, 10, 50]
    selected_step = StringVar(value=str(step_values[2]))  # výchozí krok

    step_frame = ttk.Frame(control_frame)
    step_frame.pack(pady=5)

    button_refs = {}

    def update_selected_step(val):
        selected_step.set(str(val))
        for v, b in button_refs.items():
            style = "StepSelected.TButton" if str(v) == str(val) else "Step.TButton"
            b.config(style=style)

    def render_step_buttons():
        for widget in step_frame.winfo_children():
            widget.destroy()
        button_refs.clear()
        for val in step_values:
            btn = ttk.Button(step_frame, text=str(val), width=4,
                             command=lambda v=val: update_selected_step(v))
            btn.pack(side="left", padx=2)
            button_refs[val] = btn
        update_selected_step(selected_step.get())

    render_step_buttons()

    # === FUNKCE PRO POHYBY S AKTUALIZACÍ POZICE ===
    def move_x(amount):
        move_axis("X", amount)

    def move_y(amount):
        move_axis("Y", amount)

    def move_z(amount):
        move_axis("Z", amount)

    def step():
        return float(selected_step.get())

    # Směry
    move_frame = ttk.Frame(control_frame)
    move_frame.pack(pady=10)

    def btn(txt, cmd, r, c):
        ttk.Button(move_frame, text=txt, style="Move.TButton", command=cmd, width=6).grid(row=r, column=c, padx=4, pady=4)

    btn("↑\nY+", lambda: move_y(step()), 0, 1)
    btn("←\nX-", lambda: move_x(-step()), 1, 0)
    btn("→\nX+", lambda: move_x(step()), 1, 2)
    btn("↓\nY-", lambda: move_y(-step()), 2, 1)
    btn("Z+\n↑", lambda: move_z(step()), 0, 3)
    btn("Z-\n↓", lambda: move_z(-step()), 2, 3)

    # Bind pro stisk klávesy
    def on_key_press(event):
        key = event.keysym.lower()

        try:
            if getattr(core.camera_manager, "calibration_active", False):
                if key == "q" and getattr(core.camera_manager, "_calibration_on_q", None):
                    core.camera_manager._calibration_on_q()
                    return "break"  # zastaví bubbling
        except Exception:
            pass

        if key == "up":
            move_y(step())
        elif key == "down":
            move_y(-step())
        elif key == "left":
            move_x(-step())
        elif key == "right":
            move_x(step())
        elif key == "prior":  # PgUp
            move_z(step())
        elif key == "next":   # PgDown
            move_z(-step())

    container.bind_all("<KeyPress>", on_key_press)

    # === TLAČÍTKA ===

    def add_action_button(parent, text, command):
        button = ttk.Button(parent, text=text, style="Main.TButton", command=command, cursor="hand2")
        button.pack(pady=6, ipadx=20, ipady=8)
        button.configure(width=25)

    # ZELENÁ skupina
    add_action_button(control_frame, "🏠 Domů ($H)", lambda: threading.Thread(target=grbl_home, daemon=True).start())
    add_action_button(control_frame, "🔁 Přerušit (Soft Reset)", grbl_abort)
    add_action_button(control_frame, "❌ Zrušit Alarm ($X)", grbl_clear_alarm)
    # Mezera mezi skupinami tlačítek
    ttk.Label(control_frame, text="").pack(pady=1)
    # MODRÁ skupina
    add_action_button(control_frame, "🎯 Zaostřit", lambda: threading.Thread(target=autofocus_z, daemon=True).start())
    add_action_button(control_frame, "🎥 Přepnout kameru", lambda: threading.Thread(target=switch_camera, daemon=True).start())
    add_action_button(control_frame, "🔧 Kalibrovat kamery", lambda: threading.Thread(target=calibrate_camera, args=(container, image_label, move_x, move_y, move_z, step), daemon=True).start())
    # add_action_button(control_frame, "🔎 Najdi vzorky", lambda: threading.Thread(target=find_sample_positions, args=(container, image_label, None, None, [1, 2, 3, 4]), daemon=True).start())

    # VPRAVO – kamera
    core.camera_manager.preview_running = False
    preview_frame, image_label, position_label = create_camera_preview(
        main_frame,
        config.frame_width,
        config.frame_height,
        lambda: (core.motion_controller.grbl_last_position, core.motion_controller.grbl_status),
        start_camera_preview
    )

