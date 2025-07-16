from tkinter import ttk, StringVar, Label, Frame, Button
from core.utils import create_back_button, create_header, create_footer
from core.logger import logger
from core.motion_controller import move_axis, grbl_home, grbl_clear_alarm, grbl_abort, grbl_toggle_pause_resume, \
    grbl_update_position
import core.motion_controller
from process.images_process import run_autofocus, run_fine_focus
from process.find_process import find_sample_positions
from core.camera_manager import start_camera_preview, preview_running
import threading

update_position_timer = None

def show_manual_controller(container, on_back):
    global cnc_serial, last_position, position_lock, grbl_status, update_position_timer, preview_running
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
    step_values = [0.1, 0.5, 1, 5, 10]
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

    # === TLAČÍTKA ===

    def add_action_button(parent, text, command):
        button = ttk.Button(parent, text=text, style="MainSmaller.TButton", command=command, cursor="hand2")
        button.pack(pady=6, ipadx=20, ipady=8)
        button.configure(width=25)

    # ZELENÁ skupina
    add_action_button(control_frame, "🏠 Domů ($H)", lambda: threading.Thread(target=grbl_home, daemon=True).start())
    add_action_button(control_frame, "🔎 Najdi vzorky", lambda: threading.Thread(target=find_sample_positions, daemon=True).start())
    add_action_button(control_frame, "🎥 Přepnout kameru", lambda: print("Toggle kamera"))

    # ČERVENÁ skupina
    add_action_button(control_frame, "⏸ Pause (!)", lambda: grbl_toggle_pause_resume("!"))
    add_action_button(control_frame, "🔁 Přerušit (Soft Reset)", grbl_abort)
    add_action_button(control_frame, "❌ Zrušit Alarm ($X)", grbl_clear_alarm)

    # MODRÁ skupina
    add_action_button(control_frame, "🔧 Jemné ostření", run_fine_focus)
    add_action_button(control_frame, "🎯 Zaostřit", run_autofocus)
    add_action_button(control_frame, "💾 Ulož pozici", lambda: print("Ulož pozici"))

    # VPRAVO – kamera
    preview_frame = ttk.Frame(main_frame)
    preview_frame.pack(side="left", fill="both", expand=True)

    Label(preview_frame, text="Zobrazení kamery", font=("Helvetica", 14, "bold")).pack()

    image_label = Label(preview_frame, width=640, height=480)
    image_label.pack()

    # === POZICE STROJE ===

    # Live pozice
    position_label = Label(preview_frame, text="Status: ---, X: ---, Y: ---, Z: ---", font=("Helvetica", 10))
    position_label.pack(pady=6)

    def update_position():
        try:
            last_position = core.motion_controller.grbl_last_position
            grbl_status = core.motion_controller.grbl_status
            x, y, z = map(float, last_position.split(","))
            position_label.config(text=f"Stav: {grbl_status}, X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}")
        except Exception:
            pass
        position_timer = threading.Timer(0.5, update_position)
        position_timer.daemon = True
        position_timer.start()


    preview_running = False
    start_camera_preview(image_label, update_position_callback=update_position)
    update_position()  # Spustí periodické aktualizace pozice
