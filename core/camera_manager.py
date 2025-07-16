import cv2
import numpy as np
import threading
from pypylon import pylon
from PIL import Image, ImageTk

camera = None
preview_running = False


def get_basler_frame():
    global camera
    try:
        if camera is None:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            camera.Open()
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        if camera.IsGrabbing():
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                img = grabResult.Array
                return img
    except Exception as e:
        print("[Camera] Chyba při čtení snímku:", e)
    return None


def start_camera_preview(image_label, update_position_callback=None):
    global camera, preview_running

    if preview_running:
        return

    preview_running = True

    def preview_loop():
        global camera, preview_running

        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        camera.StartGrabbing()
        camera.ExposureAuto.SetValue("Off")  # vypnout automatiku
        camera.ExposureTimeAbs.SetValue(20000)  # zvyšit hodnotu expozice

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_Mono8
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        while camera.IsGrabbing() and preview_running:
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                image = converter.Convert(grab_result)
                img = image.GetArray()

                # Převod na RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # Rozměry a střed
                h, w = img_rgb.shape[:2]
                cx, cy = w // 2, h // 2

                # Křížek ve středu
                cv2.line(img_rgb, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 2)
                cv2.line(img_rgb, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 2)

                # Zmenšit výstup na 640x480 (nebo ponechat originál)
                img_resized = cv2.resize(img_rgb, (640, 480))

                im_pil = Image.fromarray(img_resized)
                imgtk = ImageTk.PhotoImage(image=im_pil)

                def update():
                    global grbl_status
                    if not image_label.winfo_exists():
                        return  # Widget už neexistuje, ukončíme aktualizaci
                    image_label.imgtk = imgtk
                    image_label.config(image=imgtk)

                image_label.after(0, update)



            grab_result.Release()

        camera.StopGrabbing()
        camera.Close()

    # spustíme náhled ve vlákně, aby neblokoval GUI
    threading.Thread(target=preview_loop, daemon=True).start()






def stop_camera_preview():
    global preview_running
    preview_running = False



def release_camera():
    """
    Korektně zavře připojení ke kameře.
    """
    global camera
    try:
        if camera and camera.IsGrabbing():
            camera.StopGrabbing()
        if camera and camera.IsOpen():
            camera.Close()
        camera = None
        print("[Camera] Kamera uvolněna.")
    except Exception as e:
        print("[Camera] Chyba při uvolňování kamery:", e)