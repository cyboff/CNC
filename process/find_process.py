# find_process.py
"""
Modul pro najíždění na základní pozici a detekci vzorků pomocí hlavní kamery.
Vrací seznam souřadnic vzorků [(x1, y1), (x2, y2), ...].
"""
import cv2
import numpy as np
from ttkbootstrap.dialogs import Messagebox
import config
import core.camera_manager
import gui.find_samples
from core.logger import logger
from core.motion_controller import move_to_coordinates, grbl_abort, grbl_clear_alarm, cancel_move, grbl_wait_for_idle
from core.database import save_project_sample_to_db, save_sample_items_to_db, save_sample_item_positions_to_db, \
    get_sample_items_by_sample_id, get_sample_item_positions_by_item_id, update_sample_image_path, \
    get_samples_by_project_id
from PIL import Image, ImageTk
import time
from core.project_manager import save_image_to_project
from core.camera_manager import tenengrad_sharpness, autofocus_z, black_white_ratio, rectpx_to_grbl, preview_running


def find_sample_positions(container, image_label, tree, project_id: int):
    sample_positions = []

    # Počet vynechaných bodů pro interpolaci na kontuře drátu pro mikroskopické snímky - čím vyšší číslo, tím méně bodů na obvodu drátu
    precision = config.precision
    # precision = 20 # Pro 5x objektiv
    # precision = 12 # pro 10x objektiv

    if core.camera_manager.actual_camera is None or core.camera_manager.actual_camera == core.camera_manager.microscope:
        core.camera_manager.switch_camera()

    # Získáme seznam vzorků z projektu
    samples = get_samples_by_project_id(project_id)
    print(f"Pro project {project_id} nalezeny následující vzorky:")
    for sample_id, position, code, image_path in samples:
        print(f"EAN Vzorku: {code}, pozice: {position}, db_id: {sample_id}")

    while not gui.find_samples.stop_event.is_set(): # Kontrola, zda není proces přerušen, například při stisku tlačítka Opakovat hledání
        for sample_id, position, code, image_path in samples:
            items = []  # Počet detekovaných drátů pro každý vzorek
            item_points = []  # Pozice jednotlivých detekovaných bodů na drátu
            contours = []
            sample_position = next((t for t in config.sample_positions_mm if t[0] == position), None)
            (pos, x, y, z) = sample_position
            print(f"[FIND] Najíždím na pozici {pos} vzorku {code}: ({x}, {y}, {z})")
            move_to_coordinates(x, y, z)
            print("[FIND] Snímám fotku hlavní kamerou...")
            core.camera_manager.preview_running = False # Zastavíme živý náhled, abychom mohli získat snímek
            time.sleep(1.0)  # Počkáme, aby se proces náhledu zastavil

            img = core.camera_manager.get_image()
            if img is not None:
                # Aplikujeme korekční matici pro perspektivní transformaci
                img = cv2.warpPerspective(img, config.correction_matrix, (int(config.image_width), int(config.image_height)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Převod na RGB pro PIL
                save_image_to_project(project_id, img, f"sample_{code}_position_{pos}_RAW.jpg")
                # Detekce kruhů na obrázku

                print(f"[FIND] Detekuji počet a pozici drátů ve vzorku {code}")
                items, contours = find_circles(project_id, img)

                # Najdeme body po obvodu jednotlivých drátů pro pozice snímků z mikroskopu
                for i, (x, y, r) in enumerate(items):
                    # Vykreslíme detekované kruhy a jejich indexy
                    cv2.putText(img, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
                    cv2.circle(img, (x, y), 2, (0, 0, 255), 2)
                    # cv2.circle(img, (x,y), r, (0, 0, 255), 2)
                    # cv2.drawContours(img, contours[i], -1, (0, 255, 0), 2)

                    # Interpolace bodů podél kontury - aby byly rovnoměrně rozmístěny
                    points = contours[i].squeeze()
                    if len(points.shape) == 1:
                        points = points.reshape(1, 2)
                    num_points = len(points)
                    # Spočítat vzdálenosti mezi sousedními body
                    dists = np.sqrt(np.sum(np.diff(points, axis=0, append=points[:1])**2, axis=1))
                    cumlen = np.cumsum(dists)
                    cumlen = np.insert(cumlen, 0, 0)
                    total_len = cumlen[-1]
                    num_marks = max(1, num_points // precision)
                    for j in range(num_marks):
                        target_len = j * total_len / num_marks
                        idx = np.searchsorted(cumlen, target_len)
                        if idx == 0:
                            interp_point = points[0]
                        else:
                            prev_len = cumlen[idx - 1]
                            next_len = cumlen[idx]
                            alpha = (target_len - prev_len) / (next_len - prev_len) if next_len > prev_len else 0
                            interp_point = (1 - alpha) * points[(idx - 1) % num_points] + alpha * points[idx % num_points]
                        # Vykreslíme interpolovaný bod na obrázek pro kontrolu - radius 2*precision
                        cv2.circle(img, tuple(np.round(interp_point).astype(int)), 2 * precision, (255, 0, 0), 1)
                        # Uložíme bod do seznamu s indexem drátu
                        item_points.append((i, (int(interp_point[0]), int(interp_point[1]))))
                    print(f"[FIND] Detekováno {sum(1 for idx, _ in item_points if idx == i)} bodů na drátě {i+1} vzorku {code}")

                # Uložíme obrázek do projektu
                image_path = save_image_to_project(project_id, img, f"sample_{code}_position_{pos}.jpg")
                # Zobrazíme náhled obrázku v GUI
                # Rozměry dle velikosti frame
                h, w = img.shape[:2]
                target_h, target_w = config.frame_height, config.frame_width
                aspect = w / h
                target_aspect = target_w / target_h
                if aspect > target_aspect:
                    new_w = target_w
                    new_h = int(target_w / aspect)
                else:
                    new_h = target_h
                    new_w = int(target_h * aspect)

                img = cv2.resize(img, (new_w, new_h))
                # img = cv2.resize(img, (config.frame_width, config.frame_height))  # Změna velikosti na rozměry náhledu
                im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=im_pil)
                if image_label.winfo_exists():
                    image_label.imgtk = imgtk  # Uchovat referenci, aby obrázek nezmizel
                    image_label.config(image=imgtk)
                else: print("[FIND] Náhled již neexistuje, nemohu zobrazit obrázek.")
            else:
                print("[FIND] Chyba při získávání snímku z kamery, obrázek je None.")

            if not items:
                print(f"[FIND] Žádné dráty nebyly detekovány na pozici {pos} vzorku {code}.")
                image_label.after(0, lambda: Messagebox.show_warning(f"Na pozici {pos} vzorku {code} nebyly detekovány žádné dráty. Zkontrolujte, zda je vzorek správně umístěn."
                ))
            else:
                print(f"[FIND] Detekováno {len(items)} drátů na pozici {pos} vzorku {code}.")
                sample_positions.append((pos,items))  # Přidat pozici a počet detekovaných drátů
                if tree is not None and tree.winfo_exists():
                    # Přidáme řádek do tabulky s pozicí, kódem
                    container.after(0, lambda: tree.insert("", "end", values=(f"{code}", f"{pos}", f"{len(items)}")))
                # Uložit vzorek do databáze
                #sample_id = save_project_sample_to_db(project_id, pos, code, image_path)  # Uložíme vzorek do databáze
                update_sample_image_path(sample_id, image_path)
                save_sample_items_to_db(sample_id, items)  # Uložíme detekované dráty do databáze
                items_db = get_sample_items_by_sample_id(sample_id)
                for i, (id, pos_index, x_center, y_center, radius) in enumerate(items_db):
                    for step, (idx, (px, py)) in enumerate((item for item in item_points if item[0] == i)):
                        save_sample_item_positions_to_db(id, step, px, py, image_path=None)


            time.sleep(2.0)  # Počkáme, aby se obrázek načetl a zobrazil
            core.camera_manager.start_camera_preview(image_label, update_position_callback=None)  # Restart živého náhledu
            time.sleep(0.5)
        return sample_positions
    pass

def find_circles(project_id, image):
    """
    Detekuje kruhy na obrázku pomocí Houghovy transformace.
    Vrací seznam středů kruhů [(x1, y1), (x2, y2), ...] a jejich kontury.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # gray = cv2.medianBlur(gray, 5)  # Apply median blur to reduce noise
    # Detect circles using Hough Transform (HOUGH_GRADIENT_ALT)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, 10, param1=1300, param2=0.90, minRadius=90, maxRadius=600)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # # --- Step 3: Apply adaptive thresholding ---
    # adaptive_thresh = cv2.adaptiveThreshold(
    #     gray,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV,
    #     blockSize=5,
    #     C=2
    # )
    adaptive_thresh = cv2.threshold(
        blurred,
        125,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    # --- Step 4: Use watershed to separate touching circles, then find contours ---
    circles = []
    contours = []

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the unknown region with zero
    markers[unknown == 255] = 0

    # Watershed
    markers = cv2.watershed(image, markers)

    # Find contours from separated regions
    for marker in np.unique(markers):
        if marker <= 1:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == marker] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 500:  # filter out small contours
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.3:  # adjust as needed for your images
                (x, y), r = cv2.minEnclosingCircle(cnt)
                x, y, r = int(x), int(y), int(r)
                if 30 < r < (config.image_width // 2 - 10):  # filter by radius
                    circles.append((x, y, r))
                    contours.append(cnt)

    if circles is not None:
        return circles, contours
    return []

def get_microscope_images(container, image_label, tree, project_id, position, ean_code, items):
    """
    Převádí detekované kruhy (v rect prostoru) na GRBL pohyby.
    Nově využívá core.camera_manager.rectpx_to_grbl pro převod (u_rect, v_rect) -> (X,Y,Z).
    """

    # TODO: Upravit přesnost pohybu podle potřeby (asi na 0.5 mm)
    if core.camera_manager.microscope.GetDeviceInfo().GetModelName() == "acA2440-20gm":
        # pro acA2440-20gm použijeme detekci hrany kvůli zpřesnění výsledků
        edge_detection = True
    else:
        edge_detection = False
    z_step_begin = 0.4      # rozsah Z pro mikroskopické přejetí skrz fokus, po zlepšení konstrukce kazety se vzorky můžeme zmenšit
    # Najdi absolutní pozici sample slotu (base XY, výška)
    sample_position = next((t for t in config.sample_positions_mm if t[0] == position), None)
    (pos, mpos_x, mpos_y, mpos_z) = sample_position
    abs_z = mpos_z

    logger.info(f"[MICROSCOPE] Získávám mikroskopické obrázky pro pozici {position} vzorku {ean_code} s {len(items)} detekovanými dráty.")
    print(f"[MICROSCOPE] Získávám mikroskopické obrázky pro pozici {sample_position} s {len(items)} detekovanými dráty.")

    for (id, pos_index, x_center, y_center, radius) in items:
        item_positions_db = get_sample_item_positions_by_item_id(id)
        # Souřadnice středu drátu v GRBL prostoru
        cx, cy, cz = rectpx_to_grbl(float(x_center), float(y_center), base_xy=(mpos_x, mpos_y), z=abs_z)

        for id, step, x_coord, y_coord, image_path, defect_detected in item_positions_db:
            # Souřadnice bodu v GRBL prostoru
            px, py, pz = rectpx_to_grbl(float(x_coord), float(y_coord), base_xy=(mpos_x, mpos_y), z=abs_z)

            max_sharpness = 0.0
            sharpest_img = None
            max_errors = 3
            errors = 0
            attempt = 1
            max_black_ratio = 0.0
            previous_px = 0
            previous_py = 0
            z_step = z_step_begin  # Začneme s počátečním krokem Z, v případě chyb zvětšíme
            if step == 0:
                # TODO: Ověřit a doladit: Autofokus na prvním obrázku - je to celkem nespolehlivé i pomalé
                core.camera_manager.start_camera_preview(image_label, update_position_callback=None)
                core.motion_controller.move_to_position(px, py, abs_z)
                best_z, _ = autofocus_z(dof_mm = 0.003,            # hloubka ostrosti mikroskopu
                                        span_mm = 1.0,             # rozsah pro obousměrný jog
                                        feed_mm_min = 10.0,        # rychlost jogu (mm/min)
                                        sample_sleep_s = 0.02,
                                        cam_lag_s = 0.15,
                                        settle_ms_after_cancel = 120,
                                        refine = False,
                                        verbose = True)
                if best_z is not None and abs(abs_z - best_z) < 1.0:  # Pokud je změna menší než 1 mm, přijmeme ji - ale někdy autofocus "uskočí"
                    abs_z = float(best_z)
                    z_step_begin = 0.15  # Zmenšíme na jemnější rozmezí pro zaostření
                    print(f"[MICROSCOPE] Nejlepší výška pro zaostření: {abs_z:.3f} mm")

                core.camera_manager.preview_running = False
                time.sleep(0.25)
                # # Konec autofokusu

            # Opakovací smyčka (ponecháno jako v původní verzi)
            # TODO: Přidat do config nastavení max_sharpness a max_errors
            while errors > max_errors or max_sharpness < 150:
                print(f"[FIND] Získávám mikroskopický obrázek {step+1} z {len(item_positions_db)} pro drát {pos_index} z {len(items)} vzorku {ean_code} - (pokus {attempt})")
                core.motion_controller.move_to_position(px, py, abs_z - z_step)
                time.sleep(0.2)
                max_sharpness = 0.0
                sharpest_img = None
                black_ratio = 0.0

                errors = 0 # Chyby kamery resetujeme na začátku každého pokusu

                #core.motion_controller.send_gcode(f"G90 G1 Z{(abs_z+z_step):.3f} M3 S750 F5")

                # místo pohybu G1 použijeme jog $J abychom to mohli přerušit
                grbl_wait_for_idle() # musíme počkat na Idle než pošleme příkaz JOG
                core.motion_controller.send_gcode(f"$J=G90 Z{(abs_z+z_step):.3f} F5")
                #time.sleep(0.5) # Je třeba počkat na odpověď CNC, že změnilo status na RUN
                # Místo čekání na správnou odpověď ji pro urychlení procesu natvrdo změníme
                core.motion_controller.grbl_status = "Run"

                timeout = time.time() + 60
                while core.motion_controller.grbl_status != "Idle":
                    if time.time() > timeout:
                        print("[MICROSCOPE] GRBL neodpovídá, přerušuji snímání.")
                        break

                    img = core.camera_manager.get_image()
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        (h, w) = img_rgb.shape[:2]
                        sharpness = tenengrad_sharpness(img_rgb)
                        black_ratio, white_ratio = black_white_ratio(img_rgb, use_otsu=False,thresh_val=100)
                        # print(f"[MICROSCOPE] Ostrost obrázku {step}: {sharpness:.3f}, max:{max_sharpness:.3f}")
                        if sharpness > max_sharpness:
                            max_sharpness = sharpness
                            sharpest_img = img_rgb.copy()
                    else:
                        print("[MICROSCOPE] Chyba při získávání snímku z mikroskopu, obrázek je None.")
                        errors += 1
                        sharpness = None
                        if errors > max_errors:
                            cancel_move() # přeruší pojezd bez ztráty pozice
                            break

                    if sharpness is not None and sharpness < max_sharpness * 0.75 and sharpest_img is not None:
                        print(f"[MICROSCOPE] Ostrost klesla pod 75% max. ostrosti ({sharpness:.3f} z {max_sharpness:.3f}), poměr černé {black_ratio:.1%}")
                        cancel_move() # přeruší pojezd bez ztráty pozice
                        break

                if errors <= max_errors: # Pokud nebylo chyb snímání, pokračujeme v dalším pokusu

                    z_step += 0.1 # Zvětšíme rozsah Z pro další pokus - asi nerovný vzorek
                    if z_step > 5.0:
                        print("[MICROSCOPE] Dosáhli jsme maximálního rozsahu Z, ukončuji snímání.")
                        break

                    # Kontrola zda hrana drátu je na obrázku -jen pro 10x objektiv
                    if black_ratio > 0.7 and edge_detection:
                        # zmenšíme poloměr, pokud je okraj příliš velký
                        dx = px - cx
                        dy = py - cy
                        dist = np.hypot(dx, dy)
                        if dist > (0.8 * (black_ratio - 0.5)):
                            scale = (dist - (0.8 * (black_ratio - 0.5))) / dist
                            px = cx + dx * scale
                            py = cy + dy * scale
                            print(f"[MICROSCOPE] Vysoký poměr černé {black_ratio:.1%}, ostrost {max_sharpness:.1f}, zmenšuji poloměr na {np.hypot(px - cx, py - cy):.1f} mm")
                            max_sharpness = 0.0  # Vynulujeme ostrost, abychom pokračovali v hledání

                    # TODO: doladit problém, kdy jsou dráty zcela u sebe
                    if black_ratio < 0.3 and max_sharpness > 250 and edge_detection:
                        if black_ratio > max_black_ratio + 0.002: # přidáme 0.2% hysterezi
                            previous_px = px
                            previous_py = py
                            max_black_ratio = black_ratio

                        dx = px - cx
                        dy = py - cy
                        dist = np.hypot(dx, dy)

                        scale = (dist + (0.8 * (0.5 - black_ratio))) / dist
                        px = cx + dx * scale
                        py = cy + dy * scale
                        print(f"[MICROSCOPE] Nízký poměr černé {black_ratio:.1%}, ostrost {max_sharpness:.1f}, zvětšuji poloměr na {np.hypot(px - cx, py - cy):.1f} mm")
                        max_sharpness = 0.0 # Vynulujeme ostrost, abychom pokračovali v hledání

                    # pokud předchozí kroky situaci zhoršily, vrátime se
                    if max_black_ratio > black_ratio and attempt == 4 and edge_detection:
                        px = previous_px
                        py = previous_py
                        max_sharpness = 0.0
                        print(f"[MICROSCOPE] Poměr černé byl lepší {max_black_ratio:.1%} > {black_ratio:.1%}, vracím na původní souřadnice")

                attempt += 1
                if attempt > 5:  # Maximální počet pokusů
                    print("[MICROSCOPE] Příliš mnoho pokusů, ukončuji snímání.")
                    break

            if sharpest_img is not None:
                image_path = save_image_to_project(project_id, sharpest_img, f"microscope_{ean_code}_{pos_index}_{step}_{id}.jpg")
                core.database.update_sample_item_position_image(id, image_path)

                # Náhled do GUI
                h, w = sharpest_img.shape[:2]
                target_h, target_w = config.frame_height, config.frame_width
                aspect = w / h
                target_aspect = target_w / target_h
                if aspect > target_aspect:
                    new_w = target_w
                    new_h = int(target_w / aspect)
                else:
                    new_h = target_h
                    new_w = int(target_h * aspect)
                img = cv2.resize(sharpest_img, (new_w, new_h))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.putText(img, f"Vzorek {ean_code} - Drat {pos_index} z {len(items)} - Snimek {step+1} z {len(item_positions_db)} - Ostrost: {max_sharpness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                im_pil = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=im_pil)
                if image_label.winfo_exists():
                    image_label.imgtk = imgtk
                    image_label.config(image=imgtk)
                else:
                    print("[MICROSCOPE] Náhled již neexistuje, nemohu zobrazit obrázek.")
                # core.motion_controller.grbl_wait_for_idle() # Počkáme až se dokončí pohyb v ose Z - memí potřeba, příkazy se stackují
        # aktualizujeme tabulku
        if tree.winfo_exists():
            last_row = tree.get_children()[-1]  # Vybereme poslední přidaný řádek
            tree.item(last_row, values=(ean_code, position, f"{pos_index} z {len(items)}"))

    # container.after(0, lambda: Messagebox.show_info(f"Snímky z mikroskopu pro vzorek {ean_code} na pozici {position} byly úspěšně získány."))
    print(f"[MICROSCOPE] Snímky z mikroskopu pro vzorek {ean_code} na pozici {position} byly úspěšně získány.")
    logger.info(f"[MICROSCOPE] Snímky z mikroskopu pro vzorek {ean_code} na pozici {position} byly úspěšně získány.")
