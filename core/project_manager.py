import os
import re
import cv2

import config
from core.database import get_project_by_id

def create_project_folder(project_id):
    # Získání složky projektu podle ID
    result = get_project_by_id(project_id)
    if not result:
        return None
    project_id = result[0]  # ID je první sloupec v DB
    project_name = result[1]
    created_at = result[3]
    # Normalizace jména
    safe_name = re.sub(r'\W+', '_', project_name).strip('_')
    timestamp = created_at.replace(" ", "_").replace(":", "_").replace("-", "_")
    project_dir = f"{timestamp}_{project_id}_{project_name}"
    project_path = os.path.join(config.PROJECTS_DIR, project_dir)

    os.makedirs(config.PROJECTS_DIR, exist_ok=True)
    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, "images"), exist_ok=True)

    # Můžeš zde uložit i metadata např. do config.json
    with open(os.path.join(project_path, "project.txt"), "w") as f:
        f.write(f"ID: {project_id}\n")
        f.write(f"Name: {project_name}\n")
        f.write(f"Created: {timestamp}\n")

    return project_path

def get_project_folder(project_id):
    # Získání složky projektu podle ID
    result = get_project_by_id(project_id)
    if not result:
        return None
    project_id = result[0]  # ID je první sloupec v DB
    project_name = result[1]
    created_at = result[3]
    # Normalizace jména
    safe_name = re.sub(r'\W+', '_', project_name).strip('_')
    timestamp = created_at.replace(" ", "_").replace(":", "_").replace("-", "_")
    project_dir = f"{timestamp}_{project_id}_{project_name}"
    project_path = os.path.join(config.PROJECTS_DIR, project_dir)
    if os.path.exists(project_path):
        return project_path
    else:
        print(f"Složka projektu {project_path} neexistuje.")
    return None

def save_image_to_project(project_id, image, filename):
    project_folder = get_project_folder(project_id)
    if not project_folder:
        print(f"Projekt s ID {project_id} nebyl nalezen.")
        return False

    image_path = os.path.join(project_folder, "images", filename)
    cv2.imwrite(image_path, image)
    print(f"Obrázek uložen do {image_path}")
    return image_path

def get_image_from_project(image_path):
    if not os.path.exists(image_path):
        print(f"Obrázek {image_path} neexistuje.")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Chyba při načítání obrázku {image_path}.")
        return None

    return image