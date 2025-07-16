import os
from datetime import datetime
import re

def create_project_folder(project_id, project_name):
    # Normalizace jména
    safe_name = re.sub(r'\W+', '_', project_name).strip('_')
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_name = f"{timestamp}_{safe_name}"
    project_path = os.path.join("projects", folder_name)

    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, "images"), exist_ok=True)

    # Můžeš zde uložit i metadata např. do config.json
    with open(os.path.join(project_path, "project.txt"), "w") as f:
        f.write(f"ID: {project_id}\n")
        f.write(f"Name: {project_name}\n")
        f.write(f"Created: {timestamp}\n")

    return project_path
