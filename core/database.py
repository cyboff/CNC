import sqlite3
import os
import json
from datetime import datetime
from core.logger import logger

DB_PATH = os.path.join("data", "database.db")

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
    ''')
    # TODO: Pokud neexistuje, přidat defaultní nastavení parametrů do settings

    c.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            comment TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS project_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            position TEXT,
            ean_code TEXT,
            image_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS project_sample_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER,
            position_index INTEGER,
            x_center REAL,
            y_center REAL,
            radius REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(sample_id) REFERENCES project_samples(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS project_sample_item_positions
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_item_id INTEGER,
            position_index INTEGER,
            x_coord REAL,
            y_coord REAL,
            image_path TEXT,
            defect_detected BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(sample_item_id) REFERENCES project_sample_items(id)
        )
    ''')

    conn.commit()
    conn.close()

def insert_project(name, comment):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO projects (name, comment, created_at) VALUES (?, ?, ?)", (name, comment, now))
    conn.commit()
    conn.close()

def get_all_projects():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, comment, created_at FROM projects ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def get_project_by_id(project_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    result = c.fetchone()
    conn.close()
    return result


def delete_project(project_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM projects WHERE id=?", (project_id,))
    conn.commit()
    conn.close()

def save_project_sample_to_db(project_id: int, position: str, code: str, image_path: str = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #TODO: Přidat kontrolu, sample s tímto kódem již v databázi neexistuje

    c.execute("INSERT INTO project_samples (project_id, position, ean_code, image_path, created_at) VALUES (?, ?, ?, ?, ?)",
              (project_id, position, code, image_path, now))
    sample_id = c.lastrowid
    logger.info(f"[DB] Uložen vzorek {code} na pozici {position}.")

    conn.commit()
    conn.close()
    return sample_id

def get_samples_by_project_id(project_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, position, ean_code, image_path FROM project_samples WHERE project_id = ?", (project_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def save_sample_items_to_db(sample_id, items = list):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (x_center,y_center, radius) in enumerate(items):
        pos = i+1
        # TODO: Přidat kontrolu, sample s tímto kódem již v databázi neexistuje
        c.execute(
            "INSERT INTO project_sample_items (sample_id, position_index, x_center, y_center, radius, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (sample_id, pos, x_center, y_center, radius, now))
        logger.info(f"[DB] Uložen drát {pos} vzorku {sample_id} - x:{x_center} y:{y_center} r:{radius}.")

    conn.commit()
    conn.close()

def delete_sample_items_from_project(project_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM project_sample_item_positions WHERE sample_item_id IN (SELECT id FROM project_sample_items WHERE sample_id IN (SELECT id FROM project_samples WHERE project_id = ?))", (project_id,))
    c.execute("DELETE FROM project_sample_items WHERE sample_id IN (SELECT id FROM project_samples WHERE project_id = ?)", (project_id,))
    c.execute("DELETE FROM project_samples WHERE project_id = ?", (project_id,))
    conn.commit()
    conn.close()
    logger.info(f"[DB] Všechny položky vzorků pro projekt {project_id} byly smazány.")

def get_sample_items_by_sample_id(sample_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, position_index, x_center, y_center, radius FROM project_sample_items WHERE sample_id = ?", (sample_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def save_sample_item_positions_to_db(item_id:int, step:int, px:float, py:float, image_path:str = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO project_sample_item_positions (sample_item_id, position_index, x_coord, y_coord, image_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (item_id, step, px, py, image_path, now))
    logger.info(f"[DB] Do databáze byl bod mikroskopu {step} pro položku {item_id} - x:{px} y:{py}.")

    conn.commit()
    conn.close()

def get_sample_item_positions_by_item_id(item_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, position_index, x_coord, y_coord, image_path, defect_detected FROM project_sample_item_positions WHERE sample_item_id = ?", (item_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def update_sample_item_position_image(position_id: int, image_path: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE project_sample_item_positions SET image_path = ? WHERE id = ?", (image_path, position_id))
    conn.commit()
    conn.close()
    logger.info(f"[DB] Aktualizován obrázek pro pozici {position_id} na {image_path}.")

def update_sample_item_position_defect(position_id: int, defect: bool):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE project_sample_item_positions SET defect_detected = ? WHERE id = ?", (1 if defect else 0, position_id))
    conn.commit()
    conn.close()
    logger.info(f"[DB] Aktualizována detekce defektu pro pozici {position_id} na {defect}.")