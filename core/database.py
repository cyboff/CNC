import sqlite3
import os
from datetime import datetime
from core.logger import logger

DB_PATH = os.path.join("data", "database.db")

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS project_sample_items (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  sample_id INTEGER,
                  name TEXT,
                  notes TEXT,
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
            detected_detected BOOLEAN DEFAULT 0,
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


def insert_sample(project_id, position, ean_code):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO project_samples (project_id, position, ean_code) VALUES (?, ?, ?)", (project_id, position, ean_code))
    conn.commit()
    conn.close()



def get_project_by_id(project_id):
    conn = sqlite3.connect("data/database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    result = c.fetchone()
    conn.close()
    return result


def delete_project(project_id):
    conn = sqlite3.connect("data/database.db")
    c = conn.cursor()
    c.execute("DELETE FROM projects WHERE id=?", (project_id,))
    conn.commit()
    conn.close()




def save_project_samples_to_db(project_id: int, sample_codes: list[str], sample_positions: list[tuple[str, list]]):
    if len(sample_codes) != len(sample_positions):
        logger.error(f"[DB] Počet kódů ({len(sample_codes)}) neodpovídá počtu pozic ({len(sample_positions)})")
        raise ValueError("Počet EAN kódů neodpovídá počtu pozic.")

    conn = sqlite3.connect("data/database.db")
    c = conn.cursor()

    for code, (position, items) in zip(sample_codes, sample_positions):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO project_samples (project_id, position, ean_code, created_at) VALUES (?, ?, ?, ?)", (project_id, position,code, now))
        logger.info(f"[DB] Uložen vzorek {code} na pozici {position}.")

    conn.commit()
    conn.close()
