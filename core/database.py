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
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            ean_code TEXT,
            position_x REAL,
            position_y REAL,
            FOREIGN KEY(project_id) REFERENCES projects(id)
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


def insert_sample(project_id, ean_code):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO samples (project_id, ean_code) VALUES (?, ?)", (project_id, ean_code))
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




def save_sample_positions_to_db(sample_codes: list[str], sample_positions: list[tuple[int, int]]):
    if len(sample_codes) != len(sample_positions):
        logger.error(f"[DB] Počet kódů ({len(sample_codes)}) neodpovídá počtu pozic ({len(sample_positions)})")
        raise ValueError("Počet EAN kódů neodpovídá počtu pozic.")

    conn = sqlite3.connect("data/database.db")
    c = conn.cursor()

    for code, (x, y) in zip(sample_codes, sample_positions):
        c.execute("INSERT INTO sample_positions (ean_code, x, y) VALUES (?, ?, ?)", (code, x, y))
        logger.info(f"[DB] Uložen vzorek {code} na pozici ({x}, {y})")

    conn.commit()
    conn.close()
