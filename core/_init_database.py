import sqlite3

conn = sqlite3.connect("data/database.db")
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
''')

conn.commit()
conn.close()
