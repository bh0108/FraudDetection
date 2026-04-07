import sqlite3
from pathlib import Path

DB_DIR = Path("database")
DB_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_DIR / "alerts.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER,
    fraud_probability REAL,
    is_fraud INTEGER,
    explanation TEXT,
    created_at TEXT
);
""")

conn.commit()
conn.close()

print("alerts.db created.")