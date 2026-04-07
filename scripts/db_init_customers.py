import sqlite3
from pathlib import Path

DB_DIR = Path("database")
DB_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_DIR / "customers.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    segment TEXT,
    risk_score REAL,
    kyc_status TEXT
);
""")

conn.commit()
conn.close()

print("customers.db created.")