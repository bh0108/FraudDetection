import sqlite3
from pathlib import Path

DB_DIR = Path("database")
DB_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_DIR / "transactions.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    amount REAL,
    currency TEXT,
    merchant_id TEXT,
    country TEXT,
    channel TEXT,
    timestamp TEXT,
    label INTEGER
);
""")

conn.commit()
conn.close()

print("transactions.db created.")