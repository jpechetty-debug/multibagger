import sqlite3
import os

db_path = "runtime/stocks.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM signals")
    count = cursor.fetchone()[0]
    print(f"SIGNAL_COUNT: {count}")
    conn.close()
else:
    print(f"DATABASE NOT FOUND: {db_path}")
