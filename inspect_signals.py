import sqlite3
import os

db_path = "runtime/stocks.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, action, confidence_score, generated_at FROM signals ORDER BY generated_at DESC LIMIT 20")
    rows = cursor.fetchall()
    print("LATEST SIGNALS:")
    for row in rows:
        print(row)
    conn.close()
else:
    print(f"DATABASE NOT FOUND: {db_path}")
