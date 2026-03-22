import sqlite3
import os

db_path = "runtime/ops.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT level, component, message, context_json FROM logs WHERE message = 'regime detected' ORDER BY created_at DESC LIMIT 5")
    rows = cursor.fetchall()
    print("REGIME LOGS:")
    for row in rows:
        print(row)
    conn.close()
else:
    print(f"DATABASE NOT FOUND: {db_path}")
