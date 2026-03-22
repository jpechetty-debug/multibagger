import sqlite3
import os

db_path = "runtime/ops.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT level, component, message, context_json, created_at FROM logs WHERE level = 'ERROR' ORDER BY created_at DESC LIMIT 20")
    rows = cursor.fetchall()
    print("ERROR LOGS:")
    for row in rows:
        print(f"[{row[0]}] {row[1]}: {row[2]} | {row[3]}")
    conn.close()
else:
    print(f"DATABASE NOT FOUND: {db_path}")
