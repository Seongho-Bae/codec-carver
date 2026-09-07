import sqlite3
import os

db_path = "test_wal.db"
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
conn.execute("PRAGMA journal_mode=WAL")
conn.close()

conn = sqlite3.connect(db_path)
mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
conn.close()

print(f"Journal mode after reopen: {mode}")
