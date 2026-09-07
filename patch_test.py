from job_store import JobStore, _SCHEMA
from datetime import datetime, timezone
import os

db_path = "test_jobs.db"
if os.path.exists(db_path):
    os.remove(db_path)

# Test before patch (not really patched here, but checking executescript behavior)
import sqlite3
with sqlite3.connect(db_path) as conn:
    conn.executescript("PRAGMA journal_mode=WAL;\n" + _SCHEMA)

mode = sqlite3.connect(db_path).execute("PRAGMA journal_mode").fetchone()[0]
print(f"Journal mode: {mode}")
