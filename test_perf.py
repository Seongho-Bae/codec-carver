import time
import os
import sqlite3
from contextlib import contextmanager

db_path = "perf.db"

def test_redundant_pragma():
    if os.path.exists(db_path): os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    conn.close()

    start = time.perf_counter()
    for _ in range(1000):
        c = sqlite3.connect(db_path)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("SELECT * FROM t")
        c.close()
    return time.perf_counter() - start

def test_single_pragma():
    if os.path.exists(db_path): os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    conn.close()

    start = time.perf_counter()
    for _ in range(1000):
        c = sqlite3.connect(db_path)
        c.execute("SELECT * FROM t")
        c.close()
    return time.perf_counter() - start

print(f"Redundant PRAGMA: {test_redundant_pragma():.4f}s")
print(f"Single PRAGMA:    {test_single_pragma():.4f}s")
