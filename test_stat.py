import time
import os
import pathlib

p = pathlib.Path("media_shrinker.py")

s = time.time()
for _ in range(100000):
    try:
        size = pathlib.Path(p).stat().st_size
    except OSError:
        pass
print("Path(source).stat().st_size: ", time.time() - s)

s = time.time()
for _ in range(100000):
    try:
        size = p.stat().st_size
    except OSError:
        pass
print("source.stat().st_size:       ", time.time() - s)

s = time.time()
for _ in range(100000):
    try:
        size = os.stat(p).st_size
    except OSError:
        pass
print("os.stat(source).st_size:     ", time.time() - s)
