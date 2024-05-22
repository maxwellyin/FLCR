import datetime
import os
import pickle
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

SAMPLE_IN = ROOT_DIR / "data" / "samples.db"
SAMPLE_OUT = ROOT_DIR / "data" / "samplesTrim.db"

if os.path.exists(SAMPLE_OUT):
    os.remove(SAMPLE_OUT)
os.system(f"sqlite3 {SAMPLE_OUT} < {SCRIPT_DIR / 'create.sql'}")

connIn = sqlite3.connect(SAMPLE_IN)
curIn = connIn.cursor()

connOut = sqlite3.connect(SAMPLE_OUT)
curOut = connOut.cursor()
# %%
curIn.execute("SELECT COUNT(*) FROM samples")
rows = curIn.fetchall()
sampleLen = rows[0][0]
# %%
for i in range(sampleLen):
    curIn.execute("SELECT sample FROM samples where rowid=?", (i+1,))
    rows = curIn.fetchall()
    sample = rows[0][0]
    sample2 = pickle.loads(sample)
    if len(sample2['paperIDs']) < 5:
        continue
    curOut.execute("""INSERT INTO samples(sample) VALUES(?)""", (sample,))
    if i % 10000 == 0:
        print(datetime.datetime.now())
        connOut.commit()
connOut.commit()
# %%
