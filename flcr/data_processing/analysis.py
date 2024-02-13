import pickle
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

conn = sqlite3.connect(ROOT_DIR / "data" / "samplesGroups.db")
cur = conn.cursor()

cur.execute("select count(*) from samples")
rows = cur.fetchall()
length = rows[0][0]
# %%
maxGroupID = 0
for i in range(length):
    cur.execute("select sample from samples where rowid=?", (i+1,))
    rows = cur.fetchall()
    sample = pickle.loads(rows[0][0])
    maxGroupID = max(maxGroupID, sample['groupIDs'][-1])
print("maxGroupID:", maxGroupID) # maxGroupID: 445
# %%
