import pickle
import sqlite3
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent

SAMPLES = ROOT_DIR / "data" / "samplesPublic.db"

con = sqlite3.connect(SAMPLES)
cur= con.cursor()
# %%
cur.execute("select * from samples")
rows = cur.fetchall()
# %%
rows2 = [pickle.loads(row[0]) for row in rows]
# %%
df = pd.DataFrame(rows2)
# %%
with open(ROOT_DIR / 'data' / 'samplesPublic.pkl', 'wb') as f:
    pickle.dump(df, f)
