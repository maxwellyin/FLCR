import pickle
import sqlite3
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent

SAMPLES = ROOT_DIR / "data" / "samplesEnhance.db"

conn = sqlite3.connect(SAMPLES)
# %%
df = pd.read_sql_query("SELECT * from samples", conn)
# %%
for i in range(len(df)):
    df.at[i, 'sample'] = pickle.loads(df.at[i, 'sample'])
# %%
with open(ROOT_DIR / 'data' / 'samplesEnhance.pkl', 'wb') as f:
    pickle.dump(df, f)
print("finish.")
