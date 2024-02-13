# samples.db to samples.pkl
# 30GB内存不够用，算了，不转换了。
import pickle
import sqlite3
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent

SAMPLES_IN = ROOT_DIR / "data" / "samplesGroups.db"
SAMPLES_OUT = ROOT_DIR / "data" / "samplesGroups.pkl"

conn = sqlite3.connect(SAMPLES_IN)
# %%
df = pd.read_sql_query("SELECT * from samples", conn)
# %%
for i in range(len(df)):
    df.at[i, 'sample'] = pickle.loads(df.at[i, 'sample'])
# %%
with open(SAMPLES_OUT, 'wb') as f:
    pickle.dump(df, f)
# %%
print("finish.")
# %%
