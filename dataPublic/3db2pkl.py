# %%
import pickle
import sqlite3
import pandas as pd
# %%
SAMPLES = "../data/samplesPublic.db"
# %%
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
with open('../data/samplesPublic.pkl', 'wb') as f:
    pickle.dump(df, f)