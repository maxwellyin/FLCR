# samples.db to samples.pkl
# 30GB内存不够用，算了，不转换了。
# %% 
import sqlite3
import pandas as pd
import pickle
import datetime
# %%
SAMPLES_IN = "../data/samplesGroups.db"
SAMPLES_OUT = "../data/samplesGroups.pkl"
# %%
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