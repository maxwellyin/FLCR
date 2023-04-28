# samples.db to samples.pkl
# It takes about 15 minutes.
# %% 
import sqlite3
import pandas as pd
import pickle
import datetime
# %%
SAMPLES = "../data/samplesEnhance.db"
# %%
conn = sqlite3.connect(SAMPLES)
# %%
df = pd.read_sql_query("SELECT * from samples", conn)
# %%
for i in range(len(df)):
    df.at[i, 'sample'] = pickle.loads(df.at[i, 'sample'])
# %%
with open('../data/samplesEnhance.pkl', 'wb') as f:
    pickle.dump(df, f)
# %%
print("finish.")
# %%