# explore the data in kd-tree from v3
# use paperId as index for smallMag.
# %% 
import pickle
import pandas as pd
import sqlite3
# %%
conn = sqlite3.connect('../data/samplesGroups.db')
cur = conn.cursor()
# %%
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
