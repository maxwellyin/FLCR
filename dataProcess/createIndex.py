# %%
import sqlite3
import pickle
import numpy as np
import datetime
import re
# %%
# %%
CS_PARSED_PAPER = "../../unarXive/csParsedPaper.db"
LEFT_SCOPE = 3
REGHT_SCOPE = 3
BATCH_SIZE = 1000
# %%
connCsParsedPaper = sqlite3.connect(CS_PARSED_PAPER)
curCsParsedPaper = connCsParsedPaper.cursor()

# %%
curCsParsedPaper.execute(f"""SELECT count(*) FROM csUniqueID""")
rows = curCsParsedPaper.fetchall()
length = rows[0][0]
length2 = length // BATCH_SIZE + 1
# %%
tasks = []
for idx in range(length2):
    downIdx = (idx)*BATCH_SIZE+1
    upIdx = (idx+1)*BATCH_SIZE+1
    curCsParsedPaper.execute(f"""SELECT arxiveID, citationIdx FROM csUniqueID
                                WHERE rowid >= {downIdx} and rowid < {upIdx}""")
    rows = curCsParsedPaper.fetchall()
    for i, row in enumerate(rows):
        arxiveID, citationIdxs = row
        citationIdxs2 = pickle.loads(citationIdxs)
        for j in citationIdxs2:
            tasks.append((arxiveID, j))

# %%
with open("../data/taskIdx.pkl", 'wb') as f:
    pickle.dump(tasks, f)
# %%
