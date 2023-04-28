# %%
from multiprocessing import Pool, Manager
import sqlite3
import pickle
import numpy as np
import datetime
import pandas as pd
# %%
CITED = False
SMALL_MAG = "../../unarXive/data/smallMag.db" if CITED else "../../unarXive/data/citingMag.db"
BATCH_SIZE = 1000
NUM_WORKERS = 16
OUT_MAG = "../data/smallMag.pkl" if CITED else "../data/citingMag.pkl"
# %%
connSmallMag = sqlite3.connect(SMALL_MAG)
curSmallMag = connSmallMag.cursor()
curSmallMag.execute("""SELECT count(*) FROM papers""")
rows = curSmallMag.fetchall()
smallMagTitleLength = rows[0][0]
connSmallMag.close()
# %%
def f(idx: int):
    connSmallMag = sqlite3.connect(SMALL_MAG)
    curSmallMag = connSmallMag.cursor()
    curSmallMag.execute("""SELECT paperID, paperTitle, titleEmb FROM papers
                    WHERE rowid = ?""", (idx+1,))
    rows = curSmallMag.fetchall()
    paperID, paperTitle, titleEmb = rows[0]

    curSmallMag.execute("""SELECT paperID, abstractEmb FROM abstractIndex
                    WHERE paperID = ?""", (paperID,))
    rows2 = curSmallMag.fetchall()
    if len(rows2) == 0:
        return
    _, abstractEmb = rows2[0]

    curSmallMag.execute("""SELECT aIdx FROM paperAuthorIdx
                    WHERE paperID = ?""", (paperID,))
    rows3 = curSmallMag.fetchall()
    if len(rows3) == 0:
        return
    aIdxes = [row [0] for row in rows3]

    titleEmb2 = pickle.loads(titleEmb)
    abstractEmb2 = pickle.loads(abstractEmb)

    sample = {'paperID': paperID , 'paperTitle':paperTitle, 'titleEmb': [titleEmb2], 'abstractEmb': [abstractEmb2], 'authorIdxes':[aIdxes]}
    if (idx % 1000 == 0):
        print('idx:{}, time:{}'.format(idx, datetime.datetime.now()))
    return pd.DataFrame(sample)
# %%
print(datetime.datetime.now())
with Pool(NUM_WORKERS) as p:
    results = p.map(f, range(smallMagTitleLength))
print(datetime.datetime.now())
# %%
smallMagAfterFilter = pd.concat(results, ignore_index=True)
# %%
smallMagAfterFilter.to_pickle(OUT_MAG)
# %%
