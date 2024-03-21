# try to make cited papers into different groups.
# %%
import sqlite3
import pickle
import pandas as pd
import datetime
import os
from multiprocessing import Pool
from const import available_cpu_count
# %%
NUM_WORKERS = available_cpu_count()
NEIGHBOR_NUM = 30
BATCH_SIZE = 1000
SAMPLES_IN = "../data/samplesTrim.db"
SAMPLES_OUT = "../data/samplesGroups.db"
SMALL_MAG = "../../v3/check_point/smallMag2.pkl"
# %%
smallMag = pd.read_pickle(SMALL_MAG)
smallMagIdIndex = smallMag.set_index('paperID').sort_index()
# %%
conn = sqlite3.connect(SAMPLES_IN)
cur = conn.cursor()
# %%
if os.path.exists(SAMPLES_OUT):
    os.remove(SAMPLES_OUT)
os.system(f'sqlite3 {SAMPLES_OUT}<./create.sql')
# %%
connOut = sqlite3.connect(SAMPLES_OUT)
curOut = connOut.cursor()
# %%
def findNeighbors(paperIDs, neighborNum=NEIGHBOR_NUM):
    neighborsIdx = smallMagIdIndex.loc[paperIDs[0], "neighbors"][:neighborNum]
    neighbors = smallMag.loc[neighborsIdx, 'paperID']
    return neighbors.tolist()
# %%
def findSameGroup(paperIDs, neighborNum=NEIGHBOR_NUM):
    neighbors = findNeighbors(paperIDs, neighborNum=neighborNum)
    group = set(paperIDs).intersection(set(neighbors))
    remain = set(paperIDs) - group
    return list(group), list(remain)
# %%
def getGroups(paperIDs, neighborNum=NEIGHBOR_NUM):
    groups = []
    while len(paperIDs)>0:
        group, paperIDs = findSameGroup(paperIDs, neighborNum=neighborNum)
        groups.append(group)
    return groups
# %% 调参用临时函数，用来选取neighborNum的值
def adjust(idx:int, neighborNum=NEIGHBOR_NUM):
    cur.execute("select sample from samples where rowid = ?", (idx+1,))
    rows = cur.fetchall()
    sample = pickle.loads(rows[0][0])
    paperIDs = sample['paperIDs']
    print(getGroups(paperIDs, neighborNum=neighborNum))
    print(smallMagIdIndex.loc[paperIDs, "paperTitle"].values)
# %% 多次尝试后发现neighborNum为30和100结果相近，30效果更好一些。 
# adjust(2, 100)
# %%
def getGroupIDs(groups):
    paperIDsNew = sum(groups, [])
    groupIDs = [0] * len(paperIDsNew)
    lengths = [len(group) for group in groups]
    lengths2 = [0]
    for i in range(len(lengths)):
        lengths2.append(sum(lengths[:i+1]))
    for i in range(len(lengths2) - 1):
        groupIDs[lengths2[i]: lengths2[i+1]] = [i] * (lengths2[i+1] - lengths2[i])
    return paperIDsNew, groupIDs
# %%
def creatSampleGroups(sampleDumps, neighborNum=NEIGHBOR_NUM):   
    sample = pickle.loads(sampleDumps)
    paperIDs = sample['paperIDs']
    groups = getGroups(paperIDs, neighborNum=neighborNum)
    paperIDsNew, groupIDs = getGroupIDs(groups)
    sample2 = sample
    sample2['paperIDs'] = paperIDsNew
    metadata = smallMagIdIndex.loc[sample2['paperIDs'], ['titleEmb', 'abstractEmb']]
    sample2['titles'] =  metadata['titleEmb'].tolist()
    sample2['abstracts'] = metadata['abstractEmb'].tolist()
    sample2['groupIDs'] = groupIDs
    sample3 = pickle.dumps(sample2)
    return sample3
# %%
cur.execute(f"""SELECT count(*) FROM samples""")
rows = cur.fetchall()
length = rows[0][0] 
length2 = length // BATCH_SIZE + 1
# %%
def dataProcess(i: int, length = length):
    downIdx = (i)*BATCH_SIZE+1
    _upIdx = (i+1)*BATCH_SIZE
    upIdx = min(_upIdx, length)+1
    cur.execute("select sample from samples where rowid>= ? and rowid<?", (downIdx, upIdx))
    rows = cur.fetchall()
    tasks = [(creatSampleGroups(row[0]),) for row in rows]
    connOut = sqlite3.connect(SAMPLES_OUT, timeout=1000)
    curOut = connOut.cursor()
    curOut.executemany("""INSERT INTO samples(sample) VALUES(?)""", tasks)
    connOut.commit()
    connOut.close()
    x = datetime.datetime.now()
    print("rowid:", upIdx-1, x)
# %%
with Pool(NUM_WORKERS) as p:
    results = p.map(dataProcess, range(length2))
print(datetime.datetime.now(), "Finish.")
# %%
