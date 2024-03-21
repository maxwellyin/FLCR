# need to modify CHECK_STEP each time.
# %%
import pickle
import torch
import pandas as pd
import datetime
import numpy as np
import math
import sqlite3
import sklearn.cluster
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
from functools import partial
import net
import kdtree
# %%
from const import CHECK_POINT, SMALL_MAG, SAMPLES, CITING_SIZE, TORCH_SEED, BATCH_SIZE, NUM_WORKERS, DEVICE, CHECK_STEP
# %%
class recallSet(Dataset):
    def __init__(self, samples, smallMag:pd.DataFrame, transform=None):
        self.conn = sqlite3.connect(samples)
        self.smallMag = smallMag
        self.transform = transform
        self.smallMagTitleLength = len(smallMag)
        cur = self.conn.cursor()
        cur.execute(f"""SELECT count(*) FROM samples""")
        rows = cur.fetchall()
        self.length = rows[0][0] 
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):      
        cur = self.conn.cursor()
        cur.execute("""SELECT *
                        FROM samples
                        WHERE rowid=?""", (idx+1,))
        rows = cur.fetchall()
        _sample = pickle.loads(rows[0][0])
        citingTitle = _sample['citingTitle']
        citingAbstract = _sample['citingAbstract']

        sample = {'contextRaw': _sample['contextRaw'], 'context': _sample['context'], 'citingTitle': citingTitle, 'citingAbstract': citingAbstract, 'paperIDsRaw': _sample['paperIDs']}

        if self.transform:
            sample = self.transform(sample)

        return sample

# %%
def encodeCiting(data_loader, model:net.Net, device=DEVICE):
    embs = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            for key in batch.keys():
                if 'Raw' not in key:
                    batch[key] = batch[key].to(device)
            context, citingTitle, citingAbstract = batch['context'], batch['citingTitle'], batch['citingAbstract']
            metaCiting = torch.stack([citingTitle, citingAbstract], dim=1) 
            metaCiting_= metaCiting.clone()
            for i in range(CITING_SIZE):
                metaCiting_[:,i,:] = metaCiting[:,i,:] * model.p[i] 
            metaCiting_ = metaCiting_.reshape(metaCiting_.shape[0], -1)
            citing = torch.cat([context,metaCiting_], dim = 1)
            pred = model.citing_embedding(citing)
            embs.append(pred)
    outcome = torch.cat(embs, dim = 0)
    return outcome
# %%
with open(f"./check_point/tree{CHECK_STEP}.pkl", "rb") as f:
    tree = pickle.load(f)
# %%
def findNeighbor(i, batchSize, citingMaps, k=100):
    low = i*batchSize
    up = (i+1)*batchSize
    up = min(up, len(citingMaps))
    # it's kind of awkward, I cannot pass the tree as a parameter for parallel
    candidates = tree.query(citingMaps[low:up], k=k, return_distance=False)
    return candidates
# %%
smallMag = pd.read_pickle(SMALL_MAG)
rawSet = net.citationSet(SAMPLES, smallMag)
recallRawSet = recallSet(SAMPLES, smallMag)
trainSize = int(0.8 * len(rawSet))
testSize = len(rawSet) - trainSize

_, testSet = torch.utils.data.random_split(rawSet, [trainSize, testSize], generator = torch.manual_seed(TORCH_SEED))
_, recallTestSet = torch.utils.data.random_split(recallRawSet, [trainSize, testSize], generator = torch.manual_seed(TORCH_SEED))

test_loader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# %%
model = net.Net().to(DEVICE)
model.load_state_dict(torch.load(f'./check_point/{CHECK_POINT}', map_location=DEVICE))
# %%
citingMaps = encodeCiting(test_loader, model).cpu()
# %%
part = 1000
batchSize = 100
pMaps = citingMaps[:part]
length = len(pMaps) // batchSize
if len(pMaps) % batchSize != 0:
    length = length + 1
print(datetime.datetime.now())
with Pool(NUM_WORKERS) as p:
    results = p.map(partial(findNeighbor, batchSize=batchSize, citingMaps=pMaps), range(length))
print(datetime.datetime.now())

candidates = np.concatenate(results, axis = 0)
# %%
# recall:
k_list = [5,10,30,50,80]
k_num = [0] * len(k_list)
for i in range(part):
    paperIDsGroudTruth = recallTestSet[i]['paperIDsRaw']
    paperIDs = [smallMag.loc[i, 'paperID'] for i in candidates[i][:k_list[-1]]]
    for j in range(len(k_list)):
        k = k_list[j]
        for paperID in paperIDsGroudTruth:
            if paperID in paperIDs[:k]:
                k_num[j] = k_num[j] + 1
                break

print("Recall:", [i/part for i in k_num])
# %%
mrrs = []
for i in range(part):
    paperIDsGroudTruth = recallTestSet[i]['paperIDsRaw']
    paperIDs = [smallMag.loc[i, 'paperID'] for i in candidates[i][:k_list[-1]]]
    idxes = []
    for paperID in paperIDsGroudTruth:
        if paperID in paperIDs:
            idx = paperIDs.index(paperID)
            idxes.append(idx)
    if len(idxes) != 0:
        mrr = 1/(min(idxes)+1)
    else:
        mrr = 0
    mrrs.append(mrr)

print("MRR:", sum(mrrs)/len(mrrs))
# %%
k = 10
meanAps = []
for i in range(part):
    paperIDsGroudTruth = recallTestSet[i]['paperIDsRaw']
    paperIDs = [smallMag.loc[i, 'paperID'] for i in candidates[i][:k_list[-1]]]
    idxes = []
    for paperID in paperIDsGroudTruth:
        if paperID in paperIDs[:k]:
            idx = paperIDs[:k].index(paperID)
            idxes.append(idx)
    idxes.sort()
    if len(idxes) != 0:
        num = 0
        for j in range(len(idxes)):
            num = num + (j+1)/(idxes[j]+1)
        meanAp = num / len(idxes)
    else:
        meanAp = 0
    meanAps.append(meanAp)

print("MAP:", sum(meanAps)/len(meanAps))
# %%
k = 10
nDCGs = []
for i in range(part):
    paperIDsGroudTruth = recallTestSet[i]['paperIDsRaw']
    paperIDs = [smallMag.loc[i, 'paperID'] for i in candidates[i][:k_list[-1]]]
    idxes = []
    for paperID in paperIDsGroudTruth:
        if paperID in paperIDs[:k]:
            idx = paperIDs[:k].index(paperID) # start from 0
            idxes.append(idx)

    if len(idxes) != 0:
        num = 0
        for j in range(len(idxes)):
            num = num + 1/math.log2(j+1+1)

        num2 = 0
        sortedIdxes = sorted(idxes)
        for idx in sortedIdxes:
            num2 = num2 + 1/math.log2(idx+1+1)

        nDCG = num2/num
    else:
        nDCG = 0
    nDCGs.append(nDCG)

print("nDCG:", sum(nDCGs)/len(nDCGs))
# %%
