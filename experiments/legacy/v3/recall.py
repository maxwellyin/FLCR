# need to modify CHECK_STEP each time.
# %%
import pickle
import torch
import pandas as pd
import datetime
import numpy as np
import math
import sklearn.cluster
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
from functools import partial
import net
import kdtree
# %%
from const import CHECK_POINT, SMALL_MAG, SAMPLES, TORCH_SEED, BATCH_SIZE, NUM_WORKERS, DEVICE, CHECK_STEP
# %%
class recallSet(Dataset):
    def __init__(self, samples: pd.DataFrame, smallMag:pd.DataFrame, transform=None):
        self.samples = samples
        self.smallMag = smallMag
        self.transform = transform

        self.smallMagTitleLength = len(smallMag)
        self.length = len(samples)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):      
        _sample = self.samples.iat[idx,0]

        sample = {'contextRaw': _sample['contextRaw'], 'context': _sample['context'], 'paperIDsRaw': _sample['paperIDs']}

        if self.transform:
            sample = self.transform(sample)

        return sample
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
samples = pd.read_pickle(SAMPLES)
rawSet = net.citationSet(samples, smallMag)
recallRawSet = recallSet(samples, smallMag)
trainSize = int(0.8 * len(rawSet))
testSize = len(rawSet) - trainSize

_, testSet = torch.utils.data.random_split(rawSet, [trainSize, testSize], generator = torch.manual_seed(TORCH_SEED))
_, recallTestSet = torch.utils.data.random_split(recallRawSet, [trainSize, testSize], generator = torch.manual_seed(TORCH_SEED))

test_loader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# %%
model = net.Net().to(DEVICE)
model.load_state_dict(torch.load(f'./check_point/{CHECK_POINT}', map_location=DEVICE))
# %%
citingMaps = kdtree.encodeCiting(test_loader, model).cpu()
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
mag_set = kdtree.magSet(smallMag)
magLoader = DataLoader(mag_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
titleMaps = kdtree.encodeCited(magLoader, model).cpu()
# %%
# recall cluster
kcList = [1,3,5]
kcNum = [0] * len(kcList)
for i in range(part):
    paperIDsGroudTruth = recallTestSet[i]['paperIDsRaw']
    df = smallMag.iloc[candidates[i]]
    embs = titleMaps[candidates[i]]
    cluster = sklearn.cluster.Birch(n_clusters=10).fit(embs)
    label_seq = list(dict.fromkeys(cluster.labels_.tolist()))
    outcomes = []
    for label in label_seq:
        outcome = df[cluster.labels_ == label]['paperID'].values.tolist()
        outcomes.append(outcome)
    for j, k in enumerate(kcList):
        outcomes2 = sum(outcomes[:k], [])
        for paperID in paperIDsGroudTruth:
            if paperID in outcomes2:
                kcNum[j] = kcNum[j] + 1
                break

print("Cluster recall:", [i/part for i in kcNum])
# %%

