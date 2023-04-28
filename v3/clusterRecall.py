# 计算cluster 后的recall
# need to modify CHECK_STEP each time.
# %%
import pickle
import torch
import pandas as pd
import datetime
import numpy as np
from multiprocessing import Manager
from sklearn.neighbors import KDTree
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
def findNeighbor(i, batchSize, citingMaps, citedMap, k=2000):
    low = i*batchSize
    up = (i+1)*batchSize
    up = min(up, len(citingMaps))
    # it's kind of awkward, I cannot pass the tree as a parameter for parallel
    idxes = tree.query(citingMaps[low:up], k=k, return_distance=False)
    candidates = []
    for i in range(idxes.shape[0]):
        outcomes = []
        subIdxes = idxes[i]
        for _ in range(10):
            embs = citedMap[subIdxes]
            smallTree = KDTree(embs, metric='euclidean')
            emb2 = citedMap[subIdxes[0]] #总是选择分最高的一个找邻居
            emb3 = np.expand_dims(emb2, axis=0)
            idxes2 = smallTree.query(emb3, k=10, return_distance=False)
            used = [subIdxes[i] for i in idxes2[0]]
            subIdxes = list(filter(lambda x: x not in used, subIdxes[1:]))
            outcomes += used
        candidates.append(outcomes)
    return np.array(candidates)
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
citingMaps = kdtree.encodeCiting(test_loader, model).cpu().numpy()

with open(f"./check_point/citedMap{CHECK_STEP}.pkl", 'rb') as f:
    citedMap = pickle.load(f).numpy()
# %%
part = 1000
batchSize = 100
pMaps = citingMaps[:part]
length = len(pMaps) // batchSize

if len(pMaps) % batchSize != 0:
    length = length + 1
print(datetime.datetime.now())
with Pool(NUM_WORKERS) as p:
    results = p.map(partial(findNeighbor, batchSize=batchSize, citingMaps=pMaps, citedMap=citedMap), range(length))
print(datetime.datetime.now())

candidates = np.concatenate(results, axis = 0)
# %%
# recall:
k_list = [10,20,30,40,50,100]
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
#        Top1    Top2   Top3   Top4   Top5   Top10
#Recall: [0.321, 0.444, 0.502, 0.534, 0.571, 0.668]