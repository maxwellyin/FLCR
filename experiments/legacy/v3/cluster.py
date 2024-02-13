# 对一个输入句子推荐cluster，类似demo
# %%
from sklearn import neighbors
import torch
import nltk
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, Birch, DBSCAN
import kdtree
import net
# %%
from const import BERT_SIZE, CONTEXT_LENGTH, DEVICE, SMALL_MAG, BATCH_SIZE, NUM_WORKERS, CHECK_STEP, CHECK_POINT
# %%
smallMag = pd.read_pickle(SMALL_MAG)
mag_set = kdtree.magSet(smallMag)
magLoader = DataLoader(mag_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = net.Net().to(DEVICE)
model.load_state_dict(torch.load(f'./check_point/{CHECK_POINT}', map_location=DEVICE))
# %%
with open(f"../data/tree{CHECK_STEP}.pkl", "rb") as f:
    tree = pickle.load(f)
# %%
with open(f"../data/citedMap{CHECK_STEP}.pkl", 'rb') as f:
    citedMap = pickle.load(f)
# %%
sBert = SentenceTransformer('../data/all-mpnet-base-v2', device=DEVICE)
# %%
k = 2000
n_clusters = 10
inString = """in the training stage, we jointly correct the label shift and align the semantic conditional distributions; as for the querying stage, label-shift-conditioned hybrid querying (LSCHQ) strategy is proposed to balance uncertainty and diversity under label shift"""
inStrings = nltk.sent_tokenize(inString)
sEmbs = sBert.encode(inStrings)
if len(sEmbs) < CONTEXT_LENGTH:
    pad = np.zeros([CONTEXT_LENGTH - len(sEmbs), BERT_SIZE], dtype=np.float32)
    sEmbs = np.concatenate([sEmbs, pad])
if len(sEmbs) > CONTEXT_LENGTH:
    sEmbs = sEmbs[:CONTEXT_LENGTH]
context = torch.tensor(sEmbs).reshape(-1).unsqueeze(dim=0).to(DEVICE)
with torch.no_grad():
    emb = model.context_embedding(context).cpu()
idxes = tree.query(emb, k=k, return_distance=False)

df = smallMag.iloc[idxes[0]]
embs = citedMap[idxes[0]]
# %%
clusters = Birch(n_clusters=n_clusters).fit(embs)
# %%
label_seq = list(dict.fromkeys(clusters.labels_.tolist()))
# %%
outcomes = []
for label in label_seq:
    outcome = df[clusters.labels_ == label]['paperTitle'].values.tolist()
    outcomes.append(outcome)
# %% 用了一个tree，不同类里面的文章可能有重复
smallTree = KDTree(embs, metric='euclidean')
records = [0] * len(idxes[0])
outcomes2 = []
for _ in range(10):
    j = records.index(0)
    emb2 = citedMap[idxes[0][j]].unsqueeze(dim=0)
    idxes2 = smallTree.query(emb2, k=10, return_distance=False)
    for i in idxes2[0]:
        records[i] = 1
    df = smallMag.iloc[idxes[0][idxes2[0]]]
    outcome = df['paperTitle'].values
    outcomes2.append(outcome)
# %% 每次重新建tree，没有重复
outcomes3 = []
subIdxes = idxes[0]
for _ in range(10):
    embs = citedMap[subIdxes]
    smallTree = KDTree(embs, metric='euclidean')
    emb2 = citedMap[subIdxes[0]].unsqueeze(dim=0)
    idxes2 = smallTree.query(emb2, k=10, return_distance=False)
    used = [subIdxes[i] for i in idxes2[0]]
    subIdxes = list(filter(lambda x: x not in used, subIdxes[1:]))
    df = smallMag.iloc[used]
    outcome = df['paperTitle'].values
    outcomes3.append(outcome)    
# %%
outcomes3
# %%