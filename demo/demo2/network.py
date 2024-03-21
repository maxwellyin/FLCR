# %%
import pickle
from sklearn import cluster
import torch
import torch.nn as nn
import pandas as pd
import random
import nltk
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans,Birch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
# %%
DEVICE = torch.device('cpu')
SMALL_MAG = "./resource/v3smallMag2.pkl"

BERT_SIZE = 768
LEFT_SCOPE = 3
REGHT_SCOPE = 3
CONTEXT_LENGTH = LEFT_SCOPE + REGHT_SCOPE + 1
CITED_SIZE = 2
EMB_SIZE = 256

TORCH_SEED = 0
BATCH_SIZE = 128
NUM_WORKERS = 12
ALPHA = 0.5
EPOCHES = 10

# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_embedding = nn.Linear(CONTEXT_LENGTH*BERT_SIZE, 2*BERT_SIZE)
        self.cited_embedding = nn.Linear(CITED_SIZE*BERT_SIZE, 2*BERT_SIZE)

    def forward(self, context, cited):
        context_embedded = self.context_embedding(context)
        cited_embedded = self.cited_embedding(cited)
        x1 = (context_embedded - cited_embedded)**2
        x2 = (x1.sum(axis=1))**(1/2)
        return x2

class ClusterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.clusterEmbedding = nn.Embedding(700, EMB_SIZE)
        self.context_embedding = nn.Linear(CONTEXT_LENGTH*BERT_SIZE + EMB_SIZE, 2*BERT_SIZE)
        self.cited_embedding = nn.Linear(CITED_SIZE*BERT_SIZE, 2*BERT_SIZE)

    def forward(self, context, clusterID, cited):
        clusterEmb = self.clusterEmbedding(clusterID)
        context2 = torch.cat([context, clusterEmb], dim=1)
        context_embedded = self.context_embedding(context2)
        cited_embedded = self.cited_embedding(cited)
        x1 = (context_embedded - cited_embedded)**2
        x2 = (x1.sum(axis=1))**(1/2)
        return x2
# %%
smallMag = pd.read_pickle(SMALL_MAG)
# %%
model = Net().to(DEVICE)
model.load_state_dict(torch.load(f'./resource/v3second299.pt', map_location=DEVICE))
clusterModel = ClusterNet().to(DEVICE)
clusterModel.load_state_dict(torch.load(f'./resource/v5cnNum30_299.pt', map_location=DEVICE))
# %%
with open(f"./resource/v3tree299.pkl", 'rb') as f:
    tree = pickle.load(f)
with open(f"./resource/v5tree_cnNum30_299.pkl", 'rb') as f:
    clusterTree = pickle.load(f)
# %%
sBert = SentenceTransformer('../../data/all-mpnet-base-v2', device=DEVICE)
# %%
def recommend(inString: str, model = model, k=10):
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

    outcomes = smallMag.loc[idxes[0], 'paperTitle'].tolist()

    return outcomes

# %%
# input from keyboard
def readRawString(inString: str):
    inStrings = nltk.sent_tokenize(inString)
    sEmbs = sBert.encode(inStrings)
    if len(sEmbs) < CONTEXT_LENGTH:
        pad = np.zeros([CONTEXT_LENGTH - len(sEmbs), BERT_SIZE], dtype=np.float32)
        sEmbs = np.concatenate([sEmbs, pad])
    if len(sEmbs) > CONTEXT_LENGTH:
        sEmbs = sEmbs[:CONTEXT_LENGTH]
    context = torch.tensor(sEmbs).reshape(-1).unsqueeze(dim=0).to(DEVICE)
    return context
# %%
def recommendBatch(context:torch.tensor, model = clusterModel, k=10):
    with torch.no_grad():
        context2 = context.repeat([5,1])
        clusterID = torch.IntTensor(range(5))
        clusterEmb = model.clusterEmbedding(clusterID)
        context3 = torch.cat([context2, clusterEmb], dim=1)
        emb = model.context_embedding(context3).cpu()
    distances, idxes = clusterTree.query(emb, k=k)
    clusterTitles = []
    for i in range(idxes.shape[0]):
        titles = []
        for title in smallMag.loc[idxes[i],'paperTitle']:
            titles.append(title)
        clusterTitles.append(titles)
    return clusterTitles
# %%
