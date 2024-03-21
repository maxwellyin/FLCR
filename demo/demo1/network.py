# %%
import pickle
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

SAMPLES = "../../data/samples.pkl"
SMALL_MAG = "../../data/smallMagAfterFilter.pkl"
CHECK_STEP = 299
CHECK_POINT = f"good/second{CHECK_STEP}.pt"
BERT_SIZE = 768
LEFT_SCOPE = 3
REGHT_SCOPE = 3
CONTEXT_LENGTH = LEFT_SCOPE + REGHT_SCOPE + 1
CITED_SIZE = 2

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

# %%
smallMag = pd.read_pickle(SMALL_MAG)
# %%
model = Net().to(DEVICE)
model.load_state_dict(torch.load(f'./resource/{CHECK_POINT}', map_location=DEVICE))
# %%
with open(f"./resource/tree{CHECK_STEP}.pkl", 'rb') as f:
    tree = pickle.load(f)
with open(f"./resource/citedMap{CHECK_STEP}.pkl", 'rb') as f:
    citedMap = pickle.load(f)
# %%
sBert = SentenceTransformer('../../data/all-mpnet-base-v2', device=DEVICE)
# %%
def recommend(inString: str, k=10):
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
def recommendCluster(inString: str, k=2000, n_clusters=10, display=3):
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
    kmeans = Birch(n_clusters=n_clusters).fit(embs)
    label_seq = list(dict.fromkeys(kmeans.labels_.tolist()))

    outcomes = []
    for label in label_seq:
        outcome = df[kmeans.labels_ == label]['paperTitle'].values.tolist()
        outcomes.append(outcome[:display])

    return outcomes
# %%
# recommend("deep learning")