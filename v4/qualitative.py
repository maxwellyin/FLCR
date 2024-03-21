# %%
import pandas as pd
import numpy as np
import torch
import nltk
import pickle
from sklearn.neighbors import KDTree
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import net
import kdtree
# %%
from const import DEVICE, SAMPLES, SMALL_MAG, BATCH_SIZE, NUM_WORKERS, CONTEXT_LENGTH, BERT_SIZE, CHECK_POINT, CHECK_STEP
# %%
# input from dataset
def readDataset(i:int, samples, smallMag):
    sampleSet = net.citationSet(samples, smallMag)
    sample = sampleSet[i]
    groundTruthTitle = (smallMag[smallMag['paperID'] == sample['paperIDRaw']]['paperTitle'].values.item())
    context = torch.tensor(sample['context']).unsqueeze(dim=0).to(DEVICE)
    return context, groundTruthTitle
# %%
# input from keyboard
def readRawString(inString: str, sBert: SentenceTransformer):
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
def recommend(context:torch.tensor, clusterID: int, tree, model, smallMag, k=10):
    with torch.no_grad():
        clusterID2 = torch.tensor([clusterID])
        clusterEmb = model.clusterEmbedding(clusterID2)
        context2 = torch.cat([context, clusterEmb], dim=1)
        emb = model.context_embedding(context2).cpu()
    distances, idxes = tree.query(emb, k=k)
    titles = []
    for title in smallMag.loc[idxes[0],'paperTitle']:
        titles.append(title)
    return idxes[0], distances[0], titles
# %%
smallMag = pd.read_pickle(SMALL_MAG)
# %%
mag_set = kdtree.magSet(smallMag)
magLoader = DataLoader(mag_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
model = net.Net().to(DEVICE)
model.load_state_dict(torch.load(f'./check_point/{CHECK_POINT}', map_location=DEVICE))
# %%
with open(f"./check_point/tree{CHECK_STEP}.pkl", "rb") as f:
    tree = pickle.load(f)
# %%
sBert = SentenceTransformer('../data/all-mpnet-base-v2', device=DEVICE)
# %%
inString = "the machine learning is definitely an intersting area of study. Currently, the natural language process..."
inString2 = "Active learning aims to interactively select the most informative data for accelerating the learning procedure. In this paper, we proposed a novel and principled deep active learning approach under label shift -- the labeled dataset has a different class distribution w.r.t. the unlabelled data pool. By formulating the querying procedure as a distribution matching problem, we derive the generalization guarantee under label shift. Moreover, the theoretical results inspire a unified practical framework for deep batch active learning: in the training stage, we jointly correct the label shift and align the semantic conditional distributions; as for the querying stage, label-shift-conditioned hybrid querying (LSCHQ) strategy is proposed to balance uncertainty and diversity under label shift. The experimental results showed that our method achieves state-of-the-art performances compared with some modern baselines on several benchmarks, including the real-world dataset."
# %%
for i in range(5):
    print("group{}:".format(i))
    print(recommend(readRawString(inString2, sBert), i, tree, model, smallMag)[2])
