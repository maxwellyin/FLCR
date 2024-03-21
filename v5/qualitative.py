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
from const import CHECK_NAME, DEVICE, SAMPLES, SMALL_MAG, BATCH_SIZE, NUM_WORKERS, CONTEXT_LENGTH, BERT_SIZE, CHECK_POINT, CHECK_STEP
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
def recommendBatch(context:torch.tensor, tree, model, smallMag, k=10):
    with torch.no_grad():
        context2 = context.repeat([5,1])
        clusterID = torch.IntTensor(range(5))
        clusterEmb = model.clusterEmbedding(clusterID)
        context3 = torch.cat([context2, clusterEmb], dim=1)
        emb = model.context_embedding(context3).cpu()
    distances, idxes = tree.query(emb, k=k)
    clusterTitles = []
    for i in range(idxes.shape[0]):
        titles = []
        for title in smallMag.loc[idxes[i],'paperTitle']:
            titles.append(title)
        clusterTitles.append(titles)
    return idxes, distances, clusterTitles
# %%
smallMag = pd.read_pickle(SMALL_MAG)
# %%
mag_set = kdtree.magSet(smallMag)
magLoader = DataLoader(mag_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
model = net.Net().to(DEVICE)
model.load_state_dict(torch.load(f'./check_point/{CHECK_POINT}', map_location=DEVICE))
# %%
with open(f"./check_point/tree_{CHECK_NAME}.pkl", "rb") as f:
    tree = pickle.load(f)
# %%
sBert = SentenceTransformer('../data/all-mpnet-base-v2', device=DEVICE)
# %%
inString = "the machine learning is definitely an intersting area of study. Currently, the natural language process..."
inString2 = "Active learning aims to interactively select the most informative data for accelerating the learning procedure. In this paper, we proposed a novel and principled deep active learning approach under label shift -- the labeled dataset has a different class distribution w.r.t. the unlabelled data pool. By formulating the querying procedure as a distribution matching problem, we derive the generalization guarantee under label shift. Moreover, the theoretical results inspire a unified practical framework for deep batch active learning: in the training stage, we jointly correct the label shift and align the semantic conditional distributions; as for the querying stage, label-shift-conditioned hybrid querying (LSCHQ) strategy is proposed to balance uncertainty and diversity under label shift. The experimental results showed that our method achieves state-of-the-art performances compared with some modern baselines on several benchmarks, including the real-world dataset."
inString3 = "We used 300 dimension embeddings for source language, and bi-directional LSTMs have 300 hidden units. The trained parameters are source embedding, weights and bias in the model. We randomly initialized source word embeddings sampled from uniform distribution from −0.08 to 0.08. All recurrent materices with orthogonal initialization , and non-recurrent weights are initialized from scaled uniform distribution . i-batches of size 128 are used. We used Adam algorithm for optimization. We trained models with early-stopping. The perplexities on development data for English to French, German, Czech and ish are 3.80, 6.49, 6.30, 19.25 respectively. Supersenses can be thought of a generalization of words senses into a universal inventory of semantic types. That is, as the number of word senses tend to be too numerous for existing models to generalize properly with the small amounts of data available, supersenses address this problem by clustering all senses into a tractable set of tags. Table 2 show examples of supersense tags and its definition. As such, these are generally used in semantically oriented downstream tasks such as co-reference resolution  and question answering ."
# %%
for i in range(5):
    print("group{}:".format(i))
    print(recommend(readRawString(inString3, sBert), i, tree, model, smallMag)[2])
# %%
recommendBatch(readRawString(inString3, sBert), tree, model, smallMag, k=10)[2]
# %%
# 以下代码用于去除类间重复--------------------------------------------------------
# %%
def recommend2(context:torch.tensor, clusterID: int, tree, model, k):
    with torch.no_grad():
        clusterID2 = torch.tensor([clusterID])
        clusterEmb = model.clusterEmbedding(clusterID2)
        context2 = torch.cat([context, clusterEmb], dim=1)
        emb = model.context_embedding(context2).cpu()
    _, idxes = tree.query(emb, k=k)
    return idxes[0]
# %%
top = 10
finals = []
for i in range(5):
    idxes = recommend2(readRawString(inString3, sBert), i, tree, model, k=100)
    previous = sum(finals, [])
    final = [idx for idx in idxes if idx not in previous][:top]
    finals.append(final)
# %%
for i in range(len(finals)):
    final = finals[i]
    titles = []
    for title in smallMag.loc[final,'paperTitle']:
        titles.append(title)
    print("group {}:".format(i))
    print(titles)
# %%
