import datetime
import math
import pickle
import sqlite3
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from flcr import model
from flcr.config import BATCH_SIZE, CHECK_POINT, DEVICE, NUM_WORKERS, SMALL_MAG, SAMPLES, TORCH_SEED, TREE_PATH


PART = 1000


class RecallSet(Dataset):
    def __init__(self, samples, cluster_id: int, transform=None):
        self.cluster_id = cluster_id
        self.conn = sqlite3.connect(samples)
        self.transform = transform
        cur = self.conn.cursor()
        cur.execute("SELECT count(*) FROM samples")
        self.length = cur.fetchall()[0][0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM samples WHERE rowid=?", (idx + 1,))
        sample = pickle.loads(cur.fetchall()[0][0])
        item = {"context": sample["context"], "clusterID": self.cluster_id}
        return self.transform(item) if self.transform else item


class IdSet(Dataset):
    def __init__(self, samples, transform=None):
        self.conn = sqlite3.connect(samples)
        self.transform = transform
        cur = self.conn.cursor()
        cur.execute("SELECT count(*) FROM samples")
        self.length = cur.fetchall()[0][0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM samples WHERE rowid=?", (idx + 1,))
        sample = pickle.loads(cur.fetchall()[0][0])
        item = {"context": sample["context"], "paperIDsRaw": sample["paperIDs"]}
        return self.transform(item) if self.transform else item


def encode_citing(recall_test_loader, retrieval_model: model.Net, device=DEVICE):
    embeddings = []
    retrieval_model.eval()
    with torch.no_grad():
        for batch in recall_test_loader:
            for key in batch.keys():
                if "Raw" not in key:
                    batch[key] = batch[key].to(device)
            context = batch["context"]
            cluster_id = batch["clusterID"]
            cluster_emb = retrieval_model.clusterEmbedding(cluster_id)
            embeddings.append(retrieval_model.context_embedding(torch.cat([context, cluster_emb], dim=1)))
    return torch.cat(embeddings, dim=0)


with open(TREE_PATH, "rb") as f:
    tree = pickle.load(f)

small_mag = pd.read_pickle(SMALL_MAG)
dataset = model.citationSet(SAMPLES, small_mag)
id_dataset = IdSet(SAMPLES)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

_, test_set = torch.utils.data.random_split(
    dataset,
    [train_size, test_size],
    generator=torch.manual_seed(TORCH_SEED),
)
_, id_test_set = torch.utils.data.random_split(
    id_dataset,
    [train_size, test_size],
    generator=torch.manual_seed(TORCH_SEED),
)

retrieval_model = model.Net().to(DEVICE)
retrieval_model.load_state_dict(torch.load(CHECK_POINT, map_location=DEVICE))


def find_neighbor(i, batch_size, citing_maps, k=100):
    low = i * batch_size
    up = min((i + 1) * batch_size, len(citing_maps))
    return tree.query(citing_maps[low:up], k=k, return_distance=False)


def compute_candidates(citing_maps):
    batch_size = 100
    partial_maps = citing_maps[:PART]
    length = len(partial_maps) // batch_size
    if len(partial_maps) % batch_size != 0:
        length += 1
    print(datetime.datetime.now())
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(partial(find_neighbor, batch_size=batch_size, citing_maps=partial_maps), range(length))
    print(datetime.datetime.now())
    return np.concatenate(results, axis=0)


def compute_metrics(candidates):
    k_list = [5, 10, 30, 50, 80]
    k_num = [0] * len(k_list)
    for i in range(PART):
        ground_truth_ids = id_test_set[i]["paperIDsRaw"]
        paper_ids = [small_mag.loc[idx, "paperID"] for idx in candidates[i][: k_list[-1]]]
        for j, k in enumerate(k_list):
            for paper_id in ground_truth_ids:
                if paper_id in paper_ids[:k]:
                    k_num[j] += 1
                    break
    recalls = [value / PART for value in k_num]
    print("Recall:", recalls)

    mrrs = []
    for i in range(PART):
        ground_truth_ids = id_test_set[i]["paperIDsRaw"]
        paper_ids = [small_mag.loc[idx, "paperID"] for idx in candidates[i][: k_list[-1]]]
        idxes = [paper_ids.index(paper_id) for paper_id in ground_truth_ids if paper_id in paper_ids]
        mrrs.append(1 / (min(idxes) + 1) if idxes else 0)
    print("MRR:", sum(mrrs) / len(mrrs))

    mean_aps = []
    for i in range(PART):
        ground_truth_ids = id_test_set[i]["paperIDsRaw"]
        paper_ids = [small_mag.loc[idx, "paperID"] for idx in candidates[i][: k_list[-1]]]
        idxes = sorted(paper_ids[:10].index(paper_id) for paper_id in ground_truth_ids if paper_id in paper_ids[:10])
        if idxes:
            score = sum((j + 1) / (idx + 1) for j, idx in enumerate(idxes))
            mean_aps.append(score / len(idxes))
        else:
            mean_aps.append(0)
    print("MAP:", sum(mean_aps) / len(mean_aps))

    ndcgs = []
    for i in range(PART):
        ground_truth_ids = id_test_set[i]["paperIDsRaw"]
        paper_ids = [small_mag.loc[idx, "paperID"] for idx in candidates[i][: k_list[-1]]]
        idxes = [paper_ids[:10].index(paper_id) for paper_id in ground_truth_ids if paper_id in paper_ids[:10]]
        if idxes:
            ideal = sum(1 / math.log2(j + 2) for j in range(len(idxes)))
            actual = sum(1 / math.log2(idx + 2) for idx in sorted(idxes))
            ndcgs.append(actual / ideal)
        else:
            ndcgs.append(0)
    print("nDCG:", sum(ndcgs) / len(ndcgs))

    return recalls, sum(mrrs) / len(mrrs), sum(mean_aps) / len(mean_aps), sum(ndcgs) / len(ndcgs)


outcomes = []
for cluster_id in range(5):
    print(f"clusterID:{cluster_id}")
    recall_dataset = RecallSet(SAMPLES, cluster_id=cluster_id)
    _, recall_test_set = torch.utils.data.random_split(
        recall_dataset,
        [train_size, test_size],
        generator=torch.manual_seed(TORCH_SEED),
    )
    recall_test_loader = DataLoader(
        recall_test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    citing_maps = encode_citing(recall_test_loader, retrieval_model).cpu()
    outcomes.append(compute_metrics(compute_candidates(citing_maps)))
