# need to modify CHECK_STEP each time.
# %%
import pickle
import torch
import numpy as np
import pandas as pd
import nltk
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KDTree
from sentence_transformers import SentenceTransformer
import net
# %%
from const import DEVICE, SMALL_MAG, BATCH_SIZE, NUM_WORKERS, CONTEXT_LENGTH, BERT_SIZE, CHECK_POINT, CHECK_STEP
# %%
class magSet(Dataset):
    def __init__(self, mag):
        self.mag = mag
    def __len__(self):
        return len(self.mag)
    def __getitem__(self, idx):
        titleEmb, abstractEmb = self.mag.loc[idx, ["titleEmb", "abstractEmb"]]
        sample = {"title": titleEmb, "abstract": abstractEmb}
        return sample
# %%
def encodeCited(data_loader, model:net.Net, device=DEVICE):
    embs = []
    model.eval()
    encoder = model.cited_embedding
    with torch.no_grad():
        for batch in data_loader:
            for key in batch.keys():
                if 'Raw' not in key:
                    batch[key] = batch[key].to(device)
            cited = torch.cat([batch['title'], batch['abstract']], dim=1)
            pred = encoder(cited)
            embs.append(pred)
    outcome = torch.cat(embs, dim = 0)
    return outcome

# %%
if __name__ == "__main__":
    smallMag = pd.read_pickle(SMALL_MAG)
    mag_set = magSet(smallMag)
    magLoader = DataLoader(mag_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = net.Net().to(DEVICE)
    model.load_state_dict(torch.load(f'./check_point/{CHECK_POINT}', map_location=DEVICE))
# %%
# Build a kd tree
if __name__ == "__main__":
    citedMaps = encodeCited(magLoader, model)
    with open(f"./check_point/citedMap{CHECK_STEP}.pkl", "wb") as f:
        pickle.dump(citedMaps.cpu(), f)

    tree = KDTree(citedMaps.cpu(), metric='euclidean')

    with open(f"./check_point/tree{CHECK_STEP}.pkl", "wb") as f:
        pickle.dump(tree, f)
# %%
