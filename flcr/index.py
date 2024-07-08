import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from flcr import model
from flcr.config import BATCH_SIZE, CHECK_POINT, CITED_MAP_PATH, DEVICE, INDEX_PATH, NUM_WORKERS, SMALL_MAG
from flcr.search import build_index, save_index


class MagSet(Dataset):
    def __init__(self, mag):
        self.mag = mag

    def __len__(self):
        return len(self.mag)

    def __getitem__(self, idx):
        title_emb, abstract_emb = self.mag.loc[idx, ["titleEmb", "abstractEmb"]]
        return {"title": title_emb, "abstract": abstract_emb}


def encode_cited(data_loader, retrieval_model: model.Net, device=DEVICE):
    embeddings = []
    retrieval_model.eval()
    encoder = retrieval_model.cited_embedding
    with torch.no_grad():
        for batch in data_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            cited = torch.cat([batch["title"], batch["abstract"]], dim=1)
            embeddings.append(encoder(cited))
    return torch.cat(embeddings, dim=0)


if __name__ == "__main__":
    small_mag = pd.read_pickle(SMALL_MAG)
    mag_loader = DataLoader(MagSet(small_mag), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    retrieval_model = model.Net().to(DEVICE)
    retrieval_model.load_state_dict(torch.load(CHECK_POINT, map_location=DEVICE))

    cited_maps = encode_cited(mag_loader, retrieval_model)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CITED_MAP_PATH, "wb") as f:
        pickle.dump(cited_maps.cpu(), f)

    index = build_index(cited_maps)
    save_index(index, INDEX_PATH)
