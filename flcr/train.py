import pandas as pd
import torch
from torch.utils.data import DataLoader

from flcr import model
from flcr.config import BATCH_SIZE, CHECK_POINT, DEVICE, NUM_WORKERS, SMALL_MAG, SAMPLES, TORCH_SEED


small_mag = pd.read_pickle(SMALL_MAG)
dataset = model.citationSet(SAMPLES, small_mag)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = torch.utils.data.random_split(
    dataset,
    [train_size, test_size],
    generator=torch.manual_seed(TORCH_SEED),
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

retrieval_model = model.Net().to(DEVICE)
# retrieval_model.load_state_dict(torch.load(CHECK_POINT, map_location=DEVICE))
optimizer = torch.optim.Adam(retrieval_model.parameters())

model.secondTraingProcess(retrieval_model, train_loader, test_loader, optimizer)
