# %%
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import net

# %%
from const import SAMPLES, SMALL_MAG, DEVICE, TORCH_SEED, BATCH_SIZE, NUM_WORKERS
# %%
smallMag = pd.read_pickle(SMALL_MAG)
samples = pd.read_pickle(SAMPLES)
rawSet = net.citationSet(samples, smallMag)
trainSize = int(0.8 * len(rawSet))
testSize = len(rawSet) - trainSize

trainSet, testSet = torch.utils.data.random_split(rawSet, [trainSize, testSize], generator = torch.manual_seed(TORCH_SEED))

train_loader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# %%
if __name__ == "__main__":
    model = net.Net().to(DEVICE)
    # model.load_state_dict(torch.load(f'./check_point/good/euclid_a_0_5_epoch_229.pt', map_location=DEVICE))
    optimizer = torch.optim.Adam(model.parameters())
    net.traing_process(model, train_loader, test_loader, optimizer)
# %%
