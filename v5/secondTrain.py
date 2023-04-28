# 每次要检查是否修改 SMALL_MAG 和 CHECK_POINT
# %%
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import net

# %%
from const import SAMPLES, SMALL_MAG, DEVICE, TORCH_SEED, BATCH_SIZE, NUM_WORKERS, CHECK_POINT
# %%
smallMag = pd.read_pickle(SMALL_MAG)
# %%
rawSet = net.citationSet(SAMPLES, smallMag)
trainSize = int(0.8 * len(rawSet))
testSize = len(rawSet) - trainSize

trainSet, testSet = torch.utils.data.random_split(rawSet, [trainSize, testSize], generator = torch.manual_seed(TORCH_SEED))

train_loader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# %%
model = net.Net().to(DEVICE)
# model.load_state_dict(torch.load(f'./check_point/{CHECK_POINT}', map_location=DEVICE))
optimizer = torch.optim.Adam(model.parameters())
# %%
net.secondTraingProcess(model, train_loader, test_loader, optimizer)
# %%