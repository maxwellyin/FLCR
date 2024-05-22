# %%
import torch
import time
import math
# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

SAMPLES = "../data/samples.pkl"
SMALL_MAG = "../data/smallMag.pkl"
CHECK_STEP = 299
CHECK_POINT = f"base/base{CHECK_STEP}.pt"

BERT_SIZE = 768
LEFT_SCOPE = 3
REGHT_SCOPE = 3
CONTEXT_LENGTH = LEFT_SCOPE + REGHT_SCOPE + 1
CITED_SIZE = 2

TORCH_SEED = 0
BATCH_SIZE = 128
NUM_WORKERS = 16
ALPHA = 0.5
EPOCHES = 300

# %%
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)