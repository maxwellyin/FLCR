# %%
import torch
import time
import math
import os
import re
import subprocess
# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

SAMPLES = "../data/samplesEnhance.db"
SMALL_MAG = "./check_point/smallMag2.pkl"
CHECK_STEP = 249
CHECK_POINT = f"good/second{CHECK_STEP}.pt"

BERT_SIZE = 768
LEFT_SCOPE = 3
REGHT_SCOPE = 3
CONTEXT_LENGTH = LEFT_SCOPE + REGHT_SCOPE + 1
CITING_SIZE = 2
CITED_SIZE = 2

TORCH_SEED = 0
BATCH_SIZE = 128
ALPHA = 0.5
EPOCHES = 300

# %%
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# %%
def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError as e:
        print(e)
# %%
NUM_WORKERS = available_cpu_count()
# %%
