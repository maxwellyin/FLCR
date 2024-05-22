from pathlib import Path
import math
import re
import time

import torch


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
RUNS_DIR = ARTIFACTS_DIR / "runs"
DATA_DIR = ROOT_DIR / "data"
LEGACY_DIR = ROOT_DIR / "experiments" / "legacy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLES = DATA_DIR / "samplesGroups.db"
SMALL_MAG = LEGACY_DIR / "v3" / "check_point" / "smallMag2.pkl"
CHECK_STEP = 299
CHECK_NAME = f"base{CHECK_STEP}"
CHECK_POINT = CHECKPOINT_DIR / "good" / f"{CHECK_NAME}.pt"
TREE_PATH = CHECKPOINT_DIR / f"tree_{CHECK_NAME}.pkl"
CITED_MAP_PATH = CHECKPOINT_DIR / f"citedMap_{CHECK_NAME}.pkl"

BERT_SIZE = 768
LEFT_SCOPE = 3
REGHT_SCOPE = 3
CONTEXT_LENGTH = LEFT_SCOPE + REGHT_SCOPE + 1
CITED_SIZE = 2
EMB_SIZE = 256

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


NUM_WORKERS = available_cpu_count()
