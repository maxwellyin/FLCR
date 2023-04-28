# %%
import pickle
import numpy as np
import datetime
import pandas as pd
from multiprocessing import Pool
from functools import partial

# %%
from const import NUM_WORKERS, CHECK_STEP, SMALL_MAG
STEP = 3
# %%
with open(f"../data/tree{CHECK_STEP}.pkl", "rb") as f:
    tree = pickle.load(f)
# %%
with open(f"../data/citedMap{CHECK_STEP}.pkl", "rb") as f:
    citedMap = pickle.load(f)
# %%
def findNeighbor(i, batchSize, citingMaps, k=100):
    low = i*batchSize
    up = (i+1)*batchSize
    up = min(up, len(citingMaps))
    # it's kind of awkward, I cannot pass the tree as a parameter for parallel
    candidates = tree.query(citingMaps[low:up], k=k, return_distance=False)
    if i % 100 == 0:
        print("i:{}, time:{}".format(i, datetime.datetime.now()))
    return candidates
# %%
print("length of cited map: {}".format(len(citedMap)))
# %%
batchSize = 100
length = len(citedMap) // batchSize
if len(citedMap) % batchSize != 0:
    length = length + 1
print(datetime.datetime.now())
with Pool(NUM_WORKERS) as p:
    results = p.map(partial(findNeighbor, batchSize=batchSize, citingMaps=citedMap), range(length))
print(datetime.datetime.now())

candidates = np.concatenate(results, axis = 0)

# %%
with open(f"../data/citedNeighbor{STEP-1}.pkl", "wb") as f:
    pickle.dump(candidates, f)
# %%
with open(f"../data/citedNeighbor{STEP-1}.pkl", "rb") as f:
    candidates = pickle.load(f)
# %%
smallMag = pd.read_pickle(SMALL_MAG)
# %%
smallMag['neighbors'] = candidates.tolist()
# %%
with open(f"../data/smallMag{STEP}.pkl", "wb") as f:
    pickle.dump(smallMag, f)
# %%
