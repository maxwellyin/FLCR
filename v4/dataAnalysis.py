# %% 生成每个sample引用文章篇数直方图
import pandas as pd
import matplotlib.pyplot as plt
# %%
from const import SAMPLES
# %%
samples = pd.read_pickle(SAMPLES)
# %%
lengths = list([len(samples.iloc[i][0]["paperIDs"]) for i in range(len(samples))])
# %%
tmp = plt.hist(lengths, range(20))
plt.savefig("./check_point/citedNum.png")
plt.show()
# %%
max(lengths) # 634 有点夸张
# %%
