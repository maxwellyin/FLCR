# %%
from sentence_transformers import SentenceTransformer
# %%
model = SentenceTransformer('all-mpnet-base-v2')
model.save('../data/all-mpnet-base-v2')
# %%
