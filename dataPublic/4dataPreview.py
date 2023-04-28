# %%
import datasets
# %%
citationSet = datasets.load_dataset("pandas", data_files="../data/samplesPublic.pkl")
citationSet.save_to_disk("../data/samplesPublic")
# %%
citationSet = datasets.load_from_disk("../data/samplesPublic")
# %%
len(citationSet['train'].unique('citingID'))
# %%
