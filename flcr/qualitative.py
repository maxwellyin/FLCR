import nltk
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from flcr import model
from flcr.config import BERT_SIZE, CHECK_POINT, CHECK_NAME, CONTEXT_LENGTH, DATA_DIR, DEVICE, INDEX_PATH, SMALL_MAG
from flcr.search import load_index, search_index


def read_raw_string(in_string: str, sentence_model: SentenceTransformer):
    sentences = nltk.sent_tokenize(in_string)
    sentence_embeddings = sentence_model.encode(sentences)
    if len(sentence_embeddings) < CONTEXT_LENGTH:
        pad = np.zeros([CONTEXT_LENGTH - len(sentence_embeddings), BERT_SIZE], dtype=np.float32)
        sentence_embeddings = np.concatenate([sentence_embeddings, pad])
    if len(sentence_embeddings) > CONTEXT_LENGTH:
        sentence_embeddings = sentence_embeddings[:CONTEXT_LENGTH]
    return torch.tensor(sentence_embeddings).reshape(-1).unsqueeze(dim=0).to(DEVICE)


def recommend_batch(context: torch.Tensor, index, retrieval_model, small_mag, k=10):
    with torch.no_grad():
        context_batch = context.repeat([5, 1])
        cluster_id = torch.arange(5, device=DEVICE)
        cluster_emb = retrieval_model.clusterEmbedding(cluster_id)
        embedding = retrieval_model.context_embedding(torch.cat([context_batch, cluster_emb], dim=1)).cpu()
    _, idxes = search_index(index, embedding, k=k)
    return [[title for title in small_mag.loc[idxes[i], "paperTitle"]] for i in range(idxes.shape[0])]


small_mag = pd.read_pickle(SMALL_MAG)
retrieval_model = model.Net().to(DEVICE)
retrieval_model.load_state_dict(torch.load(CHECK_POINT, map_location=DEVICE))

index = load_index(INDEX_PATH)

sentence_model = SentenceTransformer(str(DATA_DIR / "all-mpnet-base-v2"), device=DEVICE)
example = (
    "Active learning aims to interactively select the most informative data for accelerating the "
    "learning procedure. In this paper, we proposed a novel and principled deep active learning "
    "approach under label shift."
)

print(f"Using retrieval index: {CHECK_NAME}")
for i, titles in enumerate(recommend_batch(read_raw_string(example, sentence_model), index, retrieval_model, small_mag)):
    print(f"group {i}:")
    print(titles)
