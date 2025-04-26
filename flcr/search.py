from pathlib import Path

import faiss
import numpy as np
import torch

from flcr.config import HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M


def _to_faiss_array(vectors) -> np.ndarray:
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    return np.ascontiguousarray(vectors, dtype=np.float32)


def build_index(vectors) -> faiss.Index:
    matrix = _to_faiss_array(vectors)
    # HNSW gives sub-linear ANN search while keeping the same L2 metric.
    index = faiss.IndexHNSWFlat(matrix.shape[1], HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(matrix)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    index = faiss.read_index(str(path))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = HNSW_EF_SEARCH
    return index


def search_index(index: faiss.Index, queries, k: int):
    query_matrix = _to_faiss_array(queries)
    capped_k = min(k, index.ntotal)
    distances, idxes = index.search(query_matrix, capped_k)
    return distances, idxes
