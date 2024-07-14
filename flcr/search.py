from pathlib import Path

import faiss
import numpy as np
import torch


def _to_faiss_array(vectors) -> np.ndarray:
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    return np.ascontiguousarray(vectors, dtype=np.float32)


def build_index(vectors) -> faiss.Index:
    matrix = _to_faiss_array(vectors)
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def search_index(index: faiss.Index, queries, k: int):
    query_matrix = _to_faiss_array(queries)
    capped_k = min(k, index.ntotal)
    distances, idxes = index.search(query_matrix, capped_k)
    return distances, idxes
