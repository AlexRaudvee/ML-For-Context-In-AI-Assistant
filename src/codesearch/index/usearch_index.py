# src/codesearch/index/usearch_index.py
from typing import Iterable, Tuple
import numpy as np
from .base import VectorIndex

class USearchIndex(VectorIndex):
    def __init__(self, dim: int, metric: str = "cos", ef_search: int = 64):
        """
        metric: one of {"cos", "l2sq", "ip"} for USearch.
        """
        from usearch.index import Index
        # Normalize metric aliases
        m = metric.lower()
        if m in {"cosine", "cos"}:
            m = "cos"
        elif m in {"l2", "l2sq"}:
            m = "l2sq"
        elif m in {"dot", "ip", "inner"}:
            m = "ip"
        else:
            raise ValueError(f"Unsupported USearch metric: {metric}")

        self.metric = m
        self.index = Index(metric=self.metric, ndim=dim)
        # ef_search exists on HNSW-like indexes; keep for parity
        self.ef_search = ef_search

    def add(self, vectors: np.ndarray, ids: Iterable[int]):
        # USearch expects float32 & contiguous
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        keys = np.ascontiguousarray(np.fromiter(ids, dtype=np.int64, count=vecs.shape[0]))
        # API: add(keys, vectors) — positional, not keywords
        self.index.add(keys, vecs)

    def finalize(self):
        # USearch builds incrementally; nothing to do
        pass

    def search(self, queries: np.ndarray, k: int):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        # Some versions return a tuple (keys, distances),
        # others an object with .keys / .distances
        res = self.index.search(q, k, exact=False)
        if isinstance(res, tuple) and len(res) == 2:
            keys, dists = res
        else:
            keys, dists = res.keys, res.distances
        return keys, dists

    def save(self, path: str):
        self.index.save(path)

    def load(self, path: str):
        from usearch.index import Index
        self.index = Index.restore(path)
