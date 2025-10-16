from __future__ import annotations
from typing import Sequence

import numpy as np


def recall_at_k_binary(retrieved_ids: Sequence[int], relevant_ids: Sequence[int], k: int = 10) -> float:
    return 1.0 if any(rid in set(relevant_ids) for rid in retrieved_ids[:k]) else 0.0

def mrr_at_k_binary(retrieved_ids: Sequence[int], relevant_ids: Sequence[int], k: int = 10) -> float:
    rel = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in rel:
            return 1.0 / rank
    return 0.0

def _dcg_at_k(rels: Sequence[float], k: int) -> float:
    rels = np.asarray(rels[:k], dtype=float)
    if rels.size == 0: return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))

def ndcg_at_k_binary(retrieved_ids: Sequence[int], relevant_ids: Sequence[int], k: int = 10) -> float:
    rel = set(relevant_ids)
    rels = [1.0 if rid in rel else 0.0 for rid in retrieved_ids]
    dcg = _dcg_at_k(rels, k)
    ideal = _dcg_at_k(sorted(rels, reverse=True), k)
    return float(dcg / ideal) if ideal > 0 else 0.0