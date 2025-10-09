
import numpy as np

def recall_at_k(ranks: np.ndarray, k: int = 10) -> float:
    return float(np.mean(ranks < k))

def mrr_at_k(ranks: np.ndarray, k: int = 10) -> float:
    mrr_vals = 1.0 / (ranks + 1)
    mrr_vals[ranks >= k] = 0.0
    return float(mrr_vals.mean())

def ndcg_at_k(ranks: np.ndarray, k: int = 10) -> float:
    # Binary relevance: DCG = 1/log2(rank+2) if rank<k else 0
    gains = np.zeros_like(ranks, dtype=float)
    mask = ranks < k
    gains[mask] = 1.0 / np.log2(ranks[mask] + 2.0)
    # Ideal DCG is always 1/log2(1+1)=1 for a single relevant doc
    return float(gains.mean())
