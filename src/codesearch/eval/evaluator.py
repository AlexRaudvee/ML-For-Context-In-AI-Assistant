
import argparse
import numpy as np
from typing import List
from codesearch.embeddings import EmbeddingModel
from codesearch.data.cosqa import load_cosqa
from codesearch.index.usearch_index import USearchIndex
from codesearch.metrics.ranking import recall_at_k, mrr_at_k, ndcg_at_k

def compute_ranks(query_embs: np.ndarray, code_embs: np.ndarray, k: int = 10):
    index = USearchIndex(dim=code_embs.shape[1])
    index.add(code_embs, range(len(code_embs)))
    ids, _ = index.search(query_embs, k=len(code_embs))
    # Rank position of the true label for each query:
    ranks = []
    for qid, row in enumerate(ids):
        pos = np.where(row == qid)[0]
        ranks.append(pos[0] if len(pos) else 1e9)
    return np.array(ranks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    data = load_cosqa(args.split)
    model = EmbeddingModel(args.model)
    q = model.encode(data["queries"])
    c = model.encode(data["codes"])
    ranks = compute_ranks(q, c, k=args.k)
    print(f"Recall@{args.k}: {recall_at_k(ranks, k=args.k):.4f}")
    print(f"MRR@{args.k}:    {mrr_at_k(ranks, k=args.k):.4f}")
    print(f"nDCG@{args.k}:   {ndcg_at_k(ranks, k=args.k):.4f}")

if __name__ == "__main__":
    main()
