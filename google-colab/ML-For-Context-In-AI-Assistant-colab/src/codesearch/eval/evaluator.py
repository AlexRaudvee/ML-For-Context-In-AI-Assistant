from __future__ import annotations
import time, argparse
from typing import List

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ..utils.eval import *
from ..metrics.ranking import *

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L12-v2")
    ap.add_argument("--qdrant-host", type=str, default="qdrant")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--qdrant-collection", type=str, default='cosqa_test_bodies')
    ap.add_argument("--K", type=int, default=3, help="top K indices from qdrant")
    args = ap.parse_args()
    
    # 1) Load CoSQA splits (HuggingFace)
    # We use the CoIR-Retrieval mirror with explicit "corpus" and "queries" configurations.
    corpus_ds  = load_dataset("CoIR-Retrieval/cosqa", name="corpus")["corpus"]     # single table with partition column
    queries_ds = load_dataset("CoIR-Retrieval/cosqa", name="queries")["queries"]

    # Filter to test partition only
    corpus_test  = [r for r in corpus_ds if str(r["partition"].lower()) == "test"]
    queries_test = [r for r in queries_ds if str(r["partition"].lower()) == "test"]

    if not corpus_test or not queries_test:
        raise RuntimeError("No rows found with partition == 'test'. Check dataset content or partition field spelling.")

    # 2) Prepare corpus (function bodies) and queries
    # Assumptions per user: corpus IDs look like 'd123', query IDs like 'q123' — one positive per query by shared numeric suffix.
    corpus_ids_str  = [str(r.get("_id", r.get("id"))) for r in corpus_test]
    corpus_texts    = [pick_text(r) for r in corpus_test]

    query_ids_str   = [str(r.get("_id", r.get("id"))) for r in queries_test]
    query_texts     = [pick_text(r) for r in queries_test]

    # Map corpus string IDs -> integer IDs for Qdrant
    corpus_int_ids, id_map = to_int_ids(corpus_ids_str)

    # Build relevance labels for each query: qNNN -> dNNN
    # Convert the target doc string ID to the corresponding integer ID via id_map.
    labels_per_query: List[List[int]] = []
    missing = 0
    for qid in query_ids_str:
        suf = suffix_id(qid)     # '123'
        target_doc = f"d{suf}"   # 'd123'
        if target_doc in id_map:
            labels_per_query.append([id_map[target_doc]])
        else:
            labels_per_query.append([])  # if missing, will count as 0 in metrics
            missing += 1

    print(f"[Info] Prepared test split: {len(corpus_texts)} corpus docs, {len(query_texts)} queries. Missing labels: {missing}")

    # 3) Embed corpus and queries
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()

    t0 = time.time()
    corpus_embs = model.encode(corpus_texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    t1 = time.time()
    queries_embs = model.encode(query_texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    t2 = time.time()
    print(f"[Timing] Encoded corpus in {t1 - t0:.2f}s, queries in {t2 - t1:.2f}s with {args.model} (dim={dim}).")

    # 4) Index corpus in Qdrant
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    client.recreate_collection(
        collection_name=args.qdrant_collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Upsert in batches
    BATCH = 256
    total = 0
    for i in range(0, len(corpus_int_ids), BATCH):
        sl = slice(i, i + BATCH)
        points = [
            PointStruct(
                id=int(corpus_int_ids[j]),
                vector=corpus_embs[j].tolist(),
                payload={
                    "doc_id": corpus_ids_str[j],
                    "text": corpus_texts[j],
                    "partition": "test",
                    "kind": "body"
                },
            )
            for j in range(sl.start, min(sl.stop, len(corpus_int_ids)))
        ]
        client.upsert(collection_name=args.qdrant_collection, points=points)
        total += len(points)
    print(f"[Index] Upserted {total} vectors into Qdrant collection '{args.qdrant_collection}' at {args.qdrant_host}:{args.qdrant_port}")

    # 5) Retrieve top-K for each query and compute metrics
    recalls, mrrs, ndcgs = [], [], []
    search_t0 = time.time()
    for qi, qvec in enumerate(queries_embs):
        res = client.search(
            collection_name=args.qdrant_collection,
            query_vector=qvec.tolist(),
            limit=args.K,
            with_payload=False,
        )
        retrieved_ids = [int(p.id) for p in res]
        relevant_ids  = labels_per_query[qi]  # list (0 or 1 element here)

        recalls.append(recall_at_k_binary(retrieved_ids, relevant_ids, args.K))
        mrrs.append(mrr_at_k_binary(retrieved_ids, relevant_ids, args.K))
        ndcgs.append(ndcg_at_k_binary(retrieved_ids, relevant_ids, args.K))
    search_t1 = time.time()

    RecallK = float(np.mean(recalls)) if recalls else 0.0
    MRRK    = float(np.mean(mrrs))    if mrrs    else 0.0
    nDCGK   = float(np.mean(ndcgs))   if ndcgs   else 0.0

    print("\n=== CoSQA (test) — Qdrant Retrieval Metrics ===")
    print(f"Model: {args.model}")
    print(f"K: {args.K}")
    print(f"Recall@{args.K}: {RecallK:.4f}")
    print(f"MRR@{args.K}:    {MRRK:.4f}")
    print(f"nDCG@{args.K}:   {nDCGK:.4f}")
    print(f"(Retrieval time for {len(queries_embs)} queries: {search_t1 - search_t0:.2f}s)")

    # Optional: return a small qualitative sample
    SAMPLE = 3
    print("\n--- Sample retrieved doc_ids for first few queries ---")
    for i in range(min(SAMPLE, len(query_ids_str))):
        res = client.search(collection_name=args.qdrant_collection, query_vector=queries_embs[i].tolist(), limit=5, with_payload=True)
        hits = [(p.id, p.payload.get("doc_id")) for p in res]
        print(f"Query {query_ids_str[i]} -> hits: {hits} ; relevant: {labels_per_query[i]}")
