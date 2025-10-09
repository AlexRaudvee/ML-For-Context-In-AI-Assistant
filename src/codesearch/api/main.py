from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Literal
import numpy as np

from codesearch.embeddings import EmbeddingModel
from codesearch.index.usearch_index import USearchIndex
from codesearch.index.qdrant_index import QdrantIndex  # NEW


app = FastAPI(title="Code Search API")

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    # Optional simple filter example (only used by Qdrant)
    lang: Optional[str] = None

class SearchResponseItem(BaseModel):
    id: int
    score: float
    text: str
    meta: Optional[dict] = None

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]


# Global state (demo)
texts: List[str] = []
model: Optional[EmbeddingModel] = None
backend: Literal["usearch", "qdrant"] = "usearch"
uindex: Optional[USearchIndex] = None
qindex: Optional[QdrantIndex] = None


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    assert model is not None, "Server not initialized"
    q = model.encode([req.query])

    # --- USEARCH path (in-process, returns text from `texts`) ---
    if backend == "usearch":
        assert uindex is not None
        ids, scores = uindex.search(q, req.k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()
        items = [
            SearchResponseItem(
                id=i,
                score=float(scores[j]),
                text=texts[i] if i < len(texts) else "",
                meta=None,
            )
            for j, i in enumerate(ids)
        ]
        return SearchResponse(results=items)

    # --- QDRANT path (payload holds the text & metadata) ---
    assert qindex is not None
    # Optional simple filter: language == req.lang
    qf = None
    if req.lang:
        qf = QdrantIndex.make_filter("lang", req.lang)

    ids, scores, payloads = qindex.search(q, req.k, payload=True, filter_=qf)
    ids = ids[0].tolist()
    scores = scores[0].tolist()
    payloads = payloads[0] if payloads else [{} for _ in ids]

    items = [
        SearchResponseItem(
            id=i,
            score=float(scores[j]),
            text=(payloads[j].get("text") or ""),
            meta=payloads[j],
        )
        for j, i in enumerate(ids)
    ]
    return SearchResponse(results=items)


def bootstrap_usearch(loaded_texts: List[str], model_name: str):
    """Start the API with an in-process USEARCH index (old behavior)."""
    global texts, model, backend, uindex, qindex
    texts = loaded_texts
    model = EmbeddingModel(model_name)
    uindex = USearchIndex(dim=model.dim)
    qindex = None
    backend = "usearch"
    embs = model.encode(texts)
    uindex.add(embs, range(len(texts)))


def bootstrap_qdrant(model_name: str, dim_hint: Optional[int] = None, **qdrant_kwargs):
    """Start the API pointing at an existing Qdrant collection."""
    global texts, model, backend, uindex, qindex
    texts = []
    model = EmbeddingModel(model_name)
    uindex = None
    backend = "qdrant"
    # dim hint is not strictly needed; used when (re)creating collections elsewhere.
    qindex = QdrantIndex(dim=dim_hint or model.dim, recreate=False, **qdrant_kwargs)
