
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from codesearch.embeddings import EmbeddingModel
from codesearch.index.usearch_index import USearchIndex

app = FastAPI(title="Code Search API")

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResponseItem(BaseModel):
    id: int
    score: float
    text: str

class SearchResponse(BaseModel):
    results: List[SearchResponseItem]

# Global state (demo only)
texts: List[str] = []
model: Optional[EmbeddingModel] = None
index: Optional[USearchIndex] = None

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    assert model is not None and index is not None, "Server not initialized"
    q = model.encode([req.query])
    ids, scores = index.search(q, req.k)

    # Normalize to 1-D lists for a single query
    if hasattr(ids, "ndim") and ids.ndim > 1:
        id_list = ids[0].tolist()
        score_list = scores[0].tolist()
    else:
        id_list = ids.tolist()
        score_list = scores.tolist()

    items = [
        SearchResponseItem(
            id=int(i),
            score=float(score_list[j]),
            text=texts[int(i)],
        )
        for j, i in enumerate(id_list)
    ]
    return SearchResponse(results=items)


def bootstrap(loaded_texts: List[str], model_name: str, index_path: str = None):
    global texts, model, index
    texts = loaded_texts
    model = EmbeddingModel(model_name)
    index = USearchIndex(dim=model.dim)
    embs = model.encode(texts)
    index.add(embs, range(len(texts)))
    if index_path:
        index.save(index_path)
