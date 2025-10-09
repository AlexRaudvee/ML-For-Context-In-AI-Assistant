from typing import Iterable, List, Optional, Tuple, Union
import numpy as np

from .base import VectorIndex

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


class QdrantIndex(VectorIndex):
    """
    Minimal Qdrant adapter that implements the VectorIndex interface.
    Assumes a single vector space per collection.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        host: str = "localhost",
        port: int = 6333,
        https: bool = False,
        collection: str = "codes",
        recreate: bool = False,
    ):
        self.dim = dim
        self.collection = collection
        self.client = QdrantClient(host=host, port=port, https=https)

        dist = Distance.COSINE if metric.lower() == "cosine" else Distance.EUCLID

        if recreate:
            # Drop & create fresh (useful for local experiments)
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=dist),
            )
        else:
            # Create if not exists
            if self.collection not in [c.name for c in self.client.get_collections().collections]:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=dist),
                )

    # --- VectorIndex API -----------------------------------------------------

    def add(
        self,
        vectors: np.ndarray,
        ids: Iterable[int],
        payloads: Optional[List[dict]] = None,
    ):
        """
        Upsert vectors with optional payloads.
        """
        ids = list(map(int, ids))
        if payloads is None:
            payloads = [{} for _ in ids]

        points = [
            PointStruct(id=ids[i], vector=vectors[i].tolist(), payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def finalize(self):
        # Qdrant is on-line; no finalize step needed.
        pass

    def search(
        self,
        queries: np.ndarray,
        k: int,
        payload: bool = True,
        filter_: Optional[Filter] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[List[dict]]]]:
        """
        Returns (ids, scores[, payloads]) for each query.
        Qdrant `score` is similarity (higher is better) for cosine.
        """
        ids_out: List[List[int]] = []
        scores_out: List[List[float]] = []
        payloads_out: List[List[dict]] = []

        for q in queries:
            res = self.client.search(
                collection_name=self.collection,
                query_vector=q.tolist(),
                limit=k,
                with_payload=payload,
                query_filter=filter_,
            )
            ids_out.append([p.id for p in res])
            scores_out.append([float(p.score) for p in res])
            if payload:
                payloads_out.append([p.payload or {} for p in res])

        ids_arr = np.array(ids_out, dtype=np.int64)
        scores_arr = np.array(scores_out, dtype=np.float32)

        if payload:
            return ids_arr, scores_arr, payloads_out
        return ids_arr, scores_arr, None

    def save(self, path: str):
        # Persistence is handled by Qdrant itself.
        pass

    def load(self, path: str):
        # Not applicable for Qdrant.
        pass

    # Convenience filter builder
    @staticmethod
    def make_filter(field: str, value: Union[str, int, float, bool]) -> Filter:
        return Filter(
            must=[FieldCondition(key=field, match=MatchValue(value=value))]
        )
