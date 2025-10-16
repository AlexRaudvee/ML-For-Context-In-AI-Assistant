
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=normalize)
        return emb
