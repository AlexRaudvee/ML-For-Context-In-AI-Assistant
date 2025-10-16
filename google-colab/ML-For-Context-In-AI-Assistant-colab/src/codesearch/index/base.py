
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple
import numpy as np

class VectorIndex(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: Iterable[int]): ...

    @abstractmethod
    def finalize(self): ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) arrays of shape (len(queries), k)."""
        ...

    @abstractmethod
    def save(self, path: str): ...

    @abstractmethod
    def load(self, path: str): ...
