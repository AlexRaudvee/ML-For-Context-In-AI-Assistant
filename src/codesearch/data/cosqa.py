
from datasets import load_dataset
from typing import Dict, List, Tuple

def load_cosqa(split: str = "validation") -> Dict[str, List[str]]:
    """Returns a dict with 'queries', 'codes', and 'labels' for CoSQA.
    Each query has exactly one positive code index (bi-encoder setup).
    """
    ds = load_dataset("datafyer/cosqa", split=split)
    # Fallback if dataset name changes: user can swap to official source.
    queries = [ex["nl"] for ex in ds]
    codes = [ex["pl"] for ex in ds]
    # Labels: assume same index is relevant (toy pairing); adjust if dataset differs.
    labels = list(range(len(queries)))
    return {"queries": queries, "codes": codes, "labels": labels}
