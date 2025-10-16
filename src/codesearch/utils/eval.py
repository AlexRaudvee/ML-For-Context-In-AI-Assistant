from __future__ import annotations
import random

from typing import Dict, List, Sequence, Tuple

import torch
import numpy as np

def pick_text(record: dict) -> str:
    """
    Robustly extract text/code body for both corpus items and queries.
    Tries common field names in order; raises if none found.
    """
    for key in ("text", "content", "body", "code", "pl", "document", "passage", "snippet", "description", "question", "nl"):
        if key in record and isinstance(record[key], str) and record[key].strip():
            return record[key]
    # As a last resort, try joining 'title' + 'text'-ish fields
    candidates = []
    if "title" in record and isinstance(record["title"], str):
        candidates.append(record["title"])
    for key in ("text", "body", "content", "code"):
        if key in record and isinstance(record[key], str):
            candidates.append(record[key])
    if candidates:
        return "\n".join([c for c in candidates if c.strip()])
    raise KeyError(f"Could not locate a text/code field in record keys: {list(record.keys())[:10]}...")

def to_int_ids(ids: Sequence[str]) -> Tuple[List[int], Dict[str, int]]:
    """
    Map string IDs to stable integer IDs for Qdrant. Returns (int_ids, map).
    """
    mapping: Dict[str, int] = {}
    int_ids: List[int] = []
    for sid in ids:
        if sid not in mapping:
            mapping[sid] = len(mapping)
        int_ids.append(mapping[sid])
    return int_ids, mapping

def suffix_id(s: str) -> str:
    """
    Extract numeric suffix from 'q123' or 'd123' as '123'.
    """
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    return s[i:] if i < len(s) else s

def fix_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
def make_labels(corpus_ids_str: List[str], query_ids_str: List[str]) -> List[List[int]]:
    idx = {sid:i for i, sid in enumerate(corpus_ids_str)}
    labs = []
    miss = 0
    for qid in query_ids_str:
        target = f"d{suffix_id(qid)}"
        if target in idx: labs.append([idx[target]])
        else: labs.append([]); miss += 1
    if miss: print(f"[warn] {miss} queries had no matching target doc in current corpus view.")
    return labs