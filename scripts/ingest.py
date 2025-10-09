
import argparse, glob, os, re, json
from pathlib import Path
from typing import List

from codesearch.embeddings import EmbeddingModel
from codesearch.index.usearch_index import USearchIndex

def read_files(glob_pattern: str) -> List[str]:
    paths = glob.glob(glob_pattern, recursive=True)
    texts = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        except Exception:
            pass
    return texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--index_path", default="models/usearch.index")
    args = ap.parse_args()

    texts = read_files(args.input_glob)
    print(f"Loaded {len(texts)} files")
    model = EmbeddingModel(args.model)
    embs = model.encode(texts)
    idx = USearchIndex(dim=model.dim)
    idx.add(embs, range(len(texts)))
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    idx.save(args.index_path)

    # Save texts alongside index for demo API
    with open(Path(args.index_path).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(texts, f)

if __name__ == "__main__":
    main()
