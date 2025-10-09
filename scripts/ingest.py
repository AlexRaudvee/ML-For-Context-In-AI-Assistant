import argparse, glob, os, json
from pathlib import Path
from typing import List, Dict

from codesearch.embeddings import EmbeddingModel
from codesearch.index.usearch_index import USearchIndex
from codesearch.index.qdrant_index import QdrantIndex


def read_files(glob_pattern: str) -> List[Dict]:
    """Return a list of dicts: {text, path, lang}"""
    paths = glob.glob(glob_pattern, recursive=True)
    docs = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
            ext = Path(p).suffix.lower()
            lang = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".java": "java", ".kt": "kotlin", ".cpp": "cpp", ".c": "c",
                ".rs": "rust", ".go": "go"
            }.get(ext, "text")
            docs.append({"text": text, "path": str(p), "lang": lang})
        except Exception:
            pass
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["usearch", "qdrant"], default="usearch")

    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")

    # USEARCH save path
    ap.add_argument("--index_path", default="models/usearch.index")
    ap.add_argument("--texts_json", default=None)

    # QDRANT params
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--qdrant-https", action="store_true")
    ap.add_argument("--qdrant-collection", default="codes")
    ap.add_argument("--qdrant-recreate", action="store_true")

    args = ap.parse_args()

    docs = read_files(args.input_glob)
    print(f"Loaded {len(docs)} files")

    model = EmbeddingModel(args.model)
    texts = [d["text"] for d in docs]
    embs = model.encode(texts)

    if args.backend == "usearch":
        idx = USearchIndex(dim=model.dim)
        idx.add(embs, range(len(texts)))
        if args.index_path:
            os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
            idx.save(args.index_path)

        # sidecar JSON to let the demo API return snippets
        if args.texts_json:
            Path(args.texts_json).write_text(json.dumps(texts), encoding="utf-8")
        else:
            Path(Path(args.index_path).with_suffix(".json")).write_text(
                json.dumps(texts), encoding="utf-8"
            )
        print("Ingested into USEARCH.")
        return

    # QDRANT path
    qidx = QdrantIndex(
        dim=model.dim,
        host=args.qdrant_host,
        port=args.qdrant_port,
        https=args.qdrant_https,
        collection=args.qdrant_collection,
        recreate=bool(args.qdrant_recreate),
    )
    payloads = [{"text": d["text"], "path": d["path"], "lang": d["lang"]} for d in docs]
    qidx.add(embs, ids=range(len(texts)), payloads=payloads)
    print(f"Ingested {len(docs)} points into Qdrant collection '{args.qdrant_collection}'.")
    # No local files needed; Qdrant persists data.


if __name__ == "__main__":
    main()
