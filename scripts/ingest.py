import argparse, glob
from typing import List, Dict

from codesearch.embeddings import EmbeddingModel
from codesearch.index.qdrant_index import QdrantIndex
from codesearch.utils.chunkers import chunk_file_auto

def read_and_chunk(
    glob_pattern: str,
    max_func_lines: int,
    window: int,
    overlap: int,
    include_comments: bool,
    min_chars: int,
) -> List[Dict]:
    """Return list of chunk dicts with rich metadata."""
    paths = glob.glob(glob_pattern, recursive=True)
    chunks: List[Dict] = []
    for p in paths:
        try:
            chunks.extend(
                chunk_file_auto(
                    path=p,
                    max_func_lines=max_func_lines,
                    window=window,
                    overlap=overlap,
                    include_comments=include_comments,
                    min_chars=min_chars,
                )
            )
        except Exception:
            # Ignore unreadable files
            continue
    return chunks

def batched(iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")

    # Chunking params
    ap.add_argument("--max-func-lines", type=int, default=200, help="Max lines before windowing a function/class")
    ap.add_argument("--win-lines", type=int, default=120, help="Window size in lines")
    ap.add_argument("--overlap-lines", type=int, default=30, help="Overlap in lines between windows")
    ap.add_argument("--min-chars", type=int, default=50, help="Drop tiny chunks below this length")
    ap.add_argument("--include-comments", action="store_true", help="Include leading comment block above defs")

    # QDRANT params
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--qdrant-https", action="store_true")
    ap.add_argument("--qdrant-collection", default="codes")
    ap.add_argument("--qdrant-recreate", action="store_true")

    # Performance
    ap.add_argument("--batch-size", type=int, default=128)

    args = ap.parse_args()

    chunks = read_and_chunk(
        args.input_glob,
        max_func_lines=args.max_func_lines,
        window=args.win_lines,
        overlap=args.overlap_lines,
        include_comments=args.include_comments,
        min_chars=args.min_chars,
    )
    print(f"Prepared {len(chunks)} chunks")

    if not chunks:
        print("No chunks found; nothing to ingest.")
        return

    model = EmbeddingModel(args.model)

    qidx = QdrantIndex(
        dim=model.dim,
        host=args.qdrant_host,
        port=args.qdrant_port,
        https=args.qdrant_https,
        collection=args.qdrant_collection,
        recreate=bool(args.qdrant_recreate),
    )

    # Assign stable integer IDs (0..N-1) for this ingest
    next_id = 0
    total = 0
    for batch in batched(chunks, args.batch_size):
        texts = [c["text"] for c in batch]
        embs = model.encode(texts)
        ids = list(range(next_id, next_id + len(batch)))
        payloads = []
        for c in batch:
            payloads.append({
                "text": c["text"],
                "path": c["path"],
                "lang": c["lang"],
                "kind": c.get("kind"),
                "symbol": c.get("symbol"),
                "start_line": c.get("start_line"),
                "end_line": c.get("end_line"),
                "part": c.get("part"),
                "n_parts": c.get("n_parts"),
            })
        qidx.add(embs, ids=ids, payloads=payloads)
        next_id += len(batch)
        total += len(batch)

    print(f"Ingested {total} chunks into Qdrant collection '{args.qdrant_collection}'.")
    print("Tip: you can filter by payload fields, e.g., lang=='python' or kind in ['function','class'].")

if __name__ == "__main__":
    main()
