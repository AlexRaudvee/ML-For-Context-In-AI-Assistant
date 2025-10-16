import argparse
import uvicorn

from codesearch.api.main import app, bootstrap_qdrant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["usearch", "qdrant"], default="usearch")

    # Common
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)

    # USEARCH-only
    ap.add_argument("--index_path", help="ignored in this demo; we rebuild in-memory", default=None)
    ap.add_argument("--texts_json", help="path to JSON with texts (from ingest step)", default=None)

    # QDRANT-only
    ap.add_argument("--qdrant-host", default="localhost")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--qdrant-https", action="store_true")
    ap.add_argument("--qdrant-collection", default="codes")

    args = ap.parse_args()


    if args.backend == "qdrant":
        bootstrap_qdrant(
            model_name=args.model,
            host=args.qdrant_host,
            port=args.qdrant_port,
            https=args.qdrant_https,
            collection=args.qdrant_collection,
        )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
