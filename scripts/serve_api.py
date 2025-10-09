
import argparse, json
import uvicorn
from pathlib import Path
from fastapi import FastAPI

from codesearch.api.main import app, bootstrap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    texts = []
    meta = Path(args.index_path).with_suffix(".json")
    if meta.exists():
        texts = json.loads(meta.read_text(encoding="utf-8"))
    bootstrap(texts, args.model, index_path=None)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
