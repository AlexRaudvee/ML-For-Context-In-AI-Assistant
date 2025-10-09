
# 🧭 Embeddings-based Code Search (CoSQA)

A minimal-but-extendable code search engine with:
- **Part 1**: Embeddings-based search + REST API (FastAPI)
- **Part 2**: Evaluation on **CoSQA** with Recall@10, MRR@10, nDCG@10
- **Part 3**: Fine-tuning on CoSQA to improve metrics (contrastive bi-encoder)
- **Bonus**: Function name vs body search, index hyperparam sweeps


---

## 🔧 Quickstart

```bash
# 1) Create env and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Ingest a small toy corpus (or your own code base)
python scripts/ingest.py --input_glob "data/examples/**/*.py" --index_path models/usearch.index

# 3) Serve the search API
python scripts/serve_api.py --index_path models/usearch.index --model sentence-transformers/all-MiniLM-L6-v2

# 4) Query
curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d '{"query":"read a csv file in pandas", "k": 5}'
```

---

## 🗂️ Repository structure

```
.
├── README.md                  # how to run each part, milestones
├── requirements.txt           # deps: sentence-transformers, usearch, etc.
├── configs/
│   └── default.yaml           # model/index/train params
├── data/                      # raw datasets (gitignored)
├── models/                    # saved indexes & finetuned models (gitignored)
├── results/                   # eval outputs & plots (gitignored)
├── notebooks/
│   └── report.ipynb           # final report notebook (fill as you go)
├── scripts/                   # CLI entry points (no business logic)
│   ├── ingest.py              # build an index from a code corpus
│   ├── serve_api.py           # run FastAPI server
│   ├── evaluate.py            # run eval (CoSQA) + print metrics
│   └── train_biencoder.py     # finetune bi-encoder on CoSQA
└── src/codesearch/
    ├── api/
    │   └── main.py            # FastAPI app + /search endpoint
    ├── data/
    │   └── cosqa.py           # CoSQA loading/prep
    ├── eval/
    │   └── evaluator.py       # evaluation driver
    ├── index/
    │   ├── base.py            # VectorIndex interface
    │   └── usearch_index.py   # USearch backend (add/search/save/load)
    ├── metrics/
    │   └── ranking.py         # Recall@10, MRR@10, nDCG@10
    ├── train/
    │   ├── losses.py          # MultipleNegatives & InfoNCE losses
    │   └── trainer.py         # simple finetuning loop (PyTorch)
    ├── utils/
    │   └── logging.py
    └── embeddings.py          # SBERT wrapper (encode texts)
```

<!-- --- -->

<!-- ## ✅ Milestones & what to implement -->

<!-- ### Part 1 — Search Engine
- [ ] `codesearch/embeddings.py`: wrap a HuggingFace/SBERT model (encode texts)
- [ ] `codesearch/index/base.py`: `VectorIndex` interface (add, finalize, search, save/load)
- [ ] `codesearch/index/usearch_index.py`: minimal in-memory ANN via `usearch`
- [ ] `scripts/ingest.py`: read a corpus (e.g., .py files), chunk to passages, build index
- [ ] `src/codesearch/api/main.py` + `scripts/serve_api.py`: FastAPI `/search`

### Part 2 — Evaluation
- [ ] `codesearch/data/cosqa.py`: download/prepare CoSQA pairs
- [ ] `codesearch/metrics/ranking.py`: Recall@10, MRR@10, nDCG@10
- [ ] `codesearch/eval/evaluator.py` & `scripts/evaluate.py`: compute metrics

### Part 3 — Fine-tuning
- [ ] `codesearch/train/losses.py`: Contrastive/MultipleNegativesRankingLoss or InfoNCE
- [ ] `codesearch/train/trainer.py` & `scripts/train_biencoder.py`: train text↔code bi-encoder
- [ ] Log training loss; save plots to `results/`

### Bonus
- [ ] Function names vs full bodies: a flag in dataset prep/eval
- [ ] Index hyperparam sweeps: `ef`, `metric`, `quantization`, `connectivity` -->

---

## 📓 Report Notebook

In `notebooks/report.ipynb`:
- shown some sample queries & results
- computed metrics (baseline vs fine-tuned)
- includes loss curve plot
- briefly justifies **loss choice** and shows **hyperparam impacts**

---

## 📜 License

MIT License
