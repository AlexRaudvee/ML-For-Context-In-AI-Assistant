from __future__ import annotations
import argparse
from typing import List

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

from ..utils.eval import *
from .losses import *


def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L12-v2", help="Model that is going to be used for finetunning")
    ap.add_argument("--finetune-dir", type=str, default="models")
    ap.add_argument("--checkpoint-path", type=str, default="checkpoint")
    ap.add_argument("--assets-dir", type=str, default="results/assets")
    
    ap.add_argument("--qdrant-host", type=str, default="qdrant")
    ap.add_argument("--qdrant-port", type=int, default=6333)
    ap.add_argument("--qdrant-collection", type=str, default='cosqa_test_bodies')
    ap.add_argument("--qdrant-collection-ft", type=str, default='cosqa_test_ft')
    ap.add_argument("--K", type=int, default=3, help="top K indices from qdrant")

    ap.add_argument("--batch-size", type=int, default=32, help="batch size for finetuning")
    ap.add_argument("--epochs", type=int, default=10, help="number of epochs the model is going to be trained")
    ap.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 means no limit (full train)")    
    ap.add_argument("--seed", type=int, default=69, help="seed for consistency of results replication")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    
    fix_seed(42)

    corpus  = load_dataset("CoIR-Retrieval/cosqa", name="corpus")["corpus"]
    queries = load_dataset("CoIR-Retrieval/cosqa", name="queries")["queries"]

    corpus_train  = [r for r in corpus  if str(r.get("partition","")).lower() == "train"]
    queries_train = [r for r in queries if str(r.get("partition","")).lower() == "train"]

    corpus_ids_train   = [str(r.get("_id", r.get("id"))) for r in corpus_train]
    corpus_texts_train = [pick_text(r) for r in corpus_train]
    queries_ids_train  = [str(r.get("_id", r.get("id"))) for r in queries_train]
    queries_texts_train= [pick_text(r) for r in queries_train]
    
    print(f"[Data] Train: {len(queries_train)} q / {len(corpus_train)} docs")

    print("\n[Train] Fine-tuning with MultipleNegativesRankingLoss (in-batch negatives)…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)

    # Build (query, positive code) pairs
    pairs: List[InputExample] = []
    cidx = {sid:i for i, sid in enumerate(corpus_ids_train)}
    for qid, qtxt in zip(queries_ids_train, queries_texts_train):
        tgt = f"d{suffix_id(qid)}"
        j = cidx.get(tgt, None)
        if j is not None:
            if args.debug:
                print(f"[Info] data in pairs {[qtxt, corpus_texts_train[j]]}")
            pairs.append(InputExample(texts=[qtxt, corpus_texts_train[j]]))

    if len(pairs) == 0:
        raise RuntimeError("No train pairs constructed; check dataset contents.")

    if args.max_steps_per_epoch > 0:
        pairs = pairs[:args.max_steps_per_epoch * args.batch_size]

    if args.debug:
        pairs = pairs[:2]

    train_dl = DataLoader(pairs, shuffle=False, batch_size=args.batch_size, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Wrap the training loss to record it
    train_loss = RecordingMNRLoss(model, log_every=1, ema=0.98)

    warmup_steps = max(0, int(0.1 * len(train_dl) * args.epochs))
    model.fit(
        train_objectives=[(train_dl, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        use_amp=True,
        show_progress_bar=True,
        output_path=args.finetune_dir,
        evaluator=None,
        evaluation_steps=0,          
        checkpoint_path=args.checkpoint_path if args.checkpoint_path else None,
    )

    print(f"[Save] Finetuned model saved to: {args.finetune_dir}")

    # Plot the recorded *training* loss
    if train_loss.train_losses:
        plt.figure()
        plt.plot(train_loss.train_losses, label="train loss (per logged step)", linewidth=1)
        if train_loss.train_losses_ema:
            plt.plot(train_loss.train_losses_ema, label="EMA", linewidth=2)
        plt.title("Training Loss (MultipleNegativesRankingLoss)")
        plt.xlabel("Logged training step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        out_png = f"{args.assets_dir}/train_loss_curve.png" if getattr(args, "assets_dir", None) else "train_loss_curve.png"
        plt.savefig(out_png, bbox_inches="tight")
        print(f"[Plot] Saved training loss curve to {out_png}")
    else:
        print("[Warn] No training loss values were recorded.")


    print("\nNotes:")
    print("- Loss used: MultipleNegativesRankingLoss (MNRL): strong retrieval baseline using in-batch negatives;")
    print("  it directly optimizes cosine similarity for (query, code) pairs and scales well without hard-negative mining.")

if __name__ == "__main__":
    main()