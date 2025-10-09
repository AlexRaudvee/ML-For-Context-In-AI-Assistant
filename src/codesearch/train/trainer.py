
import argparse, math, os, json
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel

from codesearch.data.cosqa import load_cosqa
from codesearch.train.losses import MultipleNegativesRankingLoss, InfoNCELoss

@dataclass
class PairDataset(Dataset):
    queries: List[str]
    codes: List[str]
    tokenizer
    max_len: int = 128

    def __len__(self): return len(self.queries)

    def __getitem__(self, idx):
        q = self.tokenizer(self.queries[idx], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        c = self.tokenizer(self.codes[idx], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in q.items()}, {k: v.squeeze(0) for k, v in c.items()}

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--loss", choices=["mnr", "info_nce"], default="mnr")
    ap.add_argument("--save_dir", default="models/finetuned")
    args = ap.parse_args()

    data = load_cosqa("train")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    ds = PairDataset(data["queries"], data["codes"], tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = MultipleNegativesRankingLoss() if args.loss == "mnr" else InfoNCELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    losses = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for q_batch, c_batch in dl:
            q_batch = {k: v.to(device) for k, v in q_batch.items()}
            c_batch = {k: v.to(device) for k, v in c_batch.items()}

            q_out = model(**q_batch)
            c_out = model(**c_batch)
            zq = mean_pool(q_out.last_hidden_state, q_batch["attention_mask"])
            zc = mean_pool(c_out.last_hidden_state, c_batch["attention_mask"])

            loss = loss_fn(zq, zc)
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            step += 1
            if step % 10 == 0:
                print(f"step {step}: loss={losses[-1]:.4f}")

    model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)
    with open(os.path.join(args.save_dir, "train_loss.json"), "w") as f:
        json.dump(losses, f)

if __name__ == "__main__":
    main()
