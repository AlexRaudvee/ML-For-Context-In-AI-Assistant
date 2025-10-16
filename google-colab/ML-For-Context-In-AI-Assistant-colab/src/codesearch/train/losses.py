from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer, InputExample, losses

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # Normalize
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        logits = (z_a @ z_b.t()) / self.tau
        labels = torch.arange(z_a.size(0), device=z_a.device)
        return F.cross_entropy(logits, labels)

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_query: torch.Tensor, z_code: torch.Tensor) -> torch.Tensor:
        # Equivalent to InfoNCE with tau=1 for bi-encoder pairs
        z_query = F.normalize(z_query, dim=-1)
        z_code = F.normalize(z_code, dim=-1)
        scores = z_query @ z_code.t()
        labels = torch.arange(z_query.size(0), device=z_query.device)
        return F.cross_entropy(scores, labels)

class LossCurveLogger:
    """Callable that logs training loss on a fixed probe mini-batch (no SentenceEvaluator needed)."""
    def __init__(self, model: SentenceTransformer, loss: losses.MultipleNegativesRankingLoss, examples: List[InputExample], batch_size: int = 64):
        self.model = model
        self.loss = loss
        self.examples = examples[:batch_size]
        self.bs = min(batch_size, len(self.examples))
        self.losses: List[float] = []

    def __call__(self, *args, **kwargs):
        # Compute current MNRL on the fixed probe batch without gradients
        self.model.train(False)
        a = [ex.texts[0] for ex in self.examples]
        b = [ex.texts[1] for ex in self.examples]
        a_feats = self.model.tokenize(a)
        b_feats = self.model.tokenize(b)
        for k in a_feats: a_feats[k] = a_feats[k].to(self.model.device)
        for k in b_feats: b_feats[k] = b_feats[k].to(self.model.device)
        with torch.no_grad():
            val = self.loss([a_feats, b_feats])  # returns scalar loss
        self.losses.append(float(val.detach().cpu().item()))
        

class RecordingMNRLoss(losses.MultipleNegativesRankingLoss):
    """
    Wraps MultipleNegativesRankingLoss and records the training loss each time
    forward() is called while the model is in training mode.
    """
    def __init__(self, model: SentenceTransformer, log_every: int = 1, ema: float | None = None):
        super().__init__(model)
        self.train_losses: list[float] = []
        self.train_losses_ema: list[float] = []
        self._step = 0
        self.log_every = max(1, int(log_every))
        self._ema = ema  # e.g., 0.98 for a smooth curve

    def forward(self, sentence_features, labels=None):
        loss = super().forward(sentence_features, labels)
        # Only record during training steps (fit() sets model.train(True))
        if self.model.training:
            self._step += 1
            if self._step % self.log_every == 0:
                v = float(loss.detach().cpu().item())
                self.train_losses.append(v)
                if self._ema is not None:
                    if not self.train_losses_ema:
                        self.train_losses_ema.append(v)
                    else:
                        self.train_losses_ema.append(self._ema * self.train_losses_ema[-1] + (1 - self._ema) * v)
        return loss

