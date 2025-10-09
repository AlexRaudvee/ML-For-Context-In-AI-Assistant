
import torch
import torch.nn as nn
import torch.nn.functional as F

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
