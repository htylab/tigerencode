"""Adaptor modules for TigerEncode models."""

from __future__ import annotations

import torch
from torch import nn


class ProjectionAdaptor(nn.Module):
    """A projection adaptor with residual connections and normalization."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.rescale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)
        if residual.shape[-1] == x.shape[-1]:
            x = x + residual
        x = self.ln(x)
        norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / (norm + 1e-12)
        x = x * self.rescale
        return x
