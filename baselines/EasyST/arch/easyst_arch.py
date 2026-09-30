"""BasicTS adaptation of the lightweight EasyST student architecture."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class EasyST(nn.Module):
    """Lightweight MLP with spatial and temporal prompts.

    EasyST's deployable student is intentionally graph-free: a history
    embedding is enriched with a learned node prompt and the observed calendar
    context, then processed by residual MLP blocks.  A feature gate provides a
    compact deterministic information bottleneck suitable for ordinary
    supervised BasicTS training (the original teacher-distillation pipeline is
    not required at inference time).
    """

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 256,
        d_ff: int = 512,
        node_dim: int = 64,
        time_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.history_projection = nn.Linear(seq_len, d_model)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        self.time_projection = nn.Sequential(
            nn.Linear(4, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        self.prompt_fusion = nn.Linear(d_model + node_dim + time_dim, d_model)
        self.blocks = nn.ModuleList(
            ResidualMLP(d_model, d_ff, dropout) for _ in range(num_layers)
        )
        self.bottleneck_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.head = nn.Linear(d_model, pred_len)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        del future_data, batch_seen, epoch, train, kwargs
        if history_data.shape[-1] < 5:
            raise ValueError("EasyST expects target plus four calendar features")

        target = history_data[..., 0].permute(0, 2, 1)  # [B, N, L]
        history = self.history_projection(target)
        batch_size = history.shape[0]
        node_prompt = self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        time_prompt = self.time_projection(history_data[:, -1, :, 1:5])
        hidden = self.prompt_fusion(torch.cat((history, node_prompt, time_prompt), dim=-1))
        for block in self.blocks:
            hidden = block(hidden)

        hidden = hidden * torch.sigmoid(self.bottleneck_gate(hidden))
        prediction = self.head(hidden).permute(0, 2, 1)
        return prediction.unsqueeze(-1)
