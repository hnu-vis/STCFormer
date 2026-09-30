"""BasicTS adapter for STELLA.

This is an interface-focused reimplementation of the public STELLA model:
temporal projection + spatial/temporal embeddings + residual MLP blocks.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)


class STELLA(nn.Module):
    """Spatial-Temporal Embedded Lightweight atmospheric forecaster."""

    def __init__(self, num_nodes, input_len, output_len, root_path, d_model=64,
                 num_layers=2, dropout=0.1, use_absolute_position=True, **kwargs):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.input_projection = nn.Linear(input_len, d_model)

        pos_path = Path(root_path) / "stations_sorted_reduced.npy"
        positions = np.load(pos_path).astype(np.float32)
        if positions.shape != (num_nodes, 3):
            raise ValueError(f"Expected positions [{num_nodes}, 3], got {positions.shape}")
        positions = (positions - positions.mean(0)) / (positions.std(0) + 1e-6)
        self.register_buffer("positions", torch.from_numpy(positions), persistent=False)
        self.spatial_embedding = (
            nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            if use_absolute_position else nn.Embedding(num_nodes, d_model)
        )
        self.use_absolute_position = use_absolute_position
        self.temporal_embeddings = nn.ModuleList([
            nn.Embedding(24, d_model), nn.Embedding(7, d_model),
            nn.Embedding(31, d_model), nn.Embedding(366, d_model),
        ])
        self.encoder = nn.Sequential(*[ResidualMLP(d_model, dropout) for _ in range(num_layers)])
        self.output_projection = nn.Linear(d_model, output_len)

    @staticmethod
    def _index(value, size):
        return torch.clamp((value * size).long(), min=0, max=size - 1)

    def forward(self, history_data, **kwargs):
        target = history_data[..., 0].transpose(1, 2)  # [B, N, L]
        hidden = self.input_projection(target)
        if self.use_absolute_position:
            hidden = hidden + self.spatial_embedding(self.positions).unsqueeze(0)
        else:
            node_ids = torch.arange(self.num_nodes, device=target.device)
            hidden = hidden + self.spatial_embedding(node_ids).unsqueeze(0)

        # The last observed timestamp is the forecast origin.
        marks = history_data[:, -1, 0, 1:5]
        sizes = (24, 7, 31, 366)
        temporal = sum(emb(self._index(marks[:, i], sizes[i]))
                       for i, emb in enumerate(self.temporal_embeddings))
        hidden = self.encoder(hidden + temporal.unsqueeze(1))
        return self.output_projection(hidden).transpose(1, 2).unsqueeze(-1)
