"""Dual-view temporal representation used by STCFormer."""

from typing import NamedTuple

import torch
import torch.nn as nn


class TemporalEventOutput(NamedTuple):
    """Separate state and evolution representations."""

    attention_features: torch.Tensor
    clustering_features: torch.Tensor


class TemporalSelfAttentionEventMixer(nn.Module):
    """Model state history with attention while preserving V3 clustering.

    The state path applies shared temporal self-attention and an FFN to every
    spatial patch independently.  The grouping path is deliberately identical
    to V3: it concatenates the normalized input with its first-order temporal
    difference and projects the descriptor back to ``d_model`` dimensions.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.attention_norm = nn.LayerNorm(d_model)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        # Keep these two modules structurally identical to STCFormerV3 so the
        # grouping representation is unchanged in the controlled comparison.
        self.cluster_norm = nn.LayerNorm(d_model)
        self.cluster_projection = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> TemporalEventOutput:
        batch_size, num_spatial_patches, num_time_patches, d_model = x.shape

        state = x.reshape(
            batch_size * num_spatial_patches, num_time_patches, d_model
        )
        normalized_state = self.attention_norm(state)
        attended_state, _ = self.temporal_attention(
            normalized_state,
            normalized_state,
            normalized_state,
            need_weights=False,
        )
        state = state + self.dropout(attended_state)
        state = state + self.dropout(self.ffn(self.ffn_norm(state)))
        attention_features = state.reshape_as(x)

        cluster_state = self.cluster_norm(x)
        temporal_difference = torch.zeros_like(cluster_state)
        temporal_difference[:, :, 1:] = (
            cluster_state[:, :, 1:] - cluster_state[:, :, :-1]
        )
        clustering_features = self.cluster_projection(
            torch.cat((cluster_state, temporal_difference), dim=-1)
        )
        return TemporalEventOutput(
            attention_features=attention_features,
            clustering_features=clustering_features,
        )
