"""Dual-representation Cluster-Guided Attention Block for STCFormer."""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packed_attention import packed_cluster_attention, masked_cluster_attention, fused_sdpa


class CGABOutput(NamedTuple):
    features: torch.Tensor
    soft_assignments: torch.Tensor
    hard_assignments: torch.Tensor


class MultiHeadAttention(nn.Module):
    """Batch-first attention with an optional differentiable binary gate."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.sdpa_backend = None

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_gate: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, query_length, d_model = query.shape
        key_length = key_value.shape[1]
        q = self.query(query).reshape(
            batch_size, query_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(key_value).reshape(
            batch_size, key_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(key_value).reshape(
            batch_size, key_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        if attention_gate is None and self.sdpa_backend is not None:
            output = fused_sdpa(q, k, v, dropout_p=self.dropout.p if self.training else 0.0,
                                scale=self.scale, backend=self.sdpa_backend)
            output = output.transpose(1, 2).contiguous().reshape(batch_size, query_length, d_model)
            return self.output(output)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_gate is not None:
            gate = attention_gate[:, None].to(scores.dtype)
            allowed = gate.detach() > 0.5
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if attention_gate is not None:
            weights = weights * gate
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = self.dropout(weights)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return self.output(output.reshape(batch_size, query_length, d_model))


class ClusterGuidedAttentionBlock(nn.Module):
    """Cluster evolution features, then model local and centroid-level context.

    Soft-EM receives only ``clustering_features``. Local and global attention
    receive only ``attention_features``. Crucially, global centroids are freshly
    aggregated from post-local features with the soft assignment matrix; the
    internal Soft-EM prototypes/centroids are never reused as attention values.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        num_clusters: int,
        dropout: float = 0.1,
        em_steps: int = 1,
        temperature: float = 0.35,
        balance_strength: float = 0.0,
        gumbel_tau: float = 1.0,
        centroid_blend: float = 0.95,
        local_attention_impl: str = 'auto',
        sdpa_backend: str = 'auto',
    ):
        super().__init__()
        self.num_clusters = int(num_clusters)
        if local_attention_impl not in ('dense_masked', 'packed_sdpa', 'masked_sdpa', 'auto'):
            raise ValueError('Unknown local_attention_impl')
        self.local_attention_impl = local_attention_impl
        self.sdpa_backend = sdpa_backend
        # Legacy arguments stay loadable but cannot change the fixed E-M-E.
        # Cluster balance is now encouraged solely by the information loss.
        self.em_steps = 1
        self.balance_strength = 0.0
        if not 0.0 <= centroid_blend <= 1.0:
            raise ValueError("centroid_blend must be in [0, 1]")
        self.centroid_blend = float(centroid_blend)
        if gumbel_tau <= 0:
            raise ValueError("gumbel_tau must be positive")
        self.gumbel_tau = float(gumbel_tau)
        raw_temperature = math.log(math.expm1(max(float(temperature) - 0.05, 1e-4)))
        self.raw_temperature = nn.Parameter(torch.tensor(raw_temperature))
        self.prototypes = nn.Parameter(torch.empty(self.num_clusters, d_model))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        self.cluster_norm = nn.LayerNorm(d_model)
        self.local_norm = nn.LayerNorm(d_model)
        self.centroid_norm = nn.LayerNorm(d_model)
        self.global_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.local_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.global_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.local_scale = nn.Parameter(torch.full((d_model,), 0.1))
        self.global_scale = nn.Parameter(torch.full((d_model,), 0.1))
        self.output_scale = nn.Parameter(torch.full((d_model,), 0.1))

    def _assignments(
        self, tokens: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        temperature = F.softplus(self.raw_temperature) + 0.05
        # Dot-product E-step from the specified equations; no extra cosine
        # normalization or occupancy correction is applied to these logits.
        logits = torch.matmul(tokens, centroids.transpose(-2, -1)) / temperature
        return torch.softmax(logits, dim=-1)

    def _soft_em_assignments(self, tokens: torch.Tensor) -> torch.Tensor:
        """Exactly E-M-E, independently for each sample and temporal patch.

        Prototype blending is not a persistent EMA: centroids are not carried
        across samples, minibatches, or time patches.
        """
        prototypes = self.prototypes.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        provisional = self._assignments(tokens, prototypes)
        counts = provisional.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        updated = torch.einsum("mnk,mnd->mkd", provisional, tokens) / counts
        centroids = self.centroid_blend * updated + (1.0 - self.centroid_blend) * prototypes
        return self._assignments(tokens, centroids)

    def _hard_assignments(self, probabilities: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.gumbel_softmax(
                probabilities.clamp_min(1e-8).log(),
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            )
        cluster_ids = probabilities.argmax(dim=-1)
        return F.one_hot(cluster_ids, num_classes=self.num_clusters).to(
            probabilities.dtype
        )

    @staticmethod
    def _aggregate_post_local_centroids(
        probabilities: torch.Tensor, local_features: torch.Tensor
    ) -> torch.Tensor:
        counts = probabilities.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        return torch.einsum(
            "mnk,mnd->mkd", probabilities, local_features
        ) / counts

    def forward(
        self,
        attention_features: torch.Tensor,
        clustering_features: torch.Tensor,
    ) -> CGABOutput:
        if attention_features.shape != clustering_features.shape:
            raise ValueError("attention and clustering feature shapes must match")
        batch_size, num_stations, num_time_patches, d_model = attention_features.shape
        attention_tokens = attention_features.permute(0, 2, 1, 3).reshape(
            batch_size * num_time_patches, num_stations, d_model
        )
        clustering_tokens = clustering_features.permute(0, 2, 1, 3).reshape(
            batch_size * num_time_patches, num_stations, d_model
        )

        probabilities = self._soft_em_assignments(self.cluster_norm(clustering_tokens))
        hard_assignments = self._hard_assignments(probabilities)
        local_tokens = self.local_norm(attention_tokens)
        implementation = self.local_attention_impl
        if implementation == 'auto':
            # Real French/Hunan efficiency checks: packing is worthwhile on
            # longer spatial sequences, not necessarily on very short ones.
            # Explicit Flash must route to the mask-free packed kernel.
            implementation = ('packed_sdpa' if self.sdpa_backend == 'flash'
                              or num_stations > 128 else 'dense_masked')
        if implementation == 'packed_sdpa':
            local_context = packed_cluster_attention(
                self.local_attention, local_tokens, hard_assignments, self.sdpa_backend)
        elif implementation == 'masked_sdpa':
            local_context = masked_cluster_attention(
                self.local_attention, local_tokens, hard_assignments, self.sdpa_backend)
        else:
            local_gate = torch.matmul(hard_assignments, hard_assignments.transpose(-2, -1))
            local_context = self.local_attention(local_tokens, local_tokens, attention_gate=local_gate)
        local_features = attention_tokens + self.dropout(local_context) * self.local_scale

        attention_centroids = self._aggregate_post_local_centroids(
            probabilities, self.centroid_norm(local_features)
        )
        global_context = self.global_attention(
            self.global_norm(local_features), attention_centroids
        )
        features = local_features + self.dropout(global_context) * self.global_scale
        features = features + self.dropout(
            self.feed_forward(self.output_norm(features))
        ) * self.output_scale

        features = features.reshape(
            batch_size, num_time_patches, num_stations, d_model
        ).permute(0, 2, 1, 3).contiguous()
        assignment_shape = (
            batch_size, num_time_patches, num_stations, self.num_clusters
        )
        return CGABOutput(
            features=features,
            soft_assignments=probabilities.reshape(assignment_shape),
            hard_assignments=hard_assignments.reshape(assignment_shape),
        )
