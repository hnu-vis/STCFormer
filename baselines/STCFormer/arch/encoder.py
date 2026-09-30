"""STCFormer encoder with temporal self-attention state representations."""

from typing import List, NamedTuple

import torch
import torch.nn as nn

from .cgab import ClusterGuidedAttentionBlock
from .event_mixer import TemporalSelfAttentionEventMixer
from .info_loss import patch_cluster_info_loss


class EncoderOutput(NamedTuple):
    features: torch.Tensor
    soft_assignments: List[torch.Tensor]
    hard_assignments: List[torch.Tensor]
    info_loss: torch.Tensor


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        num_clusters: int,
        dropout: float,
        em_steps: int,
        cluster_temperature: float,
        cluster_balance: float,
        gumbel_tau: float,
        centroid_blend: float = 0.95,
        local_attention_impl: str = 'auto',
        sdpa_backend: str = 'auto',
    ):
        super().__init__()
        self.event_mixer = TemporalSelfAttentionEventMixer(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.cgab = ClusterGuidedAttentionBlock(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            num_clusters=num_clusters,
            dropout=dropout,
            em_steps=em_steps,
            temperature=cluster_temperature,
            balance_strength=cluster_balance,
            gumbel_tau=gumbel_tau,
            centroid_blend=centroid_blend,
            local_attention_impl=local_attention_impl,
            sdpa_backend=sdpa_backend,
        )

    def forward(self, x: torch.Tensor):
        event_output = self.event_mixer(x)
        return self.cgab(
            attention_features=event_output.attention_features,
            clustering_features=event_output.clustering_features,
        )


class STCFormerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        centroids_num,
        layers: int,
        dropout: float,
        em_steps: int,
        cluster_temperature: float,
        cluster_balance: float,
        gumbel_tau: float,
        mi_weight: float,
        alpha: float,
        centroid_blend: float = 0.95,
        local_attention_impl: str = 'auto',
        sdpa_backend: str = 'auto',
    ):
        super().__init__()
        if len(centroids_num) < layers:
            raise ValueError("centroids_num must provide one value per encoder layer")
        self.blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                num_clusters=centroids_num[layer_index],
                dropout=dropout,
                em_steps=em_steps,
                cluster_temperature=cluster_temperature,
                cluster_balance=cluster_balance,
                gumbel_tau=gumbel_tau,
                centroid_blend=centroid_blend,
                local_attention_impl=local_attention_impl,
                sdpa_backend=sdpa_backend,
            )
            for layer_index in range(layers)
        ])
        self.layer_weights = nn.Parameter(torch.zeros(layers))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        layer_features = []
        soft_assignments = []
        hard_assignments = []
        losses = []
        for block in self.blocks:
            output = block(x)
            x = output.features
            layer_features.append(x)
            soft_assignments.append(output.soft_assignments)
            hard_assignments.append(output.hard_assignments)
            losses.append(patch_cluster_info_loss(output.soft_assignments))
        weights = torch.softmax(self.layer_weights, dim=0)
        features = sum(
            weight * layer_feature
            for weight, layer_feature in zip(weights, layer_features)
        )
        return EncoderOutput(
            features=self.output_norm(features),
            soft_assignments=soft_assignments,
            hard_assignments=hard_assignments,
            info_loss=torch.stack(losses).mean(),
        )
