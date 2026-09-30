"""STCFormer with temporal self-attention state representations."""

import os
from math import ceil
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .embedding import SpatioTemporalPatchEmbedding, serpentine_geographic_order
from .encoder import STCFormerEncoder


class DirectForecastHead(nn.Module):
    """Project all encoded time patches directly to the forecast horizon."""

    def __init__(
        self,
        num_time_patches: int,
        d_model: int,
        out_len: int,
        spatial_patch_len: int,
        dropout: float,
    ):
        super().__init__()
        input_size = num_time_patches * d_model
        self.out_len = int(out_len)
        self.spatial_patch_len = int(spatial_patch_len)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, out_len * spatial_patch_len),
        )
        nn.init.normal_(self.projection[-1].weight, std=0.01)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, num_spatial_patches, _, _ = features.shape
        prediction = self.projection(features.flatten(start_dim=2))
        prediction = prediction.reshape(
            batch_size,
            num_spatial_patches,
            self.out_len,
            self.spatial_patch_len,
        )
        return prediction.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.out_len,
            num_spatial_patches * self.spatial_patch_len,
        )


class STCFormer(nn.Module):
    """Encoder-only model with temporal attention and evolution clustering."""

    def __init__(
        self,
        num_n: int,
        in_len: int,
        out_len: int,
        time_patch_len: int = 8,
        spatial_patch_len: int = 2,
        centroids_num=(10, 10, 10),
        d_model: int = 192,
        d_ff: int = 384,
        n_heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        root_path: str = None,
        coordinate_file: str = "stations_sorted_reduced.npy",
        geographic_bands: int = 12,
        spatial_order_method: str = 'serpentine',
        spatial_order_seed: int = 2024,
        spatial_order_file: str = None,
        time_feature_dim: int = 0,
        revin_flag: bool = True,
        em_steps: int = 1,
        cluster_temperature: float = 0.35,
        cluster_balance: float = 0.0,
        gumbel_tau: float = 1.0,
        use_cluster_loss: bool = True,
        cluster_loss_weight: float = 0.1,
        mi_weight: float = 0.5,
        alpha: float = 0.2,
        anneal_epochs: int = 5,
        centroid_blend: float = 0.95,
        local_attention_impl: str = 'auto',
        sdpa_backend: str = 'auto',
        **kwargs,
    ):
        super().__init__()
        self.num_n = int(num_n)
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.time_patch_len = int(time_patch_len)
        self.spatial_patch_len = int(spatial_patch_len)
        self.time_feature_dim = int(time_feature_dim)
        self.revin_flag = bool(revin_flag)
        self.use_cluster_loss = bool(use_cluster_loss)
        self.cluster_loss_weight = float(cluster_loss_weight)
        self.anneal_epochs = int(anneal_epochs)
        self.pad_in_len = ceil(self.in_len / self.time_patch_len) * self.time_patch_len
        self.time_padding = self.pad_in_len - self.in_len
        self.num_time_patches = self.pad_in_len // self.time_patch_len
        self.padded_num_nodes = (
            ceil(self.num_n / self.spatial_patch_len) * self.spatial_patch_len
        )
        self.node_padding = self.padded_num_nodes - self.num_n

        if root_path is None:
            raise ValueError("root_path is required to load station coordinates")
        coordinate_path = os.path.join(root_path, coordinate_file)
        coordinates = np.load(coordinate_path)
        if coordinates.shape[0] != self.num_n:
            raise ValueError(
                "coordinate count {} does not match num_n {}".format(
                    coordinates.shape[0], self.num_n
                )
            )
        if spatial_order_file is not None:
            order = np.load(spatial_order_file)
        elif spatial_order_method == 'serpentine':
            order = serpentine_geographic_order(coordinates, geographic_bands)
        else:
            from .spatial_order import spatial_order
            order = spatial_order(coordinates, spatial_order_method,
                                  self.spatial_patch_len, spatial_order_seed)
        if not np.array_equal(np.sort(order), np.arange(self.num_n)):
            raise ValueError('Spatial order must contain each station exactly once')
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(self.num_n)
        sorted_coordinates = coordinates[order]
        if self.node_padding:
            sorted_coordinates = np.concatenate((
                sorted_coordinates,
                np.repeat(sorted_coordinates[-1:], self.node_padding, axis=0),
            ), axis=0)
        self.register_buffer(
            "station_order", torch.from_numpy(order).long(), persistent=True
        )
        self.register_buffer(
            "inverse_station_order",
            torch.from_numpy(inverse_order).long(),
            persistent=True,
        )

        self.embedding = SpatioTemporalPatchEmbedding(
            time_patch_len=self.time_patch_len,
            spatial_patch_len=self.spatial_patch_len,
            num_time_patches=self.num_time_patches,
            sorted_padded_coordinates=sorted_coordinates,
            d_model=d_model,
            time_feature_dim=self.time_feature_dim,
            dropout=dropout,
        )
        self.encoder = STCFormerEncoder(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            centroids_num=centroids_num,
            layers=layers,
            dropout=dropout,
            em_steps=em_steps,
            cluster_temperature=cluster_temperature,
            cluster_balance=cluster_balance,
            gumbel_tau=gumbel_tau,
            mi_weight=mi_weight,
            alpha=alpha,
            centroid_blend=centroid_blend,
            local_attention_impl=local_attention_impl,
            sdpa_backend=sdpa_backend,
        )
        self.forecast_head = DirectForecastHead(
            num_time_patches=self.num_time_patches,
            d_model=d_model,
            out_len=self.out_len,
            spatial_patch_len=self.spatial_patch_len,
            dropout=dropout,
        )
        self.persistence_delta = nn.Linear(self.in_len, self.out_len)
        nn.init.zeros_(self.persistence_delta.weight)
        nn.init.zeros_(self.persistence_delta.bias)

    def _calendar_features(self, history_data: torch.Tensor) -> torch.Tensor:
        available = max(history_data.shape[-1] - 1, 0)
        used = min(available, self.time_feature_dim)
        if used:
            calendar = history_data[..., 1:1 + used].mean(dim=2)
        else:
            calendar = history_data.new_zeros(
                history_data.shape[0], history_data.shape[1], 0
            )
        if used < self.time_feature_dim:
            zeros = history_data.new_zeros(
                history_data.shape[0],
                history_data.shape[1],
                self.time_feature_dim - used,
            )
            calendar = torch.cat((calendar, zeros), dim=-1)
        return calendar

    def _pad_time(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.time_padding:
            return tensor
        first = tensor[:, :1].expand(-1, self.time_padding, *tensor.shape[2:])
        return torch.cat((first, tensor), dim=1)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return_cluster_state = bool(kwargs.pop("return_cluster_state", False))
        values = history_data[..., 0]
        if values.shape[1] != self.in_len or values.shape[2] != self.num_n:
            raise ValueError("history_data shape does not match configured in_len/num_n")

        if self.revin_flag:
            means = values.mean(dim=1, keepdim=True).detach()
            centered = values - means
            stdev = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            normalized = centered / stdev
        else:
            means = values.new_zeros(values.shape[0], 1, values.shape[2])
            stdev = values.new_ones(values.shape[0], 1, values.shape[2])
            normalized = values

        sorted_values = normalized.index_select(2, self.station_order)
        sorted_history = sorted_values.transpose(1, 2)
        last_value = sorted_history[:, :, -1:]
        persistence = last_value + self.persistence_delta(sorted_history - last_value)
        persistence = persistence.transpose(1, 2)

        if self.node_padding:
            node_zeros = sorted_values.new_zeros(
                sorted_values.shape[0], sorted_values.shape[1], self.node_padding
            )
            padded_values = torch.cat((sorted_values, node_zeros), dim=2)
        else:
            padded_values = sorted_values
        padded_values = self._pad_time(padded_values)
        calendar = self._pad_time(self._calendar_features(history_data))

        tokens = self.embedding(padded_values, calendar)
        encoder_output = self.encoder(tokens)
        nonlinear_prediction = self.forecast_head(encoder_output.features)
        nonlinear_prediction = nonlinear_prediction[:, :, :self.num_n]
        sorted_prediction = persistence + nonlinear_prediction
        prediction = sorted_prediction.index_select(2, self.inverse_station_order)
        prediction = prediction * stdev + means

        model_return = {"prediction": prediction.unsqueeze(-1)}
        if self.use_cluster_loss:
            if epoch is None or self.anneal_epochs <= 0:
                anneal = 1.0
            else:
                anneal = min(1.0, max(0.0, float(epoch) / self.anneal_epochs))
            model_return["auxiliary_losses"] = {
                "info_loss": self.cluster_loss_weight * anneal * encoder_output.info_loss
            }
        if return_cluster_state:
            model_return["soft_assignments"] = encoder_output.soft_assignments
            model_return["hard_assignments"] = encoder_output.hard_assignments
        return model_return
