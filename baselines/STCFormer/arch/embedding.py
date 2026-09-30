"""Geographic ordering and value-only spatio-temporal patch embedding."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def serpentine_geographic_order(coordinates: np.ndarray, num_bands: int = 12) -> np.ndarray:
    """Order stations by latitude bands with alternating longitude directions."""

    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coordinates must have shape [num_stations, >=2]")
    if num_bands < 1:
        raise ValueError("num_bands must be positive")

    longitude = coordinates[:, 0]
    latitude = coordinates[:, 1]
    latitude_span = max(float(latitude.max() - latitude.min()), 1e-8)
    band_ids = np.floor(
        (latitude.max() - latitude) / latitude_span * num_bands
    ).astype(np.int64)
    band_ids = np.clip(band_ids, 0, num_bands - 1)

    order = []
    for band_index in range(num_bands):
        station_ids = np.flatnonzero(band_ids == band_index)
        station_ids = station_ids[np.argsort(longitude[station_ids], kind="stable")]
        if band_index % 2:
            station_ids = station_ids[::-1]
        order.extend(station_ids.tolist())
    return np.asarray(order, dtype=np.int64)


class SpatioTemporalPatchEmbedding(nn.Module):
    """Embed raw patches, positions, coordinates, and optional calendar features.

    Unlike V2, the input projection contains only ``vec(P_{s,p})``. Temporal
    differences are deliberately deferred to the clustering branch of the event
    mixer, so that the forecasting representation does not duplicate them.
    """

    def __init__(
        self,
        time_patch_len: int,
        spatial_patch_len: int,
        num_time_patches: int,
        sorted_padded_coordinates: np.ndarray,
        d_model: int,
        time_feature_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_patch_len = int(time_patch_len)
        self.spatial_patch_len = int(spatial_patch_len)
        self.num_time_patches = int(num_time_patches)
        self.time_feature_dim = int(time_feature_dim)

        coordinates = np.asarray(sorted_padded_coordinates, dtype=np.float32)
        coordinate_mean = coordinates.mean(axis=0, keepdims=True)
        coordinate_std = coordinates.std(axis=0, keepdims=True)
        coordinates = (coordinates - coordinate_mean) / np.maximum(coordinate_std, 1e-6)
        if coordinates.shape[0] % self.spatial_patch_len:
            raise ValueError("padded coordinates must be divisible by spatial_patch_len")
        coordinate_patches = coordinates.reshape(
            -1, self.spatial_patch_len * coordinates.shape[1]
        )
        self.register_buffer(
            "coordinate_patches", torch.from_numpy(coordinate_patches), persistent=True
        )

        raw_patch_size = self.time_patch_len * self.spatial_patch_len
        self.raw_patch_size = raw_patch_size
        self.value_projection = nn.Sequential(
            nn.LayerNorm(raw_patch_size),
            nn.Linear(raw_patch_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_patches.shape[1], d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.calendar_projection: Optional[nn.Module]
        if self.time_feature_dim > 0:
            self.calendar_projection = nn.Sequential(
                nn.LayerNorm(self.time_patch_len * self.time_feature_dim),
                nn.Linear(self.time_patch_len * self.time_feature_dim, d_model),
            )
        else:
            self.calendar_projection = None
        self.temporal_position = nn.Parameter(
            torch.zeros(1, 1, self.num_time_patches, d_model)
        )
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        calendar_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return patch tokens with shape ``[B, S, T_p, D]``."""

        batch_size, length, num_nodes = values.shape
        if length != self.num_time_patches * self.time_patch_len:
            raise ValueError("unexpected padded history length")
        if num_nodes % self.spatial_patch_len:
            raise ValueError("station dimension must be spatially padded")
        num_spatial_patches = num_nodes // self.spatial_patch_len

        raw_patches = values.reshape(
            batch_size,
            self.num_time_patches,
            self.time_patch_len,
            num_spatial_patches,
            self.spatial_patch_len,
        ).permute(0, 3, 1, 2, 4).reshape(
            batch_size, num_spatial_patches, self.num_time_patches, -1
        )
        tokens = self.value_projection(raw_patches)
        tokens = tokens + self.coordinate_projection(self.coordinate_patches)[None, :, None]
        tokens = tokens + self.temporal_position

        if self.calendar_projection is not None:
            if calendar_features is None:
                raise ValueError("calendar_features are required when time_feature_dim > 0")
            calendar_patches = calendar_features.reshape(
                batch_size,
                self.num_time_patches,
                self.time_patch_len * self.time_feature_dim,
            )
            tokens = tokens + self.calendar_projection(calendar_patches)[:, None]

        return self.dropout(self.output_norm(tokens))
