"""Fixed-station BasicTS adaptation of CDPNet's continuous grid dynamics."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _knn_weights(source: torch.Tensor, query: torch.Tensor, k: int):
    distances = torch.cdist(query.float(), source.float())
    values, indices = torch.topk(distances, k=min(k, source.shape[0]), largest=False)
    weights = (values + 1e-6).reciprocal()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return indices, weights


class CDPNet(nn.Module):
    """Continuous Diffusive Prediction Network for fixed weather stations.

    Station histories are calibrated into a regular geographic grid, evolved
    by a residual convolutional diffusion operator, and interpolated back to
    stations.  A direct MLP forecast supplies the calibrated initialization;
    the grid rollout estimates the future diffusive differences.
    """

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        position_path: str,
        grid_h: int,
        grid_w: int,
        d_model: int = 256,
        d_ff: int = 512,
        n_neighbors: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.grid_h = grid_h
        self.grid_w = grid_w

        positions = np.load(position_path).astype(np.float32)
        if positions.shape != (num_nodes, 3):
            raise ValueError(f"Expected {(num_nodes, 3)} positions, got {positions.shape}")
        coordinates = torch.from_numpy(positions[:, :2])
        coord_min = coordinates.amin(dim=0)
        coord_scale = (coordinates.amax(dim=0) - coord_min).clamp_min(1e-6)
        coordinates = (coordinates - coord_min) / coord_scale
        altitude = torch.from_numpy(positions[:, 2:3])
        altitude = (altitude - altitude.mean()) / altitude.std().clamp_min(1e-6)
        geo_features = torch.cat((coordinates, altitude), dim=-1)

        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, grid_h),
            torch.linspace(0, 1, grid_w),
            indexing="ij",
        )
        grid_coordinates = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        grid_neighbors, grid_weights = _knn_weights(coordinates, grid_coordinates, n_neighbors)
        station_neighbors, station_weights = _knn_weights(grid_coordinates, coordinates, n_neighbors)
        self.register_buffer("geo_features", geo_features)
        self.register_buffer("grid_neighbors", grid_neighbors)
        self.register_buffer("grid_weights", grid_weights)
        self.register_buffer("station_neighbors", station_neighbors)
        self.register_buffer("station_weights", station_weights)

        self.history_projection = nn.Linear(seq_len, d_model)
        self.geo_projection = nn.Linear(3, d_model)
        self.time_projection = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.initialization = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.direct_head = nn.Linear(d_model, pred_len)

        self.diffusion = nn.Sequential(
            nn.Conv2d(d_model, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_ff, d_ff, kernel_size=3, padding=1, groups=d_ff),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(d_ff, d_model, kernel_size=1),
        )
        self.diffusion_norm = nn.GroupNorm(8, d_model)
        self.difference_head = nn.Linear(d_model, 1)
        self.diffusion_scale = nn.Parameter(torch.tensor(0.1))
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def _stations_to_grid(self, station_features: torch.Tensor) -> torch.Tensor:
        gathered = station_features[:, self.grid_neighbors]  # [B, G, K, D]
        return (gathered * self.grid_weights[None, :, :, None]).sum(dim=2)

    def _grid_to_stations(self, grid_features: torch.Tensor) -> torch.Tensor:
        gathered = grid_features[:, self.station_neighbors]  # [B, N, K, D]
        return (gathered * self.station_weights[None, :, :, None]).sum(dim=2)

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
            raise ValueError("CDPNet expects target plus four calendar features")

        target = history_data[..., 0]
        mean = target.mean(dim=1, keepdim=True).detach()
        std = target.std(dim=1, keepdim=True, unbiased=False).add(1e-5).detach()
        normalized = (target - mean) / std

        hidden = self.history_projection(normalized.permute(0, 2, 1))
        hidden = hidden + self.geo_projection(self.geo_features)[None]
        hidden = hidden + self.time_projection(history_data[:, -1, :, 1:5])
        hidden = hidden + self.initialization(hidden)
        direct = self.direct_head(hidden).permute(0, 2, 1)

        grid = self._stations_to_grid(hidden)
        grid = grid.transpose(1, 2).reshape(-1, hidden.shape[-1], self.grid_h, self.grid_w)
        differences = []
        for _ in range(self.pred_len):
            delta = self.diffusion(self.diffusion_norm(grid))
            grid = grid + torch.tanh(self.diffusion_scale) * delta
            flat_grid = grid.flatten(2).transpose(1, 2)
            station_state = self._grid_to_stations(flat_grid)
            differences.append(self.difference_head(station_state).squeeze(-1))
        diffusive = torch.stack(differences, dim=1)
        prediction = direct + torch.tanh(self.output_scale) * diffusive
        prediction = prediction * std[:, :1] + mean[:, :1]
        return prediction.unsqueeze(-1)
