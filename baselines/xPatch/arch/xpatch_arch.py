"""BasicTS adaptation of xPatch (AAAI 2025, Apache-2.0).

Algorithm source: https://github.com/stitsyuk/xPatch (commit d12eeca).
The original CUDA-hardcoded EMA was replaced by a device/dtype-safe recurrence.
"""

import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, num_nodes, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_nodes))
        self.bias = nn.Parameter(torch.zeros(num_nodes))

    def normalize(self, x):
        self.mean = x.mean(1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - self.mean) / self.std * self.weight + self.bias

    def denormalize(self, x):
        return (x - self.bias) / (self.weight + self.eps ** 2) * self.std + self.mean


class ExponentialDecomposition(nn.Module):
    def __init__(self, alpha=0.3, beta=0.1, kind="ema"):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kind = kind

    def forward(self, x):
        level = x[:, 0]
        trend = x[:, 1] - x[:, 0] if self.kind == "dema" else None
        values = [level]
        for index in range(1, x.size(1)):
            if self.kind == "dema":
                previous = level
                level = self.alpha * x[:, index] + (1.0 - self.alpha) * (level + trend)
                trend = self.beta * (level - previous) + (1.0 - self.beta) * trend
            else:
                level = self.alpha * x[:, index] + (1.0 - self.alpha) * level
            values.append(level)
        moving_average = torch.stack(values, dim=1)
        return x - moving_average, moving_average


class DualStreamNetwork(nn.Module):
    def __init__(self, input_len, output_len, patch_len, stride):
        super().__init__()
        self.output_len = output_len
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (input_len - patch_len) // stride + 2
        dim = patch_len * patch_len
        self.padding = nn.ReplicationPad1d((0, stride))
        self.patch_projection = nn.Linear(patch_len, dim)
        self.patch_norm1 = nn.BatchNorm1d(self.patch_num)
        self.depthwise = nn.Conv1d(self.patch_num, self.patch_num, patch_len,
                                   stride=patch_len, groups=self.patch_num)
        self.patch_residual = nn.Linear(dim, patch_len)
        self.patch_norm2 = nn.BatchNorm1d(self.patch_num)
        self.pointwise = nn.Conv1d(self.patch_num, self.patch_num, 1)
        self.patch_norm3 = nn.BatchNorm1d(self.patch_num)
        self.nonlinear_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.patch_num * patch_len, output_len * 2), nn.GELU(),
            nn.Linear(output_len * 2, output_len),
        )
        self.linear_stream = nn.Sequential(
            nn.Linear(input_len, output_len * 4), nn.AvgPool1d(2),
            nn.LayerNorm(output_len * 2),
            nn.Linear(output_len * 2, output_len), nn.AvgPool1d(2),
            nn.LayerNorm(output_len // 2), nn.Linear(output_len // 2, output_len),
        )
        self.fusion = nn.Linear(output_len * 2, output_len)

    def forward(self, seasonal, trend):
        batch, _, channels = seasonal.shape
        seasonal = seasonal.transpose(1, 2).reshape(batch * channels, -1)
        trend = trend.transpose(1, 2).reshape(batch * channels, -1)
        patches = self.padding(seasonal).unfold(-1, self.patch_len, self.stride)
        patches = self.patch_norm1(torch.nn.functional.gelu(self.patch_projection(patches)))
        residual = self.patch_residual(patches)
        nonlinear = self.depthwise(patches)
        nonlinear = self.patch_norm2(torch.nn.functional.gelu(nonlinear)) + residual
        nonlinear = self.patch_norm3(torch.nn.functional.gelu(self.pointwise(nonlinear)))
        nonlinear = self.nonlinear_head(nonlinear)
        linear = self.linear_stream(trend)
        output = self.fusion(torch.cat([nonlinear, linear], dim=-1))
        return output.reshape(batch, channels, self.output_len).transpose(1, 2)


class xPatch(nn.Module):
    def __init__(self, input_len, output_len, num_nodes, patch_len=8, stride=4,
                 alpha=0.3, beta=0.1, ma_type="ema", use_revin=True, **kwargs):
        super().__init__()
        if output_len % 2:
            raise ValueError("xPatch requires an even output_len")
        self.use_revin = use_revin
        self.revin = RevIN(num_nodes)
        self.decomposition = ExponentialDecomposition(alpha, beta, ma_type)
        self.network = DualStreamNetwork(input_len, output_len, patch_len, stride)

    def forward(self, history_data, **kwargs):
        x = history_data[..., 0]
        if self.use_revin:
            x = self.revin.normalize(x)
        seasonal, trend = self.decomposition(x)
        prediction = self.network(seasonal, trend)
        if self.use_revin:
            prediction = self.revin.denormalize(prediction)
        return prediction.unsqueeze(-1)
