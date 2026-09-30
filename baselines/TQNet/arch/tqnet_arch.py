"""Apache-2.0 TQNet architecture adapted to the BasicTS tensor interface.

Algorithm source: https://github.com/ACAT-SCUT/TQNet (commit 15e19cb).
The adapter derives the cycle index from BasicTS time features and optionally
groups very large station sets to keep atmospheric experiments tractable.
"""

import torch
from torch import nn


class TQNet(nn.Module):
    def __init__(self, input_len, output_len, num_nodes, cycle_len=24,
                 d_model=128, dropout=0.1, num_heads=4,
                 attention_group_size=256, use_revin=True, **kwargs):
        super().__init__()
        if input_len % num_heads:
            raise ValueError("input_len must be divisible by num_heads")
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.cycle_len = cycle_len
        self.group_size = attention_group_size
        self.use_revin = use_revin
        self.temporal_query = nn.Parameter(torch.zeros(cycle_len, num_nodes))
        nn.init.normal_(self.temporal_query, std=0.02)
        self.channel_aggregator = nn.MultiheadAttention(
            embed_dim=input_len, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.input_projection = nn.Linear(input_len, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.output_projection = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, output_len))

    def _aggregate(self, query, values):
        # Exact TQNet attention for normal MTS; bounded blocks for 3k+ stations.
        if self.group_size <= 0 or values.size(1) <= self.group_size:
            return self.channel_aggregator(query, values, values, need_weights=False)[0]
        chunks = []
        for start in range(0, values.size(1), self.group_size):
            stop = min(start + self.group_size, values.size(1))
            chunks.append(self.channel_aggregator(
                query[:, start:stop], values[:, start:stop], values[:, start:stop],
                need_weights=False,
            )[0])
        return torch.cat(chunks, dim=1)

    def forward(self, history_data, **kwargs):
        x = history_data[..., 0]  # [B, L, N]
        if self.use_revin:
            mean = x.mean(1, keepdim=True)
            variance = x.var(1, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(variance + 1e-5)

        values = x.transpose(1, 2)
        if history_data.size(-1) > 1:
            cycle_index = torch.remainder(
                (history_data[:, -1, 0, 1] * self.cycle_len).long(), self.cycle_len
            )
        else:
            cycle_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        offsets = torch.arange(self.input_len, device=x.device)
        query_index = (cycle_index[:, None] - self.input_len + 1 + offsets) % self.cycle_len
        query = self.temporal_query[query_index].transpose(1, 2)
        channel_information = self._aggregate(query, values)
        embedded = self.input_projection(values + channel_information)
        prediction = self.output_projection(embedded + self.mlp(embedded)).transpose(1, 2)
        if self.use_revin:
            prediction = prediction * torch.sqrt(variance + 1e-5) + mean
        return prediction.unsqueeze(-1)
