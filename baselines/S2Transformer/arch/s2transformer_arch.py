"""BasicTS adaptation of hongyichenhitsz/S2Transformer (see ../SOURCE.md).

Keeps the released intra/inter attention, residual FFNs, mean pooling, fusion,
and direct horizon head. SDPA includes BOTH the geographic bias and pad mask.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .graph import prepare_graph


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, bias):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, dim//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj_drop(self.proj(attended.transpose(1, 2).reshape(batch, length, dim)))


def mlp(dim, d_ff, dropout):
    return nn.Sequential(nn.Linear(dim, d_ff), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(d_ff, dim), nn.Dropout(dropout))


class StructuredBlock(nn.Module):
    def __init__(self, dim, d_ff, heads, dropout):
        super().__init__()
        self.snorm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.snorm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.nnorm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.nnorm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(dim, heads, dropout)
        self.nattn = Attention(dim, heads, dropout)
        self.smlp = mlp(dim, d_ff, dropout)
        self.nmlp = mlp(dim, d_ff, dropout)
        self.fuse_lin = nn.Linear(2*dim, dim)

    def forward(self, x, patch, valid_flat, inverse, local_bias, global_bias):
        batch, _, dim = x.shape
        parts, size = patch.shape
        padded = F.pad(x, (0, 0, 0, 1))
        local = padded[:, patch].reshape(batch*parts, size, dim)
        local = local + self.sattn(self.snorm1(local), local_bias)
        local = local + self.smlp(self.snorm2(local))
        local = local.reshape(batch, parts, size, dim)
        # Intentionally matches the released code's padded mean pooling.
        global_x = self.nnorm1(local.mean(2))
        global_x = global_x + self.nattn(global_x, global_bias)
        global_x = global_x + self.nmlp(self.nnorm2(global_x))
        fused = local + F.relu(self.fuse_lin(torch.cat(
            (local, global_x[:, :, None].expand(-1, -1, size, -1)), -1)))
        # Each real node occurs exactly once. Avoid upstream repeated dummy-node
        # indexed writes (ambiguous backward for repeated padding indices).
        return fused.flatten(1, 2)[:, valid_flat][:, inverse]


class S2Transformer(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, root_path,
                 d_model=256, d_ff=512, n_heads=8, num_layers=2,
                 num_parts=32, dropout=0.1):
        super().__init__()
        if d_model <= 128 or d_model % n_heads:
            raise ValueError("d_model must exceed 128 and be divisible by n_heads")
        self.num_nodes, self.input_len = num_nodes, input_len
        graph = prepare_graph(root_path, num_nodes, num_parts, node_dim=48)
        for name, array in graph.items():
            self.register_buffer(name, torch.from_numpy(array))
        valid = self.patch.flatten() != num_nodes
        valid_flat = valid.nonzero().flatten()
        self.register_buffer("valid_flat", valid_flat)
        self.register_buffer("inverse", self.patch.flatten()[valid].argsort())
        mask = (self.patch != num_nodes)
        mask = mask[:, :, None] & mask[:, None, :]
        mask |= torch.eye(mask.shape[-1], dtype=torch.bool)[None]
        self.register_buffer("attention_mask", mask)
        self.input_projection = nn.Linear(input_len, d_model-128)
        # Actual prepared channels are tod, dow, day-of-year, month-of-year;
        # desc.json labels are stale. Use existing information, no future marks.
        self.calendar_sizes = (24, 7, 366, 12)
        self.calendar = nn.ModuleList([nn.Embedding(size, 16) for size in self.calendar_sizes])
        for embedding in self.calendar:
            nn.init.xavier_uniform_(embedding.weight)
        self.lonlat_weight = nn.Parameter(torch.zeros(16))
        self.learnable_scalar = nn.Parameter(torch.zeros(num_nodes+1))
        self.patch_learnable_scalar = nn.Parameter(torch.zeros(self.patch.shape[0]))
        self.blocks = nn.ModuleList([
            StructuredBlock(d_model, d_ff, n_heads, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, output_len)

    def forward(self, history_data, future_data=None, **kwargs):
        batch, length, nodes, features = history_data.shape
        if (length, nodes) != (self.input_len, self.num_nodes) or features < 5:
            raise ValueError("Expected [B, input_len, num_nodes, 5] history")
        value = self.input_projection(history_data[..., 0].transpose(1, 2))
        calendar = []
        for i, (embedding, size) in enumerate(zip(self.calendar, self.calendar_sizes)):
            indices = (history_data[:, -1, :, i+1]*size).round().long().clamp(0, size-1)
            calendar.append(embedding(indices))
        x = torch.cat([value, self.node_features[None].expand(batch, -1, -1),
                       *calendar, (self.spherical*self.lonlat_weight)[None].expand(batch, -1, -1)], -1)
        parts, size = self.patch.shape
        bias = self.intra*self.learnable_scalar[self.patch, None]
        bias = bias.masked_fill(~self.attention_mask, float("-inf"))
        local_bias = bias[None].expand(batch, -1, -1, -1).reshape(batch*parts, 1, size, size)
        global_bias = (self.inter*self.patch_learnable_scalar[:, None])[None, None]
        for block in self.blocks:
            x = block(x, self.patch, self.valid_flat, self.inverse, local_bias, global_bias)
        return self.head(x).transpose(1, 2).unsqueeze(-1)
