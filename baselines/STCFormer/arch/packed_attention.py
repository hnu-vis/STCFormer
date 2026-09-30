"""Cluster-packed SDPA with the original straight-through gate derivative.

For a fixed hard cluster c, G_ij = H_ic H_jc. The query factor cancels
in row normalization; adding log(H_jc) to the key logit preserves the gate
gradient at one-hot H. Routing indices alone would silently drop this gradient.
The key-only logit bias is encoded in an extra Q/K channel so fused kernels
do not need an N x N additive mask (or its dense backward).
"""
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def fused_sdpa(q, k, v, *, dropout_p, scale, backend='auto', attn_mask=None):
    choices = {'flash': SDPBackend.FLASH_ATTENTION,
               'efficient': SDPBackend.EFFICIENT_ATTENTION,
               'math': SDPBackend.MATH}
    if backend not in ('auto', *choices):
        raise ValueError(f'Unknown SDPA backend: {backend}')
    context = sdpa_kernel(choices[backend]) if backend != 'auto' else nullcontext()
    with context:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=dropout_p, scale=scale)


def masked_cluster_attention(module, tokens, assignments, backend='auto'):
    """Fused dense-mask path for short sequences; preserves the same gate VJP.

    This saves dense per-head scores but does NOT claim block-sparse FLOPs.
    A small shared [M,1,N,N] mask remains. No differentiable N x N gate.
    """
    m,n,width = tokens.shape
    heads,dim = module.n_heads,module.head_dim
    q,k,v = [proj(tokens).reshape(m,n,heads,dim).transpose(1,2)
             for proj in (module.query,module.key,module.value)]
    labels = assignments.detach().argmax(-1)
    membership = assignments.gather(-1,labels[...,None]).squeeze(-1)
    bias = membership.clamp_min(1e-8).log().to(q.dtype)/module.scale
    extra = 8-dim%8
    q = F.pad(torch.cat((q,torch.ones_like(q[...,:1])),dim=-1),(0,extra-1))
    k = F.pad(torch.cat((k,bias[:,None,:,None].expand(-1,heads,-1,-1)),dim=-1),(0,extra-1))
    v = F.pad(v,(0,extra))
    same = labels[:,:,None] == labels[:,None,:]
    aligned = ((n+7)//8)*8
    mask = tokens.new_full((m,1,n,aligned),float('-inf'),dtype=q.dtype)
    mask[:,:,:,:n].masked_fill_(same[:,None],0)
    out = fused_sdpa(q,k,v,attn_mask=mask[:,:,:,:n],
                     dropout_p=module.dropout.p if module.training else 0.,
                     scale=module.scale,backend=backend)
    return module.output(out[...,:dim].transpose(1,2).reshape(m,n,width))


def packed_cluster_attention(module, tokens, assignments, backend='auto'):
    """Project once, pack real clusters, attend, then scatter to original order.

    No global N x N tensor is built. Empty clusters are skipped. Length buckets
    limit skew-related padding; balanced groups share one rounded launch.
    Gradients go through QKV
    and the selected one-hot membership, but not through sorting indices.
    Dropout has the same distribution, not the same random-number ordering.
    """
    m, n, width = tokens.shape
    if assignments.shape[:2] != (m, n):
        raise ValueError('Expected assignments [M,N,K] for tokens [M,N,D]')
    clusters = assignments.shape[-1]
    heads, dim = module.n_heads, module.head_dim
    q, k, v = [proj(tokens).reshape(m*n, heads, dim)
               for proj in (module.query, module.key, module.value)]
    labels = assignments.detach().argmax(-1).reshape(-1)
    group = torch.arange(m, device=tokens.device).repeat_interleave(n)*clusters + labels
    order = torch.argsort(group, stable=True)
    counts = torch.bincount(group, minlength=m*clusters)
    starts = counts.cumsum(0)-counts
    # A single small synchronization is included in all reported benchmarks.
    counts_cpu = counts.detach().cpu().tolist()
    buckets = {}
    for group_id, count in enumerate(counts_cpu):
        if count:
            capacity = max(4, 1 << (count-1).bit_length())
            buckets.setdefault(capacity, []).append(group_id)
    nonempty = [i for i,c in enumerate(counts_cpu) if c]
    largest = max(counts_cpu)
    # Balanced clusters benefit from a SINGLE fused launch. Switch to length
    # buckets only when skew would make a common capacity wasteful.
    if largest <= 3 * (m*n/len(nonempty)):
        buckets = {((largest+7)//8)*8: nonempty}
    membership = assignments.reshape(m*n, clusters).gather(1, labels[:, None]).squeeze(1)
    key_bias = membership.clamp_min(1e-8).log().to(q.dtype)
    restored = torch.zeros_like(q)
    dropout = module.dropout.p if module.training else 0.0
    for capacity, ids in buckets.items():
        ids = torch.tensor(ids, device=tokens.device, dtype=torch.long)
        position = torch.arange(capacity, device=tokens.device)
        valid = position[None] < counts[ids, None]
        indices = order[(starts[ids, None]+position[None]).clamp_max(m*n-1)]
        qp, kp, vp = [x[indices].transpose(1, 2) for x in (q, k, v)]
        # Pad head width to a multiple of 8, preserving the ORIGINAL scale.
        extra = 8 - dim % 8
        ones = torch.ones_like(qp[..., :1])
        bias = key_bias[indices] / module.scale
        # Finite sentinel avoids inf*0 in fused backward. Invalid keys have
        # zero softmax probability at normal finite model logits.
        bias = torch.where(valid, bias, torch.full_like(bias, torch.finfo(q.dtype).min/16))
        bias = bias[:, None, :, None].expand(-1, heads, -1, -1)
        qp = F.pad(torch.cat((qp, ones), dim=-1), (0, extra-1))
        kp = F.pad(torch.cat((kp, bias), dim=-1), (0, extra-1))
        vp = F.pad(vp, (0, extra))
        out = fused_sdpa(qp, kp, vp, dropout_p=dropout, scale=module.scale, backend=backend)
        # Boolean indexing would call nonzero and synchronize CUDA for each
        # bucket. Keep padded shapes static and scatter-add ZERO padded rows.
        # Every genuine token has exactly one nonzero contribution.
        out = out[..., :dim].transpose(1, 2).reshape(-1, heads, dim)
        out = out * valid.reshape(-1, 1, 1)
        restored.index_add_(0, indices.reshape(-1), out)
    return module.output(restored.reshape(m, n, width))
