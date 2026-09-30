"""Within-patch mutual information between spatial tokens and cluster labels."""

import torch


def patch_cluster_info_loss(assignments: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """mean_b sum_p [mean_s H(R[b,p,s,:]) - H(mean_s R[b,p,s,:])].

    N denotes spatial patches, not raw stations if spatial_patch_len > 1.
    Marginals are independent per sample/time patch; there is no adjacent-time
    joint. Sum over time as in the supplied equation; the encoder averages
    layers. The caller applies the sole coefficient cluster_loss_weight.
    """
    if assignments.ndim != 4 or any(size == 0 for size in assignments.shape):
        raise ValueError("assignments must have nonempty shape [B, T, N, K]")
    probabilities = assignments.float()
    conditional_entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(-1).mean(-1)
    marginal = probabilities.mean(dim=-2)
    marginal_entropy = -(marginal * marginal.clamp_min(eps).log()).sum(-1)
    return (conditional_entropy - marginal_entropy).sum(dim=1).mean()


def temporal_cluster_info_loss(assignments: torch.Tensor, mi_weight=None, alpha=None,
                               eps: float = 1e-8) -> torch.Tensor:
    """Compatibility alias; legacy mi_weight/alpha no longer change the loss."""
    return patch_cluster_info_loss(assignments, eps=eps)
