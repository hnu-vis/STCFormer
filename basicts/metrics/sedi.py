import torch
from typing import Optional

def sedi_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction 和 target 的形状必须一致")

    if mask is not None:
        mask = torch.broadcast_to(mask.bool(), prediction.shape)
        prediction = prediction[mask]
        target = target[mask]

    if prediction.numel() == 0:
        return torch.tensor(float("nan"), device=prediction.device)

    obs_extreme = (target >= threshold)
    pred_extreme = (prediction >= threshold)

    hits = (pred_extreme & obs_extreme).sum().to(torch.float64)
    misses = ((~pred_extreme) & obs_extreme).sum().to(torch.float64)
    false_alarms = (pred_extreme & (~obs_extreme)).sum().to(torch.float64)
    correct_negatives = ((~pred_extreme) & (~obs_extreme)).sum().to(torch.float64)

    total_events = hits + misses
    total_non_events = false_alarms + correct_negatives

    if total_events == 0 or total_non_events == 0:
        return torch.tensor(float("nan"), device=prediction.device)

    H = hits / total_events
    F = false_alarms / total_non_events

    H = torch.clamp(H, eps, 1 - eps)
    F = torch.clamp(F, eps, 1 - eps)

    num = torch.log(F) - torch.log(H) + torch.log1p(-H) - torch.log1p(-F)
    den = torch.log(F) + torch.log(H) + torch.log1p(-H) + torch.log1p(-F)

    if torch.abs(den) < eps:
        return torch.tensor(float("nan"), device=prediction.device)

    return (num / den).to(prediction.dtype if prediction.is_floating_point() else torch.float32)