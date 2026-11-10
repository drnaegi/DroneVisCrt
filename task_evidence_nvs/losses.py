"""Optional PyTorch losses for task and cross-view evidence constraints."""

from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("task_evidence_nvs.losses requires PyTorch") from error
    return torch


def task_evidence_loss(prediction: Any, target: Any, attention: Any, support: Any, *, eps: float = 1e-6):
    torch = _torch()
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if attention.shape != support.shape:
        raise ValueError("attention and support must have the same shape")
    mask = (attention * support).to(dtype=prediction.dtype)
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    error = (prediction - target).abs() * mask
    return error.sum() / mask.sum().clamp_min(eps)


def cross_view_consistency_loss(features: Any, correspondences: Any, valid_mask: Any, *, eps: float = 1e-6):
    torch = _torch()
    if features.ndim != 3 or features.shape[0] < 2:
        raise ValueError("features must have shape (views, tokens, channels)")
    if correspondences.shape[:2] != features.shape[:2]:
        raise ValueError("correspondences must index each view and token")
    if valid_mask.shape != correspondences.shape:
        raise ValueError("valid_mask must have the same shape as correspondences")
    anchor = features[0][correspondences[0].clamp(min=0)]
    pair_features = features[1:]
    pair_indices = correspondences[1:].clamp(min=0)
    gathered = torch.stack([pair_features[index][pair_indices[index]] for index in range(len(pair_features))])
    mask = valid_mask[1:].to(dtype=features.dtype)
    difference = (gathered - anchor.unsqueeze(0)).pow(2).mean(dim=-1)
    return (difference * mask).sum() / mask.sum().clamp_min(eps)


def evidence_conditioned_loss(diffusion_loss: Any, prediction: Any, target: Any,
                              attention: Any, support: Any, features: Any,
                              correspondences: Any, valid_mask: Any, *, lambda_task: float = 1.0,
                              lambda_cross: float = 1.0) -> dict[str, Any]:
    task_loss = task_evidence_loss(prediction, target, attention, support)
    cross_loss = cross_view_consistency_loss(features, correspondences, valid_mask)
    total = diffusion_loss + lambda_task * task_loss + lambda_cross * cross_loss
    return {"total": total, "diffusion": diffusion_loss, "task": task_loss, "cross_view": cross_loss}
