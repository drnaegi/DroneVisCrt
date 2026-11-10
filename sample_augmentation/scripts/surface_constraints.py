"""Constrain ray termination near a predicted surface instead of its mean alone."""

import torch
import torch.nn.functional as F


def surface_losses(weights, edges, ranges, valid, relative_band=0.08, absolute_band=0.02):
    centers = (edges[:, 1:] + edges[:, :-1]) * 0.5
    widths = ranges.abs() * relative_band + absolute_band
    valid = valid & torch.isfinite(ranges) & (ranges > edges[:, 0]) & (ranges < edges[:, -1])
    if not valid.any():
        zero = weights.sum() * 0
        return {"surface": zero, "free_space": zero, "depth": zero}
    centers = centers[valid]
    weights = weights[valid]
    ranges = ranges[valid]
    widths = widths[valid]
    support = torch.exp(-0.5 * ((centers - ranges[:, None]) / widths[:, None]).square())
    surface_mass = (weights * support).sum(-1)
    near_mass = (weights * (centers < (ranges - 2 * widths)[:, None])).sum(-1)
    opacity = weights.sum(-1)
    expected_depth = (weights * centers).sum(-1) / opacity.clamp_min(1e-6)
    return {
        "surface": -torch.log(surface_mass.clamp_min(1e-7)).mean(),
        "free_space": -torch.log((1 - near_mass).clamp_min(1e-7)).mean(),
        "depth": F.smooth_l1_loss(expected_depth.clamp_min(1e-4).log(), ranges.log()),
    }
