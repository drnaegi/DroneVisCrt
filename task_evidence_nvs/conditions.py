"""Projection of scene evidence into task-aware generation conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import Camera, project_point_cloud
from .scene import SceneRepresentation
from .task import TaskSpec


@dataclass
class EvidenceCondition:
    view_id: str
    rgb: np.ndarray
    depth: np.ndarray
    support: np.ndarray
    task_attention: np.ndarray
    instance_map: np.ndarray
    instance_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "rgb": self.rgb.astype(np.float32),
            "depth": self.depth.astype(np.float32),
            "support": self.support.astype(np.float32),
            "task_attention": self.task_attention.astype(np.float32),
            "instance_map": self.instance_map.astype(np.int32),
        }

    def summary(self) -> dict[str, Any]:
        valid_depth = np.isfinite(self.depth)
        return {
            "view_id": self.view_id,
            "shape": [int(self.rgb.shape[0]), int(self.rgb.shape[1])],
            "valid_projection_fraction": float(valid_depth.mean()),
            "support_mean_on_projection": float(self.support[valid_depth].mean()) if valid_depth.any() else 0.0,
            "task_attention_fraction": float((self.task_attention > 0).mean()),
            "task_evidence_mass": float(self.task_attention.sum()),
            "instance_names": list(self.instance_names),
            "metadata": self.metadata,
        }


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        y0, y1 = max(0, dy), min(height, height + dy)
        sy0, sy1 = max(0, -dy), min(height, height - dy)
        for dx in range(-radius, radius + 1):
            x0, x1 = max(0, dx), min(width, width + dx)
            sx0, sx1 = max(0, -dx), min(width, width - dx)
            result[y0:y1, x0:x1] |= mask[sy0:sy1, sx0:sx1]
    return result.astype(np.float32)


def build_evidence_condition(scene: SceneRepresentation, task: TaskSpec, camera: Camera,
                             *, view_id: str, context_radius: int = 8) -> EvidenceCondition:
    if scene.point_cloud is None:
        raise ValueError("a point cloud is required to build a projected evidence condition")
    projected = project_point_cloud(scene.point_cloud, camera)
    attention = np.zeros((camera.height, camera.width), dtype=np.float32)
    instance_names = tuple(str(item) for item in projected["instance_names"])
    for index, instance_id in enumerate(instance_names):
        instance = scene.instances.get(instance_id)
        if instance is None:
            continue
        pixels = projected["instance_map"] == index
        relevance = instance.task_relevance(task)
        attention = np.maximum(attention, relevance * _dilate(pixels, context_radius))
    support = projected["support"]
    attention *= np.clip(support + 0.25 * (support == 0), 0.0, 1.0)
    metadata = {
        "task_parser_version": task.parser_version,
        "condition_policy": "task relevance is gated by projected geometry support",
        "context_radius_pixels": context_radius,
        "is_generation_reliability": False,
        "is_target_label": False,
    }
    return EvidenceCondition(view_id, projected["rgb"], projected["depth"], support, attention,
                             projected["instance_map"], instance_names, metadata)
