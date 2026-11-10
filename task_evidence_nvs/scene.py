"""Scene and instance identity records for multi-reference input."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from .geometry import PointCloud
from .task import TaskSpec


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not self.x2 > self.x1 or not self.y2 > self.y1:
            raise ValueError("box must have positive area")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def clip(self, width: int, height: int) -> "Box":
        x1 = max(0.0, min(self.x1, float(width)))
        y1 = max(0.0, min(self.y1, float(height)))
        x2 = max(0.0, min(self.x2, float(width)))
        y2 = max(0.0, min(self.y2, float(height)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box has no visible area after clipping")
        return Box(x1, y1, x2, y2)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class InstanceObservation:
    instance_id: str
    category: str
    bbox: Box
    image_id: str
    attributes: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if not self.category:
            raise ValueError("category must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class SceneInstance:
    instance_id: str
    category: str
    observations: list[InstanceObservation] = field(default_factory=list)
    point_count: int = 0
    geometry_support: float = 0.0
    identity_confidence: float = 0.0
    anchor_xyz: np.ndarray | None = None

    def task_relevance(self, task: TaskSpec) -> float:
        attributes = [attribute for observation in self.observations for attribute in observation.attributes]
        return task.relevance(self.category, attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "category": self.category,
            "observations": [
                {
                    "instance_id": item.instance_id,
                    "category": item.category,
                    "bbox": item.bbox.to_list(),
                    "image_id": item.image_id,
                    "attributes": list(item.attributes),
                    "confidence": item.confidence,
                }
                for item in self.observations
            ],
            "point_count": self.point_count,
            "geometry_support": self.geometry_support,
            "identity_confidence": self.identity_confidence,
            "anchor_xyz": self.anchor_xyz.tolist() if self.anchor_xyz is not None else None,
        }


@dataclass
class SceneRepresentation:
    scene_id: str
    references: list[dict[str, Any]]
    instances: dict[str, SceneInstance]
    point_cloud: PointCloud | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "references": self.references,
            "instances": {key: value.to_dict() for key, value in self.instances.items()},
            "point_count": len(self.point_cloud.xyz) if self.point_cloud is not None else 0,
            "metadata": self.metadata,
        }


class SceneBuilder:
    def __init__(self, scene_id: str, task: TaskSpec | None = None):
        self.scene_id = scene_id
        self.task = task

    def build(self, references: Iterable[dict[str, Any]], point_cloud: PointCloud | None = None) -> SceneRepresentation:
        reference_list = list(references)
        if not reference_list:
            raise ValueError("at least one reference image is required")
        instances: dict[str, SceneInstance] = {}
        for reference_index, reference in enumerate(reference_list):
            image_id = str(reference.get("image_id", f"reference-{reference_index}"))
            width = int(reference["width"])
            height = int(reference["height"])
            for item_index, raw in enumerate(reference.get("instances", [])):
                category = str(raw["category"])
                instance_id = str(raw.get("instance_id", f"{category}:{item_index}"))
                box_values = raw["bbox"]
                if len(box_values) != 4:
                    raise ValueError(f"bbox for {instance_id} must contain four values")
                box = Box(*[float(value) for value in box_values]).clip(width, height)
                observation = InstanceObservation(
                    instance_id=instance_id,
                    category=category,
                    bbox=box,
                    image_id=image_id,
                    attributes=tuple(str(value) for value in raw.get("attributes", [])),
                    confidence=float(raw.get("confidence", 1.0)),
                )
                if instance_id not in instances:
                    instances[instance_id] = SceneInstance(instance_id, category)
                instances[instance_id].observations.append(observation)
        if point_cloud is not None:
            names = np.asarray(point_cloud.instance_ids, dtype=object)
            for instance_id, instance in instances.items():
                selected = names == instance_id
                instance.point_count = int(selected.sum())
                if instance.point_count:
                    confidence = point_cloud.confidence[selected]
                    instance.geometry_support = float(np.mean(confidence))
                    instance.anchor_xyz = np.mean(point_cloud.xyz[selected], axis=0)
                else:
                    instance.geometry_support = 0.0
                explicit_identity = all(item.instance_id == instance_id for item in instance.observations)
                instance.identity_confidence = min(1.0, 0.25 + 0.25 * len(instance.observations)) if explicit_identity else 0.25
        else:
            for instance in instances.values():
                instance.identity_confidence = min(1.0, 0.25 + 0.25 * len(instance.observations))
        metadata = {
            "identity_policy": "explicit instance_id is required for full cross-view identity confidence",
            "task_categories": list(self.task.categories) if self.task else [],
        }
        return SceneRepresentation(self.scene_id, reference_list, instances, point_cloud, metadata)
