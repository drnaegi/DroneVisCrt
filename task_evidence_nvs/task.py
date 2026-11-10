"""Controlled parsing of inspection task descriptions.

The parser intentionally produces a small, inspectable vocabulary.  A task
description assigns importance to observed evidence; it never creates an
object label by itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable


_CATEGORY_ALIASES = {
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "car": "car",
    "cars": "car",
    "汽车": "car",
    "车辆": "vehicle",
    "轿车": "car",
    "truck": "truck",
    "trucks": "truck",
    "卡车": "truck",
    "货车": "truck",
    "bus": "bus",
    "buses": "bus",
    "公交车": "bus",
    "行人": "person",
    "人": "person",
    "person": "person",
    "people": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "自行车": "bicycle",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "摩托车": "motorcycle",
    "motorcycle": "motorcycle",
    "建筑": "building",
    "建筑物": "building",
    "building": "building",
    "buildings": "building",
    "道路": "road",
    "道路区域": "road",
    "road": "road",
    "roads": "road",
    "树": "tree",
    "树木": "tree",
    "tree": "tree",
    "trees": "tree",
}

_ATTRIBUTE_ALIASES = {
    "红色": "red",
    "红": "red",
    "red": "red",
    "蓝色": "blue",
    "蓝": "blue",
    "blue": "blue",
    "白色": "white",
    "白": "white",
    "white": "white",
    "黑色": "black",
    "黑": "black",
    "black": "black",
    "小目标": "small",
    "小型": "small",
    "small": "small",
    "远距离": "far",
    "远处": "far",
    "far": "far",
    "遮挡": "occluded",
    "被遮挡": "occluded",
    "occluded": "occluded",
    "斜视": "oblique",
    "倾斜视角": "oblique",
    "oblique": "oblique",
    "低照度": "low_light",
    "夜间": "low_light",
    "low light": "low_light",
}

_RELATION_ALIASES = {
    "道路边缘": "near_road_edge",
    "靠近道路": "near_road",
    "道路附近": "near_road",
    "附近": "near",
    "相邻": "adjacent",
    "左侧": "left_of",
    "右侧": "right_of",
    "上方": "above",
    "下方": "below",
    "near road edge": "near_road_edge",
    "near road": "near_road",
    "near": "near",
    "adjacent": "adjacent",
    "left of": "left_of",
    "right of": "right_of",
}


@dataclass(frozen=True)
class EvidenceRequirement:
    category: str
    attributes: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskSpec:
    raw_text: str
    categories: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    view_conditions: tuple[str, ...] = ()
    requirements: tuple[EvidenceRequirement, ...] = ()
    parser_version: str = "controlled-v1"

    def relevance(self, category: str, attributes: Iterable[str] = ()) -> float:
        category = _normalise_category(category)
        observed = {_normalise_attribute(item) for item in attributes}
        for requirement in self.requirements:
            family_match = requirement.category == "vehicle" and category in {"vehicle", "car", "truck", "bus", "bicycle", "motorcycle"}
            if requirement.category == category or family_match:
                if not requirement.attributes or not observed:
                    return requirement.weight
                overlap = len(observed.intersection(requirement.attributes))
                return requirement.weight * (0.5 + 0.5 * overlap / len(requirement.attributes))
        return 0.35 if not self.categories else 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["requirements"] = [item.to_dict() for item in self.requirements]
        return result


def _normalise_category(value: str) -> str:
    return _CATEGORY_ALIASES.get(value.strip().lower(), value.strip().lower())


def _normalise_attribute(value: str) -> str:
    return _ATTRIBUTE_ALIASES.get(value.strip().lower(), value.strip().lower())


def _find_aliases(text: str, aliases: dict[str, str]) -> list[str]:
    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for alias, canonical in aliases.items():
        match = re.search(re.escape(alias.lower()), lowered)
        if match:
            found.append((match.start(), canonical))
    return list(dict.fromkeys(item[1] for item in sorted(found)))


def parse_task(text: str, *, category_weights: dict[str, float] | None = None) -> TaskSpec:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("task description must be a non-empty string")
    categories = _find_aliases(text, _CATEGORY_ALIASES)
    attributes = _find_aliases(text, _ATTRIBUTE_ALIASES)
    relations = _find_aliases(text, _RELATION_ALIASES)
    view_conditions = [item for item in attributes if item in {"far", "oblique", "small", "low_light", "occluded"}]
    weights = category_weights or {}
    requirements = tuple(
        EvidenceRequirement(
            category=category,
            attributes=tuple(attributes),
            relations=tuple(relations),
            weight=float(weights.get(category, 1.0)),
        )
        for category in categories
    )
    return TaskSpec(
        raw_text=text,
        categories=tuple(categories),
        attributes=tuple(attributes),
        relations=tuple(relations),
        view_conditions=tuple(view_conditions),
        requirements=requirements,
    )
