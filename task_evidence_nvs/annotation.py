"""Review and reconstruction of generated-view supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .scene import Box
from .task import TaskSpec


@dataclass(frozen=True)
class ReviewedObject:
    object_id: str
    category: str
    bbox: Box
    status: str
    source_instance_id: str | None = None
    attributes: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        valid = {"preserved", "occluded", "out_of_view", "identity_unknown", "generated_new"}
        if self.status not in valid:
            raise ValueError(f"unsupported object status: {self.status}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass
class GeneratedViewAnnotation:
    view_id: str
    objects: list[ReviewedObject] = field(default_factory=list)
    ignored_regions: list[Box] = field(default_factory=list)
    unknown_claims: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def usable_objects(self) -> list[ReviewedObject]:
        return [item for item in self.objects if item.status in {"preserved", "generated_new"} and item.confidence >= 0.5]

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "objects": [
                {
                    "object_id": item.object_id,
                    "category": item.category,
                    "bbox": item.bbox.to_list(),
                    "status": item.status,
                    "source_instance_id": item.source_instance_id,
                    "attributes": list(item.attributes),
                    "relations": list(item.relations),
                    "confidence": item.confidence,
                }
                for item in self.objects
            ],
            "ignored_regions": [item.to_list() for item in self.ignored_regions],
            "unknown_claims": self.unknown_claims,
            "source_metadata": self.source_metadata,
        }


def review_generated_view(view_id: str, proposals: Iterable[dict[str, Any]], confirmations: Iterable[dict[str, Any]],
                          *, source_metadata: dict[str, Any] | None = None) -> GeneratedViewAnnotation:
    confirmation_list = list(confirmations)
    confirmed = {str(item.get("object_id")): item for item in confirmation_list}
    objects: list[ReviewedObject] = []
    ignored_regions: list[Box] = []
    unknown_claims: list[str] = []
    for proposal in proposals:
        object_id = str(proposal["object_id"])
        result = confirmed.get(object_id)
        if result is None:
            unknown_claims.append(object_id)
            if "bbox" in proposal:
                ignored_regions.append(Box(*[float(value) for value in proposal["bbox"]]))
            continue
        bbox = Box(*[float(value) for value in result["bbox"]])
        status = str(result.get("status", "identity_unknown"))
        if status == "identity_unknown":
            ignored_regions.append(bbox)
        objects.append(ReviewedObject(
            object_id=object_id,
            category=str(result["category"]),
            bbox=bbox,
            status=status,
            source_instance_id=result.get("source_instance_id"),
            attributes=tuple(str(value) for value in result.get("attributes", [])),
            relations=tuple(str(value) for value in result.get("relations", [])),
            confidence=float(result.get("confidence", 1.0)),
        ))
    known_ids = {item.object_id for item in objects}
    for result in confirmation_list:
        if str(result.get("object_id")) in known_ids:
            continue
        if result.get("status") == "generated_new":
            reviewed = ReviewedObject(
                object_id=str(result["object_id"]), category=str(result["category"]),
                bbox=Box(*[float(value) for value in result["bbox"]]), status="generated_new",
                attributes=tuple(str(value) for value in result.get("attributes", [])),
                relations=tuple(str(value) for value in result.get("relations", [])),
                confidence=float(result.get("confidence", 1.0)),
            )
            objects.append(reviewed)
            known_ids.add(reviewed.object_id)
    return GeneratedViewAnnotation(view_id, objects, ignored_regions, unknown_claims, source_metadata or {})


def rebuild_grounded_text(annotation: GeneratedViewAnnotation, task: TaskSpec) -> dict[str, Any]:
    usable = annotation.usable_objects
    counts: dict[str, int] = {}
    for item in usable:
        counts[item.category] = counts.get(item.category, 0) + 1
    count_text = "、".join(f"{category}{count}个" for category, count in sorted(counts.items())) or "没有可确认目标"
    assertions = [
        f"可确认目标：{count_text}。",
        f"任务关注类别：{'、'.join(task.categories) if task.categories else '未指定'}。",
    ]
    for item in usable:
        relations = "、".join(item.relations)
        if relations:
            assertions.append(f"{item.object_id}是{item.category}，关系为{relations}。")
        else:
            assertions.append(f"{item.object_id}是{item.category}。")
    if annotation.unknown_claims:
        assertions.append("以下对象无法确认身份，不能作为确定监督：" + "、".join(annotation.unknown_claims) + "。")
    return {
        "caption": "".join(assertions),
        "questions": [
            {"question": "当前视角中可确认的目标数量是多少？", "answer": len(usable)},
            {"question": "当前视角是否存在可核验的任务证据？", "answer": bool(usable)},
        ],
        "grounded_object_ids": [item.object_id for item in usable],
        "unknown_claims": annotation.unknown_claims,
        "task_raw_text": task.raw_text,
    }
