"""Budgeted candidate selection with explicit reliability gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class CandidateView:
    candidate_id: str
    condition_summary: dict[str, Any]
    reliability_estimate: float
    utility_estimate: float
    generation_cost: float = 1.0
    annotation_cost: float = 1.0
    feature: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability_estimate <= 1.0:
            raise ValueError("reliability_estimate must be in [0, 1]")
        if self.generation_cost <= 0 or self.annotation_cost <= 0:
            raise ValueError("candidate costs must be positive")


@dataclass
class SelectionResult:
    selected: list[CandidateView]
    rejected: list[dict[str, Any]]
    generation_cost: float
    annotation_cost: float
    policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [item.candidate_id for item in self.selected],
            "rejected": self.rejected,
            "generation_cost": self.generation_cost,
            "annotation_cost": self.annotation_cost,
            "policy": self.policy,
        }


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    return sum((left[index] - right[index]) ** 2 for index in range(length)) ** 0.5


def select_candidates(candidates: Iterable[CandidateView], *, max_generation_cost: float,
                      max_annotation_cost: float, reliability_threshold: float = 0.5,
                      diversity_weight: float = 0.1) -> SelectionResult:
    if max_generation_cost < 0 or max_annotation_cost < 0:
        raise ValueError("budgets must be non-negative")
    candidate_list = list(candidates)
    ranked = sorted(candidate_list, key=lambda item: item.utility_estimate, reverse=True)
    selected: list[CandidateView] = []
    rejected: list[dict[str, Any]] = []
    generation_cost = 0.0
    annotation_cost = 0.0
    for candidate in ranked:
        reason = None
        if candidate.reliability_estimate < reliability_threshold:
            reason = "below_reliability_threshold"
        elif generation_cost + candidate.generation_cost > max_generation_cost:
            reason = "generation_budget"
        elif annotation_cost + candidate.annotation_cost > max_annotation_cost:
            reason = "annotation_budget"
        else:
            redundancy = max((_distance(candidate.feature, item.feature) for item in selected), default=float("inf"))
            if selected and redundancy == 0.0 and diversity_weight > 0:
                reason = "duplicate_feature"
        if reason:
            rejected.append({"candidate_id": candidate.candidate_id, "reason": reason})
            continue
        selected.append(candidate)
        generation_cost += candidate.generation_cost
        annotation_cost += candidate.annotation_cost
    return SelectionResult(
        selected=selected,
        rejected=rejected,
        generation_cost=generation_cost,
        annotation_cost=annotation_cost,
        policy={
            "reliability_threshold": reliability_threshold,
            "max_generation_cost": max_generation_cost,
            "max_annotation_cost": max_annotation_cost,
            "diversity_weight": diversity_weight,
            "reliability_is_proxy": any(item.metadata.get("reliability_is_proxy", False) for item in candidate_list),
        },
    )
