"""Adapters that keep evidence conditions separate from model-specific runners."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .conditions import EvidenceCondition
from .geometry import Camera
from .scene import SceneRepresentation
from .selection import CandidateView
from .task import TaskSpec


@dataclass(frozen=True)
class GenerationRequest:
    scene: SceneRepresentation
    task: TaskSpec
    candidate: CandidateView
    camera: Camera
    condition: EvidenceCondition
    reference_paths: tuple[str, ...]


class GeneratorRunner(Protocol):
    def __call__(self, request: GenerationRequest) -> dict[str, Any]: ...


class DryRunGenerator:
    """Persist conditions and model-neutral metadata without claiming a label."""

    def __init__(self, output: str | Path):
        self.output = Path(output)

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        directory = self.output / request.scene.scene_id / request.candidate.candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "evidence_condition.npz", **request.condition.to_arrays())
        metadata = {
            "scene_id": request.scene.scene_id,
            "candidate_id": request.candidate.candidate_id,
            "task": request.task.to_dict(),
            "camera": request.camera.to_dict(),
            "condition": request.condition.summary(),
            "reference_paths": list(request.reference_paths),
            "generated_image": None,
            "label_status": "not_generated",
        }
        (directory / "request.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata


class CallableGenerator:
    def __init__(self, runner: GeneratorRunner):
        self.runner = runner

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        result = self.runner(request)
        if not isinstance(result, dict):
            raise TypeError("generator runner must return a dictionary")
        result.setdefault("label_status", "needs_review")
        result.setdefault("condition_view_id", request.condition.view_id)
        return result


def viewcrafter_parameters(camera: Camera, base_camera: Camera) -> dict[str, float]:
    """Return relative pose metadata for an external ViewCrafter adapter.

    The values are descriptive and do not silently convert unknown scene scale
    into physical altitude.
    """
    base_center = base_camera.center
    center = camera.center
    delta = center - base_center
    return {
        "relative_translation_x": float(delta[0]),
        "relative_translation_y": float(delta[1]),
        "relative_translation_z": float(delta[2]),
        "relative_distance": float(np.linalg.norm(delta)),
        "scale_semantics": "relative_scene_units",
    }
