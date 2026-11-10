#!/usr/bin/env python3
"""Build task-aware scene evidence conditions from a JSON scene manifest.

The command deliberately stops before image generation.  Its output is a
model-neutral condition package that an external ViewCrafter or diffusion
runner can consume, while preserving the distinction between geometry support
and reviewed generated labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task_evidence_nvs.conditions import build_evidence_condition
from task_evidence_nvs.geometry import Camera, PointCloud, orbit_camera
from task_evidence_nvs.scene import SceneBuilder
from task_evidence_nvs.selection import CandidateView, select_candidates
from task_evidence_nvs.task import parse_task


def _load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def _camera(data: dict) -> Camera:
    return Camera(
        np.asarray(data["intrinsics"], dtype=np.float32),
        np.asarray(data["world_to_camera"], dtype=np.float32),
        int(data["width"]),
        int(data["height"]),
    )


def _point_cloud(data: dict | None) -> PointCloud | None:
    if not data:
        return None
    points = data.get("points", data)
    xyz = np.asarray([item["xyz"] for item in points], dtype=np.float32)
    rgb = np.asarray([item.get("rgb", [0.5, 0.5, 0.5]) for item in points], dtype=np.float32)
    confidence = np.asarray([item.get("confidence", 1.0) for item in points], dtype=np.float32)
    ids = tuple(item.get("instance_id") for item in points)
    return PointCloud(xyz, rgb, confidence, ids)


def _proxy_scores(condition, task, scene):
    valid = np.isfinite(condition.depth)
    projection = float(valid.mean())
    support = float(condition.support[valid].mean()) if valid.any() else 0.0
    evidence_mass = float(condition.task_attention.sum())
    target_pixels = max(1.0, evidence_mass)
    reliability = max(0.0, min(1.0, 0.5 * support + 0.5 * projection))
    utility = float(min(1.0, evidence_mass / (condition.rgb.shape[0] * condition.rgb.shape[1] * 0.15)))
    difficulty = float(max(0.0, min(1.0, 1.0 - target_pixels / (condition.rgb.shape[0] * condition.rgb.shape[1] * 0.05))))
    if not task.categories:
        utility *= 0.5
    return reliability, utility, {
        "geometry_projection_fraction": projection,
        "geometry_support_mean": support,
        "task_evidence_mass": evidence_mass,
        "difficulty_proxy": difficulty,
        "reliability_is_proxy": True,
        "utility_is_proxy": True,
        "proxy_warning": "not a generated-image reliability or downstream training utility label",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-radius", type=int, default=8)
    parser.add_argument("--yaw", type=float, nargs="*", default=[-10.0, 0.0, 10.0])
    parser.add_argument("--pitch", type=float, nargs="*", default=[-5.0, 0.0, 5.0])
    parser.add_argument("--radius-scale", type=float, nargs="*", default=[-0.1, 0.0, 0.1])
    parser.add_argument("--max-generation-cost", type=float, default=4.0)
    parser.add_argument("--max-annotation-cost", type=float, default=16.0)
    parser.add_argument("--reliability-threshold", type=float, default=0.35)
    args = parser.parse_args()
    manifest = _load_manifest(args.manifest)
    task = parse_task(str(manifest["task"]))
    base_camera = _camera(manifest["camera"])
    cloud = _point_cloud(manifest.get("point_cloud"))
    if cloud is None:
        raise ValueError("manifest must provide point_cloud for projected conditions")
    scene = SceneBuilder(str(manifest.get("scene_id", args.manifest.stem)), task).build(manifest["references"], cloud)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "task.json").write_text(json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "scene.json").write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    candidates: list[CandidateView] = []
    conditions = []
    for yaw in args.yaw:
        for pitch in args.pitch:
            for radius_scale in args.radius_scale:
                if yaw == 0 and pitch == 0 and radius_scale == 0:
                    continue
                candidate_id = f"yaw{yaw:+g}_pitch{pitch:+g}_radius{radius_scale:+g}".replace("+", "p").replace("-", "m")
                camera = orbit_camera(base_camera, yaw_deg=yaw, pitch_deg=pitch, radius_scale=radius_scale)
                condition = build_evidence_condition(scene, task, camera, view_id=candidate_id, context_radius=args.context_radius)
                reliability, utility, proxy = _proxy_scores(condition, task, scene)
                summary = condition.summary()
                summary.update({"yaw_deg": yaw, "pitch_deg": pitch, "radius_scale": radius_scale, **proxy})
                candidates.append(CandidateView(candidate_id, summary, reliability, utility,
                                                generation_cost=1.0, annotation_cost=1.0,
                                                feature=(float(yaw) / 30.0, float(pitch) / 20.0, float(radius_scale)),
                                                metadata=proxy))
                conditions.append((candidate_id, camera, condition, summary))
    selection = select_candidates(candidates, max_generation_cost=args.max_generation_cost,
                                  max_annotation_cost=args.max_annotation_cost,
                                  reliability_threshold=args.reliability_threshold)
    selected_ids = {item.candidate_id for item in selection.selected}
    records = []
    for candidate_id, camera, condition, summary in conditions:
        directory = args.output / "conditions" / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "evidence_condition.npz", **condition.to_arrays())
        record = {
            "candidate_id": candidate_id,
            "selected": candidate_id in selected_ids,
            "camera": camera.to_dict(),
            "condition": summary,
            "selection": next((item for item in selection.rejected if item["candidate_id"] == candidate_id), None),
        }
        (directory / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    manifest_out = {
        "schema_version": "task-evidence-conditions-v1",
        "scene_id": scene.scene_id,
        "task": task.to_dict(),
        "selection": selection.to_dict(),
        "candidates": records,
        "condition_policy": {
            "source_boxes_are_not_inherited": True,
            "projected_geometry_is_proposal_only": True,
            "generated_images_require_review": True,
            "final_test_labels_not_consumed": True,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selection.selected), "candidates": len(candidates), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
