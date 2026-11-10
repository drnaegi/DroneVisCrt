#!/usr/bin/env python3
"""Run the evidence-condition pipeline on a saved ViewCrafter scene trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task_evidence_nvs.conditions import build_evidence_condition
from task_evidence_nvs.geometry import Camera, PointCloud
from task_evidence_nvs.scene import SceneBuilder
from task_evidence_nvs.task import parse_task


def _camera(camera_data: dict, index: int) -> Camera:
    focal_x, focal_y = camera_data["focal_length"][index]
    principal_x, principal_y = camera_data["principal_point"][index]
    width, height = camera_data["geometry_image_size"]
    intrinsics = np.array([[focal_x, 0.0, principal_x], [0.0, focal_y, principal_y], [0.0, 0.0, 1.0]], dtype=np.float32)
    row_world_to_view = np.asarray(camera_data["R"][index], dtype=np.float32)
    translation = np.asarray(camera_data["T"][index], dtype=np.float32)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = row_world_to_view.T
    world_to_camera[:3, 3] = translation
    return Camera(intrinsics, world_to_camera, int(width), int(height))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="saved wide_view_trials scene directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="保持场景结构和可见物体，适应斜视和远距离观测")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 8, 15])
    parser.add_argument("--context-radius", type=int, default=3)
    args = parser.parse_args()
    geometry = np.load(args.scene / "geometry.npz")
    camera_data = json.loads((args.scene / "cameras.json").read_text(encoding="utf-8"))
    points = geometry["points"].reshape(-1, 3)
    colors = geometry["colors"].reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points, colors = points[valid], colors[valid]
    cloud = PointCloud(points, colors, np.ones(len(points), dtype=np.float32), tuple(["scene"] * len(points)))
    task = parse_task(args.task)
    width, height = camera_data["geometry_image_size"]
    references = [{
        "image_id": "source",
        "path": str((args.scene / "input.png").resolve()),
        "width": int(width),
        "height": int(height),
        "instances": [{"instance_id": "scene", "category": "scene", "bbox": [0, 0, width, height], "confidence": 1.0}],
    }]
    scene = SceneBuilder(args.scene.name, task).build(references, cloud)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "task.json").write_text(json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "scene.json").write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    existing_conditions = np.load(args.scene / "conditions.npy")
    records = []
    for index in args.frames:
        if index < 0 or index >= len(camera_data["R"]):
            raise ValueError(f"frame index out of range: {index}")
        camera = _camera(camera_data, index)
        condition = build_evidence_condition(scene, task, camera, view_id=f"frame_{index:02d}", context_radius=args.context_radius)
        directory = args.output / f"frame_{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "evidence_condition.npz", **condition.to_arrays())
        condition_image = Image.fromarray(np.rint(np.clip(condition.rgb, 0, 1) * 255).astype(np.uint8))
        condition_image.save(directory / "projected_condition.png")
        existing = Image.fromarray(np.rint(np.clip(existing_conditions[index], 0, 1) * 255).astype(np.uint8)).resize((int(width), int(height)), Image.Resampling.BILINEAR)
        existing.save(directory / "saved_geometry_condition.png")
        valid_depth = np.isfinite(condition.depth)
        record = {
            "frame": index,
            "condition": condition.summary(),
            "saved_geometry_condition_mae": float(np.abs(np.asarray(condition_image, dtype=np.float32) / 255.0 - np.asarray(existing, dtype=np.float32) / 255.0).mean()),
            "source_dataset": str(args.scene.resolve()),
            "label_status": "geometry_condition_only_no_object_annotation",
            "viewcrafter_generated_image": str((args.scene / "generated" / f"{index:03d}.png").resolve()) if (args.scene / "generated" / f"{index:03d}.png").exists() else None,
            "valid_projection_fraction": float(valid_depth.mean()),
        }
        (directory / "metadata.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    report = {
        "schema_version": "existing-scene-evidence-trial-v1",
        "scene": str(args.scene.resolve()),
        "task": task.to_dict(),
        "frames": records,
        "interpretation": {
            "is_downstream_detection_result": False,
            "is_generated_label_result": False,
            "uses_existing_saved_point_cloud": True,
            "object_identity_available": False,
            "next_required_step": "add independent object annotations before detection or grounded-text claims",
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scene": args.scene.name, "frames": len(records), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
