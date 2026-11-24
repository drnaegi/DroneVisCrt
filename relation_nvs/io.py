from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import Camera, Reference


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_camera(record: dict) -> Camera:
    if set(record) != {"camera_id", "intrinsics", "camera_to_world", "width", "height"}:
        raise ValueError("Camera records must contain only the declared calibration fields")
    return Camera(**record)


def camera_dict(camera: Camera) -> dict:
    return {"camera_id": camera.camera_id, "intrinsics": camera.intrinsics.tolist(),
            "camera_to_world": camera.camera_to_world.tolist(), "width": camera.width, "height": camera.height}


def load_source_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"schema_version", "scene_id", "scene_group", "split", "protocol", "references", "cameras", "provenance"}
    if set(manifest) != allowed or manifest["schema_version"] != 1:
        raise ValueError("Source manifest has missing or forbidden fields; evaluation truth belongs in a separate manifest")
    if manifest["protocol"] not in {"controlled_source", "rgb_source", "fixture"}:
        raise ValueError("Unknown source protocol")
    if manifest["split"] not in {"train", "development", "heldout"}:
        raise ValueError("Unknown split")
    references = []
    for item in manifest["references"]:
        if set(item) != {"source_id", "camera", "rgb", "depth", "labels", "depth_kind", "hashes", "modality_origin"}:
            raise ValueError("Unexpected reference fields")
        origin = item["modality_origin"]
        if not isinstance(origin, dict) or set(origin) != {"depth", "labels"}:
            raise ValueError("Missing modality origins")
        if manifest["protocol"] == "rgb_source" and any(value != "predicted" for value in origin.values()):
            raise ValueError("RGB protocol cannot use ground-truth depth or labels")
        files = {name: path.parent / item[name] for name in ["rgb", "depth", "labels"]}
        for name, filename in files.items():
            if sha256(filename) != item["hashes"][name]:
                raise ValueError(f"Source hash mismatch: {filename}")
        rgb = np.asarray(Image.open(files["rgb"]).convert("RGB"))
        labels = np.asarray(Image.open(files["labels"]))
        depth = np.load(files["depth"], allow_pickle=False)
        references.append(Reference(item["source_id"], load_camera(item["camera"]), rgb, depth, labels, item["depth_kind"]))
    cameras = [load_camera(record) for record in manifest["cameras"]]
    if len({camera.camera_id for camera in cameras}) != len(cameras):
        raise ValueError("Duplicate candidate camera IDs")
    return manifest, references, cameras


def validate_splits(manifests: list[dict]):
    groups, source_hashes = {}, {}
    for manifest in manifests:
        split = manifest["split"]
        group = manifest["scene_group"]
        if not group:
            raise ValueError("A verified scene group is required")
        if group in groups and groups[group] != split:
            raise ValueError(f"Scene group crosses splits: {group}")
        groups[group] = split
        for reference in manifest["references"]:
            digest = reference["hashes"]["rgb"]
            if digest in source_hashes and source_hashes[digest] != split:
                raise ValueError("Reference RGB crosses splits")
            source_hashes[digest] = split
    return {"scene_groups": len(groups), "unique_reference_rgb": len(source_hashes),
            "geographic_independence_certified": False}
