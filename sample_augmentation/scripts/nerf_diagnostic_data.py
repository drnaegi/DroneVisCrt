"""Camera-only split preparation for Lego and the original LLFF Fern scene."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def camera_rays(pose, intrinsics, height, width):
    rows, columns = torch.meshgrid(torch.arange(height, device=pose.device),
                                  torch.arange(width, device=pose.device), indexing="ij")
    camera = torch.stack([(columns + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
                          -(rows + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1],
                          -torch.ones_like(columns)], -1).float().reshape(-1, 3)
    lengths = camera.norm(dim=-1)
    directions = F.normalize(camera @ pose[:3, :3].T, dim=-1)
    return pose[:3, 3].expand_as(directions), directions, lengths


def llff_to_opengl(raw):
    output = np.broadcast_to(np.eye(4), (len(raw), 4, 4)).copy()
    output[:, :3, :4] = np.concatenate([raw[:, :, 1:2], -raw[:, :, 0:1], raw[:, :, 2:4]], axis=2)
    return output


def direction_angles(first, others):
    return np.degrees(np.arccos(np.clip(others[:, :3, 2] @ first[:3, 2], -1, 1)))


def reference_order(poses, maximum=8):
    maximum = min(maximum, len(poses))
    centers = poses[:, :3, 3]
    forward = -poses[:, :3, 2]
    center_scale = max(float(np.linalg.norm(centers - centers.mean(0), axis=1).max()), 1e-6)
    features = np.concatenate([centers / center_scale, forward], axis=1)
    selected = [0]
    if maximum > 1:
        angles = direction_angles(poses[0], poses)
        candidates = np.flatnonzero((angles >= 10) & (angles <= 35))
        candidates = candidates[candidates != 0]
        if len(candidates):
            second = int(candidates[np.argmin(abs(angles[candidates] - 20))])
        else:
            distances = np.linalg.norm(features[1:] - features[0], axis=1)
            target = max(float(distances.max()) * 0.35, 1e-6)
            second = int(np.argmin(abs(distances - target))) + 1
        selected.append(second)
    while len(selected) < maximum:
        distances = np.linalg.norm(features[:, None] - features[selected][None], axis=-1).min(1)
        distances[selected] = -1
        selected.append(int(distances.argmax()))
    return selected


def prepare_scene(root, scene, long_side=128):
    root = Path(root)
    frames = []
    source_files = []
    if scene == "lego":
        folder = root / "nerf_synthetic/lego"
        for split in ("train", "val", "test"):
            metadata_path = folder / f"transforms_{split}.json"
            metadata = json.loads(metadata_path.read_text())
            source_files.append(metadata_path)
            for index, frame in enumerate(metadata["frames"]):
                path = folder / (frame["file_path"] + ".png")
                width = height = long_side
                focal = 0.5 * width / np.tan(0.5 * metadata["camera_angle_x"])
                intrinsics = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
                frames.append({"id": f"{split}/{index:03d}", "path": str(path.resolve()), "split": split,
                               "pose": frame["transform_matrix"], "intrinsics": intrinsics,
                               "height": height, "width": width, "near": 2.0, "far": 6.0})
        notes = ["Official Blender train/val/test split; exact render cameras.",
                 "RGBA composited onto white before resizing; no mask or depth supervision."]
        background = 1.0
    elif scene == "fern":
        folder = root / "nerf_llff_data/fern"
        metadata_path = folder / "poses_bounds.npy"
        source_files.append(metadata_path)
        metadata = np.load(metadata_path)
        raw = metadata[:, :15].reshape(-1, 3, 5)
        poses = llff_to_opengl(raw)
        image_folder = folder / "images_8"
        if not image_folder.is_dir():
            image_folder = folder / "images"
        images = sorted(path for path in image_folder.iterdir() if path.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if len(images) != len(poses):
            raise ValueError("LLFF image/camera count mismatch")
        train_indices = [index for index in range(len(poses)) if index % 8 not in (0, 4)]
        bounds = metadata[:, -2:]
        train_bounds = bounds[train_indices]
        focus = float(np.median(train_bounds))
        centers = poses[train_indices, :3, 3]
        forwards = -poses[train_indices, :3, 2]
        center = (centers + forwards * focus).mean(0)
        poses[:, :3, 3] = (poses[:, :3, 3] - center) / focus
        near = float(train_bounds[:, 0].min() * 0.9 / focus)
        far = float(train_bounds[:, 1].max() * 1.1 / focus)
        for index, path in enumerate(images):
            original_height, original_width, focal = raw[index, :, 4]
            width = long_side
            height = round(width * original_height / original_width)
            intrinsics = [[float(focal * width / original_width), 0, width / 2],
                          [0, float(focal * height / original_height), height / 2], [0, 0, 1]]
            split = "test" if index % 8 == 0 else "val" if index % 8 == 4 else "train"
            frames.append({"id": path.stem, "path": str(path.resolve()), "split": split,
                           "pose": poses[index].tolist(), "intrinsics": intrinsics,
                           "height": height, "width": width, "near": near, "far": far})
        notes = ["Official LLFF SfM poses, not ground truth calibration; poses were estimated jointly upstream.",
                 "Sorted image indices modulo 8: 0 test, 4 development, others training pool.",
                 "Normalization/bounds from training-pool metadata only; no test RGB/depth supervision.",
                 "Euclidean ray rendering without NDC; not a reproduction of the original NeRF paper."]
        background = 0.5
    else:
        raise ValueError(scene)
    pool = [frame for frame in frames if frame["split"] == "train"]
    order = reference_order(np.array([frame["pose"] for frame in pool]))
    validation = [frame for frame in frames if frame["split"] == "val"]
    test = [frame for frame in frames if frame["split"] == "test"]
    if scene == "lego":
        validation = validation[::10]
        test = test[::10]
        notes.append("Development uses every 10th official val frame (10); test every 10th test frame (20).")
    selected = [pool[index] for index in order]
    for frame in selected + validation + test:
        frame["sha256"] = checksum(frame["path"])
    poses = np.array([frame["pose"] for frame in selected])
    rotations = poses[:, :3, :3]
    errors = np.max(abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)))
    if errors > 1e-4 or np.min(np.linalg.det(rotations)) < 0.999:
        raise ValueError("Invalid rotation convention")
    return {"scene": scene, "background": background, "long_side": long_side, "notes": notes,
            "train_pool_size": len(pool), "source_frame_count": len(frames),
            "training_order": selected, "validation": validation, "test": test,
            "reference_selection": "First training-pool frame; second camera near 20 degrees if available, then camera-feature farthest sampling; no RGB scoring.",
            "source_files": [{"path": str(path.resolve()), "sha256": checksum(path)} for path in source_files],
            "rotation_orthogonality_max_error": float(errors),
            "second_reference_angle_deg": float(direction_angles(poses[0], poses[1:2])[0])}


def load_pixels(frame, background):
    path = Path(frame["path"])
    if checksum(path) != frame["sha256"]:
        raise ValueError("Source image changed: " + str(path))
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    colors = rgba[:, :, :3] * rgba[:, :, 3:] + background * (1 - rgba[:, :, 3:])
    channels = [np.asarray(Image.fromarray(colors[:, :, channel], mode="F").resize(
        (frame["width"], frame["height"]), Image.Resampling.LANCZOS)) for channel in range(3)]
    return np.stack(channels, -1).clip(0, 1).copy()


def foreground_mask(frame):
    with Image.open(frame["path"]) as image:
        alpha = image.convert("RGBA").getchannel("A").resize(
            (frame["width"], frame["height"]), Image.Resampling.LANCZOS)
    return np.asarray(alpha) >= 128
