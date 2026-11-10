"""Small NumPy geometry primitives used by the evidence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Camera:
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        pose = np.asarray(self.world_to_camera, dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape (3, 3)")
        if pose.shape != (4, 4):
            raise ValueError("world_to_camera must have shape (4, 4)")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "world_to_camera", pose)

    @property
    def center(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera)[:3, 3]

    def to_dict(self) -> dict:
        return {
            "intrinsics": self.intrinsics.tolist(),
            "world_to_camera": self.world_to_camera.tolist(),
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    confidence: np.ndarray
    instance_ids: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float32)
        rgb = np.asarray(self.rgb, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        if rgb.shape != xyz.shape:
            raise ValueError("rgb must have shape (N, 3)")
        if confidence.shape != (len(xyz),):
            raise ValueError("confidence must have shape (N,)")
        if self.instance_ids and len(self.instance_ids) != len(xyz):
            raise ValueError("instance_ids must have one item per point")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "rgb", np.clip(rgb, 0.0, 1.0))
        object.__setattr__(self, "confidence", np.clip(confidence, 0.0, 1.0))
        if not self.instance_ids:
            object.__setattr__(self, "instance_ids", tuple([None] * len(xyz)))


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=np.float32)


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]], dtype=np.float32)


def _look_at(center: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - center
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        raise ValueError("camera center and target must differ")
    forward = forward / norm
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    camera_down = np.cross(forward, right)
    rotation = np.stack([right, camera_down, forward], axis=0)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ center
    return pose


def orbit_camera(base: Camera, *, yaw_deg: float = 0.0, pitch_deg: float = 0.0,
                 radius_scale: float = 0.0, target: Iterable[float] = (0.0, 0.0, 0.0)) -> Camera:
    target_array = np.asarray(tuple(target), dtype=np.float32)
    if target_array.shape != (3,):
        raise ValueError("target must contain three coordinates")
    center = base.center
    offset = center - target_array
    radius = float(np.linalg.norm(offset))
    if radius < 1e-6:
        raise ValueError("base camera must have a non-zero orbit radius")
    offset = offset * (1.0 + float(radius_scale))
    offset = _rotation_y(math.radians(yaw_deg)) @ offset
    offset = _rotation_x(math.radians(pitch_deg)) @ offset
    new_center = target_array + offset
    return Camera(base.intrinsics.copy(), _look_at(new_center, target_array, np.array([0.0, 1.0, 0.0], dtype=np.float32)), base.width, base.height)


def project_point_cloud(points: PointCloud, camera: Camera) -> dict[str, np.ndarray]:
    count = len(points.xyz)
    homogeneous = np.concatenate([points.xyz, np.ones((count, 1), dtype=np.float32)], axis=1)
    camera_points = (camera.world_to_camera @ homogeneous.T).T[:, :3]
    z = camera_points[:, 2]
    valid = np.isfinite(camera_points).all(axis=1) & (z > 1e-5)
    pixels_h = (camera.intrinsics @ camera_points.T).T
    pixels = np.zeros((count, 2), dtype=np.float32)
    pixels[valid] = pixels_h[valid, :2] / z[valid, None]
    in_frame = valid & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
    depth = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    rgb = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
    support = np.zeros((camera.height, camera.width), dtype=np.float32)
    instance_map = np.full((camera.height, camera.width), -1, dtype=np.int32)
    point_indices = np.flatnonzero(in_frame)
    order = point_indices[np.argsort(z[point_indices])]
    instance_names: list[str] = []
    instance_to_index: dict[str, int] = {}
    for point_index in order:
        x, y = np.rint(pixels[point_index]).astype(int)
        if x < 0 or x >= camera.width or y < 0 or y >= camera.height:
            continue
        if z[point_index] >= depth[y, x]:
            continue
        depth[y, x] = z[point_index]
        rgb[y, x] = points.rgb[point_index]
        support[y, x] = points.confidence[point_index]
        name = points.instance_ids[point_index]
        if name is None:
            instance_map[y, x] = -1
        else:
            if name not in instance_to_index:
                instance_to_index[name] = len(instance_names)
                instance_names.append(name)
            instance_map[y, x] = instance_to_index[name]
    return {
        "depth": depth,
        "rgb": rgb,
        "support": support,
        "instance_map": instance_map,
        "instance_names": np.asarray(instance_names, dtype=object),
        "projected_pixels": pixels,
        "valid_points": in_frame,
    }
