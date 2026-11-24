from __future__ import annotations

import numpy as np

from .geometry import Camera, UNKNOWN


def carla_camera(record: dict, camera_id: str, width: int = 1920, height: int = 1080,
                 horizontal_fov: float = 90) -> Camera:
    pitch, yaw, roll = np.deg2rad([record["pitch"], record["yaw"], record["roll"]])
    cosine_pitch, sine_pitch = np.cos(pitch), np.sin(pitch)
    cosine_yaw, sine_yaw = np.cos(yaw), np.sin(yaw)
    cosine_roll, sine_roll = np.cos(roll), np.sin(roll)
    rotation = np.array([
        [cosine_pitch * cosine_yaw, cosine_yaw * sine_pitch * sine_roll - sine_yaw * cosine_roll,
         -cosine_yaw * sine_pitch * cosine_roll - sine_yaw * sine_roll],
        [cosine_pitch * sine_yaw, sine_yaw * sine_pitch * sine_roll + cosine_yaw * cosine_roll,
         -sine_yaw * sine_pitch * cosine_roll + cosine_yaw * sine_roll],
        [sine_pitch, -cosine_pitch * sine_roll, cosine_pitch * cosine_roll],
    ])
    world_reflection = np.diag([1, -1, 1])
    cv_to_unreal = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    transform = np.eye(4)
    transform[:3, :3] = world_reflection @ rotation @ cv_to_unreal
    transform[:3, 3] = world_reflection @ np.array([record["x"], record["y"], record["z"]])
    focal = width / (2 * np.tan(np.deg2rad(horizontal_fov) / 2))
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    return Camera(camera_id, intrinsics, transform, width, height)


def decode_depth(encoded: np.ndarray) -> np.ndarray:
    if encoded.ndim != 2 or encoded.dtype != np.uint16:
        raise ValueError("SynDrone expects uint16 normalized CARLA depth")
    depth = encoded.astype(np.float32) * (1000.0 / 65535.0)
    depth[(encoded == 0) | (encoded == 65535)] = np.nan
    return depth


def map_labels(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2 or raw.dtype != np.uint8:
        raise ValueError("SynDrone expects uint8 scalar semantic IDs")
    known_ids = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                 40, 41, 100, 101, 102, 103, 104, 105]
    result = np.full(raw.shape, UNKNOWN, dtype=np.uint8)
    result[np.isin(raw, known_ids)] = 0
    result[np.isin(raw, [6, 7])] = 1
    result[raw == 21] = 2
    result[np.isin(raw, [8, 14, 22])] = 3
    result[np.isin(raw, [13, 20, 40, 41, 100, 101, 102, 103, 104, 105])] = UNKNOWN
    return result
