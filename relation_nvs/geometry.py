from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

UNKNOWN = 255


@dataclass(frozen=True)
class Camera:
    camera_id: str
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.camera_id) or self.camera_id in {".", ".."}:
            raise ValueError("Camera ID must be a safe filename component")
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        transform = np.asarray(self.camera_to_world, dtype=np.float64)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive")
        if intrinsics.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError("Expected 3x3 intrinsics and 4x4 OpenCV camera-to-world")
        if not np.isfinite(intrinsics).all() or not np.isfinite(transform).all():
            raise ValueError("Nonfinite camera")
        if not np.allclose(intrinsics[2], [0, 0, 1]) or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0:
            raise ValueError("Invalid pinhole intrinsics")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Camera rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Convert handedness to OpenCV before loading")
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous camera transform")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "camera_to_world", transform)

    def resized(self, width: int, height: int) -> Camera:
        scale_x, scale_y = width / self.width, height / self.height
        pixel_transform = np.array([[scale_x, 0, (scale_x - 1) / 2],
                                    [0, scale_y, (scale_y - 1) / 2], [0, 0, 1]])
        return Camera(self.camera_id, pixel_transform @ self.intrinsics,
                      self.camera_to_world.copy(), width, height)


@dataclass
class Reference:
    source_id: str
    camera: Camera
    rgb: np.ndarray
    depth: np.ndarray
    labels: np.ndarray
    depth_kind: str = "z"

    def __post_init__(self):
        shape = (self.camera.height, self.camera.width)
        if self.rgb.shape != (*shape, 3) or self.depth.shape != shape or self.labels.shape != shape:
            raise ValueError("Reference modalities must align exactly with the camera")
        if self.rgb.dtype != np.uint8 or self.labels.dtype != np.uint8:
            raise ValueError("RGB and mapped labels must be uint8")
        if self.depth_kind not in {"z", "range"}:
            raise ValueError("Depth must explicitly be z-depth or Euclidean range")
        if not set(np.unique(self.labels)).issubset({0, 1, 2, 3, UNKNOWN}):
            raise ValueError("Unmapped semantic label")


@dataclass
class SceneEvidence:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray
    vertex_labels: np.ndarray
    face_labels: np.ndarray
    face_supported: np.ndarray
    source_index: np.ndarray
    source_pixel: np.ndarray
    source_ids: tuple[str, ...]


@dataclass
class Render:
    rgb: np.ndarray
    labels: np.ndarray
    depth: np.ndarray
    face_index: np.ndarray
    barycentric: np.ndarray
    geometry_mask: np.ndarray
    support_mask: np.ndarray


def source_free_space_rejections(reference: Reference, points: np.ndarray, faces: np.ndarray,
                                 discontinuous: np.ndarray) -> np.ndarray:
    rejected = np.zeros(len(faces), dtype=bool)
    candidate_indices = np.flatnonzero(discontinuous)
    if not len(candidate_indices):
        return rejected
    weights = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    samples = np.einsum("sk,fkd->fsd", weights, points[faces[candidate_indices]])
    projected = samples @ reference.camera.intrinsics.T
    pixels = np.rint(projected[..., :2] / np.maximum(projected[..., 2:3], 1e-8)).astype(np.int64)
    inside = (samples[..., 2] > 0) & (pixels[..., 0] >= 0) & (pixels[..., 0] < reference.camera.width) & (pixels[..., 1] >= 0) & (pixels[..., 1] < reference.camera.height)
    observed = np.full(inside.shape, np.nan)
    observed[inside] = reference.depth[pixels[..., 1][inside], pixels[..., 0][inside]]
    predicted = samples[..., 2] if reference.depth_kind == "z" else np.linalg.norm(samples, axis=-1)
    contradicted = inside & np.isfinite(observed) & (observed > 0) & (predicted < observed - np.maximum(0.1, 0.01 * observed))
    rejected[candidate_indices] = contradicted.sum(axis=1) >= 2
    return rejected


def build_scene(references: list[Reference], stride: int = 4,
                depth_jump_relative: float = 0.05, geometry_filter: str = "legacy_unknown_occlusion") -> SceneEvidence:
    if not references or stride < 1 or not 0 < depth_jump_relative < 1:
        raise ValueError("Invalid mesh construction parameters")
    if len({reference.source_id for reference in references}) != len(references):
        raise ValueError("Duplicate reference IDs")
    if geometry_filter not in {"legacy_unknown_occlusion", "source_free_space_v1"}:
        raise ValueError("Unknown geometry filter")
    pieces = []
    offset = 0
    for source_index, reference in enumerate(references):
        camera = reference.camera
        columns = np.unique(np.r_[np.arange(0, camera.width, stride), camera.width - 1])
        rows = np.unique(np.r_[np.arange(0, camera.height, stride), camera.height - 1])
        grid_x, grid_y = np.meshgrid(columns, rows)
        pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(camera.intrinsics).T
        if reference.depth_kind == "range":
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        depth = reference.depth[grid_y, grid_x].ravel().astype(np.float64)
        valid = np.isfinite(depth) & (depth > 0)
        points = rays * np.where(valid, depth, 0)[:, None]
        vertices = points @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3]
        indices = np.arange(len(pixels)).reshape(grid_x.shape)
        upper_left = indices[:-1, :-1].ravel()
        upper_right = indices[:-1, 1:].ravel()
        lower_left = indices[1:, :-1].ravel()
        lower_right = indices[1:, 1:].ravel()
        faces = np.concatenate([np.stack([upper_left, upper_right, lower_left], axis=1),
                                np.stack([upper_right, lower_right, lower_left], axis=1)])
        faces = faces[valid[faces].all(axis=1)]
        labels = reference.labels[grid_y, grid_x].ravel()
        face_depth = depth[faces]
        continuous = np.ptp(face_depth, axis=1) <= depth_jump_relative * np.maximum(face_depth.min(axis=1), 1e-8)
        if geometry_filter == "source_free_space_v1":
            keep = ~source_free_space_rejections(reference, points, faces, ~continuous)
            faces, continuous = faces[keep], continuous[keep]
        same_class = (labels[faces] == labels[faces[:, :1]]).all(axis=1)
        face_supported = continuous & same_class & (labels[faces[:, 0]] != UNKNOWN)
        face_labels = np.where(face_supported, labels[faces[:, 0]], UNKNOWN).astype(np.uint8)
        colors = reference.rgb[grid_y, grid_x].reshape(-1, 3).astype(np.float32) / 255
        pieces.append((vertices, faces + offset, colors, labels, face_labels, face_supported,
                       np.full(len(pixels), source_index, dtype=np.int32), pixels))
        offset += len(pixels)
    arrays = [np.concatenate([piece[index] for piece in pieces]) for index in range(8)]
    return SceneEvidence(*arrays, tuple(reference.source_id for reference in references))


def rasterize(scene: SceneEvidence, camera: Camera, colors: np.ndarray | None = None) -> Render:
    if colors is None:
        colors = scene.colors
    if colors.shape != scene.colors.shape or not np.isfinite(colors).all():
        raise ValueError("Invalid vertex colors")
    points = (scene.vertices - camera.camera_to_world[:3, 3]) @ camera.camera_to_world[:3, :3]
    projected = points @ camera.intrinsics.T
    screen = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
    shape = (camera.height, camera.width)
    depths = np.full(shape, np.inf, dtype=np.float64)
    face_image = np.full(shape, -1, dtype=np.int32)
    weights_image = np.zeros((*shape, 3), dtype=np.float32)
    for face_index, face in enumerate(scene.faces):
        triangle_depth = points[face, 2]
        if (triangle_depth <= 1e-4).any():
            continue
        triangle = screen[face]
        minimum = np.maximum(np.ceil(triangle.min(axis=0)).astype(int), [0, 0])
        maximum = np.minimum(np.floor(triangle.max(axis=0)).astype(int), [camera.width - 1, camera.height - 1])
        if (maximum < minimum).any():
            continue
        first, second, third = triangle
        denominator = (second[1] - third[1]) * (first[0] - third[0]) + (third[0] - second[0]) * (first[1] - third[1])
        if abs(denominator) < 1e-10:
            continue
        grid_y, grid_x = np.mgrid[minimum[1]:maximum[1] + 1, minimum[0]:maximum[0] + 1]
        first_weight = ((second[1] - third[1]) * (grid_x - third[0]) + (third[0] - second[0]) * (grid_y - third[1])) / denominator
        second_weight = ((third[1] - first[1]) * (grid_x - third[0]) + (first[0] - third[0]) * (grid_y - third[1])) / denominator
        screen_weights = np.stack([first_weight, second_weight, 1 - first_weight - second_weight], axis=-1)
        inside = (screen_weights >= -1e-7).all(axis=-1)
        perspective = screen_weights / triangle_depth
        inverse_depth = perspective.sum(axis=-1)
        current_depth = 1 / np.maximum(inverse_depth, 1e-12)
        update = inside & (current_depth < depths[grid_y, grid_x] - 1e-8)
        update_y, update_x = grid_y[update], grid_x[update]
        depths[update_y, update_x] = current_depth[update]
        face_image[update_y, update_x] = face_index
        weights_image[update_y, update_x] = perspective[update] / inverse_depth[update, None]
    geometry_mask = face_image >= 0
    labels = np.full(shape, UNKNOWN, dtype=np.uint8)
    support = np.zeros(shape, dtype=bool)
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    visible_faces = face_image[geometry_mask]
    labels[geometry_mask] = scene.face_labels[visible_faces]
    support[geometry_mask] = scene.face_supported[visible_faces]
    rgb[geometry_mask] = (colors[scene.faces[visible_faces]] * weights_image[geometry_mask, :, None]).sum(axis=1)
    return Render(rgb, labels, depths, face_image, weights_image, geometry_mask, support)


def surface_edges(scene: SceneEvidence, same_class: bool = True) -> np.ndarray:
    faces = scene.faces
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    if same_class:
        labels = scene.vertex_labels[edges]
        edges = edges[(labels[:, 0] == labels[:, 1]) & (labels[:, 0] != UNKNOWN)]
    return edges


def recolor(scene: SceneEvidence, render: Render, colors: np.ndarray) -> np.ndarray:
    if colors.shape != scene.colors.shape or not np.isfinite(colors).all():
        raise ValueError("Invalid vertex colors")
    result = render.rgb.copy()
    visible = render.geometry_mask
    vertices = scene.faces[render.face_index[visible]]
    result[visible] += ((colors[vertices] - scene.colors[vertices]) * render.barycentric[visible, :, None]).sum(axis=1)
    return np.clip(result, 0, 1)
