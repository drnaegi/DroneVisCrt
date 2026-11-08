"""Differentiable unbounded hash-grid NeRF using standard PyTorch CUDA ops.

This implementation replaces tiny-cuda-nn/nerfacc on Windows. It uses spatial
contraction, separate density/albedo MLPs, Magic3D density initialization,
inverse-depth sampling and alpha compositing. It is not binary-compatible with
tiny-cuda-nn scene checkpoints.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def camera_pose(elevation, azimuth, distance=1.0, device='cuda'):
    e, a = math.radians(elevation), math.radians(azimuth)
    position = torch.tensor([distance * math.cos(e) * math.cos(a),
                             distance * math.cos(e) * math.sin(a),
                             distance * math.sin(e)], device=device)
    forward = F.normalize(-position, dim=0)
    right = F.normalize(torch.linalg.cross(forward, position.new_tensor([0., 0., 1.])), dim=0)
    up = F.normalize(torch.linalg.cross(right, forward), dim=0)
    pose = torch.eye(4, device=device)
    pose[:3, :3] = torch.stack([right, up, -forward], dim=-1)
    pose[:3, 3] = position
    return pose


def rays(pose, resolution, fov):
    p = torch.arange(resolution, device=pose.device, dtype=torch.float32) + 0.5
    y, x = torch.meshgrid(p, p, indexing='ij')
    focal = resolution / (2 * math.tan(math.radians(fov) / 2))
    directions = torch.stack([(x - resolution / 2) / focal,
                               -(y - resolution / 2) / focal, -torch.ones_like(x)], -1)
    directions = F.normalize(directions.reshape(-1, 3) @ pose[:3, :3].T, dim=-1)
    return pose[:3, 3].expand_as(directions), directions


class HashGrid(nn.Module):
    def __init__(self, levels=16, log2_size=19):
        super().__init__()
        self.levels, self.table_size = levels, 1 << log2_size
        self.table = nn.Parameter(torch.empty(levels * self.table_size, 2).uniform_(-1e-4, 1e-4))
        self.register_buffer('scales', torch.tensor([16 * 1.447269237440378 ** i - 1 for i in range(levels)]))
        self.register_buffer('offsets', torch.arange(levels, dtype=torch.int64) * self.table_size)
        self.register_buffer('corners', torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=torch.int64))

    def forward(self, x):
        positions = x[:, None, :] * self.scales[None, :, None] + 0.5
        floors = torch.floor(positions)
        frac = positions - floors
        vertices = floors.to(torch.int64)[:, :, None, :] + self.corners[None, None]
        h = (vertices[..., 0] ^ (vertices[..., 1] * 2654435761) ^ (vertices[..., 2] * 805459861))
        indices = (h & (self.table_size - 1)) + self.offsets[None, :, None]
        features = F.embedding(indices, self.table)
        weights = torch.where(self.corners[None, None].bool(), frac[:, :, None], 1 - frac[:, :, None]).prod(-1)
        return (features * weights[..., None]).sum(2).flatten(1)


class NeRF(nn.Module):
    def __init__(self, log2_size=19):
        super().__init__()
        self.encoding = HashGrid(log2_size=log2_size)
        def mlp(out):
            return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU(),
                                 nn.Linear(32, 32, bias=False), nn.ReLU(), nn.Linear(32, out, bias=False))
        self.density = mlp(1)
        self.albedo = mlp(3)

    def forward(self, points):
        x = points / 2.0  # upstream radius=2 bounding box
        magnitude = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-6)
        contracted = torch.where(magnitude > 1, (2 - magnitude.reciprocal()) * (x / magnitude), x)
        features = self.encoding(contracted / 4 + 0.5)
        bias = 5 * (1 - torch.linalg.vector_norm(points, dim=-1, keepdim=True) / 0.5)
        density = F.softplus(self.density(features) + bias)
        return density.squeeze(-1), torch.sigmoid(self.albedo(features))


def compose(field, origins, directions, edges, background):
    mids = (edges[..., 1:] + edges[..., :-1]) * 0.5
    points = origins[:, None] + directions[:, None] * mids[..., None]
    sigma, color = field(points.reshape(-1, 3))
    sigma = sigma.reshape_as(mids)
    color = color.reshape(*mids.shape, 3)
    optical = sigma * (edges[..., 1:] - edges[..., :-1])
    alpha = -torch.expm1(-optical)
    transmittance = torch.exp(-torch.cat([torch.zeros_like(optical[:, :1]), optical.cumsum(-1)[:, :-1]], -1))
    weights = alpha * transmittance
    opacity = weights.sum(-1)
    rgb = (weights[..., None] * color).sum(-2) + (1 - opacity[..., None]) * background
    depth = (weights * mids).sum(-1)
    return rgb, depth, opacity, weights


def sample_edges(field, origins, directions, samples=64, near=0.5, far=1000., randomized=True):
    """Deterministic per-step samples can be reused in image-gradient replay."""
    n = len(origins)
    u = torch.linspace(0, 1, samples + 1, device=origins.device).expand(n, -1)
    if randomized:
        u = u.clone()
        u[:, 1:-1] += (torch.rand_like(u[:, 1:-1]) - 0.5) / samples
    return 1 / ((1 - u) / near + u / far)


def render(field, origins, directions, edges, background, chunk=512):
    results = []
    for start in range(0, len(origins), chunk):
        results.append(compose(field, origins[start:start+chunk], directions[start:start+chunk],
                               edges[start:start+chunk], background))
    return tuple(torch.cat([result[i] for result in results]) for i in range(3))
