"""Check actual CUDA rendering, camera conditioning and replay gradients."""
import json
import pathlib
import sys
import torch
from portable_nerf import NeRF, camera_pose, rays, sample_edges, compose, render

torch.manual_seed(0)
torch.set_num_threads(4)
field = NeRF(log2_size=10).cuda()
pose = camera_pose(20, 0)
origins, directions = rays(pose, 8, 55)
origins, directions = origins[:16], directions[:16]
edges = sample_edges(field, origins, directions, 16, randomized=False)
background = torch.tensor([0.1, 0.2, 0.3], device='cuda')
gradient = torch.randn(16, 3, device='cuda')
rgb, depth, opacity, weights = compose(field, origins, directions, edges, background)
assert torch.isfinite(rgb).all() and (rgb >= 0).all() and (rgb <= 1).all()
assert torch.allclose(opacity, weights.sum(-1))
(rgb * gradient).sum().backward()
expected = {name: p.grad.clone() for name, p in field.named_parameters()}
field.zero_grad(set_to_none=True)
for start in range(0, len(origins), 4):
    image, _, _, _ = compose(field, origins[start:start+4], directions[start:start+4], edges[start:start+4], background)
    (image * gradient[start:start+4]).sum().backward()
max_error = 0.
for name, p in field.named_parameters():
    torch.testing.assert_close(p.grad, expected[name], rtol=1e-4, atol=1e-7)
    max_error = max(max_error, float((p.grad - expected[name]).abs().max()))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ZeroNVS/zeronvs_diffusion/zero123'))
from ldm.data.common import compute_T
from omegaconf import OmegaConf
config = OmegaConf.create({'params': {'mode': '7dof_quantile_scale'}})
encoded = compute_T(config, None, {'target_cam2world': pose[None], 'cond_cam2world': pose[None],
                                  'fov_deg': torch.tensor([55.], device='cuda')}, precomputed_scale=.7)
torch.testing.assert_close(encoded[0, :16].reshape(4, 4), torch.eye(4, device='cuda'), atol=1e-6, rtol=1e-6)
assert encoded.shape == (1, 19)
result = {'status': 'PASS', 'cuda_device': torch.cuda.get_device_name(0),
          'torch': torch.__version__, 'replay_gradient_max_abs_error': max_error,
          'official_camera_embedding_shape': list(encoded.shape)}
print(json.dumps(result, indent=2))
