"""Verify final artifacts and measure reference-view fitting after reload."""
import hashlib
import json
import math
import pathlib

import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from portable_nerf import NeRF, camera_pose, rays, sample_edges, render

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'aerialmegadepth_00572'


def main():
    print('Checking checkpoint...', flush=True)
    torch.set_num_threads(8)
    status = json.loads((OUT / 'status.json').read_text())
    assert status['status'] == 'complete', status
    checkpoint = torch.load(OUT / 'last.pt', map_location='cpu', weights_only=False)
    config = checkpoint['config']
    assert checkpoint['step'] == config['steps'] == 4000
    state = checkpoint['field']
    assert all(torch.isfinite(value).all() for value in state.values())
    field = NeRF(log2_size=config['hash_log2_size']).cuda().eval()
    field.load_state_dict(state, strict=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    transform = json.loads((OUT / 'transforms.json').read_text())
    assert len(transform['frames']) == 120
    images = sorted((OUT / 'renders').glob('*.png'))
    assert len(images) == 120
    hashes = set()
    print('Checking PNGs and camera matrices...', flush=True)
    for record in transform['frames']:
        path = OUT / record['file_path']
        assert path.is_file()
        assert Image.open(path).size == (256, 256)
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        pose = np.asarray(record['transform_matrix'])
        assert pose.shape == (4, 4) and np.isfinite(pose).all()
        p = record['transform_matrix']
        assert all(abs(a - b) < 1e-6 for a, b in zip(p[3], [0, 0, 0, 1]))
        assert all(abs(sum(p[k][i] * p[k][j] for k in range(3)) - int(i == j)) < 1e-6
                   for i in range(3) for j in range(3))
        det = (p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
               - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0])
               + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]))
        assert abs(det - 1) < 1e-6
        assert record['step'] == 4000
    assert len(hashes) == 120
    print('Checking video...', flush=True)
    with imageio.get_reader(str(OUT / 'orbit.mp4')) as reader:
        assert reader.count_frames() == 120
        assert reader.get_data(0).shape == (256, 256, 3)
        fps = reader.get_meta_data()['fps']
        assert abs(fps - 24) < 1e-6
    print('Re-rendering reference camera...', flush=True)
    with torch.no_grad():
        pose = camera_pose(config['elevation'], 0.)
        ro, rd = rays(pose, 256, config['fov'])
        edges = sample_edges(field, ro, rd, config['samples'], randomized=False)
        rgb, depth, opacity = render(field, ro, rd, edges, torch.full((3,), .5, device='cuda'), config['ray_chunk'])
    prediction = rgb.reshape(256, 256, 3).cpu().numpy()
    target = np.asarray(Image.open(ROOT / config['image']).convert('RGB')).astype(np.float32) / 255
    mse = float(np.mean((prediction - target) ** 2))
    # SSIM with a 7x7 uniform window, sample covariance, and valid borders,
    # matching the default skimage convention without loading a second OpenMP.
    a = torch.from_numpy(target).permute(2, 0, 1)[None]
    b = torch.from_numpy(prediction).permute(2, 0, 1)[None]
    mean_a, mean_b = F.avg_pool2d(a, 7, 1), F.avg_pool2d(b, 7, 1)
    var_a = (F.avg_pool2d(a * a, 7, 1) - mean_a.square()) * (49 / 48)
    var_b = (F.avg_pool2d(b * b, 7, 1) - mean_b.square()) * (49 / 48)
    covariance = (F.avg_pool2d(a * b, 7, 1) - mean_a * mean_b) * (49 / 48)
    ssim = float((((2 * mean_a * mean_b + .01**2) * (2 * covariance + .03**2)) /
                  ((mean_a.square() + mean_b.square() + .01**2) * (var_a + var_b + .03**2))).mean())
    first_png = np.asarray(Image.open(OUT / 'renders' / '0000.png')).astype(np.int16)
    quantized = np.clip(np.rint(prediction * 255), 0, 255).astype(np.int16)
    reload_error = int(np.abs(first_png - quantized).max())
    assert reload_error <= 1, reload_error
    metrics = {}
    for line in (OUT / 'metrics.jsonl').read_text().splitlines():
        record = json.loads(line)
        metrics[record['step']] = record
    assert set(metrics) == set(range(1, 4001))
    assert all(math.isfinite(r['loss_sds']) and math.isfinite(r['sds_image_gradient_norm'])
               and r['sds_image_gradient_norm'] > 0 for r in metrics.values())
    provenance = json.loads((ROOT / 'source_provenance.json').read_text())
    for name, expected in provenance['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    result = {
        'status': 'PASS', 'training_steps': 4000, 'valid_unique_training_steps': len(metrics),
        'checkpoint_sha256': hashlib.file_digest((OUT / 'last.pt').open('rb'), 'sha256').hexdigest(),
        'png_count': len(images), 'distinct_png_count': len(hashes), 'video_frames': 120,
        'video_fps': fps, 'render_resolution': [256, 256],
        'camera_matrices': 'valid OpenGL c2w rotations, all frames at step 4000',
        'reload_max_pixel_difference_8bit': reload_error,
        'reference_view_only': {'mse': mse, 'psnr': -10 * math.log10(mse),
                                'ssim': ssim, 'ssim_window': '7x7 uniform, sample covariance, valid borders'},
        'peak_training_vram_GiB': max(r['peak_vram_GiB'] for r in metrics.values()),
        'all_training_sds_image_gradients_finite_nonzero': True,
        'upstream_guidance_and_camera_source_hashes_unchanged': True,
        'novel_view_quality': 'No ground truth; must be judged visually. Reference metrics do not establish novel-view accuracy.',
        'paper_reproduction': False,
    }
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
