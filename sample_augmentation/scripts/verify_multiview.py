"""Verify saved multiview experiment artifacts and checkpoint re-rendering."""
import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from portable_nerf import NeRF
from run_multiview_experiment import OUT, ROOT, calibrated_rays, render_camera, write_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    calibration = torch.load(OUT / 'cameras.pt', map_location='cpu', weights_only=False)
    common = json.loads((OUT / 'experiment_config.json').read_text())
    results = json.loads((OUT / 'results.json').read_text())
    transforms = json.loads((OUT / 'transforms.json').read_text())
    assert len(transforms['frames']) == 120
    report = {'status': 'PASS', 'steps_per_run': common['steps'], 'runs': {},
              'limitations': results['calibration_caveat'], 'sampling_coverage': {}}
    report['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'Pillow', 'roma', 'timm', 'transformers', 'diffusers', 'safetensors', 'imageio']}}
    for i, name in enumerate(calibration['ids']):
        pose = calibration['poses'][i]
        assert torch.allclose(pose[:3, :3].T @ pose[:3, :3], torch.eye(3), atol=1e-4)
        assert float(torch.linalg.det(pose[:3, :3])) > .999
        _, directions = calibrated_rays(pose, calibration['intrinsics_512'][i], 512)
        cosine = (directions * -pose[:3, 2]).sum(-1)
        ranges = calibration['depths'][i].flatten() / cosine
        reliable = (calibration['confidence'][i].flatten() > 3) & torch.isfinite(ranges)
        values = ranges[reliable]
        fraction = float(((values > common['near']) & (values < common['far'])).float().mean())
        report['sampling_coverage'][name] = {'reliable_depth_fraction_inside_range': fraction,
            'range_quantiles_0_50_100': torch.quantile(values, torch.tensor([0., .5, 1.])).tolist()}
        assert fraction > .99
    for mode, refs in [('single', [0]), ('multi', [0, 1])]:
        folder = OUT / mode
        config = json.loads((folder / 'config.json').read_text())
        assert config['reference_indices'] == refs and 2 not in refs
        assert {key: config[key] for key in common} == common
        status = json.loads((folder / 'status.json').read_text())
        assert status['status'] == 'complete' and status['step'] == common['steps']
        metrics = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert [row['step'] for row in metrics] == list(range(1, common['steps'] + 1))
        assert all(math.isfinite(value) for row in metrics for value in row.values()
                   if isinstance(value, (float, int)))
        sds_steps = sum(row['sds_image_gradient_norm'] > 0 for row in metrics)
        expected_sds = len(range(common['warmup_steps'], common['steps'], common['sds_every']))
        assert sds_steps == expected_sds
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['step'] == common['steps'] and saved['config'] == config
        assert all(torch.isfinite(value).all() for value in saved['field'].values())
        field = NeRF(log2_size=config['hash_log2_size']).cuda().eval()
        field.load_state_dict(saved['field'], strict=True)
        with torch.no_grad():
            rgb, _, _ = render_camera(field, calibration['poses'][2].cuda(),
                                      calibration['intrinsics_512'][2].cuda(), config)
        reloaded = (rgb.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
        exported = np.asarray(Image.open(folder / 'camera_00616.png'))
        difference = int(np.abs(reloaded.astype(np.int16) - exported.astype(np.int16)).max())
        assert difference == 0
        frames = sorted((folder / 'renders').glob('*.png'))
        assert len(frames) == 120
        assert all(Image.open(path).size == (256, 256) for path in frames)
        distinct = len({sha256(path) for path in frames})
        assert distinct >= 61  # Symmetric forward/back interpolation repeats frames.
        report['runs'][mode] = {'reference_indices': refs, 'completed_steps': len(metrics),
            'nonzero_sds_gradient_steps': sds_steps, 'exported_frames': len(frames),
            'distinct_frames': distinct, 'checkpoint_reload_max_pixel_difference': difference,
            'training_elapsed_s': metrics[-1]['elapsed_s'],
            'peak_allocated_vram_GiB': max(row['peak_vram_GiB'] for row in metrics)}
        del field, saved
        torch.cuda.empty_cache()
    with imageio.get_reader(str(OUT / 'comparison_path.mp4')) as reader:
        assert reader.count_frames() == 120
        assert reader.get_meta_data()['size'] == (512, 256)
    report['video'] = {'frames': 120, 'size': [512, 256], 'fps': 24}
    report['heldout_psnr_delta_db'] = results['heldout_psnr_delta_db']
    report['heldout_ssim_delta'] = results['heldout_ssim_delta']
    scripts = ['prepare_multiview.py', 'run_multiview_experiment.py',
               'portable_nerf.py', 'official_guidance.py', 'verify_multiview.py']
    report['script_sha256'] = {name: sha256(ROOT / 'scripts' / name) for name in scripts}
    write_json(OUT / 'verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
