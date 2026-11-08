"""Estimate cameras from three official demo photos with aerial MASt3R.

All three RGB images are used for camera estimation. Image 00616 is excluded
from subsequent NeRF losses, so evaluation is a held-out rendering check with
transductive camera calibration, not a fully unseen-image benchmark.
"""
import hashlib
import json
import pathlib
import sys
import time

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from safetensors.torch import load_file
from safetensors import safe_open

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/aerial-megadepth/mast3r'))


def prepare_inputs():
    ids = ['00572', '00563', '00616']
    scene = 't04_v13_s00_r01_VaryingAltitudes_WACV_test_A10'
    folder = ROOT / 'inputs/multiview_a10'
    folder.mkdir(parents=True, exist_ok=True)
    expected = ['db3ba053586711d90aa82bfa9160a5f49fe0bad95d0fce54eb5903e91e9217bf',
                '04ccd3354ee9ea3713c516afa08fcd7e3d64a02b4428eb79527c8d9716b81118',
                '79d492972109c2bba6f538f4f970fed4a9b34724e324eb7cb64209eb2e423a1c']
    provenance = []
    board = Image.new('RGB', (768, 282), 'white')
    for i, (name, digest) in enumerate(zip(ids, expected)):
        source = ROOT / 'third_party/aerial-megadepth/assets' / scene / (name + '.jpg')
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f'Source photo SHA256 mismatch: {name}')
        with Image.open(source) as opened:
            source_size = list(opened.size)
            image = ImageOps.exif_transpose(opened).convert('RGB')
        image = ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS)
        for size in [512, 256]:
            resized = image.resize((size, size), Image.Resampling.LANCZOS)
            path = folder / f'{name}_{size}.png'
            # Existing inputs are immutable evidence for a completed run.
            if path.exists():
                with Image.open(path) as old:
                    if old.convert('RGB').tobytes() != resized.tobytes():
                        raise RuntimeError(f'Existing prepared image differs: {path}')
            else:
                resized.save(path)
        board.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (i * 256, 0))
        ImageDraw.Draw(board).text((i * 256 + 6, 262), name, fill='black')
        provenance.append({'id': name, 'source_path': str(source.relative_to(ROOT)),
            'sha256': actual, 'source_size': source_size,
            'processing': 'EXIF transpose; ImageOps.fit center square to 512 LANCZOS; resize 512 to 256 LANCZOS',
            'source_url': f'https://raw.githubusercontent.com/kvuong2711/aerial-megadepth/main/assets/{scene}/{name}.jpg'})
    board.save(folder / 'inputs.png')
    (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


def main():
    prepare_inputs()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    folder = ROOT / 'weights/aerial_mast3r'
    config = json.loads((folder / 'config.json').read_text())
    print('Constructing aerial MASt3R...', flush=True)
    model = AsymmetricMASt3R(**config)
    state = load_file(str(folder / 'model.safetensors'))
    with safe_open(str(folder / 'model.safetensors'), framework='pt', device='cpu') as checkpoint:
        for alias, original in (checkpoint.metadata() or {}).items():
            if original in state and alias in model.state_dict():
                state[alias] = state[original]
    model.load_state_dict(state, strict=True)
    del state
    model = model.cuda().eval()
    ids = ['00572', '00563', '00616']
    inputs = ROOT / 'inputs/multiview_a10'
    images = load_images([str(inputs / (name + '_512.png')) for name in ids], size=512, square_ok=True)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    out = ROOT / 'outputs/multiview_a10'
    out.mkdir(exist_ok=True)
    start = time.monotonic()
    predictions = inference(pairs, model, 'cuda', batch_size=1)
    torch.save(predictions, out / 'pair_predictions.pt')
    del model
    torch.cuda.empty_cache()
    print('Aligning 3 cameras and dense point maps...', flush=True)
    scene = global_aligner(predictions, device='cuda', mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
    loss = scene.compute_global_alignment(init='mst', niter=400, schedule='cosine', lr=0.01)
    with torch.no_grad():
        poses = scene.get_im_poses().detach().cpu()
        intrinsics = scene.get_intrinsics().detach().cpu()
        points = torch.stack(scene.get_pts3d()).detach().cpu()
        depths = torch.stack(scene.get_depthmaps()).detach().cpu()
        conf = torch.stack(list(scene.im_conf)).detach().cpu()
    # Put camera 0 at the origin in OpenCV axes; normalize by median reliable
    # depth in image 0. This frame and scale are identical for both experiments.
    inverse = torch.linalg.inv(poses[0])
    camera_poses = inverse[None] @ poses
    reliable = (conf[0] > 3) & torch.isfinite(depths[0]) & (depths[0] > 0)
    if reliable.sum() < 1000:
        raise RuntimeError('Insufficient reliable geometry in reference image')
    scale = torch.median(depths[0][reliable])
    camera_poses[:, :3, 3] /= scale
    points = (points @ inverse[:3, :3].T + inverse[:3, 3]) / scale
    depths = depths / scale
    # Place the observed scene around zero, leaving camera 0 at z=-1.
    center = torch.tensor([0., 0., 1.])
    points -= center
    camera_poses[:, :3, 3] -= center
    # OpenCV rays use +z forwards and +y down. Convert to OpenGL camera axes.
    convert = torch.diag(torch.tensor([1., -1., -1., 1.]))
    opengl_poses = camera_poses @ convert
    assert torch.isfinite(opengl_poses).all() and torch.isfinite(intrinsics).all()
    quality = []
    for i, name in enumerate(ids):
        rotation = camera_poses[i, :3, :3]
        cos = ((torch.trace(rotation) - 1) / 2).clamp(-1, 1)
        quality.append({'id': name, 'focal_512': float(intrinsics[i, 0, 0]),
                        'fov_deg': float(2 * torch.atan(256 / intrinsics[i, 0, 0]) * 180 / torch.pi),
                        'median_confidence': float(conf[i].median()),
                        'fraction_confidence_gt3': float((conf[i] > 3).float().mean()),
                        'rotation_from_reference_deg': float(torch.acos(cos) * 180 / torch.pi),
                        'camera_center': camera_poses[i, :3, 3].tolist()})
    metadata = {'ids': ids, 'train_indices_multi': [0, 1], 'train_indices_single': [0],
                'evaluation_indices': [2], 'camera_calibration_uses_evaluation_image': True,
                'camera_model': 'estimated pinhole; no ground truth intrinsics or extrinsics',
                'checkpoint_repository': 'kvuong2711/checkpoint-aerial-mast3r',
                'checkpoint_revision': '0cf76e1da5c9b2bb7fb9d17cce03e0e6582c034e',
                'global_alignment_loss': float(loss), 'quality': quality,
                'normalization_depth_scale': float(scale), 'elapsed_s': time.monotonic() - start}
    torch.save({'ids': ids, 'poses': opengl_poses, 'intrinsics_512': intrinsics,
                'depths': depths, 'confidence': conf, 'points': points, 'metadata': metadata}, out / 'cameras.pt')
    (out / 'cameras.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # Save a small confidence-filtered colored point cloud for inspection.
    colors = torch.stack([((im['img'][0].permute(1, 2, 0) + 1) / 2).cpu() for im in images])
    valid = (conf > 3) & torch.isfinite(points).all(-1) & (points.norm(dim=-1) < 10)
    valid[:, ::2] = False
    valid[:, :, ::2] = False
    xyz = points[valid].numpy(); rgb = (colors[valid].numpy().clip(0, 1) * 255).astype(np.uint8)
    with (out / 'estimated_geometry.ply').open('w', encoding='ascii') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(xyz, rgb):
            stream.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
