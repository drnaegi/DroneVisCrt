"""Controlled one-photo vs two-photo ZeroNVS-guided NeRF experiment.

Shared: estimated cameras, architecture, seed, optimizer, ray/step budgets,
depth-prior weight and SDS schedule. Changed: training reference pool [0] vs
[0,1]. Photo 2 supplies camera calibration but never NeRF RGB/depth/SDS input.
This is an experimental multi-reference extension, not official ZeroNVS code.
"""
import argparse
import json
import math
import pathlib
import random
import time

import torch
import torch.nn.functional as F
import numpy as np
import roma
from PIL import Image, ImageDraw
import imageio.v2 as imageio

from portable_nerf import NeRF, compose, render

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/multiview_a10'


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def calibrated_rays(pose, intrinsics, resolution):
    p = torch.arange(resolution, device=pose.device, dtype=torch.float32) + .5
    y, x = torch.meshgrid(p, p, indexing='ij')
    k = intrinsics * (resolution / 512)
    directions = torch.stack([(x - k[0, 2]) / k[0, 0], -(y - k[1, 2]) / k[1, 1], -torch.ones_like(x)], -1)
    directions = F.normalize(directions.reshape(-1, 3) @ pose[:3, :3].T, dim=-1)
    return pose[:3, 3].expand_as(directions), directions


def sampling_edges(origins, count, randomized=True):
    u = torch.linspace(0, 1, count + 1, device=origins.device).expand(len(origins), -1)
    if randomized:
        u = u.clone()
        u[:, 1:-1] += (torch.rand_like(u[:, 1:-1]) - .5) / count
    # Reserve half the intervals for distant aerial geometry. Pure inverse
    # depth places almost all samples near the ground camera and cannot
    # resolve the aerial reference around range 7 in this normalization.
    near, split, far = .15, 2., 12.
    lower = 1 / ((1 - 2 * u) / near + 2 * u / split).clamp_min(1e-6)
    upper = split + (2 * u - 1) * (far - split)
    return torch.where(u <= .5, lower, upper)


def interpolate_pose(first, second, fraction):
    q = roma.rotmat_to_unitquat(torch.stack([first[:3, :3], second[:3, :3]]))
    rotation = roma.unitquat_to_rotmat(roma.unitquat_slerp(q[0], q[1], first.new_tensor(fraction)))
    pose = torch.eye(4, device=first.device)
    pose[:3, :3] = rotation
    pose[:3, 3] = first[:3, 3].lerp(second[:3, 3], fraction)
    return pose


def array_image(value):
    return (value.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)


@torch.no_grad()
def render_camera(field, pose, intrinsics, config, resolution=256):
    origins, directions = calibrated_rays(pose, intrinsics, resolution)
    edges = sampling_edges(origins, config['samples'], randomized=False)
    rgb, depth, opacity = render(field, origins, directions, edges, origins.new_full((3,), .5), config['ray_chunk'])
    return rgb.reshape(resolution, resolution, 3), depth.reshape(resolution, resolution), opacity.reshape(resolution, resolution)


def ssim(first, second):
    a, b = (x.permute(2, 0, 1)[None].float().cpu() for x in (first, second))
    ma, mb = F.avg_pool2d(a, 7, 1), F.avg_pool2d(b, 7, 1)
    va = (F.avg_pool2d(a * a, 7, 1) - ma.square()) * (49 / 48)
    vb = (F.avg_pool2d(b * b, 7, 1) - mb.square()) * (49 / 48)
    cov = (F.avg_pool2d(a * b, 7, 1) - ma * mb) * (49 / 48)
    return float((((2 * ma * mb + .01**2) * (2 * cov + .03**2)) /
                  ((ma.square() + mb.square() + .01**2) * (va + vb + .03**2))).mean())


def train(mode, calibration, teacher, embeddings, config):
    folder = OUT / mode
    folder.mkdir(exist_ok=True)
    if (folder / 'last.pt').exists():
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if saved['step'] == config['steps']:
            if any(saved['config'].get(key) != value for key, value in config.items()):
                raise RuntimeError('Completed checkpoint configuration differs from requested experiment')
            print('Already trained:', mode, flush=True)
            return
        raise RuntimeError('Partial checkpoint exists; refusing to silently overwrite')
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    generator = torch.Generator(device='cuda').manual_seed(config['seed'])
    poses = calibration['poses'].cuda()
    intrinsics = calibration['intrinsics_512'].cuda()
    indices = [0] if mode == 'single' else [0, 1]
    targets = torch.stack([torch.tensor(np.asarray(Image.open(ROOT / f'inputs/multiview_a10/{name}_256.png')).copy(), device='cuda').float() / 255
                           for name in calibration['ids']])
    all_rays = [calibrated_rays(poses[i], intrinsics[i], 256) for i in range(3)]
    field = NeRF(log2_size=config['hash_log2_size']).cuda()
    optimizer = torch.optim.Adam(field.parameters(), lr=config['lr'], betas=(.9, .99), eps=1e-8)
    # Predicted depths supervise only the training views. No held-out RGB,
    # pseudo-depth or reference embedding enters the optimization.
    depths = F.interpolate(calibration['depths'][:, None].cuda(), (256, 256), mode='nearest')[:, 0]
    confidences = F.interpolate(calibration['confidence'][:, None].cuda(), (256, 256), mode='nearest')[:, 0]
    depth_ranges = []
    for i in range(3):
        _, directions = all_rays[i]
        forward = -poses[i, :3, 2]
        cosine = (directions * forward).sum(-1).clamp_min(.1)
        depth_ranges.append(depths[i].flatten() / cosine)
    start = time.monotonic()
    config = dict(config, mode=mode, reference_indices=indices)
    write_json(folder / 'config.json', config)
    status = {'status': 'training', 'mode': mode, 'step': 0, 'total_steps': config['steps']}
    write_json(folder / 'status.json', status)
    with (folder / 'metrics.jsonl').open('w', encoding='utf-8') as log:
        for step in range(config['steps']):
            optimizer.zero_grad(set_to_none=True)
            # Same total RGB-ray budget; spread across training images.
            reference_mse = 0.
            depth_loss_value = 0.
            for ref in indices:
                n = config['reference_rays'] // len(indices)
                pixels = torch.randint(0, 256 * 256, (n,), generator=generator, device='cuda')
                ro, rd = (x[pixels] for x in all_rays[ref])
                edges = sampling_edges(ro, config['samples'])
                gt = targets[ref].reshape(-1, 3)[pixels]
                gt_depth = depth_ranges[ref][pixels]
                valid = (confidences[ref].flatten()[pixels] > 3) & (gt_depth > .15) & (gt_depth < 12)
                bg = torch.full((3,), .5, device='cuda')
                rgb, depth, opacity, weights = compose(field, ro, rd, edges, bg)
                mse = (rgb - gt).square().mean()
                if valid.any():
                    predicted_depth = depth / opacity.clamp_min(1e-4)
                    dl = F.smooth_l1_loss(predicted_depth[valid].clamp_min(.01).log(), gt_depth[valid].log())
                else:
                    dl = depth.sum() * 0
                loss = (config['lambda_rgb'] * mse + config['lambda_depth'] * dl) / len(indices)
                loss.backward()
                reference_mse += float(mse.detach()) / len(indices)
                depth_loss_value += float(dl.detach()) / len(indices)
            sds_value = gradient_norm = 0.
            if step >= config['warmup_steps'] and step % config['sds_every'] == 0:
                # Target-camera distribution is identical in both runs. The
                # reference chosen for SDS is restricted to each run's inputs.
                fraction = random.random()
                target_pose = interpolate_pose(poses[0], poses[1], fraction)
                target_k = intrinsics[0].lerp(intrinsics[1], fraction)
                ref = min(indices, key=lambda i: float((poses[i, :3, 3] - target_pose[:3, 3]).square().sum()))
                teacher.c_crossattn, teacher.c_concat = embeddings[ref]
                teacher.cfg.precomputed_scale = float(depths[ref][confidences[ref] > 3].quantile(.2))
                virtual_step = int(step / config['steps'] * 4000)
                teacher.update_step(0, virtual_step)
                resolution = config['sds_resolution']
                ro, rd = calibrated_rays(target_pose, target_k, resolution)
                edges = sampling_edges(ro, config['samples'])
                bg = torch.full((3,), .5, device='cuda')
                with torch.no_grad():
                    rgb, _, _ = render(field, ro, rd, edges, bg, config['ray_chunk'])
                image = rgb.reshape(1, resolution, resolution, 3).detach().requires_grad_(True)
                relative_rot = poses[ref, :3, :3].T @ target_pose[:3, :3]
                angle = torch.acos(((relative_rot.trace() - 1) / 2).clamp(-1, 1)) * (180 / math.pi)
                fov = 2 * torch.atan(256 / target_k[0, 0]) * (180 / math.pi)
                camera = {'elevation': ro.new_tensor([0.]), 'azimuth': angle[None],
                          'camera_distances': target_pose[:3, 3].norm()[None],
                          'canonical_c2w': poses[ref:ref+1], 'c2w': target_pose[None],
                          'fov_deg': fov[None], 'aux_c2ws': []}
                result = teacher(image, camera)
                gradient = torch.autograd.grad(config['lambda_sds'] * result['loss_sds'], image)[0].detach().reshape(-1, 3)
                if not torch.isfinite(gradient).all():
                    raise RuntimeError('Nonfinite SDS image gradient')
                sds_value = float(result['loss_sds'].detach())
                gradient_norm = float(gradient.norm())
                del result, image
                for first in range(0, len(ro), config['ray_chunk']):
                    last = first + config['ray_chunk']
                    color, _, _, _ = compose(field, ro[first:last], rd[first:last], edges[first:last], bg)
                    (color * gradient[first:last]).sum().backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in field.parameters()):
                raise RuntimeError('Nonfinite NeRF gradient')
            optimizer.step()
            metric = {'step': step + 1, 'mode': mode, 'reference_mse': reference_mse,
                      'reference_psnr_sampled': -10 * math.log10(max(reference_mse, 1e-12)),
                      'predicted_depth_loss': depth_loss_value, 'sds_loss': sds_value,
                      'sds_image_gradient_norm': gradient_norm,
                      'elapsed_s': time.monotonic() - start,
                      'peak_vram_GiB': torch.cuda.max_memory_allocated() / 2**30}
            log.write(json.dumps(metric) + '\n'); log.flush()
            if (step + 1) % 50 == 0 or step == 0:
                print(f"{mode} {step+1}/{config['steps']} PSNR(sample)={metric['reference_psnr_sampled']:.2f} depth={depth_loss_value:.4f} elapsed={metric['elapsed_s']:.1f}s", flush=True)
                status.update(metric)
                write_json(folder / 'status.json', status)
            if (step + 1) % 500 == 0 or step + 1 == config['steps']:
                temporary = folder / 'last.tmp'
                torch.save({'step': step + 1, 'field': field.state_dict(), 'optimizer': optimizer.state_dict(),
                            'config': config, 'calibration': calibration['metadata']}, temporary)
                temporary.replace(folder / 'last.pt')
        status.update(status='trained', step=config['steps'])
        write_json(folder / 'status.json', status)
    del field, optimizer
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(calibration, config):
    poses = calibration['poses'].cuda()
    intrinsics = calibration['intrinsics_512'].cuda()
    targets = [torch.tensor(np.asarray(Image.open(ROOT / f'inputs/multiview_a10/{name}_256.png')).copy(), device='cuda').float() / 255
               for name in calibration['ids']]
    fields = {}
    results = {}
    board = Image.new('RGB', (3 * 256, 3 * 282), 'white')
    draw = ImageDraw.Draw(board)
    for mode in ['single', 'multi']:
        saved = torch.load(OUT / mode / 'last.pt', map_location='cpu', weights_only=False)
        assert saved['step'] == config['steps']
        assert all(saved['config'].get(key) == value for key, value in config.items())
        field = NeRF(log2_size=config['hash_log2_size']).cuda().eval()
        field.load_state_dict(saved['field'], strict=True)
        fields[mode] = field
        metrics = []
        for i, name in enumerate(calibration['ids']):
            rgb, _, _ = render_camera(field, poses[i], intrinsics[i], config)
            mse = float((rgb - targets[i]).square().mean())
            metrics.append({'image': name, 'psnr': -10 * math.log10(max(mse, 1e-12)), 'ssim': ssim(targets[i], rgb),
                            'used_for_nerf_training': i in ([0] if mode == 'single' else [0, 1])})
            Image.fromarray(array_image(rgb)).save(OUT / mode / f'camera_{name}.png')
            column = 1 if mode == 'single' else 2
            board.paste(Image.fromarray(array_image(rgb)), (column * 256, i * 282))
            draw.text((column * 256 + 6, i * 282 + 260), f'{mode}: {metrics[-1]["psnr"]:.2f} dB', fill='black')
        results[mode] = metrics
    for i, name in enumerate(calibration['ids']):
        board.paste(Image.fromarray(array_image(targets[i])), (0, i * 282))
        draw.text((6, i * 282 + 260), f'photo {name}' + (' (held out)' if i == 2 else ''), fill='black')
    board.save(OUT / 'comparison.png')
    frames = []
    previews = []
    for mode in fields:
        (OUT / mode / 'renders').mkdir(exist_ok=True)
    with imageio.get_writer(str(OUT / 'comparison_path.mp4'), fps=24, codec='libx264', quality=8) as writer:
        for index in range(120):
            t = .5 - .5 * math.cos(2 * math.pi * index / 120)
            pose = interpolate_pose(poses[0], poses[1], t)
            k = intrinsics[0].lerp(intrinsics[1], t)
            panels = []
            for mode, field in fields.items():
                image, _, _ = render_camera(field, pose, k, config)
                pixels = array_image(image)
                Image.fromarray(pixels).save(OUT / mode / 'renders' / f'{index:04d}.png')
                panels.append(pixels)
            pair = np.concatenate(panels, axis=1)
            writer.append_data(pair)
            if index in [0, 15, 30, 45, 60, 75, 90, 105]:
                previews.append(Image.fromarray(pair))
            frames.append({'index': index, 'fraction': t, 'c2w': pose.cpu().tolist(), 'intrinsics_512': k.cpu().tolist()})
            if index % 20 == 0:
                print('Export comparison:', index, '/120', flush=True)
    sheet = Image.new('RGB', (1024, 1024))
    for i, im in enumerate(previews):
        sheet.paste(im, ((i % 2) * 512, (i // 2) * 256))
    sheet.save(OUT / 'path_contact_sheet.png')
    write_json(OUT / 'transforms.json', {'convention': 'OpenGL camera-to-world', 'resolution': 256,
                                        'path': 'interpolation between estimated cameras 00572 and 00563, forward and back', 'frames': frames})
    heldout_single, heldout_multi = results['single'][2], results['multi'][2]
    results['heldout_psnr_delta_db'] = heldout_multi['psnr'] - heldout_single['psnr']
    results['heldout_ssim_delta'] = heldout_multi['ssim'] - heldout_single['ssim']
    results['calibration_caveat'] = 'All 3 images used to estimate cameras/geometry. 00616 excluded from RGB, pseudo-depth and SDS reference losses. Not a fully unseen-image benchmark.'
    results['method'] = 'Experimental multi-reference ZeroNVS SDS + RGB + MASt3R predicted-depth NeRF supervision. Not official paper reproduction.'
    results['steps_per_run'] = config['steps']
    results['camera_estimation'] = calibration['metadata']
    write_json(OUT / 'results.json', results)
    for mode in ['single', 'multi']:
        status = json.loads((OUT / mode / 'status.json').read_text())
        status.update(status='complete', exported_frames=120)
        write_json(OUT / mode / 'status.json', status)
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--mode', choices=['all', 'single', 'multi', 'evaluate'], default='all')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    calibration = torch.load(OUT / 'cameras.pt', map_location='cpu', weights_only=False)
    config = {'seed': 0, 'steps': args.steps, 'reference_rays': 4096, 'samples': 96,
              'ray_chunk': 2048, 'hash_log2_size': 19, 'lr': .005,
              'lambda_rgb': 1500., 'lambda_depth': 20., 'lambda_sds': .1,
              'warmup_steps': 100, 'sds_every': 2, 'sds_resolution': 64,
              'sampling': 'piecewise inverse depth 0.15-2 and linear 2-12, equal intervals',
              'near': .15, 'far': 12.,
              'calibration': 'MASt3R aerial, common shared calibration for both runs'}
    config_path = OUT / 'experiment_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Existing experiment configuration differs; use a separate output folder')
    write_json(config_path, config)
    if args.mode != 'evaluate':
        from official_guidance import create_guidance
        # Only these two reference embeddings are computed; the held-out photo
        # is never given to the diffusion teacher.
        image = ROOT / 'inputs/aerialmegadepth/00572_256_rgba.png'
        teacher = create_guidance(image, ROOT / 'weights/zeronvs_ema_inference.safetensors', 55., 0., .7)
        embeddings = []
        for name in calibration['ids'][:2]:
            rgb = torch.tensor(np.asarray(Image.open(ROOT / f'inputs/multiview_a10/{name}_256.png')).copy(), device='cuda').float().permute(2, 0, 1)[None] / 255
            embeddings.append(teacher.get_img_embeds(rgb))
        for mode in ['single', 'multi'] if args.mode == 'all' else [args.mode]:
            train(mode, calibration, teacher, embeddings, config)
        del teacher
        torch.cuda.empty_cache()
    if args.mode in ['all', 'evaluate']:
        evaluate(calibration, config)


if __name__ == '__main__':
    main()
