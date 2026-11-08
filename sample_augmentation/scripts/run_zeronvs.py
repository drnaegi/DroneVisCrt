"""Train a portable ZeroNVS SDS NeRF, resume it, and export actual renders."""
import argparse
import hashlib
import json
import math
import pathlib
import random
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from portable_nerf import NeRF, camera_pose, rays, render, compose, sample_edges

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACTIVE_OUTPUT = None


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path, field, optimizer, step, config):
    data = {'step': step, 'config': config, 'field': field.state_dict(),
            'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(), 'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state()}
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def save_image(path, tensor):
    Image.fromarray((tensor.detach().float().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)).save(path)


@torch.no_grad()
def export(field, config, out, step, frames=120, resolution=256):
    folder = out / 'renders'
    folder.mkdir(exist_ok=True)
    background = torch.full((3,), 0.5, device='cuda')
    metadata = []
    previews = []
    with imageio.get_writer(str(out / 'orbit.mp4'), fps=24, codec='libx264', quality=8) as writer:
        for i in range(frames):
            azimuth = 360 * i / frames
            pose = camera_pose(config['elevation'], azimuth)
            origins, directions = rays(pose, resolution, config['fov'])
            edges = sample_edges(field, origins, directions, samples=config['samples'], randomized=False)
            rgb, depth, opacity = render(field, origins, directions, edges, background, config['ray_chunk'])
            rgb = rgb.reshape(resolution, resolution, 3)
            filename = f'{i:04d}.png'
            save_image(folder / filename, rgb)
            array = (rgb.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
            writer.append_data(array)
            if i % max(1, frames // 8) == 0:
                previews.append(Image.fromarray(array))
            metadata.append({'file_path': f'renders/{filename}', 'transform_matrix': pose.cpu().tolist(),
                             'azimuth_deg': azimuth, 'elevation_deg': config['elevation'], 'step': step})
            if i % 10 == 0:
                print(f'Export {i+1}/{frames}', flush=True)
    sheet = Image.new('RGB', (resolution * 4, resolution * math.ceil(len(previews) / 4)))
    for i, image in enumerate(previews):
        sheet.paste(image, ((i % 4) * resolution, (i // 4) * resolution))
    sheet.save(out / 'contact_sheet.png')
    write_json(out / 'transforms.json', {'camera_angle_x': math.radians(config['fov']),
                                       'coordinate_system': 'OpenGL c2w, Z-up world', 'w': resolution,
                                       'h': resolution, 'frames': metadata})


def main():
    global ACTIVE_OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'aerialmegadepth.json'))
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--output')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--skip-export', action='store_true')
    args = parser.parse_args()
    config = json.loads(pathlib.Path(args.config).read_text(encoding='utf-8'))
    if args.steps is not None:
        config['steps'] = args.steps
    if args.resolution is not None:
        config['resolution'] = args.resolution
        config['final_resolution'] = args.resolution
    out = ROOT / (args.output or config['output'])
    out.mkdir(parents=True, exist_ok=True)
    ACTIVE_OUTPUT = out
    if (out / 'last.pt').exists() and not (args.resume or args.render_only):
        raise RuntimeError('Existing checkpoint: use --resume or a new --output')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    field = NeRF(log2_size=config['hash_log2_size']).cuda()
    optimizer = torch.optim.Adam(field.parameters(), lr=config['lr'], betas=(0.9, 0.99), eps=1e-8)
    step = 0
    if args.resume or args.render_only:
        data = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        field.load_state_dict(data['field'])
        optimizer.load_state_dict(data['optimizer'])
        step = data['step']
        for key in ['fov', 'elevation', 'scale', 'samples', 'hash_log2_size', 'image']:
            if config[key] != data['config'][key]:
                raise ValueError(f'Cannot resume with changed {key}')
        torch.set_rng_state(data['torch_rng'])
        torch.cuda.set_rng_state(data['cuda_rng'])
        random.setstate(data['python_rng'])
        np.random.set_state(data['numpy_rng'])
        del data
    if args.render_only:
        export(field, config, out, step, args.frames)
        return
    write_json(out / 'config.json', config)
    status = {'status': 'loading_teacher', 'step': step, 'total_steps': config['steps'],
              'device': torch.cuda.get_device_name(0), 'backend': 'PyTorch hash-grid NeRF + upstream ZeroNVS SDS',
              'paper_reproduction': False}
    write_json(out / 'status.json', status)
    from official_guidance import create_guidance
    checkpoint = ROOT / 'weights' / 'zeronvs_ema_inference.safetensors'
    image_path = ROOT / config['image']
    teacher = create_guidance(image_path, checkpoint, config['fov'], config['elevation'], config['scale'])
    target_full = torch.tensor(np.asarray(Image.open(image_path).convert('RGB')).copy(), device='cuda').float().permute(2, 0, 1)[None] / 255
    canonical = camera_pose(config['elevation'], 0.0)
    started = time.monotonic()
    initial_step = step
    metrics_file = out / 'metrics.jsonl'
    field.train()
    while step < config['steps']:
        current = step
        resolution = config['resolution'] if step < config['resolution_milestone'] else config['final_resolution']
        target = F.interpolate(target_full, (resolution, resolution), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).reshape(-1, 3)
        progress = min(1., (step + 1) / 1000)
        azimuth = random.uniform(-180, 180) * progress
        elevation = config['elevation'] + (random.uniform(5, 80) - config['elevation']) * progress
        distance = random.uniform(0.8, 1.8)
        pose = camera_pose(elevation, azimuth, distance)
        origins, directions = rays(pose, resolution, config['fov'])
        edges = sample_edges(field, origins, directions, config['samples'])
        background = torch.rand(3, device='cuda')
        optimizer.zero_grad(set_to_none=True)
        # First pass holds no NeRF graph. SDS differentiates the VAE with respect
        # to the rendered image; an exact replay then propagates that image
        # gradient into NeRF chunks using identical ray samples/background.
        with torch.no_grad():
            rgb, _, _ = render(field, origins, directions, edges, background, config['ray_chunk'])
        image = rgb.reshape(1, resolution, resolution, 3).detach().requires_grad_(True)
        camera = {'elevation': torch.tensor([elevation], device='cuda'),
                  'azimuth': torch.tensor([azimuth], device='cuda'),
                  'camera_distances': torch.tensor([distance], device='cuda'),
                  'canonical_c2w': canonical[None], 'c2w': pose[None],
                  'fov_deg': torch.tensor([config['fov']], device='cuda'), 'aux_c2ws': []}
        teacher.update_step(0, step)
        guidance = teacher(image, camera)
        sds_loss = guidance['loss_sds'] * config['lambda_sds']
        image_grad = torch.autograd.grad(sds_loss, image)[0].detach().reshape(-1, 3)
        if not torch.isfinite(image_grad).all() or image_grad.norm() == 0:
            raise RuntimeError('SDS produced invalid/zero image gradients')
        sds_value = float(guidance['loss_sds'].detach())
        grad_norm = float(image_grad.norm())
        del guidance, sds_loss, image
        for start in range(0, len(origins), config['ray_chunk']):
            stop = start + config['ray_chunk']
            rgb_chunk, _, _, _ = compose(field, origins[start:stop], directions[start:stop], edges[start:stop], background)
            (rgb_chunk * image_grad[start:stop]).sum().backward()
        ref_origins, ref_directions = rays(canonical, resolution, config['fov'])
        indices = torch.randperm(len(ref_origins), device='cuda')[:config['reference_rays']]
        ref_origins, ref_directions = ref_origins[indices], ref_directions[indices]
        ref_edges = sample_edges(field, ref_origins, ref_directions, config['samples'])
        reference_mse = 0.
        for start in range(0, len(indices), config['ray_chunk']):
            stop = start + config['ray_chunk']
            rgb_chunk, _, _, _ = compose(field, ref_origins[start:stop], ref_directions[start:stop], ref_edges[start:stop], background)
            mse = (rgb_chunk - target[indices[start:stop]]).square().sum() / (len(indices) * 3)
            (config['lambda_rgb'] * mse).backward()
            reference_mse += float(mse.detach())
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in field.parameters()):
            raise RuntimeError('Nonfinite NeRF gradients')
        optimizer.step()
        step += 1
        torch.cuda.synchronize()
        metric = {'step': step, 'resolution': resolution, 'loss_sds': sds_value,
                  'sds_image_gradient_norm': grad_norm, 'reference_mse': reference_mse,
                  'reference_psnr': -10 * math.log10(max(reference_mse, 1e-12)),
                  'azimuth_deg': azimuth, 'elevation_deg': elevation,
                  'peak_vram_GiB': torch.cuda.max_memory_allocated() / 2**30,
                  'elapsed_s': time.monotonic() - started}
        with metrics_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps(metric) + '\n')
        if step <= initial_step + 3 or step % 10 == 0:
            seconds = metric['elapsed_s'] / (step - initial_step)
            print(f"step {step}/{config['steps']} SDS={sds_value:.3f} ref_PSNR={metric['reference_psnr']:.2f} grad={grad_norm:.3f} VRAM={metric['peak_vram_GiB']:.2f}GiB {seconds:.2f}s/step", flush=True)
            status.update(metric, status='training', estimated_remaining_s=seconds * (config['steps'] - step))
            write_json(out / 'status.json', status)
        if step % config['checkpoint_every'] == 0 or step == config['steps']:
            save_checkpoint(out / 'last.pt', field, optimizer, step, config)
            with torch.no_grad():
                ro, rd = rays(canonical, 128, config['fov'])
                ed = sample_edges(field, ro, rd, config['samples'], randomized=False)
                preview, _, _ = render(field, ro, rd, ed, torch.full((3,), 0.5, device='cuda'), config['ray_chunk'])
                save_image(out / f'reference_step_{step:04d}.png', preview.reshape(128, 128, 3))
    del teacher
    torch.cuda.empty_cache()
    status.update(status='trained', step=step)
    write_json(out / 'status.json', status)
    if not args.skip_export:
        export(field, config, out, step, args.frames)
        status.update(status='complete', exported_frames=args.frames, render_resolution=256)
        write_json(out / 'status.json', status)
    print('Completed:', out, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if ACTIVE_OUTPUT is not None:
            status_path = ACTIVE_OUTPUT / 'status.json'
            status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
            status.update(status='failed', error=f'{type(error).__name__}: {error}')
            write_json(status_path, status)
        raise
