import argparse
import json
import math
import os
import resource
import struct
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, default=ROOT / 'code/aerial-megadepth/assets/siteACC0003-finearts/siteACC0003-camA005-2023-10-16-13-32-00-000181.jpg')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/zeronvs-megascenes/model-inference.safetensors')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/zeronvs_cpu')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fov', type=float, default=60)
    parser.add_argument('--depth-scale', type=float, default=1.0)
    parser.add_argument('--angles', type=float, nargs='+', default=[-10, 10])
    parser.add_argument('--check-header', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 500 or args.threads < 1 or args.depth_scale <= 0:
        parser.error('Use 1–500 steps, positive thread count and depth scale')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT / 'code/megascenes-nvs'))

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image, ImageDraw, ImageOps
    from safetensors import safe_open
    from ldm.util import instantiate_from_config
    from zeronvs_cpu_support import DeviceDDIMSampler

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    configuration = OmegaConf.load(ROOT / 'code/megascenes-nvs/configs/zeronvs_finetune/config.yaml')
    configuration.model.params.use_ema = False
    configuration.model.params.cond_stage_config.target = 'zeronvs_cpu_support.LocalCLIPImageEmbedder'
    started = time.perf_counter()
    model = instantiate_from_config(configuration.model).cpu().eval()
    if args.check_header:
        with args.check_header.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(size))
        expected = model.state_dict()
        available = {name: value for name, value in header.items()
                     if name != '__metadata__' and not name.startswith('model_ema.')}
        assert set(expected) == set(available), {'missing': sorted(set(expected) - set(available)),
                                                'extra': sorted(set(available) - set(expected))}
        for name, tensor in expected.items():
            assert list(tensor.shape) == available[name]['shape'], name
        print('Preflight passed: %d tensors match the official checkpoint header' % len(expected), flush=True)
        return
    with safe_open(str(args.checkpoint), framework='pt', device='cpu') as archive:
        state = {name: archive.get_tensor(name) for name in archive.keys() if not name.startswith('model_ema.')}
        model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False)
    load_seconds = time.perf_counter() - started
    print('Strict checkpoint loading completed in %.2f seconds' % load_seconds, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    image = ImageOps.fit(image, (256, 256), method=Image.Resampling.LANCZOS)
    image.save(args.output / 'input.png')
    tensor = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    with torch.no_grad():
        embedding = model.get_learned_conditioning(tensor)
        latent_reference = model.encode_first_stage(tensor).mode()
        sampler = DeviceDDIMSampler(model)
        outputs = []
        records = []
        for angle in args.angles:
            radians = math.radians(angle)
            rotation = torch.tensor([[math.cos(radians), 0, math.sin(radians)],
                                     [0, 1, 0], [-math.sin(radians), 0, math.cos(radians)]])
            center = torch.tensor([0., 0., -1.])
            relative = torch.eye(4)
            relative[:3, :3] = rotation
            relative[:3, 3] = center - rotation @ center
            relative_unscaled = relative.clone()
            relative[:3, 3] /= max(args.depth_scale, 0.01)
            relative[:3, 3] = 5 * torch.tanh(relative[:3, 3] / 10)
            field_of_view = math.radians(args.fov)
            pose = torch.cat([relative.flatten(), torch.tensor([
                field_of_view, math.sin(field_of_view), math.cos(field_of_view)])]).reshape(1, 1, 19)
            condition = {'c_concat': [latent_reference],
                         'c_crossattn': [model.cc_projection(torch.cat([embedding, pose], dim=-1))]}
            unconditional = {'c_concat': [torch.zeros_like(latent_reference)],
                             'c_crossattn': [torch.zeros_like(condition['c_crossattn'][0])]}
            torch.manual_seed(args.seed)
            started = time.perf_counter()
            executed_steps = []

            def progress(step, current, prediction):
                executed_steps.append(step)
                if (step + 1) % 10 == 0:
                    print('Orbit %+g: %d denoising updates' % (angle, step + 1), flush=True)
                return current

            samples, unused = sampler.sample(
                S=args.steps, batch_size=1, shape=[4, 32, 32], conditioning=condition,
                unconditional_conditioning=unconditional, unconditional_guidance_scale=3.0,
                eta=1.0, verbose=False, callback=progress,
            )
            decoded = model.decode_first_stage(samples)
            elapsed = time.perf_counter() - started
            assert torch.isfinite(decoded).all(), 'Generated pixels contain nonfinite values'
            pixels = ((decoded[0].permute(1, 2, 0).clamp(-1, 1).numpy() + 1) * 127.5).round().astype(np.uint8)
            result = Image.fromarray(pixels)
            filename = 'orbit_%+g.png' % angle
            result.save(args.output / filename)
            outputs.append((angle, result))
            records.append({'orbit_degrees': angle, 'seconds': elapsed, 'image': filename,
                            'executed_denoising_updates': len(executed_steps),
                            'target_camera_to_reference': relative_unscaled.tolist(),
                            'conditioning_vector': pose.flatten().tolist()})
            (args.output / ('orbit_%+g.json' % angle)).write_text(json.dumps({
                'model': 'ZeroNVS-MegaScenes iter_90000',
                'steps': args.steps, 'seed': args.seed, 'threads': args.threads,
                'fov_degrees': args.fov, 'depth_scale': args.depth_scale,
                'result': records[-1],
            }, indent=2) + '\n')
            print('Orbit %+g degrees: %.2f seconds' % (angle, elapsed), flush=True)
        canvas = Image.new('RGB', (256 * (len(outputs) + 1), 302), 'white')
        draw = ImageDraw.Draw(canvas)
        canvas.paste(image, (0, 36))
        draw.text((8, 10), 'Input (center crop)', fill='black')
        for index, (angle, result) in enumerate(outputs, 1):
            canvas.paste(result, (256 * index, 36))
            draw.text((256 * index + 8, 10), 'ZeroNVS-MS / orbit %+g deg' % angle, fill='black')
        canvas.save(args.output / 'comparison.png')
    report = {
        'model': 'ZeroNVS fine-tuned on MegaScenes, iteration 90000',
        'checkpoint': str(args.checkpoint), 'checkpoint_loading': 'strict; non-EMA parameters',
        'input': str(args.image), 'input_preprocessing': 'EXIF transpose, RGB, center crop to 256x256',
        'device': 'cpu', 'dtype': 'float32', 'torch': torch.__version__,
        'threads': args.threads, 'steps': args.steps, 'sampler': 'DDIM', 'eta': 1.0,
        'guidance_scale': 3.0, 'seed': args.seed, 'assumed_fov_degrees': args.fov,
        'assumed_reference_depth_quantile': args.depth_scale,
        'pose_convention': 'OpenGL camera-to-world; reference identity; orbit about (0,0,-1)',
        'load_seconds': load_seconds, 'outputs': records,
        'peak_process_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
        'scope': 'Single-image pose-conditioned diffusion smoke test; not original ZeroNVS NeRF distillation, not AerialMegaDepth fine-tuned weights, no ground truth accuracy evaluation.',
    }
    (args.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
