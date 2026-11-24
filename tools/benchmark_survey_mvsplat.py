"""Run upstream MVSplat-ACID on the source images and cameras frozen by the survey benchmark."""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from tools.benchmark_stereo_transport import OUT, ROOT, read
from relation_nvs.io import load_camera, sha256, write_json

WEIGHTS_SHA = '253785dee0804e6d7f312bd5b927a47454ba55c32735c0a2685b5d07594e142f'


def register_upstream(path):
    sys.path.insert(0, str(path.resolve()))
    # The training registries import loggers and video exporters. Import only
    # the original inference modules, keeping their source and weights intact.
    for name in ('src.dataset', 'src.model.encoder', 'src.model.decoder'):
        package = types.ModuleType(name)
        package.__path__ = [str(path.joinpath(*name.split('.')).resolve())]
        sys.modules[name] = package


def load_upstream(path, checkpoint):
    if sha256(checkpoint) != WEIGHTS_SHA:
        raise ValueError('Unexpected MVSplat-ACID checkpoint')
    register_upstream(path)
    from src.global_cfg import set_cfg
    from src.model.encoder.encoder_costvolume import EncoderCostVolume
    from src.model.decoder.cuda_splatting import render_cuda

    config = OmegaConf.load(path / 'config/experiment/acid.yaml')
    encoder_config = OmegaConf.merge(OmegaConf.load(path / 'config/model/encoder/costvolume.yaml'), config.model.encoder)
    set_cfg(OmegaConf.create({'mode': 'test', 'dataset': {'view_sampler': {'num_context_views': 2}}}))
    encoder = EncoderCostVolume(encoder_config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')}, strict=True)
    return encoder.cuda().eval(), render_cuda, config.dataset


@torch.inference_mode()
def generate(group, model, renderer, config, size, composite=False, model_name='mvsplat_acid'):
    folder = OUT / group
    manifest = folder / 'predictions.json'
    predictions = read(manifest)
    name = f'{model_name}_{size}'
    composites = {}
    for row in predictions:
        cameras = [load_camera(c) for c in row['cameras']]
        images = []
        for path, expected, camera in zip(row['paths'][:2], row['source_hashes'], cameras[:2]):
            if sha256(ROOT / path) != expected:
                raise ValueError(path)
            with Image.open(ROOT / path) as im:
                shape = {'256': (256,256), '256x448': (448,256), 'native': tuple(max(32,32 * round(n/32)) for n in (camera.width,camera.height))}[size]
                images.append(np.asarray(im.convert('RGB').resize(shape, Image.Resampling.LANCZOS)))
        rgb = torch.as_tensor(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255
        # ACID's bounds are fixed in SfM units, while our input is in metres.
        # One common similarity sets the mean source altitude to 10 model units,
        # well inside the upstream 1..100 interval. Source 1 defines the frame.
        poses = np.stack([c.camera_to_world for c in cameras])
        units = np.mean(poses[:2, 2, 3]) / 10.
        if units <= 0:
            raise ValueError('Survey altitude must be positive')
        poses = np.linalg.inv(poses[0]) @ poses
        poses[:, :3, 3] /= units
        pose = torch.as_tensor(poses, dtype=torch.float32, device='cuda')
        intrinsics = np.stack([np.diag([1/c.width, 1/c.height, 1]) @ c.intrinsics for c in cameras])
        K = torch.as_tensor(intrinsics, dtype=torch.float32, device='cuda')
        near = torch.full((3,), float(config.near), device='cuda')
        far = torch.full_like(near, float(config.far))
        scene = model(dict(image=rgb[None], extrinsics=pose[None,:2], intrinsics=K[None,:2],
                           near=near[None,:2], far=far[None,:2]), 300000, deterministic=True)
        if isinstance(scene,dict):
            scene = scene['gaussians']
        target = cameras[-1]
        rendered = renderer(pose[2:], K[2:], near[2:], far[2:], (target.height, target.width),
                            torch.zeros(1, 3, device='cuda'), scene.means, scene.covariances,
                            scene.harmonics, scene.opacities)
        if not torch.isfinite(rendered).all():
            raise ValueError('Nonfinite rendering at ' + row['case'])
        dest = folder / row['case']
        if composite:
            colors = torch.ones_like(scene.harmonics[..., :1])
            # Rendering unit precomputed colors over black returns alpha. Use the
            # same planar rendering for missing radiance in this control.
            alpha = renderer(pose[2:], K[2:], near[2:], far[2:], (target.height, target.width),
                             torch.zeros(1,3,device='cuda'), scene.means, scene.covariances,
                             colors, scene.opacities, use_sh=False)
            with Image.open(dest/'plane.png') as bg:
                plane = torch.as_tensor(np.asarray(bg).copy(),device='cuda').permute(2,0,1)/255.
            output_name = name + '_plane_bg'
            rendered = rendered + (1 - alpha.clamp(0,1)) * plane
        else:
            output_name = name
        result = rendered[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        output = dest / (output_name + '.png')
        Image.fromarray(result).save(output)
        if composite or model_name == 'depthsplat_dl3dv':
            composites[row['case']] = dict(rgb_sha256=sha256(output))
            if composite:
                composites[row['case']]['mean_alpha'] = alpha.mean().item()
        else:
            row['artifacts'][name] = sha256(output)
        print(group, row['case'], output_name, flush=True)
    if composite or model_name == 'depthsplat_dl3dv':
        write_json(folder/(output_name + '.json'), composites)
    else:
        write_json(manifest, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/MVSplat/acid.ckpt')
    parser.add_argument('--size', choices=('256', 'native'), default='256')
    parser.add_argument('--composite', action='store_true')
    parser.add_argument('groups', nargs='+')
    args = parser.parse_args()
    model, renderer, config = load_upstream(args.upstream, args.checkpoint)
    for group in args.groups:
        generate(group, model, renderer, config, args.size, args.composite)
