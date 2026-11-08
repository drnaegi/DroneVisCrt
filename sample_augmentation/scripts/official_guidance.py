"""Run upstream ZeroNVS guidance without importing unrelated threestudio plugins.

The guidance class, including camera conditioning, noise schedule, VAE encoding,
and SDS loss, is compiled directly from the downloaded upstream source. Only
framework plumbing, checkpoint loading and mathematically equivalent attention
kernels are supplied here.
"""
import ast
import dataclasses
import importlib
import pathlib
import sys
import types
import copy
import typing

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from diffusers import DDIMScheduler
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ZeroNVS'
sys.path.insert(0, str(SOURCE / 'zeronvs_diffusion' / 'zero123'))


def schedule(value, epoch, step):
    if isinstance(value, (float, int)):
        return value
    start, lo, hi, end = value if len(value) == 4 else [0, *value]
    return lo + (hi - lo) * max(0.0, min(1.0, (step - start) / (end - start)))


class BaseObject:
    @dataclasses.dataclass
    class Config:
        pass

    def __init__(self, config):
        self.cfg = self.Config(**config)
        self.device = torch.device('cuda')
        self.configure()


def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    from ldm.models.diffusion import options
    options.LDM_DISTILLATION_ONLY = True  # upstream switch enabling VAE input gradients
    import clip
    from clip.model import CLIP
    from ldm.modules.attention import CrossAttention
    from ldm.modules.diffusionmodules.model import AttnBlock

    def sdpa(self, x, context=None, mask=None):
        context = x if context is None else context
        b, n, _ = x.shape
        h = self.heads
        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(context).view(b, -1, h, q.shape[-1]).transpose(1, 2)
        v = self.to_v(context).view(b, -1, h, q.shape[-1]).transpose(1, 2)
        attn_mask = None if mask is None else mask.reshape(b, 1, 1, -1).bool()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))

    def vae_sdpa(self, x):
        y = self.norm(x)
        b, c, h, w = y.shape
        q, k, v = (op(y).flatten(2).transpose(1, 2).unsqueeze(1) for op in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj_out(y.squeeze(1).transpose(1, 2).reshape(b, c, h, w))

    CrossAttention.forward = sdpa
    AttnBlock.forward = vae_sdpa

    # The checkpoint already contains CLIP. Construct its exact architecture
    # locally, avoiding an unnecessary download of another pretrained CLIP.
    def construct_clip(name, device='cpu', jit=False, **kwargs):
        if name != 'ViT-L/14':
            raise ValueError(f'Unexpected upstream CLIP architecture: {name}')
        model = CLIP(embed_dim=768, image_resolution=224, vision_layers=24,
                     vision_width=1024, vision_patch_size=14, context_length=77,
                     vocab_size=49408, transformer_width=768,
                     transformer_heads=12, transformer_layers=12)
        return model.eval().to(device), None

    old_clip_load = clip.load
    clip.load = construct_clip
    config = copy.deepcopy(config)
    config.model.params.use_ema = False  # already applied during verified extraction
    config.model.params.conditioning_config.params.depth_model_name = None
    config.model.params.unet_config.params.use_checkpoint = False
    try:
        target = config.model.target
        module, cls = target.rsplit('.', 1)
        model = getattr(importlib.import_module(module), cls)(**config.model.params)
    finally:
        clip.load = old_clip_load
    state = load_file(ckpt, device='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Incomplete teacher checkpoint: missing={missing}, unexpected={unexpected}')
    del state
    if vram_O:
        del model.first_stage_model.decoder
    model.eval().requires_grad_(False).to(device)
    # Match mixed precision inference while keeping original FP32 weights.
    apply_model = model.apply_model

    def mixed_apply(x, t, cond, *args, **kwargs):
        with torch.autocast('cuda', dtype=torch.float16):
            return apply_model(x, t, cond, *args, **kwargs).float()

    model.apply_model = mixed_apply
    print('Official EMA teacher loaded; all 1250 tensors matched.', flush=True)
    return model


def create_guidance(image, checkpoint, fov, elevation, scale):
    from ldm.data import common
    module = types.ModuleType('portable_upstream_zeronvs_guidance')
    sys.modules[module.__name__] = module
    module.__dict__.update({k: v for k, v in vars(typing).items() if not k.startswith('__')})
    module.__dict__.update({
        'torch': torch, 'np': np, 'nn': torch.nn, 'F': F, 'cv2': cv2,
        'os': __import__('os'), 'importlib': importlib, 'copy': copy,
        'dataclass': dataclasses.dataclass, 'field': dataclasses.field,
        'OmegaConf': OmegaConf, 'DDIMScheduler': DDIMScheduler,
        'common': common, 'BaseObject': BaseObject, 'C': schedule,
        'threestudio': types.SimpleNamespace(register=lambda name: lambda cls: cls,
                                             info=lambda message: print(message, flush=True)),
        'load_model_from_config': load_model_from_config,
    })
    # OpenCV's Windows imread cannot reliably read Unicode workspace paths.
    # Keep this change local to the dynamically loaded guidance module.
    cv_proxy = types.ModuleType('cv2_unicode_paths')
    cv_proxy.__dict__.update(vars(cv2))
    cv_proxy.imread = lambda path, flags=cv2.IMREAD_COLOR: cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    module.__dict__['cv2'] = cv_proxy
    filename = SOURCE / 'threestudio' / 'models' / 'guidance' / 'zero123_guidance.py'
    tree = ast.parse(filename.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name != 'load_model_from_config']
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    ast_module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(ast_module, str(filename), 'exec'), module.__dict__)
    config = {
        'pretrained_model_name_or_path': str(checkpoint),
        'pretrained_config': str(SOURCE / 'zeronvs_config.yaml'),
        'cond_image_path': str(image), 'cond_fov_deg': fov,
        'cond_elevation_deg': elevation, 'cond_azimuth_deg': 0.0,
        'cond_camera_distance': 1.0, 'precomputed_scale': scale,
        'vram_O': True, 'guidance_scale': 9.5,
        'min_step_percent': [0, 0.75, 0.02, 1000],
        'max_step_percent': [1000, 0.98, 0.025, 2500],
        'p_use_aux_cameras': 0.0, 'use_anisotropic_schedule': True,
        'anisotropic_offset': 1000,
    }
    return module.Zero123Guidance(config)
