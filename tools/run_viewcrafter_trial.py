import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as functional
import torch.utils.checkpoint
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "code/viewcrafter-sparse"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / "extern/dust3r"))


def save_frames(frames, directory, video_name):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = np.rint(np.clip(frames, 0, 1) * 255).astype(np.uint8)
    for index, frame in enumerate(arrays):
        Image.fromarray(frame).save(directory / f"{index:03d}.png")
    with imageio.get_writer(str(directory.parent / video_name), fps=8, codec="libx264",
                           quality=9, macro_block_size=2) as writer:
        for frame in arrays:
            writer.append_data(frame)


def attention_sdpa(module, hidden, context=None, mask=None):
    if module.relative_position:
        return module.original_forward(hidden, context=context, mask=mask)
    self_attention = context is None
    context = hidden if self_attention else context
    query = module.to_q(hidden)
    image_context = None
    if module.image_cross_attention and not self_attention:
        image_context = context[:, module.text_context_len:]
        context = context[:, :module.text_context_len]
    elif not self_attention:
        context = context[:, :module.text_context_len]
    batch, length, _ = query.shape

    def heads(tensor):
        return tensor.reshape(batch, tensor.shape[1], module.heads, module.dim_head).transpose(1, 2)

    query = heads(query)
    attention_mask = (mask[:, None] > 0.5) if mask is not None else None
    result = functional.scaled_dot_product_attention(
        query, heads(module.to_k(context)), heads(module.to_v(context)),
        attn_mask=attention_mask, dropout_p=0.0, scale=module.scale,
    )
    if image_context is not None:
        image_result = functional.scaled_dot_product_attention(
            query, heads(module.to_k_ip(image_context)), heads(module.to_v_ip(image_context)),
            dropout_p=0.0, scale=module.scale,
        )
        factor = module.image_cross_attention_scale
        if module.image_cross_attention_scale_learnable:
            factor = factor * (torch.tanh(module.alpha) + 1)
        result = result + factor * image_result
    return module.to_out(result.transpose(1, 2).reshape(batch, length, -1))


def autoencoder_sdpa(module, hidden):
    normalized = module.norm(hidden)
    batch, channels, height, width = normalized.shape
    query, key, value = [layer(normalized).flatten(2).transpose(1, 2)[:, None].contiguous()
                         for layer in (module.q, module.k, module.v)]
    result = functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
    result = result[:, 0].transpose(1, 2).reshape(batch, channels, height, width)
    return hidden + module.proj_out(result)


def install_attention():
    from lvdm.modules.attention import CrossAttention
    from lvdm.modules.networks.ae_modules import AttnBlock

    CrossAttention.original_forward = CrossAttention.forward
    CrossAttention.forward = attention_sdpa
    CrossAttention.efficient_forward = attention_sdpa
    AttnBlock.forward = autoencoder_sdpa


def load_dust3r(device):
    from safetensors.torch import load_file
    from dust3r.model import AsymmetricCroCo3DStereo

    model = AsymmetricCroCo3DStereo(
        pos_embed="RoPE100", img_size=(512, 512), head_type="dpt", output_mode="pts3d",
        depth_mode=("exp", -float("inf"), float("inf")), conf_mode=("exp", 1, float("inf")),
        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
        dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
        patch_embed_cls="PatchEmbedDust3R", landscape_only=False, freeze="none",
    )
    state = load_file(str(ROOT / "checkpoints/DUSt3R/model.safetensors"))
    for head in ("downstream_head1", "downstream_head2"):
        for index in range(4):
            state[f"{head}.dpt.scratch.layer_rn.{index}.weight"] = state[
                f"{head}.dpt.scratch.layer{index + 1}_rn.weight"
            ]
    result = model.load_state_dict(state, strict=True)
    print("DUSt3R strict state load:", result, flush=True)
    return model.eval().to(device)


def options(args):
    from configs.infer_config import get_parser

    opts = get_parser().parse_args([])
    opts.image_dir = str(args.image)
    opts.save_dir = str(args.output)
    opts.config = str(SOURCE / "configs/inference_pvd_1024.yaml")
    opts.ckpt_path = str(ROOT / "checkpoints/ViewCrafter_16/model.ckpt")
    opts.seed = args.seed
    opts.video_length = 16
    opts.ddim_steps = args.steps
    opts.elevation = args.elevation
    opts.center_scale = args.center_scale
    opts.d_phi = [args.phi]
    opts.d_theta = [args.theta]
    opts.d_r = [0.0]
    opts.d_x = [0.0]
    opts.d_y = [0.0]
    opts.mode = "single_view_target"
    return opts


def geometry(args, opts, metadata):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
    from viewcrafter import ViewCrafter
    from utils.pvd_utils import generate_traj_specified, world_point_to_obj

    class GeometryEngine(ViewCrafter):
        def setup_diffusion(self):
            return None

        def setup_dust3r(self):
            self.dust3r = load_dust3r(self.device)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine = GeometryEngine(opts)
    with torch.no_grad():
        poses = engine.scene.get_im_poses().detach()[1:]
        principal = engine.scene.get_principal_points().detach()[1:]
        focals = engine.scene.get_focals().detach()[1:]
        height, width = map(int, engine.images[0]["true_shape"][0])
        points = torch.stack([value.detach() for value in engine.scene.get_pts3d(clip_thred=opts.dpt_trd)])
        depth = engine.scene.get_depthmaps()[-1].detach()
        center_depth = depth[height // 2, width // 2]
        radius = center_depth * opts.center_scale
        poses, points = world_point_to_obj(poses, points, -1, radius, opts.elevation, opts.device)
        zero = torch.zeros((), device=opts.device)
        cameras, frame_count = generate_traj_specified(
            poses, height, width, focals, principal, args.theta, args.phi, 0.0,
            zero, zero, opts.video_length, opts.device,
        )
        colors = np.array(engine.scene.imgs)
        renderings, _ = engine.run_render([points[-1]], [colors[-1]], None,
                                          height, width, cameras, frame_count)
        renderings = functional.interpolate(renderings.permute(0, 3, 1, 2),
                                            size=(576, 1024), mode="bilinear",
                                            align_corners=False).permute(0, 2, 3, 1)
        renderings[0] = engine.img_ori
        arrays = renderings.cpu().numpy()
        centers = cameras.get_camera_center().cpu().numpy()
        rotation = cameras.R.cpu().numpy()
        relative_rotation = rotation[0].T @ rotation[-1]
        angle = np.degrees(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)))
        camera_data = {
            "convention": "PyTorch3D row-vector world-to-view R,T; scene scale is predicted",
            "R": rotation.tolist(), "T": cameras.T.cpu().tolist(),
            "centers": centers.tolist(), "focal_length": cameras.focal_length.cpu().tolist(),
            "principal_point": cameras.principal_point.cpu().tolist(),
            "geometry_image_size": [width, height], "rendered_size": [1024, 576],
            "center_depth": float(center_depth), "orbit_radius": float(radius),
            "endpoint_rotation_degrees": float(angle),
            "baseline_over_radius": float(np.linalg.norm(centers[-1] - centers[0]) / float(radius)),
            "requested_phi_degrees": args.phi, "requested_theta_degrees": args.theta,
            "assumed_source_elevation_degrees": args.elevation,
        }
        np.savez_compressed(args.output / "geometry.npz", depth=depth.cpu().numpy(),
                            points=points[-1].cpu().numpy(), colors=colors[-1])
        (args.output / "cameras.json").write_text(json.dumps(camera_data, indent=2))
        torch.cuda.synchronize()
        metadata["geometry_seconds"] = time.perf_counter() - started
        metadata["geometry_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        np.save(args.output / "conditions.npy", arrays)
        save_frames(arrays, args.output / "conditions", "conditions.mp4")
        Image.fromarray(np.rint(arrays[0] * 255).astype(np.uint8)).save(args.output / "input.png")
    del engine, points, depth, renderings, cameras
    gc.collect()
    torch.cuda.empty_cache()
    print("Geometry and camera path saved", flush=True)


def generate(args, opts, metadata):
    from omegaconf import OmegaConf
    from utils.diffusion_utils import instantiate_from_config, load_model_checkpoint, image_guided_synthesis

    install_attention()
    config = OmegaConf.load(opts.config).model
    config.params.unet_config.params.use_checkpoint = False
    config.params.cond_stage_config.params.version = None
    config.params.img_cond_stage_config.params.version = None
    config.params.perframe_ae = True
    OmegaConf.save(config, args.output / "effective_model_config.yaml")
    started = time.perf_counter()
    model = instantiate_from_config(config)
    model = load_model_checkpoint(model, opts.ckpt_path).eval().to(opts.device)
    model.cond_stage_model.device = opts.device
    model.perframe_ae = True
    torch.cuda.synchronize()
    metadata["diffusion_model_load_seconds"] = time.perf_counter() - started
    arrays = np.load(args.output / "conditions.npy")
    if len(arrays) != opts.video_length:
        raise ValueError("Condition frame count differs from the model video length")
    videos = (torch.from_numpy(arrays).to(opts.device) * 2 - 1).permute(3, 0, 1, 2)[None]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        samples = image_guided_synthesis(
            model, [opts.prompt], videos, [1, 4, opts.video_length, 72, 128],
            n_samples=1, ddim_steps=args.steps, ddim_eta=opts.ddim_eta,
            unconditional_guidance_scale=opts.unconditional_guidance_scale,
            cfg_img=opts.cfg_img, fs=opts.frame_stride, text_input=opts.text_input,
            multiple_cond_cfg=False, timestep_spacing=opts.timestep_spacing,
            guidance_rescale=opts.guidance_rescale, condition_index=[0],
        )
    torch.cuda.synchronize()
    metadata["diffusion_seconds"] = time.perf_counter() - started
    metadata["diffusion_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    result = ((samples[0, 0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) / 2).float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite generated pixels")
    save_frames(result, args.output / "generated", "generated.mp4")
    metadata["generated_frames"] = len(result)
    metadata["source_reconstruction_mae_0_255"] = float(np.abs(result[0] - arrays[0]).mean() * 255)
    print("Generated views saved", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phi", type=float, default=30)
    parser.add_argument("--theta", type=float, default=0)
    parser.add_argument("--elevation", type=float, default=35)
    parser.add_argument("--center-scale", type=float, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--phase", choices=["geometry", "generate", "all"], default="all")
    args = parser.parse_args()
    args.image = args.image.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(SOURCE)
    torch.set_num_threads(8)
    from pytorch_lightning import seed_everything
    seed_everything(args.seed)
    opts = options(args)
    metadata_path = args.output / "run.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if args.phase == "generate":
        expected = {"input": str(args.image), "phi": args.phi, "theta": args.theta,
                    "elevation": args.elevation, "center_scale": args.center_scale}
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Cached geometry differs from requested {key}")
    metadata.update({
        "method": "ViewCrafter_16", "input": str(args.image),
        "input_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "seed": args.seed, "ddim_steps": args.steps, "ddim_eta": opts.ddim_eta,
        "guidance_scale": opts.unconditional_guidance_scale, "guidance_rescale": opts.guidance_rescale,
        "text_input": opts.text_input, "prompt": opts.prompt,
        "perframe_ae": True, "attention": "PyTorch SDPA preserving original attention equations",
        "model_weights": "official ViewCrafter checkpoint; strict state load including OpenCLIP",
        "geometry_weights": "official DUSt3R safetensors; strict state load",
        "phi": args.phi, "theta": args.theta, "elevation": args.elevation,
        "center_scale": args.center_scale, "target_ground_truth": False,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if args.phase in {"geometry", "all"}:
        geometry(args, opts, metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    if args.phase in {"generate", "all"}:
        seed_everything(args.seed)
        generate(args, opts, metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
