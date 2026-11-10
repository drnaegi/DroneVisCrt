import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from run_viewcrafter_trial import ROOT, SOURCE, generate, load_dust3r, save_frames


COMMON_INDICES = np.rint(np.linspace(0, 24, 16)).astype(int).tolist()


def prepare_geometry(args, opts):
    from viewcrafter import ViewCrafter
    from utils.pvd_utils import generate_traj, world_point_to_kth

    class GeometryEngine(ViewCrafter):
        def setup_diffusion(self):
            return None

        def setup_dust3r(self):
            self.dust3r = load_dust3r(self.device)

    directory = args.output / "geometry"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    engine = GeometryEngine(opts)
    if len(engine.images) != 25:
        raise ValueError("The benchmark sequence must contain exactly 25 real frames")
    with torch.no_grad():
        poses = engine.scene.get_im_poses().detach()
        principal = engine.scene.get_principal_points().detach()
        focals = engine.scene.get_focals().detach()
        height, width = map(int, engine.images[0]["true_shape"][0])
        points = torch.stack([value.detach() for value in engine.scene.get_pts3d(clip_thred=opts.dpt_trd)])
        poses, points = world_point_to_kth(poses, points, 0, opts.device)
        cameras, count = generate_traj(poses, height, width, focals, principal, opts.device)
        truth = torch.stack(engine.img_ori).cpu().numpy()
        relative = poses[0, :3, :3].T[None] @ poses[:, :3, :3]
        traces = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
        angles = torch.rad2deg(torch.acos(((traces - 1) / 2).clamp(-1, 1)))
        camera_data = {
            "pose_source": "DUSt3R complete-graph alignment of the 25 real frames, following official eval",
            "c2w_source_relative": poses.cpu().tolist(),
            "pytorch3d_R": cameras.R.cpu().tolist(), "pytorch3d_T": cameras.T.cpu().tolist(),
            "focal_length": focals.cpu().tolist(), "principal_point": principal.cpu().tolist(),
            "geometry_image_size": [width, height],
            "rotation_from_source_degrees": angles.cpu().tolist(),
            "endpoint_rotation_degrees": float(angles[-1]),
        }
        del points
        reference_images = [engine.images[0], copy.deepcopy(engine.images[0])]
        reference_images[1]["idx"] = 1
    del engine.scene
    gc.collect()
    torch.cuda.empty_cache()
    engine.run_dust3r(input_images=reference_images)
    with torch.no_grad():
        source_points = engine.scene.get_pts3d(clip_thred=opts.dpt_trd)[0].detach()
        source_colors = np.array(engine.scene.imgs)[0]
        rendered, _ = engine.run_render([source_points], [source_colors], None, height, width, cameras, count)
        rendered = functional.interpolate(rendered.permute(0, 3, 1, 2), size=(576, 1024),
                                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        rendered[0] = engine.img_ori[0]
        conditions = rendered.cpu().numpy()
    if not np.isfinite(conditions).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite geometry or ground-truth pixels")
    torch.cuda.synchronize()
    metadata = {
        "input_directory": str(args.images), "frames": 25,
        "geometry_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "point_cloud_source": "Reference image only, re-estimated from two identical copies",
        "target_images_used_for_pose_estimation": True,
        "target_images_used_as_diffusion_image_conditions": False,
        "common_frame_indices": COMMON_INDICES,
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(args.images.glob("*.jpg"))},
    }
    np.save(directory / "conditions.npy", conditions)
    save_frames(conditions, directory / "conditions", "conditions.mp4")
    save_frames(truth, directory / "ground_truth", "ground_truth.mp4")
    (directory / "cameras.json").write_text(json.dumps(camera_data, indent=2))
    (directory / "run.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def run_generation(args, opts, length):
    geometry_directory = args.output / "geometry"
    geometry_metadata = json.loads((geometry_directory / "run.json").read_text())
    actual_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(args.images.glob("*.jpg"))}
    if actual_hashes != geometry_metadata["input_sha256"]:
        raise ValueError("Cached geometry does not match the requested sequence")
    directory = args.output / f"viewcrafter{length}"
    directory.mkdir(parents=True, exist_ok=True)
    indices = COMMON_INDICES if length == 16 else list(range(25))
    conditions = np.load(geometry_directory / "conditions.npy")[indices]
    np.save(directory / "conditions.npy", conditions)
    opts.video_length = length
    opts.ckpt_path = str(ROOT / f"checkpoints/ViewCrafter_{length}/model.ckpt")
    opts.save_dir = str(directory)
    metadata = {
        "method": f"ViewCrafter_{length}", "source_revision": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        "source_frame_indices": indices, "evaluation_common_indices": COMMON_INDICES[1:],
        "seed": args.seed, "steps": args.steps, "ddim_eta": opts.ddim_eta,
        "guidance_scale": opts.unconditional_guidance_scale, "guidance_rescale": opts.guidance_rescale,
        "prompt": opts.prompt, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "checkpoint": json.loads(Path(opts.ckpt_path + ".json").read_text()),
        "geometry_metadata": str(geometry_directory / "run.json"),
    }
    generation_args = argparse.Namespace(output=directory, steps=args.steps)
    generate(generation_args, opts, metadata)
    (directory / "run.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["geometry", "16", "25"], required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    args.images = args.images.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(SOURCE)
    Image.ANTIALIAS = Image.Resampling.LANCZOS
    torch.set_num_threads(8)
    from configs.infer_config import get_parser
    from pytorch_lightning import seed_everything

    seed_everything(args.seed)
    opts = get_parser().parse_args([])
    opts.image_dir = str(args.images)
    opts.save_dir = str(args.output)
    opts.config = str(SOURCE / "configs/inference_pvd_1024.yaml")
    opts.mode = "single_view_eval"
    opts.seed = args.seed
    opts.ddim_steps = args.steps
    if args.phase == "geometry":
        prepare_geometry(args, opts)
    else:
        run_generation(args, opts, int(args.phase))


if __name__ == "__main__":
    main()
