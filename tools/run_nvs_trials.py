import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
for directory in ["code/ml-sharp/src", "code/moge", "code/utils3d"]:
    sys.path.insert(0, str(ROOT / directory))


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(directory):
    return subprocess.check_output(
        ["git", "-C", str(ROOT / directory), "rev-parse", "HEAD"], text=True
    ).strip()


def camera_matrix(offset):
    position = np.array([offset, 0, 0], dtype=np.float32)
    forward = np.array([0, 0, 1], dtype=np.float32) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0, 1, 0], dtype=np.float32), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ position
    return matrix


def save_depth(depth, path):
    from matplotlib import colormaps

    valid = np.isfinite(depth) & (depth > 0)
    inverse = np.zeros_like(depth)
    inverse[valid] = 1 / depth[valid]
    lower, upper = np.percentile(inverse[valid], [2, 98])
    values = np.clip((inverse - lower) / max(upper - lower, 1e-6), 0, 1)
    colors = (colormaps["turbo"](values)[..., :3] * 255).astype(np.uint8)
    colors[~valid] = 0
    Image.fromarray(colors).save(path)


def contact_sheet(panels, path):
    panel_width = 480
    panel_height = 320
    columns = 3
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * panel_width, rows * (panel_height + 34)), "#141b26")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        row, column = divmod(index, columns)
        panel = Image.fromarray(array)
        panel.thumbnail((panel_width, panel_height))
        left = column * panel_width
        top = row * (panel_height + 34)
        draw.text((left + 12, top + 8), label, fill="white")
        canvas.paste(panel, (left + (panel_width - panel.width) // 2, top + 34))
    canvas.save(path, quality=92)


def load_model(method, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if method == "sharp":
        from sharp.models import PredictorParams, create_predictor

        model = create_predictor(PredictorParams())
        model.load_state_dict(state, strict=True)
    else:
        from moge.model.v2 import MoGeModel

        model = MoGeModel(**state["model_config"])
        model.load_state_dict(state["model"], strict=True)
    return model.eval().cuda()


def infer_sharp(model, image, focal, output):
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import save_ply
    from sharp.utils.gsplat import GSplatRenderer

    height, width = image.shape[:2]
    torch.cuda.synchronize()
    started = time.perf_counter()
    gaussians = predict_image(model, image, focal, torch.device("cuda"))
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    inference_peak = torch.cuda.max_memory_allocated() / 1024**3
    save_ply(gaussians, focal, (height, width), output / "scene.ply")
    intrinsics = torch.tensor(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
        dtype=torch.float32,
        device="cuda",
    )
    renderer = GSplatRenderer(color_space="linearRGB")
    initial = renderer(
        gaussians,
        torch.eye(4, device="cuda")[None],
        intrinsics[None],
        width,
        height,
    )
    initial_depth = initial.depth[0, 0]
    valid = (initial.alpha[0, 0] > 0.9) & torch.isfinite(initial_depth) & (initial_depth > 0)
    median_depth = float(initial_depth[valid].median())
    gaussians = gaussians._replace(
        mean_vectors=gaussians.mean_vectors / median_depth,
        singular_values=gaussians.singular_values / median_depth,
    )

    def render(offset):
        result = renderer(
            gaussians,
            torch.from_numpy(camera_matrix(offset)).cuda()[None],
            intrinsics[None],
            width,
            height,
        )
        rgb = result.color[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        alpha = result.alpha[0, 0].cpu().numpy()
        return np.rint(rgb * 255).astype(np.uint8), alpha

    info = {
        "inference_seconds": inference_seconds,
        "inference_peak_allocated_gib": inference_peak,
        "median_predicted_depth_before_normalization": median_depth,
        "gaussian_count": gaussians.mean_vectors.shape[1],
        "internal_resolution": [1536, 1536],
        "rendering": "official SHARP GSplatRenderer; custom documented camera path",
        "coverage_definition": "Gaussian alpha >= 0.9; not target correctness",
    }
    return render, initial_depth.cpu().numpy(), info


def infer_moge(model, image, fov, output):
    import utils3d
    from moge.utils.io import save_glb

    height, width = image.shape[:2]
    image_tensor = torch.from_numpy(image.copy()).cuda().float().permute(2, 0, 1) / 255
    torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = model.infer(image_tensor, fov_x=fov, resolution_level=9, use_fp16=True)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    inference_peak = torch.cuda.max_memory_allocated() / 1024**3
    arrays = {key: value.cpu().numpy() for key, value in prediction.items()}
    np.savez_compressed(output / "geometry.npz", **arrays)
    depth = arrays["depth"]
    mask = arrays["mask"] & np.isfinite(depth) & (depth > 0)
    median_depth = float(np.median(depth[mask]))
    mask_cleaned = mask & ~utils3d.np.depth_map_edge(depth, rtol=0.03)
    faces, vertices, colors, vertex_uvs = utils3d.np.build_mesh_from_map(
        arrays["points"] / median_depth,
        image.astype(np.float32) / 255,
        utils3d.np.uv_map(height, width),
        mask=mask_cleaned,
        tri=True,
    )
    vertices = vertices.astype(np.float32)
    save_glb(
        output / "scene.glb",
        vertices * np.array([1, -1, -1], dtype=np.float32),
        faces,
        vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32),
        image,
    )
    context = utils3d.np.RastContext(standalone=True, backend="egl")
    projection = utils3d.np.intrinsics_to_perspective(
        arrays["intrinsics"], near=0.001, far=1000.0
    ).astype(np.float32)
    color_and_coverage = np.concatenate([colors, np.ones((len(colors), 1), dtype=np.float32)], axis=1)
    background = np.zeros((height, width, 4), dtype=np.float32)

    def render(offset):
        result = utils3d.np.rasterize_triangles(
            context,
            (height, width),
            vertices=vertices,
            faces=faces,
            attributes=color_and_coverage,
            view=utils3d.np.extrinsics_to_view(camera_matrix(offset)).astype(np.float32),
            projection=projection,
            background_image=background,
        )
        rgb = np.clip(result["image"][..., :3], 0, 1)
        alpha = result["image"][..., 3]
        return np.rint(rgb * 255).astype(np.uint8), alpha

    info = {
        "inference_seconds": inference_seconds,
        "inference_peak_allocated_gib": inference_peak,
        "median_predicted_depth_before_normalization": median_depth,
        "valid_geometry_fraction": float(mask.mean()),
        "mesh_face_count": len(faces),
        "mesh_depth_edge_rtol": 0.03,
        "resolution_level": 9,
        "rendering": "MoGe-2 geometry plus utils3d mesh rendering; no generative hole filling",
        "coverage_definition": "rasterized mesh alpha (vertex=1, background=0); not target correctness",
        "opengl_renderer": context.mgl_ctx.info.get("GL_RENDERER"),
    }
    return render, depth, info


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sharp", "moge2"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20261023)
    args.output.mkdir(parents=True, exist_ok=True)
    input_paths = [args.input] if args.input.is_file() else sorted(
        path for path in args.input.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    metadata = {
        "method": args.method,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checksum(args.checkpoint),
        "source_revision": revision("code/ml-sharp" if args.method == "sharp" else "code/moge"),
        "utils3d_revision": revision("code/utils3d"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "horizontal_fov_assumed_degrees": args.fov,
        "calibration": "unknown; shared assumed FOV for diagnostics, not true camera calibration",
        "camera_protocol": "each geometry normalized to median depth 1; look at [0,0,1]; camera x offset relative to that depth",
        "target_ground_truth": False,
        "seed": 20261023,
        "samples": [],
    }
    started = time.perf_counter()
    model = load_model(args.method, args.checkpoint)
    torch.cuda.synchronize()
    metadata["model_load_seconds"] = time.perf_counter() - started
    for image_path in input_paths:
        print("Processing", args.method, image_path, flush=True)
        output = args.output / image_path.stem
        output.mkdir(parents=True, exist_ok=True)
        image_pil = Image.open(image_path).convert("RGB")
        image_pil.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
        image = np.array(image_pil)
        image_pil.save(output / "input.png")
        height, width = image.shape[:2]
        focal = width / (2 * math.tan(math.radians(args.fov) / 2))
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if args.method == "sharp":
            render, depth, info = infer_sharp(model, image, focal, output)
        else:
            render, depth, info = infer_moge(model, image, args.fov, output)
        torch.cuda.synchronize()
        save_depth(depth, output / "depth.png")
        np.save(output / "depth.npy", depth)
        panels = [("Input", image)]
        offsets = [0.0, -0.05, 0.05, -0.15, 0.15]
        views = []
        for offset in offsets:
            rendered, alpha = render(offset)
            label = f"view_{offset:+.2f}"
            Image.fromarray(rendered).save(output / f"{label}.png")
            Image.fromarray(np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)).save(
                output / f"{label}_coverage.png"
            )
            view_info = {
                "offset": offset,
                "extrinsics_world_to_camera": camera_matrix(offset).tolist(),
                "low_coverage_fraction": float((alpha < 0.9).mean()),
            }
            if offset == 0:
                view_info["source_reconstruction_mae_0_255"] = float(
                    np.abs(rendered.astype(np.float32) - image).mean()
                )
            views.append(view_info)
            panels.append((f"{args.method} | relative camera x = {offset:+.2f}", rendered))
        contact_sheet(panels, output / "contact_sheet.jpg")
        video_path = output / "nearby_views.mp4"
        trajectory = []
        with imageio.get_writer(str(video_path), fps=16, codec="libx264", quality=8, macro_block_size=2) as writer:
            for frame_index in range(args.frames):
                offset = 0.10 * math.sin(2 * math.pi * frame_index / args.frames)
                rendered, _ = render(offset)
                writer.append_data(rendered)
                trajectory.append({"offset": offset, "extrinsics": camera_matrix(offset).tolist()})
        torch.cuda.synchronize()
        info.update({
            "input": str(image_path.resolve()),
            "input_sha256": checksum(image_path),
            "resolution": [width, height],
            "total_sample_seconds": time.perf_counter() - started,
            "total_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "views": views,
            "video_frames": args.frames,
            "video_encoded_resolution": [math.ceil(width / 2) * 2, math.ceil(height / 2) * 2],
            "output": str(output.resolve()),
        })
        (output / "metrics.json").write_text(json.dumps(info, indent=2))
        (output / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        metadata["samples"].append(info)
        (args.output / "run_manifest.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(info), flush=True)
        del render, depth
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
