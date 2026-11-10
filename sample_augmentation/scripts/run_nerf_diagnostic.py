"""Train equal-budget sparse-view NeRF controls and evaluate frozen scene splits."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from nerf_diagnostic_data import camera_rays, checksum, foreground_mask, load_pixels, prepare_scene
from portable_nerf import NeRF


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def composite(density, colors, positions, near, far, background):
    edges = torch.cat([near[:, None], (positions[:, 1:] + positions[:, :-1]) * 0.5, far[:, None]], -1)
    optical = density * (edges[:, 1:] - edges[:, :-1]).clamp_min(0)
    alpha = -torch.expm1(-optical)
    transmission = torch.exp(-torch.cat([torch.zeros_like(optical[:, :1]), optical.cumsum(-1)[:, :-1]], -1))
    weights = alpha * transmission
    rgb = (weights[..., None] * colors).sum(-2) + (1 - weights.sum(-1, keepdim=True)) * background
    return rgb, weights, edges


def sample_pdf(edges, weights, count, randomized, generator=None):
    probabilities = weights.detach() + 1e-5
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(probabilities[:, :1]), probabilities.cumsum(-1)], -1)
    fractions = torch.arange(count, device=weights.device)[None].expand(len(weights), -1).float()
    jitter = torch.rand(fractions.shape, device=weights.device, generator=generator) if randomized else 0.5
    uniform = ((fractions + jitter) / count).contiguous()
    upper = torch.searchsorted(cdf.contiguous(), uniform, right=True).clamp(1, cdf.shape[-1] - 1)
    lower = upper - 1
    low_cdf, high_cdf = cdf.gather(1, lower), cdf.gather(1, upper)
    fraction = (uniform - low_cdf) / (high_cdf - low_cdf).clamp_min(1e-6)
    return edges.gather(1, lower) + fraction * (edges.gather(1, upper) - edges.gather(1, lower))


def query(field, origins, directions, positions):
    points = origins[:, None] + directions[:, None] * positions[..., None]
    density, colors = field(points.reshape(-1, 3))
    return density.reshape_as(positions), colors.reshape(*positions.shape, 3)


def render_rays(field, origins, directions, near, far, background, samples, sampler, randomized, generator=None):
    coarse_count = samples if sampler == "uniform" else samples // 2
    fractions = torch.arange(coarse_count, device=origins.device)[None].expand(len(origins), -1).float()
    jitter = torch.rand(fractions.shape, device=origins.device, generator=generator) if randomized else 0.5
    positions = near[:, None] + (far - near)[:, None] * (fractions + jitter) / coarse_count
    density, colors = query(field, origins, directions, positions)
    rgb, weights, edges = composite(density, colors, positions, near, far, background)
    if sampler == "uniform":
        return rgb
    fine = sample_pdf(edges.detach(), weights, samples - coarse_count, randomized, generator)
    fine_density, fine_colors = query(field, origins, directions, fine)
    all_positions, order = torch.cat([positions, fine], -1).sort(-1)
    all_density = torch.cat([density, fine_density], -1).gather(1, order)
    all_colors = torch.cat([colors, fine_colors], 1).gather(1, order[..., None].expand(-1, -1, 3))
    rgb, _, _ = composite(all_density, all_colors, all_positions, near, far, background)
    return rgb


def ssim(first, second):
    first, second = [value.permute(2, 0, 1)[None].float() for value in (first, second)]
    first_mean, second_mean = [F.avg_pool2d(value, 7, 1) for value in (first, second)]
    first_var = (F.avg_pool2d(first.square(), 7, 1) - first_mean.square()) * 49 / 48
    second_var = (F.avg_pool2d(second.square(), 7, 1) - second_mean.square()) * 49 / 48
    covariance = (F.avg_pool2d(first * second, 7, 1) - first_mean * second_mean) * 49 / 48
    return float((((2 * first_mean * second_mean + 0.01**2) * (2 * covariance + 0.03**2)) /
                  ((first_mean.square() + second_mean.square() + 0.01**2) * (first_var + second_var + 0.03**2))).mean())


def frame_tensors(frame, background, pixels=True):
    pose = torch.tensor(frame["pose"], device="cuda", dtype=torch.float32)
    intrinsics = torch.tensor(frame["intrinsics"], device="cuda", dtype=torch.float32)
    origins, directions, lengths = camera_rays(pose, intrinsics, frame["height"], frame["width"])
    result = {"origins": origins, "directions": directions,
              "near": lengths * frame["near"], "far": lengths * frame["far"]}
    if pixels:
        result["colors"] = torch.from_numpy(load_pixels(frame, background)).cuda().reshape(-1, 3)
    return result


def save_image(path, pixels):
    image = (pixels.detach().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(image).save(path)


@torch.no_grad()
def evaluate(field, scene, frames, config, folder):
    folder.mkdir(exist_ok=True)
    rows = []
    board = None
    for index, frame in enumerate(frames):
        rays = frame_tensors(frame, scene["background"])
        pieces = []
        for start in range(0, len(rays["origins"]), config["render_chunk"]):
            stop = start + config["render_chunk"]
            rgb = render_rays(field, *[rays[name][start:stop] for name in ("origins", "directions", "near", "far")],
                              scene["background"], config["samples"], config["sampler"], False)
            pieces.append(rgb)
        prediction = torch.cat(pieces).reshape(frame["height"], frame["width"], 3)
        target = rays["colors"].reshape_as(prediction)
        mse = float((prediction - target).square().mean())
        rows.append({"id": frame["id"], "mse": mse, "psnr": -10 * math.log10(max(mse, 1e-12)),
                     "ssim": ssim(target, prediction)})
        if scene["scene"] == "lego":
            mask = torch.from_numpy(foreground_mask(frame).copy()).cuda()
            if not mask.any():
                raise ValueError("Empty Lego foreground")
            foreground_mse = float((prediction[mask] - target[mask]).square().mean())
            rows[-1]["foreground_psnr"] = -10 * math.log10(max(foreground_mse, 1e-12))
        save_image(folder / f"{index:03d}.png", prediction)
        if index < 6:
            width, height = frame["width"], frame["height"]
            if board is None:
                board = Image.new("RGB", (width * 2, (height + 20) * min(6, len(frames))), "white")
            for column, pixels in enumerate((target, prediction)):
                image = Image.fromarray((pixels.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8))
                board.paste(image, (column * width, index * (height + 20)))
            ImageDraw.Draw(board).text((4, index * (height + 20) + height + 2),
                                      f"{frame['id']} PSNR {rows[-1]['psnr']:.2f}", fill="black")
    if board is not None:
        board.save(folder / "comparison.png")
    report = {"views": rows, "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
              "mean_ssim": float(np.mean([row["ssim"] for row in rows])), "count": len(rows)}
    if scene["scene"] == "lego":
        report["mean_foreground_psnr"] = float(np.mean([row["foreground_psnr"] for row in rows]))
    write_json(folder / "metrics.json", report)
    return report


def train(scene, base_config, count, seed, sampler, output):
    folder = output / f"{sampler}_{count:02d}views_seed{seed}"
    config = dict(base_config, count=count, seed=seed, sampler=sampler)
    protocol = {"config": config, "scene_sha256": checksum(output / "scene.json"),
                "training_ids": [frame["id"] for frame in scene["training_order"][:count]],
                "implementation_sha256": {name: checksum(Path(__file__).with_name(name)) for name in
                                          ("run_nerf_diagnostic.py", "nerf_diagnostic_data.py", "portable_nerf.py")}}
    if folder.exists():
        if (folder / "complete.json").exists() and json.loads((folder / "protocol.json").read_text()) == protocol:
            print("SKIP completed", folder.name, flush=True)
            return
        raise FileExistsError("Partial/different experiment exists: " + str(folder))
    folder.mkdir()
    write_json(folder / "protocol.json", protocol)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    field = NeRF(log2_size=config["hash_log2_size"]).cuda()
    field.train()
    optimizer = torch.optim.Adam(field.parameters(), lr=config["lr"], betas=(0.9, 0.99))
    frames = scene["training_order"][:count]
    ray_batches = [frame_tensors(frame, scene["background"]) for frame in frames]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (folder / "train.jsonl").open("w", encoding="utf-8") as log:
        for step in range(config["steps"]):
            optimizer.zero_grad(set_to_none=True)
            rays_per_frame = config["rays"] // count
            selected = {name: [] for name in ray_batches[0]}
            for batch in ray_batches:
                indices = torch.randint(len(batch["origins"]), (rays_per_frame,), generator=generator, device="cuda")
                for name, values in batch.items():
                    selected[name].append(values[indices])
            selected = {name: torch.cat(values) for name, values in selected.items()}
            prediction = render_rays(field, *[selected[name] for name in ("origins", "directions", "near", "far")],
                                     scene["background"], config["samples"], sampler, True, generator)
            loss = (prediction - selected["colors"]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] = config["lr"] * 0.1 ** ((step + 1) / config["steps"])
            if step == 0 or (step + 1) % 100 == 0:
                row = {"step": step + 1, "mse": float(loss.detach()), "elapsed_s": time.monotonic() - started}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(folder.name, json.dumps(row), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    torch.save({"field": field.state_dict(), "config": config, "step": config["steps"]}, folder / "last.pt")
    field.eval()
    train_report = evaluate(field, scene, frames, config, folder / "training_views")
    validation = evaluate(field, scene, scene["validation"], config, folder / "validation")
    write_json(folder / "complete.json", {"training_seconds": elapsed, "step": config["steps"],
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "training": train_report, "validation": validation, "checkpoint_sha256": checksum(folder / "last.pt")})
    del field, optimizer, ray_batches
    torch.cuda.empty_cache()


def test_scene(scene, output):
    study = json.loads((output / "study.json").read_text())
    expected = set(study["planned_runs"])
    completed = {folder.name for folder in output.iterdir() if (folder / "complete.json").is_file()}
    if expected != completed:
        raise RuntimeError("Finish exactly the predeclared training matrix before opening test: " + str(expected - completed))
    seal = output / "test_seal.json"
    if seal.exists():
        raise FileExistsError("Final test has already been opened")
    write_json(seal, {"study_sha256": checksum(output / "study.json"), "scene_sha256": checksum(output / "scene.json"),
                     "checkpoints": {name: checksum(output / name / "last.pt") for name in sorted(completed)}})
    experiments = []
    for folder in sorted(output.iterdir()):
        if not folder.is_dir() or not (folder / "complete.json").is_file():
            continue
        if (folder / "test/metrics.json").exists():
            raise FileExistsError("Test is already evaluated: " + str(folder))
        config = json.loads((folder / "protocol.json").read_text())["config"]
        checkpoint = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        field = NeRF(log2_size=config["hash_log2_size"]).cuda().eval()
        field.load_state_dict(checkpoint["field"], strict=True)
        report = evaluate(field, scene, scene["test"], config, folder / "test")
        experiments.append({"run": folder.name, "config": config, **report})
        print("TEST", folder.name, report["mean_psnr"], report["mean_ssim"], flush=True)
        del field, checkpoint
        torch.cuda.empty_cache()
    write_json(output / "test_summary.json", {"scene": scene["scene"], "experiments": experiments})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/nerf_diagnostic"))
    parser.add_argument("--scene", choices=("lego", "fern"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "test"), default="train")
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sampler", choices=("uniform", "hierarchical"), default="hierarchical")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--long-side", type=int, default=128)
    args = parser.parse_args()
    if args.steps < 1 or args.long_side < 16 or any(count not in (1, 2, 4, 8) for count in args.counts):
        raise ValueError("Invalid diagnostic budget")
    args.output.mkdir(parents=True, exist_ok=True)
    scene_path = args.output / "scene.json"
    if scene_path.exists():
        scene = json.loads(scene_path.read_text())
        if scene["scene"] != args.scene or scene["long_side"] != args.long_side:
            raise ValueError("Existing scene protocol differs")
    else:
        scene = prepare_scene(args.data, args.scene, args.long_side)
        write_json(scene_path, scene)
    study_path = args.output / "study.json"
    study = {"scene": args.scene, "steps": args.steps, "long_side": args.long_side,
             "planned_runs": [f"hierarchical_{count:02d}views_seed{seed}" for seed in (0, 1, 2) for count in (1, 2, 4, 8)] +
                             [f"uniform_{count:02d}views_seed0" for count in (1, 2)],
             "purpose": "Geometry/ray-sampling diagnostic with existing PyTorch NeRF; no diffusion teacher or depth prior.",
             "primary_comparison": "Hierarchical two-view minus one-view paired by seed on identical test cameras.",
             "controls": "Four/eight-view coverage; uniform one/two-view seed-zero sampler ablation.",
             "selection": "All configurations declared before training; test opened once after full training matrix.",
             "limitations": "Low-resolution fixed-budget diagnostic, not a reproduction of NeRF or ZeroNVS paper results."}
    if study_path.exists():
        if json.loads(study_path.read_text()) != study:
            raise ValueError("Study configuration is already frozen")
    else:
        write_json(study_path, study)
    if args.stage == "prepare":
        print(json.dumps({key: scene[key] for key in ("scene", "source_frame_count", "train_pool_size", "second_reference_angle_deg", "notes")}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.stage == "test":
        test_scene(scene, args.output)
        return
    if (args.output / "test_seal.json").exists():
        raise RuntimeError("Test results already opened; use a new explicitly exploratory experiment")
    config = {"steps": args.steps, "rays": 1024, "samples": 64, "render_chunk": 1024,
              "hash_log2_size": 17, "lr": 0.005, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "teacher": None,
              "selection": "Fixed final step; no checkpoint selection from validation or test"}
    for seed in args.seeds:
        for count in args.counts:
            train(scene, config, count, seed, args.sampler, args.output)


if __name__ == "__main__":
    main()
