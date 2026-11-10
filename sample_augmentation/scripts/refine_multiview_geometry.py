"""Equal-budget geometric refinement of archived single/dual ZeroNVS NeRFs."""

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

from portable_nerf import NeRF, compose
from run_multiview_experiment import calibrated_rays, render_camera, ssim, array_image
from surface_constraints import surface_losses


ROOT = Path(__file__).resolve().parents[1]
TRAIN_IDS = ("00572", "00563")
EVALUATION_ID = "00616"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_field(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    field = NeRF(log2_size=saved["config"]["hash_log2_size"]).cuda()
    field.load_state_dict(saved["field"], strict=True)
    return field, saved["config"], saved["step"]


def build_training_views(calibration, mode):
    count = 1 if mode == "single" else 2
    if calibration["ids"] != [*TRAIN_IDS, EVALUATION_ID]:
        raise ValueError("Unexpected calibration IDs or order")
    views = []
    for index, name in enumerate(TRAIN_IDS[:count]):
        pose = calibration["poses"][index].cuda()
        intrinsics = calibration["intrinsics_512"][index].cuda()
        origins, directions = calibrated_rays(pose, intrinsics, 256)
        depth = F.interpolate(calibration["depths"][index:index + 1, None].cuda(), (256, 256), mode="nearest").flatten()
        confidence = F.interpolate(calibration["confidence"][index:index + 1, None].cuda(), (256, 256), mode="nearest").flatten()
        cosine = (directions * -pose[:3, 2]).sum(-1).clamp_min(0.1)
        ranges = depth / cosine
        image = np.asarray(Image.open(ROOT / f"inputs/multiview_a10/{name}_256.png").convert("RGB")).copy()
        colors = torch.from_numpy(image).cuda().float().reshape(-1, 3) / 255
        valid = (confidence > 3) & torch.isfinite(ranges) & (ranges > 0.15) & (ranges < 12)
        views.append({"origins": origins, "directions": directions, "ranges": ranges,
                      "valid": valid, "colors": colors})
    return views


def sample_edges(origins, count, generator):
    intervals = torch.linspace(0, 1, count + 1, device=origins.device).expand(len(origins), -1).clone()
    noise = torch.rand((len(origins), count - 1), generator=generator, device=origins.device)
    intervals[:, 1:-1] += (noise - 0.5) / count
    inverse = 1 / ((1 - 2 * intervals) / 0.15 + 2 * intervals / 2).clamp_min(1e-6)
    linear = 2 + (2 * intervals - 1) * 10
    return torch.where(intervals <= 0.5, inverse, linear)


def refine(mode, source, calibration, config, output):
    torch.manual_seed(config["seed"])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    field, original_config, original_step = load_field(source)
    field.train()
    views = build_training_views(calibration, mode)
    optimizer = torch.optim.Adam(field.parameters(), lr=config["lr"], betas=(0.9, 0.99))
    folder = output / mode
    folder.mkdir()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with (folder / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(config["steps"]):
            optimizer.zero_grad(set_to_none=True)
            totals = {"rgb": 0.0, "surface": 0.0, "free_space": 0.0, "depth": 0.0}
            for view in views:
                pixels = torch.randint(256 * 256, (config["reference_rays"] // len(views),), generator=generator, device="cuda")
                origins, directions = view["origins"][pixels], view["directions"][pixels]
                edges = sample_edges(origins, config["samples"], generator)
                rgb, _, _, weights = compose(field, origins, directions, edges, origins.new_full((3,), 0.5))
                losses = surface_losses(weights, edges, view["ranges"][pixels], view["valid"][pixels],
                                        config["relative_band"], config["absolute_band"])
                losses["rgb"] = (rgb - view["colors"][pixels]).square().mean()
                loss = sum(config["lambda_" + name] * value for name, value in losses.items()) / len(views)
                loss.backward()
                for name, value in losses.items():
                    totals[name] += float(value.detach()) / len(views)
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in field.parameters()):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
            record = {"step": step + 1, **totals, "elapsed_s": time.monotonic() - started}
            stream.write(json.dumps(record) + "\n")
            if step == 0 or (step + 1) % 50 == 0:
                stream.flush()
                print(mode, json.dumps(record), flush=True)
    torch.cuda.synchronize()
    status = {"status": "complete", "step": config["steps"], "source_step": original_step,
              "training_ids": list(TRAIN_IDS[:len(views)]), "source_sha256": digest(source),
              "elapsed_s": time.monotonic() - started,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    torch.save({"field": field.state_dict(), "config": original_config, "refinement_config": config,
                "step": original_step + config["steps"], "provenance": status}, folder / "last.pt")
    write_json(folder / "status.json", status)
    del field, optimizer, views
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(calibration, original, output):
    variants = {"original_single": original / "single/last.pt", "original_multi": original / "multi/last.pt",
                "refined_single": output / "single/last.pt", "refined_multi": output / "multi/last.pt"}
    board = Image.new("RGB", (5 * 256, 3 * 286), "white")
    draw = ImageDraw.Draw(board)
    results = {}
    for column, (name, path) in enumerate(variants.items(), 1):
        field, config, _ = load_field(path)
        field.eval()
        results[name] = []
        for index, photo in enumerate(calibration["ids"]):
            pose = calibration["poses"][index].cuda()
            intrinsics = calibration["intrinsics_512"][index].cuda()
            rgb, depth, opacity = render_camera(field, pose, intrinsics, config)
            real = Image.open(ROOT / f"inputs/multiview_a10/{photo}_256.png").convert("RGB")
            target = torch.from_numpy(np.asarray(real).copy()).cuda().float() / 255
            mse = float((rgb - target).square().mean())
            score = {"id": photo, "psnr": -10 * math.log10(max(mse, 1e-12)), "ssim": ssim(target, rgb)}
            results[name].append(score)
            picture = Image.fromarray(array_image(rgb))
            picture.save(output / f"{name}_{photo}.png")
            board.paste(real, (0, index * 286))
            board.paste(picture, (column * 256, index * 286))
            draw.text((6, index * 286 + 259), photo + (" HELD OUT" if index == 2 else ""), fill="black")
            draw.text((column * 256 + 5, index * 286 + 259), name, fill="black")
            draw.text((column * 256 + 5, index * 286 + 272), f"PSNR {score['psnr']:.2f} / SSIM {score['ssim']:.3f}", fill="black")
        del field
        torch.cuda.empty_cache()
    board.save(output / "comparison.png")
    comparison = {}
    for metric in ("psnr", "ssim"):
        comparison[metric + "_dual_minus_single"] = results["refined_multi"][2][metric] - results["refined_single"][2][metric]
        comparison[metric + "_dual_minus_original_dual"] = results["refined_multi"][2][metric] - results["original_multi"][2][metric]
    report = {"metrics": results, "heldout_comparison": comparison,
              "both_heldout_metrics_improved_over_refined_single": all(comparison[metric + "_dual_minus_single"] > 0 for metric in ("psnr", "ssim")),
              "limitations": ["Exploratory single scene and seed; previously inspected held-out image is now a development check.",
                              "Shared calibration used all three images; held-out RGB/depth never used by refinement.",
                              "Refinement has no new SDS updates; both models start from their own archived SDS checkpoint.",
                              "Pseudo-depth is estimated, not ground truth. No universal improvement guarantee."]}
    write_json(output / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surface_refinement.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; archived experiments must not be overwritten")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["steps"] < 1 or config["reference_rays"] < 2 or config["reference_rays"] % 2 or config["samples"] < 2:
        raise ValueError("Invalid step, sample or equal ray budget")
    calibration_path = args.source / "cameras.pt"
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
    for mode in ("single", "multi"):
        if not (args.source / mode / "last.pt").is_file():
            raise FileNotFoundError(args.source / mode / "last.pt")
    args.output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    write_json(args.output / "protocol.json", {"config": config, "calibration_sha256": digest(calibration_path),
               "training_reference_ids": {"single": [TRAIN_IDS[0]], "multi": list(TRAIN_IDS)},
               "evaluation_id": EVALUATION_ID, "selection": "Fixed final step; no selection using evaluation metrics",
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
               "script_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                  ("refine_multiview_geometry.py", "surface_constraints.py", "portable_nerf.py", "run_multiview_experiment.py")}})
    for mode in ("single", "multi"):
        refine(mode, args.source / mode / "last.pt", calibration, config, args.output)
    evaluate(calibration, args.source, args.output)


if __name__ == "__main__":
    main()
