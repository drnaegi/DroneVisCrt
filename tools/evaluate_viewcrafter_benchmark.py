import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity


COMMON_INDICES = np.rint(np.linspace(0, 24, 16)).astype(int).tolist()
METRICS = ("psnr", "ssim", "lpips_alex")


def load_rgb(path):
    with Image.open(path) as picture:
        pixels = np.asarray(picture.convert("RGB"), dtype=np.float32) / 255
    if pixels.shape != (576, 1024, 3):
        raise ValueError(f"Unexpected image shape at {path}: {pixels.shape}")
    return pixels


def score(prediction, truth, perceptual, device):
    error = float(np.mean((prediction.astype(np.float64) - truth.astype(np.float64)) ** 2))
    psnr = -10 * math.log10(error) if error > 0 else float("inf")
    ssim = structural_similarity(
        prediction, truth, data_range=1.0, channel_axis=2,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    tensors = [torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)[None].to(device) * 2 - 1
               for pixels in (prediction, truth)]
    with torch.inference_mode():
        perceptual_error = float(perceptual(*tensors).item())
    return {"psnr": psnr, "ssim": float(ssim), "lpips_alex": perceptual_error}


def verify_metrics(perceptual, device):
    generator = np.random.default_rng(123)
    pixels = generator.uniform(0.1, 0.8, (64, 64, 3)).astype(np.float32)
    identity = score(pixels, pixels, perceptual, device)
    offset = score(pixels + 0.1, pixels, perceptual, device)
    if not math.isinf(identity["psnr"]) or abs(identity["ssim"] - 1) > 1e-6:
        raise AssertionError("Identity PSNR/SSIM check failed")
    if abs(identity["lpips_alex"]) > 1e-6 or abs(offset["psnr"] - 20) > 1e-4:
        raise AssertionError("LPIPS identity or known-error PSNR check failed")
    return {"identity_ssim": identity["ssim"], "identity_lpips": identity["lpips_alex"],
            "constant_offset_psnr_expected": 20, "constant_offset_psnr_actual": offset["psnr"]}


def make_panel(items, width=384):
    height = width * 576 // 1024
    canvas = Image.new("RGB", (width * len(items), height + 30), "#17212b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for column, (label, path) in enumerate(items):
        with Image.open(path) as picture:
            canvas.paste(picture.convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
                         (column * width, 30))
        draw.text((column * width + 8, 6), label, fill="white", font=font)
    return canvas


def visualize(directory, models):
    geometry = directory / "geometry"
    items = [("Reference (real)", geometry / "ground_truth/000.png"),
             ("Geometry only", geometry / "conditions/024.png")]
    for method in models:
        endpoint = 15 if method == "viewcrafter16" else 24
        items.append((method, directory / method / "generated" / f"{endpoint:03d}.png"))
    items.append(("Target (real)", geometry / "ground_truth/024.png"))
    make_panel(items).save(directory / "endpoint_comparison.jpg", quality=94)
    preferred = "viewcrafter25" if "viewcrafter25" in models else "viewcrafter16"
    with imageio.get_writer(str(directory / "comparison.mp4"), fps=8, codec="libx264",
                            quality=9, macro_block_size=2) as writer:
        for frame_index, source_index in enumerate(COMMON_INDICES):
            generated_index = source_index if preferred == "viewcrafter25" else frame_index
            frames = [("Reference (real)", geometry / "ground_truth/000.png"),
                      ("Geometry only", geometry / "conditions" / f"{source_index:03d}.png"),
                      (preferred, directory / preferred / "generated" / f"{generated_index:03d}.png"),
                      ("Target (real)", geometry / "ground_truth" / f"{source_index:03d}.png")]
            writer.append_data(np.asarray(make_panel(frames)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(8)
    perceptual = lpips.LPIPS(net="alex", version="0.1").eval().to(args.device)
    checks = verify_metrics(perceptual, args.device)
    rows = []
    clips = sorted(path for path in args.root.iterdir() if (path / "geometry/run.json").exists())
    for directory in clips:
        geometry = directory / "geometry"
        source = load_rgb(geometry / "ground_truth/000.png")
        models = [method for method in ("viewcrafter16", "viewcrafter25")
                  if (directory / method / "run.json").exists()]
        if not models:
            raise ValueError(f"No completed model generation at {directory}")
        camera = json.loads((geometry / "cameras.json").read_text())
        for source_index in range(1, 25):
            truth = load_rgb(geometry / "ground_truth" / f"{source_index:03d}.png")
            predictions = {"copy_source": source,
                           "geometry_only": load_rgb(geometry / "conditions" / f"{source_index:03d}.png")}
            for method in models:
                if method == "viewcrafter16" and source_index not in COMMON_INDICES:
                    continue
                generated_index = COMMON_INDICES.index(source_index) if method == "viewcrafter16" else source_index
                predictions[method] = load_rgb(directory / method / "generated" / f"{generated_index:03d}.png")
            for method, prediction in predictions.items():
                for crop in ("full", "center_square"):
                    left = 224 if crop == "center_square" else 0
                    right = 800 if crop == "center_square" else 1024
                    metrics = score(prediction[:, left:right], truth[:, left:right], perceptual, args.device)
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise ValueError(f"Nonfinite metric: {directory.name}/{source_index}/{method}/{crop}")
                    rows.append({"clip": directory.name, "method": method, "crop": crop,
                                 "target_index": source_index, "common_target": source_index in COMMON_INDICES,
                                 "target_rotation_degrees": camera["rotation_from_source_degrees"][source_index],
                                 **metrics})
        visualize(directory, models)
        print(f"Scored and visualized {directory.name}", flush=True)
    groups = defaultdict(list)
    for row in rows:
        for subset in (["all_targets", "common_targets"] if row["common_target"] else ["all_targets"]):
            if subset == "all_targets" and row["method"] == "viewcrafter16":
                continue
            for clip_name in (row["clip"], "ALL"):
                groups[(clip_name, row["method"], row["crop"], subset)].append(row)
    aggregates = []
    for (clip, method, crop, subset), entries in sorted(groups.items()):
        aggregates.append({"clip": clip, "method": method, "crop": crop, "subset": subset,
                           "targets": len(entries),
                           **{metric: float(np.mean([entry[metric] for entry in entries])) for metric in METRICS}})
    result = {"protocol": "Fixed two-scene Tanks-and-Temples subset; not the paper test split",
              "common_target_indices": COMMON_INDICES[1:], "metric_checks": checks,
              "aggregates": aggregates, "per_frame": rows}
    (args.root / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    for filename, records in (("metrics_per_frame.csv", rows), ("metrics_summary.csv", aggregates)):
        with (args.root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps([row for row in aggregates if row["clip"] == "ALL" and row["subset"] == "common_targets"],
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
