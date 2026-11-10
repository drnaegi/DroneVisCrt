import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def draw_panel(canvas, image, label, column, row, panel_size):
    panel_width, panel_height = panel_size
    top = row * (panel_height + 36)
    left = column * panel_width
    font = ImageFont.truetype(FONT_PATH, 19)
    ImageDraw.Draw(canvas).text((left + 10, top + 7), label, font=font, fill="white")
    image = image.copy()
    image.thumbnail(panel_size, Image.Resampling.LANCZOS)
    canvas.paste(image, (left + (panel_width - image.width) // 2, top + 36))


def comparison(root, sample, output, crop=None):
    panel_size = (480, 300 if crop else 270)
    canvas = Image.new("RGB", (panel_size[0] * 3, (panel_size[1] + 36) * 2), "#141b26")
    for row, method in enumerate(("sharp", "moge2")):
        directory = root / method / sample
        labels = ["Source" if crop is None else "Source: fixed image crop",
                  f"{method.upper()} | x = +0.05", f"{method.upper()} | x = +0.15"]
        for column, (name, label) in enumerate(zip(
            ("input.png", "view_+0.05.png", "view_+0.15.png"), labels
        )):
            with Image.open(directory / name) as image:
                if crop is not None:
                    image = image.crop(crop)
                draw_panel(canvas, image, label, column, row, panel_size)
    canvas.save(output, quality=95)


def video_comparison(root, sample, output):
    readers = [imageio.get_reader(root / method / sample / "nearby_views.mp4")
               for method in ("sharp", "moge2")]
    try:
        with imageio.get_writer(str(output), fps=16, codec="libx264", quality=8,
                               macro_block_size=2) as writer:
            for sharp_frame, moge_frame in zip(*readers):
                canvas = Image.new("RGB", (1440, 306), "#141b26")
                with Image.open(root / "sharp" / sample / "input.png") as source:
                    draw_panel(canvas, source, "Source (fixed)", 0, 0, (480, 270))
                draw_panel(canvas, Image.fromarray(sharp_frame), "SHARP", 1, 0, (480, 270))
                draw_panel(canvas, Image.fromarray(moge_frame), "MoGe-2 + mesh", 2, 0, (480, 270))
                writer.append_data(np.array(canvas))
    finally:
        for reader in readers:
            reader.close()


def validate_sample(directory, sample):
    expected_size = tuple(sample["resolution"])
    image_count = 0
    for path in directory.glob("*.png"):
        with Image.open(path) as image:
            image.load()
            if image.size != expected_size:
                raise ValueError(f"Unexpected resolution: {path}: {image.size}")
            image_count += 1
    depth = np.load(directory / "depth.npy")
    valid_depth = np.isfinite(depth) & (depth > 0)
    if valid_depth.shape != expected_size[::-1] or not valid_depth.any():
        raise ValueError(f"Invalid depth: {directory}")
    frame_count = 0
    with imageio.get_reader(directory / "nearby_views.mp4") as reader:
        metadata = reader.get_meta_data()
        for frame in reader:
            if frame.shape[1::-1] != tuple(sample["video_encoded_resolution"]):
                raise ValueError(f"Unexpected video resolution: {directory}")
            frame_count += 1
    if frame_count != sample["video_frames"]:
        raise ValueError(f"Incomplete video: {directory}: {frame_count}")
    trajectory = json.loads((directory / "trajectory.json").read_text())
    if len(trajectory) != frame_count:
        raise ValueError(f"Trajectory/frame mismatch: {directory}")
    for view in sample["views"]:
        name = f"view_{view['offset']:+.2f}_coverage.png"
        with Image.open(directory / name) as image:
            uncovered = float((np.array(image) < 229.5).mean())
        if abs(uncovered - view["low_coverage_fraction"]) > 0.002:
            raise ValueError(f"Coverage/metric mismatch: {directory / name}")
    return {"sample": directory.name, "png_count": image_count, "decoded_frames": frame_count,
            "fps": metadata["fps"], "valid_depth_fraction": float(valid_depth.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    comparisons = args.root / "comparisons"
    comparisons.mkdir(exist_ok=True)
    rows = []
    summary = {}
    validation = {}
    sample_names = {}
    for method in ("sharp", "moge2"):
        manifest = json.loads((args.root / method / "run_manifest.json").read_text())
        sample_names[method] = [Path(sample["input"]).stem for sample in manifest["samples"]]
        validation[method] = []
        for sample in manifest["samples"]:
            name = Path(sample["input"]).stem
            validation[method].append(validate_sample(args.root / method / name, sample))
            rows.append({"method": method, "sample": name,
                         "uav": name != "demo_traffic",
                         "inference_seconds": sample["inference_seconds"],
                         "inference_peak_allocated_gib": sample["inference_peak_allocated_gib"],
                         "total_sample_seconds": sample["total_sample_seconds"],
                         "source_reconstruction_mae_0_255": sample["views"][0]["source_reconstruction_mae_0_255"],
                         "uncovered_source": sample["views"][0]["low_coverage_fraction"],
                         "uncovered_plus_005": sample["views"][2]["low_coverage_fraction"],
                         "uncovered_plus_015": sample["views"][4]["low_coverage_fraction"]})
        uav_rows = [row for row in rows if row["method"] == method and row["uav"]]
        summary[method] = {"uav_count": len(uav_rows), "all_count": len(manifest["samples"])}
        for field in ("inference_seconds", "inference_peak_allocated_gib",
                      "total_sample_seconds", "source_reconstruction_mae_0_255"):
            values = [row[field] for row in uav_rows]
            summary[method][field] = {"mean": float(np.mean(values)),
                                      "min": min(values), "max": max(values)}
    if sample_names["sharp"] != sample_names["moge2"]:
        raise ValueError("The two methods have different samples")
    for sample in sample_names["sharp"]:
        comparison(args.root, sample, comparisons / f"{sample}.jpg")
        video_comparison(args.root, sample, comparisons / f"{sample}.mp4")
    comparison(args.root, "0000002_00005_d_0000014", comparisons / "parking_detail.jpg",
               crop=(280, 240, 700, 520))
    with (args.root / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.root / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
