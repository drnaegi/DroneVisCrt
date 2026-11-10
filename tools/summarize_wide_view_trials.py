import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def panel(image, label, width=512):
    height = round(image.height * width / image.width)
    canvas = Image.new("RGB", (width, height + 36), "#15202b")
    canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS), (0, 36))
    ImageDraw.Draw(canvas).text((12, 8), label, fill="white", font=ImageFont.truetype(FONT, 18))
    return canvas


def row(images, labels):
    panels = [panel(image, label) for image, label in zip(images, labels)]
    canvas = Image.new("RGB", (sum(image.width for image in panels), panels[0].height))
    left = 0
    for image in panels:
        canvas.paste(image, (left, 0))
        left += image.width
    return canvas


def process(directory):
    metadata = json.loads((directory / "run.json").read_text())
    cameras = json.loads((directory / "cameras.json").read_text())
    inputs = Image.open(directory / "input.png").convert("RGB")
    conditions = [Image.open(path).convert("RGB") for path in sorted((directory / "conditions").glob("*.png"))]
    generated = [Image.open(path).convert("RGB") for path in sorted((directory / "generated").glob("*.png"))]
    if len(conditions) != 16 or len(generated) != 16:
        raise ValueError(f"Incomplete output: {directory}")
    labels = ["Source image", "Geometry condition: endpoint", "ViewCrafter: generated endpoint"]
    row([inputs, conditions[-1], generated[-1]], labels).save(directory / "endpoint_comparison.jpg", quality=95)
    row([inputs, generated[7], generated[-1]],
        ["Source image", "Generated: frame 7", "Generated: frame 15"]).save(directory / "keyframes.jpg", quality=95)
    flicker_frames = [panel(inputs, "SOURCE", 800), panel(generated[-1], "GENERATED ENDPOINT", 800)]
    flicker_frames[0].save(directory / "source_vs_endpoint.gif", save_all=True,
                           append_images=flicker_frames[1:], duration=[1200, 1200], loop=0)
    with imageio.get_writer(str(directory / "comparison.mp4"), fps=8, codec="libx264",
                           quality=9, macro_block_size=2) as writer:
        for index, (condition, prediction) in enumerate(zip(conditions, generated)):
            canvas = row([inputs, condition, prediction],
                         ["Source (fixed)", f"Geometry condition: {index:02d}", f"ViewCrafter: {index:02d}"])
            writer.append_data(np.asarray(canvas))
    decoded = {}
    for name in ("conditions.mp4", "generated.mp4", "comparison.mp4"):
        with imageio.get_reader(directory / name) as reader:
            frame_count = sum(1 for _ in reader)
        if frame_count != 16:
            raise ValueError(f"Video frame count mismatch: {directory / name}")
        decoded[name] = frame_count
    return {
        "experiment": directory.name, "phi": metadata["phi"], "theta": metadata["theta"],
        "requested_rotation_degrees": float(np.hypot(metadata["phi"], metadata["theta"])),
        "camera_endpoint_rotation_degrees": cameras["endpoint_rotation_degrees"],
        "baseline_over_radius": cameras["baseline_over_radius"],
        "geometry_seconds": metadata["geometry_seconds"],
        "diffusion_seconds": metadata["diffusion_seconds"],
        "diffusion_peak_allocated_gib": metadata["diffusion_peak_allocated_gib"],
        "source_reconstruction_mae_0_255": metadata["source_reconstruction_mae_0_255"],
        "decoded_frames": decoded,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for directory in sorted(args.root.iterdir()):
        if directory.is_dir() and (directory / "generated/015.png").exists():
            summaries.append(process(directory))
    if not summaries:
        raise ValueError("No completed generations")
    for scene in ("parking", "buildings"):
        directories = [args.root / f"{scene}_{motion}" for motion in ("phi30", "theta_minus20")]
        if all((directory / "generated/015.png").exists() for directory in directories):
            source = Image.open(directories[0] / "input.png")
            lateral = Image.open(directories[0] / "generated/015.png")
            vertical = Image.open(directories[1] / "generated/015.png")
            row([source, lateral, vertical],
                ["Source", "Generated: azimuth +30 deg", "Generated: elevation +20 deg"]).save(
                    args.root / f"{scene}_directions.jpg", quality=95)
    (args.root / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
