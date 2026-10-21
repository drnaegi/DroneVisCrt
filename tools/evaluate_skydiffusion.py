import argparse
import json
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def upstream_metrics(target, prediction):
    mse = float(np.mean((prediction - target) ** 2))
    return {
        "psnr_rgb_capped40_db": float(10 * np.log(1.0 / max(mse, 0.0001)) / np.log(10)),
        "ssim_y": float(structural_similarity(color.rgb2yuv(target)[:, :, 0], color.rgb2yuv(prediction)[:, :, 0], data_range=1.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results/skydiffusion_disaster_smoke")
    parser.add_argument("--profile", choices=["rgb", "upstream"], default="rgb")
    parser.add_argument("--reference", choices=["target", "vae_reconstruction"], default="target")
    args = parser.parse_args()
    samples = json.loads((args.results / "samples.json").read_text())
    if not samples:
        raise ValueError("No generated samples")
    torch.set_num_threads(8)
    metric = lpips.LPIPS(net="alex").eval().cuda()
    records = []
    canvas = Image.new("RGB", (4 * 256, len(samples) * 284), "white")
    draw = ImageDraw.Draw(canvas)
    with torch.no_grad():
        for index, sample in enumerate(samples):
            prediction = np.asarray(Image.open(sample["generated"]).convert("RGB"), dtype=np.float32) / 255.0
            reference_path = sample["target"] if args.reference == "target" else sample.get("reconstruction")
            if reference_path is None:
                raise ValueError("VAE reconstruction reference is missing for " + sample["name"])
            target_pixels = np.asarray(Image.open(reference_path).convert("RGB"))
            target_resized = target_pixels.shape != prediction.shape
            if target_resized:
                target_pixels = cv2.resize(target_pixels, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_LINEAR)
            target = target_pixels.astype(np.float32) / 255.0
            if target.shape != prediction.shape:
                raise ValueError("Prediction and target shapes differ")
            prediction_tensor = torch.from_numpy(prediction).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
            target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
            record = {
                "name": sample["name"],
                "psnr_db": float(peak_signal_noise_ratio(target, prediction, data_range=1.0)),
                "ssim_rgb": float(structural_similarity(target, prediction, data_range=1.0, channel_axis=-1)),
                "lpips_alex": float(metric(prediction_tensor, target_tensor).item()),
                "target_resized": target_resized,
                "reference": str(reference_path),
            }
            record.update(upstream_metrics(target, prediction))
            records.append(record)
            images = [Image.open(sample["street"]), Image.open(args.results / (sample["name"] + "_bev.png")), Image.open(sample["generated"]), Image.open(sample["target"])]
            for column, (label, image) in enumerate(zip(["Street", "Curved-BEV", "Generated", "Ground truth"], images)):
                image.thumbnail((256, 256))
                origin = (column * 256, index * 284)
                draw.text((origin[0] + 4, origin[1] + 4), label, fill="black")
                canvas.paste(image, (origin[0], origin[1] + 22))
            print(json.dumps(record), flush=True)
    keys = ["psnr_db", "ssim_rgb", "lpips_alex", "psnr_rgb_capped40_db", "ssim_y"]
    means = {key: float(np.mean([record[key] for record in records])) for key in keys}
    report = {
        "sample_count": len(records), "mean": means, "per_sample": records,
        "definitions": {
            "psnr_db": "skimage PSNR, RGB [0,1], data_range=1, no 40 dB cap",
            "ssim_rgb": "skimage SSIM, RGB [0,1], channel_axis=-1, default uniform 7x7 window",
            "lpips_alex": "lpips 0.1.4, pretrained AlexNet, RGB [-1,1]",
            "target_resize": "only if needed: OpenCV INTER_LINEAR to generated image size; G2A-3 targets are already 512x512",
            "psnr_rgb_capped40_db": "matches upstream eval/eval.py: RGB float32 MSE, eps=0.0001, max 40 dB",
            "ssim_y": "matches upstream eval/eval.py: skimage rgb2yuv Y, data_range=1, default uniform 7x7 window",
        },
        "evaluation_profile": args.profile,
        "reference": args.reference,
        "primary_metrics": {
            "psnr_db": means["psnr_db"] if args.profile == "rgb" else means["psnr_rgb_capped40_db"],
            "ssim": means["ssim_rgb"] if args.profile == "rgb" else means["ssim_y"],
            "lpips_alex": means["lpips_alex"],
        },
        "paper_protocol_verified": False,
        "lpips_status": "AlexNet matches active eval/evaluation.py; eval/eval.py comments out LPIPS; paper table protocol is unconfirmed",
        "reference_status": "eval/evaluation.py uses *_reconstruction.png (VAE decoded target), not original target PNG; paper reference is unconfirmed",
        "scope": "diagnostic sample evaluation; not paper benchmark scores",
        "fid": "not computed: sample count is too small for a useful FID comparison",
    }
    filename = "metrics.json" if args.profile == "rgb" else "metrics_upstream.json"
    if args.reference == "vae_reconstruction":
        filename = filename.replace(".json", "_reconstruction.json")
    (args.results / filename).write_text(json.dumps(report, indent=2) + "\n")
    if args.profile == "rgb" and args.reference == "target":
        canvas.save(args.results / "comparison.jpg", quality=95)
    print(json.dumps({"sample_count": len(records), "mean": means}, indent=2))


if __name__ == "__main__":
    main()
