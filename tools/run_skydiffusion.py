import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
import cv2

from skydiffusion_preflight import check_profile, local_path, sha256


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "code/skydiffusion"
sys.path.insert(0, str(REPOSITORY))

from cruvedbev import CruvedBEV
from ldm.util import instantiate_from_config


def save_rgb(tensor, path, quantization="round"):
    array = tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite image: " + str(path))
    pixels = np.clip((array + 1.0) / 2.0 * 255.0, 0, 255)
    if quantization == "round":
        pixels = pixels.round()
    pixels = pixels.astype(np.uint8)
    Image.fromarray(pixels).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/G2A-3/disaster_smoke/pairs.tsv")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/SkyDiffusion/CVACT_Skydiffusion.ckpt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", choices=["G2A-3", "CVACT", "CVUSA"], default="G2A-3")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--meter-per-pixel", type=float, default=0.06)
    parser.add_argument("--camera-height", type=float, default=-1.5)
    parser.add_argument("--curve-height", type=float, default=3.0)
    parser.add_argument("--north-offset-deg", type=float, default=0.0)
    parser.add_argument("--panorama-height", type=int)
    parser.add_argument("--panorama-width", type=int)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--quantization", choices=["round", "upstream"], default="round")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--mode", choices=["diagnostic", "benchmark"], default="diagnostic")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--save-reconstruction-reference", action="store_true")
    args = parser.parse_args()
    profile = None
    preflight = None
    checkpoint_hash = None
    if args.profile:
        allowed = {"--profile", "--mode", "--output", "--preflight-only", "--save-reconstruction-reference"}
        overrides = [argument.split("=", 1)[0] for argument in sys.argv[1:] if argument.startswith("--") and argument.split("=", 1)[0] not in allowed]
        if overrides:
            raise ValueError("Locked profile does not permit parameter overrides: " + str(overrides))
        profile = json.loads(args.profile.read_text())
        args.checkpoint = local_path(profile["checkpoint"]["path"])
        args.manifest = local_path(profile["inputs"]["manifest"])
        args.dataset = profile["dataset"]
        for section in ["sampling", "projection"]:
            for key in ["steps", "guidance", "seed", "count", "prompt", "negative_prompt", "precision", "quantization", "meter_per_pixel", "camera_height", "curve_height", "north_offset_deg", "panorama_height", "panorama_width"]:
                if key in profile[section]:
                    setattr(args, key, profile[section][key])
        preflight = check_profile(profile, args.mode)
        print(json.dumps(preflight, indent=2), flush=True)
        checkpoint_hash = preflight["checkpoint_sha256"]
        if not preflight["can_run"]:
            raise RuntimeError("Preflight failed; benchmark settings remain unverified")
    elif args.mode == "benchmark" or args.preflight_only:
        raise ValueError("Benchmark/preflight mode requires a locked --profile")
    if args.output is None:
        args.output = ROOT / "results" / ("skydiffusion_alignment_locked" if profile else "skydiffusion_disaster_smoke")
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.count < 1 or not 1 < args.steps < 1000 or args.meter_per_pixel <= 0 or args.curve_height < 0 or not np.isfinite([args.guidance, args.meter_per_pixel, args.camera_height, args.curve_height, args.north_offset_deg]).all():
        raise ValueError("Invalid count, step count or map scale")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.profile:
        (args.output / "locked_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    with args.manifest.open() as handle:
        rows = list(csv.reader(handle, delimiter="\t"))[:args.count]
    if not rows or any(len(row) < 3 for row in rows):
        raise ValueError("Expected name, source and target TSV columns")
    for row in rows:
        row[1:3] = [str(Path(filename).resolve()) for filename in row[1:3]]
        for filename in row[1:3]:
            if not Path(filename).is_file():
                raise FileNotFoundError(filename)

    config = OmegaConf.load(REPOSITORY / "models/lacldm_v15.yaml")
    config.model.params.cond_stage_config.params = {
        "version": str(ROOT / "checkpoints/clip-vit-large-patch14"),
        "load_pretrained": False,
    }
    started = time.monotonic()
    checkpoint_hash = checkpoint_hash or sha256(args.checkpoint)
    if args.checkpoint == ROOT / "checkpoints/SkyDiffusion/CVACT_Skydiffusion.ckpt" and checkpoint_hash != "a9108a3394fe127ae7cde9a75019b34ea11da9d3375a51efb4236200bbedfc6a":
        raise ValueError("Default checkpoint failed SHA256 verification")
    print("Checkpoint SHA256:", checkpoint_hash, flush=True)
    print("Building official LAControlLDM architecture", flush=True)
    model = instantiate_from_config(config.model).cpu()
    print("Loading checkpoint with weights_only=True", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatches = {name: {"expected": list(expected[name].shape), "actual": list(state_dict[name].shape)} for name in set(expected) & set(state_dict) if expected[name].shape != state_dict[name].shape}
    report = {"missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": mismatches, "model_keys": len(expected), "checkpoint_keys": len(state_dict)}
    (args.output / "checkpoint_compatibility.json").write_text(json.dumps(report, indent=2) + "\n")
    if missing or unexpected or mismatches:
        raise RuntimeError("Checkpoint is incompatible; see checkpoint_compatibility.json")
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict, expected
    model = model.eval().cuda()
    print("Strict checkpoint loading passed; model ready in %.1f s" % (time.monotonic() - started), flush=True)
    torch.cuda.reset_peak_memory_stats()

    records = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            name, street_path, aerial_path = row[:3]
            if Path(name).name != name:
                raise ValueError("Unsafe sample name")
            sample_seed = args.seed + index
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            sample_started = time.monotonic()
            bev = CruvedBEV(street_path, args.dataset, meter_per_pixel=args.meter_per_pixel, camera_height=args.camera_height, curve_height=args.curve_height, north_offset_deg=args.north_offset_deg, panorama_height=args.panorama_height, panorama_width=args.panorama_width)
            hint = torch.from_numpy(bev.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.precision == "fp16"):
                latent_hint = model.apply_condition_encoder(hint)
                text = model.get_learned_conditioning([args.prompt])
                condition = {"c_concat": [latent_hint], "c_crossattn": [text]}
                unconditional = None
                if args.guidance != 1.0:
                    negative = model.get_learned_conditioning([args.negative_prompt])
                    unconditional = {"c_concat": [latent_hint], "c_crossattn": [negative]}
                samples, _ = model.sample_log(
                    cond=condition, batch_size=1, ddim=True, ddim_steps=args.steps,
                    eta=0.0, unconditional_guidance_scale=args.guidance,
                    unconditional_conditioning=unconditional,
                )
                prediction = model.decode_first_stage(samples)
            generated_path = args.output / (name + "_generated.png")
            save_rgb(prediction, generated_path, args.quantization)
            Image.fromarray(bev).save(args.output / (name + "_bev.png"))
            torch.cuda.synchronize()
            elapsed = time.monotonic() - sample_started
            record = {"name": name, "street": street_path, "target": aerial_path, "generated": str(generated_path), "seed": sample_seed, "seconds": elapsed}
            if args.save_reconstruction_reference:
                target_pixels = np.asarray(Image.open(aerial_path).convert("RGB"))
                target_pixels = cv2.resize(target_pixels, (512, 512), interpolation=cv2.INTER_LINEAR)
                target_tensor = torch.from_numpy(target_pixels.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).cuda()
                target_latent = model.get_first_stage_encoding(model.encode_first_stage(target_tensor))
                reference_path = args.output / (name + "_reconstruction.png")
                save_rgb(model.decode_first_stage(target_latent), reference_path, args.quantization)
                record["reconstruction"] = str(reference_path)
            records.append(record)
            print("Generated %d/%d in %.1f s: %s" % (index + 1, len(rows), elapsed, name), flush=True)
            (args.output / "samples.json").write_text(json.dumps(records, indent=2) + "\n")

    metadata = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "official_code_commit": "461ccc2fff11882d48f5ab62cae65eef5c45c647",
        "checkpoint_source": profile["checkpoint"]["source"] if profile else "SkyDiff1109/SkyDiffusion_ckpt" if args.checkpoint == ROOT / "checkpoints/SkyDiffusion/CVACT_Skydiffusion.ckpt" else "user-supplied; provenance not verified",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_training_dataset_label": profile["checkpoint"]["training_task"] if profile else "CVACT" if args.checkpoint == ROOT / "checkpoints/SkyDiffusion/CVACT_Skydiffusion.ckpt" else "unknown",
        "evaluation_dataset": "SkyDiff/G2A-3" if args.dataset == "G2A-3" else args.dataset,
        "geometry_status": "Explicit projection parameters; training calibration is unverified" if not profile or not profile["projection"]["paper_parameters_verified"] else "profile-verified projection parameters",
        "target_used_for_conditioning": False,
        "profile_sha256": sha256(args.profile) if args.profile else None,
        "preflight": preflight,
        "mode": args.mode,
        "paper_reproduction_claim_allowed": bool(preflight and preflight["paper_reproduction_claim_allowed"]),
        "projection": {key: getattr(args, key) for key in ["meter_per_pixel", "camera_height", "curve_height", "north_offset_deg", "panorama_height", "panorama_width"]},
        "cfg_condition_branches_identical": args.prompt == args.negative_prompt,
        "reconstruction_reference_used_for_conditioning": False,
        "png_quantization": args.quantization,
        "sample_count": len(records),
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "torch_version": torch.__version__,
        "total_seconds": time.monotonic() - started,
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
