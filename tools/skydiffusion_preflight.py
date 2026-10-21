import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "configs/skydiffusion_disaster_diagnostic.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def check_profile(profile, mode="diagnostic"):
    errors = []
    blockers = []
    checkpoint = profile["checkpoint"]
    checkpoint_path = local_path(checkpoint["path"])
    checkpoint_hash = None
    if not checkpoint_path.is_file():
        errors.append("Checkpoint file is missing")
    elif checkpoint_path.stat().st_size != checkpoint["bytes"]:
        errors.append("Checkpoint byte count does not match the locked profile")
    else:
        checkpoint_hash = sha256(checkpoint_path)
        if checkpoint_hash != checkpoint["sha256"]:
            errors.append("Checkpoint SHA256 does not match the locked profile")
    if checkpoint["training_task"] != profile["task"] or not checkpoint["paper_benchmark_verified"]:
        blockers.append("Task-matched, paper-verified checkpoint is unavailable")

    inputs = profile["inputs"]
    manifest = local_path(inputs["manifest"])
    rows = []
    if not manifest.is_file():
        errors.append("Input manifest is missing")
    else:
        if sha256(manifest) != inputs["manifest_sha256"]:
            errors.append("Input manifest SHA256 does not match the locked profile")
        with manifest.open() as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        if len(rows) != profile["sampling"]["count"] or any(len(row) < 3 for row in rows):
            errors.append("Input manifest count or column layout differs from the profile")
        names = [row[0] for row in rows if row]
        if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
            errors.append("Duplicate or unsafe sample names")
        for row in rows:
            if len(row) >= 3 and any(not Path(value).is_file() for value in row[1:3]):
                errors.append("Input files are missing for " + row[0])
    if inputs["split_status"] != "official_verified":
        blockers.append("Official regional test IDs/boundaries are unavailable; diagnostic samples are not a formal split")
    for category in ["train", "test"]:
        if inputs.get(category + "_ids") is None:
            if mode == "benchmark":
                errors.append("Benchmark requires official " + category + " IDs")
    if mode == "benchmark" and all(inputs.get(category + "_ids") for category in ["train", "test"]):
        train_ids = set(local_path(inputs["train_ids"]).read_text().splitlines())
        test_ids = set(local_path(inputs["test_ids"]).read_text().splitlines())
        if train_ids & test_ids or set(row[0] for row in rows) != test_ids:
            errors.append("Split overlap or incomplete official test coverage")

    geometry = profile["projection"]
    values = [geometry[key] for key in ["meter_per_pixel", "camera_height", "curve_height", "north_offset_deg"]]
    if not all(math.isfinite(value) for value in values) or geometry["meter_per_pixel"] <= 0 or geometry["curve_height"] < 0:
        errors.append("Projection parameters are invalid")
    if min(geometry["panorama_height"], geometry["panorama_width"]) <= 0:
        errors.append("Panorama dimensions are invalid")
    if not geometry["paper_parameters_verified"]:
        blockers.append("Disaster projection scale/height/curvature and training preprocessing are unconfirmed")
    if not profile["evaluation"]["paper_protocol_verified"]:
        blockers.append("Paper evaluation reference and FID protocol are unconfirmed")
    for filename, expected_hash in profile["local_code_sha256"].items():
        path = local_path(filename)
        if not path.is_file() or sha256(path) != expected_hash:
            errors.append("Local code differs from the locked profile: " + filename)
    if mode not in ["diagnostic", "benchmark"]:
        errors.append("Unknown execution mode")
    return {
        "mode": mode,
        "task": profile["task"],
        "checkpoint_sha256": checkpoint_hash,
        "manifest_count": len(rows),
        "errors": errors,
        "benchmark_blockers": blockers,
        "can_run": not errors and (mode == "diagnostic" or not blockers),
        "paper_reproduction_claim_allowed": not errors and not blockers and mode == "benchmark",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--mode", choices=["diagnostic", "benchmark"], default="diagnostic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_profile(json.loads(args.profile.read_text()), args.mode)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["can_run"] else 2)


if __name__ == "__main__":
    main()
