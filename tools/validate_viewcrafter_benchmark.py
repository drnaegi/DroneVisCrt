import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    expected_clips = {"train_stride1", "train_stride3", "truck_stride1", "truck_stride3"}
    indices = np.rint(np.linspace(0, 24, 16)).astype(int).tolist()
    metrics = json.loads((args.root / "metrics.json").read_text())
    assert {row["clip"] for row in metrics["per_frame"]} == expected_clips
    images_checked = 0
    videos_checked = 0
    video_frames_checked = 0
    for clip in sorted(expected_clips):
        directory = args.root / clip
        geometry = directory / "geometry"
        metadata = json.loads((geometry / "run.json").read_text())
        for filename, expected_digest in metadata["input_sha256"].items():
            source = Path(metadata["input_directory"]) / filename
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
        conditions = np.load(geometry / "conditions.npy")
        assert conditions.shape == (25, 576, 1024, 3) and np.isfinite(conditions).all()
        framesets = [(geometry / "conditions", 25), (geometry / "ground_truth", 25)]
        videos = [(geometry / "conditions.mp4", 25), (geometry / "ground_truth.mp4", 25),
                  (directory / "comparison.mp4", 16)]
        for length in (16, 25):
            model = directory / f"viewcrafter{length}"
            run = json.loads((model / "run.json").read_text())
            assert run["generated_frames"] == length
            expected_indices = indices if length == 16 else list(range(25))
            assert run["source_frame_indices"] == expected_indices
            assert np.array_equal(np.load(model / "conditions.npy"), conditions[expected_indices])
            framesets.append((model / "generated", length))
            videos.append((model / "generated.mp4", length))
        for frame_directory, expected_count in framesets:
            frames = sorted(frame_directory.glob("*.png"))
            assert len(frames) == expected_count
            for frame in frames:
                with Image.open(frame) as picture:
                    picture.load()
                    assert picture.size == (1024, 576) and picture.mode == "RGB"
                images_checked += 1
        for video, expected_count in videos:
            with imageio.get_reader(str(video)) as reader:
                count = sum(1 for _ in reader)
            assert count == expected_count, (video, count)
            videos_checked += 1
            video_frames_checked += count
    common = [row for row in metrics["aggregates"] if row["clip"] == "ALL"
              and row["subset"] == "common_targets"]
    assert len(common) == 8
    for aggregate in common:
        assert aggregate["targets"] == 60
        rows = [row for row in metrics["per_frame"] if row["common_target"]
                and row["method"] == aggregate["method"] and row["crop"] == aggregate["crop"]]
        assert len(rows) == 60
        for metric in ("psnr", "ssim", "lpips_alex"):
            assert abs(np.mean([row[metric] for row in rows]) - aggregate[metric]) < 1e-10
    result = {"status": "passed", "clips": sorted(expected_clips), "pngs_checked": images_checked,
              "videos_decoded": videos_checked, "video_frames_decoded": video_frames_checked,
              "common_targets_per_method": 60, "input_hashes_checked": True,
              "shared_geometry_conditions_identical": True, "aggregate_means_checked": True}
    (args.root / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
