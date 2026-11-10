import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import requests

from network_dns import enable


DATASET = "nvs-bench/tanksandtemples"
REVISION = "f5c9529539ab3e93dc33362f84034c6704bd7256"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enable()
    args.output.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://hf-mirror.com/api/datasets/{DATASET}/tree/{REVISION}?recursive=true&limit=1000",
        timeout=60,
    )
    response.raise_for_status()
    entries = response.json()
    clips = []
    selected = {}
    for scene in ("train", "truck"):
        images = sorted(
            [entry for entry in entries if entry["path"].startswith(f"{scene}/images/")],
            key=lambda entry: entry["path"],
        )
        for stride in (1, 3):
            chosen = [images[index * stride] for index in range(25)]
            clips.append({
                "name": f"{scene}_stride{stride}", "scene": scene, "stride": stride,
                "source_index": 0, "frames": [entry["path"] for entry in chosen],
            })
            selected.update({entry["path"]: entry for entry in chosen})

    def fetch(entry):
        destination = args.output / "raw" / entry["path"]
        expected = entry["lfs"]["oid"]
        url = f"https://hf-mirror.com/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}"
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            content = destination.read_bytes()
        else:
            for attempt in range(4):
                try:
                    result = requests.get(url, timeout=(20, 60))
                    result.raise_for_status()
                    content = result.content
                    if hashlib.sha256(content).hexdigest() != expected:
                        raise ValueError(f"Publisher SHA256 mismatch: {entry['path']}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
        return {"path": entry["path"], "sha256": expected, "bytes": len(content), "url": url}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        files = list(executor.map(fetch, selected.values()))
    for clip in clips:
        directory = args.output / "clips" / clip["name"]
        directory.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(clip["frames"]):
            destination = directory / f"{index:03d}.jpg"
            source_path = (args.output / "raw" / source).resolve()
            if not destination.exists():
                destination.symlink_to(source_path)
    manifest = {
        "dataset": "Tanks-and-Temples, processed Train and Truck scenes",
        "mirror_repository": DATASET, "revision": REVISION,
        "split": "Local deterministic subset; not the paper's undisclosed Easy/Hard split",
        "selection": "First sorted image as source; 25 images with stride 1 or 3; fixed before inference",
        "clips": clips, "files": sorted(files, key=lambda entry: entry["path"]),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified {len(files)} source images; prepared {len(clips)} clips", flush=True)


if __name__ == "__main__":
    main()
