import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("/root/DroneVisCrt/datasets/G2A-3/Disaster.zip"))
    parser.add_argument("--output", type=Path, default=Path("/root/DroneVisCrt/datasets/G2A-3/disaster_smoke"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(args.archive) as archive:
        sources = {Path(name).stem: name for name in archive.namelist() if name.startswith("disaster/SVI/") and name.endswith(".png")}
        targets = {Path(name).stem: name for name in archive.namelist() if name.startswith("disaster/Arieal/") and name.endswith(".png")}
        if sources.keys() != targets.keys():
            raise ValueError("Street and aerial filenames do not match")
        if not 0 < args.count <= len(sources):
            raise ValueError("count is out of range")
        selected = sorted(random.Random(args.seed).sample(sorted(sources), args.count))
        for name in selected:
            paths = []
            for category, member, expected_size in [("street", sources[name], (1024, 512)), ("aerial", targets[name], (512, 512))]:
                path = args.output / category / (name + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(member))
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if image.size != expected_size or image.mode != "RGB":
                        raise ValueError("Unexpected image format: " + str(path))
                paths.append(str(path.resolve()))
            rows.append([name, *paths])
    manifest = args.output / "pairs.tsv"
    with manifest.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)
    metadata = {"dataset": "SkyDiff/G2A-3", "subset": "Disaster", "available_pairs": len(sources), "selected_pairs": len(rows), "seed": args.seed, "purpose": "smoke test, not an official evaluation split"}
    (args.output / "subset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(manifest)


if __name__ == "__main__":
    main()
