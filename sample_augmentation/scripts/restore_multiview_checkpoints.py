"""Download verified Release archives in ranges and restore to an isolated directory."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_range(endpoint, start, stop):
    for attempt in range(3):
        try:
            result = subprocess.run(["gh", "api", "-H", "Accept: application/octet-stream",
                                     "-H", f"Range: bytes={start}-{stop - 1}", endpoint],
                                    capture_output=True, timeout=180, check=True)
            if len(result.stdout) != stop - start:
                raise ValueError("Unexpected range response size")
            return start, result.stdout
        except (subprocess.SubprocessError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    manifest = json.loads((ROOT / "CHECKPOINTS_MANIFEST.json").read_text())
    repository, tag = manifest["repository"], manifest["tag"]
    release = json.loads(subprocess.check_output(["gh", "api", f"repos/{repository}/releases/tags/{tag}"]))
    asset_ids = {asset["name"]: asset["id"] for asset in release["assets"]}
    folder = ROOT / "checkpoint_downloads"
    folder.mkdir(exist_ok=True)
    destination = folder / "restored"
    required = [asset for asset in manifest["assets"] if asset["name"].startswith("multiview-")]
    for asset in required:
        path = folder / asset["name"]
        size = asset["size_bytes"]
        verified = path.is_file() and path.stat().st_size == size and digest(path) == asset["sha256"]
        if not verified:
            prefix = path.stat().st_size if path.exists() and path.stat().st_size < size else 0
            partial = path.with_suffix(".part")
            if prefix:
                path.replace(partial)
            endpoint = f"repos/{repository}/releases/assets/{asset_ids[asset['name']]}"
            block = 2 * 1024 * 1024
            with partial.open("r+b" if prefix else "w+b") as stream:
                stream.truncate(size)
                completed = prefix
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    pending = [executor.submit(fetch_range, endpoint, start, min(start + block, size))
                               for start in range(prefix, size, block)]
                    for future in as_completed(pending):
                        start, data = future.result()
                        stream.seek(start)
                        stream.write(data)
                        completed += len(data)
                        print(asset["name"], completed, "/", size, flush=True)
            if digest(partial) != asset["sha256"]:
                raise ValueError("Archive SHA256 mismatch: " + asset["name"])
            partial.replace(path)
        with zipfile.ZipFile(path) as archive:
            expected = {item["path"]: item for item in asset["files"]}
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if member.filename not in expected:
                    raise ValueError("Unexpected archive member: " + member.filename)
                target = destination / member.filename
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Archive path escapes destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                contents = archive.read(member)
                entry = expected[member.filename]
                if len(contents) != entry["size_bytes"] or hashlib.sha256(contents).hexdigest() != entry["sha256"]:
                    raise ValueError("Extracted file checksum mismatch: " + member.filename)
                if target.exists() and digest(target) != entry["sha256"]:
                    raise FileExistsError("Refusing to overwrite changed restored file: " + str(target))
                if not target.exists():
                    target.write_bytes(contents)
            if not set(expected).issubset(archive.namelist()):
                raise ValueError("Archive is missing manifest entries")
        print("VERIFIED", asset["name"], flush=True)
    print("Restored source:", destination / "outputs/multiview_a10", flush=True)


if __name__ == "__main__":
    main()
