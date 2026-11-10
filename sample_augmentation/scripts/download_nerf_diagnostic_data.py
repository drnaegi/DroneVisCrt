"""Fetch the original NeRF example archive (Lego and Fern) with recorded hashes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import requests


URL = "https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_range(start, stop, total, folder):
    part = folder / str(start)
    if part.exists() and part.stat().st_size == stop - start:
        return part
    for attempt in range(3):
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(URL, headers={"Range": f"bytes={start}-{stop - 1}"}, timeout=(15, 90))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop - 1}/{total}":
                    raise ValueError("Unexpected range response")
                if len(response.content) != stop - start:
                    raise ValueError("Incomplete range")
                temporary = part.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(part)
                return part
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/nerf_diagnostic"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "nerf_example_data.zip"
    manifest_path = args.output / "source_manifest.json"
    with requests.Session() as session:
        session.trust_env = False
        response = session.head(URL, timeout=30)
        response.raise_for_status()
        size = int(response.headers["Content-Length"])
        metadata = {"url": URL, "size_bytes": size, "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "source": "bmild/nerf download_example_data.sh", "publisher_sha256_available": False}
    if not archive.exists():
        parts = args.output / "download_parts"
        parts.mkdir(exist_ok=True)
        block = 4 * 1024**2
        ranges = [(start, min(start + block, size)) for start in range(0, size, block)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(fetch_range, start, stop, size, parts) for start, stop in ranges]
            for count, future in enumerate(as_completed(pending), 1):
                future.result()
                print(f"Downloaded {count}/{len(ranges)} ranges", flush=True)
        temporary = archive.with_suffix(".part")
        with temporary.open("wb") as stream:
            for start, _ in ranges:
                with (parts / str(start)).open("rb") as source:
                    shutil.copyfileobj(source, stream)
        temporary.replace(archive)
    if archive.stat().st_size != size:
        raise ValueError("Archive length mismatch")
    metadata["sha256"] = sha256(archive)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["sha256"] != metadata["sha256"]:
            raise ValueError("Source archive changed")
    files = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir() or member.filename.startswith("__MACOSX/"):
                continue
            target = args.output / member.filename
            if not target.resolve().is_relative_to(args.output.resolve()):
                raise ValueError("Unsafe archive member")
            contents = bundle.read(member)
            checksum = hashlib.sha256(contents).hexdigest()
            if target.exists():
                if sha256(target) != checksum:
                    raise FileExistsError("Existing dataset file differs: " + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
            files.append({"path": member.filename, "bytes": len(contents), "sha256": checksum})
    metadata["files"] = files
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "files"}, indent=2))
    print("Extracted", len(files), "files", flush=True)


if __name__ == "__main__":
    main()
