import argparse
import concurrent.futures
import hashlib
import subprocess
from pathlib import Path

import requests

from network_dns import enable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    enable()
    revision = subprocess.check_output(["git", "-C", str(args.directory), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(args.directory), "ls-tree", "-r", "HEAD"], text=True)
    files = []
    for entry in tree.splitlines():
        metadata, name = entry.split("\t", 1)
        _, kind, digest = metadata.split()
        path = Path(name)
        if kind != "blob" or any(part in {"assets", "__pycache__", "build", "tests", "docs"} for part in path.parts):
            continue
        if path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".txt", ".cpp", ".cu", ".h", ".cuh", ".in", ".md"} or path.name in {"LICENSE", "NOTICE"} or name == "test/images/boy.png":
            files.append((name, digest))

    def fetch(item):
        name, expected = item
        destination = args.directory / name
        def verify(content):
            return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == expected
        if destination.exists() and verify(destination.read_bytes()):
            return name
        urls = [f"https://cdn.jsdelivr.net/gh/{args.repository}@{revision}/{name}",
                f"https://raw.githubusercontent.com/{args.repository}/{revision}/{name}"]
        for attempt in range(4):
            try:
                response = requests.get(urls[attempt % 2], timeout=(10, 25))
                response.raise_for_status()
                if not verify(response.content):
                    raise RuntimeError(f"Git blob mismatch: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(response.content)
                return name
            except requests.RequestException:
                if attempt == 3:
                    raise

    print(f"Fetching {len(files)} verified blobs at {revision}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(fetch, item) for item in files]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if count % 20 == 0 or count == len(files):
                print(f"Verified {count}/{len(files)}", flush=True)


if __name__ == "__main__":
    main()
