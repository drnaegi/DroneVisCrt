import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests

from network_dns import enable


ROOT = Path("/root/DroneVisCrt/checkpoints/SkyDiffusion")
FILENAME = "CVACT_Skydiffusion.ckpt"
SIZE = 12811886329
SHA256 = "a9108a3394fe127ae7cde9a75019b34ea11da9d3375a51efb4236200bbedfc6a"
REVISION = "31b574c1750cc6bb8e285d566b381a0946db9ee2"
URL = "https://hf-mirror.com/SkyDiff1109/SkyDiffusion_ckpt/resolve/main/" + FILENAME
CHUNK_SIZE = 64 * 1024 * 1024


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    enable()
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / FILENAME
    partial = ROOT / (FILENAME + ".part")
    state_path = ROOT / (FILENAME + ".progress.json")
    if target.exists():
        if target.stat().st_size != SIZE or checksum(target) != SHA256:
            raise ValueError("Existing checkpoint failed checksum")
        print("Checkpoint already verified", flush=True)
        return
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["sha256"] != SHA256 or not partial.exists():
            raise ValueError("Resume state does not match checkpoint")
    else:
        candidates = list((ROOT / ".cache/huggingface/download").glob("*." + SHA256 + ".incomplete"))
        if candidates and not partial.exists():
            candidates[0].rename(partial)
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > SIZE:
            raise ValueError("Partial checkpoint is too large")
        state = {"sha256": SHA256, "size": SIZE, "revision": REVISION, "prefix_bytes": prefix, "completed_chunks": []}

    def save_state():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    save_state()
    descriptor = os.open(str(partial), os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(descriptor, SIZE)
    completed = set(state["completed_chunks"])
    ranges = [(start, min(start + CHUNK_SIZE, SIZE) - 1) for start in range(state["prefix_bytes"], SIZE, CHUNK_SIZE) if start not in completed]
    with requests.head(URL, allow_redirects=False, timeout=(15, 30)) as response:
        response.raise_for_status()
        linked_hash = response.headers.get("X-Linked-ETag", "").strip('"')
        if linked_hash != SHA256:
            raise ValueError("Remote checkpoint hash changed: " + linked_hash)
        download_url = response.headers["Location"]
    print("Resuming contiguous prefix:", state["prefix_bytes"], "pending ranges:", len(ranges), flush=True)
    started = time.monotonic()

    def fetch_range(bounds):
        start, end = bounds
        for attempt in range(4):
            try:
                with requests.get(download_url, headers={"Range": "bytes=%d-%d" % (start, end), "Accept-Encoding": "identity"}, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    expected_range = "bytes %d-%d/%d" % (start, end, SIZE)
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Server returned an incorrect range: " + str(response.headers.get("Content-Range")))
                    position = start
                    for block in response.iter_content(1024 * 1024):
                        if position + len(block) > end + 1:
                            raise ValueError("Too many bytes in range response")
                        written = os.pwrite(descriptor, block, position)
                        if written != len(block):
                            raise OSError("Short file write")
                        position += written
                    if position != end + 1:
                        raise ValueError("Incomplete range response")
                return start, end - start + 1
            except (requests.RequestException, OSError, ValueError):
                if attempt == 3:
                    raise
        raise RuntimeError("Range retry exhausted")

    transferred = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch_range, bounds) for bounds in ranges]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                start, byte_count = future.result()
                os.fsync(descriptor)
                state["completed_chunks"].append(start)
                save_state()
                transferred += byte_count
                if index % 8 == 0 or index == len(futures):
                    print("Completed %d/%d ranges; %.1f MB/s" % (index, len(futures), transferred / (time.monotonic() - started) / 1e6), flush=True)
    finally:
        os.close(descriptor)
    actual_hash = checksum(partial)
    if actual_hash != SHA256:
        raise ValueError("SHA256 mismatch: " + actual_hash)
    partial.replace(target)
    (ROOT / (FILENAME + ".sha256")).write_text(SHA256 + "  " + FILENAME + "\n")
    print("Verified SHA256:", actual_hash, flush=True)
    print(target, flush=True)


if __name__ == "__main__":
    main()
