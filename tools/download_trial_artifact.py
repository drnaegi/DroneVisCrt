import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
from pathlib import Path

import requests

from network_dns import enable


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parallel_download(args, temporary):
    offset = temporary.stat().st_size if temporary.exists() else 0
    parts = args.output.with_suffix(args.output.suffix + ".parts")
    parts.mkdir(exist_ok=True)
    ranges = [(start, min(start + 128 * 1024**2, args.size) - 1)
              for start in range(offset, args.size, 128 * 1024**2)]

    def fetch_range(bounds):
        start, end = bounds
        destination = parts / str(start)
        if destination.exists() and destination.stat().st_size == end - start + 1:
            return destination
        for attempt in range(6):
            try:
                with requests.get(args.url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{args.size}":
                        raise RuntimeError("Unexpected parallel byte range")
                    with destination.open("wb") as handle:
                        for chunk in response.iter_content(4 * 1024**2):
                            handle.write(chunk)
                if destination.stat().st_size != end - start + 1:
                    raise RuntimeError("Incomplete parallel byte range")
                return destination
            except requests.RequestException:
                if attempt == 5:
                    raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_range, bounds) for bounds in ranges]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            print(f"Completed range {count}/{len(ranges)} after initial {offset / 1024**3:.3f} GiB", flush=True)
    with temporary.open("ab") as handle:
        for start, _ in ranges:
            with (parts / str(start)).open("rb") as chunk:
                shutil.copyfileobj(chunk, handle, 8 * 1024**2)
    if args.sha256 and digest_file(temporary) != args.sha256:
        raise RuntimeError("Parallel download SHA256 mismatch")
    temporary.rename(args.output)
    for start, _ in ranges:
        (parts / str(start)).unlink()
    parts.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--size", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    enable()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    if not args.output.exists() and args.workers > 1:
        if not args.size or not args.sha256:
            raise ValueError("Parallel download requires expected size and SHA256")
        parallel_download(args, temporary)
    if not args.output.exists():
        for attempt in range(8):
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with requests.get(args.url, headers=headers, stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        raise RuntimeError("Server did not honor resume request")
                    if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Unexpected resume range")
                    with temporary.open("ab" if offset else "wb") as handle:
                        last_report = time.monotonic()
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            handle.write(chunk)
                            offset += len(chunk)
                            if time.monotonic() - last_report > 15:
                                print(f"Downloaded {offset / 1024**3:.3f} GiB", flush=True)
                                last_report = time.monotonic()
                if args.size and temporary.stat().st_size != args.size:
                    raise RuntimeError(f"Unexpected size {temporary.stat().st_size}")
                actual = digest_file(temporary)
                if args.sha256 and actual != args.sha256:
                    raise RuntimeError("SHA256 mismatch")
                temporary.rename(args.output)
                break
            except requests.RequestException as error:
                print(f"Attempt {attempt + 1}: {type(error).__name__}", flush=True)
                if attempt == 7:
                    raise
                time.sleep(3)
    actual = digest_file(args.output)
    if args.sha256 and actual != args.sha256:
        raise RuntimeError("Existing file SHA256 mismatch")
    if args.size and args.output.stat().st_size != args.size:
        raise RuntimeError("Existing file size mismatch")
    metadata = {"url": args.url, "size": args.output.stat().st_size, "sha256": actual,
                "verified_publisher_sha256": bool(args.sha256)}
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
