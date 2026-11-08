import concurrent.futures
import hashlib
from pathlib import Path
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent
URL = "https://github.com/microsoft/AirSim/releases/download/v1.8.1-windows/Blocks.zip"
SIZE = 259463081
CHUNK_SIZE = 4 * 1024 * 1024
download_url = URL


def download_chunk(index):
    start = index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, SIZE) - 1
    destination = ROOT / "downloads" / f"Blocks.part{index:03d}"
    if destination.exists() and destination.stat().st_size == end - start + 1:
        return destination
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                download_url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                expected_range = f"bytes {start}-{end}/{SIZE}"
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"Unexpected response for chunk {index}")
                data = response.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"Incomplete chunk {index}: {len(data)}")
            destination.write_bytes(data)
            return destination
        except Exception:
            if attempt == 5:
                raise
            time.sleep((attempt + 1) * 3)


def main():
    global download_url
    (ROOT / "downloads").mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(f"{URL}?download=1", headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        download_url = response.geturl()
    count = (SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for completed, destination in enumerate(executor.map(download_chunk, range(count)), 1):
            print(f"Downloaded {completed}/{count}: {destination.name}", flush=True)
    archive = ROOT / "downloads" / "Blocks-v1.8.1.zip"
    with archive.open("wb") as output:
        for index in range(count):
            output.write((ROOT / "downloads" / f"Blocks.part{index:03d}").read_bytes())
    with zipfile.ZipFile(archive) as package:
        damaged = package.testzip()
        if damaged is not None:
            raise RuntimeError(f"ZIP checksum failed: {damaged}")
        package.extractall(ROOT / "environments")
    print(f"SHA256: {hashlib.sha256(archive.read_bytes()).hexdigest()}", flush=True)
    print("Extracted Blocks successfully", flush=True)


if __name__ == "__main__":
    main()
