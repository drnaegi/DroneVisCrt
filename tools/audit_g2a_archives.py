import argparse
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from network_dns import enable


ROOT = Path(__file__).resolve().parents[1]


class RemoteArchive(io.RawIOBase):
    def __init__(self, url, size, expected_sha256):
        self.size = size
        self.position = 0
        self.session = requests.Session()
        response = self.session.head(url, allow_redirects=False, timeout=(10, 30))
        response.raise_for_status()
        if response.headers.get("X-Linked-ETag", "").strip('"') != expected_sha256:
            raise ValueError("Remote archive metadata changed")
        self.url = response.headers["Location"]

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Invalid archive position")
        return self.position

    def read(self, size=-1):
        length = min(size if size >= 0 else self.size, self.size - self.position)
        if length == 0:
            return b""
        end = self.position + length - 1
        response = self.session.get(self.url, headers={"Range": "bytes=%d-%d" % (self.position, end), "Accept-Encoding": "identity"}, timeout=(15, 60))
        response.raise_for_status()
        if response.status_code != 206 or response.headers.get("Content-Range") != "bytes %d-%d/%d" % (self.position, end, self.size):
            raise ValueError("Unexpected HTTP range")
        content = response.content
        if len(content) != length:
            raise ValueError("Truncated HTTP range")
        self.position += length
        return content

    def close(self):
        self.session.close()
        super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="disaster.zip")
    parser.add_argument("--local-filename", default="Disaster.zip")
    parser.add_argument("--sample-count", type=int, default=0)
    args = parser.parse_args()
    enable()
    output = ROOT / "results/alignment_audit"
    metadata = json.loads((output / "sources/official_dataset.json").read_text())
    entry = next(item for item in metadata["siblings"] if item["rfilename"] == args.filename)
    url = "https://hf-mirror.com/datasets/Yejy53/G2A-3/resolve/" + metadata["sha"] + "/" + args.filename
    with RemoteArchive(url, entry["size"], entry["lfs"]["sha256"]) as remote, zipfile.ZipFile(remote) as archive, zipfile.ZipFile(ROOT / "datasets/G2A-3" / args.local_filename) as local:
        remote_entries = [{"name": item.filename, "crc32": item.CRC, "size": item.file_size} for item in archive.infolist() if not item.is_dir()]
        local_entries = [{"name": item.filename, "crc32": item.CRC, "size": item.file_size} for item in local.infolist() if not item.is_dir()]
        remote_by_name = {item["name"]: (item["crc32"], item["size"]) for item in remote_entries}
        local_by_name = {item["name"]: (item["crc32"], item["size"]) for item in local_entries}
        shared = remote_by_name.keys() & local_by_name.keys()
        different = sorted(name for name in shared if remote_by_name[name] != local_by_name[name])
        report = {
            "official_repository": "Yejy53/G2A-3", "official_revision": metadata["sha"],
            "official_zip_sha256": entry["lfs"]["sha256"], "remote_entries": remote_entries,
            "local_file_count": len(local_entries), "remote_file_count": len(remote_entries),
            "only_local": sorted(local_by_name.keys() - remote_by_name.keys()),
            "only_official": sorted(remote_by_name.keys() - local_by_name.keys()),
            "different_crc_or_size": different,
            "same_name_crc_size_count": len(shared) - len(different),
        }
        remote_by_basename = {Path(item["name"]).name: item for item in remote_entries}
        local_by_basename = {Path(item["name"]).name: item for item in local_entries}
        if len(remote_by_basename) == len(remote_entries) and len(local_by_basename) == len(local_entries):
            common = remote_by_basename.keys() & local_by_basename.keys()
            report["basename_comparison"] = {
                "shared_count": len(common),
                "same_crc_size_count": sum(
                    (remote_by_basename[name]["crc32"], remote_by_basename[name]["size"])
                    == (local_by_basename[name]["crc32"], local_by_basename[name]["size"])
                    for name in common
                ),
            }
        sample_hashes = []
        if args.filename == "disaster.zip":
            subset = json.loads((ROOT / "datasets/G2A-3/disaster_smoke/subset.json").read_text())
            for path in sorted((ROOT / "datasets/G2A-3/disaster_smoke/street").glob("*.png")):
                for category in ["SVI", "Arieal"]:
                    name = "disaster/" + category + "/" + path.name
                    if name in shared:
                        official_hash = hashlib.sha256(archive.read(name)).hexdigest()
                        local_hash = hashlib.sha256(local.read(name)).hexdigest()
                        sample_hashes.append({"name": name, "official_sha256": official_hash, "local_sha256": local_hash, "equal": official_hash == local_hash})
                        print(name, sample_hashes[-1]["equal"], flush=True)
            report["sample_subset"] = subset
        elif args.sample_count:
            candidates = sorted(name for name in remote_by_basename.keys() & local_by_basename.keys() if Path(name).suffix.lower() in [".jpg", ".jpeg"])
            for basename in sorted(random.Random(42).sample(candidates, min(args.sample_count, len(candidates)))):
                official_content = archive.read(remote_by_basename[basename]["name"])
                local_content = local.read(local_by_basename[basename]["name"])
                official_pixels = np.asarray(Image.open(io.BytesIO(official_content)).convert("RGB"))
                local_pixels = np.asarray(Image.open(io.BytesIO(local_content)).convert("RGB"))
                equal_shape = official_pixels.shape == local_pixels.shape
                sample_hashes.append({
                    "name": basename,
                    "official_member": remote_by_basename[basename]["name"],
                    "local_member": local_by_basename[basename]["name"],
                    "official_sha256": hashlib.sha256(official_content).hexdigest(),
                    "local_sha256": hashlib.sha256(local_content).hexdigest(),
                    "equal": official_content == local_content,
                    "official_shape": list(official_pixels.shape),
                    "local_shape": list(local_pixels.shape),
                    "decoded_pixels_equal": equal_shape and bool(np.array_equal(official_pixels, local_pixels)),
                    "rgb_mae_255": float(np.abs(official_pixels.astype(np.float32) - local_pixels).mean()) if equal_shape else None,
                })
                sample_path = output / "archive_samples/History_official" / basename
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                sample_path.write_bytes(official_content)
        report["sample_content_sha256"] = sample_hashes
    destination = output / (args.filename + ".audit.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ["remote_entries", "sample_content_sha256", "only_local", "only_official", "different_crc_or_size"]}, indent=2), flush=True)
    print(destination)


if __name__ == "__main__":
    main()
