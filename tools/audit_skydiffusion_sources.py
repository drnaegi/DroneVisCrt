import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path

import requests

from network_dns import enable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/alignment_audit/sources"
SOURCES = {
    "github_issues.json": "https://api.github.com/repos/opendatalab/skydiffusion/issues?state=all&per_page=100",
    "github_issue4_comments.json": "https://api.github.com/repos/opendatalab/skydiffusion/issues/4/comments",
    "github_releases.json": "https://api.github.com/repos/opendatalab/skydiffusion/releases",
    "github_commits.json": "https://api.github.com/repos/opendatalab/skydiffusion/commits?per_page=100",
    "github_branches.json": "https://api.github.com/repos/opendatalab/skydiffusion/branches",
    "github_tree.json": "https://api.github.com/repos/opendatalab/skydiffusion/git/trees/main?recursive=1",
    "models_search.json": "https://hf-mirror.com/api/models?search=skydiffusion&limit=100&full=true",
    "author_models.json": "https://hf-mirror.com/api/models?author=Yejy53&limit=100&full=true",
    "official_dataset.json": "https://hf-mirror.com/api/datasets/Yejy53/G2A-3?blobs=true",
    "downloaded_dataset.json": "https://hf-mirror.com/api/datasets/SkyDiff/G2A-3?blobs=true",
    "model_metadata.json": "https://hf-mirror.com/api/models/SkyDiff1109/SkyDiffusion_ckpt?blobs=true",
    "model_card.md": "https://hf-mirror.com/SkyDiff1109/SkyDiffusion_ckpt/raw/main/README.md",
    "project.html": "https://opendatalab.github.io/skydiffusion/",
    "iccv_index.html": "https://openaccess.thecvf.com/ICCV2025?day=all",
    "arxiv_abs.html": "https://arxiv.org/abs/2408.01812",
}


def fetch(source):
    name, url = source
    record = {"name": name, "url": url, "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=(10, 60))
        record["status_code"] = response.status_code
        response.raise_for_status()
        (OUTPUT / name).write_bytes(response.content)
        record.update(sha256=hashlib.sha256(response.content).hexdigest(), bytes=len(response.content))
    except (requests.RequestException, OSError) as error:
        record["error"] = str(error)
    print(name, record.get("status_code"), record.get("bytes", record.get("error")), flush=True)
    return record


def main():
    enable()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        records = list(executor.map(fetch, SOURCES.items()))
    (OUTPUT.parent / "source_index.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
