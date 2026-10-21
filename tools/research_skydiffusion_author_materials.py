import base64
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path

import requests

from network_dns import enable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/author_materials_search"
SOURCES = {
    "author_home.html": "https://yejy53.github.io/",
    "corresponding_author_home.html": "https://liweijia.github.io/",
    "author_repos.json": "https://api.github.com/users/Yejy53/repos?per_page=100",
    "repository_search.json": "https://api.github.com/search/repositories?q=skydiffusion&per_page=100",
    "legacy_repository.json": "https://api.github.com/repos/SkyDiffusion/SkyDiffusion-code",
    "official_branches.json": "https://api.github.com/repos/opendatalab/skydiffusion/branches",
    "project_tree.json": "https://api.github.com/repos/opendatalab/skydiffusion/git/trees/gh-pages?recursive=1",
    "author_site_tree.json": "https://api.github.com/repos/yejy53/yejy53.github.io/git/trees/main?recursive=1",
    "author_hf_models.json": "https://hf-mirror.com/api/models?author=Yejy53&full=true&limit=100",
    "uploader_hf_models.json": "https://hf-mirror.com/api/models?author=SkyDiff1109&full=true&limit=100",
    "skydiff_hf_models.json": "https://hf-mirror.com/api/models?author=SkyDiff&full=true&limit=100",
    "g2a_hf_models.json": "https://hf-mirror.com/api/models?search=G2A&full=true&limit=100",
    "official_dataset_tree.json": "https://hf-mirror.com/api/datasets/Yejy53/G2A-3/tree/main?recursive=true&expand=false",
    "official_dataset_readme.md": "https://hf-mirror.com/datasets/Yejy53/G2A-3/raw/main/README.md",
    "official_dataset_discussions.json": "https://huggingface.co/api/datasets/Yejy53/G2A-3/discussions",
    "checkpoint_discussions.json": "https://huggingface.co/api/models/SkyDiff1109/SkyDiffusion_ckpt/discussions",
    "author_skydiffusion_tree.json": "https://api.github.com/repos/yejy53/Skydiffusion/git/trees/main?recursive=1",
    "author_skydiffusion_readme.md": "https://api.github.com/repos/yejy53/Skydiffusion/contents/README.md",
    "alternate_project_tree.json": "https://api.github.com/repos/SkyDiffusion0307/skydiffusion0307.github.io/git/trees/main?recursive=1",
    "author_about.md": "https://api.github.com/repos/yejy53/yejy53.github.io/contents/_pages/about.md",
    "author_config.yml": "https://api.github.com/repos/yejy53/yejy53.github.io/contents/_config.yml",
    "corresponding_author_index.html": "https://api.github.com/repos/liweijia/liweijia.github.io/contents/index.html",
    "official_dataset_discussions_mirror.json": "https://hf-mirror.com/api/datasets/Yejy53/G2A-3/discussions",
    "checkpoint_discussions_mirror.json": "https://hf-mirror.com/api/models/SkyDiff1109/SkyDiffusion_ckpt/discussions",
    "old_project_repository.json": "https://api.github.com/repos/longhz140516/SkyDiffusion.github.io",
    "old_project_contents.json": "https://api.github.com/repos/longhz140516/SkyDiffusion.github.io/contents",
    "corresponding_author_repository.json": "https://api.github.com/repos/liweijia/liweijia.github.io",
    "corresponding_author_config.yml": "https://api.github.com/repos/liweijia/liweijia.github.io/contents/_config.yml",
    "alternate_project_readme.md": "https://api.github.com/repos/SkyDiffusion0307/skydiffusion0307.github.io/contents/README.md",
    "alternate_projection.py": "https://api.github.com/repos/SkyDiffusion0307/skydiffusion0307.github.io/contents/Cruved-BEV.py",
    "legacy_dataset_metadata.json": "https://hf-mirror.com/api/datasets/SkyDiff1109/G2A-3?blobs=true",
    "g2a_datasets_search.json": "https://hf-mirror.com/api/datasets?search=G2A-3&full=true&limit=100",
}


def fetch(source):
    filename, url = source
    record = {"filename": filename, "url": url, "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=(10, 35))
        record["status_code"] = response.status_code
        record["resolved_url"] = response.url
        content = response.content
        if response.status_code == 200 and "/contents/" in url:
            content = base64.b64decode(response.json()["content"])
            record["github_blob_sha"] = response.json()["sha"]
        record["bytes"] = len(content)
        record["sha256"] = hashlib.sha256(content).hexdigest()
        (OUTPUT / filename).write_bytes(content)
        response.raise_for_status()
    except (requests.RequestException, OSError) as error:
        record["error"] = str(error)
    print(filename, record.get("status_code"), record.get("bytes", record.get("error")), flush=True)
    return record


def main():
    enable()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        previous = json.loads((OUTPUT / "source_index.json").read_text()) if (OUTPUT / "source_index.json").exists() else []
        completed = {item["filename"] for item in previous}
        records = list(executor.map(fetch, [(name, url) for name, url in SOURCES.items() if name not in completed]))
    records_by_name = {item["filename"]: item for item in previous}
    records_by_name.update({item["filename"]: item for item in records})
    records = list(records_by_name.values())
    (OUTPUT / "source_index.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Saved source snapshots to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
