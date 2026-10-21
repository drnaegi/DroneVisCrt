import ast
import csv
import datetime
import difflib
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import color
from skimage.metrics import structural_similarity

from evaluate_skydiffusion import upstream_metrics
from skydiffusion_preflight import sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/alignment_audit"
sys.path.insert(0, str(ROOT / "code/skydiffusion"))
from cruvedbev import BEV_transform, CruvedBEV


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def functions_from_file(path, names, namespace):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in tree.body} != set(names):
        raise ValueError("Requested functions are absent: " + str(path))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def main():
    torch.set_num_threads(8)
    original = OUTPUT / "upstream"
    patches = []
    changed = ["skydiffusion_test.py", "cruvedbev.py", "ldm/modules/encoders/modules.py"]
    for filename in changed:
        patches.extend(difflib.unified_diff((original / filename).read_text().splitlines(True), (ROOT / "code/skydiffusion" / filename).read_text().splitlines(True), fromfile="a/" + filename, tofile="b/" + filename))
    (ROOT / "tools/skydiffusion_upstream_fixes.patch").write_text("".join(patches))

    legacy_projection = functions_from_file(original / "cruvedbev.py", ["BEV_transform"], {"torch": torch, "np": np})["BEV_transform"]
    rotation = torch.tensor([90.0], dtype=torch.float32)
    geometry = (rotation, 512, 512, 1024, 0.06, -1.5)
    current_uv = BEV_transform(*geometry)
    assert torch.equal(current_uv, legacy_projection(*geometry)), "Default UV projection changed"
    flat_uv = BEV_transform(*geometry, curve_height=0.0)
    assert torch.isfinite(flat_uv).all() and torch.isfinite(current_uv).all()
    cardinals = {}
    for label, row, column, expected in [("north", 0, 256, 512), ("east", 256, 511, 768), ("south", 511, 256, 0), ("west", 256, 0, 256)]:
        horizontal = float(current_uv[0, row, column, 0])
        error = abs((horizontal - expected + 512) % 1024 - 512)
        assert error < 1.0, "Cardinal orientation mismatch: " + label
        cardinals[label] = {"u": horizontal, "expected_u": expected, "circular_error_pixels": error}
    shifted = BEV_transform(torch.tensor([90.0 + 90.0 / 180.0]), *geometry[1:])
    delta = (shifted[..., 0] - current_uv[..., 0] - 256 + 512) % 1024 - 512
    assert float(delta.abs().max()) < 0.01
    projection_records = []
    metric_records = []
    namespaces = [functions_from_file(ROOT / "code/skydiffusion/eval" / filename, ["PSNR", "SSIM"], {"np": np, "color": color, "ssim": structural_similarity}) for filename in ["eval.py", "evaluation.py"]]
    for run_name, dataset in [("skydiffusion_disaster_smoke", "G2A-3"), ("skydiffusion_cvact_examples", "CVACT")]:
        samples = json.loads((ROOT / "results" / run_name / "samples.json").read_text())
        for sample in samples:
            bev = CruvedBEV(sample["street"], dataset)
            previous = np.asarray(Image.open(ROOT / "results" / run_name / (sample["name"] + "_bev.png")))
            assert np.array_equal(bev, previous), "Default BEV pixels changed: " + sample["name"]
            projection_records.append({"run": run_name, "name": sample["name"], "bev_pixels_identical": True})
            prediction = np.asarray(Image.open(sample["generated"]).convert("RGB"), dtype=np.float32) / 255.0
            target = np.asarray(Image.open(sample["target"]).convert("RGB"))
            if target.shape != prediction.shape:
                target = cv2.resize(target, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_LINEAR)
            target = target.astype(np.float32) / 255.0
            metrics = upstream_metrics(target, prediction)
            for namespace in namespaces:
                assert abs(float(namespace["PSNR"](target, prediction)) - metrics["psnr_rgb_capped40_db"]) < 1e-12
                assert float(namespace["SSIM"](target, prediction)) == metrics["ssim_y"]
            assert np.allclose((prediction - 0.5) / 0.5, prediction * 2 - 1, atol=1e-7)
            identity = upstream_metrics(target, target)
            assert abs(identity["psnr_rgb_capped40_db"] - 40) < 1e-12 and identity["ssim_y"] == 1
            metric_records.append({"run": run_name, "name": sample["name"], "psnr_ssim_equal_to_both_upstream_functions": True})
    write_json(OUTPUT / "projection_validation.json", {
        "default_uv_bitwise_equal_to_pinned_upstream": True,
        "default_bev_matches_previous_runs": projection_records,
        "cardinals": cardinals,
        "north_offset_90_degree_max_error_pixels": float(delta.abs().max()),
        "rotation_convention": "upstream multiplies rot=90 by pi (45 full turns); preserve this default, do not convert it into a 90-degree turn",
        "vertical_convention": "signed camera_height=-1.5; public atan2(radial_meters,camera_height+curve) differs in vertical sign convention from printed equation (2); training convention is not established",
        "paper_calibration_verified": False,
    })
    write_json(OUTPUT / "metric_validation.json", {
        "checks": metric_records,
        "identity_psnr_db": 40,
        "identity_ssim_y": 1,
        "lpips": "AlexNet and normalization match active eval/evaluation.py; no claim that paper used the same reference",
        "reference_gap": "evaluation.py evaluates VAE reconstruction targets; existing local metrics use original targets",
        "paper_protocol_verified": False,
    })

    inventory = ROOT / "datasets/G2A-3/disaster_inventory.tsv"
    with zipfile.ZipFile(ROOT / "datasets/G2A-3/Disaster.zip") as archive:
        sources = {Path(item.filename).stem: item for item in archive.infolist() if item.filename.startswith("disaster/SVI/") and item.filename.endswith(".png")}
        targets = {Path(item.filename).stem: item for item in archive.infolist() if item.filename.startswith("disaster/Arieal/") and item.filename.endswith(".png")}
        assert sources.keys() == targets.keys() and len(sources) == 2714
        with inventory.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["sample_id", "filename_latitude", "filename_longitude", "street_member", "street_size", "street_crc32", "aerial_member", "aerial_size", "aerial_crc32", "official_split"])
            for name in sorted(sources):
                source, target = sources[name], targets[name]
                latitude, longitude = name.split("_")[:2]
                writer.writerow([name, float(latitude), float(longitude), source.filename, source.file_size, source.CRC, target.filename, target.file_size, target.CRC, "unknown"])
    write_json(OUTPUT / "split_status.json", {
        "dataset": "G2A-3/Disaster", "pair_count": len(sources), "inventory_sha256": sha256(inventory),
        "coordinates_status": "parsed from filenames; not independently geolocated",
        "paper_split_rule": "regional 4:1 train:test", "official_train_ids": None, "official_test_ids": None,
        "official_region_boundaries": None, "local_split_created": False,
        "searched": ["public main and historical repository trees", "downloaded and official Disaster/History ZIP central directories", "project page", "ICCV main paper"],
        "uav_scope": "outer ZIPs contain six nested ZIPs; nested official UAV contents have not been audited",
    })

    pinned_files = ["code/skydiffusion/" + filename for filename in changed] + [
        "code/skydiffusion/models/lacldm_v15.yaml", "code/skydiffusion/cldm/cldm.py",
        "code/skydiffusion/cldm/logger.py", "code/skydiffusion/skydiffusion_dataset.py",
        "code/skydiffusion/ldm/models/diffusion/ddim.py", "tools/run_skydiffusion.py",
        "tools/evaluate_skydiffusion.py", "tools/skydiffusion_preflight.py",
    ]
    profile = {
        "schema_version": 1, "task": "G2A-3/Disaster", "dataset": "G2A-3",
        "scope": "locked diagnostic profile, NOT the official Disaster benchmark",
        "checkpoint": {
            "path": "checkpoints/SkyDiffusion/CVACT_Skydiffusion.ckpt", "bytes": 12811886329,
            "sha256": "a9108a3394fe127ae7cde9a75019b34ea11da9d3375a51efb4236200bbedfc6a",
            "source": "SkyDiff1109/SkyDiffusion_ckpt", "revision": "31b574c1750cc6bb8e285d566b381a0946db9ee2",
            "training_task": "CVACT", "official_historical_link_confirmed": True, "paper_benchmark_verified": False,
            "provenance": "https://github.com/opendatalab/skydiffusion/blob/e3ef0b6da5f46875ba24bedf355937daf23518ba/README.md",
        },
        "inputs": {
            "manifest": "datasets/G2A-3/disaster_smoke/pairs.tsv", "manifest_sha256": sha256(ROOT / "datasets/G2A-3/disaster_smoke/pairs.tsv"),
            "purpose": "random 8-pair diagnostic subset, selection seed=42", "split_status": "not_published_in_audited_sources",
            "train_ids": None, "test_ids": None,
        },
        "projection": {
            "meter_per_pixel": 0.06, "camera_height": -1.5, "curve_height": 3.0, "north_offset_deg": 0.0,
            "panorama_height": 512, "panorama_width": 1024, "paper_parameters_verified": False,
        },
        "sampling": {"steps": 50, "guidance": 9.0, "seed": 42, "count": 8, "prompt": "", "negative_prompt": "", "precision": "fp32", "quantization": "round"},
        "evaluation": {"profile": "upstream", "reference": "target", "lpips_net": "alex", "fid": "not_computed", "paper_protocol_verified": False},
        "official_code_commit": "461ccc2fff11882d48f5ab62cae65eef5c45c647",
        "local_patch_sha256": sha256(ROOT / "tools/skydiffusion_upstream_fixes.patch"),
        "local_code_sha256": {filename: sha256(ROOT / filename) for filename in pinned_files},
    }
    write_json(ROOT / "configs/skydiffusion_disaster_diagnostic.json", profile)
    write_json(OUTPUT / "alignment_status.json", {
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "weight": "official historical link, hash and architecture verified; CVACT only; Disaster benchmark match not verified",
        "projection": "public defaults preserved and validated; Disaster calibration unconfirmed",
        "split": "2714 pairs inventoried; no verified official train/test IDs",
        "evaluation": "PSNR/Y-SSIM and AlexNet definition aligned to public functions; target reference and FID remain unconfirmed",
        "disaster_dataset": "all 5428 name/size/CRC entries agree; 16 current input/target files agree by SHA256",
        "history_dataset": "2128 aerial CRC/size entries agree; 2128 street CRC/size entries differ; 3 decoded street samples differ",
        "paper_results_reproduced": False,
    })
    known_sources = {item["name"]: item["url"] for item in json.loads((OUTPUT / "source_index.json").read_text())}
    known_sources.update({
        "historic_README.md": "https://api.github.com/repos/opendatalab/skydiffusion/contents/README.md?ref=e3ef0b6da5f46875ba24bedf355937daf23518ba",
        "historic_code_commit.json": "https://api.github.com/repos/opendatalab/skydiffusion/commits/e3ef0b6da5f46875ba24bedf355937daf23518ba",
        "projection_commits.json": "https://api.github.com/repos/opendatalab/skydiffusion/commits?path=cruvedbev.py&per_page=100",
        "prs.json": "https://api.github.com/repos/opendatalab/skydiffusion/pulls?state=all&per_page=100",
        "iccv_paper.html": "https://openaccess.thecvf.com/content/ICCV2025/html/Ye_Leveraging_BEV_Paradigm_for_Ground-to-Aerial_Image_Synthesis_ICCV_2025_paper.html",
    })
    evidence = [{"path": str(path.relative_to(ROOT)), "url": known_sources[path.name], "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted((OUTPUT / "sources").iterdir()) if path.name in known_sources]
    paper = ROOT / "papers/SkyDiffusion_ICCV2025_paper.pdf"
    evidence.append({"path": str(paper.relative_to(ROOT)), "url": "https://openaccess.thecvf.com/content/ICCV2025/papers/Ye_Leveraging_BEV_Paradigm_for_Ground-to-Aerial_Image_Synthesis_ICCV_2025_paper.pdf", "bytes": paper.stat().st_size, "sha256": sha256(paper)})
    write_json(OUTPUT / "evidence_index.json", evidence)
    print(json.dumps({"projection_samples_verified": len(projection_records), "metric_samples_verified": len(metric_records), "disaster_inventory_pairs": len(sources), "profile": str(ROOT / "configs/skydiffusion_disaster_diagnostic.json")}, indent=2))


if __name__ == "__main__":
    main()
