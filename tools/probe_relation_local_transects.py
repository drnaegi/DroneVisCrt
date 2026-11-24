"""Locate local road-to-water cross sections in one source and check them in two withheld views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relation_nvs.io import load_camera, sha256, write_json

EVIDENCE = ROOT / "outputs/relation_nvs_20261120"
TOWNS = ("Town01", "Town03", "Town04", "Town07")
HEIGHTS = (20, 50, 80)
SIZE = (512, 288)


def read(path):
    return json.loads(path.read_text())


def checked(folder, entry, key):
    path = folder / entry[key]
    if sha256(path) != entry["hashes"][key]:
        raise ValueError(f"Changed {key} input: {path}")
    return path


def world_points(camera, xy, depth):
    rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(camera.intrinsics).T
    return (rays * depth[:, None]) @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3]


def project(camera, xyz):
    local = (xyz - camera.camera_to_world[:3, 3]) @ camera.camera_to_world[:3, :3]
    screen = local @ camera.intrinsics.T
    xy = screen[:, :2] / np.maximum(screen[:, 2:], 1e-8)
    return xy, local[:, 2]


def transects(labels, depth, camera):
    # Use interior pixels so a single jagged boundary pixel cannot define either endpoint.
    road = binary_erosion(labels == 1, iterations=2)
    water = binary_erosion(labels == 2, iterations=2)
    if not road.any() or not water.any():
        return []
    distance, nearest = distance_transform_edt(~road, return_indices=True)
    ys, xs = np.nonzero(water & ~binary_erosion(water) & (distance <= 96))
    found = []
    for y, x in zip(ys, xs):
        ry, rx = nearest[:, y, x]
        length = int(np.ceil(distance[y, x])) + 1
        xy = np.rint(np.linspace([rx, ry], [x, y], length)).astype(int)
        seq = labels[xy[:, 1], xy[:, 0]]
        z = depth[xy[:, 1], xy[:, 0]]
        if (seq == 255).any() or not (np.isfinite(z) & (z > 0)).all():
            continue
        first_water = np.flatnonzero(seq == 2)[0]
        last_road = np.flatnonzero(seq == 1)[-1]
        if last_road >= first_water:
            continue
        xyz = world_points(camera, xy, z)
        span = np.linalg.norm(xyz[-1] - xyz[0])
        if span > 15 or np.linalg.norm(np.diff(xyz, axis=0), axis=1).max() > 2:
            continue
        indices = [0, len(seq) - 1]
        middle = seq[last_road+1:first_water]
        if len(middle):
            # An intervening mapped class is only a witness when sampled explicitly.
            separator = np.flatnonzero(middle == 3)
            mid = separator[len(separator)//2] if len(separator) else len(middle)//2
            indices.insert(1, int(last_road + 1 + mid))
        found.append((float(span), xy[indices], xyz[indices], seq[indices]))
    retained = []
    for span, xy, xyz, seq in sorted(found, key=lambda row: row[0]):
        if all(np.linalg.norm(xyz[-1] - np.array(old["xyz"][-1])) >= 4 for old in retained):
            retained.append(dict(span_m=span, source_xy=xy.tolist(), xyz=xyz.tolist(), classes=seq.tolist()))
    return retained


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "protocol.json", dict(towns=TOWNS, source_heights=HEIGHTS, source_size=SIZE,
        target_size=[256,144], target_views="other two heights, same time", code_sha256=sha256(Path(__file__)),
        source="renderer depth and semantic classes; one source per trial", global_stride_m=4,
        max_span_m=15, max_neighbor_jump_m=2, target_min_span_px=3,
        truth_test="all witness labels match at rounded projection and |dz| <= max(1 m, 0.03*z)",
        units="overlapping trials in four development towns; localized probes are not independent instances"))
    cases = []
    for town in TOWNS:
        folder = EVIDENCE / "multitown_extension" / town / "converted"
        manifests = {h: read(folder/f"source_to_height{h}.json") for h in HEIGHTS}
        for h in HEIGHTS:
            others = [other for other in HEIGHTS if other != h]
            source = next(r for r in manifests[others[0]]["references"] if r["source_id"] == f"height{h}")
            camera = load_camera(source["camera"]).resized(*SIZE)
            labels = np.asarray(Image.open(checked(folder, source, "labels")).resize(SIZE, Image.Resampling.NEAREST))
            depth = np.asarray(Image.fromarray(np.load(checked(folder, source, "depth"))).resize(SIZE, Image.Resampling.NEAREST))
            witnesses = transects(labels, depth, camera)
            for witness in witnesses:
                views = {}
                for target in others:
                    cam = load_camera(manifests[target]["cameras"][0])
                    xy, z = project(cam, np.array(witness["xyz"]))
                    pix = np.rint(xy).astype(int)
                    visible = ((z > 0) & (pix[:,0] >= 0) & (pix[:,0] < cam.width) &
                               (pix[:,1] >= 0) & (pix[:,1] < cam.height)).all()
                    resolved = np.linalg.norm(xy[-1]-xy[0]) >= 3 and len(np.unique(pix, axis=0)) == len(pix)
                    views[str(target)] = dict(xy=xy.tolist(), z=z.tolist(), eligible=bool(visible and resolved))
                witness["views"] = views
            cases.append(dict(town=town, source_height=h, source_hashes=source["hashes"], witnesses=witnesses))
    write_json(destination/"source_predictions.json", cases)


def score(destination):
    protocol = read(destination/"protocol.json")
    if protocol["code_sha256"] != sha256(Path(__file__)):
        raise ValueError("Prediction and evaluation code differ")
    prediction = destination/"source_predictions.json"
    cases = read(prediction)
    output = []
    for case in cases:
        folder = EVIDENCE/"multitown_extension"/case["town"]/"converted"
        targets = {}
        for h in HEIGHTS:
            if h == case["source_height"]:
                continue
            item = read(folder/f"evaluation_height{h}.json")["targets"][0]
            targets[str(h)] = (np.asarray(Image.open(checked(folder,item,"labels"))), np.load(checked(folder,item,"depth")))
        rows = []
        for index, witness in enumerate(case["witnesses"]):
            result = {}
            for h, view in witness["views"].items():
                if not view["eligible"]:
                    result[h] = "out_of_frame_or_unresolved"
                    continue
                x, y = np.rint(view["xy"]).astype(int).T
                labels, depth = targets[h]
                z = np.array(view["z"])
                if not (np.abs(depth[y,x]-z) <= np.maximum(1, 0.03*z)).all():
                    result[h] = "depth_disagreement"
                elif not np.array_equal(labels[y,x], witness["classes"]):
                    result[h] = "class_disagreement"
                else:
                    result[h] = "confirmed"
            rows.append(dict(index=index, views=result, jointly_eligible=all(w["eligible"] for w in witness["views"].values()),
                             jointly_confirmed=all(v=="confirmed" for v in result.values())))
        output.append(dict(town=case["town"], source_height=case["source_height"], rows=rows))
    write_json(destination/"results.json", dict(source_predictions_sha256=sha256(prediction), cases=output))
    for case in output:
        rows=case["rows"]
        print(case["town"], case["source_height"], len(rows), sum(r["jointly_eligible"] for r in rows),
              sum(r["jointly_confirmed"] for r in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare","score"))
    parser.add_argument("--output", type=Path, default=EVIDENCE/"local_transects_v1")
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else score)(args.output)
