"""Select one ordered road/grass/water strip per recorded flight pair and audit both real views."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

from probe_relation_floodnet_transfer import BASE, W, H, calibration, plane
from probe_relation_local_transects import project, read, world_points, write_json, sha256
from probe_relation_spatial_neighbors import neighbors, surface_samples
from relation_nvs.geometry import Camera

SITES = BASE / "floodnet_calibrated_sites"
OUT = BASE / "grass_strip_pairs"
METHODS = ("endpoints", "midpoint", "three_transects")


def label_image(frame):
    with Image.open(BASE / "mask_inventory" / f"{frame}_lab.png") as image:
        raw = np.asarray(image.resize((W, H), Image.Resampling.NEAREST))
    mask = np.zeros_like(raw)
    mask[np.isin(raw, (3, 4))] = 1
    mask[raw == 9] = 3
    mask[raw == 5] = 2
    return mask


def load_cameras(ids):
    cameras = {}
    for i in ids:
        lat, lon, meta, rotation, f = calibration(SITES / "images" / f"{i}.jpg")
        if not cameras:
            origin = lat, lon
        dy, dx = np.radians([lat - origin[0], lon - origin[1]]) * 6378137
        transform = np.eye(4)
        transform[:3, 3] = [dx * np.cos(np.radians(origin[0])), dy, meta["RelativeAltitude"]]
        transform[:3, :3] = rotation
        k = np.array([[f, 0, (W-1)/2], [0, f, (H-1)/2], [0, 0, 1]])
        cameras[i] = Camera(str(i), k, transform, W, H)
    return cameras


def fits(xy):
    return np.all((xy >= 0) & (xy <= np.array([W-1, H-1])))


def ordered(line):
    # Require continuous grass between the last road sample and the first water sample.
    if line[0] != 1 or line[-1] != 2:
        return False
    last_road = np.flatnonzero(line == 1)[-1]
    first_water = np.flatnonzero(line == 2)[0]
    band = line[last_road + 1:first_water]
    return (len(band) >= 3 and np.all(band == 3) and
            np.all(line[:last_road+1] == 1) and np.all(line[first_water:] == 2))


def observed(mask, xy):
    x, y = np.rint(xy).astype(int).T
    return mask[y, x].reshape(3, -1)


def candidates(mask, source_cam):
    _, depth = plane(source_cam)
    depth = depth.reshape(H, W)
    road, water = [distance_transform_edt(mask == c) >= 8 for c in (1, 2)]
    boundary = water & ~binary_erosion(water)
    y, x = np.nonzero(boundary)
    if not road.any() or not len(y):
        return
    _, nearest = distance_transform_edt(~road, return_indices=True)
    for j in range(0, len(y), 4):
        end = np.array([x[j], y[j]])
        start = nearest[:, y[j], x[j]][::-1]
        v = end-start
        length = np.linalg.norm(v)
        if length < 1:
            continue
        normal = np.array([-v[1], v[0]]) / length
        center = np.linspace(start, end, int(np.ceil(length))+1)
        coords = np.concatenate([center+normal*offset for offset in (-3, 0, 3)])
        if fits(coords):
            # A ground plane permits subpixel backprojection without depth rounding.
            dirs = np.column_stack((coords, np.ones(len(coords)))) @ np.linalg.inv(source_cam.intrinsics).T
            z = -source_cam.camera_to_world[2,3] / (dirs @ source_cam.camera_to_world[:3,:3].T)[:,2]
            if (z > 0).all():
                yield coords, world_points(source_cam, coords, z)


def prepare():
    OUT.mkdir(exist_ok=False)
    inventory = read(SITES / "inventory.json")
    write_json(OUT / "protocol.json", dict(code=sha256(Path(__file__)), inventory=sha256(SITES/"inventory.json"),
        plans=inventory["possible_pairs"], budget_per_method_per_plan=1,
        methods=METHODS, sampling="road/water interiors >=8 pixels, every fourth water rim pixel in raster order; nearest road interior",
        tracks="three parallel lines at -3,0,3 source pixels with <=1 pixel sample spacing; each ordered road then >=3 grass samples then water",
        ranking="largest minimum margin from frame edges in both projected target views; ties use raster order",
        geometry="DJI telemetry and horizontal plane; far relation sampled separately from the existing 15 m proximity test",
        target_score="each projected track must independently exhibit the ordered classes in each of two recorded target label images",
        split="two development sites; frame co-presence used during inventory; ground plane has no measured target depth"))
    plans = []
    for spec in inventory["possible_pairs"]:
        src, targets = spec["source"], spec["targets"]
        cams = load_cameras([src, *targets])
        mask = label_image(src)
        _, z = plane(cams[src])
        depth = z.reshape(H, W)
        local = neighbors(mask, depth, cams[src])
        _, road = surface_samples(mask, depth, cams[src], 1)
        _, water = surface_samples(mask, depth, cams[src], 2)
        minimum = float(cKDTree(road).query(water)[0].min()) if len(road) and len(water) else None
        best = {m:None for m in METHODS}
        count = dict.fromkeys(METHODS, 0)
        for xy, xyz in candidates(mask, cams[src]):
            seq = observed(mask, xy)
            path = {}
            margin = np.inf
            for target in targets:
                uv, dz = project(cams[target], xyz)
                if (dz<=0).any() or not fits(uv):
                    break
                path[target] = uv
                margin = min(margin, uv.min(), (np.array([W-1,H-1])-uv).min())
            if len(path) != 2:
                continue
            conditions = [True, seq[1, seq.shape[1]//2] == 3, all(ordered(line) for line in seq)]
            for method, ok in zip(METHODS, conditions):
                if not ok:
                    continue
                count[method] += 1
                if best[method] is None or margin > best[method]["margin"]:
                    best[method] = dict(margin=float(margin), source_tracks=xy.tolist(),
                        target_tracks={str(t):uv.tolist() for t,uv in path.items()})
        plans.append(dict(source=src, targets=targets, near_15m=len(local), minimum_m=minimum,
                          eligible_counts=count, chosen=best))
    write_json(OUT/"predictions.json", plans)


def audit():
    rows = []
    for p in read(OUT/"predictions.json"):
        labels = {t:label_image(t) for t in p["targets"]}
        outcomes = {}
        for method, selection in p["chosen"].items():
            if selection is None:
                outcomes[method] = None
                continue
            results = {}
            for t,mask in labels.items():
                seq = observed(mask, np.array(selection["target_tracks"][str(t)]))
                results[t] = [bool(ordered(line)) for line in seq]
            outcomes[method] = dict(per_target=results, both=all(all(v) for v in results.values()))
        rows.append(dict(source=p["source"], targets=p["targets"], near_15m=p["near_15m"],
                         minimum_m=p["minimum_m"], outcomes=outcomes))
    totals = {m:dict(selected=sum(r["outcomes"][m] is not None for r in rows),
                     verified=sum(bool(r["outcomes"][m] and r["outcomes"][m]["both"]) for r in rows))
              for m in METHODS}
    write_json(OUT/"results.json", dict(predictions_sha256=sha256(OUT/"predictions.json"), rows=rows, totals=totals))
    print(totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "audit"))
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else audit)()
