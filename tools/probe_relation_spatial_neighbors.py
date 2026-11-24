"""Find road and water surface neighbors in 3D from one source view."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from probe_relation_local_transects import (
    EVIDENCE, TOWNS, HEIGHTS, SIZE, checked, load_camera, project, read, sha256,
    world_points, write_json,
)


def surface_samples(labels, depth, camera, category):
    interior = binary_erosion(labels == category, iterations=2)
    # Search all valid road interior points. For water, query from just inside its perimeter.
    if category == 2:
        interior &= ~binary_erosion(interior)
    y, x = np.nonzero(interior & np.isfinite(depth) & (depth > 0))
    xy = np.column_stack((x,y))
    return xy, world_points(camera, xy, depth[y,x])


def neighbors(labels, depth, camera):
    road_uv, road = surface_samples(labels, depth, camera, 1)
    water_uv, water = surface_samples(labels, depth, camera, 2)
    if not len(road) or not len(water):
        return []
    distance, nearest = cKDTree(road).query(water)
    retained = []
    for index in np.argsort(distance, kind="stable"):
        if distance[index] > 15:
            break
        if any(np.linalg.norm(water[index] - row["xyz"][-1]) < 4 for row in retained):
            continue
        road_index = nearest[index]
        retained.append(dict(classes=[1,2], span_m=float(distance[index]),
            source_xy=[road_uv[road_index].tolist(),water_uv[index].tolist()],
            xyz=[road[road_index].tolist(),water[index].tolist()]))
    return retained


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    shared = Path(__file__).with_name("probe_relation_local_transects.py")
    write_json(out/"protocol.json", dict(towns=TOWNS, source_heights=HEIGHTS,
        proposal="closest road interior point within 15m of each water interior rim point in world coordinates",
        thinning="shortest distance first, 4m separation between water points",
        prediction="positive depth, both points in frame and at least 3 target pixels apart",
        truth_test="both exact target classes and |dz| <= max(1 m, 0.03*z) at both projected pixels",
        scope="physical proximity of observed samples only; intervening surfaces unspecified",
        source_classes_and_depth="renderer ground truth", target_independent_scenes=False,
        code_sha256=sha256(Path(__file__)), shared_code_sha256=sha256(shared)))
    cases = []
    for town in TOWNS:
        folder = EVIDENCE/"multitown_extension"/town/"converted"
        manifests = {h:read(folder/f"source_to_height{h}.json") for h in HEIGHTS}
        for height in HEIGHTS:
            others = [h for h in HEIGHTS if h != height]
            record = next(ref for ref in manifests[others[0]]["references"] if ref["source_id"]==f"height{height}")
            camera = load_camera(record["camera"]).resized(*SIZE)
            depth = np.asarray(Image.fromarray(np.load(checked(folder,record,"depth"))).resize(SIZE, Image.Resampling.NEAREST))
            labels = np.asarray(Image.open(checked(folder,record,"labels")).resize(SIZE, Image.Resampling.NEAREST))
            anchors = neighbors(labels, depth, camera)
            for anchor in anchors:
                anchor["views"] = {}
                for h in others:
                    cam = load_camera(manifests[h]["cameras"][0])
                    xy, z = project(cam, np.asarray(anchor["xyz"]))
                    x,y = np.rint(xy).astype(int).T
                    in_frame = ((z>0)&(x>=0)&(x<cam.width)&(y>=0)&(y<cam.height)).all()
                    eligible = in_frame and np.linalg.norm(xy[0]-xy[1]) >= 3
                    anchor["views"][str(h)] = dict(xy=xy.tolist(),z=z.tolist(),eligible=bool(eligible))
            cases.append(dict(town=town,source_height=height,witnesses=anchors,source_hashes=record["hashes"]))
    write_json(out/"source_predictions.json",cases)


def score(out):
    summaries = []
    for case in read(out / "source_predictions.json"):
        folder = EVIDENCE/"multitown_extension"/case["town"]/"converted"
        observations = [{} for _ in case["witnesses"]]
        for height in HEIGHTS:
            if height == case["source_height"]:
                continue
            key = str(height)
            target = read(folder/f"evaluation_height{height}.json")["targets"][0]
            labels = np.asarray(Image.open(checked(folder,target,"labels")))
            depth = np.load(checked(folder,target,"depth"))
            for anchor, record in zip(case["witnesses"], observations):
                view = anchor["views"][key]
                if not view["eligible"]:
                    record[key] = "out_of_frame_or_unresolved"
                    continue
                x,y = np.rint(view["xy"]).astype(int).T
                z = np.asarray(view["z"])
                if not (np.abs(depth[y,x]-z) <= np.maximum(1,0.03*z)).all():
                    record[key] = "depth_disagreement"
                elif not np.array_equal(labels[y,x],anchor["classes"]):
                    record[key] = "class_disagreement"
                else:
                    record[key] = "confirmed"
        counts = dict(proposed=len(observations),
            jointly_eligible=sum("out_of_frame_or_unresolved" not in row.values() for row in observations),
            jointly_confirmed=sum(set(row.values())=={"confirmed"} for row in observations))
        summaries.append(dict(town=case["town"], source_height=case["source_height"],
                              observations=observations, **counts))
        print(case["town"], case["source_height"], counts)
    write_json(out/"results.json",dict(cases=summaries,source_predictions_sha256=sha256(out/"source_predictions.json")))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","score"))
    parser.add_argument("--output",type=Path,default=EVIDENCE/"local_neighbors_v2")
    args=parser.parse_args()
    if args.stage=="prepare":
        prepare(args.output)
    else:
        protocol = read(args.output/"protocol.json")
        shared = Path(__file__).with_name("probe_relation_local_transects.py")
        if protocol["shared_code_sha256"] != sha256(shared) or protocol["code_sha256"] != sha256(Path(__file__)):
            raise ValueError("Code changed since prediction")
        score(args.output)
