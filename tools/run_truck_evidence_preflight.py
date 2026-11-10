import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = [0, 3, 6, 9, 12, 16]
DEVELOPMENT = [1, 5, 11]
EVALUATION = [2, 4, 7, 10, 14, 15]
POLYGONS = {
    0: [[58, 133], [105, 128], [110, 85], [176, 82], [238, 94], [243, 117], [428, 117], [434, 174], [427, 210], [391, 200], [291, 197], [254, 238], [213, 240], [191, 217], [190, 183], [104, 178], [91, 198], [75, 198], [59, 174]],
    3: [[40, 132], [96, 128], [105, 82], [196, 78], [198, 101], [286, 102], [290, 127], [512, 128], [512, 213], [347, 215], [329, 261], [286, 268], [255, 249], [238, 213], [240, 190], [94, 185], [87, 211], [62, 213], [47, 193]],
    6: [[0, 50], [171, 60], [173, 100], [246, 110], [246, 130], [394, 138], [398, 174], [383, 199], [290, 209], [282, 238], [256, 260], [219, 257], [194, 235], [178, 209], [0, 209]],
    9: [[0, 30], [206, 28], [223, 76], [339, 78], [342, 118], [302, 156], [292, 206], [265, 238], [240, 234], [232, 203], [118, 281], [0, 288]],
    12: [[0, 169], [44, 160], [58, 95], [88, 68], [190, 60], [200, 74], [312, 63], [322, 90], [475, 91], [480, 120], [395, 150], [392, 186], [366, 211], [340, 210], [322, 176], [214, 202], [181, 238], [65, 268], [62, 288], [0, 288]],
    16: [[0, 54], [107, 47], [114, 70], [232, 58], [239, 84], [376, 87], [377, 117], [325, 142], [320, 171], [299, 191], [280, 184], [266, 157], [159, 193], [123, 221], [0, 250]],
}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sheet(items, path, columns=3):
    canvas = Image.new("RGB", (512 * columns, 316 * ((len(items) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(canvas)
    for slot, (label, array) in enumerate(items):
        left, top = slot % columns * 512, slot // columns * 316
        canvas.paste(Image.fromarray(array).resize((512, 288)), (left, top + 28))
        draw.text((left + 8, top + 7), label, fill="black")
    canvas.save(path)


def load_image(path):
    source = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    width, height = source.size
    scale = max(1024 / width, 576 / height)
    source = source.resize((max(1024, int(width * scale)), max(576, int(height * scale))), Image.Resampling.LANCZOS)
    source = ImageOps.fit(source, (1024, 576), Image.Resampling.BICUBIC)
    return np.array(source.resize((512, 288), Image.Resampling.LANCZOS))


def project(points, intrinsic, pose):
    world_to_camera = np.linalg.inv(pose)
    camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    homogeneous = camera_points @ intrinsic.T
    pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-8)
    return pixels, camera_points[:, 2]


def sparse_geometry(images, intrinsics, poses):
    detector = cv2.SIFT_create(nfeatures=6000)
    features = {index: detector.detectAndCompute(cv2.cvtColor(images[index], cv2.COLOR_RGB2GRAY), None) for index in REFERENCES}
    points_all, records = [], []
    matcher = cv2.BFMatcher()
    for first, second in itertools.combinations(REFERENCES, 2):
        key_first, desc_first = features[first]
        key_second, desc_second = features[second]
        matches = matcher.knnMatch(desc_first, desc_second, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
        if len(good) < 8:
            continue
        pixels_first = np.float64([key_first[match.queryIdx].pt for match in good])
        pixels_second = np.float64([key_second[match.trainIdx].pt for match in good])
        _, mask = cv2.findFundamentalMat(pixels_first, pixels_second, cv2.FM_RANSAC, 1.5, 0.999)
        if mask is None:
            continue
        pixels_first, pixels_second = pixels_first[mask.ravel() > 0], pixels_second[mask.ravel() > 0]
        projection_first = intrinsics[first] @ np.linalg.inv(poses[first])[:3]
        projection_second = intrinsics[second] @ np.linalg.inv(poses[second])[:3]
        homogeneous = cv2.triangulatePoints(projection_first, projection_second, pixels_first.T, pixels_second.T).T
        points = homogeneous[:, :3] / homogeneous[:, 3:4]
        check_first, depth_first = project(points, intrinsics[first], poses[first])
        check_second, depth_second = project(points, intrinsics[second], poses[second])
        errors = np.maximum(np.linalg.norm(check_first - pixels_first, axis=1), np.linalg.norm(check_second - pixels_second, axis=1))
        valid = np.isfinite(points).all(axis=1) & (depth_first > 0) & (depth_second > 0) & (errors < 2.5)
        points_all.append(points[valid])
        records.append({"references": [first, second], "ransac_matches": len(points), "accepted_points": int(valid.sum()), "median_reprojection_error_px": float(np.median(errors[valid])) if valid.any() else None})
    if not points_all or sum(len(points) for points in points_all) < 30:
        raise RuntimeError("Insufficient reference-only triangulated points for dense depth bounds")
    return np.concatenate(points_all), records


def dense_geometry(images, intrinsics, poses, sparse):
    height, width = images[REFERENCES[0]].shape[:2]
    pixel_y, pixel_x = np.mgrid[:height, :width].astype(np.float32)
    pixels = np.stack([pixel_x, pixel_y, np.ones_like(pixel_x)], -1)
    gray = {index: cv2.cvtColor(images[index], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255 for index in REFERENCES}
    depths, costs, masks = {}, {}, {}
    for reference in REFERENCES:
        location, sparse_depth = project(sparse, intrinsics[reference], poses[reference])
        inside = (sparse_depth > 0) & (location[:, 0] > 0) & (location[:, 0] < width) & (location[:, 1] > 0) & (location[:, 1] < height)
        near, far = np.quantile(sparse_depth[inside], [0.02, 0.98]) * np.array([0.7, 1.3])
        planes = 1 / np.linspace(1 / near, 1 / far, 80)
        rays = pixels @ np.linalg.inv(intrinsics[reference]).T
        source = gray[reference]
        source_mean = cv2.boxFilter(source, -1, (7, 7))
        source_variance = cv2.boxFilter(source * source, -1, (7, 7)) - source_mean**2
        volume = []
        for depth in planes:
            points = (rays * depth) @ poses[reference][:3, :3].T + poses[reference][:3, 3]
            neighbor_costs = []
            for neighbor in REFERENCES:
                if neighbor == reference:
                    continue
                projected, projected_depth = project(points.reshape(-1, 3), intrinsics[neighbor], poses[neighbor])
                warp_x, warp_y = projected.reshape(height, width, 2).astype(np.float32).transpose(2, 0, 1)
                valid = (warp_x >= 3) & (warp_x < width - 3) & (warp_y >= 3) & (warp_y < height - 3) & (projected_depth.reshape(height, width) > 0)
                warped = cv2.remap(gray[neighbor], warp_x, warp_y, cv2.INTER_LINEAR)
                mean = cv2.boxFilter(warped, -1, (7, 7))
                variance = cv2.boxFilter(warped**2, -1, (7, 7)) - mean**2
                covariance = cv2.boxFilter(source * warped, -1, (7, 7)) - source_mean * mean
                cost = 1 - covariance / np.sqrt(np.maximum(source_variance * variance, 1e-6))
                cost[~valid] = 2
                neighbor_costs.append(cost)
            best_neighbors = np.partition(np.stack(neighbor_costs), 1, axis=0)[:2]
            volume.append(best_neighbors.mean(axis=0))
        volume = np.stack(volume)
        best = volume.argmin(axis=0)
        depths[reference] = planes[best].astype(np.float32)
        costs[reference] = np.take_along_axis(volume, best[None], axis=0)[0]
        masks[reference] = (costs[reference] < 0.35) & (source_variance > 0.0004) & (best > 0) & (best < len(planes) - 1)
        print(f"depth {reference:03d}: bounds {near:.3f}..{far:.3f}, photo-valid {masks[reference].mean():.3f}", flush=True)
    all_points, all_colors, all_support, all_task, all_source = [], [], [], [], []
    for reference in REFERENCES:
        rays = pixels @ np.linalg.inv(intrinsics[reference]).T
        points = (rays * depths[reference][..., None]) @ poses[reference][:3, :3].T + poses[reference][:3, 3]
        support = np.ones((height, width), dtype=np.uint8)
        for neighbor in REFERENCES:
            if neighbor == reference:
                continue
            projected, expected_depth = project(points.reshape(-1, 3), intrinsics[neighbor], poses[neighbor])
            warp_x, warp_y = projected.reshape(height, width, 2).astype(np.float32).transpose(2, 0, 1)
            sampled = cv2.remap(depths[neighbor], warp_x, warp_y, cv2.INTER_NEAREST)
            valid_neighbor = cv2.remap(masks[neighbor].astype(np.uint8), warp_x, warp_y, cv2.INTER_NEAREST) > 0
            expected_depth = expected_depth.reshape(height, width)
            support += valid_neighbor & (expected_depth > 0) & (np.abs(sampled - expected_depth) < 0.03 * np.maximum(expected_depth, 1e-8))
        valid = masks[reference] & (support >= 2)
        roi = np.zeros((height, width), np.uint8)
        cv2.fillPoly(roi, [np.array(POLYGONS[reference], np.int32)], 1)
        all_points.append(points[valid])
        all_colors.append(images[reference][valid])
        all_support.append(support[valid])
        all_task.append(roi[valid])
        all_source.append(np.full(valid.sum(), reference, np.uint8))
    return {"points": np.concatenate(all_points), "colors": np.concatenate(all_colors), "support": np.concatenate(all_support), "task_roi": np.concatenate(all_task), "source_frame": np.concatenate(all_source)}, depths, masks


def render(cloud, intrinsic, pose, radius):
    width, height = 512, 288
    projected, depths = project(cloud["points"], intrinsic, pose)
    eligible = np.isfinite(projected).all(axis=1) & (depths > 0)
    eligible &= (projected[:, 0] >= -radius - 1) & (projected[:, 0] <= width + radius)
    eligible &= (projected[:, 1] >= -radius - 1) & (projected[:, 1] <= height + radius)
    centers = np.zeros(projected.shape, np.int32)
    centers[eligible] = np.rint(projected[eligible]).astype(np.int32)
    point_indices, pixel_indices = [], []
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            horizontal = centers[:, 0] + offset_x
            vertical = centers[:, 1] + offset_y
            valid = eligible & (horizontal >= 0) & (horizontal < width) & (vertical >= 0) & (vertical < height)
            point_indices.append(np.flatnonzero(valid))
            pixel_indices.append(vertical[valid] * width + horizontal[valid])
    point_indices, pixel_indices = np.concatenate(point_indices), np.concatenate(pixel_indices)
    order = np.argsort(depths[point_indices], kind="stable")
    unique_pixels, first = np.unique(pixel_indices[order], return_index=True)
    selected = point_indices[order[first]]
    rgb = np.zeros((height * width, 3), np.uint8)
    support = np.zeros(height * width, np.uint8)
    task = np.zeros(height * width, np.uint8)
    depth = np.zeros(height * width, np.float32)
    rgb[unique_pixels] = cloud["colors"][selected]
    support[unique_pixels] = cloud["support"][selected]
    task[unique_pixels] = cloud["task_roi"][selected]
    depth[unique_pixels] = depths[selected]
    return rgb.reshape(height, width, 3), support.reshape(height, width), task.reshape(height, width), depth.reshape(height, width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/truck_evidence_preflight_20261110")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cv2.setNumThreads(8)
    dataset = ROOT / "datasets/viewcrafter_tnt_subset_20261026/raw/truck/images"
    historical = ROOT / "outputs/viewcrafter_benchmark_20261026/truck_stride3"
    camera_path = historical / "geometry/cameras.json"
    camera = json.loads(camera_path.read_text())
    poses = np.array(camera["c2w_source_relative"])
    intrinsics = []
    for focal, principal in zip(camera["focal_length"], camera["principal_point"]):
        intrinsics.append(np.array([[focal[0], 0, principal[0]], [0, focal[0], principal[1]], [0, 0, 1]]))
    selected = REFERENCES + DEVELOPMENT + EVALUATION
    manifest = {
        "task": "检测卡车，保持可见车身、车轮及邻近地面结构；不推断未观察细节。",
        "references": REFERENCES, "development": DEVELOPMENT, "evaluation": EVALUATION,
        "camera_source": str(camera_path), "camera_sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
        "target_images_used_in_historical_camera_estimation": True,
        "target_pixels_used_in_new_geometry": False,
        "geometry_method": "CPU SIFT triangulation + 80-plane ZNCC multiview stereo; not DUSt3R",
        "thresholds": {"sift_ratio": 0.7, "reprojection_px": 2.5, "zncc_cost": 0.35, "depth_consistency_relative": 0.03, "minimum_support": 2},
        "annotation_status": "assistant visually traced coarse reference polygons; proposals, not human-reviewed ground truth",
        "reference_polygons_512x288": POLYGONS,
        "new_diffusion_status": "not_run_cuda_unavailable_and_original_environment_missing",
        "task_adapter_status": "not_implemented_or_trained",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"opencv": cv2.__version__, "numpy": np.__version__, "device": "cpu"},
        "frame_to_raw_file": {index: f"{3 * index + 1:06d}.jpg" for index in selected},
        "input_sha256": {f"{3 * index + 1:06d}.jpg": hashlib.sha256((dataset / f"{3 * index + 1:06d}.jpg").read_bytes()).hexdigest() for index in selected},
    }
    write_json(args.output / "manifest.json", manifest)
    images = {index: load_image(dataset / f"{3 * index + 1:06d}.jpg") for index in REFERENCES}
    previews = []
    for index in REFERENCES:
        preview = images[index].copy()
        cv2.polylines(preview, [np.array(POLYGONS[index], np.int32)], True, (255, 180, 0), 2)
        previews.append((f"REFERENCE {index:03d} / coarse truck ROI", preview))
    sheet(previews, args.output / "references.png")
    sparse, pairs = sparse_geometry(images, intrinsics, poses)
    write_json(args.output / "reference_matching.json", pairs)
    print(f"Accepted sparse reference points: {len(sparse)}", flush=True)
    cloud, depths, masks = dense_geometry(images, intrinsics, poses, sparse)
    if len(cloud["points"]) == 0:
        raise RuntimeError("No multiview-consistent points")
    np.savez_compressed(args.output / "reference_cloud.npz", **cloud)
    np.savez_compressed(args.output / "reference_depths.npz", **{f"depth_{index:03d}": depths[index] for index in REFERENCES}, **{f"photo_valid_{index:03d}": masks[index] for index in REFERENCES})
    metrics, comparisons, support_panels, history_panels = [], [], [], []
    for split, indices in [("reference", REFERENCES), ("development", DEVELOPMENT), ("evaluation", EVALUATION)]:
        for index in indices:
            truth = images[index] if index in images else load_image(dataset / f"{3 * index + 1:06d}.jpg")
            raw, raw_support, _, _ = render(cloud, intrinsics[index], poses[index], 0)
            splat, support, task, depth = render(cloud, intrinsics[index], poses[index], 1)
            valid = support > 0
            raster_filled = valid & (raw_support == 0)
            task_region = (task > 0) & valid
            error = np.abs(splat.astype(np.float32) - truth.astype(np.float32)).mean(axis=-1) / 255
            record = {"frame": index, "split": split, "point_coverage": float((raw_support > 0).mean()), "splat_coverage": float(valid.mean()), "splat_added_fraction": float(raster_filled.mean()), "supported_rgb_mae": float(error[valid].mean()), "task_roi_rgb_mae": float(error[task_region].mean()) if task_region.any() else None}
            metrics.append(record)
            if split != "evaluation":
                continue
            directory = args.output / f"frame_{index:03d}"
            directory.mkdir()
            for name, array in [("truth", truth), ("point_projection", raw), ("splat_projection", splat)]:
                Image.fromarray(array).save(directory / f"{name}.png")
            context = cv2.dilate(task, np.ones((13, 13), np.uint8)) > 0
            attention = np.where(task > 0, 1, np.where(context, 0.4, 0.1)) * (support / len(REFERENCES))
            np.savez_compressed(directory / "evidence_condition.npz", rgb=splat / 255, depth=depth, support_count=support, task_roi=task, task_attention=attention, valid_projection=valid, raster_added=raster_filled, unresolved=~valid)
            status = np.full_like(splat, (45, 45, 45))
            status[raw_support > 0] = (40, 180, 85)
            status[raster_filled] = (240, 165, 35)
            Image.fromarray(status).save(directory / "support_status.png")
            comparisons.extend([(f"REAL {index:03d}", truth), ("NEW: multi-reference point projection", raw), ("NEW: 3x3 splat (not diffusion)", splat)])
            task_preview = (splat * 0.45).astype(np.uint8)
            task_preview[task_region] = splat[task_region]
            support_panels.extend([(f"REAL {index:03d}", truth), ("GREEN point / ORANGE splat / GRAY unknown", status), ("Projected coarse truck ROI", task_preview)])
            old_condition = np.array(Image.open(historical / "geometry/conditions" / f"{index:03d}.png").resize((512, 288)))
            old_generated = np.array(Image.open(historical / "viewcrafter25/generated" / f"{index:03d}.png").resize((512, 288)))
            history_panels.extend([(f"REAL {index:03d}", truth), ("HISTORICAL single-reference condition", old_condition), ("HISTORICAL ViewCrafter25 (not new)", old_generated)])
    sheet(comparisons, args.output / "new_geometry_comparison.png")
    sheet(support_panels, args.output / "evidence_support.png")
    sheet(history_panels, args.output / "historical_diffusion_comparison.png")
    evaluation = [entry for entry in metrics if entry["split"] == "evaluation"]
    report = {"new_diffusion_ran": False, "task_adapter_ran": False, "reference_count": len(REFERENCES), "accepted_sparse_points": len(sparse), "dense_points": len(cloud["points"]), "seconds": time.perf_counter() - started, "metrics": metrics, "evaluation_mean": {key: float(np.mean([entry[key] for entry in evaluation])) for key in ["point_coverage", "splat_coverage", "supported_rgb_mae", "task_roi_rgb_mae"]}, "limitations": ["Historical cameras jointly estimated using target photos", "CPU stereo baseline is not proposed DUSt3R-plus-adapter method", "Coarse ROI polygons are not human-reviewed instance ground truth", "Splat additions do not prove observed surface continuity; unresolved does not mean never-observed", "MAE is measured only on projected pixels; no detector, identity, or downstream gain claim"]}
    write_json(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
