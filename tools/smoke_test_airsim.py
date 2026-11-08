import argparse
import json
import time
from pathlib import Path

import airsim
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/airsim_smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = airsim.VehicleClient(ip=args.host, port=args.port, timeout_value=args.timeout)
    client.confirmConnection()
    original_pose = client.simGetVehiclePose()
    report = {
        "client_version": client.getClientVersion(),
        "server_version": client.getServerVersion(),
        "scene_object_count": len(client.simListSceneObjects()),
        "views": [],
    }
    rgb_views = []
    try:
        for name, position, yaw in [
            ("reference", (0, 0, -2), 0),
            ("moved", (3, 0, -4), 0.35),
        ]:
            requested_pose = airsim.Pose(
                airsim.Vector3r(*position), airsim.to_quaternion(0, 0, yaw)
            )
            client.simSetVehiclePose(requested_pose, True)
            time.sleep(2)
            vehicle_pose = client.simGetVehiclePose()
            started = time.monotonic()
            responses = client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True, False),
                airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),
            ])
            elapsed = time.monotonic() - started
            if len(responses) != 3:
                raise RuntimeError("Expected RGB, planar depth, and segmentation responses")
            scene, depth_response, segmentation_response = responses
            for response in responses:
                if response.width <= 0 or response.height <= 0:
                    raise RuntimeError(f"Empty camera response: {response.message}")
            rgb = np.frombuffer(scene.image_data_uint8, dtype=np.uint8).reshape(
                scene.height, scene.width, 3
            )
            segmentation = np.frombuffer(
                segmentation_response.image_data_uint8, dtype=np.uint8
            ).reshape(segmentation_response.height, segmentation_response.width, 3)
            depth = np.asarray(depth_response.image_data_float, dtype=np.float32).reshape(
                depth_response.height, depth_response.width
            )
            if not np.isfinite(depth).all() or np.any(depth < 0):
                raise RuntimeError("Depth contains nonfinite or negative values")
            if float(rgb.std()) < 1:
                raise RuntimeError("RGB image is nearly constant")
            actual_position = np.array([
                scene.camera_position.x_val,
                scene.camera_position.y_val,
                scene.camera_position.z_val,
            ])
            vehicle_position = np.array([
                vehicle_pose.position.x_val,
                vehicle_pose.position.y_val,
                vehicle_pose.position.z_val,
            ])
            position_error = float(np.linalg.norm(vehicle_position - np.asarray(position)))
            if position_error > 0.05:
                raise RuntimeError(f"Unexpected vehicle position error: {position_error} m")
            Image.fromarray(rgb).save(args.output / f"{name}_rgb.png")
            Image.fromarray(segmentation).save(args.output / f"{name}_segmentation.png")
            np.save(args.output / f"{name}_depth_m.npy", depth)
            depth_display = (np.clip(depth, 0, 50) / 50 * 255).astype(np.uint8)
            Image.fromarray(depth_display).save(args.output / f"{name}_depth_0_50m.png")
            camera = client.simGetCameraInfo("front_center")
            report["views"].append({
                "name": name,
                "width": scene.width,
                "height": scene.height,
                "capture_seconds": elapsed,
                "timestamp": scene.time_stamp,
                "requested_position_ned_m": position,
                "vehicle_position_ned_m": vehicle_position.tolist(),
                "camera_position_ned_m": actual_position.tolist(),
                "vehicle_position_error_m": position_error,
                "camera_orientation_xyzw": [
                    scene.camera_orientation.x_val,
                    scene.camera_orientation.y_val,
                    scene.camera_orientation.z_val,
                    scene.camera_orientation.w_val,
                ],
                "fov_degrees": camera.fov,
                "rgb_std": float(rgb.std()),
                "depth_min_m": float(depth.min()),
                "depth_max_m": float(depth.max()),
                "segmentation_color_count": int(
                    len(np.unique(segmentation.reshape(-1, 3), axis=0))
                ),
            })
            rgb_views.append(rgb)
            print(f"Captured {name}: {scene.width}x{scene.height}, {elapsed:.2f}s", flush=True)
    finally:
        try:
            client.simSetVehiclePose(original_pose, True)
        except Exception as error:
            print(f"Could not restore original pose: {error}", flush=True)
    difference = float(np.abs(rgb_views[0].astype(float) - rgb_views[1]).mean())
    if difference < 1:
        raise RuntimeError("Camera movement did not meaningfully change the rendered image")
    report["mean_absolute_rgb_difference"] = difference
    report["status"] = "passed"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
