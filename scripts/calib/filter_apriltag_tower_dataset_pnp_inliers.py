#!/usr/bin/env python3
"""Filter an AprilTag tower dataset to per-view fixed-intrinsic PnP inliers."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import struct
import time

import numpy as np


def load_refine_module():
    module_path = Path(__file__).resolve().parent / "refine_outer_tower_delta_prior.py"
    spec = importlib.util.spec_from_file_location("refine_outer_tower_delta_prior", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def u32(value):
    return struct.pack(">I", int(value))


def i32(value):
    return struct.pack(">i", int(value))


def f32(value):
    return struct.pack("<f", float(value))


def write_dataset(path, image_sizes, imagesets, known_points, cell_length_m):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"calib_data")
        stream.write(u32(1))
        stream.write(u32(len(image_sizes)))
        for width, height in image_sizes:
            stream.write(u32(width))
            stream.write(u32(height))

        stream.write(u32(len(imagesets)))
        for imageset in imagesets:
            encoded = imageset["filename"].encode("utf-8")
            stream.write(u32(len(encoded)))
            stream.write(encoded)
            for features in imageset["features"]:
                stream.write(u32(len(features)))
                for x, y, feature_id in features:
                    stream.write(f32(x))
                    stream.write(f32(y))
                    stream.write(i32(feature_id))

        stream.write(u32(1))
        stream.write(f32(cell_length_m))
        stream.write(u32(0))
        stream.write(u32(len(known_points)))
        for feature_id in sorted(known_points):
            x, y, z = known_points[feature_id]
            stream.write(i32(feature_id))
            stream.write(f32(x))
            stream.write(f32(y))
            stream.write(f32(z))


def cv_intrinsics(intrinsic):
    p = intrinsic["params"]
    camera_matrix = np.asarray([
        [p[0], 0.0, p[2]],
        [0.0, p[1], p[3]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.asarray(
        [p[4], p[5], p[10], p[11], p[6], p[7], p[8], p[9]],
        dtype=np.float64)
    return camera_matrix, dist_coeffs


def solve_view(cv2, object_points, image_points, camera_matrix, dist_coeffs, args):
    if len(object_points) < args.min_points_per_view:
        return None, []
    object_points = np.asarray(object_points, dtype=np.float32)
    image_points = np.asarray(image_points, dtype=np.float32)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            None,
            None,
            False,
            args.ransac_iterations,
            args.ransac_reprojection_threshold_px,
            0.99,
            flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None, []
    if not ok or inliers is None or len(inliers) < args.min_points_per_view:
        return None, []
    inlier_indices = [int(index) for index in inliers.ravel()]
    inlier_object = object_points[inlier_indices]
    inlier_image = image_points[inlier_indices]
    try:
        cv2.solvePnP(
            inlier_object,
            inlier_image,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE)
        projected, _ = cv2.projectPoints(
            inlier_object,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs)
    except cv2.error:
        return None, []
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - inlier_image, axis=1)
    median_error = float(np.median(errors)) if errors.size else float("inf")
    mean_error = float(np.mean(errors)) if errors.size else float("inf")
    if median_error > args.max_median_error_px:
        return None, []
    return {
        "inlier_count": len(inlier_indices),
        "mean_error_px": mean_error,
        "median_error_px": median_error,
    }, inlier_indices


def run(args):
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV with aruco/calib3d support is required.") from exc

    refine = load_refine_module()
    dataset = refine.read_dataset(args.dataset)
    manifest = refine.read_manifest(args.manifest, dataset["camera_count"])
    intrinsics = refine.collect_intrinsics(
        args.intrinsics_dir,
        manifest,
        dataset["image_sizes"],
        args.intrinsics_mode)
    cv_models = [cv_intrinsics(intrinsic) for intrinsic in intrinsics]

    start = time.time()
    output_imagesets = []
    per_camera = [
        {
            "camera_index": idx,
            "camera_id": row["camera_id"],
            "input_observations": 0,
            "kept_observations": 0,
            "positive_views": 0,
            "solved_views": 0,
            "median_view_error_px": [],
        }
        for idx, row in enumerate(manifest)
    ]
    view_rows = []

    for frame_index, imageset in enumerate(dataset["imagesets"]):
        camera_features_out = []
        for camera_index, features in enumerate(imageset["features"]):
            stats = per_camera[camera_index]
            stats["input_observations"] += len(features)
            if features:
                stats["positive_views"] += 1
            object_points = []
            image_points = []
            feature_rows = []
            for x, y, feature_id in features:
                point = dataset["known_points"].get(feature_id)
                if point is None:
                    continue
                object_points.append(point)
                image_points.append([x, y])
                feature_rows.append((x, y, feature_id))

            camera_matrix, dist_coeffs = cv_models[camera_index]
            view, inlier_indices = solve_view(
                cv2,
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                args)
            if view is None:
                filtered = []
                status = "failed"
                mean_error = ""
                median_error = ""
            else:
                filtered = [feature_rows[index] for index in inlier_indices]
                status = "solved"
                stats["solved_views"] += 1
                stats["median_view_error_px"].append(view["median_error_px"])
                mean_error = f"{view['mean_error_px']:.8g}"
                median_error = f"{view['median_error_px']:.8g}"
            stats["kept_observations"] += len(filtered)
            camera_features_out.append(filtered)
            if args.views_tsv:
                view_rows.append({
                    "frame_index": frame_index,
                    "filename": imageset["filename"],
                    "camera_index": camera_index,
                    "camera_id": manifest[camera_index]["camera_id"],
                    "status": status,
                    "input_observations": len(features),
                    "kept_observations": len(filtered),
                    "mean_error_px": mean_error,
                    "median_error_px": median_error,
                })
        output_imagesets.append({
            "filename": imageset["filename"],
            "features": camera_features_out,
        })

    write_dataset(
        args.output_dataset,
        dataset["image_sizes"],
        output_imagesets,
        dataset["known_points"],
        args.cell_length_m)

    per_camera_rows = []
    for row in per_camera:
        medians = row.pop("median_view_error_px")
        output = dict(row)
        output["kept_ratio"] = (
            output["kept_observations"] / output["input_observations"]
            if output["input_observations"] else 0.0)
        output["median_view_error_px"] = float(np.median(medians)) if medians else None
        per_camera_rows.append(output)

    summary = {
        "mode": "apriltag_tower_pnp_inlier_filter",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": time.time() - start,
        "input_dataset": str(args.dataset),
        "output_dataset": str(args.output_dataset),
        "camera_count": dataset["camera_count"],
        "imageset_count": len(dataset["imagesets"]),
        "input_observations": int(sum(row["input_observations"] for row in per_camera_rows)),
        "kept_observations": int(sum(row["kept_observations"] for row in per_camera_rows)),
        "kept_ratio": (
            sum(row["kept_observations"] for row in per_camera_rows)
            / max(1, sum(row["input_observations"] for row in per_camera_rows))),
        "settings": {
            "intrinsics_mode": args.intrinsics_mode,
            "min_points_per_view": args.min_points_per_view,
            "ransac_reprojection_threshold_px": args.ransac_reprojection_threshold_px,
            "max_median_error_px": args.max_median_error_px,
        },
        "per_camera": per_camera_rows,
    }

    if args.per_camera_tsv:
        args.per_camera_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_camera_tsv.open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "camera_index", "camera_id", "input_observations", "kept_observations",
                "kept_ratio", "positive_views", "solved_views", "median_view_error_px",
            ]
            writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_camera_rows)

    if args.views_tsv:
        args.views_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.views_tsv.open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "frame_index", "filename", "camera_index", "camera_id", "status",
                "input_observations", "kept_observations",
                "mean_error_px", "median_error_px",
            ]
            writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
            writer.writeheader()
            writer.writerows(view_rows)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--intrinsics-dir", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--intrinsics-mode", choices=["central_opencv", "colmap_fixed"], default="central_opencv")
    parser.add_argument("--min-points-per-view", type=int, default=4)
    parser.add_argument("--ransac-iterations", type=int, default=100)
    parser.add_argument("--ransac-reprojection-threshold-px", type=float, default=8.0)
    parser.add_argument("--max-median-error-px", type=float, default=8.0)
    parser.add_argument("--cell-length-m", type=float, default=0.1)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--per-camera-tsv", type=Path)
    parser.add_argument("--views-tsv", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
