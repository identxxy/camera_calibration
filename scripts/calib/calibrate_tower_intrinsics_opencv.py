#!/usr/bin/env python3
"""Fast per-camera OpenCV intrinsics from an AprilTag tower calib_data dataset."""

import argparse
import csv
import importlib.util
import json
import math
import multiprocessing
import os
import re
import time
from pathlib import Path

import numpy as np
import yaml


CALIBRATION_MODES = (
    "fxfycxcy_k1k2p1p2k3",
    "fxfycxcy_no_dist",
    "fxfy_fixed_center_no_dist",
)

SUMMARY_FIELDS = [
    "camera_index",
    "camera_id",
    "user_id",
    "stage_name",
    "machine",
    "width",
    "height",
    "status",
    "reason",
    "input_observations",
    "candidate_views",
    "candidate_points",
    "usable_views",
    "usable_points",
    "bbox_area_ratio",
    "rms",
    "mean_error_px",
    "median_error_px",
    "p90_error_px",
    "max_error_px",
    "fx",
    "fy",
    "cx",
    "cy",
    "k1",
    "k2",
    "k3",
    "p1",
    "p2",
    "prior_source",
    "output_source",
    "calibration_mode",
    "intrinsics_yaml",
    "opencv_intrinsics_yaml",
]


def load_refine_module():
    module_path = Path(__file__).resolve().parent / "refine_outer_tower_delta_prior.py"
    spec = importlib.util.spec_from_file_location("refine_outer_tower_delta_prior", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def padded_params(params):
    return (list(params) + [0.0] * 12)[:12]


def initial_intrinsic_from_image_size(image_size):
    width, height = image_size
    focal = float(max(width, height))
    return {
        "width": int(width),
        "height": int(height),
        "params": [focal, focal, float(width) * 0.5, float(height) * 0.5] + [0.0] * 8,
    }


def intrinsic_from_opencv5(width, height, camera_matrix, dist_coeffs):
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    return {
        "width": int(width),
        "height": int(height),
        "params": [
            float(camera_matrix[0, 0]),
            float(camera_matrix[1, 1]),
            float(camera_matrix[0, 2]),
            float(camera_matrix[1, 2]),
            float(k1),
            float(k2),
            float(k3),
            0.0,
            0.0,
            0.0,
            float(p1),
            float(p2),
        ],
    }


def opencv5_from_intrinsic(intrinsic):
    params = padded_params(intrinsic["params"])
    camera_matrix = np.asarray([
        [params[0], 0.0, params[2]],
        [0.0, params[1], params[3]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.asarray(
        [params[4], params[5], params[10], params[11], params[6]],
        dtype=np.float64).reshape(5, 1)
    return camera_matrix, dist_coeffs


def force_no_distortion(intrinsic, fixed_center=False):
    result = {
        "width": int(intrinsic["width"]),
        "height": int(intrinsic["height"]),
        "params": padded_params(intrinsic["params"]),
    }
    if fixed_center:
        result["params"][2] = result["width"] * 0.5
        result["params"][3] = result["height"] * 0.5
    result["params"][4:] = [0.0] * 8
    return result


def safe_camera_label(camera_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(camera_id))


def uniform_sample_views(views, max_views):
    views = list(views)
    max_views = int(max_views)
    if max_views <= 0:
        return []
    if len(views) <= max_views:
        return views
    indices = [int(i) for i in np.linspace(0, len(views) - 1, num=max_views, dtype=np.int64)]
    seen = set()
    unique = []
    for index in indices:
        if index not in seen:
            unique.append(index)
            seen.add(index)
    if len(unique) < max_views:
        for index in range(len(views)):
            if index not in seen:
                unique.append(index)
                seen.add(index)
                if len(unique) == max_views:
                    break
    return [views[index] for index in sorted(unique)]


def collect_camera_views(dataset, camera_index, min_points_per_view):
    known_points = dataset["known_points"]
    views = []
    input_observations = 0
    for frame_index, imageset in enumerate(dataset["imagesets"]):
        features = imageset["features"][camera_index]
        input_observations += len(features)
        object_points = []
        image_points = []
        for x, y, feature_id in features:
            point = known_points.get(feature_id)
            if point is None:
                continue
            object_points.append(point)
            image_points.append([x, y])
        if len(object_points) < min_points_per_view:
            continue
        views.append({
            "frame_index": frame_index,
            "filename": imageset["filename"],
            "object_points": np.asarray(object_points, dtype=np.float32),
            "image_points": np.asarray(image_points, dtype=np.float32),
        })
    return views, input_observations


def camera_id_from_manifest_row(row, index):
    return str(row.get("camera_id") or row.get("user_id") or index)


def build_camera_jobs(dataset, manifest, initial_intrinsics, prior_sources, args):
    jobs = []
    for camera_index, row in enumerate(manifest):
        views, input_observations = collect_camera_views(
            dataset,
            camera_index,
            args.min_points_per_view)
        sampled_views = uniform_sample_views(views, args.max_views_per_camera)
        camera_id = camera_id_from_manifest_row(row, camera_index)
        jobs.append({
            "camera_index": camera_index,
            "camera_id": camera_id,
            "user_id": str(row.get("user_id") or camera_id),
            "stage_name": str(row.get("stage_name") or ""),
            "machine": str(row.get("machine") or ""),
            "image_size": tuple(dataset["image_sizes"][camera_index]),
            "views": sampled_views,
            "candidate_views": len(views),
            "candidate_points": int(sum(len(view["object_points"]) for view in views)),
            "input_observations": int(input_observations),
            "initial_intrinsic": initial_intrinsics[camera_index],
            "prior_source": prior_sources[camera_index],
            "min_views": int(args.min_views),
            "calibration_mode": args.calibration_mode,
        })
    return jobs


def calibration_flags(cv2, mode):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if mode in ("fxfycxcy_no_dist", "fxfy_fixed_center_no_dist"):
        flags |= getattr(cv2, "CALIB_ZERO_TANGENT_DIST", 0)
        flags |= getattr(cv2, "CALIB_FIX_TANGENT_DIST", 0)
        for name in ("CALIB_FIX_K1", "CALIB_FIX_K2", "CALIB_FIX_K3", "CALIB_FIX_K4", "CALIB_FIX_K5", "CALIB_FIX_K6"):
            flags |= getattr(cv2, name, 0)
    if mode == "fxfy_fixed_center_no_dist":
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    return flags


def bbox_area_ratio(views, width, height):
    if not views:
        return 0.0
    points = [np.asarray(view["image_points"], dtype=np.float64).reshape(-1, 2) for view in views]
    points = [p for p in points if p.size]
    if not points:
        return 0.0
    points = np.concatenate(points, axis=0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    area = max(0.0, float(hi[0] - lo[0])) * max(0.0, float(hi[1] - lo[1]))
    denom = max(1.0, float(width) * float(height))
    return area / denom


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def residual_stats(cv2, object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    residuals = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        observed = img.reshape(-1, 2)
        residuals.append(projected - observed)
    if not residuals:
        return {
            "mean_error_px": None,
            "median_error_px": None,
            "p90_error_px": None,
            "max_error_px": None,
        }
    errors = np.linalg.norm(np.concatenate(residuals, axis=0), axis=1)
    return {
        "mean_error_px": finite_float(np.mean(errors)),
        "median_error_px": finite_float(np.median(errors)),
        "p90_error_px": finite_float(np.percentile(errors, 90)),
        "max_error_px": finite_float(np.max(errors)),
    }


def residual_rows(cv2, views, rvecs, tvecs, camera_matrix, dist_coeffs):
    rows = []
    for view, rvec, tvec in zip(views, rvecs, tvecs):
        object_points = np.asarray(view["object_points"], dtype=np.float32)
        image_points = np.asarray(view["image_points"], dtype=np.float32).reshape(-1, 2)
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs)
        projected = projected.reshape(-1, 2)
        for point_index, (observed, predicted) in enumerate(zip(image_points, projected)):
            error = predicted - observed
            rows.append({
                "frame_index": int(view["frame_index"]),
                "filename": view["filename"],
                "point_index": int(point_index),
                "observed_x": float(observed[0]),
                "observed_y": float(observed[1]),
                "projected_x": float(predicted[0]),
                "projected_y": float(predicted[1]),
                "error_x": float(error[0]),
                "error_y": float(error[1]),
                "error_px": float(np.linalg.norm(error)),
            })
    return rows


def result_row(job, status, reason, intrinsic, output_source, metrics=None):
    width, height = job["image_size"]
    params = padded_params(intrinsic["params"])
    metrics = dict(metrics or {})
    row = {
        "camera_index": int(job["camera_index"]),
        "camera_id": job["camera_id"],
        "user_id": job["user_id"],
        "stage_name": job["stage_name"],
        "machine": job["machine"],
        "width": int(width),
        "height": int(height),
        "status": status,
        "reason": reason,
        "input_observations": int(job["input_observations"]),
        "candidate_views": int(job["candidate_views"]),
        "candidate_points": int(job["candidate_points"]),
        "usable_views": int(len(job["views"])),
        "usable_points": int(sum(len(view["object_points"]) for view in job["views"])),
        "bbox_area_ratio": finite_float(bbox_area_ratio(job["views"], width, height)),
        "rms": metrics.get("rms"),
        "mean_error_px": metrics.get("mean_error_px"),
        "median_error_px": metrics.get("median_error_px"),
        "p90_error_px": metrics.get("p90_error_px"),
        "max_error_px": metrics.get("max_error_px"),
        "fx": finite_float(params[0]),
        "fy": finite_float(params[1]),
        "cx": finite_float(params[2]),
        "cy": finite_float(params[3]),
        "k1": finite_float(params[4]),
        "k2": finite_float(params[5]),
        "k3": finite_float(params[6]),
        "p1": finite_float(params[10]),
        "p2": finite_float(params[11]),
        "prior_source": job["prior_source"],
        "output_source": output_source,
        "calibration_mode": job["calibration_mode"],
        "intrinsics_yaml": "",
        "opencv_intrinsics_yaml": "",
    }
    return row


def camera_result(job, intrinsic, row):
    result = {
        "camera_index": int(job["camera_index"]),
        "intrinsic": intrinsic,
        "row": row,
    }
    result.update(row)
    return result


def invalid_calibration(camera_matrix, rms):
    if not math.isfinite(float(rms)):
        return True
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    return not (math.isfinite(fx) and math.isfinite(fy) and fx > 0.0 and fy > 0.0)


def calibrate_camera_job(job):
    fallback_intrinsic = job["initial_intrinsic"]
    if job["calibration_mode"] in ("fxfycxcy_no_dist", "fxfy_fixed_center_no_dist"):
        fallback_intrinsic = force_no_distortion(
            fallback_intrinsic,
            fixed_center=job["calibration_mode"] == "fxfy_fixed_center_no_dist")

    if len(job["views"]) < job["min_views"]:
        row = result_row(
            job,
            "insufficient_views",
            f"usable_views_lt_{job['min_views']}",
            fallback_intrinsic,
            job["prior_source"])
        return camera_result(job, fallback_intrinsic, row)

    try:
        import cv2
    except ImportError as exc:
        row = result_row(
            job,
            "opencv_import_error",
            str(exc),
            fallback_intrinsic,
            job["prior_source"])
        return camera_result(job, fallback_intrinsic, row)

    width, height = job["image_size"]
    camera_matrix, dist_coeffs = opencv5_from_intrinsic(fallback_intrinsic)
    if job["calibration_mode"] in ("fxfycxcy_no_dist", "fxfy_fixed_center_no_dist"):
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    if job["calibration_mode"] == "fxfy_fixed_center_no_dist":
        camera_matrix[0, 2] = float(width) * 0.5
        camera_matrix[1, 2] = float(height) * 0.5

    object_points = [
        np.ascontiguousarray(view["object_points"], dtype=np.float32)
        for view in job["views"]
    ]
    image_points = [
        np.ascontiguousarray(view["image_points"], dtype=np.float32)
        for view in job["views"]
    ]

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
        100,
        1e-9,
    )
    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            (int(width), int(height)),
            camera_matrix,
            dist_coeffs,
            flags=calibration_flags(cv2, job["calibration_mode"]),
            criteria=criteria)
    except cv2.error as exc:
        row = result_row(
            job,
            "opencv_error",
            str(exc).splitlines()[0],
            fallback_intrinsic,
            job["prior_source"])
        return camera_result(job, fallback_intrinsic, row)

    if invalid_calibration(camera_matrix, rms):
        row = result_row(
            job,
            "invalid_result",
            "nonfinite_or_nonpositive_focal",
            fallback_intrinsic,
            job["prior_source"])
        return camera_result(job, fallback_intrinsic, row)

    intrinsic = intrinsic_from_opencv5(width, height, camera_matrix, dist_coeffs)
    if job["calibration_mode"] == "fxfycxcy_no_dist":
        intrinsic = force_no_distortion(intrinsic)
    elif job["calibration_mode"] == "fxfy_fixed_center_no_dist":
        intrinsic = force_no_distortion(intrinsic, fixed_center=True)
    stats = residual_stats(cv2, object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs)
    stats["rms"] = finite_float(rms)
    row = result_row(job, "solved", "", intrinsic, "opencv_calibration", stats)
    result = camera_result(job, intrinsic, row)
    result["residual_rows"] = residual_rows(
        cv2,
        job["views"],
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs)
    return result


def ensure_cv2_available():
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise SystemExit("OpenCV with calib3d support is required.") from exc


def collect_initial_intrinsics(refine, args, manifest, image_sizes):
    if args.prior_intrinsics_dir:
        intrinsics = refine.collect_intrinsics(
            args.prior_intrinsics_dir,
            manifest,
            image_sizes,
            args.intrinsics_mode)
        if args.intrinsics_mode == "colmap_fixed":
            source = "colmap_fixed_prior"
        else:
            source = "prior_intrinsics"
        return intrinsics, [source] * len(intrinsics)
    intrinsics = [initial_intrinsic_from_image_size(image_size) for image_size in image_sizes]
    return intrinsics, ["initial_guess"] * len(intrinsics)


def known_points_from_points_yaml(path):
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    points = np.asarray(node["points"], dtype=np.float64).reshape((-1, 3))
    result = {}
    for item in node["feature_id_to_point_index"]:
        result[int(item["feature_id"])] = points[int(item["point_index"])]
    return result


def format_tsv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return str(value)


def write_summary_tsv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_tsv_value(row.get(field)) for field in SUMMARY_FIELDS})


def write_residuals_tsv(path, rows):
    fields = [
        "frame_index",
        "filename",
        "point_index",
        "observed_x",
        "observed_y",
        "projected_x",
        "projected_y",
        "error_x",
        "error_y",
        "error_px",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_tsv_value(row.get(field)) for field in fields})


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_opencv_intrinsics_yaml(path, intrinsic):
    camera_matrix, dist_coeffs = opencv5_from_intrinsic(intrinsic)
    k_data = [float(v) for v in camera_matrix.reshape(-1)]
    d_data = [float(v) for v in dist_coeffs.reshape(-1)[:5]]
    text = "\n".join([
        "%YAML:1.0",
        f"image_width: {int(intrinsic['width'])}",
        f"image_height: {int(intrinsic['height'])}",
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  dt: d",
        "  data: [" + ", ".join(f"{value:.14g}" for value in k_data) + "]",
        "distortion_coefficients:",
        "  rows: 1",
        "  cols: 5",
        "  dt: d",
        "  data: [" + ", ".join(f"{value:.14g}" for value in d_data) + "]",
        "",
    ])
    Path(path).write_text(text, encoding="utf-8")


def write_outputs(refine, output_dir, results, write_opencv_yaml):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in sorted(results, key=lambda item: item["camera_index"]):
        row = dict(result["row"])
        label = safe_camera_label(row["camera_id"])
        intrinsic_path = output_dir / f"intrinsics{row['camera_index']}_{label}.yaml"
        refine.write_intrinsics_yaml(intrinsic_path, result["intrinsic"])
        row["intrinsics_yaml"] = str(intrinsic_path)
        if write_opencv_yaml:
            opencv_path = output_dir / f"opencv_intrinsics{row['camera_index']}_{label}.yaml"
            write_opencv_intrinsics_yaml(opencv_path, result["intrinsic"])
            row["opencv_intrinsics_yaml"] = str(opencv_path)
        if result.get("residual_rows"):
            write_residuals_tsv(
                output_dir / f"residuals_camera{row['camera_index']}_{label}.tsv",
                result["residual_rows"])
        rows.append(row)
    write_summary_tsv(output_dir / "intrinsics_summary.tsv", rows)
    return rows


def run(args):
    start = time.time()
    refine = load_refine_module()
    dataset = refine.read_dataset(args.dataset)
    if args.points_yaml:
        dataset["known_points"] = known_points_from_points_yaml(args.points_yaml)
    manifest = refine.read_manifest(args.manifest, dataset["camera_count"])
    initial_intrinsics, prior_sources = collect_initial_intrinsics(
        refine,
        args,
        manifest,
        dataset["image_sizes"])
    jobs = build_camera_jobs(dataset, manifest, initial_intrinsics, prior_sources, args)

    if any(len(job["views"]) >= job["min_views"] for job in jobs):
        ensure_cv2_available()

    if args.jobs == 1 or len(jobs) <= 1:
        results = [calibrate_camera_job(job) for job in jobs]
    else:
        worker_count = min(max(1, int(args.jobs)), len(jobs))
        with multiprocessing.Pool(processes=worker_count) as pool:
            results = pool.map(calibrate_camera_job, jobs)

    rows = write_outputs(
        refine,
        args.output_dir,
        results,
        write_opencv_yaml=not args.no_opencv_yaml)
    solved = sum(1 for row in rows if row["status"] == "solved")
    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "prior_intrinsics_dir": str(Path(args.prior_intrinsics_dir).resolve()) if args.prior_intrinsics_dir else None,
        "intrinsics_mode": args.intrinsics_mode,
        "calibration_mode": args.calibration_mode,
        "min_points_per_view": int(args.min_points_per_view),
        "min_views": int(args.min_views),
        "max_views_per_camera": int(args.max_views_per_camera),
        "jobs": int(args.jobs),
        "camera_count": len(rows),
        "solved_camera_count": solved,
        "runtime_sec": time.time() - start,
        "cameras": rows,
    }
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {Path(args.output_dir).resolve()}")
    print(f"solved {solved}/{len(rows)} cameras")
    print(f"summary {summary_path.resolve()}")
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fast per-camera OpenCV CentralOpenCV intrinsics from AprilTag tower calib_data.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prior-intrinsics-dir", type=Path)
    parser.add_argument(
        "--points-yaml",
        type=Path,
        help=(
            "Optional feature_id-to-3D point mapping exported by a fixed-rig state. "
            "Use this for repo board datasets whose binary dataset does not carry "
            "known 3D geometry in the tower-specific reader."
        ),
    )
    parser.add_argument(
        "--intrinsics-mode",
        choices=("central_opencv", "colmap_fixed"),
        default="central_opencv")
    parser.add_argument("--min-points-per-view", type=int, default=8)
    parser.add_argument("--min-views", type=int, default=30)
    parser.add_argument("--max-views-per-camera", type=int, default=160)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--calibration-mode",
        choices=CALIBRATION_MODES,
        default="fxfycxcy_k1k2p1p2k3")
    parser.add_argument("--no-opencv-yaml", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.min_points_per_view < 4:
        raise SystemExit("--min-points-per-view must be >= 4")
    if args.min_views < 1:
        raise SystemExit("--min-views must be >= 1")
    if args.max_views_per_camera < 1:
        raise SystemExit("--max-views-per-camera must be >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
