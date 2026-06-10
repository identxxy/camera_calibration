#!/usr/bin/env python3
"""Export feature-level calibration correspondence reprojection residuals."""

import argparse
import csv
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import yaml


IMAGE_ID_RE = re.compile(r"(?:^|_)(\d+)(?=\.[^.]+$)")

TSV_COLUMNS = [
    "dataset",
    "imageset_index",
    "frame_index",
    "camera_index",
    "source_camera_index",
    "camera_label",
    "stage_name",
    "machine",
    "camera_id",
    "user_id",
    "filename",
    "feature_index",
    "feature_id",
    "point_index",
    "local_x",
    "local_y",
    "local_z",
    "state_x",
    "state_y",
    "state_z",
    "world_x",
    "world_y",
    "world_z",
    "camera_center_x",
    "camera_center_y",
    "camera_center_z",
    "observed_x",
    "observed_y",
    "projected_x",
    "projected_y",
    "residual_x_px",
    "residual_y_px",
    "residual_px",
    "projection_status",
]


def read_u32(data, offset):
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def read_i32(data, offset):
    return struct.unpack_from(">i", data, offset)[0], offset + 4


def read_f32(data, offset):
    return struct.unpack_from("<f", data, offset)[0], offset + 4


def read_dataset(path):
    path = Path(path)
    data = path.read_bytes()
    offset = 0
    if data[:10] != b"calib_data":
        raise ValueError(f"{path} is not a calib_data dataset")
    offset += 10
    version, offset = read_u32(data, offset)
    if version not in (0, 1):
        raise ValueError(f"{path} has unsupported dataset version {version}")

    camera_count, offset = read_u32(data, offset)
    image_sizes = []
    for _ in range(camera_count):
        width, offset = read_u32(data, offset)
        height, offset = read_u32(data, offset)
        image_sizes.append((width, height))

    imageset_count, offset = read_u32(data, offset)
    imagesets = []
    for imageset_index in range(imageset_count):
        filename_len, offset = read_u32(data, offset)
        filename = data[offset:offset + filename_len].decode("utf-8", errors="replace")
        offset += filename_len
        features_by_camera = []
        for _camera_index in range(camera_count):
            feature_count, offset = read_u32(data, offset)
            features = []
            for feature_index in range(feature_count):
                x, offset = read_f32(data, offset)
                y, offset = read_f32(data, offset)
                feature_id, offset = read_i32(data, offset)
                features.append({
                    "feature_index": feature_index,
                    "feature_id": int(feature_id),
                    "x": float(x),
                    "y": float(y),
                })
            features_by_camera.append(features)
        imagesets.append({
            "index": imageset_index,
            "filename": filename,
            "features": features_by_camera,
        })

    geometry_count, offset = read_u32(data, offset)
    for _ in range(geometry_count):
        _cell_length, offset = read_f32(data, offset)
        count_2d, offset = read_u32(data, offset)
        offset += count_2d * 12
        if version >= 1:
            count_3d, offset = read_u32(data, offset)
            offset += count_3d * 16

    if offset != len(data):
        raise ValueError(f"{path} has trailing bytes: parsed={offset}, size={len(data)}")

    return {
        "version": version,
        "camera_count": camera_count,
        "image_sizes": image_sizes,
        "imagesets": imagesets,
    }


def quat_to_matrix(w, x, y, z):
    q = np.asarray([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0:
        raise ValueError("pose quaternion has zero norm")
    q /= norm
    w, x, y, z = q
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def load_pose_file(path):
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pose_count = int(node["pose_count"])
    used = np.zeros(pose_count, dtype=bool)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], pose_count, axis=0)
    translations = np.zeros((pose_count, 3), dtype=np.float64)
    for pose in node.get("poses", []) or []:
        index = int(pose["index"])
        used[index] = True
        translations[index] = [
            float(pose["tx"]),
            float(pose["ty"]),
            float(pose["tz"]),
        ]
        rotations[index] = quat_to_matrix(
            float(pose["qw"]),
            float(pose["qx"]),
            float(pose["qy"]),
            float(pose["qz"]),
        )
    return used, rotations, translations


def pose_matrix(rotation, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def invert_pose(matrix):
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv


def pose_from_yaml_node(node):
    pose = node.get("camera_tr_studio_rig", node)
    return pose_matrix(
        quat_to_matrix(
            float(pose.get("qw", 1.0)),
            float(pose.get("qx", 0.0)),
            float(pose.get("qy", 0.0)),
            float(pose.get("qz", 0.0)),
        ),
        [
            float(pose.get("tx", 0.0)),
            float(pose.get("ty", 0.0)),
            float(pose.get("tz", 0.0)),
        ],
    )


def load_reference_camera_poses(path):
    return load_reference_cameras(path)[0]


def reference_intrinsics_from_camera(camera):
    intrinsics = camera.get("intrinsics") or {}
    model_type = str(intrinsics.get("model") or intrinsics.get("type") or "").strip()
    if model_type in {"CentralOpenCV", "CentralOpenCVModel"}:
        model_type = "CentralOpenCVModel"
    parameters = [float(value) for value in intrinsics.get("parameters", [])]
    if model_type in {"CentralOpenCVModel", "CentralThinPrismFisheyeModel"}:
        parameters = (parameters + [0.0] * 12)[:12]
    return {
        "type": model_type,
        "width": int(intrinsics.get("width", 0)),
        "height": int(intrinsics.get("height", 0)),
        "parameters": np.asarray(parameters, dtype=np.float64),
        "use_equidistant_projection": bool(intrinsics.get("use_equidistant_projection", True)),
    }


def load_reference_cameras(path):
    if not path:
        return {}, {}
    path = Path(path)
    if not path.is_file():
        return {}, {}
    node = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    poses = {}
    intrinsics = {}
    for camera in node.get("cameras", []) or []:
        index = int(camera.get("index", len(poses)))
        poses[index] = pose_from_yaml_node(camera)
        if camera.get("intrinsics"):
            intrinsics[index] = reference_intrinsics_from_camera(camera)
    for pose in node.get("poses", []) or []:
        index = int(pose.get("index", len(poses)))
        poses[index] = pose_from_yaml_node(pose)
    return poses, intrinsics


def average_rotation(rotations):
    if not rotations:
        return np.eye(3, dtype=np.float64)
    matrix = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _s, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def estimate_reference_tr_state(camera_rotations, camera_translations, reference_camera_poses, camera_index_offset):
    transforms = []
    for source_index in range(len(camera_rotations)):
        reference_index = source_index + int(camera_index_offset)
        camera_tr_reference = reference_camera_poses.get(reference_index)
        if camera_tr_reference is None:
            continue
        camera_tr_state = pose_matrix(camera_rotations[source_index], camera_translations[source_index])
        transforms.append(invert_pose(camera_tr_reference) @ camera_tr_state)
    if not transforms:
        return np.eye(4, dtype=np.float64), {
            "reference_alignment_camera_count": 0,
            "reference_alignment_center_rms_m": None,
            "reference_alignment_rotation_rms_deg": None,
        }

    reference_tr_state = pose_matrix(
        average_rotation([transform[:3, :3] for transform in transforms]),
        np.mean([transform[:3, 3] for transform in transforms], axis=0),
    )
    center_residuals = []
    rotation_residuals = []
    for source_index in range(len(camera_rotations)):
        reference_index = source_index + int(camera_index_offset)
        camera_tr_reference = reference_camera_poses.get(reference_index)
        if camera_tr_reference is None:
            continue
        camera_tr_state = pose_matrix(camera_rotations[source_index], camera_translations[source_index])
        predicted_camera_tr_reference = camera_tr_state @ invert_pose(reference_tr_state)
        predicted_center = invert_pose(predicted_camera_tr_reference)[:3, 3]
        target_center = invert_pose(camera_tr_reference)[:3, 3]
        center_residuals.append(float(np.linalg.norm(predicted_center - target_center)))
        delta_r = predicted_camera_tr_reference[:3, :3] @ camera_tr_reference[:3, :3].T
        cos_angle = max(-1.0, min(1.0, (float(np.trace(delta_r)) - 1.0) * 0.5))
        rotation_residuals.append(math.degrees(math.acos(cos_angle)))
    return reference_tr_state, {
        "reference_alignment_camera_count": len(transforms),
        "reference_alignment_center_rms_m": float(np.sqrt(np.mean(np.square(center_residuals)))) if center_residuals else None,
        "reference_alignment_rotation_rms_deg": float(np.sqrt(np.mean(np.square(rotation_residuals)))) if rotation_residuals else None,
    }


def transform_point(matrix, point):
    return matrix[:3, :3] @ np.asarray(point, dtype=np.float64) + matrix[:3, 3]


def load_points(path):
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    flat = np.asarray(node["points"], dtype=np.float64)
    if flat.size % 3:
        raise ValueError(f"{path} points size is not divisible by 3")
    points = flat.reshape((-1, 3))
    feature_to_point = {
        int(item["feature_id"]): int(item["point_index"])
        for item in node.get("feature_id_to_point_index", []) or []
    }
    return points, feature_to_point


def load_manifest(path):
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            try:
                camera_index = int(float(row.get("camera_index", row.get("camera", ""))))
            except (TypeError, ValueError):
                continue
            rows[camera_index] = row
    return rows


def load_intrinsics(path):
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model_type = str(node.get("type", "")).strip()
    parameters = [float(value) for value in node.get("parameters", [])]
    if model_type in {"CentralOpenCV", "CentralOpenCVModel"}:
        parameters = (parameters + [0.0] * 12)[:12]
        model_type = "CentralOpenCVModel"
    elif model_type == "CentralThinPrismFisheyeModel":
        parameters = (parameters + [0.0] * 12)[:12]
    return {
        "type": model_type,
        "width": int(node.get("width", 0)),
        "height": int(node.get("height", 0)),
        "parameters": np.asarray(parameters, dtype=np.float64),
        "use_equidistant_projection": bool(node.get("use_equidistant_projection", True)),
    }


def find_intrinsics_path(state_dir, intrinsics_dir, camera_index):
    candidates = []
    if intrinsics_dir:
        candidates.append(Path(intrinsics_dir) / f"intrinsics{camera_index}.yaml")
    candidates.append(Path(state_dir) / f"intrinsics{camera_index}.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_intrinsics_set(state_dir, intrinsics_dir, camera_count):
    intrinsics = []
    for camera_index in range(camera_count):
        path = find_intrinsics_path(state_dir, intrinsics_dir, camera_index)
        if path.is_file():
            intrinsics.append(load_intrinsics(path))
        else:
            intrinsics.append(None)
    return intrinsics


def project_central_opencv(point, intrinsics):
    params = intrinsics["parameters"]
    x, y, z = [float(value) for value in point]
    if z <= 0:
        return "behind_camera", None
    nx = x / z
    ny = y / z
    x2 = nx * nx
    y2 = ny * ny
    xy = nx * ny
    r2 = x2 + y2
    r4 = r2 * r2
    r6 = r4 * r2
    fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2 = params
    denom = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    if abs(denom) <= 1e-12:
        return "invalid_projection", None
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / denom
    dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2)
    dy = 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2)
    pixel = np.asarray([
        fx * (nx * radial + dx) + cx,
        fy * (ny * radial + dy) + cy,
    ], dtype=np.float64)
    return projection_bounds_status(pixel, intrinsics), pixel


def project_central_thin_prism_fisheye(point, intrinsics):
    params = intrinsics["parameters"]
    x, y, z = [float(value) for value in point]
    if z <= 0:
        return "behind_camera", None
    undistorted_x = x / z
    undistorted_y = y / z
    radius = math.hypot(undistorted_x, undistorted_y)
    if intrinsics.get("use_equidistant_projection", True) and radius > 1e-6:
        theta_by_r = math.atan(radius) / radius
        fisheye_x = theta_by_r * undistorted_x
        fisheye_y = theta_by_r * undistorted_y
    else:
        fisheye_x = undistorted_x
        fisheye_y = undistorted_y

    x2 = fisheye_x * fisheye_x
    y2 = fisheye_y * fisheye_y
    xy = fisheye_x * fisheye_y
    r2 = x2 + y2
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r6 * r2
    fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1 = params
    radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8
    dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2) + sx1 * r2
    dy = 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2) + sy1 * r2
    pixel = np.asarray([
        fx * (fisheye_x + radial * fisheye_x + dx) + cx,
        fy * (fisheye_y + radial * fisheye_y + dy) + cy,
    ], dtype=np.float64)
    return projection_bounds_status(pixel, intrinsics), pixel


def projection_bounds_status(pixel, intrinsics):
    if not np.all(np.isfinite(pixel)):
        return "invalid_projection"
    width = int(intrinsics.get("width", 0))
    height = int(intrinsics.get("height", 0))
    if width > 0 and height > 0:
        if pixel[0] < 0 or pixel[1] < 0 or pixel[0] >= width or pixel[1] >= height:
            return "outside_image"
    return "ok"


def project_point(point, intrinsics):
    if intrinsics is None:
        return "missing_intrinsics", None
    model_type = intrinsics.get("type", "")
    if model_type == "CentralOpenCVModel":
        return project_central_opencv(point, intrinsics)
    if model_type == "CentralThinPrismFisheyeModel":
        return project_central_thin_prism_fisheye(point, intrinsics)
    return f"unsupported_model:{model_type or 'unknown'}", None


def format_float(value):
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.12g}"


def parse_frame_index(filename, fallback):
    match = IMAGE_ID_RE.search(str(filename))
    if match:
        return str(int(match.group(1)))
    return str(fallback)


def camera_label(camera_index, manifest_row):
    if manifest_row:
        for key in ("user_id", "camera_id", "stage_name"):
            value = manifest_row.get(key)
            if value:
                return value
    return f"camera{camera_index}"


def make_base_row(dataset_name, imageset, camera_index, output_camera_index, manifest_row, feature):
    row = {
        "dataset": dataset_name,
        "imageset_index": str(imageset["index"]),
        "frame_index": parse_frame_index(imageset["filename"], imageset["index"]),
        "camera_index": str(output_camera_index),
        "source_camera_index": str(camera_index),
        "camera_label": camera_label(camera_index, manifest_row),
        "stage_name": manifest_row.get("stage_name", "") if manifest_row else "",
        "machine": manifest_row.get("machine", "") if manifest_row else "",
        "camera_id": manifest_row.get("camera_id", "") if manifest_row else "",
        "user_id": manifest_row.get("user_id", "") if manifest_row else "",
        "filename": imageset["filename"],
        "feature_index": str(feature["feature_index"]),
        "feature_id": str(feature["feature_id"]),
        "point_index": "",
        "local_x": "",
        "local_y": "",
        "local_z": "",
        "state_x": "",
        "state_y": "",
        "state_z": "",
        "world_x": "",
        "world_y": "",
        "world_z": "",
        "camera_center_x": "",
        "camera_center_y": "",
        "camera_center_z": "",
        "observed_x": format_float(feature["x"]),
        "observed_y": format_float(feature["y"]),
        "projected_x": "",
        "projected_y": "",
        "residual_x_px": "",
        "residual_y_px": "",
        "residual_px": "",
        "projection_status": "",
    }
    return row


def export_rows(args):
    dataset = read_dataset(args.dataset)
    state_dir = Path(args.state_dir)
    image_used, rig_rotations, rig_translations = load_pose_file(state_dir / "rig_tr_global.yaml")
    _camera_used, camera_rotations, camera_translations = load_pose_file(state_dir / "camera_tr_rig.yaml")
    points, feature_to_point = load_points(state_dir / "points.yaml")
    intrinsics = load_intrinsics_set(state_dir, args.intrinsics_dir, dataset["camera_count"])
    manifest = load_manifest(args.manifest)
    reference_poses, reference_intrinsics = load_reference_cameras(args.reference_studio32_yaml)
    reference_tr_state, reference_summary = estimate_reference_tr_state(
        camera_rotations,
        camera_translations,
        reference_poses,
        args.camera_index_offset,
    )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    status_counts = {}
    residuals = []
    row_count = 0
    with args.output_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TSV_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for imageset in dataset["imagesets"]:
            imageset_index = int(imageset["index"])
            rig_pose_available = imageset_index < len(image_used)
            for camera_index, features in enumerate(imageset["features"]):
                manifest_row = manifest.get(camera_index, {})
                output_camera_index = camera_index + int(args.camera_index_offset)
                camera_center = None
                if camera_index < len(camera_rotations):
                    camera_tr_state = pose_matrix(camera_rotations[camera_index], camera_translations[camera_index])
                    state_tr_camera = invert_pose(camera_tr_state)
                    camera_center = transform_point(reference_tr_state, state_tr_camera[:3, 3])
                for feature in features:
                    row = make_base_row(
                        args.dataset_name,
                        imageset,
                        camera_index,
                        output_camera_index,
                        manifest_row,
                        feature,
                    )
                    status = ""
                    point_index = feature_to_point.get(int(feature["feature_id"]))
                    if point_index is None:
                        status = "missing_point"
                    elif point_index < 0 or point_index >= len(points):
                        status = "point_index_out_of_range"
                        row["point_index"] = str(point_index)
                    else:
                        local_point = points[point_index]
                        row["point_index"] = str(point_index)
                        row["local_x"] = format_float(local_point[0])
                        row["local_y"] = format_float(local_point[1])
                        row["local_z"] = format_float(local_point[2])
                        row["camera_center_x"] = format_float(camera_center[0] if camera_center is not None else None)
                        row["camera_center_y"] = format_float(camera_center[1] if camera_center is not None else None)
                        row["camera_center_z"] = format_float(camera_center[2] if camera_center is not None else None)
                        if not rig_pose_available:
                            status = "missing_image_pose"
                        elif not image_used[imageset_index]:
                            status = "unused_image"
                        elif camera_index >= len(camera_rotations):
                            status = "missing_camera_pose"
                        else:
                            rig_point = rig_rotations[imageset_index] @ local_point + rig_translations[imageset_index]
                            world_point = transform_point(reference_tr_state, rig_point)
                            row["state_x"] = format_float(rig_point[0])
                            row["state_y"] = format_float(rig_point[1])
                            row["state_z"] = format_float(rig_point[2])
                            row["world_x"] = format_float(world_point[0])
                            row["world_y"] = format_float(world_point[1])
                            row["world_z"] = format_float(world_point[2])
                            if args.project_with_reference_yaml:
                                reference_index = output_camera_index
                                reference_pose = reference_poses.get(reference_index)
                                projection_intrinsics = reference_intrinsics.get(reference_index)
                                if reference_pose is None:
                                    status = "missing_reference_camera_pose"
                                    projected = None
                                elif projection_intrinsics is None:
                                    status = "missing_reference_intrinsics"
                                    projected = None
                                else:
                                    camera_point = transform_point(reference_pose, world_point)
                                    status, projected = project_point(camera_point, projection_intrinsics)
                            else:
                                camera_point = camera_rotations[camera_index] @ rig_point + camera_translations[camera_index]
                                status, projected = project_point(camera_point, intrinsics[camera_index])
                            if projected is not None:
                                observed = np.asarray([feature["x"], feature["y"]], dtype=np.float64)
                                residual = projected - observed
                                residual_px = float(np.linalg.norm(residual))
                                row["projected_x"] = format_float(projected[0])
                                row["projected_y"] = format_float(projected[1])
                                row["residual_x_px"] = format_float(residual[0])
                                row["residual_y_px"] = format_float(residual[1])
                                row["residual_px"] = format_float(residual_px)
                                if status == "ok":
                                    residuals.append(residual_px)
                    row["projection_status"] = status
                    status_counts[status] = status_counts.get(status, 0) + 1
                    writer.writerow(row)
                    row_count += 1
    summary = {
        "dataset": args.dataset_name,
        "dataset_path": str(Path(args.dataset).resolve(strict=False)),
        "state_dir": str(state_dir.resolve(strict=False)),
        "output_tsv": str(args.output_tsv.resolve(strict=False)),
        "reference_studio32_yaml": str(Path(args.reference_studio32_yaml).resolve(strict=False)) if args.reference_studio32_yaml else "",
        "project_with_reference_yaml": bool(args.project_with_reference_yaml),
        "camera_index_offset": int(args.camera_index_offset),
        "row_count": row_count,
        "status_counts": status_counts,
        "ok_count": len(residuals),
        "median_residual_px": float(np.median(residuals)) if residuals else None,
        "p90_residual_px": float(np.percentile(residuals, 90)) if residuals else None,
        "max_residual_px": float(np.max(residuals)) if residuals else None,
    }
    summary.update(reference_summary)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export feature-level calibration correspondence reprojection residuals.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    parser.add_argument("--dataset-name", default="calibration")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--intrinsics-dir", type=Path)
    parser.add_argument("--camera-index-offset", type=int, default=0)
    parser.add_argument("--reference-studio32-yaml", type=Path)
    parser.add_argument(
        "--project-with-reference-yaml",
        action="store_true",
        help=(
            "Project with camera poses and intrinsics from --reference-studio32-yaml "
            "after aligning the local solver state into that reference frame. "
            "Without this flag, residuals are computed in the local solver state."
        ),
    )
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = export_rows(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
