#!/usr/bin/env python3
"""Generate an advanced correspondence viewer for studio calibration runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path

import numpy as np
import yaml

try:
    from studio_canonical_frame import (
        estimate_studio_canonical_frame,
        transform_point_to_aligned,
    )
except ModuleNotFoundError:
    from scripts.calib.studio_canonical_frame import (
        estimate_studio_canonical_frame,
        transform_point_to_aligned,
    )


def to_three(point):
    x, y, z = [float(v) for v in point]
    return [x, -y, -z]


def vector_to_three(vector):
    vector = np.asarray(vector, dtype=np.float64)
    mapped = np.asarray([vector[0], -vector[1], -vector[2]], dtype=np.float64)
    norm = np.linalg.norm(mapped)
    if norm > 0:
        mapped /= norm
    return [float(v) for v in mapped]


def finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def finite_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def quat_xyzw_to_matrix(qx, qy, qz, qw):
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        return np.eye(3, dtype=np.float64)
    q /= norm
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_matrix_from_node(node):
    pose = node.get("camera_tr_studio_rig", node)
    rotation = quat_xyzw_to_matrix(
        finite_float(pose.get("qx"), 0.0),
        finite_float(pose.get("qy"), 0.0),
        finite_float(pose.get("qz"), 0.0),
        finite_float(pose.get("qw"), 1.0),
    )
    translation = np.asarray([
        finite_float(pose.get("tx"), 0.0),
        finite_float(pose.get("ty"), 0.0),
        finite_float(pose.get("tz"), 0.0),
    ], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def invert_pose(matrix):
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv


def transform_point(matrix, point):
    point = np.asarray(point, dtype=np.float64)
    return matrix[:3, :3] @ point + matrix[:3, 3]


def load_studio32_cameras(path):
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cameras = []
    for camera in node.get("cameras", []):
        matrix = pose_matrix_from_node(camera)
        center = invert_pose(matrix)[:3, 3]
        index = finite_int(camera.get("index"), len(cameras))
        label = str(camera.get("label") or camera.get("camera_id") or index)
        group = str(camera.get("group") or ("inner" if label.startswith("inner") else "outer"))
        cameras.append({
            "index": index,
            "label": label,
            "camera_id": str(camera.get("camera_id") or label),
            "group": group,
            "center": center.tolist(),
            "center_three": to_three(center),
        })
    cameras.sort(key=lambda item: item["index"])
    return cameras


def load_studio32_coordinate_transform(path):
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    transform = node.get("coordinate_transform") or {}
    if not transform:
        return None
    required = ["origin_source", "aligned_from_source_rotation"]
    if any(key not in transform for key in required):
        return None
    try:
        origin = np.asarray(transform["origin_source"], dtype=np.float64)
        rotation = np.asarray(transform["aligned_from_source_rotation"], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if origin.shape != (3,) or rotation.shape != (3, 3):
        return None
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(rotation)):
        return None
    return transform


def estimate_viewer_up_alignment(cameras):
    label_to_center = {
        str(camera.get("label")): np.asarray(camera.get("center"), dtype=np.float64)
        for camera in cameras
    }
    frame = estimate_studio_canonical_frame(label_to_center)
    if frame is None:
        return {
            "method": "canonical_yaml_default_y_axis",
            "display_up_vector": vector_to_three([0.0, 1.0, 0.0]),
            "metric_up_vector": [0.0, 1.0, 0.0],
        }
    return {
        "method": frame["method"],
        "source": frame["source"],
        "display_up_vector": vector_to_three(frame["axes_source"]["y"]),
        "metric_up_vector": [float(v) for v in frame["axes_source"]["y"]],
        "level_plane_count": frame["level_plane_count"],
        "used_columns": frame["used_columns"],
    }


def apply_coordinate_transform(point, coordinate_transform):
    if not coordinate_transform:
        return [float(v) for v in point]
    return [float(v) for v in transform_point_to_aligned(point, coordinate_transform)]


def normalized_three_delta(origin, endpoint):
    delta = np.asarray(endpoint, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    three = np.asarray([delta[0], -delta[1], -delta[2]], dtype=np.float64)
    norm = np.linalg.norm(three)
    if norm <= 0:
        return None
    return [float(v) for v in three / norm]


def frame_face_pose_to_viewer_entry(frame_index, face_id, pose, coordinate_transform):
    origin = apply_coordinate_transform(transform_point(pose, [0.0, 0.0, 0.0]), coordinate_transform)
    axis_x_endpoint = apply_coordinate_transform(transform_point(pose, [1.0, 0.0, 0.0]), coordinate_transform)
    axis_y_endpoint = apply_coordinate_transform(transform_point(pose, [0.0, 1.0, 0.0]), coordinate_transform)
    axis_z_endpoint = apply_coordinate_transform(transform_point(pose, [0.0, 0.0, 1.0]), coordinate_transform)
    axis_x = normalized_three_delta(origin, axis_x_endpoint)
    axis_y = normalized_three_delta(origin, axis_y_endpoint)
    axis_z = normalized_three_delta(origin, axis_z_endpoint)
    if axis_x is None or axis_y is None or axis_z is None:
        return None
    return {
        "frame_index": int(frame_index),
        "face_id": int(face_id),
        "origin_world": [float(v) for v in origin],
        "origin_three": to_three(origin),
        "axis_x_three": axis_x,
        "axis_y_three": axis_y,
        "axis_z_three": axis_z,
    }


def estimate_rigid_transform(source_points, target_points):
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] == 0:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), {
            "method": "identity_no_pairs",
            "sample_count": 0,
            "rms_error_m": None,
            "max_error_m": None,
        }
    if source.shape[0] < 3:
        translation = np.mean(target - source, axis=0)
        aligned = source + translation
        residuals = np.linalg.norm(aligned - target, axis=1)
        return np.eye(3, dtype=np.float64), translation, {
            "method": "translation_only_fewer_than_3_pairs",
            "sample_count": int(source.shape[0]),
            "rms_error_m": float(np.sqrt(np.mean(residuals ** 2))),
            "max_error_m": float(np.max(residuals)),
        }
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    aligned = (rotation @ source.T).T + translation
    residuals = np.linalg.norm(aligned - target, axis=1)
    return rotation, translation, {
        "method": "rigid_kabsch_camera_center_alignment",
        "sample_count": int(source.shape[0]),
        "rms_error_m": float(np.sqrt(np.mean(residuals ** 2))),
        "max_error_m": float(np.max(residuals)),
        "singular_values": [float(value) for value in singular_values],
    }


def align_marker_observations_to_cameras(observations, cameras):
    if not observations or not cameras:
        return {
            "method": "not_available",
            "sample_count": 0,
            "rms_error_m": None,
            "max_error_m": None,
        }
    camera_by_index = {int(camera["index"]): camera for camera in cameras}
    camera_by_id = {str(camera.get("camera_id")): camera for camera in cameras}
    camera_by_label = {str(camera.get("label")): camera for camera in cameras}
    per_camera = {}
    for obs in observations:
        center = obs.get("camera_center")
        if not center:
            continue
        camera = (
            camera_by_id.get(str(obs.get("camera_id")))
            or camera_by_label.get(str(obs.get("camera_id")))
            or camera_by_index.get(finite_int(obs.get("camera_index")))
        )
        if not camera:
            continue
        key = int(camera["index"])
        per_camera.setdefault(key, {
            "source": [],
            "target": np.asarray(camera["center"], dtype=np.float64),
        })
        per_camera[key]["source"].append(np.asarray(center, dtype=np.float64))
    source_points = []
    target_points = []
    for key in sorted(per_camera):
        item = per_camera[key]
        source_points.append(np.mean(np.asarray(item["source"], dtype=np.float64), axis=0))
        target_points.append(item["target"])
    rotation, translation, stats = estimate_rigid_transform(source_points, target_points)
    for obs in observations:
        world = obs.get("world")
        center = obs.get("camera_center")
        camera = (
            camera_by_id.get(str(obs.get("camera_id")))
            or camera_by_label.get(str(obs.get("camera_id")))
            or camera_by_index.get(finite_int(obs.get("camera_index")))
        )
        if world:
            aligned_world = rotation @ np.asarray(world, dtype=np.float64) + translation
            obs["world"] = [float(value) for value in aligned_world]
            obs["three"] = to_three(aligned_world)
        if center:
            if camera:
                aligned_center = np.asarray(camera["center"], dtype=np.float64)
                obs["viewer_camera_label"] = str(camera.get("label") or obs.get("camera_id"))
            else:
                aligned_center = rotation @ np.asarray(center, dtype=np.float64) + translation
            obs["camera_center"] = [float(value) for value in aligned_center]
            obs["camera_center_three"] = to_three(aligned_center)
            if world:
                obs["line_three"] = [obs["camera_center_three"], obs["three"]]
    stats["source"] = "marker_correspondence_camera_centers_to_final_studio32_camera_centers"
    return stats


def load_pose_yaml_by_index(path):
    path = Path(path)
    if not path.is_file():
        return {}
    node = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    result = {}
    for pose in node.get("poses", []):
        index = finite_int(pose.get("index"), len(result))
        result[index] = pose_matrix_from_node(pose)
    return result


def load_frame_face_poses(path):
    path = Path(path)
    node = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pose_type = str(node.get("type") or "")
    invert_input = pose_type == "frame_face_tr_rig" or path.name == "frame_face_tr_rig.yaml"
    result = {}
    for pose in node.get("poses", []):
        frame_index = finite_int(pose.get("frame_index"))
        face_id = finite_int(pose.get("face_id"))
        if frame_index is None or face_id is None:
            continue
        matrix = pose_matrix_from_node(pose)
        result[(frame_index, face_id)] = invert_pose(matrix) if invert_input else matrix
    return result


def residual_color_value(residual_px):
    value = finite_float(residual_px, 0.0) or 0.0
    return max(0.0, min(1.0, math.log1p(value) / math.log1p(10.0)))


def load_outer_observations(path, frame_face_pose_yaml, cameras, coordinate_transform=None):
    path = Path(path)
    if not path.is_file():
        return {
            "observations": [],
            "frames": [],
            "camera_ids": [],
            "frame_face_poses": [],
            "summary": empty_summary(),
        }

    camera_by_index = {int(camera["index"]): camera for camera in cameras}
    camera_by_id = {camera["camera_id"]: camera for camera in cameras}
    frame_face_poses = load_frame_face_poses(frame_face_pose_yaml)
    observations = []
    residuals = []
    frames = set()
    camera_ids = set()
    used_frame_faces = set()
    missing_pose = 0

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            frame_index = finite_int(row.get("frame_index"))
            face_id = finite_int(row.get("face_id"))
            camera_index = finite_int(row.get("camera_index"))
            if frame_index is None or face_id is None or camera_index is None:
                continue
            local = [
                finite_float(row.get("local_x"), 0.0),
                finite_float(row.get("local_y"), 0.0),
                finite_float(row.get("local_z"), 0.0),
            ]
            pose = frame_face_poses.get((frame_index, face_id))
            if pose is None:
                missing_pose += 1
                world = local
            else:
                used_frame_faces.add((frame_index, face_id))
                world = transform_point(pose, local).tolist()
            world = apply_coordinate_transform(world, coordinate_transform)
            camera = camera_by_index.get(camera_index) or camera_by_id.get(str(row.get("camera_id")))
            camera_center = camera["center"] if camera else None
            residual_px = finite_float(row.get("residual_px"))
            if residual_px is not None:
                residuals.append(residual_px)
            camera_id = str(row.get("camera_id") or camera_index)
            frames.add(frame_index)
            camera_ids.add(camera_id)
            observations.append({
                "dataset": "whole",
                "frame_index": frame_index,
                "filename": str(row.get("filename") or ""),
                "camera_index": camera_index,
                "camera_id": camera_id,
                "feature_id": finite_int(row.get("feature_id")),
                "tag_id": finite_int(row.get("tag_id")),
                "corner_id": finite_int(row.get("corner_id")),
                "face_id": face_id,
                "local": [float(v) for v in local],
                "world": [float(v) for v in world],
                "three": to_three(world),
                "camera_center": camera_center,
                "camera_center_three": to_three(camera_center) if camera_center else None,
                "line_three": [to_three(camera_center), to_three(world)] if camera_center else None,
                "observed": [
                    finite_float(row.get("observed_x")),
                    finite_float(row.get("observed_y")),
                ],
                "projected": [
                    finite_float(row.get("projected_x")),
                    finite_float(row.get("projected_y")),
                ],
                "residual": [
                    finite_float(row.get("residual_x_px")),
                    finite_float(row.get("residual_y_px")),
                ],
                "residual_px": residual_px,
                "residual_color": residual_color_value(residual_px),
                "projection_status": str(row.get("projection_status") or ""),
            })

    frame_face_pose_entries = []
    for frame_index, face_id in sorted(used_frame_faces):
        pose = frame_face_poses.get((frame_index, face_id))
        if pose is None:
            continue
        entry = frame_face_pose_to_viewer_entry(frame_index, face_id, pose, coordinate_transform)
        if entry:
            frame_face_pose_entries.append(entry)

    summary = summarize_residuals(residuals, len(observations), missing_pose)
    summary["frame_face_pose_count"] = len(frame_face_pose_entries)
    return {
        "observations": observations,
        "frames": sorted(frames),
        "camera_ids": sorted(camera_ids),
        "frame_face_poses": frame_face_pose_entries,
        "summary": summary,
    }


def empty_summary():
    return {
        "observation_count": 0,
        "missing_pose_count": 0,
        "median_residual_px": None,
        "p90_residual_px": None,
        "max_residual_px": None,
    }


def summarize_residuals(residuals, observation_count, missing_pose_count=0):
    if not residuals:
        summary = empty_summary()
        summary["observation_count"] = int(observation_count)
        summary["missing_pose_count"] = int(missing_pose_count)
        return summary
    values = np.asarray(residuals, dtype=np.float64)
    return {
        "observation_count": int(observation_count),
        "missing_pose_count": int(missing_pose_count),
        "median_residual_px": float(np.median(values)),
        "p90_residual_px": float(np.percentile(values, 90)),
        "max_residual_px": float(np.max(values)),
    }


def load_points_yaml(path):
    path = Path(path)
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float64)
    node = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat = np.asarray(node.get("points", []), dtype=np.float64)
    if flat.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return flat.reshape((-1, 3))


def row_value(row, *names, default=None):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def load_pnp_dataset(name, pnp_dir, cameras, coordinate_transform=None):
    if not pnp_dir:
        return empty_pnp_dataset(name)
    pnp_dir = Path(pnp_dir)
    if not pnp_dir.is_dir():
        return empty_pnp_dataset(name)

    cameras_by_index = {int(camera["index"]): camera for camera in cameras}
    points = load_points_yaml(pnp_dir / "points.yaml")
    rig_poses = load_pose_yaml_by_index(pnp_dir / "rig_tr_global.yaml")
    pnp_views = pnp_dir / "pnp_views.tsv"
    views = []
    frames = set()
    camera_ids = set()

    if pnp_views.is_file():
        with pnp_views.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            for row in reader:
                camera_index = finite_int(row_value(row, "camera_index", "camera"))
                frame_index = finite_int(row_value(row, "imageset_index", "frame_index"))
                if camera_index is None or frame_index is None:
                    continue
                status = str(row.get("status") or "")
                if status.lower() == "failed":
                    continue
                board_pose = rig_poses.get(frame_index)
                board_center = board_pose[:3, 3].tolist() if board_pose is not None else None
                if board_center is not None:
                    board_center = apply_coordinate_transform(board_center, coordinate_transform)
                camera = cameras_by_index.get(camera_index)
                camera_center = camera["center"] if camera else None
                camera_id = camera["camera_id"] if camera else str(row.get("user_id") or camera_index)
                frames.add(frame_index)
                camera_ids.add(camera_id)
                views.append({
                    "kind": "per_view_pose_summary",
                    "dataset": name,
                    "frame_index": frame_index,
                    "camera_index": camera_index,
                    "camera_id": camera_id,
                    "filename": str(row.get("filename") or ""),
                    "status": status,
                    "points": finite_int(row_value(row, "points", "point_count")),
                    "inliers": finite_int(row_value(row, "inliers", "inlier_count")),
                    "mean_error_px": finite_float(row_value(row, "mean_error_px", "reprojection_rmse_px")),
                    "median_error_px": finite_float(row_value(row, "median_error_px")),
                    "board_center": board_center,
                    "board_center_three": to_three(board_center) if board_center else None,
                    "camera_center": camera_center,
                    "camera_center_three": to_three(camera_center) if camera_center else None,
                    "line_three": [to_three(camera_center), to_three(board_center)]
                    if camera_center and board_center else None,
                })

    return {
        "name": name,
        "kind": "board_pose_summary",
        "note": "Large/small markers show board pose summaries, not per-feature residuals.",
        "point_count": int(points.shape[0]),
        "sample_points_three": [to_three(point) for point in points[:512]],
        "view_count": len(views),
        "frames": sorted(frames),
        "camera_ids": sorted(camera_ids),
        "views": views,
    }


def load_marker_correspondences(name, path, max_rows, cameras=None):
    path = Path(path) if path else None
    if path is None or not path.is_file():
        return None

    ok_count = 0
    total_rows = 0
    frame_counts = {}
    residuals = []
    camera_ids = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            total_rows += 1
            if row.get("projection_status") != "ok":
                continue
            world = [
                finite_float(row.get("world_x")),
                finite_float(row.get("world_y")),
                finite_float(row.get("world_z")),
            ]
            center = [
                finite_float(row.get("camera_center_x")),
                finite_float(row.get("camera_center_y")),
                finite_float(row.get("camera_center_z")),
            ]
            if any(value is None for value in world + center):
                continue
            ok_count += 1
            frame = finite_int(row.get("imageset_index"))
            if frame is not None:
                frame_counts[frame] = frame_counts.get(frame, 0) + 1
            camera_ids.add(str(row.get("camera_label") or row.get("reference_camera_index") or row.get("camera_index")))
            residual_px = finite_float(row.get("residual_px"))
            if residual_px is not None:
                residuals.append(residual_px)

    stride = max(1, int(math.ceil(ok_count / max(1, max_rows)))) if max_rows else 1
    observations = []
    kept_ok_index = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            if row.get("projection_status") != "ok":
                continue
            world = [
                finite_float(row.get("world_x")),
                finite_float(row.get("world_y")),
                finite_float(row.get("world_z")),
            ]
            center = [
                finite_float(row.get("camera_center_x")),
                finite_float(row.get("camera_center_y")),
                finite_float(row.get("camera_center_z")),
            ]
            if any(value is None for value in world + center):
                continue
            if kept_ok_index % stride != 0:
                kept_ok_index += 1
                continue
            kept_ok_index += 1
            residual_px = finite_float(row.get("residual_px"))
            camera_id = str(row.get("camera_label") or row.get("reference_camera_index") or row.get("camera_index"))
            frame_index = finite_int(row.get("imageset_index"))
            observations.append({
                "kind": "feature_correspondence",
                "dataset": name,
                "frame_index": frame_index,
                "filename": str(row.get("filename") or ""),
                "camera_index": finite_int(row.get("reference_camera_index"), finite_int(row.get("camera_index"))),
                "camera_id": camera_id,
                "feature_id": finite_int(row.get("feature_id")),
                "point_index": finite_int(row.get("point_index")),
                "world": [float(v) for v in world],
                "three": to_three(world),
                "camera_center": [float(v) for v in center],
                "camera_center_three": to_three(center),
                "line_three": [to_three(center), to_three(world)],
                "observed": [
                    finite_float(row.get("observed_x")),
                    finite_float(row.get("observed_y")),
                ],
                "projected": [
                    finite_float(row.get("projected_x")),
                    finite_float(row.get("projected_y")),
                ],
                "residual": [
                    finite_float(row.get("residual_x_px")),
                    finite_float(row.get("residual_y_px")),
                ],
                "residual_px": residual_px,
                "residual_color": residual_color_value(residual_px),
                "projection_status": "ok",
            })

    alignment = align_marker_observations_to_cameras(observations, cameras or [])
    frames = sorted(frame_counts)
    top_frames = [
        {"frame_index": int(frame), "observation_count": int(count)}
        for frame, count in sorted(frame_counts.items(), key=lambda item: (-item[1], item[0]))[:64]
    ]
    summary = summarize_residuals(residuals, ok_count, 0)
    summary.update({
        "row_count": int(total_rows),
        "loaded_observation_count": int(len(observations)),
        "sampling_stride": int(stride),
        "source_tsv": str(path.resolve()),
    })
    view_keys = {
        (obs.get("frame_index"), obs.get("camera_id"))
        for obs in observations
        if obs.get("frame_index") is not None and obs.get("camera_id") not in (None, "")
    }
    return {
        "name": name,
        "kind": "feature_correspondence",
        "note": "Feature-level observed/projected/world correspondences exported from calibration residuals.",
        "point_count": int(len(observations)),
        "sample_points_three": [obs["three"] for obs in observations[:512]],
        "view_count": int(len(view_keys)),
        "frames": frames,
        "top_frames": top_frames,
        "camera_ids": sorted(camera_ids),
        "views": [],
        "observations": observations,
        "alignment": alignment,
        "summary": summary,
    }


def empty_pnp_dataset(name):
    return {
        "name": name,
        "kind": "board_pose_summary",
        "note": "Large/small markers show board pose summaries, not per-feature residuals.",
        "point_count": 0,
        "sample_points_three": [],
        "view_count": 0,
        "frames": [],
        "camera_ids": [],
        "views": [],
    }


def find_assets_dir(explicit_dir):
    if explicit_dir:
        return Path(explicit_dir)
    return None


def copy_assets(output_dir, assets_dir):
    assets_dir = find_assets_dir(assets_dir)
    if assets_dir is None:
        return
    for name in ("three.min.js", "OrbitControls.js", "TransformControls.js"):
        src = assets_dir / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)


def write_html(output_dir):
    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Advanced Studio Correspondence Viewer</title>
  <style>
    body { margin: 0; overflow: hidden; font-family: Inter, system-ui, sans-serif; background: #f7f7f3; color: #23262a; }
    #viewer { position: fixed; inset: 0; }
    #panel { position: fixed; top: 14px; left: 14px; width: 340px; max-height: calc(100vh - 28px); overflow: auto; background: rgba(255,255,255,.94); border: 1px solid #d7d7cf; border-radius: 8px; padding: 12px; box-shadow: 0 8px 28px rgba(20,20,20,.12); }
    h1 { margin: 0 0 8px; font-size: 18px; }
    label { display: block; font-size: 12px; color: #5b6068; margin-top: 9px; }
    select, input[type="range"], input[type="text"] { width: 100%; box-sizing: border-box; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .toggles { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; margin-top: 8px; font-size: 13px; }
    .stats { margin-top: 10px; font-size: 12px; line-height: 1.45; background: #eeeeea; border-radius: 6px; padding: 8px; }
    .note { color: #62666b; font-size: 12px; line-height: 1.4; }
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="panel">
    <h1>Correspondence Viewer</h1>
    <p class="note">Shows feature-level observed/projected/world correspondences when residual TSVs are available; marker datasets fall back to board pose summaries only when TSVs are missing.</p>
    <div class="row">
      <div><label>Dataset</label><select id="dataset"><option value="all">all</option><option value="whole">whole</option><option value="large">large</option><option value="small">small</option></select></div>
      <div><label>Frame</label><select id="frame"><option value="all">all</option></select></div>
    </div>
    <label>Camera filter</label><input id="cameraFilter" type="text" placeholder="all, 1-1, inner0">
    <label>Residual threshold px: <span id="thresholdLabel">10</span></label><input id="residualThreshold" type="range" min="0" max="20" step="0.1" value="10">
    <div class="toggles">
      <label><input id="showCameras" type="checkbox" checked> cameras</label>
      <label><input id="showPoints" type="checkbox" checked> points</label>
      <label><input id="showRays" type="checkbox" checked> rays</label>
      <label><input id="showLabels" type="checkbox"> labels</label>
    </div>
    <div class="stats" id="stats">loading...</div>
  </div>
  <script src="./three.min.js"></script>
  <script src="./OrbitControls.js"></script>
  <script>
let scene, camera, renderer, controls, data, root, grid;
const pointMeshes = [];
const lineMeshes = [];
const cameraMeshes = [];
const labelSprites = [];

function colorFromResidual(value) {
  const t = Math.max(0, Math.min(1, Math.log1p(value || 0) / Math.log1p(10)));
  return new THREE.Color().setHSL((1 - t) * 0.33, 0.9, 0.48);
}

function makeTextSprite(text) {
  const canvas = document.createElement('canvas');
  canvas.width = 256; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#202124';
  ctx.font = '24px sans-serif';
  ctx.fillText(text, 12, 40);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.28, 0.07, 1);
  return sprite;
}

function init() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf7f7f3);
  root = new THREE.Group();
  scene.add(root);
  camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 100);
  camera.position.set(2.8, 2.2, 5.0);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.getElementById('viewer').appendChild(renderer.domElement);
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const light = new THREE.DirectionalLight(0xffffff, 0.5);
  light.position.set(3, 5, 4);
  scene.add(light);
  grid = new THREE.GridHelper(6, 24, 0x999999, 0xd5d5d0);
  scene.add(grid);
  window.addEventListener('resize', onResize);
  document.getElementById('dataset').addEventListener('input', () => { updateFrameOptions(true); rebuild(); });
  for (const id of ['frame', 'cameraFilter', 'residualThreshold', 'showCameras', 'showPoints', 'showRays', 'showLabels']) {
    document.getElementById(id).addEventListener('input', rebuild);
  }
}

function clearObjects() {
  for (const arr of [pointMeshes, lineMeshes, cameraMeshes, labelSprites]) {
    for (const obj of arr) root.remove(obj);
    arr.length = 0;
  }
}

function applyWorldUp() {
  const up = data && data.viewer_options ? data.viewer_options.default_reference_up_vector_three : null;
  if (!up || up.length !== 3) return;
  const upVector = new THREE.Vector3(up[0], up[1], up[2]).normalize();
  if (upVector.lengthSq() === 0) return;
  camera.up.copy(upVector);
  if (grid) {
    grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), upVector);
  }
  controls.update();
}

function addCameraMarker(cam, showLabels) {
  const pos = new THREE.Vector3(...cam.center_three);
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 8), new THREE.MeshBasicMaterial({ color: cam.group === 'inner' ? 0x2b6cb0 : 0x2f855a }));
  mesh.position.copy(pos);
  root.add(mesh); cameraMeshes.push(mesh);
  if (showLabels) {
    const label = makeTextSprite(cam.label);
    label.position.copy(pos).add(new THREE.Vector3(0, 0.08, 0));
    root.add(label); labelSprites.push(label);
  }
}

function addPoint(point, color, size=0.018) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 8, 6), new THREE.MeshBasicMaterial({ color }));
  mesh.position.set(point[0], point[1], point[2]);
  root.add(mesh); pointMeshes.push(mesh);
}

function addLine(a, b, color, opacity=0.28) {
  const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(geom, mat);
  root.add(line); lineMeshes.push(line);
}

function frameLabel(datasetName, frame) {
  const ds = data.datasets[datasetName];
  const top = ds && ds.top_frames ? ds.top_frames.find(item => String(item.frame_index) === String(frame)) : null;
  return top ? `${frame} (${top.observation_count})` : String(frame);
}

function updateFrameOptions(useDefault=false) {
  const select = document.getElementById('frame');
  const datasetName = document.getElementById('dataset').value;
  const current = select.value;
  const frames = new Set();
  if (datasetName === 'all' || datasetName === 'whole') for (const f of data.outer.frames) frames.add(f);
  for (const name of ['large', 'small']) {
    if (datasetName === 'all' || datasetName === name) for (const f of data.datasets[name].frames) frames.add(f);
  }
  const sorted = [...frames].sort((a,b)=>a-b);
  select.innerHTML = '<option value="all">all</option>' + sorted.map(f => `<option value="${f}">${frameLabel(datasetName, f)}</option>`).join('');
  const defaultFrame = data.defaults && data.defaults.frame_by_dataset ? data.defaults.frame_by_dataset[datasetName] : null;
  if (useDefault && defaultFrame != null && [...select.options].some(o => o.value === String(defaultFrame))) {
    select.value = String(defaultFrame);
  } else if ([...select.options].some(o => o.value === current)) {
    select.value = current;
  }
}

function cameraMatches(id, filter) {
  if (!filter) return true;
  return String(id).toLowerCase().includes(filter.toLowerCase());
}

function rebuild() {
  if (!data) return;
  clearObjects();
  const datasetName = document.getElementById('dataset').value;
  const frameValue = document.getElementById('frame').value;
  const cameraFilter = document.getElementById('cameraFilter').value.trim();
  const threshold = parseFloat(document.getElementById('residualThreshold').value);
  document.getElementById('thresholdLabel').textContent = threshold.toFixed(1);
  const showCameras = document.getElementById('showCameras').checked;
  const showPoints = document.getElementById('showPoints').checked;
  const showRays = document.getElementById('showRays').checked;
  const showLabels = document.getElementById('showLabels').checked;
  if (showCameras) for (const cam of data.cameras) if (cameraMatches(cam.label, cameraFilter) || cameraMatches(cam.camera_id, cameraFilter)) addCameraMarker(cam, showLabels);
  let pointCount = 0, lineCount = 0;
  if (datasetName === 'all' || datasetName === 'whole') {
    for (const obs of data.outer.observations) {
      if (frameValue !== 'all' && String(obs.frame_index) !== frameValue) continue;
      if (!cameraMatches(obs.camera_id, cameraFilter)) continue;
      if ((obs.residual_px || 0) > threshold) continue;
      const color = colorFromResidual(obs.residual_px || 0);
      if (showPoints) { addPoint(obs.three, color, 0.016); pointCount++; }
      if (showRays && obs.line_three) { addLine(obs.line_three[0], obs.line_three[1], color, 0.22); lineCount++; }
    }
  }
  for (const name of ['large', 'small']) {
    if (!(datasetName === 'all' || datasetName === name)) continue;
    const ds = data.datasets[name];
    if (ds.observations && ds.observations.length) {
      for (const obs of ds.observations) {
        if (frameValue !== 'all' && String(obs.frame_index) !== frameValue) continue;
        if (!cameraMatches(obs.camera_id, cameraFilter)) continue;
        if ((obs.residual_px || 0) > threshold) continue;
        const color = colorFromResidual(obs.residual_px || 0);
        if (showPoints) { addPoint(obs.three, color, name === 'large' ? 0.014 : 0.012); pointCount++; }
        if (showRays && obs.line_three) { addLine(obs.line_three[0], obs.line_three[1], color, 0.16); lineCount++; }
      }
    } else {
      for (const view of ds.views) {
        if (frameValue !== 'all' && String(view.frame_index) !== frameValue) continue;
        if (!cameraMatches(view.camera_id, cameraFilter)) continue;
        if (showPoints && view.board_center_three) { addPoint(view.board_center_three, name === 'large' ? 0x805ad5 : 0xdd6b20, 0.025); pointCount++; }
        if (showRays && view.line_three) { addLine(view.line_three[0], view.line_three[1], name === 'large' ? 0x805ad5 : 0xdd6b20, 0.18); lineCount++; }
      }
    }
  }
  document.getElementById('stats').innerHTML =
    `<b>Displayed</b><br>points: ${pointCount}<br>rays: ${lineCount}<br>` +
    `<b>Outer residuals</b><br>obs: ${data.summary.outer.observation_count}, median: ${fmt(data.summary.outer.median_residual_px)} px, p90: ${fmt(data.summary.outer.p90_residual_px)} px<br>` +
    `<b>Marker correspondences</b><br>` +
    `large: ${markerSummary(data.summary.large)}<br>` +
    `small: ${markerSummary(data.summary.small)}`;
}

function fmt(v) { return v == null ? 'n/a' : Number(v).toFixed(3); }
function markerSummary(s) {
  if (!s) return 'n/a';
  if (s.kind === 'feature_correspondence') {
    return `${s.observation_count || 0} obs, loaded ${s.loaded_observation_count || 0}, median ${fmt(s.median_residual_px)} px, p90 ${fmt(s.p90_residual_px)} px`;
  }
  return `${s.view_count || 0} pose views`;
}
function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }

init();
fetch('./correspondence_data.json').then(r => r.json()).then(json => {
  data = json;
  applyWorldUp();
  document.getElementById('dataset').value = data.default_dataset || 'large';
  updateFrameOptions(true);
  rebuild();
  animate();
});
  </script>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def build_data(args):
    cameras = load_studio32_cameras(args.studio32_yaml)
    coordinate_transform = load_studio32_coordinate_transform(args.studio32_yaml)
    up_alignment = estimate_viewer_up_alignment(cameras)
    outer = load_outer_observations(
        args.outer_observation_residuals_tsv,
        args.outer_frame_face_pose_yaml,
        cameras,
        coordinate_transform,
    )
    large = (
        load_marker_correspondences(
            "large",
            args.large_correspondence_tsv,
            args.max_marker_correspondences_per_dataset,
            cameras,
        )
        or load_pnp_dataset("large", args.large_pnp_dir, cameras, coordinate_transform)
    )
    small = (
        load_marker_correspondences(
            "small",
            args.small_correspondence_tsv,
            args.max_marker_correspondences_per_dataset,
            cameras,
        )
        or load_pnp_dataset("small", args.small_pnp_dir, cameras, coordinate_transform)
    )
    frame_defaults = {}
    for name, dataset in (("large", large), ("small", small)):
        top_frames = dataset.get("top_frames") or []
        if top_frames:
            frame_defaults[name] = top_frames[0]["frame_index"]
    return {
        "title": "Advanced Studio Correspondence Viewer",
        "coordinate_note": "Studio/OpenCV coordinates are displayed as Three.js x,-y,-z.",
        "viewer_options": {
            "default_reference_up_vector_three": up_alignment["display_up_vector"],
            "up_alignment": up_alignment,
            "coordinate_transform": coordinate_transform or {},
        },
        "default_dataset": "large" if large.get("observations") else "whole",
        "defaults": {
            "frame_by_dataset": frame_defaults,
        },
        "cameras": cameras,
        "outer": outer,
        "datasets": {
            "large": large,
            "small": small,
        },
        "summary": {
            "camera_count": len(cameras),
            "outer": outer["summary"],
            "large": {
                "kind": large["kind"],
                "view_count": large["view_count"],
                "point_count": large["point_count"],
                **large.get("summary", {}),
            },
            "small": {
                "kind": small["kind"],
                "view_count": small["view_count"],
                "point_count": small["point_count"],
                **small.get("summary", {}),
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an advanced studio correspondence viewer.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--studio32-yaml", required=True, type=Path)
    parser.add_argument("--outer-observation-residuals-tsv", required=True, type=Path)
    parser.add_argument("--outer-frame-face-pose-yaml", required=True, type=Path)
    parser.add_argument("--large-correspondence-tsv", type=Path, default=None)
    parser.add_argument("--small-correspondence-tsv", type=Path, default=None)
    parser.add_argument("--large-pnp-dir", type=Path, default=None)
    parser.add_argument("--small-pnp-dir", type=Path, default=None)
    parser.add_argument("--viewer-assets-dir", type=Path, default=None)
    parser.add_argument("--max-marker-correspondences-per-dataset", type=int, default=30000)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_data(args)
    (args.output_dir / "correspondence_data.json").write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    copy_assets(args.output_dir, args.viewer_assets_dir)
    write_html(args.output_dir)
    print(args.output_dir / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
