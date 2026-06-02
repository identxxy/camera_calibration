#!/usr/bin/env python3
"""Canonical coordinate-frame helpers for the studio 24+8 rig."""

import math
import re

import numpy as np


OUTER_SIDE_RE = re.compile(r"^[1-8]-[123]$")
SIDE_LEVEL_RE = re.compile(r"^([1-8])-([123])$")


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm <= 0 or not np.all(np.isfinite(vector)):
        return None
    return vector / norm


def camera_center_from_camera_tr_rig(camera_tr_rig):
    rotation = camera_tr_rig[:3, :3]
    translation = camera_tr_rig[:3, 3]
    return -rotation.T @ translation


def fit_plane_normal(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return None
    centered = points - np.mean(points, axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        return None
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return normalize(vt[-1])


def side_level(label):
    match = SIDE_LEVEL_RE.match(str(label))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def collect_outer_side_centers(label_to_center):
    centers = {}
    for label, center in label_to_center.items():
        parsed = side_level(label)
        if parsed is None:
            continue
        side, level = parsed
        if side == 4:
            continue
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            continue
        centers[(side, level)] = center
    return centers


def estimate_studio_canonical_frame(label_to_center):
    """Estimate canonical studio axes from outer non-4 side camera centers.

    The published frame is:
      origin: average center of side-level *-2 cameras, excluding 4-2
      +Y: average normal of the three side-camera layers, oriented *-1 -> *-3
      -Z: level-2 gap direction between 3-2 and 5-2
      +X: right-handed completion, so X x Y = Z
    """
    centers = collect_outer_side_centers(label_to_center)
    origin_points = [centers[(side, 2)] for side in range(1, 9) if side != 4 and (side, 2) in centers]
    if len(origin_points) < 3:
        return None
    origin = np.mean(np.asarray(origin_points, dtype=np.float64), axis=0)

    column_vectors = []
    used_columns = []
    for side in range(1, 9):
        if side == 4:
            continue
        if (side, 1) in centers and (side, 3) in centers:
            direction = normalize(centers[(side, 3)] - centers[(side, 1)])
            if direction is not None:
                column_vectors.append(direction)
                used_columns.append(side)
    if not column_vectors:
        return None
    vertical_ref = normalize(np.mean(np.asarray(column_vectors, dtype=np.float64), axis=0))
    if vertical_ref is None:
        return None

    layer_entries = []
    layer_normals = []
    for level in (1, 2, 3):
        labels = []
        points = []
        for side in range(1, 9):
            if side == 4:
                continue
            key = (side, level)
            if key in centers:
                labels.append(f"{side}-{level}")
                points.append(centers[key])
        normal = fit_plane_normal(points)
        if normal is None:
            continue
        if float(np.dot(normal, vertical_ref)) < 0:
            normal = -normal
        layer_normals.append(normal)
        layer_entries.append({
            "level": level,
            "labels": labels,
            "normal": [float(v) for v in normal],
        })
    if not layer_normals:
        return None
    y_axis = normalize(np.mean(np.asarray(layer_normals, dtype=np.float64), axis=0))
    if y_axis is None:
        return None
    if float(np.dot(y_axis, vertical_ref)) < 0:
        y_axis = -y_axis

    if (3, 2) not in centers or (5, 2) not in centers:
        return None
    gap_midpoint = 0.5 * (centers[(3, 2)] + centers[(5, 2)])
    gap_direction = gap_midpoint - origin
    gap_direction = gap_direction - float(np.dot(gap_direction, y_axis)) * y_axis
    gap_direction = normalize(gap_direction)
    if gap_direction is None:
        return None

    z_seed = -gap_direction
    x_axis = normalize(np.cross(y_axis, z_seed))
    if x_axis is None:
        return None
    z_axis = normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    axes = np.vstack([x_axis, y_axis, z_axis])
    if abs(float(np.linalg.det(axes)) - 1.0) > 1e-3:
        return None

    return {
        "method": "outer_level_plane_mean_normal_origin_level2_gap4",
        "source": (
            "outer non-4 side cameras: origin is mean of *-2 centers, +Y is "
            "the mean normal of levels *-1/*-2/*-3 oriented *-1 -> *-3, "
            "and -Z points toward the missing 4-2 side gap"
        ),
        "source_coordinate_frame": "studio_rig_current",
        "aligned_coordinate_frame": "studio_rig_level2_gravity_aligned",
        "point_transform": "p_aligned = R_aligned_from_source @ (p_source - origin_source)",
        "origin_source": [float(v) for v in origin],
        "aligned_from_source_rotation": [[float(v) for v in row] for row in axes],
        "source_from_aligned_rotation": [[float(v) for v in row] for row in axes.T],
        "axes_source": {
            "x": [float(v) for v in x_axis],
            "y": [float(v) for v in y_axis],
            "z": [float(v) for v in z_axis],
        },
        "axis_meaning": {
            "x": "right-handed horizontal right axis",
            "y": "gravity direction, oriented from *-1 toward *-3",
            "z": "opposite of the missing 4-2 side gap; -Z points toward that gap",
        },
        "negative_z_gap_direction_source": [float(v) for v in gap_direction],
        "negative_z_gap_labels": ["3-2", "5-2"],
        "level_plane_count": len(layer_entries),
        "level_plane_normals_source": layer_entries,
        "used_columns": [str(side) for side in used_columns],
        "column_count": len(used_columns),
        "origin_level2_labels": [f"{side}-2" for side in range(1, 9) if side != 4 and (side, 2) in centers],
    }


def transform_point_to_aligned(point, frame):
    rotation = np.asarray(frame["aligned_from_source_rotation"], dtype=np.float64)
    origin = np.asarray(frame["origin_source"], dtype=np.float64)
    return rotation @ (np.asarray(point, dtype=np.float64) - origin)


def transform_vector_to_aligned(vector, frame):
    rotation = np.asarray(frame["aligned_from_source_rotation"], dtype=np.float64)
    return rotation @ np.asarray(vector, dtype=np.float64)


def transform_pose_to_aligned(camera_tr_source, frame):
    source_from_aligned = np.eye(4, dtype=np.float64)
    source_from_aligned[:3, :3] = np.asarray(frame["source_from_aligned_rotation"], dtype=np.float64)
    source_from_aligned[:3, 3] = np.asarray(frame["origin_source"], dtype=np.float64)
    return np.asarray(camera_tr_source, dtype=np.float64) @ source_from_aligned


def estimate_frame_from_camera_poses(poses, camera_rows):
    label_to_center = {}
    for row in camera_rows:
        label = str(row.get("label") or "")
        index = int(row["index"])
        if not OUTER_SIDE_RE.match(label):
            continue
        pose = poses[index]
        if pose is None:
            continue
        label_to_center[label] = camera_center_from_camera_tr_rig(pose)
    return estimate_studio_canonical_frame(label_to_center)


def yaw_pitch_roll_from_rotation(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return {
        "yaw_deg": math.degrees(yaw),
        "pitch_deg": math.degrees(pitch),
        "roll_deg": math.degrees(roll),
    }
