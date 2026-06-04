#!/usr/bin/env python3
"""Smoke refine outer-camera SE(3) deltas from sparse AprilTag tower observations."""

import argparse
import csv
import json
import math
import re
import struct
from pathlib import Path

import numpy as np


INTRINSICS_REFINE_MODES = (
    "fixed",
    "shared_fxfy",
    "per_camera_fxfy",
    "per_camera_fxfycxcy",
    "per_camera_opencv5",
)
PNP_POSE_AVERAGING_ERROR_FLOOR_PX = 0.25
PNP_POSE_AVERAGING_MODE = "robust_weighted_median_error"
BACKTRACKING_STEP_SCALES = (1.0,)


def read_exact(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise EOFError("Unexpected end of dataset file")
    return data


def read_u32(stream):
    return struct.unpack(">I", read_exact(stream, 4))[0]


def read_i32(stream):
    return struct.unpack(">i", read_exact(stream, 4))[0]


def read_f32(stream):
    return struct.unpack("<f", read_exact(stream, 4))[0]


def read_dataset(path):
    with Path(path).open("rb") as stream:
        header = read_exact(stream, 10)
        if header != b"calib_data":
            raise ValueError(f"Invalid dataset header: {path}")
        version = read_u32(stream)
        if version not in (0, 1):
            raise ValueError(f"Unsupported dataset version {version}: {path}")
        camera_count = read_u32(stream)
        image_sizes = [(read_u32(stream), read_u32(stream)) for _ in range(camera_count)]
        imagesets = []
        for _ in range(read_u32(stream)):
            name_len = read_u32(stream)
            filename = read_exact(stream, name_len).decode("utf-8")
            camera_features = []
            for _camera in range(camera_count):
                features = []
                for _feature in range(read_u32(stream)):
                    features.append((read_f32(stream), read_f32(stream), read_i32(stream)))
                camera_features.append(features)
            imagesets.append({"filename": filename, "features": camera_features})
        known_points = {}
        for _ in range(read_u32(stream)):
            _cell_length = read_f32(stream)
            for _item in range(read_u32(stream)):
                read_i32(stream)
                read_i32(stream)
                read_i32(stream)
            if version >= 1:
                for _item in range(read_u32(stream)):
                    feature_id = read_i32(stream)
                    known_points[feature_id] = np.asarray(
                        [read_f32(stream), read_f32(stream), read_f32(stream)],
                        dtype=np.float64)
    return {
        "camera_count": camera_count,
        "image_sizes": image_sizes,
        "imagesets": imagesets,
        "known_points": known_points,
    }


def read_manifest(path, camera_count):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)
    if len(rows) != camera_count:
        raise ValueError(f"Manifest rows {len(rows)} != camera count {camera_count}")
    return rows


def quat_xyzw_to_matrix(qx, qy, qz, qw):
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_wxyz_to_matrix(qw, qx, qy, qz):
    return quat_xyzw_to_matrix(qx, qy, qz, qw)


def matrix_to_quat_xyzw(rotation):
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q *= -1
    return q


def pose_matrix(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def invert_pose(matrix):
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv


def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)


def so3_exp(w):
    theta = float(np.linalg.norm(w))
    wx = skew(w)
    if theta < 1e-12:
        return np.eye(3) + wx
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + a * wx + b * (wx @ wx)


def so3_log(rotation):
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(float(value))
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return theta / (2.0 * math.sin(theta)) * np.asarray([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], dtype=np.float64)


def se3_exp(xi):
    w = np.asarray(xi[:3], dtype=np.float64)
    v = np.asarray(xi[3:6], dtype=np.float64)
    theta = float(np.linalg.norm(w))
    wx = skew(w)
    rotation = so3_exp(w)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * wx
    else:
        theta2 = theta * theta
        theta3 = theta2 * theta
        V = (
            np.eye(3)
            + (1.0 - math.cos(theta)) / theta2 * wx
            + (theta - math.sin(theta)) / theta3 * (wx @ wx)
        )
    return pose_matrix(rotation, V @ v)


def se3_log_approx(matrix):
    return np.concatenate([so3_log(matrix[:3, :3]), matrix[:3, 3]])


def average_rotations(rotations):
    if not rotations:
        return np.eye(3)
    acc = np.zeros((4, 4), dtype=np.float64)
    ref = None
    for rotation in rotations:
        qx, qy, qz, qw = matrix_to_quat_xyzw(rotation)
        q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
        if ref is None:
            ref = q
        elif np.dot(ref, q) < 0:
            q *= -1
        acc += np.outer(q, q)
    _values, vectors = np.linalg.eigh(acc)
    q = vectors[:, -1]
    if q[0] < 0:
        q *= -1
    return quat_wxyz_to_matrix(q[0], q[1], q[2], q[3])


def weighted_average_rotations(rotations, weights):
    if not rotations:
        return np.eye(3)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size != len(rotations):
        raise ValueError(f"Rotation count {len(rotations)} != weight count {weights.size}")
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(rotations), dtype=np.float64)
    ref_idx = int(np.argmax(weights))
    ref_qx, ref_qy, ref_qz, ref_qw = matrix_to_quat_xyzw(rotations[ref_idx])
    ref = np.asarray([ref_qw, ref_qx, ref_qy, ref_qz], dtype=np.float64)
    acc = np.zeros((4, 4), dtype=np.float64)
    for rotation, weight in zip(rotations, weights):
        qx, qy, qz, qw = matrix_to_quat_xyzw(rotation)
        q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
        if np.dot(ref, q) < 0:
            q *= -1
        acc += float(weight) * np.outer(q, q)
    _values, vectors = np.linalg.eigh(acc)
    q = vectors[:, -1]
    if q[0] < 0:
        q *= -1
    return quat_wxyz_to_matrix(q[0], q[1], q[2], q[3])


def average_poses(poses):
    return pose_matrix(
        average_rotations([p[:3, :3] for p in poses]),
        np.asarray([p[:3, 3] for p in poses]).mean(axis=0))


def median_error_pose_weights(median_errors_px, error_floor_px=PNP_POSE_AVERAGING_ERROR_FLOOR_PX):
    weights = []
    for error in median_errors_px:
        if not math.isfinite(float(error)) or float(error) < 0.0:
            weights.append(0.0)
            continue
        clamped = max(float(error), error_floor_px)
        weights.append(1.0 / (clamped * clamped))
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return np.ones(len(weights), dtype=np.float64) / max(len(weights), 1)
    return weights / total


def robust_weighted_average_poses(poses, median_errors_px):
    if not poses:
        return np.eye(4, dtype=np.float64)
    if len(poses) == 1:
        return poses[0].copy()
    weights = median_error_pose_weights(median_errors_px)
    rotations = [p[:3, :3] for p in poses]
    translations = np.asarray([p[:3, 3] for p in poses], dtype=np.float64)
    return pose_matrix(
        weighted_average_rotations(rotations, weights),
        np.average(translations, axis=0, weights=weights))


def umeyama_similarity(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    variance = np.sum(source_centered ** 2) / source.shape[0]
    scale = float(np.sum(singular_values * np.diag(sign)) / variance)
    translation = target_mean - scale * rotation @ source_mean
    residuals = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
    return scale, rotation, translation, residuals


def rigid_transform(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be Nx3 arrays")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, _singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    translation = target_mean - rotation @ source_mean
    residuals = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
    return pose_matrix(rotation, translation), residuals


def parse_colmap_label(name):
    match = re.search(r"cam\d+_([^_]+)_f\d+", name)
    return match.group(1) if match else Path(name).stem


def load_colmap_images(path):
    images = {}
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        point_line = lines[i].strip() if i < len(lines) else ""
        i += 1
        point_ids = point_line.split()[2::3]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_tr_world = pose_matrix(quat_wxyz_to_matrix(qw, qx, qy, qz), [tx, ty, tz])
        label = parse_colmap_label(parts[9])
        images[label] = {
            "camera_tr_world": camera_tr_world,
            "world_tr_camera": invert_pose(camera_tr_world),
            "center_world": invert_pose(camera_tr_world)[:3, 3],
            "triangulated_point_count": sum(1 for p in point_ids if p != "-1"),
        }
    return images


def parse_label_pose_indices(text):
    result = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        label, index = item.split(":", 1)
        result[label.strip()] = int(index)
    return result


def load_bridge_centers(path, labels, label_to_index):
    poses = load_pose_yaml(path)
    centers = {}
    for label in labels:
        if label not in label_to_index:
            raise ValueError(f"Missing anchor label mapping for {label}")
        index = label_to_index[label]
        if index >= len(poses):
            raise ValueError(f"Anchor label {label} maps to pose {index}, but {path} only has {len(poses)} poses")
        centers[label] = invert_pose(poses[index])[:3, 3]
    return centers


def load_pose_yaml(path):
    pose_count = None
    current = None
    nodes = {}

    def flush():
        if current is None:
            return
        idx = int(current["index"])
        nodes[idx] = pose_matrix(
            quat_xyzw_to_matrix(
                float(current["qx"]),
                float(current["qy"]),
                float(current["qz"]),
                float(current["qw"])),
            [float(current["tx"]), float(current["ty"]), float(current["tz"])])

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pose_count:"):
            pose_count = int(line.split(":", 1)[1].strip())
        elif line.startswith("- index:"):
            flush()
            current = {"index": line.split(":", 1)[1].strip()}
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    flush()
    if pose_count is None:
        raise ValueError(f"Could not parse pose_count from {path}")
    return [nodes.get(i) for i in range(pose_count)]


def write_pose_yaml(path, poses):
    lines = [
        "# Each pose gives the B_tr_A transformation (i.e., A to B with right-multiplication), where the spaces A and B are defined by the filename. Quaternions are written as used by the Eigen library.",
        f"pose_count: {len(poses)}",
        "poses:",
    ]
    for index, pose in enumerate(poses):
        if pose is None:
            continue
        qx, qy, qz, qw = matrix_to_quat_xyzw(pose[:3, :3])
        tx, ty, tz = pose[:3, 3]
        lines.extend([
            f"  - index: {index}",
            f"    tx: {tx:.14g}",
            f"    ty: {ty:.14g}",
            f"    tz: {tz:.14g}",
            f"    qx: {qx:.14g}",
            f"    qy: {qy:.14g}",
            f"    qz: {qz:.14g}",
            f"    qw: {qw:.14g}",
        ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_intrinsics(path, image_size, mode):
    width, height = image_size
    if mode == "colmap_fixed" or not path or not Path(path).exists():
        return {
            "width": width,
            "height": height,
            "params": [4915.2, 4915.2, width * 0.5, height * 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"parameters\s*:\s*\[([^\]]+)\]", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not parse CentralOpenCV parameters from {path}")
    params = [float(x) for x in re.split(r"[,\s]+", match.group(1).strip()) if x]
    width_match = re.search(r"width\s*:\s*(\d+)", text)
    height_match = re.search(r"height\s*:\s*(\d+)", text)
    return {
        "width": int(width_match.group(1)) if width_match else width,
        "height": int(height_match.group(1)) if height_match else height,
        "params": (params + [0.0] * 12)[:12],
    }


def collect_intrinsics(intrinsics_dir, manifest, image_sizes, mode):
    intrinsics = []
    for idx, row in enumerate(manifest):
        path = None
        if intrinsics_dir:
            root = Path(intrinsics_dir)
            candidates = [
                root / f"intrinsics{idx}_{row['camera_id']}.yaml",
                root / f"intrinsics{idx}.yaml",
            ]
            candidates.extend(sorted(root.glob(f"intrinsics{idx}_*.yaml")))
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
        intrinsics.append(read_intrinsics(path, image_sizes[idx], mode))
    return intrinsics


def copy_intrinsic(intrinsic):
    return {
        "width": int(intrinsic["width"]),
        "height": int(intrinsic["height"]),
        "params": list(intrinsic["params"]),
    }


def copy_intrinsics(intrinsics):
    return [copy_intrinsic(intrinsic) for intrinsic in intrinsics]


def write_intrinsics_yaml(path, intrinsic):
    params = (list(intrinsic["params"]) + [0.0] * 12)[:12]
    text = "\n".join([
        "type: CentralOpenCVModel",
        f"width: {int(intrinsic['width'])}",
        f"height: {int(intrinsic['height'])}",
        "parameters: [" + ", ".join(f"{float(value):.14g}" for value in params) + "]",
        "",
    ])
    Path(path).write_text(text, encoding="utf-8")


def write_intrinsics_dir(path, manifest, intrinsics):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(manifest):
        write_intrinsics_yaml(path / f"intrinsics{idx}_{row['camera_id']}.yaml", intrinsics[idx])


def intrinsics_delta_from_values(values):
    return np.asarray(values, dtype=np.float64)


def intrinsics_delta_dimension(mode):
    if mode == "fixed":
        return 0
    if mode in ("shared_fxfy", "per_camera_fxfy"):
        return 2
    if mode == "per_camera_fxfycxcy":
        return 4
    if mode == "per_camera_opencv5":
        return 9
    raise ValueError(f"Unsupported intrinsics refine mode: {mode}")


def clamp_intrinsics_delta_to_total_bounds(
        delta,
        max_total_focal_delta_frac=0.0,
        max_total_principal_delta_px=0.0,
        max_total_distortion_delta=0.0):
    result = np.asarray(delta, dtype=np.float64).copy()
    if result.size >= 2 and max_total_focal_delta_frac > 0.0:
        limit = float(max_total_focal_delta_frac)
        focal_lo = math.log(max(1e-9, 1.0 - limit))
        focal_hi = math.log(1.0 + limit)
        result[:2] = np.clip(result[:2], focal_lo, focal_hi)
    if result.size >= 4 and max_total_principal_delta_px > 0.0:
        limit = float(max_total_principal_delta_px)
        result[2:4] = np.clip(result[2:4], -limit, limit)
    if result.size >= 9 and max_total_distortion_delta > 0.0:
        limit = float(max_total_distortion_delta)
        result[4:9] = np.clip(result[4:9], -limit, limit)
    return result


def apply_intrinsic_delta(intrinsic, delta, include_principal=False, include_distortion=False):
    result = copy_intrinsic(intrinsic)
    params = (list(result["params"]) + [0.0] * 12)[:12]
    delta = np.asarray(delta, dtype=np.float64)
    params[0] = max(1e-9, params[0] * math.exp(float(delta[0])))
    params[1] = max(1e-9, params[1] * math.exp(float(delta[1])))
    if include_principal:
        params[2] += float(delta[2])
        params[3] += float(delta[3])
    if include_distortion:
        # Delta order matches OpenCV's five-coefficient model:
        # k1, k2, p1, p2, k3. The repo stores k3 at index 6 and p1/p2 at 10/11.
        params[4] += float(delta[4])
        params[5] += float(delta[5])
        params[10] += float(delta[6])
        params[11] += float(delta[7])
        params[6] += float(delta[8])
    result["params"] = params
    return result


def apply_intrinsics_refinement(base_intrinsics, mode, shared_delta, per_camera_deltas):
    if mode == "fixed":
        return copy_intrinsics(base_intrinsics)
    if mode == "shared_fxfy":
        delta = np.zeros(2, dtype=np.float64) if shared_delta is None else np.asarray(shared_delta, dtype=np.float64)
        return [apply_intrinsic_delta(intrinsic, delta, include_principal=False) for intrinsic in base_intrinsics]
    dim = intrinsics_delta_dimension(mode)
    include_principal = mode in ("per_camera_fxfycxcy", "per_camera_opencv5")
    include_distortion = mode == "per_camera_opencv5"
    refined = []
    for idx, intrinsic in enumerate(base_intrinsics):
        if per_camera_deltas is None or idx >= len(per_camera_deltas):
            delta = np.zeros(dim, dtype=np.float64)
        else:
            delta = np.asarray(per_camera_deltas[idx], dtype=np.float64)
        refined.append(apply_intrinsic_delta(
            intrinsic,
            delta,
            include_principal=include_principal,
            include_distortion=include_distortion))
    return refined


def intrinsic_delta_stats(base_intrinsic, refined_intrinsic):
    base = (list(base_intrinsic["params"]) + [0.0] * 12)[:12]
    refined = (list(refined_intrinsic["params"]) + [0.0] * 12)[:12]
    fx_delta_frac = (refined[0] - base[0]) / max(abs(base[0]), 1e-12)
    fy_delta_frac = (refined[1] - base[1]) / max(abs(base[1]), 1e-12)
    cx_delta_px = refined[2] - base[2]
    cy_delta_px = refined[3] - base[3]
    k1_delta = refined[4] - base[4]
    k2_delta = refined[5] - base[5]
    k3_delta = refined[6] - base[6]
    p1_delta = refined[10] - base[10]
    p2_delta = refined[11] - base[11]
    distortion_deltas = [k1_delta, k2_delta, p1_delta, p2_delta, k3_delta]
    return {
        "fx_delta_frac": float(fx_delta_frac),
        "fy_delta_frac": float(fy_delta_frac),
        "max_abs_focal_delta_frac": float(max(abs(fx_delta_frac), abs(fy_delta_frac))),
        "cx_delta_px": float(cx_delta_px),
        "cy_delta_px": float(cy_delta_px),
        "principal_delta_px": float(np.linalg.norm([cx_delta_px, cy_delta_px])),
        "k1_delta": float(k1_delta),
        "k2_delta": float(k2_delta),
        "k3_delta": float(k3_delta),
        "p1_delta": float(p1_delta),
        "p2_delta": float(p2_delta),
        "max_abs_distortion_delta": float(max(abs(value) for value in distortion_deltas)),
    }


def format_float(value):
    return f"{float(value):.8g}"


def accepted_refined_intrinsics(manifest, base_intrinsics, refined_intrinsics, accepted_pose_camera_ids, args, mode):
    accepted_pose_camera_ids = set(accepted_pose_camera_ids)
    accepted = []
    accepted_ids = []
    rows = []
    for idx, row in enumerate(manifest):
        camera_id = row["camera_id"]
        stats = intrinsic_delta_stats(base_intrinsics[idx], refined_intrinsics[idx])
        if mode == "fixed":
            keep_refined = False
            decision = "fixed_prior"
            reason = "intrinsics_refine_mode_fixed"
        elif camera_id not in accepted_pose_camera_ids:
            keep_refined = False
            decision = "pose_prior_intrinsics_prior"
            reason = "pose_not_accepted"
        else:
            focal_limit = args.accept_camera_max_intrinsic_focal_delta_frac
            principal_limit = args.accept_camera_max_intrinsic_principal_delta_px
            distortion_limit = args.accept_camera_max_intrinsic_distortion_delta
            focal_ok = focal_limit <= 0 or stats["max_abs_focal_delta_frac"] <= focal_limit
            principal_ok = principal_limit <= 0 or stats["principal_delta_px"] <= principal_limit
            distortion_ok = (
                mode != "per_camera_opencv5"
                or distortion_limit <= 0
                or stats["max_abs_distortion_delta"] <= distortion_limit
            )
            keep_refined = focal_ok and principal_ok and distortion_ok
            decision = "accepted_refined" if keep_refined else "rejected_to_prior"
            reason = "passes_acceptance_gate" if keep_refined else "intrinsic_delta_exceeds_acceptance_gate"
        accepted.append(refined_intrinsics[idx] if keep_refined else base_intrinsics[idx])
        if keep_refined:
            accepted_ids.append(camera_id)
        base = (list(base_intrinsics[idx]["params"]) + [0.0] * 12)[:12]
        refined = (list(refined_intrinsics[idx]["params"]) + [0.0] * 12)[:12]
        output = (list(accepted[-1]["params"]) + [0.0] * 12)[:12]
        rows.append({
            "camera_index": idx,
            "camera_id": camera_id,
            "decision": decision,
            "output_intrinsics": "refined" if keep_refined else "prior",
            "reason": reason,
            "base_fx": format_float(base[0]),
            "base_fy": format_float(base[1]),
            "base_cx": format_float(base[2]),
            "base_cy": format_float(base[3]),
            "base_k1": format_float(base[4]),
            "base_k2": format_float(base[5]),
            "base_k3": format_float(base[6]),
            "base_p1": format_float(base[10]),
            "base_p2": format_float(base[11]),
            "refined_fx": format_float(refined[0]),
            "refined_fy": format_float(refined[1]),
            "refined_cx": format_float(refined[2]),
            "refined_cy": format_float(refined[3]),
            "refined_k1": format_float(refined[4]),
            "refined_k2": format_float(refined[5]),
            "refined_k3": format_float(refined[6]),
            "refined_p1": format_float(refined[10]),
            "refined_p2": format_float(refined[11]),
            "output_fx": format_float(output[0]),
            "output_fy": format_float(output[1]),
            "output_cx": format_float(output[2]),
            "output_cy": format_float(output[3]),
            "output_k1": format_float(output[4]),
            "output_k2": format_float(output[5]),
            "output_k3": format_float(output[6]),
            "output_p1": format_float(output[10]),
            "output_p2": format_float(output[11]),
            "fx_delta_frac": format_float(stats["fx_delta_frac"]),
            "fy_delta_frac": format_float(stats["fy_delta_frac"]),
            "max_abs_focal_delta_frac": format_float(stats["max_abs_focal_delta_frac"]),
            "cx_delta_px": format_float(stats["cx_delta_px"]),
            "cy_delta_px": format_float(stats["cy_delta_px"]),
            "principal_delta_px": format_float(stats["principal_delta_px"]),
            "k1_delta": format_float(stats["k1_delta"]),
            "k2_delta": format_float(stats["k2_delta"]),
            "k3_delta": format_float(stats["k3_delta"]),
            "p1_delta": format_float(stats["p1_delta"]),
            "p2_delta": format_float(stats["p2_delta"]),
            "max_abs_distortion_delta": format_float(stats["max_abs_distortion_delta"]),
        })
    return accepted, accepted_ids, rows


def summarize_intrinsics_acceptance(rows):
    def parse(row, key):
        try:
            return abs(float(row[key]))
        except (KeyError, TypeError, ValueError):
            return 0.0

    accepted = [row["camera_id"] for row in rows if row["decision"] == "accepted_refined"]
    prior = [row["camera_id"] for row in rows if row["output_intrinsics"] == "prior"]
    return {
        "accepted_refined": accepted,
        "accepted_refined_count": len(accepted),
        "output_prior_intrinsics": prior,
        "output_prior_intrinsics_count": len(prior),
        "max_abs_focal_delta_frac": max((parse(row, "max_abs_focal_delta_frac") for row in rows), default=0.0),
        "max_principal_delta_px": max((parse(row, "principal_delta_px") for row in rows), default=0.0),
        "max_abs_distortion_delta": max((parse(row, "max_abs_distortion_delta") for row in rows), default=0.0),
    }


def project_point(point_camera, intrinsic):
    x, y, z = point_camera
    if z <= 1e-9:
        return None
    p = intrinsic["params"]
    xn = x / z
    yn = y / z
    r2 = xn * xn + yn * yn
    r4 = r2 * r2
    r6 = r4 * r2
    radial_num = 1.0 + p[4] * r2 + p[5] * r4 + p[6] * r6
    radial_den = 1.0 + p[7] * r2 + p[8] * r4 + p[9] * r6
    radial = radial_num / radial_den if abs(radial_den) > 1e-12 else radial_num
    p1 = p[10]
    p2 = p[11]
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return np.asarray([p[0] * xd + p[2], p[1] * yd + p[3]], dtype=np.float64)


def load_pnp_views(path, max_error, imagesets=None, camera_count=None):
    views = []
    with Path(path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            image_index = int(row["imageset_index"])
            if imagesets is not None:
                if image_index < 0 or image_index >= len(imagesets):
                    raise ValueError(
                        f"PnP view imageset_index {image_index} is outside dataset range 0..{len(imagesets) - 1}")
                filename = row.get("filename", "")
                if filename and filename != imagesets[image_index]["filename"]:
                    raise ValueError(
                        "PnP view filename does not match dataset imageset: "
                        f"index {image_index}, pnp={filename}, dataset={imagesets[image_index]['filename']}")
            camera_index = int(row["camera_index"])
            if camera_count is not None and (camera_index < 0 or camera_index >= camera_count):
                raise ValueError(
                    f"PnP view camera_index {camera_index} is outside dataset range 0..{camera_count - 1}")
            if row["status"] != "solved":
                continue
            try:
                median_error = float(row["median_error_px"])
            except ValueError:
                continue
            if median_error > max_error:
                continue
            views.append({
                "camera_index": camera_index,
                "imageset_index": image_index,
                "median_error_px": median_error,
                "camera_tr_tower": pose_matrix(
                    quat_xyzw_to_matrix(
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                        float(row["qw"])),
                    [float(row["tx"]), float(row["ty"]), float(row["tz"])]),
            })
    return views


def build_colmap_prior(manifest, colmap_images, bridge_pose_yaml, anchor_labels, label_to_index):
    labels = [row["camera_id"] for row in manifest]
    missing = [label for label in labels if label not in colmap_images]
    if missing:
        raise ValueError(f"COLMAP prior missing labels: {missing}")
    bridge_centers = load_bridge_centers(bridge_pose_yaml, anchor_labels, label_to_index)
    source = np.asarray([colmap_images[label]["center_world"] for label in anchor_labels])
    target = np.asarray([bridge_centers[label] for label in anchor_labels])
    scale, rotation, translation, residuals = umeyama_similarity(source, target)
    poses = []
    for label in labels:
        world_tr_camera = colmap_images[label]["world_tr_camera"]
        rig_R_camera = rotation @ world_tr_camera[:3, :3]
        rig_center = scale * rotation @ world_tr_camera[:3, 3] + translation
        rig_tr_camera = pose_matrix(rig_R_camera, rig_center)
        poses.append(invert_pose(rig_tr_camera))
    return poses, {
        "source": "bridge_topdown",
        "anchor_labels": anchor_labels,
        "scale": scale,
        "anchor_residuals_m": residuals.tolist(),
        "anchor_rms_m": float(np.sqrt(np.mean(residuals ** 2))),
        "anchor_max_m": float(np.max(residuals)),
    }


def load_pnp_anchor_filter(path, max_median_error, min_solved_views):
    if not path:
        return None
    result = {}
    with Path(path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                median_error = float(row["median_view_error_px"])
            except (ValueError, KeyError):
                median_error = float("inf")
            solved_views = int(row.get("solved_views", "0") or 0)
            result[int(row["camera_index"])] = (
                row.get("connected") == "yes"
                and solved_views >= min_solved_views
                and median_error <= max_median_error
            )
    return result


def build_colmap_prior_from_alignment_poses(
        manifest,
        colmap_images,
        alignment_pose_yaml,
        pnp_summary_tsv,
        max_anchor_median_error,
        min_anchor_solved_views):
    labels = [row["camera_id"] for row in manifest]
    pnp_filter = load_pnp_anchor_filter(
        pnp_summary_tsv,
        max_anchor_median_error,
        min_anchor_solved_views)
    alignment_poses = load_pose_yaml(alignment_pose_yaml)
    source = []
    target = []
    anchor_labels = []
    for idx, label in enumerate(labels):
        if label not in colmap_images:
            raise ValueError(f"COLMAP prior missing label: {label}")
        if idx >= len(alignment_poses) or alignment_poses[idx] is None:
            continue
        if pnp_filter is not None and not pnp_filter.get(idx, False):
            continue
        source.append(colmap_images[label]["center_world"])
        target.append(invert_pose(alignment_poses[idx])[:3, 3])
        anchor_labels.append(label)
    if len(anchor_labels) < 3:
        raise ValueError(f"Need at least 3 alignment anchors, got {anchor_labels}")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale, rotation, translation, residuals = umeyama_similarity(source, target)
    poses = []
    for label in labels:
        world_tr_camera = colmap_images[label]["world_tr_camera"]
        rig_R_camera = rotation @ world_tr_camera[:3, :3]
        rig_center = scale * rotation @ world_tr_camera[:3, 3] + translation
        rig_tr_camera = pose_matrix(rig_R_camera, rig_center)
        poses.append(invert_pose(rig_tr_camera))
    return poses, {
        "source": "tower_pnp_rig",
        "anchor_labels": anchor_labels,
        "scale": scale,
        "anchor_residuals_m": residuals.tolist(),
        "anchor_rms_m": float(np.sqrt(np.mean(residuals ** 2))),
        "anchor_max_m": float(np.max(residuals)),
        "max_anchor_median_error_px": max_anchor_median_error,
        "min_anchor_solved_views": min_anchor_solved_views,
    }


def build_pose_yaml_prior(manifest, pose_yaml):
    poses = load_pose_yaml(pose_yaml)
    if len(poses) < len(manifest):
        raise ValueError(f"Pose prior has {len(poses)} poses but manifest has {len(manifest)} cameras")
    missing = [
        row["camera_id"]
        for idx, row in enumerate(manifest)
        if idx >= len(poses) or poses[idx] is None
    ]
    if missing:
        raise ValueError(f"Pose prior missing cameras: {missing}")
    return poses[:len(manifest)], {
        "source": "camera_prior_pose_yaml",
        "pose_yaml": str(Path(pose_yaml).resolve()),
    }


def pose_center(pose):
    return invert_pose(pose)[:3, 3]


def pose_rotation_delta_deg(a, b):
    value = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(value)))


def apply_bridge_prior_overrides(camera_priors, manifest, bridge_pose_yaml, labels, label_to_index):
    labels = [label.strip() for label in str(labels or "").split(",") if label.strip()]
    if not labels:
        return [], camera_priors
    bridge_poses = load_pose_yaml(bridge_pose_yaml)
    manifest_index_by_label = {row["camera_id"]: idx for idx, row in enumerate(manifest)}
    updated = list(camera_priors)
    rows = []
    for label in labels:
        if label not in manifest_index_by_label:
            raise ValueError(f"Bridge prior override label {label} is not in the outer manifest")
        if label not in label_to_index:
            raise ValueError(f"Bridge prior override label {label} has no bridge pose index mapping")
        manifest_index = manifest_index_by_label[label]
        bridge_index = label_to_index[label]
        if bridge_index >= len(bridge_poses) or bridge_poses[bridge_index] is None:
            raise ValueError(
                f"Bridge prior override label {label} maps to pose {bridge_index}, "
                f"but {bridge_pose_yaml} has no such pose")
        before = updated[manifest_index]
        after = bridge_poses[bridge_index]
        rows.append({
            "camera_id": label,
            "outer_manifest_index": manifest_index,
            "bridge_pose_index": bridge_index,
            "center_delta_m": float(np.linalg.norm(pose_center(before) - pose_center(after))),
            "rotation_delta_deg": float(pose_rotation_delta_deg(before, after)),
        })
        updated[manifest_index] = after
    return rows, updated


def make_observation(frame_idx, cam_idx, xy, point, feature_id=None):
    return (
        int(frame_idx),
        int(cam_idx),
        np.asarray(xy, dtype=np.float64),
        np.asarray(point, dtype=np.float64),
        None if feature_id is None else int(feature_id),
    )


def unpack_observation(observation):
    if len(observation) == 5:
        frame_idx, cam_idx, xy, point, feature_id = observation
    elif len(observation) == 4:
        frame_idx, cam_idx, xy, point = observation
        feature_id = None
    else:
        raise ValueError(f"Unexpected observation tuple length: {len(observation)}")
    return int(frame_idx), int(cam_idx), xy, point, feature_id


def observation_camera_index(observation):
    return unpack_observation(observation)[1]


def observation_feature_fields(feature_id):
    if feature_id is None:
        return "", "", ""
    try:
        feature_id = int(feature_id)
    except (TypeError, ValueError):
        return "", "", ""
    tag_id = feature_id // 4
    corner_id = feature_id % 4
    face_id = tag_id // 32 if tag_id >= 0 else ""
    return tag_id, corner_id, face_id


def observation_face_id(feature_id):
    _tag_id, _corner_id, face_id = observation_feature_fields(feature_id)
    if face_id == "":
        return None
    return int(face_id)


def physical_corner_for_opencv_corner(corner_id, tag_rotation_degrees):
    if tag_rotation_degrees == 0:
        return [3, 2, 1, 0][corner_id]
    if tag_rotation_degrees == 180:
        return [1, 0, 3, 2][corner_id]
    raise ValueError(f"Unsupported tag rotation: {tag_rotation_degrees}")


def tower_layout_from_args(args):
    return {
        "first_tag_id": int(args.tower_first_tag_id),
        "face_id_stride": int(args.tower_face_id_stride),
        "tag_columns": int(args.tower_tag_columns),
        "tag_rows": int(args.tower_tag_rows),
        "tag_size_m": float(args.tower_tag_size_m),
        "tag_spacing_m": float(args.tower_tag_spacing_m),
        "tag_rotation_degrees": int(args.tower_tag_rotation_degrees),
    }


def tower_face_and_local_tag_id(feature_id, layout):
    tag_id = int(feature_id) // 4
    first_tag_id = int(layout["first_tag_id"])
    face_stride = int(layout["face_id_stride"])
    local = tag_id - first_tag_id
    if local < 0:
        return None, None
    face_id = local // face_stride
    local_tag_id = local - face_id * face_stride
    return int(face_id), int(local_tag_id)


def tower_face_local_point_for_feature(feature_id, layout):
    if feature_id is None:
        return None
    face_id, local_tag_id = tower_face_and_local_tag_id(feature_id, layout)
    if face_id is None:
        return None
    columns = int(layout["tag_columns"])
    rows = int(layout["tag_rows"])
    if local_tag_id < 0 or local_tag_id >= columns * rows:
        return None
    row = local_tag_id // columns
    col = local_tag_id % columns
    tag_size = float(layout["tag_size_m"])
    spacing = float(layout["tag_spacing_m"])
    pitch = tag_size + spacing
    half_tag = 0.5 * tag_size
    center_u = (col - 0.5 * (columns - 1)) * pitch
    center_z = (row - 0.5 * (rows - 1)) * pitch
    corners = [
        np.asarray([0.0, center_u - half_tag, center_z - half_tag], dtype=np.float64),
        np.asarray([0.0, center_u + half_tag, center_z - half_tag], dtype=np.float64),
        np.asarray([0.0, center_u + half_tag, center_z + half_tag], dtype=np.float64),
        np.asarray([0.0, center_u - half_tag, center_z + half_tag], dtype=np.float64),
    ]
    corner_id = int(feature_id) % 4
    physical_corner_id = physical_corner_for_opencv_corner(
        corner_id,
        int(layout["tag_rotation_degrees"]))
    return corners[physical_corner_id]


def initialize_tower_face_base_poses(dataset, layout, face_count):
    local_by_face = [[] for _ in range(face_count)]
    target_by_face = [[] for _ in range(face_count)]
    for feature_id, target_point in dataset["known_points"].items():
        face_id, _local_tag_id = tower_face_and_local_tag_id(feature_id, layout)
        if face_id is None or face_id < 0 or face_id >= face_count:
            continue
        local_point = tower_face_local_point_for_feature(feature_id, layout)
        if local_point is None:
            continue
        local_by_face[face_id].append(local_point)
        target_by_face[face_id].append(np.asarray(target_point, dtype=np.float64))
    poses = []
    rows = []
    for face_id in range(face_count):
        if len(local_by_face[face_id]) >= 3:
            pose, residuals = rigid_transform(
                np.asarray(local_by_face[face_id], dtype=np.float64),
                np.asarray(target_by_face[face_id], dtype=np.float64))
        else:
            pose = np.eye(4, dtype=np.float64)
            residuals = np.asarray([], dtype=np.float64)
        poses.append(pose)
        rows.append({
            "face_id": face_id,
            "point_count": len(local_by_face[face_id]),
            "init_rms_m": (
                float(np.sqrt(np.mean(residuals ** 2)))
                if residuals.size
                else None
            ),
            "init_max_m": float(np.max(residuals)) if residuals.size else None,
        })
    return poses, rows


def face_normal_for_feature(feature_id, faces=8, face0_angle_degrees=0.0):
    if feature_id is None:
        return None
    try:
        tag_id = int(feature_id) // 4
    except (TypeError, ValueError):
        return None
    if tag_id < 0:
        return None
    face_id = tag_id // 32
    theta = math.radians(float(face0_angle_degrees)) + face_id * 2.0 * math.pi / int(faces)
    return np.asarray([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)


def adjusted_tower_point(
        point,
        feature_id,
        tower_face_width_delta_m=0.0,
        faces=8,
        face0_angle_degrees=0.0):
    delta_width = float(tower_face_width_delta_m or 0.0)
    if abs(delta_width) < 1e-12:
        return point
    normal = face_normal_for_feature(feature_id, faces, face0_angle_degrees)
    if normal is None:
        return point
    apothem_delta = delta_width / (2.0 * math.tan(math.pi / int(faces)))
    return point + normal * apothem_delta


def apply_tower_face_pose_delta(point, feature_id, tower_face_pose_deltas=None):
    if tower_face_pose_deltas is None:
        return point
    face_id = observation_face_id(feature_id)
    if face_id is None or face_id < 0 or face_id >= len(tower_face_pose_deltas):
        return point
    delta = tower_face_pose_deltas[face_id]
    if delta is None:
        return point
    return delta[:3, :3] @ point + delta[:3, 3]


def adjusted_tower_model_point(
        point,
        feature_id,
        tower_face_width_delta_m=0.0,
        tower_face_pose_deltas=None,
        tower_point_model="dataset_3d",
        tower_face_base_poses=None,
        tower_layout=None):
    if tower_point_model == "independent_face_planes":
        if tower_layout is None:
            return point
        local_point = tower_face_local_point_for_feature(feature_id, tower_layout)
        face_id = observation_face_id(feature_id)
        if local_point is None or face_id is None:
            return point
        if tower_face_base_poses is None or face_id >= len(tower_face_base_poses):
            base_pose = np.eye(4, dtype=np.float64)
        else:
            base_pose = tower_face_base_poses[face_id]
        if tower_face_pose_deltas is None or face_id >= len(tower_face_pose_deltas):
            delta_pose = np.eye(4, dtype=np.float64)
        else:
            delta_pose = tower_face_pose_deltas[face_id]
        pose = delta_pose @ base_pose
        return pose[:3, :3] @ local_point + pose[:3, 3]
    if tower_point_model != "dataset_3d":
        raise ValueError(f"Unsupported tower point model: {tower_point_model}")
    point = adjusted_tower_point(point, feature_id, tower_face_width_delta_m)
    return apply_tower_face_pose_delta(point, feature_id, tower_face_pose_deltas)


def observation_gate_key(observation):
    frame_idx, cam_idx, xy, _point, feature_id = unpack_observation(observation)
    if feature_id is None:
        return ("legacy", id(observation))
    return (
        "feature",
        frame_idx,
        cam_idx,
        int(feature_id),
        round(float(xy[0]), 9),
        round(float(xy[1]), 9),
    )


def build_observations(dataset):
    observations_by_frame = [[] for _ in dataset["imagesets"]]
    observations_by_camera = [[] for _ in range(dataset["camera_count"])]
    for frame_idx, imageset in enumerate(dataset["imagesets"]):
        for cam_idx, features in enumerate(imageset["features"]):
            for x, y, feature_id in features:
                point = dataset["known_points"].get(feature_id)
                if point is None:
                    continue
                obs = make_observation(frame_idx, cam_idx, [x, y], point, feature_id)
                observations_by_frame[frame_idx].append(obs)
                observations_by_camera[cam_idx].append(obs)
    return observations_by_frame, observations_by_camera


def projection_residuals(
        observations,
        camera_poses,
        tower_poses,
        intrinsics,
        tower_face_width_delta_m=0.0,
        tower_face_pose_deltas=None,
        tower_point_model="dataset_3d",
        tower_face_base_poses=None,
        tower_layout=None):
    residuals = []
    for observation in observations:
        frame_idx, cam_idx, xy, point, feature_id = unpack_observation(observation)
        tower_pose = tower_poses[frame_idx]
        if tower_pose is None:
            continue
        camera_tr_tower = camera_poses[cam_idx] @ tower_pose
        point = adjusted_tower_model_point(
            point,
            feature_id,
            tower_face_width_delta_m,
            tower_face_pose_deltas,
            tower_point_model,
            tower_face_base_poses,
            tower_layout)
        pixel = project_point(camera_tr_tower[:3, :3] @ point + camera_tr_tower[:3, 3], intrinsics[cam_idx])
        if pixel is None or not np.all(np.isfinite(pixel)):
            residuals.extend([1000.0, 1000.0])
        else:
            residuals.extend((pixel - xy).tolist())
    return np.asarray(residuals, dtype=np.float64)


def summarize_residuals(residuals):
    if residuals.size < 2:
        return {"count": 0, "mean_px": None, "median_px": None, "p90_px": None, "max_px": None}
    norms = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
    return {
        "count": int(norms.size),
        "mean_px": float(np.mean(norms)),
        "median_px": float(np.median(norms)),
        "p90_px": float(np.percentile(norms, 90)),
        "max_px": float(np.max(norms)),
    }


def residual_norms(residuals):
    if residuals.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(residuals.reshape(-1, 2), axis=1)


def summarize_camera_reprojection(
        manifest,
        observations_by_camera,
        prior_camera_poses,
        refined_camera_poses,
        tower_poses,
        prior_intrinsics,
        refined_intrinsics,
        tower_face_width_delta_m=0.0,
        tower_face_pose_deltas=None,
        tower_point_model="dataset_3d",
        tower_face_base_poses=None,
        tower_layout=None):
    rows = []
    for idx, row in enumerate(manifest):
        before_norms = residual_norms(
            projection_residuals(
                observations_by_camera[idx],
                prior_camera_poses,
                tower_poses,
                prior_intrinsics,
                tower_face_width_delta_m,
                tower_face_pose_deltas,
                tower_point_model,
                tower_face_base_poses,
                tower_layout))
        after_norms = residual_norms(
            projection_residuals(
                observations_by_camera[idx],
                refined_camera_poses,
                tower_poses,
                refined_intrinsics,
                tower_face_width_delta_m,
                tower_face_pose_deltas,
                tower_point_model,
                tower_face_base_poses,
                tower_layout))

        def stat(norms, fn, default=""):
            if norms.size == 0:
                return default
            return f"{float(fn(norms)):.8g}"

        under_100 = float(np.mean(after_norms < 100.0)) if after_norms.size else 0.0
        under_300 = float(np.mean(after_norms < 300.0)) if after_norms.size else 0.0
        rows.append({
            "camera_index": idx,
            "camera_id": row["camera_id"],
            "observation_count": int(after_norms.size),
            "before_median_px": stat(before_norms, np.median),
            "before_p90_px": stat(before_norms, lambda x: np.percentile(x, 90)),
            "after_median_px": stat(after_norms, np.median),
            "after_p90_px": stat(after_norms, lambda x: np.percentile(x, 90)),
            "after_max_px": stat(after_norms, np.max),
            "after_under_100_fraction": f"{under_100:.8g}",
            "after_under_300_fraction": f"{under_300:.8g}",
        })
    return rows


def write_camera_reprojection_tsv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "camera_index", "camera_id", "observation_count",
                "before_median_px", "before_p90_px",
                "after_median_px", "after_p90_px", "after_max_px",
                "after_under_100_fraction", "after_under_300_fraction",
            ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def accepted_refined_poses(camera_priors, refined_camera_poses, camera_rows, active_camera, used_camera, args):
    if args.accept_camera_median_px <= 0:
        if args.allow_ungated_accepted_output:
            return list(refined_camera_poses), []
        return list(camera_priors), []
    accepted = []
    output = []
    for idx, row in enumerate(camera_rows):
        try:
            median_px = float(row["after_median_px"])
            p90_px = float(row["after_p90_px"])
            under_fraction = float(row["after_under_300_fraction"])
        except (TypeError, ValueError):
            median_px = float("inf")
            p90_px = float("inf")
            under_fraction = 0.0
        delta_stats = delta_pose_stats(refined_camera_poses[idx] @ invert_pose(camera_priors[idx]))
        p90_ok = args.accept_camera_p90_px <= 0 or p90_px <= args.accept_camera_p90_px
        rotation_ok = (
            args.accept_camera_max_delta_rotation_deg <= 0
            or delta_stats["rotation_deg"] <= args.accept_camera_max_delta_rotation_deg
        )
        translation_ok = (
            args.accept_camera_max_delta_translation_m <= 0
            or delta_stats["translation_m"] <= args.accept_camera_max_delta_translation_m
        )
        keep_refined = (
            active_camera[idx]
            and used_camera[idx]
            and row["observation_count"] > 0
            and median_px <= args.accept_camera_median_px
            and p90_ok
            and under_fraction >= args.accept_camera_min_under_300_fraction
            and rotation_ok
            and translation_ok
        )
        output.append(refined_camera_poses[idx] if keep_refined else camera_priors[idx])
        if keep_refined:
            accepted.append(row["camera_id"])
    return output, accepted


def summarize_camera_acceptance(manifest, camera_rows, active_camera, used_camera, accepted_camera_ids, args):
    accepted_camera_ids = set(accepted_camera_ids)
    acceptance_enabled = args.accept_camera_median_px > 0
    rows = []
    for idx, row in enumerate(camera_rows):
        camera_id = row["camera_id"]
        if camera_id in accepted_camera_ids:
            decision = "accepted_refined"
            output_pose = "refined"
            reason = "passes_acceptance_gate"
        elif not acceptance_enabled:
            if args.allow_ungated_accepted_output:
                decision = "refined_no_acceptance_gate"
                output_pose = "refined"
                reason = "acceptance_gate_disabled_allowed_by_flag"
            else:
                decision = "ungated_prior_only"
                output_pose = "prior"
                reason = "acceptance_gate_disabled_without_allow_ungated_output"
        elif not used_camera[idx]:
            decision = "excluded_prior_only"
            output_pose = "prior"
            reason = "below_min_camera_observations_for_use"
        elif not active_camera[idx]:
            decision = "inactive_prior_only"
            output_pose = "prior"
            reason = "below_min_camera_observations_for_delta"
        else:
            decision = "rejected_to_prior"
            output_pose = "prior"
            reason = "failed_acceptance_gate"
        rows.append({
            "camera_index": idx,
            "camera_id": manifest[idx]["camera_id"],
            "decision": decision,
            "output_pose": output_pose,
            "reason": reason,
            "active_delta": "yes" if active_camera[idx] else "no",
            "used_observation": "yes" if used_camera[idx] else "no",
            "observation_count": row["observation_count"],
            "after_median_px": row["after_median_px"],
            "after_under_300_fraction": row["after_under_300_fraction"],
        })
    return rows


def filter_observations_by_camera(observations_by_frame, observations_by_camera, min_observations):
    use_camera = [
        len(observations) >= min_observations
        for observations in observations_by_camera
    ]
    filtered_by_frame = []
    for observations in observations_by_frame:
        filtered_by_frame.append([
            obs for obs in observations
            if use_camera[observation_camera_index(obs)]
        ])
    filtered_by_camera = [
        observations if use_camera[idx] else []
        for idx, observations in enumerate(observations_by_camera)
    ]
    return filtered_by_frame, filtered_by_camera, use_camera


def observation_residual_norm(
        observation,
        camera_poses,
        tower_poses,
        intrinsics,
        tower_face_width_delta_m=0.0,
        tower_face_pose_deltas=None,
        tower_point_model="dataset_3d",
        tower_face_base_poses=None,
        tower_layout=None):
    frame_idx, cam_idx, xy, point, feature_id = unpack_observation(observation)
    tower_pose = tower_poses[frame_idx]
    if tower_pose is None:
        return None
    camera_tr_tower = camera_poses[cam_idx] @ tower_pose
    point = adjusted_tower_model_point(
        point,
        feature_id,
        tower_face_width_delta_m,
        tower_face_pose_deltas,
        tower_point_model,
        tower_face_base_poses,
        tower_layout)
    pixel = project_point(
        camera_tr_tower[:3, :3] @ point + camera_tr_tower[:3, 3],
        intrinsics[cam_idx])
    if pixel is None or not np.all(np.isfinite(pixel)):
        return None
    return float(np.linalg.norm(pixel - xy))


def format_optional_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return format_float(value)


def project_observation(
        observation,
        camera_poses,
        tower_poses,
        intrinsics,
        tower_face_width_delta_m=0.0,
        tower_face_pose_deltas=None,
        tower_point_model="dataset_3d",
        tower_face_base_poses=None,
        tower_layout=None):
    frame_idx, cam_idx, xy, point, feature_id = unpack_observation(observation)
    tower_pose = tower_poses[frame_idx]
    if tower_pose is None:
        return "missing_tower_pose", None, None
    camera_tr_tower = camera_poses[cam_idx] @ tower_pose
    point = adjusted_tower_model_point(
        point,
        feature_id,
        tower_face_width_delta_m,
        tower_face_pose_deltas,
        tower_point_model,
        tower_face_base_poses,
        tower_layout)
    pixel = project_point(
        camera_tr_tower[:3, :3] @ point + camera_tr_tower[:3, 3],
        intrinsics[cam_idx])
    if pixel is None or not np.all(np.isfinite(pixel)):
        return "invalid_projection", None, None
    residual = pixel - xy
    return "ok", pixel, residual


def write_observation_residuals(
        path,
        manifest,
        observations_by_frame_before_gate,
        observations_by_frame_after_gate,
        camera_poses,
        tower_poses,
        intrinsics,
        tower_face_width_delta_m=0.0,
        tower_face_pose_deltas=None,
        tower_point_model="dataset_3d",
        tower_face_base_poses=None,
        tower_layout=None):
    used_after_gate = {
        observation_gate_key(obs)
        for frame in observations_by_frame_after_gate
        for obs in frame
    }
    fieldnames = [
        "frame_index",
        "camera_index",
        "camera_id",
        "feature_id",
        "tag_id",
        "corner_id",
        "face_id",
        "observed_x",
        "observed_y",
        "projected_x",
        "projected_y",
        "residual_x_px",
        "residual_y_px",
        "residual_px",
        "used_after_gate",
        "projection_status",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for observations in observations_by_frame_before_gate:
            for obs in observations:
                frame_idx, cam_idx, xy, _point, feature_id = unpack_observation(obs)
                tag_id, corner_id, face_id = observation_feature_fields(feature_id)
                projection_status, pixel, residual = project_observation(
                    obs,
                    camera_poses,
                    tower_poses,
                    intrinsics,
                    tower_face_width_delta_m,
                    tower_face_pose_deltas,
                    tower_point_model,
                    tower_face_base_poses,
                    tower_layout)
                residual_px = None
                if residual is not None:
                    residual_px = float(np.linalg.norm(residual))
                writer.writerow({
                    "frame_index": frame_idx,
                    "camera_index": cam_idx,
                    "camera_id": manifest[cam_idx]["camera_id"] if cam_idx < len(manifest) else "",
                    "feature_id": "" if feature_id is None else int(feature_id),
                    "tag_id": tag_id,
                    "corner_id": corner_id,
                    "face_id": face_id,
                    "observed_x": format_optional_float(xy[0]),
                    "observed_y": format_optional_float(xy[1]),
                    "projected_x": format_optional_float(pixel[0] if pixel is not None else None),
                    "projected_y": format_optional_float(pixel[1] if pixel is not None else None),
                    "residual_x_px": format_optional_float(residual[0] if residual is not None else None),
                    "residual_y_px": format_optional_float(residual[1] if residual is not None else None),
                    "residual_px": format_optional_float(residual_px),
                    "used_after_gate": "yes" if observation_gate_key(obs) in used_after_gate else "no",
                    "projection_status": projection_status,
                })


def filter_observations_by_projection_gate(
        observations_by_frame,
        camera_count,
        camera_poses,
        tower_poses,
        intrinsics,
        max_residual_px,
        tower_face_width_delta_m=0.0,
        tower_face_pose_deltas=None,
        tower_point_model="dataset_3d",
        tower_face_base_poses=None,
        tower_layout=None):
    if max_residual_px <= 0:
        observations_by_camera = [[] for _ in range(camera_count)]
        for frame in observations_by_frame:
            for obs in frame:
                observations_by_camera[observation_camera_index(obs)].append(obs)
        return observations_by_frame, observations_by_camera, {
            "enabled": False,
            "max_residual_px": max_residual_px,
            "input_observations": int(sum(len(frame) for frame in observations_by_frame)),
            "kept_observations": int(sum(len(frame) for frame in observations_by_frame)),
            "removed_observations": 0,
            "missing_pose_or_invalid_projection": 0,
        }

    filtered_by_frame = []
    filtered_by_camera = [[] for _ in range(camera_count)]
    input_count = 0
    kept_count = 0
    invalid_count = 0
    removed_count = 0
    for observations in observations_by_frame:
        kept_frame = []
        for obs in observations:
            input_count += 1
            norm = observation_residual_norm(
                obs,
                camera_poses,
                tower_poses,
                intrinsics,
                tower_face_width_delta_m,
                tower_face_pose_deltas,
                tower_point_model,
                tower_face_base_poses,
                tower_layout)
            if norm is None:
                invalid_count += 1
                removed_count += 1
                continue
            if norm <= max_residual_px:
                kept_frame.append(obs)
                filtered_by_camera[observation_camera_index(obs)].append(obs)
                kept_count += 1
            else:
                removed_count += 1
        filtered_by_frame.append(kept_frame)

    return filtered_by_frame, filtered_by_camera, {
        "enabled": True,
        "max_residual_px": max_residual_px,
        "input_observations": input_count,
        "kept_observations": kept_count,
        "removed_observations": removed_count,
        "missing_pose_or_invalid_projection": invalid_count,
    }


def count_observations(observations_by_frame):
    if isinstance(observations_by_frame, int):
        return int(observations_by_frame)
    return int(sum(len(frame) for frame in observations_by_frame))


def observations_by_camera_from_frames(observations_by_frame, camera_count):
    observations_by_camera = [[] for _ in range(camera_count)]
    for frame in observations_by_frame:
        for obs in frame:
            observations_by_camera[observation_camera_index(obs)].append(obs)
    return observations_by_camera


def observations_by_face_from_frames(observations_by_frame, face_count):
    observations_by_face = [[] for _ in range(face_count)]
    for frame in observations_by_frame:
        for obs in frame:
            face_id = observation_face_id(unpack_observation(obs)[4])
            if face_id is None or face_id < 0 or face_id >= face_count:
                continue
            observations_by_face[face_id].append(obs)
    return observations_by_face


def make_post_refine_observation_gate_summary(
        enabled,
        threshold_px,
        outer_iterations,
        input_observations,
        kept_observations,
        removed_observations,
        missing_pose_or_invalid_projection):
    missing_count = int(missing_pose_or_invalid_projection)
    return {
        "enabled": bool(enabled),
        "threshold": float(threshold_px),
        "threshold_px": float(threshold_px),
        "outer_iterations": int(outer_iterations),
        "input_observations": count_observations(input_observations),
        "kept_observations": count_observations(kept_observations),
        "removed_observations": int(removed_observations),
        "missing_observations": missing_count,
        "missing_pose_or_invalid_projection": missing_count,
    }


def filter_post_refine_observations_by_projection_gate(
        observations_by_frame,
        camera_count,
        camera_poses,
        tower_poses,
        intrinsics,
        max_residual_px,
        outer_iterations,
        tower_face_width_delta_m=0.0,
        tower_face_pose_deltas=None,
        tower_point_model="dataset_3d",
        tower_face_base_poses=None,
        tower_layout=None):
    if max_residual_px <= 0:
        observations_by_camera = observations_by_camera_from_frames(observations_by_frame, camera_count)
        return observations_by_frame, observations_by_camera, make_post_refine_observation_gate_summary(
            enabled=False,
            threshold_px=max_residual_px,
            outer_iterations=outer_iterations,
            input_observations=observations_by_frame,
            kept_observations=observations_by_frame,
            removed_observations=0,
            missing_pose_or_invalid_projection=0)

    filtered_by_frame, filtered_by_camera, gate = filter_observations_by_projection_gate(
        observations_by_frame,
        camera_count,
        camera_poses,
        tower_poses,
        intrinsics,
        max_residual_px,
        tower_face_width_delta_m,
        tower_face_pose_deltas,
        tower_point_model,
        tower_face_base_poses,
        tower_layout)
    return filtered_by_frame, filtered_by_camera, make_post_refine_observation_gate_summary(
        enabled=True,
        threshold_px=max_residual_px,
        outer_iterations=outer_iterations,
        input_observations=gate["input_observations"],
        kept_observations=gate["kept_observations"],
        removed_observations=gate["removed_observations"],
        missing_pose_or_invalid_projection=gate["missing_pose_or_invalid_projection"])


def clip_residual_norms(residuals, max_norm):
    if max_norm <= 0 or residuals.size < 2:
        return residuals
    shaped = residuals.reshape(-1, 2).copy()
    norms = np.linalg.norm(shaped, axis=1)
    mask = norms > max_norm
    if np.any(mask):
        shaped[mask] *= (max_norm / norms[mask])[:, None]
    return shaped.reshape(-1)


def weighted_residual_and_jacobian(transform, residual_fn, eps):
    r0 = residual_fn(transform)
    if r0.size == 0:
        return r0, np.zeros((0, 6), dtype=np.float64)
    J = np.zeros((r0.size, 6), dtype=np.float64)
    for j in range(6):
        step = np.zeros(6, dtype=np.float64)
        step[j] = eps[j]
        r1 = residual_fn(se3_exp(step) @ transform)
        J[:, j] = (r1 - r0) / eps[j]
    return r0, J


def optimize_block(transform, residual_fn, iterations, damping, max_rotation_step, max_translation_step):
    eps = np.asarray([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4], dtype=np.float64)
    current = transform.copy()
    for _ in range(iterations):
        r, J = weighted_residual_and_jacobian(current, residual_fn, eps)
        if r.size < 6:
            break
        lhs = J.T @ J + damping * np.eye(6)
        rhs = -J.T @ r
        try:
            dx = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        rot_norm = np.linalg.norm(dx[:3])
        trans_norm = np.linalg.norm(dx[3:])
        if rot_norm > max_rotation_step:
            dx[:3] *= max_rotation_step / rot_norm
        if trans_norm > max_translation_step:
            dx[3:] *= max_translation_step / trans_norm
        if np.linalg.norm(dx) < 1e-9:
            break
        current_norm = np.linalg.norm(r)
        accepted = False
        for scale in BACKTRACKING_STEP_SCALES:
            candidate = se3_exp(scale * dx) @ current
            if np.linalg.norm(residual_fn(candidate)) <= current_norm:
                current = candidate
                accepted = True
                break
        if not accepted:
            break
    return current


def vector_residual_and_jacobian(values, residual_fn, eps):
    values = np.asarray(values, dtype=np.float64)
    r0 = residual_fn(values)
    if r0.size == 0:
        return r0, np.zeros((0, values.size), dtype=np.float64)
    J = np.zeros((r0.size, values.size), dtype=np.float64)
    for j in range(values.size):
        step = np.zeros(values.size, dtype=np.float64)
        step[j] = eps[j]
        r1 = residual_fn(values + step)
        J[:, j] = (r1 - r0) / eps[j]
    return r0, J


def optimize_vector(values, residual_fn, iterations, damping, eps, max_step):
    current = np.asarray(values, dtype=np.float64).copy()
    eps = np.asarray(eps, dtype=np.float64)
    max_step = np.asarray(max_step, dtype=np.float64)
    for _ in range(iterations):
        r, J = vector_residual_and_jacobian(current, residual_fn, eps)
        if r.size < current.size:
            break
        lhs = J.T @ J + damping * np.eye(current.size)
        rhs = -J.T @ r
        try:
            dx = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        dx = np.clip(dx, -max_step, max_step)
        if np.linalg.norm(dx) < 1e-10:
            break
        current_norm = np.linalg.norm(r)
        accepted = False
        for scale in BACKTRACKING_STEP_SCALES:
            candidate = current + scale * dx
            if np.linalg.norm(residual_fn(candidate)) <= current_norm:
                current = candidate
                accepted = True
                break
        if not accepted:
            break
    return current


def initialize_tower_poses_from_pnp(pnp_views, camera_priors, frame_count, min_votes):
    votes_by_frame = [[] for _ in range(frame_count)]
    for view in pnp_views:
        rig_tr_tower = invert_pose(camera_priors[view["camera_index"]]) @ view["camera_tr_tower"]
        votes_by_frame[view["imageset_index"]].append({
            "pose": rig_tr_tower,
            "median_error_px": float(view["median_error_px"]),
        })
    tower_poses = [None for _ in range(frame_count)]
    frame_pnp_quality = []
    for idx, votes in enumerate(votes_by_frame):
        errors = [vote["median_error_px"] for vote in votes]
        row = {
            "pnp_vote_count": len(votes),
            "pnp_median_error_px": float(np.median(errors)) if errors else None,
            "pnp_pose_average": "none" if not votes else "insufficient_votes",
        }
        if len(votes) >= min_votes:
            poses = [vote["pose"] for vote in votes]
            if len(votes) == 1:
                tower_poses[idx] = poses[0].copy()
                row["pnp_pose_average"] = "single_vote"
            else:
                tower_poses[idx] = robust_weighted_average_poses(poses, errors)
                row["pnp_pose_average"] = "robust_weighted"
        frame_pnp_quality.append(row)
    return tower_poses, frame_pnp_quality


def delta_pose_stats(delta):
    xi = se3_log_approx(delta)
    return {
        "rotation_deg": float(np.linalg.norm(xi[:3]) * 180.0 / math.pi),
        "translation_m": float(np.linalg.norm(xi[3:])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pnp_views", required=True, type=Path)
    parser.add_argument("--outer_colmap_images_txt", type=Path)
    parser.add_argument("--bridge_pose_yaml", required=True, type=Path)
    parser.add_argument("--camera_prior_pose_yaml", type=Path)
    parser.add_argument("--alignment_pose_yaml", type=Path)
    parser.add_argument("--alignment_pnp_summary_tsv", type=Path)
    parser.add_argument("--intrinsics_dir", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--intrinsics_mode", choices=["colmap_fixed", "central_opencv"], default="colmap_fixed")
    parser.add_argument(
        "--intrinsics_refine_mode",
        choices=INTRINSICS_REFINE_MODES,
        default="fixed",
        help="Opt-in intrinsic refinement mode. Distortion terms are always fixed.",
    )
    parser.add_argument(
        "--optimize_intrinsics",
        action="store_true",
        help="Shortcut for --intrinsics_refine_mode per_camera_fxfycxcy.",
    )
    parser.add_argument("--anchor_labels", default="4-1,4-2,4-3")
    parser.add_argument(
        "--anchor_label_to_pose_index",
        default="4-1:9,4-2:10,4-3:11",
        help=(
            "Bridge YAML pose indices for top-down anchors. The default matches "
            "the current all32 bridge order: outer 0..23, inner 24..31."
        ),
    )
    parser.add_argument(
        "--bridge_prior_override_labels",
        default="",
        help=(
            "Comma-separated outer camera labels whose camera_tr_rig prior should be "
            "replaced by the full pose from --bridge_pose_yaml before tower tag refine. "
            "This is intended for top-down bridge cameras after the bridge metric gate passes."
        ),
    )
    parser.add_argument("--max_anchor_median_error_px", type=float, default=2.0)
    parser.add_argument("--min_anchor_solved_views", type=int, default=10)
    parser.add_argument("--max_pnp_median_error_px", type=float, default=8.0)
    parser.add_argument("--min_frame_pnp_votes", type=int, default=1)
    parser.add_argument("--min_camera_observations_for_delta", type=int, default=32)
    parser.add_argument("--min_camera_observations_for_use", type=int, default=0)
    parser.add_argument("--outer_iterations", type=int, default=5)
    parser.add_argument("--block_iterations", type=int, default=8)
    parser.add_argument("--observation_residual_gate_px", type=float, default=600.0)
    parser.add_argument("--post_refine_observation_residual_gate_px", type=float, default=0.0)
    parser.add_argument("--post_refine_outer_iterations", type=int, default=2)
    parser.add_argument("--optimizer_residual_clip_px", type=float, default=500.0)
    parser.add_argument("--accept_camera_median_px", type=float, default=350.0)
    parser.add_argument("--accept_camera_p90_px", type=float, default=450.0)
    parser.add_argument("--accept_camera_min_under_300_fraction", type=float, default=0.45)
    parser.add_argument("--accept_camera_max_delta_translation_m", type=float, default=0.35)
    parser.add_argument("--accept_camera_max_delta_rotation_deg", type=float, default=6.5)
    parser.add_argument("--allow_ungated_accepted_output", action="store_true")
    parser.add_argument("--delta_translation_sigma_m", type=float, default=0.12)
    parser.add_argument("--delta_rotation_sigma_deg", type=float, default=3.0)
    parser.add_argument("--camera_delta_max_rotation_step_deg", type=float, default=0.02 * 180.0 / math.pi)
    parser.add_argument("--camera_delta_max_translation_step_m", type=float, default=0.03)
    parser.add_argument("--tower_pose_max_rotation_step_deg", type=float, default=0.05 * 180.0 / math.pi)
    parser.add_argument("--tower_pose_max_translation_step_m", type=float, default=0.10)
    parser.add_argument("--intrinsics_focal_sigma_frac", type=float, default=0.01)
    parser.add_argument("--intrinsics_principal_sigma_px", type=float, default=8.0)
    parser.add_argument("--intrinsics_distortion_sigma", type=float, default=0.05)
    parser.add_argument("--intrinsics_max_focal_step_frac", type=float, default=0.002)
    parser.add_argument("--intrinsics_max_principal_step_px", type=float, default=1.0)
    parser.add_argument("--intrinsics_max_distortion_step", type=float, default=0.01)
    parser.add_argument("--intrinsics_max_total_focal_delta_frac", type=float, default=0.0)
    parser.add_argument("--intrinsics_max_total_principal_delta_px", type=float, default=0.0)
    parser.add_argument("--intrinsics_max_total_distortion_delta", type=float, default=0.0)
    parser.add_argument("--intrinsics_block_iterations", type=int, default=4)
    parser.add_argument("--accept_camera_max_intrinsic_focal_delta_frac", type=float, default=0.02)
    parser.add_argument("--accept_camera_max_intrinsic_principal_delta_px", type=float, default=16.0)
    parser.add_argument("--accept_camera_max_intrinsic_distortion_delta", type=float, default=0.15)
    parser.add_argument(
        "--optimize_tower_face_width",
        action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument("--tower_face_width_initial_m", type=float, default=0.0)
    parser.add_argument("--tower_face_width_sigma_m", type=float, default=0.03)
    parser.add_argument("--tower_face_width_min_m", type=float, default=0.18)
    parser.add_argument("--tower_face_width_max_m", type=float, default=0.32)
    parser.add_argument("--tower_face_width_max_step_m", type=float, default=0.005)
    parser.add_argument(
        "--tower_point_model",
        choices=["dataset_3d", "independent_face_planes"],
        default="dataset_3d",
        help=(
            "dataset_3d uses the dataset's known 3D points. independent_face_planes "
            "rebuilds each tag corner in its own face-local plane and estimates the "
            "face pose instead of relying on ideal octagonal-prism geometry."
        ),
    )
    parser.add_argument("--tower_first_tag_id", type=int, default=0)
    parser.add_argument("--tower_face_id_stride", type=int, default=32)
    parser.add_argument("--tower_tag_columns", type=int, default=2)
    parser.add_argument("--tower_tag_rows", type=int, default=16)
    parser.add_argument("--tower_tag_size_m", type=float, default=0.08)
    parser.add_argument("--tower_tag_spacing_m", type=float, default=0.02)
    parser.add_argument("--tower_tag_rotation_degrees", type=int, default=180)
    parser.add_argument(
        "--optimize_tower_face_poses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine one small SE(3) delta per AprilTag tower face.")
    parser.add_argument("--tower_face_count", type=int, default=8)
    parser.add_argument("--tower_face_pose_min_observations", type=int, default=128)
    parser.add_argument("--tower_face_pose_rotation_sigma_deg", type=float, default=3.0)
    parser.add_argument("--tower_face_pose_translation_sigma_m", type=float, default=0.025)
    parser.add_argument("--tower_face_pose_max_rotation_step_deg", type=float, default=0.25)
    parser.add_argument("--tower_face_pose_max_translation_step_m", type=float, default=0.002)
    args = parser.parse_args()

    if args.optimize_intrinsics and args.intrinsics_refine_mode == "fixed":
        args.intrinsics_refine_mode = "per_camera_fxfycxcy"
    if args.intrinsics_refine_mode != "fixed" and args.intrinsics_focal_sigma_frac <= 0:
        raise ValueError("--intrinsics_focal_sigma_frac must be positive when refining intrinsics")
    if args.intrinsics_refine_mode == "per_camera_fxfycxcy" and args.intrinsics_principal_sigma_px <= 0:
        raise ValueError("--intrinsics_principal_sigma_px must be positive when refining cx/cy")
    if args.intrinsics_refine_mode == "per_camera_opencv5":
        if args.intrinsics_principal_sigma_px <= 0:
            raise ValueError("--intrinsics_principal_sigma_px must be positive when refining cx/cy")
        if args.intrinsics_distortion_sigma <= 0:
            raise ValueError("--intrinsics_distortion_sigma must be positive when refining distortion")
    if args.intrinsics_max_total_focal_delta_frac < 0:
        raise ValueError("--intrinsics_max_total_focal_delta_frac must be non-negative")
    if args.intrinsics_max_total_principal_delta_px < 0:
        raise ValueError("--intrinsics_max_total_principal_delta_px must be non-negative")
    if args.intrinsics_max_total_distortion_delta < 0:
        raise ValueError("--intrinsics_max_total_distortion_delta must be non-negative")
    if args.intrinsics_max_distortion_step <= 0:
        raise ValueError("--intrinsics_max_distortion_step must be positive")
    if args.optimize_tower_face_width and args.tower_face_width_initial_m <= 0:
        raise ValueError("--tower_face_width_initial_m must be positive when optimizing tower face width")
    if args.tower_face_width_sigma_m <= 0:
        raise ValueError("--tower_face_width_sigma_m must be positive")
    if args.tower_face_width_min_m <= 0 or args.tower_face_width_max_m <= 0:
        raise ValueError("--tower_face_width_min_m and --tower_face_width_max_m must be positive")
    if args.tower_face_width_min_m > args.tower_face_width_max_m:
        raise ValueError("--tower_face_width_min_m must be <= --tower_face_width_max_m")
    if args.tower_face_width_max_step_m <= 0:
        raise ValueError("--tower_face_width_max_step_m must be positive")
    if args.camera_delta_max_rotation_step_deg <= 0:
        raise ValueError("--camera_delta_max_rotation_step_deg must be positive")
    if args.camera_delta_max_translation_step_m <= 0:
        raise ValueError("--camera_delta_max_translation_step_m must be positive")
    if args.tower_pose_max_rotation_step_deg <= 0:
        raise ValueError("--tower_pose_max_rotation_step_deg must be positive")
    if args.tower_pose_max_translation_step_m <= 0:
        raise ValueError("--tower_pose_max_translation_step_m must be positive")
    if args.tower_point_model == "independent_face_planes" and args.optimize_tower_face_width:
        raise ValueError("--optimize_tower_face_width is incompatible with independent_face_planes")
    if args.tower_first_tag_id < 0:
        raise ValueError("--tower_first_tag_id must be non-negative")
    if args.tower_face_id_stride <= 0:
        raise ValueError("--tower_face_id_stride must be positive")
    if args.tower_tag_columns <= 0 or args.tower_tag_rows <= 0:
        raise ValueError("--tower_tag_columns and --tower_tag_rows must be positive")
    if args.tower_tag_size_m <= 0 or args.tower_tag_spacing_m < 0:
        raise ValueError("--tower_tag_size_m must be positive and --tower_tag_spacing_m must be non-negative")
    if args.tower_tag_rotation_degrees not in (0, 180):
        raise ValueError("--tower_tag_rotation_degrees must be 0 or 180")
    if args.tower_face_count <= 0:
        raise ValueError("--tower_face_count must be positive")
    if args.tower_face_pose_min_observations < 0:
        raise ValueError("--tower_face_pose_min_observations must be non-negative")
    if args.tower_face_pose_rotation_sigma_deg <= 0:
        raise ValueError("--tower_face_pose_rotation_sigma_deg must be positive")
    if args.tower_face_pose_translation_sigma_m <= 0:
        raise ValueError("--tower_face_pose_translation_sigma_m must be positive")
    if args.tower_face_pose_max_rotation_step_deg <= 0:
        raise ValueError("--tower_face_pose_max_rotation_step_deg must be positive")
    if args.tower_face_pose_max_translation_step_m <= 0:
        raise ValueError("--tower_face_pose_max_translation_step_m must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    dataset = read_dataset(args.dataset)
    manifest = read_manifest(args.manifest, dataset["camera_count"])
    tower_layout = tower_layout_from_args(args)
    tower_face_base_poses, tower_face_base_pose_rows = initialize_tower_face_base_poses(
        dataset,
        tower_layout,
        args.tower_face_count)
    base_intrinsics = collect_intrinsics(args.intrinsics_dir, manifest, dataset["image_sizes"], args.intrinsics_mode)
    colmap_images = load_colmap_images(args.outer_colmap_images_txt) if args.outer_colmap_images_txt else {}
    anchor_labels = [x for x in args.anchor_labels.split(",") if x]
    label_to_index = parse_label_pose_indices(args.anchor_label_to_pose_index)
    if args.camera_prior_pose_yaml:
        camera_priors, prior_alignment = build_pose_yaml_prior(manifest, args.camera_prior_pose_yaml)
    elif args.alignment_pose_yaml:
        camera_priors, prior_alignment = build_colmap_prior_from_alignment_poses(
            manifest,
            colmap_images,
            args.alignment_pose_yaml,
            args.alignment_pnp_summary_tsv,
            args.max_anchor_median_error_px,
            args.min_anchor_solved_views)
    else:
        camera_priors, prior_alignment = build_colmap_prior(
            manifest, colmap_images, args.bridge_pose_yaml, anchor_labels, label_to_index)
    bridge_prior_overrides, camera_priors = apply_bridge_prior_overrides(
        camera_priors,
        manifest,
        args.bridge_pose_yaml,
        args.bridge_prior_override_labels,
        label_to_index)
    observations_by_frame_all, observations_by_camera_all = build_observations(dataset)
    observations_by_frame, observations_by_camera, used_camera = filter_observations_by_camera(
        observations_by_frame_all,
        observations_by_camera_all,
        args.min_camera_observations_for_use)
    observations_by_frame_before_gate = observations_by_frame
    pnp_views = load_pnp_views(
        args.pnp_views,
        args.max_pnp_median_error_px,
        dataset["imagesets"],
        dataset["camera_count"])
    tower_poses, frame_pnp_quality = initialize_tower_poses_from_pnp(
        pnp_views, camera_priors, len(dataset["imagesets"]), args.min_frame_pnp_votes)

    deltas = [np.eye(4, dtype=np.float64) for _ in range(dataset["camera_count"])]
    intrinsics_dim = intrinsics_delta_dimension(args.intrinsics_refine_mode)
    shared_intrinsics_delta = (
        np.zeros(intrinsics_dim, dtype=np.float64)
        if args.intrinsics_refine_mode == "shared_fxfy"
        else None
    )
    per_camera_intrinsics_delta = (
        [np.zeros(intrinsics_dim, dtype=np.float64) for _ in range(dataset["camera_count"])]
        if args.intrinsics_refine_mode.startswith("per_camera")
        else None
    )
    tower_face_width_delta_m = 0.0
    tower_face_pose_deltas = [
        np.eye(4, dtype=np.float64)
        for _ in range(args.tower_face_count)
    ]

    def clamp_tower_face_width_delta(delta_m):
        if args.tower_face_width_initial_m <= 0:
            return float(delta_m)
        minimum = args.tower_face_width_min_m - args.tower_face_width_initial_m
        maximum = args.tower_face_width_max_m - args.tower_face_width_initial_m
        return float(np.clip(float(delta_m), minimum, maximum))

    def current_camera_poses():
        return [deltas[i] @ camera_priors[i] for i in range(dataset["camera_count"])]

    def current_intrinsics():
        return apply_intrinsics_refinement(
            base_intrinsics,
            args.intrinsics_refine_mode,
            shared_intrinsics_delta,
            per_camera_intrinsics_delta)

    def current_tower_face_pose_deltas():
        return tower_face_pose_deltas if args.optimize_tower_face_poses else None

    observations_by_frame, observations_by_camera, observation_gate = filter_observations_by_projection_gate(
        observations_by_frame,
        dataset["camera_count"],
        current_camera_poses(),
        tower_poses,
        current_intrinsics(),
        args.observation_residual_gate_px,
        tower_face_width_delta_m,
        current_tower_face_pose_deltas(),
        args.tower_point_model,
        tower_face_base_poses,
        tower_layout)

    active_camera = [
        len(observations_by_camera[i]) >= args.min_camera_observations_for_delta
        for i in range(dataset["camera_count"])
    ]

    def optimizer_residuals(
            observations,
            camera_poses,
            local_tower_poses,
            local_intrinsics,
            tower_width_delta=None,
            local_face_pose_deltas=None):
        width_delta = tower_face_width_delta_m if tower_width_delta is None else float(tower_width_delta)
        face_pose_deltas = (
            current_tower_face_pose_deltas()
            if local_face_pose_deltas is None
            else local_face_pose_deltas
        )
        return clip_residual_norms(
            projection_residuals(
                observations,
                camera_poses,
                local_tower_poses,
                local_intrinsics,
                width_delta,
                face_pose_deltas,
                args.tower_point_model,
                tower_face_base_poses,
                tower_layout),
            args.optimizer_residual_clip_px)

    before = summarize_residuals(
        projection_residuals(
            [obs for frame in observations_by_frame for obs in frame],
            current_camera_poses(),
            tower_poses,
            current_intrinsics(),
            tower_face_width_delta_m,
            current_tower_face_pose_deltas(),
            args.tower_point_model,
            tower_face_base_poses,
            tower_layout))
    raw_before = summarize_residuals(
        projection_residuals(
            [obs for frame in observations_by_frame_before_gate for obs in frame],
            current_camera_poses(),
            tower_poses,
            current_intrinsics(),
            tower_face_width_delta_m,
            current_tower_face_pose_deltas(),
            args.tower_point_model,
            tower_face_base_poses,
            tower_layout))

    sigma_r = args.delta_rotation_sigma_deg * math.pi / 180.0
    sigma_t = args.delta_translation_sigma_m
    focal_sigma = args.intrinsics_focal_sigma_frac
    principal_sigma = args.intrinsics_principal_sigma_px
    distortion_sigma = args.intrinsics_distortion_sigma
    camera_delta_max_rotation_step = args.camera_delta_max_rotation_step_deg * math.pi / 180.0
    tower_pose_max_rotation_step = args.tower_pose_max_rotation_step_deg * math.pi / 180.0
    face_pose_sigma_r = args.tower_face_pose_rotation_sigma_deg * math.pi / 180.0
    face_pose_sigma_t = args.tower_face_pose_translation_sigma_m
    face_pose_max_rotation_step = args.tower_face_pose_max_rotation_step_deg * math.pi / 180.0

    def intrinsics_prior_residual(delta):
        delta = np.asarray(delta, dtype=np.float64)
        if delta.size == 0:
            return np.asarray([], dtype=np.float64)
        residual = [
            delta[0] / focal_sigma,
            delta[1] / focal_sigma,
        ]
        if delta.size >= 4:
            residual.extend([
                delta[2] / principal_sigma,
                delta[3] / principal_sigma,
            ])
        if delta.size >= 9:
            residual.extend((delta[4:9] / distortion_sigma).tolist())
        return np.asarray(residual, dtype=np.float64)

    def intrinsics_eps_and_step(dim):
        eps = [1e-5, 1e-5]
        max_step = [args.intrinsics_max_focal_step_frac, args.intrinsics_max_focal_step_frac]
        if dim >= 4:
            eps.extend([1e-3, 1e-3])
            max_step.extend([args.intrinsics_max_principal_step_px, args.intrinsics_max_principal_step_px])
        if dim >= 9:
            eps.extend([1e-5] * 5)
            max_step.extend([args.intrinsics_max_distortion_step] * 5)
        return np.asarray(eps[:dim], dtype=np.float64), np.asarray(max_step[:dim], dtype=np.float64)

    def run_outer_optimization(outer_iterations):
        nonlocal shared_intrinsics_delta, tower_face_width_delta_m
        for _outer in range(max(0, int(outer_iterations))):
            camera_poses = current_camera_poses()
            intrinsics = current_intrinsics()
            for frame_idx, observations in enumerate(observations_by_frame):
                if tower_poses[frame_idx] is None or len(observations) < 4:
                    continue

                def frame_residual(T, obs=observations, frame_idx=frame_idx):
                    local_tower_poses = list(tower_poses)
                    local_tower_poses[frame_idx] = T
                    return optimizer_residuals(obs, camera_poses, local_tower_poses, intrinsics)

                tower_poses[frame_idx] = optimize_block(
                    tower_poses[frame_idx],
                    frame_residual,
                    args.block_iterations,
                    damping=1e-3,
                    max_rotation_step=tower_pose_max_rotation_step,
                    max_translation_step=args.tower_pose_max_translation_step_m)

            for cam_idx, observations in enumerate(observations_by_camera):
                if not active_camera[cam_idx]:
                    continue

                def camera_residual(delta, cam_idx=cam_idx, obs=observations):
                    poses = current_camera_poses()
                    poses[cam_idx] = delta @ camera_priors[cam_idx]
                    reproj = optimizer_residuals(obs, poses, tower_poses, current_intrinsics())
                    xi = se3_log_approx(delta)
                    prior = np.concatenate([xi[:3] / sigma_r, xi[3:] / sigma_t])
                    return np.concatenate([reproj, prior])

                deltas[cam_idx] = optimize_block(
                    deltas[cam_idx],
                    camera_residual,
                    args.block_iterations,
                    damping=1e-3,
                    max_rotation_step=camera_delta_max_rotation_step,
                    max_translation_step=args.camera_delta_max_translation_step_m)

            if args.intrinsics_refine_mode == "shared_fxfy":
                shared_observations = [
                    obs
                    for frame in observations_by_frame
                    for obs in frame
                    if active_camera[observation_camera_index(obs)]
                ]
                eps, max_step = intrinsics_eps_and_step(intrinsics_dim)

                def shared_intrinsics_residual(delta, obs=shared_observations):
                    local_intrinsics = apply_intrinsics_refinement(
                        base_intrinsics,
                        args.intrinsics_refine_mode,
                        delta,
                        per_camera_intrinsics_delta)
                    reproj = optimizer_residuals(obs, current_camera_poses(), tower_poses, local_intrinsics)
                    return np.concatenate([reproj, intrinsics_prior_residual(delta)])

                shared_intrinsics_delta = optimize_vector(
                    shared_intrinsics_delta,
                    shared_intrinsics_residual,
                    args.intrinsics_block_iterations,
                    damping=1e-2,
                    eps=eps,
                    max_step=max_step)
                shared_intrinsics_delta = clamp_intrinsics_delta_to_total_bounds(
                    shared_intrinsics_delta,
                    args.intrinsics_max_total_focal_delta_frac,
                    args.intrinsics_max_total_principal_delta_px,
                    args.intrinsics_max_total_distortion_delta)
            elif args.intrinsics_refine_mode.startswith("per_camera"):
                eps, max_step = intrinsics_eps_and_step(intrinsics_dim)
                for cam_idx, observations in enumerate(observations_by_camera):
                    if not active_camera[cam_idx]:
                        continue

                    def camera_intrinsics_residual(delta, cam_idx=cam_idx, obs=observations):
                        local_deltas = list(per_camera_intrinsics_delta)
                        local_deltas[cam_idx] = delta
                        local_intrinsics = apply_intrinsics_refinement(
                            base_intrinsics,
                            args.intrinsics_refine_mode,
                            shared_intrinsics_delta,
                            local_deltas)
                        reproj = optimizer_residuals(obs, current_camera_poses(), tower_poses, local_intrinsics)
                        return np.concatenate([reproj, intrinsics_prior_residual(delta)])

                    per_camera_intrinsics_delta[cam_idx] = optimize_vector(
                        per_camera_intrinsics_delta[cam_idx],
                        camera_intrinsics_residual,
                        args.intrinsics_block_iterations,
                        damping=1e-2,
                        eps=eps,
                        max_step=max_step)
                    per_camera_intrinsics_delta[cam_idx] = clamp_intrinsics_delta_to_total_bounds(
                        per_camera_intrinsics_delta[cam_idx],
                        args.intrinsics_max_total_focal_delta_frac,
                        args.intrinsics_max_total_principal_delta_px,
                        args.intrinsics_max_total_distortion_delta)

            if args.optimize_tower_face_width:
                width_observations = [
                    obs
                    for frame in observations_by_frame
                    for obs in frame
                    if active_camera[observation_camera_index(obs)]
                ]

                def tower_face_width_residual(delta, obs=width_observations):
                    delta_m = clamp_tower_face_width_delta(float(delta[0]))
                    reproj = optimizer_residuals(
                        obs,
                        current_camera_poses(),
                        tower_poses,
                        current_intrinsics(),
                        delta_m)
                    prior = np.asarray([delta_m / args.tower_face_width_sigma_m], dtype=np.float64)
                    return np.concatenate([reproj, prior])

                optimized_delta = optimize_vector(
                    np.asarray([tower_face_width_delta_m], dtype=np.float64),
                    tower_face_width_residual,
                    args.block_iterations,
                    damping=1e-2,
                    eps=np.asarray([1e-4], dtype=np.float64),
                    max_step=np.asarray([args.tower_face_width_max_step_m], dtype=np.float64))
                tower_face_width_delta_m = clamp_tower_face_width_delta(float(optimized_delta[0]))

            if args.optimize_tower_face_poses:
                observations_by_face = observations_by_face_from_frames(
                    observations_by_frame,
                    args.tower_face_count)
                for face_idx, face_observations in enumerate(observations_by_face):
                    if len(face_observations) < args.tower_face_pose_min_observations:
                        continue

                    def face_pose_residual(delta, face_idx=face_idx, obs=face_observations):
                        local_face_pose_deltas = list(tower_face_pose_deltas)
                        local_face_pose_deltas[face_idx] = delta
                        reproj = optimizer_residuals(
                            obs,
                            current_camera_poses(),
                            tower_poses,
                            current_intrinsics(),
                            tower_face_width_delta_m,
                            local_face_pose_deltas)
                        xi = se3_log_approx(delta)
                        prior = np.concatenate([
                            xi[:3] / face_pose_sigma_r,
                            xi[3:] / face_pose_sigma_t,
                        ])
                        return np.concatenate([reproj, prior])

                    tower_face_pose_deltas[face_idx] = optimize_block(
                        tower_face_pose_deltas[face_idx],
                        face_pose_residual,
                        args.block_iterations,
                        damping=1e-2,
                        max_rotation_step=face_pose_max_rotation_step,
                        max_translation_step=args.tower_face_pose_max_translation_step_m)

    run_outer_optimization(args.outer_iterations)

    observations_by_frame, observations_by_camera, post_refine_observation_gate = (
        filter_post_refine_observations_by_projection_gate(
            observations_by_frame,
            dataset["camera_count"],
            current_camera_poses(),
            tower_poses,
            current_intrinsics(),
            args.post_refine_observation_residual_gate_px,
            args.post_refine_outer_iterations,
            tower_face_width_delta_m,
            current_tower_face_pose_deltas(),
            args.tower_point_model,
            tower_face_base_poses,
            tower_layout))
    if post_refine_observation_gate["enabled"]:
        active_camera = [
            len(observations_by_camera[i]) >= args.min_camera_observations_for_delta
            for i in range(dataset["camera_count"])
        ]
        run_outer_optimization(args.post_refine_outer_iterations)

    refined_camera_poses = current_camera_poses()
    refined_intrinsics = current_intrinsics()
    after = summarize_residuals(
        projection_residuals(
            [obs for frame in observations_by_frame for obs in frame],
            refined_camera_poses,
            tower_poses,
            refined_intrinsics,
            tower_face_width_delta_m,
            current_tower_face_pose_deltas(),
            args.tower_point_model,
            tower_face_base_poses,
            tower_layout))
    raw_after = summarize_residuals(
        projection_residuals(
            [obs for frame in observations_by_frame_before_gate for obs in frame],
            refined_camera_poses,
            tower_poses,
            refined_intrinsics,
            tower_face_width_delta_m,
            current_tower_face_pose_deltas(),
            args.tower_point_model,
            tower_face_base_poses,
            tower_layout))

    write_pose_yaml(args.output_dir / "camera_tr_rig_prior.yaml", camera_priors)
    write_pose_yaml(args.output_dir / "camera_tr_rig_delta_refined.yaml", refined_camera_poses)
    write_pose_yaml(args.output_dir / "camera_delta_from_prior.yaml", deltas)
    write_pose_yaml(args.output_dir / "rig_tr_global.yaml", tower_poses)
    write_pose_yaml(args.output_dir / "tower_face_pose_delta.yaml", tower_face_pose_deltas)
    write_pose_yaml(
        args.output_dir / "tower_face_pose.yaml",
        [
            tower_face_pose_deltas[i] @ tower_face_base_poses[i]
            for i in range(args.tower_face_count)
        ])

    camera_reprojection_rows = summarize_camera_reprojection(
        manifest,
        observations_by_camera,
        camera_priors,
        refined_camera_poses,
        tower_poses,
        base_intrinsics,
        refined_intrinsics,
        tower_face_width_delta_m,
        current_tower_face_pose_deltas(),
        args.tower_point_model,
        tower_face_base_poses,
        tower_layout)
    accepted_camera_poses, accepted_refined_camera_ids = accepted_refined_poses(
        camera_priors,
        refined_camera_poses,
        camera_reprojection_rows,
        active_camera,
        used_camera,
        args)
    camera_acceptance_rows = summarize_camera_acceptance(
        manifest,
        camera_reprojection_rows,
        active_camera,
        used_camera,
        accepted_refined_camera_ids,
        args)
    pose_refined_camera_ids = [
        row["camera_id"]
        for row in camera_acceptance_rows
        if row["output_pose"] == "refined"
    ]
    accepted_intrinsics, accepted_refined_intrinsics_ids, camera_intrinsics_rows = accepted_refined_intrinsics(
        manifest,
        base_intrinsics,
        refined_intrinsics,
        pose_refined_camera_ids,
        args,
        args.intrinsics_refine_mode)
    accepted_camera_reprojection_rows = summarize_camera_reprojection(
        manifest,
        observations_by_camera,
        camera_priors,
        accepted_camera_poses,
        tower_poses,
        base_intrinsics,
        accepted_intrinsics,
        tower_face_width_delta_m,
        current_tower_face_pose_deltas(),
        args.tower_point_model,
        tower_face_base_poses,
        tower_layout)
    accepted_after = summarize_residuals(
        projection_residuals(
            [obs for frame in observations_by_frame for obs in frame],
            accepted_camera_poses,
            tower_poses,
            accepted_intrinsics,
            tower_face_width_delta_m,
            current_tower_face_pose_deltas(),
            args.tower_point_model,
            tower_face_base_poses,
            tower_layout))
    raw_accepted_after = summarize_residuals(
        projection_residuals(
            [obs for frame in observations_by_frame_before_gate for obs in frame],
            accepted_camera_poses,
            tower_poses,
            accepted_intrinsics,
            tower_face_width_delta_m,
            current_tower_face_pose_deltas(),
            args.tower_point_model,
            tower_face_base_poses,
            tower_layout))
    write_pose_yaml(args.output_dir / "camera_tr_rig_delta_refined_accepted.yaml", accepted_camera_poses)
    write_intrinsics_dir(args.output_dir / "intrinsics_prior", manifest, base_intrinsics)
    write_intrinsics_dir(args.output_dir / "intrinsics_refined", manifest, refined_intrinsics)
    write_intrinsics_dir(args.output_dir / "intrinsics_refined_accepted", manifest, accepted_intrinsics)
    write_observation_residuals(
        diagnostics_dir / "observation_residuals.tsv",
        manifest,
        observations_by_frame_before_gate,
        observations_by_frame,
        refined_camera_poses,
        tower_poses,
        refined_intrinsics,
        tower_face_width_delta_m,
        current_tower_face_pose_deltas(),
        args.tower_point_model,
        tower_face_base_poses,
        tower_layout)
    write_observation_residuals(
        diagnostics_dir / "observation_residuals_accepted.tsv",
        manifest,
        observations_by_frame_before_gate,
        observations_by_frame,
        accepted_camera_poses,
        tower_poses,
        accepted_intrinsics,
        tower_face_width_delta_m,
        current_tower_face_pose_deltas(),
        args.tower_point_model,
        tower_face_base_poses,
        tower_layout)

    write_camera_reprojection_tsv(
        diagnostics_dir / "camera_reprojection.tsv",
        camera_reprojection_rows)
    write_camera_reprojection_tsv(
        diagnostics_dir / "camera_reprojection_accepted.tsv",
        accepted_camera_reprojection_rows)

    with (diagnostics_dir / "camera_delta.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "camera_index", "camera_id", "active", "used",
                "observation_count", "used_observation_count",
                "delta_rotation_deg", "delta_translation_m",
                "colmap_tracks",
            ])
        writer.writeheader()
        for idx, row in enumerate(manifest):
            stats = delta_pose_stats(deltas[idx])
            colmap_image = colmap_images.get(row["camera_id"], {})
            writer.writerow({
                "camera_index": idx,
                "camera_id": row["camera_id"],
                "active": "yes" if active_camera[idx] else "no",
                "used": "yes" if used_camera[idx] else "no",
                "observation_count": len(observations_by_camera_all[idx]),
                "used_observation_count": len(observations_by_camera[idx]),
                "delta_rotation_deg": f"{stats['rotation_deg']:.8g}",
                "delta_translation_m": f"{stats['translation_m']:.8g}",
                "colmap_tracks": colmap_image.get("triangulated_point_count", ""),
            })

    with (diagnostics_dir / "camera_intrinsics.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "camera_index", "camera_id", "decision", "output_intrinsics", "reason",
                "base_fx", "base_fy", "base_cx", "base_cy",
                "base_k1", "base_k2", "base_k3", "base_p1", "base_p2",
                "refined_fx", "refined_fy", "refined_cx", "refined_cy",
                "refined_k1", "refined_k2", "refined_k3", "refined_p1", "refined_p2",
                "output_fx", "output_fy", "output_cx", "output_cy",
                "output_k1", "output_k2", "output_k3", "output_p1", "output_p2",
                "fx_delta_frac", "fy_delta_frac", "max_abs_focal_delta_frac",
                "cx_delta_px", "cy_delta_px", "principal_delta_px",
                "k1_delta", "k2_delta", "k3_delta", "p1_delta", "p2_delta",
                "max_abs_distortion_delta",
            ])
        writer.writeheader()
        for row in camera_intrinsics_rows:
            writer.writerow(row)

    observations_by_face_final = observations_by_face_from_frames(
        observations_by_frame,
        args.tower_face_count)
    tower_face_pose_rows = []
    with (diagnostics_dir / "tower_face_pose_delta.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "face_id", "active", "used_observation_count",
                "delta_rotation_deg", "delta_translation_m",
            ])
        writer.writeheader()
        for face_idx, face_delta in enumerate(tower_face_pose_deltas):
            stats = delta_pose_stats(face_delta)
            active_face = (
                args.optimize_tower_face_poses
                and len(observations_by_face_final[face_idx]) >= args.tower_face_pose_min_observations
            )
            writer.writerow({
                "face_id": face_idx,
                "active": "yes" if active_face else "no",
                "used_observation_count": len(observations_by_face_final[face_idx]),
                "delta_rotation_deg": f"{stats['rotation_deg']:.8g}",
                "delta_translation_m": f"{stats['translation_m']:.8g}",
            })
            tower_face_pose_rows.append({
                "face_id": face_idx,
                "active": bool(active_face),
                "used_observation_count": len(observations_by_face_final[face_idx]),
                "delta_rotation_deg": float(stats["rotation_deg"]),
                "delta_translation_m": float(stats["translation_m"]),
            })

    with (diagnostics_dir / "camera_acceptance.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "camera_index", "camera_id", "decision", "output_pose", "reason",
                "active_delta", "used_observation", "observation_count",
                "after_median_px", "after_under_300_fraction",
            ])
        writer.writeheader()
        for row in camera_acceptance_rows:
            writer.writerow(row)

    with (diagnostics_dir / "frame_quality.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "frame_index", "filename", "feature_count",
                "pnp_support", "pnp_vote_count", "pnp_median_error_px",
                "pnp_pose_average", "active",
            ])
        writer.writeheader()
        for idx, imageset in enumerate(dataset["imagesets"]):
            pnp_quality = frame_pnp_quality[idx]
            writer.writerow({
                "frame_index": idx,
                "filename": imageset["filename"],
                "feature_count": len(observations_by_frame[idx]),
                "pnp_support": pnp_quality["pnp_vote_count"],
                "pnp_vote_count": pnp_quality["pnp_vote_count"],
                "pnp_median_error_px": pnp_quality["pnp_median_error_px"],
                "pnp_pose_average": pnp_quality["pnp_pose_average"],
                "active": "yes" if tower_poses[idx] is not None else "no",
            })

    inactive_prior_only_camera_ids = [
        manifest[i]["camera_id"] for i, active in enumerate(active_camera) if not active
    ]
    output_prior_pose_camera_ids = [
        row["camera_id"]
        for row in camera_acceptance_rows
        if row["output_pose"] == "prior"
    ]

    summary = {
        "inputs": {
            "dataset": str(args.dataset),
            "manifest": str(args.manifest),
            "pnp_views": str(args.pnp_views),
            "outer_colmap_images_txt": str(args.outer_colmap_images_txt) if args.outer_colmap_images_txt else "",
            "bridge_pose_yaml": str(args.bridge_pose_yaml),
            "camera_prior_pose_yaml": str(args.camera_prior_pose_yaml) if args.camera_prior_pose_yaml else "",
            "intrinsics_dir": str(args.intrinsics_dir) if args.intrinsics_dir else "",
        },
        "settings": {
            "intrinsics_mode": args.intrinsics_mode,
            "intrinsics_refine_mode": args.intrinsics_refine_mode,
            "tower_point_model": args.tower_point_model,
            "tower_first_tag_id": args.tower_first_tag_id,
            "tower_face_id_stride": args.tower_face_id_stride,
            "tower_tag_columns": args.tower_tag_columns,
            "tower_tag_rows": args.tower_tag_rows,
            "tower_tag_size_m": args.tower_tag_size_m,
            "tower_tag_spacing_m": args.tower_tag_spacing_m,
            "tower_tag_rotation_degrees": args.tower_tag_rotation_degrees,
            "outer_iterations": args.outer_iterations,
            "block_iterations": args.block_iterations,
            "intrinsics_block_iterations": args.intrinsics_block_iterations,
            "min_camera_observations_for_use": args.min_camera_observations_for_use,
            "observation_residual_gate_px": args.observation_residual_gate_px,
            "post_refine_observation_residual_gate_px": args.post_refine_observation_residual_gate_px,
            "post_refine_outer_iterations": args.post_refine_outer_iterations,
            "optimizer_residual_clip_px": args.optimizer_residual_clip_px,
            "accept_camera_median_px": args.accept_camera_median_px,
            "accept_camera_p90_px": args.accept_camera_p90_px,
            "accept_camera_min_under_300_fraction": args.accept_camera_min_under_300_fraction,
            "accept_camera_max_delta_translation_m": args.accept_camera_max_delta_translation_m,
            "accept_camera_max_delta_rotation_deg": args.accept_camera_max_delta_rotation_deg,
            "delta_translation_sigma_m": args.delta_translation_sigma_m,
            "delta_rotation_sigma_deg": args.delta_rotation_sigma_deg,
            "camera_delta_max_rotation_step_deg": args.camera_delta_max_rotation_step_deg,
            "camera_delta_max_translation_step_m": args.camera_delta_max_translation_step_m,
            "tower_pose_max_rotation_step_deg": args.tower_pose_max_rotation_step_deg,
            "tower_pose_max_translation_step_m": args.tower_pose_max_translation_step_m,
            "intrinsics_focal_sigma_frac": args.intrinsics_focal_sigma_frac,
            "intrinsics_principal_sigma_px": args.intrinsics_principal_sigma_px,
            "intrinsics_distortion_sigma": args.intrinsics_distortion_sigma,
            "intrinsics_max_focal_step_frac": args.intrinsics_max_focal_step_frac,
            "intrinsics_max_principal_step_px": args.intrinsics_max_principal_step_px,
            "intrinsics_max_distortion_step": args.intrinsics_max_distortion_step,
            "intrinsics_max_total_focal_delta_frac": args.intrinsics_max_total_focal_delta_frac,
            "intrinsics_max_total_principal_delta_px": args.intrinsics_max_total_principal_delta_px,
            "intrinsics_max_total_distortion_delta": args.intrinsics_max_total_distortion_delta,
            "accept_camera_max_intrinsic_focal_delta_frac": args.accept_camera_max_intrinsic_focal_delta_frac,
            "accept_camera_max_intrinsic_principal_delta_px": args.accept_camera_max_intrinsic_principal_delta_px,
            "accept_camera_max_intrinsic_distortion_delta": args.accept_camera_max_intrinsic_distortion_delta,
            "optimize_tower_face_poses": bool(args.optimize_tower_face_poses),
            "tower_face_count": args.tower_face_count,
            "tower_face_pose_min_observations": args.tower_face_pose_min_observations,
            "tower_face_pose_rotation_sigma_deg": args.tower_face_pose_rotation_sigma_deg,
            "tower_face_pose_translation_sigma_m": args.tower_face_pose_translation_sigma_m,
            "tower_face_pose_max_rotation_step_deg": args.tower_face_pose_max_rotation_step_deg,
            "tower_face_pose_max_translation_step_m": args.tower_face_pose_max_translation_step_m,
            "allow_ungated_accepted_output": bool(args.allow_ungated_accepted_output),
            "anchor_label_to_pose_index": args.anchor_label_to_pose_index,
            "bridge_prior_override_labels": args.bridge_prior_override_labels,
            "pnp_pose_averaging": PNP_POSE_AVERAGING_MODE,
            "pnp_pose_averaging_error_floor_px": PNP_POSE_AVERAGING_ERROR_FLOOR_PX,
            "optimizer_backtracking_step_scales": list(BACKTRACKING_STEP_SCALES),
        },
        "prior_alignment": prior_alignment,
        "bridge_prior_overrides": bridge_prior_overrides,
        "observation_gate": observation_gate,
        "post_refine_observation_gate": post_refine_observation_gate,
        "frames": {
            "total": len(dataset["imagesets"]),
            "active": int(sum(p is not None for p in tower_poses)),
        },
        "cameras": {
            "total": dataset["camera_count"],
            "active_delta": int(sum(active_camera)),
            "used_observation": int(sum(used_camera)),
            "excluded_from_observation_residual": [
                manifest[i]["camera_id"]
                for i, used in enumerate(used_camera)
                if not used
            ],
            "prior_only": output_prior_pose_camera_ids,
            "inactive_prior_only": inactive_prior_only_camera_ids,
            "accepted_refined": accepted_refined_camera_ids,
            "accepted_refined_count": len(accepted_refined_camera_ids),
            "rejected_to_prior": [
                row["camera_id"]
                for row in camera_acceptance_rows
                if row["decision"] == "rejected_to_prior"
            ],
            "output_prior_pose": output_prior_pose_camera_ids,
            "acceptance_enabled": args.accept_camera_median_px > 0,
        },
        "intrinsics": {
            "refine_mode": args.intrinsics_refine_mode,
            "prior_dir": str(args.output_dir / "intrinsics_prior"),
            "refined_dir": str(args.output_dir / "intrinsics_refined"),
            "accepted_dir": str(args.output_dir / "intrinsics_refined_accepted"),
            "accepted_refined": accepted_refined_intrinsics_ids,
            **summarize_intrinsics_acceptance(camera_intrinsics_rows),
        },
        "tower_geometry": {
            "optimize_face_width": bool(args.optimize_tower_face_width),
            "face_width_initial_m": (
                float(args.tower_face_width_initial_m)
                if args.tower_face_width_initial_m > 0
                else None
            ),
            "face_width_delta_m": float(tower_face_width_delta_m),
            "face_width_final_m": (
                float(args.tower_face_width_initial_m + tower_face_width_delta_m)
                if args.tower_face_width_initial_m > 0
                else None
            ),
            "face_width_sigma_m": float(args.tower_face_width_sigma_m),
            "face_width_min_m": float(args.tower_face_width_min_m),
            "face_width_max_m": float(args.tower_face_width_max_m),
            "face_width_max_step_m": float(args.tower_face_width_max_step_m),
            "optimize_face_poses": bool(args.optimize_tower_face_poses),
            "face_pose_model": args.tower_point_model,
            "face_base_pose_initialization": tower_face_base_pose_rows,
            "face_pose_delta_yaml": str(args.output_dir / "tower_face_pose_delta.yaml"),
            "face_pose_yaml": str(args.output_dir / "tower_face_pose.yaml"),
            "face_pose_delta_tsv": str(diagnostics_dir / "tower_face_pose_delta.tsv"),
            "face_pose_delta_by_face": tower_face_pose_rows,
            "max_face_pose_delta_rotation_deg": max(
                (row["delta_rotation_deg"] for row in tower_face_pose_rows),
                default=0.0),
            "max_face_pose_delta_translation_m": max(
                (row["delta_translation_m"] for row in tower_face_pose_rows),
                default=0.0),
        },
	        "diagnostics": {
            "camera_reprojection_tsv": str(diagnostics_dir / "camera_reprojection.tsv"),
            "camera_reprojection_semantics": "candidate refined pose/intrinsics before acceptance fallback",
            "camera_reprojection_accepted_tsv": str(diagnostics_dir / "camera_reprojection_accepted.tsv"),
            "camera_reprojection_accepted_semantics": "final accepted-output pose/intrinsics after fallback",
            "observation_residuals_tsv": str(diagnostics_dir / "observation_residuals.tsv"),
            "observation_residuals_semantics": "candidate refined pose/intrinsics before acceptance fallback",
            "observation_residuals_accepted_tsv": str(diagnostics_dir / "observation_residuals_accepted.tsv"),
            "observation_residuals_accepted_semantics": "final accepted-output pose/intrinsics after fallback",
        },
        "residual_before": before,
        "residual_after": after,
        "residual_after_output_accepted": accepted_after,
        "raw_residual_before": raw_before,
        "raw_residual_after": raw_after,
        "raw_residual_after_output_accepted": raw_accepted_after,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
