#!/usr/bin/env python3
"""Estimate Seeker camera-IMU rotation from full-board tagged-pattern features."""

import argparse
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import yaml


NS_PER_S = 1_000_000_000
CAMERA_TO_MCAP = {
    "cam0": "up",
    "cam1": "down",
    "cam2": "down",
    "cam3": "up",
}
CAMERA_LABELS = {
    "cam0": "left_up",
    "cam1": "left_down",
    "cam2": "right_down",
    "cam3": "right_up",
}


def read_u32(data, offset):
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def read_i32(data, offset):
    return struct.unpack_from(">i", data, offset)[0], offset + 4


def read_f32(data, offset):
    return struct.unpack_from("<f", data, offset)[0], offset + 4


def parse_timestamp_from_filename(filename):
    match = re.search(r"_src(\d+)_(\d+)\.", filename)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def read_dataset(path):
    path = Path(path)
    data = path.read_bytes()
    offset = 0
    if data[:10] != b"calib_data":
        raise ValueError(f"{path} is not a calib_data dataset")
    offset += 10
    version, offset = read_u32(data, offset)
    if version not in (0, 1):
        raise ValueError(f"Unsupported dataset version {version} in {path}")
    num_cameras, offset = read_u32(data, offset)
    image_sizes = []
    for _ in range(num_cameras):
        width, offset = read_u32(data, offset)
        height, offset = read_u32(data, offset)
        image_sizes.append([width, height])

    num_imagesets, offset = read_u32(data, offset)
    imagesets = []
    for image_index in range(num_imagesets):
        filename_len, offset = read_u32(data, offset)
        filename = data[offset:offset + filename_len].decode("utf-8", errors="replace")
        offset += filename_len
        cameras = []
        for _camera_index in range(num_cameras):
            num_features, offset = read_u32(data, offset)
            features = []
            for _ in range(num_features):
                x, offset = read_f32(data, offset)
                y, offset = read_f32(data, offset)
                feature_id, offset = read_i32(data, offset)
                features.append((feature_id, x, y))
            cameras.append(features)
        src, stamp_ns = parse_timestamp_from_filename(filename)
        imagesets.append({
            "index": image_index,
            "filename": filename,
            "src": src,
            "timestamp_ns": stamp_ns,
            "cameras": cameras,
        })

    geometries = []
    num_geometries, offset = read_u32(data, offset)
    for _ in range(num_geometries):
        cell_length, offset = read_f32(data, offset)
        feature_id_to_position = {}
        count_2d, offset = read_u32(data, offset)
        for _ in range(count_2d):
            feature_id, offset = read_i32(data, offset)
            gx, offset = read_i32(data, offset)
            gy, offset = read_i32(data, offset)
            feature_id_to_position[feature_id] = (gx, gy)
        feature_id_to_position3d = {}
        if version >= 1:
            count_3d, offset = read_u32(data, offset)
            for _ in range(count_3d):
                feature_id, offset = read_i32(data, offset)
                x, offset = read_f32(data, offset)
                y, offset = read_f32(data, offset)
                z, offset = read_f32(data, offset)
                feature_id_to_position3d[feature_id] = (x, y, z)
        geometries.append({
            "cell_length_in_meters": cell_length,
            "feature_id_to_position": feature_id_to_position,
            "feature_id_to_position3d": feature_id_to_position3d,
        })
    if offset != len(data):
        raise ValueError(f"{path} has trailing bytes: parsed={offset}, size={len(data)}")
    return {
        "version": version,
        "num_cameras": num_cameras,
        "image_sizes": image_sizes,
        "imagesets": imagesets,
        "geometries": geometries,
    }


def object_point_for_feature(feature_id, geometry):
    pos3d = geometry["feature_id_to_position3d"].get(feature_id)
    if pos3d is not None:
        return pos3d
    pos2d = geometry["feature_id_to_position"].get(feature_id)
    if pos2d is None:
        return None
    cell = geometry["cell_length_in_meters"]
    return (cell * pos2d[0], cell * pos2d[1], 0.0)


def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)


def so3_exp(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    axis = w / theta
    K = skew(axis)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def so3_log(R):
    cos_theta = 0.5 * (float(np.trace(R)) - 1.0)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    if abs(math.pi - theta) < 1e-5:
        U, _, Vt = np.linalg.svd(R - np.eye(3))
        axis = Vt[-1]
        return theta * axis / max(1e-12, np.linalg.norm(axis))
    return theta / (2.0 * math.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def rot_to_quat_xyzw(R):
    qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    qy = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    qz = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = np.asarray([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


def quat_to_rot_xyzw(q):
    x, y, z, w = [float(v) for v in q]
    return np.asarray([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=float)


def average_rotations(rotations):
    quats = []
    ref = None
    for R in rotations:
        q = rot_to_quat_xyzw(R)
        if ref is None:
            ref = q
        elif float(q @ ref) < 0:
            q = -q
        quats.append(q)
    A = np.zeros((4, 4), dtype=float)
    for q in quats:
        A += np.outer(q, q)
    vals, vecs = np.linalg.eigh(A)
    q = vecs[:, int(np.argmax(vals))]
    if ref is not None and float(q @ ref) < 0:
        q = -q
    q = q / np.linalg.norm(q)
    return quat_to_rot_xyzw(q), q


def rotation_angle(R):
    c = 0.5 * (float(np.trace(R)) - 1.0)
    c = min(1.0, max(-1.0, c))
    return math.acos(c)


def load_kb8_prior(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {}
    for cam, entry in data.items():
        if not cam.startswith("cam"):
            continue
        fx, fy, cx, cy = [float(v) for v in entry["intrinsics"]]
        out[cam] = {
            "entry": entry,
            "width": int(entry["resolution"][0]),
            "height": int(entry["resolution"][1]),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k": [float(v) for v in entry["distortion_coeffs"]],
            "T_cam_imu_prior": np.asarray(entry["T_cam_imu"], dtype=float),
        }
    return data, out


def kb8_project(points_cam, intr):
    points_cam = np.asarray(points_cam, dtype=float)
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    theta2 = theta * theta
    k1, k2, k3, k4 = intr["k"]
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
    scale = np.zeros_like(theta_d)
    ok = rho > 1e-12
    scale[ok] = theta_d[ok] / rho[ok]
    mx = scale * x
    my = scale * y
    proj = np.column_stack([
        intr["fx"] * mx + intr["cx"],
        intr["fy"] * my + intr["cy"],
    ])
    valid = np.isfinite(proj).all(axis=1) & (z > 1e-4)
    return proj, valid


def kb8_unproject(pixels, intr):
    pixels = np.asarray(pixels, dtype=float)
    x = (pixels[:, 0] - intr["cx"]) / intr["fx"]
    y = (pixels[:, 1] - intr["cy"]) / intr["fy"]
    rd = np.hypot(x, y)
    theta = rd.copy()
    k1, k2, k3, k4 = intr["k"]
    for _ in range(20):
        theta2 = theta * theta
        f = theta * (1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4) - rd
        df = 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta2**2 + 7.0 * k3 * theta2**3 + 9.0 * k4 * theta2**4
        step = f / np.where(np.abs(df) < 1e-12, 1e-12, df)
        theta -= step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    rays = np.zeros((len(pixels), 3), dtype=float)
    scale = np.zeros_like(rd)
    ok = rd > 1e-12
    scale[ok] = np.sin(theta[ok]) / rd[ok]
    rays[:, 0] = scale * x
    rays[:, 1] = scale * y
    rays[:, 2] = np.cos(theta)
    rays[~ok] = np.asarray([0.0, 0.0, 1.0])
    norm = np.linalg.norm(rays, axis=1)
    return rays / norm[:, None]


def homography_initial_pose(object_xy, image_uv, intr):
    rays = kb8_unproject(image_uv, intr)
    keep = np.abs(rays[:, 2]) > 1e-9
    if int(np.sum(keep)) < 4:
        raise ValueError("Too few rays for homography initialization")
    normalized = rays[keep, :2] / rays[keep, 2:3]
    object_xy = object_xy[keep]
    A = []
    for (X, Y), (u, v) in zip(object_xy, normalized):
        A.append([-X, -Y, -1.0, 0.0, 0.0, 0.0, u * X, u * Y, u])
        A.append([0.0, 0.0, 0.0, -X, -Y, -1.0, v * X, v * Y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    H = Vt[-1].reshape(3, 3)
    if np.linalg.det(H) < 0:
        H = -H
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    scale = 2.0 / max(1e-12, np.linalg.norm(h1) + np.linalg.norm(h2))
    r1 = scale * h1
    r2 = scale * h2
    t = scale * h3
    r3 = np.cross(r1, r2)
    R0 = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    if t[2] < 0:
        R[:, 0:2] *= -1.0
        t *= -1.0
    return R, t


def pose_residual(params, object_xyz, image_uv, intr):
    R = so3_exp(params[:3])
    t = params[3:6]
    points_cam = (R @ object_xyz.T).T + t.reshape(1, 3)
    proj, valid = kb8_project(points_cam, intr)
    residual = proj - image_uv
    residual[~valid] = 1e6
    return residual.reshape(-1)


def refine_pose(R, t, object_xyz, image_uv, intr, max_iterations=20):
    params = np.concatenate([so3_log(R), t.astype(float)])
    damping = 1e-3
    residual = pose_residual(params, object_xyz, image_uv, intr)
    best_cost = float(residual @ residual)
    for _ in range(max_iterations):
        J = np.zeros((residual.size, 6), dtype=float)
        eps = 1e-6
        for j in range(6):
            stepped = params.copy()
            stepped[j] += eps
            J[:, j] = (pose_residual(stepped, object_xyz, image_uv, intr) - residual) / eps
        H = J.T @ J
        g = J.T @ residual
        try:
            delta = -np.linalg.solve(H + damping * np.eye(6), g)
        except np.linalg.LinAlgError:
            break
        candidate = params + delta
        candidate_residual = pose_residual(candidate, object_xyz, image_uv, intr)
        candidate_cost = float(candidate_residual @ candidate_residual)
        if candidate_cost < best_cost:
            params = candidate
            residual = candidate_residual
            best_cost = candidate_cost
            damping = max(1e-8, damping * 0.5)
            if np.linalg.norm(delta) < 1e-10:
                break
        else:
            damping *= 5.0
    return so3_exp(params[:3]), params[3:6], residual.reshape(-1, 2)


def robust_pose_from_observations(object_xyz, image_uv, intr, min_features, max_rmse, max_inlier_px):
    if len(object_xyz) < min_features:
        return None
    R, t = homography_initial_pose(object_xyz[:, :2], image_uv, intr)
    R, t, residual = refine_pose(R, t, object_xyz, image_uv, intr, max_iterations=15)
    errors = np.linalg.norm(residual, axis=1)
    finite = np.isfinite(errors)
    if int(np.sum(finite)) < min_features:
        return None
    med = float(np.median(errors[finite]))
    mad = float(np.median(np.abs(errors[finite] - med)))
    adaptive = med + 4.0 * 1.4826 * mad
    threshold = min(float(max_inlier_px), max(2.0, adaptive))
    inliers = finite & (errors <= threshold)
    if int(np.sum(inliers)) < min_features:
        inliers = finite & (errors <= max_inlier_px)
    if int(np.sum(inliers)) < min_features:
        return None
    R, t, residual = refine_pose(R, t, object_xyz[inliers], image_uv[inliers], intr, max_iterations=15)
    errors = np.linalg.norm(residual, axis=1)
    rmse = float(math.sqrt(np.mean(errors * errors)))
    if not np.isfinite(rmse) or rmse > max_rmse:
        return None
    return {
        "R_cam_board": R,
        "t_cam_board": t,
        "reprojection_rmse_px": rmse,
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_p95_px": float(np.percentile(errors, 95)),
        "features": int(len(object_xyz)),
        "inliers": int(len(errors)),
    }


def align_cdr(offset, alignment):
    return (offset + alignment - 1) & ~(alignment - 1)


def parse_sensor_msgs_imu_cdr(data):
    data = bytes(data)
    if data[:4] == b"\x00\x01\x00\x00":
        endian = "<"
    elif data[:4] == b"\x00\x00\x00\x00":
        endian = ">"
    else:
        endian = "<"
    offset = 4
    sec = struct.unpack_from(endian + "i", data, offset)[0]
    offset += 4
    nanosec = struct.unpack_from(endian + "I", data, offset)[0]
    offset += 4
    frame_len = struct.unpack_from(endian + "I", data, offset)[0]
    offset += 4
    offset += frame_len
    offset = align_cdr(offset, 8)

    def read_doubles(count):
        nonlocal offset
        offset = align_cdr(offset, 8)
        values = struct.unpack_from(endian + "d" * count, data, offset)
        offset += 8 * count
        return values

    read_doubles(4)
    read_doubles(9)
    angular_velocity = read_doubles(3)
    read_doubles(9)
    linear_acceleration = read_doubles(3)
    return {
        "stamp_ns": int(sec) * NS_PER_S + int(nanosec),
        "angular_velocity": angular_velocity,
        "linear_acceleration": linear_acceleration,
    }


def load_imu_gyro(mcap_path, imu_topic):
    from mcap.reader import make_reader

    stamps = []
    gyros = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=[imu_topic], log_time_order=True):
            if channel.topic != imu_topic or not schema or schema.name != "sensor_msgs/msg/Imu":
                continue
            imu = parse_sensor_msgs_imu_cdr(message.data)
            stamps.append(int(imu["stamp_ns"]))
            gyros.append([float(v) for v in imu["angular_velocity"]])
    return np.asarray(stamps, dtype=np.int64), np.asarray(gyros, dtype=float)


def interpolate_gyro(stamps_ns, gyros, query_ns):
    if query_ns < stamps_ns[0] or query_ns > stamps_ns[-1]:
        return None
    idx = int(np.searchsorted(stamps_ns, query_ns))
    if idx == 0:
        return gyros[0]
    if idx >= len(stamps_ns):
        return gyros[-1]
    t0 = stamps_ns[idx - 1]
    t1 = stamps_ns[idx]
    alpha = (query_ns - t0) / max(1, t1 - t0)
    return (1.0 - alpha) * gyros[idx - 1] + alpha * gyros[idx]


def solve_rotation(imu_vectors, cam_vectors):
    A = np.asarray(imu_vectors, dtype=float)
    B = np.asarray(cam_vectors, dtype=float)
    H = B.T @ A
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def summarize(values):
    if not values:
        return {}
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def estimate_camera_rotation(cam, dataset_path, intr, mcap_path, args):
    dataset = read_dataset(dataset_path)
    if dataset["num_cameras"] != 1:
        raise ValueError(f"{dataset_path} is expected to contain one camera")
    geometry = dataset["geometries"][0]
    poses = []
    rejected = {"too_few_features": 0, "pose_failed": 0, "missing_timestamp": 0}
    for imageset in dataset["imagesets"]:
        if imageset["timestamp_ns"] is None:
            rejected["missing_timestamp"] += 1
            continue
        object_points = []
        image_points = []
        for feature_id, x, y in imageset["cameras"][0]:
            point = object_point_for_feature(feature_id, geometry)
            if point is None:
                continue
            object_points.append(point)
            image_points.append((x, y))
        if len(object_points) < args.min_features:
            rejected["too_few_features"] += 1
            continue
        object_xyz = np.asarray(object_points, dtype=float)
        image_uv = np.asarray(image_points, dtype=float)
        try:
            pose = robust_pose_from_observations(
                object_xyz,
                image_uv,
                intr,
                args.min_features,
                args.max_pose_rmse_px,
                args.max_pose_inlier_px)
        except Exception:
            pose = None
        if pose is None:
            rejected["pose_failed"] += 1
            continue
        pose.update({
            "index": int(imageset["index"]),
            "src": imageset["src"],
            "timestamp_ns": int(imageset["timestamp_ns"]),
        })
        poses.append(pose)
        if len(poses) % 100 == 0:
            print(f"{cam}: poses={len(poses)} / imagesets={len(dataset['imagesets'])}", flush=True)

    stamps_ns, gyros = load_imu_gyro(mcap_path, args.imu_topic)
    visual_omegas = []
    imu_omegas = []
    samples = []
    delta_frames = max(1, int(args.delta_frames))
    for i in range(0, max(0, len(poses) - delta_frames)):
        a = poses[i]
        b = poses[i + delta_frames]
        dt = (b["timestamp_ns"] - a["timestamp_ns"]) / NS_PER_S
        if dt <= args.min_dt_s or dt >= args.max_dt_s:
            continue
        midpoint_ns = (a["timestamp_ns"] + b["timestamp_ns"]) // 2
        gyro = interpolate_gyro(stamps_ns, gyros, midpoint_ns)
        if gyro is None:
            continue
        R_a = a["R_cam_board"]
        R_b = b["R_cam_board"]
        omega_cam = so3_log(R_a @ R_b.T) / dt
        if np.linalg.norm(omega_cam) < args.min_omega_rad_s or np.linalg.norm(gyro) < args.min_omega_rad_s:
            continue
        visual_omegas.append(omega_cam)
        imu_omegas.append(gyro)
        samples.append({
            "timestamp_ns": int(midpoint_ns),
            "dt_s": float(dt),
            "visual_omega_norm": float(np.linalg.norm(omega_cam)),
            "imu_omega_norm": float(np.linalg.norm(gyro)),
        })
    if len(samples) < args.min_angular_samples:
        raise RuntimeError(f"{cam}: not enough angular samples: {len(samples)}")

    candidates = []
    for sign in (1.0, -1.0):
        cam_vectors = [sign * v for v in visual_omegas]
        R = solve_rotation(imu_omegas, cam_vectors)
        residuals = [float(np.linalg.norm(R @ imu - cam_vec)) for imu, cam_vec in zip(imu_omegas, cam_vectors)]
        candidates.append((float(np.median(residuals)), sign, R, residuals, cam_vectors))
    _, sign, R_initial, residuals_initial, cam_vectors = min(candidates, key=lambda item: item[0])
    cutoff = float(np.percentile(residuals_initial, args.angular_inlier_percentile))
    inliers = [i for i, r in enumerate(residuals_initial) if r <= cutoff]
    R_final = solve_rotation([imu_omegas[i] for i in inliers], [cam_vectors[i] for i in inliers])
    residuals_final = [float(np.linalg.norm(R_final @ imu_omegas[i] - cam_vectors[i])) for i in inliers]
    pose_rmse = [float(p["reprojection_rmse_px"]) for p in poses]
    pose_p95 = [float(p["reprojection_p95_px"]) for p in poses]
    feature_counts = [int(p["features"]) for p in poses]
    inlier_counts = [int(p["inliers"]) for p in poses]
    return {
        "camera": cam,
        "dataset": str(Path(dataset_path).resolve()),
        "mcap": str(Path(mcap_path).resolve()),
        "image_size": dataset["image_sizes"][0],
        "imagesets": len(dataset["imagesets"]),
        "valid_board_poses": len(poses),
        "rejected": rejected,
        "board_features": summarize(feature_counts),
        "pose_inlier_features": summarize(inlier_counts),
        "pose_reprojection_rmse_px": summarize(pose_rmse),
        "pose_reprojection_p95_px": summarize(pose_p95),
        "angular_samples": len(samples),
        "angular_inliers": len(inliers),
        "visual_omega_sign": sign,
        "R_cam_imu_raw": R_final,
        "q_cam_imu_raw_xyzw": rot_to_quat_xyzw(R_final).tolist(),
        "initial_residual_rad_s": summarize(residuals_initial),
        "inlier_residual_rad_s": summarize(residuals_final),
        "samples_preview": samples[:20],
    }


def invert_rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T


def matrix_to_list(m):
    return [[float(v) for v in row] for row in np.asarray(m)]


def write_outputs(reference_yaml, prior, camera_results, output_yaml, raw_yaml, summary_json, report_html):
    raw_common_rotations = {}
    for cam, result in camera_results.items():
        R_cam_cam0 = prior[cam]["T_cam_imu_prior"][:3, :3]
        raw_common_rotations[cam] = R_cam_cam0.T @ result["R_cam_imu_raw"]
    R_cam0_imu_avg, q_avg = average_rotations(raw_common_rotations.values())

    output = {}
    raw_output = {}
    final_R = {}
    raw_R = {}
    for cam in sorted(prior):
        entry = dict(reference_yaml[cam])
        raw_entry = dict(reference_yaml[cam])
        R_cam_cam0 = prior[cam]["T_cam_imu_prior"][:3, :3]
        t_cam_imu = prior[cam]["T_cam_imu_prior"][:3, 3]
        R_cam_imu = R_cam_cam0 @ R_cam0_imu_avg
        raw_R_cam_imu = camera_results[cam]["R_cam_imu_raw"]
        T = np.eye(4)
        T[:3, :3] = R_cam_imu
        T[:3, 3] = t_cam_imu
        raw_T = np.eye(4)
        raw_T[:3, :3] = raw_R_cam_imu
        raw_T[:3, 3] = t_cam_imu
        entry["T_cam_imu"] = matrix_to_list(T)
        entry["T_imu_cam"] = matrix_to_list(invert_rt(R_cam_imu, t_cam_imu))
        raw_entry["T_cam_imu"] = matrix_to_list(raw_T)
        raw_entry["T_imu_cam"] = matrix_to_list(invert_rt(raw_R_cam_imu, t_cam_imu))
        output[cam] = entry
        raw_output[cam] = raw_entry
        final_R[cam] = R_cam_imu
        raw_R[cam] = raw_R_cam_imu

    cams = sorted(output)
    for i, cam in enumerate(cams):
        prev = cams[i - 1]
        T_cam = np.asarray(output[cam]["T_cam_imu"], dtype=float)
        T_prev = np.asarray(output[prev]["T_cam_imu"], dtype=float)
        output[cam]["T_cn_cnm1"] = matrix_to_list(T_cam @ np.linalg.inv(T_prev))
        raw_T_cam = np.asarray(raw_output[cam]["T_cam_imu"], dtype=float)
        raw_T_prev = np.asarray(raw_output[prev]["T_cam_imu"], dtype=float)
        raw_output[cam]["T_cn_cnm1"] = matrix_to_list(raw_T_cam @ np.linalg.inv(raw_T_prev))

    pair_consistency = []
    for a, b in (("cam1", "cam2"), ("cam0", "cam3"), ("cam0", "cam1"), ("cam2", "cam3")):
        R_b_a = prior[b]["T_cam_imu_prior"][:3, :3] @ prior[a]["T_cam_imu_prior"][:3, :3].T
        predicted = R_b_a @ raw_R[a]
        measured = raw_R[b]
        angle = rotation_angle(measured @ predicted.T)
        pair_consistency.append({
            "from": a,
            "to": b,
            "angle_error_rad": float(angle),
            "angle_error_deg": float(math.degrees(angle)),
        })

    cam0_frame_residual = {}
    for cam, R_common in raw_common_rotations.items():
        cam0_frame_residual[cam] = float(math.degrees(rotation_angle(R_common @ R_cam0_imu_avg.T)))

    summary = {
        "format": "seeker_full_board_vi_rotation_summary_v0",
        "method": "full-board tagged-pattern pose + IMU gyro angular velocity alignment",
        "prior": {
            "camera_model": "kb8",
            "reference": "kalibr_cam_chain_kb8_generated_20260526.yaml",
            "translation_policy": "inherited_from_prior_not_solved",
        },
        "clock_assumption": "camera and IMU timestamps are in the same clock domain",
        "time_offset_s": 0.0,
        "averaged_R_cam0_imu": matrix_to_list(R_cam0_imu_avg),
        "averaged_q_cam0_imu_xyzw": [float(v) for v in q_avg.tolist()],
        "final_R_cam_imu": {cam: matrix_to_list(R) for cam, R in final_R.items()},
        "raw_R_cam_imu": {cam: matrix_to_list(R) for cam, R in raw_R.items()},
        "cam0_frame_residual_deg": cam0_frame_residual,
        "pair_consistency": pair_consistency,
        "cameras": {},
        "warnings": [
            "Rotation is aligned to the physical IMU gyro frame; camera-to-IMU translation is inherited from the prior and not measured in this pass.",
            "The prior KB8 rig rotations are used to enforce a rig-consistent final YAML.",
            "This uses the repository's full-board tagged star/checker detector, not Kalibr's AprilGrid detector.",
        ],
    }
    for cam, result in camera_results.items():
        serializable = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        serializable["R_cam_imu_raw"] = matrix_to_list(result["R_cam_imu_raw"])
        summary["cameras"][cam] = serializable

    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    raw_yaml = Path(raw_yaml)
    raw_yaml.write_text(yaml.safe_dump(raw_output, sort_keys=False), encoding="utf-8")
    summary_json = Path(summary_json)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html_report(report_html, summary, output_yaml, raw_yaml, summary_json)
    return summary


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_html_report(path, summary, output_yaml, raw_yaml, summary_json):
    rows = []
    for cam in sorted(summary["cameras"]):
        c = summary["cameras"][cam]
        pose_rmse = c.get("pose_reprojection_rmse_px", {})
        residual = c.get("inlier_residual_rad_s", {})
        features = c.get("board_features", {})
        rows.append(
            f"<tr><td>{cam}</td><td>{c.get('valid_board_poses')}</td>"
            f"<td>{fmt(features.get('median'), 1)}</td>"
            f"<td>{fmt(pose_rmse.get('median'), 3)}</td>"
            f"<td>{fmt(pose_rmse.get('p95'), 3)}</td>"
            f"<td>{c.get('angular_samples')}</td><td>{c.get('angular_inliers')}</td>"
            f"<td>{fmt(residual.get('median'), 4)}</td><td>{fmt(residual.get('p95'), 4)}</td>"
            f"<td>{fmt(summary['cam0_frame_residual_deg'].get(cam), 3)}</td></tr>")
    pair_rows = []
    for p in summary["pair_consistency"]:
        pair_rows.append(
            f"<tr><td>{p['from']} -> {p['to']}</td>"
            f"<td>{fmt(p['angle_error_deg'], 3)}</td>"
            f"<td>{fmt(p['angle_error_rad'], 5)}</td></tr>")
    warnings = "".join(f"<li>{w}</li>" for w in summary["warnings"])
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Full-Board Visual-IMU Calibration</title>
<style>
body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fb; color: #1f2937; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 48px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
h2 {{ margin: 28px 0 12px; font-size: 20px; letter-spacing: 0; }}
p {{ line-height: 1.6; }}
code {{ background: #edf2f7; border: 1px solid #d8dee9; border-radius: 4px; padding: 1px 5px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9e0ea; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e9f0; text-align: left; vertical-align: top; }}
th {{ background: #eef3f8; font-weight: 650; }}
.panel {{ background: white; border: 1px solid #d9e0ea; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }}
.path {{ word-break: break-all; }}
</style>
</head>
<body><main>
<h1>Full-Board Visual-IMU Calibration</h1>
<p>本报告使用 repository full-board tagged-pattern detector 的角点观测、prior KB8 内外参和 MCAP IMU gyro，估计 camera-to-IMU rotation。最终 YAML 对四路相机使用 rig-consistent SO(3) average；translation 沿用 prior。</p>
<section class="panel path">
<p>Final YAML: <code>{output_yaml}</code></p>
<p>Raw per-camera YAML: <code>{raw_yaml}</code></p>
<p>Summary JSON: <code>{summary_json}</code></p>
</section>
<h2>Per-Camera Quality</h2>
<table><thead><tr><th>Camera</th><th>Valid poses</th><th>Median features</th><th>Pose RMSE median px</th><th>Pose RMSE p95 px</th><th>Angular samples</th><th>Angular inliers</th><th>Gyro residual median rad/s</th><th>Gyro residual p95 rad/s</th><th>Cam0-frame residual deg</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Raw Pair Consistency</h2>
<table><thead><tr><th>Pair</th><th>Angle error deg</th><th>Angle error rad</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table>
<h2>Warnings</h2>
<section class="panel"><ul>{warnings}</ul></section>
</main></body></html>
"""
    Path(path).write_text(doc, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-yaml", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--down-mcap", required=True)
    parser.add_argument("--up-mcap", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--raw-output-yaml", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--report-html", required=True)
    parser.add_argument("--imu-topic", default="/seeker/imu")
    parser.add_argument("--min-features", type=int, default=80)
    parser.add_argument("--max-pose-rmse-px", type=float, default=3.0)
    parser.add_argument("--max-pose-inlier-px", type=float, default=5.0)
    parser.add_argument("--delta-frames", type=int, default=10)
    parser.add_argument("--min-dt-s", type=float, default=0.1)
    parser.add_argument("--max-dt-s", type=float, default=0.6)
    parser.add_argument("--min-omega-rad-s", type=float, default=0.08)
    parser.add_argument("--min-angular-samples", type=int, default=40)
    parser.add_argument("--angular-inlier-percentile", type=float, default=90.0)
    return parser.parse_args()


def main():
    args = parse_args()
    reference_yaml, prior = load_kb8_prior(args.prior_yaml)
    mcaps = {"down": args.down_mcap, "up": args.up_mcap}
    camera_results = {}
    for cam in sorted(prior):
        dataset_path = Path(args.feature_root) / f"{cam}_features.bin"
        result = estimate_camera_rotation(
            cam,
            dataset_path,
            prior[cam],
            mcaps[CAMERA_TO_MCAP[cam]],
            args)
        camera_results[cam] = result
        print(json.dumps({
            "camera": cam,
            "valid_board_poses": result["valid_board_poses"],
            "angular_samples": result["angular_samples"],
            "angular_inliers": result["angular_inliers"],
            "pose_rmse_median_px": result["pose_reprojection_rmse_px"].get("median"),
            "gyro_residual_median_rad_s": result["inlier_residual_rad_s"].get("median"),
        }, indent=2), flush=True)
    summary = write_outputs(
        reference_yaml,
        prior,
        camera_results,
        args.output_yaml,
        args.raw_output_yaml,
        args.summary_json,
        args.report_html)
    print(json.dumps({
        "output_yaml": str(Path(args.output_yaml).resolve()),
        "raw_output_yaml": str(Path(args.raw_output_yaml).resolve()),
        "summary_json": str(Path(args.summary_json).resolve()),
        "report_html": str(Path(args.report_html).resolve()),
        "pair_consistency": summary["pair_consistency"],
        "cam0_frame_residual_deg": summary["cam0_frame_residual_deg"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
