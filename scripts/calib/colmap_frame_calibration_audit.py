#!/usr/bin/env python3
"""Audit one synchronized frame with current studio calibration and COLMAP BA."""

import argparse
import csv
import datetime as _datetime
import html
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import time

import numpy as np


COLMAP_FULL_OPENCV_MODEL_ID = 6
COLMAP_PAIR_ID_BASE = 2147483647


def run_command(cmd, cwd, log_path, env=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        stream.flush()
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    return proc.returncode


def parse_scalar(raw):
    raw = raw.strip().strip('"').strip("'")
    if raw == "":
        return raw
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_inline_float_list(raw):
    start = raw.index("[") + 1
    end = raw.rindex("]")
    text = raw[start:end].strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",")]


def load_current_yaml(path):
    cameras = []
    current = None
    section = None
    with Path(path).open(encoding="utf-8") as stream:
        for raw in stream:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("- index:"):
                if current is not None:
                    cameras.append(current)
                current = {
                    "index": int(stripped.split(":", 1)[1].strip()),
                    "intrinsics": {},
                    "camera_tr_studio_rig": {},
                }
                section = None
                continue
            if current is None:
                continue
            if stripped == "intrinsics:":
                section = "intrinsics"
                continue
            if stripped == "camera_tr_studio_rig:":
                section = "pose"
                continue
            if stripped == "sources:":
                section = "sources"
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if section == "intrinsics":
                if key == "parameters":
                    current["intrinsics"][key] = parse_inline_float_list(value)
                else:
                    current["intrinsics"][key] = parse_scalar(value)
            elif section == "pose":
                current["camera_tr_studio_rig"][key] = float(value)
            elif section is None:
                current[key] = parse_scalar(value)
    if current is not None:
        cameras.append(current)
    cameras.sort(key=lambda row: int(row["index"]))
    return cameras


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def find_frame_rows(staged_root, frame_id):
    staged_root = Path(staged_root)
    frames = read_tsv(staged_root / "selected_frames.tsv")
    images = read_tsv(staged_root / "selected_images.tsv")
    exact_frames = [row for row in frames if row.get("frame_id") == str(frame_id)]
    exact_images = [row for row in images if row.get("frame_id") == str(frame_id)]
    if exact_frames:
        frame_row = exact_frames[0]
        image_rows = exact_images
        mapping_note = "exact_frame_id"
    else:
        valid_frames = []
        for row in frames:
            try:
                valid_frames.append((abs(int(row["frame_id"]) - frame_id), int(row["frame_id"]), row))
            except Exception:
                pass
        if not valid_frames:
            raise RuntimeError(f"No usable frame_id rows in {staged_root / 'selected_frames.tsv'}")
        _, nearest, frame_row = min(valid_frames, key=lambda item: (item[0], item[1]))
        image_rows = [row for row in images if row.get("frame_id") == str(nearest)]
        mapping_note = "nearest_frame_id"
    image_rows.sort(key=lambda row: int(row["camera_index"]))
    return frame_row, image_rows, mapping_note


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


def matrix_to_quat_wxyz(rotation):
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
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1
    return q


def pose_matrix(rotation, translation):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def invert_pose(pose):
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = pose[:3, :3].T
    inv[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inv


def rotation_angle_deg(rotation):
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(value)))


def camera_pose_from_yaml(camera):
    pose = camera["camera_tr_studio_rig"]
    rotation = quat_xyzw_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    translation = [pose["tx"], pose["ty"], pose["tz"]]
    return pose_matrix(rotation, translation)


def full_opencv_params(camera):
    params = list(camera["intrinsics"]["parameters"])
    if len(params) < 12:
        params = params + [0.0] * (12 - len(params))
    return [float(x) for x in params[:12]]


def stage_frame_images(image_rows, yaml_by_label, image_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in image_rows:
        label = row["camera_id"]
        if label not in yaml_by_label:
            continue
        src = Path(row["filtered_image"])
        if not src.exists():
            raise FileNotFoundError(src)
        camera = yaml_by_label[label]
        name = f"cam{int(row['camera_index']):02d}_{label}_frame{int(row['frame_id']):06d}_out{int(row['out_frame']):06d}{src.suffix.lower()}"
        dst = image_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        records.append({
            "label": label,
            "camera_index": int(row["camera_index"]),
            "yaml_index": int(camera["index"]),
            "image_name": name,
            "source": str(src),
            "raw_source": row.get("source", ""),
            "tag_count": int(row.get("tag_count", "0") or 0),
            "corner_count": int(row.get("corner_count", "0") or 0),
            "frame_id": int(row["frame_id"]),
            "out_frame": int(row["out_frame"]),
        })
    records.sort(key=lambda row: row["camera_index"])
    return records


def sqlite_params_blob(params):
    return struct.pack("<" + "d" * len(params), *params)


def rewrite_colmap_database(database_path, records, yaml_by_label):
    conn = sqlite3.connect(str(database_path))
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    name_to_image_id = {name: image_id for image_id, name in cur.fetchall()}
    cur.execute("DELETE FROM cameras")
    missing = []
    for rec in records:
        image_id = name_to_image_id.get(rec["image_name"])
        if image_id is None:
            missing.append(rec["image_name"])
            continue
        camera = yaml_by_label[rec["label"]]
        intr = camera["intrinsics"]
        camera_id = int(camera["index"]) + 1
        params = full_opencv_params(camera)
        cur.execute(
            "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                COLMAP_FULL_OPENCV_MODEL_ID,
                int(intr["width"]),
                int(intr["height"]),
                sqlite_params_blob(params),
                1,
            ),
        )
        cur.execute("UPDATE images SET camera_id=? WHERE image_id=?", (camera_id, image_id))
        rec["image_id"] = int(image_id)
        rec["camera_id"] = camera_id
        rec["initial_params"] = params
    conn.commit()
    conn.close()
    if missing:
        raise RuntimeError(f"Feature extractor did not create DB image rows: {missing}")


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % COLMAP_PAIR_ID_BASE
    image_id1 = (pair_id - image_id2) // COLMAP_PAIR_ID_BASE
    return int(image_id1), int(image_id2)


def read_match_stats(database_path, records):
    id_to_label = {int(row["image_id"]): row["label"] for row in records if "image_id" in row}
    pair_rows = []
    per_label = {row["label"]: {"verified_pairs": 0, "verified_inliers": 0} for row in records}
    conn = sqlite3.connect(str(database_path))
    cur = conn.cursor()
    cur.execute("SELECT pair_id, rows FROM two_view_geometries")
    for pair_id, rows in cur.fetchall():
        image_id1, image_id2 = pair_id_to_image_ids(int(pair_id))
        label1 = id_to_label.get(image_id1)
        label2 = id_to_label.get(image_id2)
        if not label1 or not label2:
            continue
        rows = int(rows)
        if rows <= 0:
            continue
        pair_rows.append({"label1": label1, "label2": label2, "inlier_matches": rows})
        per_label[label1]["verified_pairs"] += 1
        per_label[label1]["verified_inliers"] += rows
        per_label[label2]["verified_pairs"] += 1
        per_label[label2]["verified_inliers"] += rows
    conn.close()
    pair_rows.sort(key=lambda row: row["inlier_matches"], reverse=True)
    return pair_rows, per_label


def write_initial_model(model_dir, records, yaml_by_label):
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "cameras.txt").open("w", encoding="utf-8") as stream:
        stream.write("# Camera list with one line of data per camera:\n")
        stream.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for rec in records:
            camera = yaml_by_label[rec["label"]]
            intr = camera["intrinsics"]
            params = full_opencv_params(camera)
            stream.write(
                "{} FULL_OPENCV {} {} {}\n".format(
                    rec["camera_id"],
                    int(intr["width"]),
                    int(intr["height"]),
                    " ".join(f"{x:.17g}" for x in params),
                )
            )
    with (model_dir / "images.txt").open("w", encoding="utf-8") as stream:
        stream.write("# Image list with two lines of data per image:\n")
        stream.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        stream.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for rec in records:
            pose = camera_pose_from_yaml(yaml_by_label[rec["label"]])
            qw, qx, qy, qz = matrix_to_quat_wxyz(pose[:3, :3])
            tx, ty, tz = pose[:3, 3]
            stream.write(
                "{} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {} {}\n\n".format(
                    rec["image_id"], qw, qx, qy, qz, tx, ty, tz, rec["camera_id"], rec["image_name"]
                )
            )
    with (model_dir / "points3D.txt").open("w", encoding="utf-8") as stream:
        stream.write("# 3D point list with one line of data per point:\n")
        stream.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")


def load_colmap_cameras(path):
    cameras = {}
    with (Path(path) / "cameras.txt").open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cameras[int(parts[0])] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]],
            }
    return cameras


def load_colmap_images(path):
    images = {}
    lines = (Path(path) / "images.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        point_line = lines[i].strip() if i < len(lines) else ""
        if i < len(lines):
            i += 1
        xys = []
        if point_line:
            p = point_line.split()
            for j in range(0, len(p), 3):
                xys.append((float(p[j]), float(p[j + 1]), int(p[j + 2])))
        camera_tr_world = pose_matrix(quat_wxyz_to_matrix(qw, qx, qy, qz), [tx, ty, tz])
        images[image_id] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "name": name,
            "camera_tr_world": camera_tr_world,
            "world_tr_camera": invert_pose(camera_tr_world),
            "xys": xys,
        }
    return images


def load_colmap_points(path):
    points = {}
    p = Path(path) / "points3D.txt"
    if not p.exists():
        return points
    with p.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            error = float(parts[7])
            track_parts = parts[8:]
            track = []
            for j in range(0, len(track_parts), 2):
                track.append((int(track_parts[j]), int(track_parts[j + 1])))
            points[point_id] = {"xyz": xyz, "error": error, "track": track}
    return points


def project_full_opencv(params, xyz_camera):
    x = xyz_camera[0] / xyz_camera[2]
    y = xyz_camera[1] / xyz_camera[2]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params[:12]
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = numerator / denominator if abs(denominator) > 1e-12 else numerator
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.asarray([fx * xd + cx, fy * yd + cy], dtype=np.float64)


def compute_reprojection_stats(model_dir, records):
    cameras = load_colmap_cameras(model_dir)
    images = load_colmap_images(model_dir)
    points = load_colmap_points(model_dir)
    image_id_to_label = {int(row["image_id"]): row["label"] for row in records}
    per_label = {
        row["label"]: {"residuals": [], "triangulated_observations": 0, "point2d_count": 0}
        for row in records
    }
    all_residuals = []
    observation_keys = set()
    for image in images.values():
        label = image_id_to_label.get(int(image["image_id"]))
        if label in per_label:
            per_label[label]["point2d_count"] = len(image["xys"])
    for point_id, point in points.items():
        for image_id, point2d_idx in point["track"]:
            image = images.get(image_id)
            if image is None or point2d_idx >= len(image["xys"]):
                continue
            label = image_id_to_label.get(image_id)
            camera = cameras.get(image["camera_id"])
            if label is None or camera is None:
                continue
            xy = np.asarray(image["xys"][point2d_idx][:2], dtype=np.float64)
            xyz_camera = image["camera_tr_world"][:3, :3] @ point["xyz"] + image["camera_tr_world"][:3, 3]
            if xyz_camera[2] <= 1e-9:
                continue
            projected = project_full_opencv(camera["params"], xyz_camera)
            residual = float(np.linalg.norm(projected - xy))
            all_residuals.append(residual)
            per_label[label]["residuals"].append(residual)
            per_label[label]["triangulated_observations"] += 1
            observation_keys.add((point_id, image_id, point2d_idx))
    stats = summarize_residuals(all_residuals)
    stats.update({
        "points3D_count": len(points),
        "registered_image_count": len(images),
        "observation_count": len(all_residuals),
    })
    per_camera = {}
    for label, row in per_label.items():
        out = summarize_residuals(row["residuals"])
        out["triangulated_observations"] = row["triangulated_observations"]
        out["point2d_count"] = row["point2d_count"]
        per_camera[label] = out
    return stats, per_camera, observation_keys


def summarize_residuals(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean_px": None,
            "rmse_px": None,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "count": 0,
        }
    return {
        "mean_px": float(np.mean(arr)),
        "rmse_px": float(np.sqrt(np.mean(arr * arr))),
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90)),
        "max_px": float(np.max(arr)),
        "count": int(arr.size),
    }


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
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    variance = np.sum(source_centered ** 2) / source.shape[0]
    scale = float(np.sum(singular_values * np.diag(sign)) / variance)
    translation = target_mean - scale * rotation @ source_mean
    residuals = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
    return scale, rotation, translation, residuals


def compare_models(before_dir, after_dir, records):
    before_cameras = load_colmap_cameras(before_dir)
    after_cameras = load_colmap_cameras(after_dir)
    before_images = load_colmap_images(before_dir)
    after_images = load_colmap_images(after_dir)
    rows = []
    before_centers = []
    after_centers = []
    aligned_labels = []
    for rec in records:
        before_image = before_images.get(int(rec["image_id"]))
        after_image = after_images.get(int(rec["image_id"]))
        if before_image is None or after_image is None:
            continue
        before_centers.append(before_image["world_tr_camera"][:3, 3])
        after_centers.append(after_image["world_tr_camera"][:3, 3])
        aligned_labels.append(rec["label"])
    scale, sim_R, sim_t, sim_residuals = umeyama_similarity(after_centers, before_centers)
    sim_residual_by_label = {label: float(res) for label, res in zip(aligned_labels, sim_residuals)}
    for rec in records:
        label = rec["label"]
        before_image = before_images.get(int(rec["image_id"]))
        after_image = after_images.get(int(rec["image_id"]))
        before_camera = before_cameras.get(int(rec["camera_id"]))
        after_camera = after_cameras.get(int(rec["camera_id"]))
        if before_image is None or after_image is None or before_camera is None or after_camera is None:
            continue
        before_center = before_image["world_tr_camera"][:3, 3]
        after_center_raw = after_image["world_tr_camera"][:3, 3]
        after_center_aligned = scale * sim_R @ after_center_raw + sim_t
        before_world_R_camera = before_image["world_tr_camera"][:3, :3]
        after_world_R_camera = sim_R @ after_image["world_tr_camera"][:3, :3]
        pose_delta_rot = rotation_angle_deg(after_world_R_camera @ before_world_R_camera.T)
        params_before = before_camera["params"]
        params_after = after_camera["params"]
        diffs = [params_after[i] - params_before[i] for i in range(min(len(params_before), len(params_after)))]
        rows.append({
            "label": label,
            "camera_index": rec["camera_index"],
            "tag_count": rec["tag_count"],
            "corner_count": rec["corner_count"],
            "fx_delta": diffs[0],
            "fy_delta": diffs[1],
            "cx_delta": diffs[2],
            "cy_delta": diffs[3],
            "dist_l2_delta": float(np.linalg.norm(np.asarray(diffs[4:], dtype=np.float64))),
            "dist_max_abs_delta": float(np.max(np.abs(np.asarray(diffs[4:], dtype=np.float64)))) if len(diffs) > 4 else 0.0,
            "translation_delta_m": float(np.linalg.norm(after_center_aligned - before_center)),
            "rotation_delta_deg": pose_delta_rot,
            "raw_translation_delta_m": float(np.linalg.norm(after_center_raw - before_center)),
            "sim3_center_residual_m": sim_residual_by_label.get(label, 0.0),
        })
    rows.sort(key=lambda row: row["camera_index"])
    summary = summarize_camera_deltas(rows)
    summary["sim3_after_to_before"] = {
        "scale": scale,
        "translation": [float(x) for x in sim_t],
        "center_rmse_m": float(np.sqrt(np.mean(np.asarray(sim_residuals) ** 2))),
        "center_max_m": float(np.max(sim_residuals)),
    }
    return rows, summary


def percentile(arr, q):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def summarize_camera_deltas(rows):
    keys = [
        "fx_delta",
        "fy_delta",
        "cx_delta",
        "cy_delta",
        "dist_l2_delta",
        "dist_max_abs_delta",
        "translation_delta_m",
        "rotation_delta_deg",
        "raw_translation_delta_m",
    ]
    summary = {}
    for key in keys:
        vals = np.asarray([abs(row[key]) for row in rows], dtype=np.float64)
        summary[key] = {
            "mean_abs": float(np.mean(vals)) if vals.size else None,
            "median_abs": percentile(vals, 50),
            "p90_abs": percentile(vals, 90),
            "max_abs": float(np.max(vals)) if vals.size else None,
        }
    return summary


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def html_table(rows, columns, limit=None):
    rows = rows[:limit] if limit is not None else rows
    out = ["<table><thead><tr>"]
    for key, label in columns:
        out.append(f"<th>{html.escape(label)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value, 5)
            out.append(f"<td>{html.escape(str(value))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def make_report(output_root, summary, camera_rows, per_camera_before, per_camera_after, match_pairs, inactive_cameras):
    before = summary["reprojection_before"]
    after = summary["reprojection_after"]
    error_drop = summary["error_drop"]
    camera_summary = summary["camera_delta_summary"]
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>COLMAP Frame 380 BA Audit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
h1, h2 {{ margin-bottom: 0.35rem; }}
code {{ background: #eef2f6; padding: 0.1rem 0.25rem; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.25rem; font-size: 13px; }}
th, td {{ border: 1px solid #d8dee8; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f6fa; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
.metric {{ border: 1px solid #d8dee8; border-radius: 6px; padding: 12px; }}
.metric strong {{ display: block; font-size: 22px; margin-bottom: 4px; }}
.note {{ background: #f8fafc; border-left: 4px solid #52616f; padding: 10px 14px; }}
</style>
</head>
<body>
<h1>COLMAP Frame 380 BA Audit</h1>
<p>Run directory: <code>{html.escape(str(output_root))}</code></p>
<p>Input frame mapping: raw <code>frame_id={summary['frame']['requested_frame_id']}</code>,
selected <code>out_frame={summary['frame']['out_frame']}</code>,
selected filename <code>{html.escape(summary['frame']['selected_filename'])}</code>,
mapping status <code>{html.escape(summary['frame']['mapping_note'])}</code>.</p>

<h2>Headline Metrics</h2>
<div class="grid">
<div class="metric"><strong>{fmt(before['rmse_px'], 5)} px</strong><span>before BA RMSE</span></div>
<div class="metric"><strong>{fmt(after['rmse_px'], 5)} px</strong><span>after BA RMSE</span></div>
<div class="metric"><strong>{fmt(error_drop['rmse_abs_px'], 5)} px</strong><span>RMSE drop</span></div>
<div class="metric"><strong>{fmt(error_drop['rmse_rel_pct'], 5)}%</strong><span>relative RMSE drop</span></div>
</div>
<table>
<thead><tr><th>metric</th><th>before</th><th>after</th><th>delta</th></tr></thead>
<tbody>
<tr><td>mean reprojection error px</td><td>{fmt(before['mean_px'], 6)}</td><td>{fmt(after['mean_px'], 6)}</td><td>{fmt(error_drop['mean_abs_px'], 6)}</td></tr>
<tr><td>RMSE reprojection error px</td><td>{fmt(before['rmse_px'], 6)}</td><td>{fmt(after['rmse_px'], 6)}</td><td>{fmt(error_drop['rmse_abs_px'], 6)}</td></tr>
<tr><td>median reprojection error px</td><td>{fmt(before['median_px'], 6)}</td><td>{fmt(after['median_px'], 6)}</td><td>{fmt(error_drop['median_abs_px'], 6)}</td></tr>
<tr><td>p90 reprojection error px</td><td>{fmt(before['p90_px'], 6)}</td><td>{fmt(after['p90_px'], 6)}</td><td>{fmt(error_drop['p90_abs_px'], 6)}</td></tr>
<tr><td>observations</td><td>{before['observation_count']}</td><td>{after['observation_count']}</td><td></td></tr>
<tr><td>points3D</td><td>{before['points3D_count']}</td><td>{after['points3D_count']}</td><td></td></tr>
</tbody>
</table>

<h2>Camera Delta Summary</h2>
<p class="note">Pose deltas are reported after aligning the BA world frame back to the current studio rig frame with a global Sim(3). This removes the unobservable single-frame gauge before measuring per-camera changes. Raw translation deltas are retained in TSV/JSON only as a gauge diagnostic.</p>
<table>
<thead><tr><th>quantity</th><th>mean abs</th><th>median abs</th><th>p90 abs</th><th>max abs</th></tr></thead>
<tbody>
"""
    for key, label in [
        ("fx_delta", "fx delta px"),
        ("fy_delta", "fy delta px"),
        ("cx_delta", "cx delta px"),
        ("cy_delta", "cy delta px"),
        ("dist_l2_delta", "distortion L2 delta"),
        ("dist_max_abs_delta", "distortion max abs delta"),
        ("translation_delta_m", "translation delta m, Sim(3)-aligned"),
        ("rotation_delta_deg", "rotation delta deg, Sim(3)-aligned"),
    ]:
        row = camera_summary[key]
        report += f"<tr><td>{html.escape(label)}</td><td>{fmt(row['mean_abs'], 6)}</td><td>{fmt(row['median_abs'], 6)}</td><td>{fmt(row['p90_abs'], 6)}</td><td>{fmt(row['max_abs'], 6)}</td></tr>\n"
    sim3 = summary["camera_delta_summary"]["sim3_after_to_before"]
    report += f"""</tbody></table>
<p>BA-to-current Sim(3): scale <code>{fmt(sim3['scale'], 8)}</code>, center RMSE <code>{fmt(sim3['center_rmse_m'], 6)} m</code>, max center residual <code>{fmt(sim3['center_max_m'], 6)} m</code>.</p>

<h2>Per-Camera BA Deltas</h2>
{html_table(camera_rows, [
    ("label", "camera"),
    ("tag_count", "tags"),
    ("corner_count", "corners"),
    ("fx_delta", "dfx"),
    ("fy_delta", "dfy"),
    ("cx_delta", "dcx"),
    ("cy_delta", "dcy"),
    ("dist_l2_delta", "dist L2"),
    ("translation_delta_m", "dT m"),
    ("rotation_delta_deg", "dR deg"),
])}

<h2>Participation And Observations</h2>
{html_table(summary['participation_rows'], [
    ("label", "camera"),
    ("tag_count", "tags"),
    ("corner_count", "corners"),
    ("verified_pairs", "verified pairs"),
    ("verified_inliers", "verified inliers"),
    ("before_triangulated_observations", "before obs"),
    ("after_triangulated_observations", "after obs"),
    ("status", "status"),
])}

<h2>Top Verified COLMAP Pairs</h2>
{html_table(match_pairs, [("label1", "camera A"), ("label2", "camera B"), ("inlier_matches", "inlier matches")], limit=30)}

<h2>Unparticipating Cameras</h2>
<p>{html.escape(', '.join(inactive_cameras) if inactive_cameras else 'None among staged outer24. Inner cameras were absent from this whole_outer24 staged root.')}</p>

<h2>Evidence Conclusion</h2>
<p>For this frame, the AprilTag-selected metadata reports {summary['frame']['passing_camera_count']} passing cameras out of {summary['frame']['active_camera_count']} active staged cameras. COLMAP does not require tag decoding; it uses SIFT features, verified pair geometry, and triangulated tracks. Cameras with zero decoded tags can still obtain COLMAP matches if the scene texture or the repeated board texture provides enough local features. Conversely, tag BA reports obs=0 whenever decoded/filtered tag corners are absent, even if SIFT matches exist.</p>
<p>The important caveat is that this is a single synchronized frame. The reprojection reduction measures whether the current rig is locally compatible with the SIFT tracks, not whether the metric studio rig is fully observable from one frame. The Sim(3) gauge alignment above is therefore part of the reported pose comparison.</p>
</body>
</html>
"""
    (Path(output_root) / "report.html").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-yaml", required=True, type=Path)
    parser.add_argument("--staged-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frame-id", type=int, default=380)
    parser.add_argument("--colmap-bin", default="/home/ubuntu/miniconda3/envs/colmap4/bin/colmap")
    parser.add_argument("--max-image-size", type=int, default=3200)
    parser.add_argument("--max-num-features", type=int, default=12000)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--ba-iterations", type=int, default=80)
    parser.add_argument("--fix-intrinsics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    logs_dir = output_root / "logs"
    logs_dir.mkdir()

    start = time.time()
    cameras = load_current_yaml(args.current_yaml)
    yaml_by_label = {row["label"]: row for row in cameras}
    frame_row, image_rows, mapping_note = find_frame_rows(args.staged_root, args.frame_id)
    image_dir = output_root / "images"
    records = stage_frame_images(image_rows, yaml_by_label, image_dir)
    present_labels = {row["label"] for row in records}
    inactive_cameras = [
        row["label"] for row in cameras
        if row["label"] not in present_labels
    ]

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.num_threads)

    database_path = output_root / "database.db"
    feature_cmd = [
        args.colmap_bin,
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera_per_image", "1",
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.num_threads", str(args.num_threads),
        "--FeatureExtraction.max_image_size", str(args.max_image_size),
        "--SiftExtraction.max_num_features", str(args.max_num_features),
    ]
    rc = run_command(feature_cmd, output_root, logs_dir / "01_feature_extractor.log", env)
    if rc != 0:
        raise RuntimeError(f"feature_extractor failed with rc={rc}")
    rewrite_colmap_database(database_path, records, yaml_by_label)

    matcher_cmd = [
        args.colmap_bin,
        "exhaustive_matcher",
        "--database_path", database_path,
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.num_threads", str(args.num_threads),
        "--FeatureMatching.guided_matching", "1",
    ]
    rc = run_command(matcher_cmd, output_root, logs_dir / "02_exhaustive_matcher.log", env)
    if rc != 0:
        raise RuntimeError(f"exhaustive_matcher failed with rc={rc}")
    match_pairs, per_label_matches = read_match_stats(database_path, records)

    initial_model = output_root / "sparse_initial_txt"
    write_initial_model(initial_model, records, yaml_by_label)

    triangulated_model = output_root / "sparse_triangulated"
    triangulated_model.mkdir()
    triangulator_cmd = [
        args.colmap_bin,
        "point_triangulator",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--input_path", initial_model,
        "--output_path", triangulated_model,
        "--clear_points", "1",
        "--refine_intrinsics", "0",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.fix_existing_frames", "1",
        "--Mapper.tri_ignore_two_view_tracks", "0",
        "--Mapper.tri_min_angle", "0.2",
        "--Mapper.filter_max_reproj_error", "20",
        "--Mapper.num_threads", str(args.num_threads),
    ]
    rc = run_command(triangulator_cmd, output_root, logs_dir / "03_point_triangulator.log", env)
    if rc != 0:
        raise RuntimeError(f"point_triangulator failed with rc={rc}")

    triangulated_txt = output_root / "sparse_triangulated_txt"
    triangulated_txt.mkdir()
    rc = run_command([
        args.colmap_bin, "model_converter",
        "--input_path", triangulated_model,
        "--output_path", triangulated_txt,
        "--output_type", "TXT",
    ], output_root, logs_dir / "04_model_converter_triangulated.log", env)
    if rc != 0:
        raise RuntimeError(f"model_converter triangulated failed with rc={rc}")

    ba_model = output_root / "sparse_ba"
    ba_model.mkdir()
    ba_cmd = [
        args.colmap_bin,
        "bundle_adjuster",
        "--input_path", triangulated_model,
        "--output_path", ba_model,
        "--BundleAdjustment.refine_focal_length", "0" if args.fix_intrinsics else "1",
        "--BundleAdjustment.refine_principal_point", "0" if args.fix_intrinsics else "1",
        "--BundleAdjustment.refine_extra_params", "0" if args.fix_intrinsics else "1",
        "--BundleAdjustment.refine_points3D", "1",
        "--BundleAdjustmentCeres.max_num_iterations", str(args.ba_iterations),
        "--BundleAdjustmentCeres.use_gpu", "0",
    ]
    rc = run_command(ba_cmd, output_root, logs_dir / "05_bundle_adjuster.log", env)
    if rc != 0:
        raise RuntimeError(f"bundle_adjuster failed with rc={rc}")

    ba_txt = output_root / "sparse_ba_txt"
    ba_txt.mkdir()
    rc = run_command([
        args.colmap_bin, "model_converter",
        "--input_path", ba_model,
        "--output_path", ba_txt,
        "--output_type", "TXT",
    ], output_root, logs_dir / "06_model_converter_ba.log", env)
    if rc != 0:
        raise RuntimeError(f"model_converter BA failed with rc={rc}")

    run_command([args.colmap_bin, "model_analyzer", "--path", triangulated_model], output_root, logs_dir / "07_model_analyzer_before.log", env)
    run_command([args.colmap_bin, "model_analyzer", "--path", ba_model], output_root, logs_dir / "08_model_analyzer_after.log", env)

    repro_before, per_camera_before, before_obs = compute_reprojection_stats(triangulated_txt, records)
    repro_after, per_camera_after, after_obs = compute_reprojection_stats(ba_txt, records)
    camera_rows, camera_delta_summary = compare_models(triangulated_txt, ba_txt, records)
    common_obs_count = len(before_obs & after_obs)

    def drop(before_key, after_key):
        b = repro_before[before_key]
        a = repro_after[after_key]
        if b is None or a is None:
            return None, None
        abs_drop = b - a
        rel = 100.0 * abs_drop / b if b else None
        return abs_drop, rel

    rmse_drop, rmse_rel = drop("rmse_px", "rmse_px")
    mean_drop, mean_rel = drop("mean_px", "mean_px")
    median_drop, median_rel = drop("median_px", "median_px")
    p90_drop, p90_rel = drop("p90_px", "p90_px")

    participation_rows = []
    for rec in records:
        label = rec["label"]
        before_cam = per_camera_before.get(label, {})
        after_cam = per_camera_after.get(label, {})
        matches = per_label_matches.get(label, {})
        status = "ok"
        if after_cam.get("triangulated_observations", 0) == 0:
            status = "no_triangulated_observations"
        elif after_cam.get("triangulated_observations", 0) < 20:
            status = "low_observation_count"
        participation_rows.append({
            "label": label,
            "camera_index": rec["camera_index"],
            "tag_count": rec["tag_count"],
            "corner_count": rec["corner_count"],
            "verified_pairs": matches.get("verified_pairs", 0),
            "verified_inliers": matches.get("verified_inliers", 0),
            "before_triangulated_observations": before_cam.get("triangulated_observations", 0),
            "after_triangulated_observations": after_cam.get("triangulated_observations", 0),
            "before_rmse_px": before_cam.get("rmse_px"),
            "after_rmse_px": after_cam.get("rmse_px"),
            "status": status,
        })
    inactive_all = list(inactive_cameras)
    inactive_all.extend(
        row["label"] for row in participation_rows
        if row["status"] != "ok"
    )

    summary = {
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start,
        "command": " ".join(sys.argv),
        "current_yaml": str(args.current_yaml),
        "staged_root": str(args.staged_root),
        "output_root": str(output_root),
        "colmap_bin": str(args.colmap_bin),
        "colmap_camera_model": "FULL_OPENCV",
        "ba_intrinsics_mode": "fixed" if args.fix_intrinsics else "refined",
        "coordinate_conversion": {
            "initial_colmap_world": "studio_rig_current",
            "initial_image_pose": "COLMAP image qvec/tvec stores camera_tr_studio_rig from YAML",
            "yaml_quaternion_order": "qx qy qz qw",
            "colmap_quaternion_order": "qw qx qy qz",
            "pose_delta_alignment": "BA camera centers are Sim(3)-aligned back to current centers before per-camera pose deltas",
        },
        "frame": {
            "requested_frame_id": args.frame_id,
            "mapping_note": mapping_note,
            "out_frame": int(frame_row["out_frame"]),
            "frame_id": int(frame_row["frame_id"]),
            "frame_key": frame_row["frame_key"],
            "selected_filename": frame_row["selected_filename"],
            "observed_camera_count": int(frame_row["observed_camera_count"]),
            "passing_camera_count": int(frame_row["passing_camera_count"]),
            "active_camera_count": int(frame_row["active_camera_count"]),
        },
        "records": records,
        "inactive_cameras_from_yaml": inactive_cameras,
        "inner_camera_note": "The staged whole root contains outer24 only; current YAML inner0..inner7 have no image rows here.",
        "match_pair_count": len(match_pairs),
        "match_pairs_top": match_pairs[:50],
        "reprojection_before": repro_before,
        "reprojection_after": repro_after,
        "common_observation_count": common_obs_count,
        "error_drop": {
            "rmse_abs_px": rmse_drop,
            "rmse_rel_pct": rmse_rel,
            "mean_abs_px": mean_drop,
            "mean_rel_pct": mean_rel,
            "median_abs_px": median_drop,
            "median_rel_pct": median_rel,
            "p90_abs_px": p90_drop,
            "p90_rel_pct": p90_rel,
        },
        "camera_delta_summary": camera_delta_summary,
        "participation_rows": participation_rows,
    }

    write_tsv(
        output_root / "camera_deltas.tsv",
        camera_rows,
        [
            "camera_index", "label", "tag_count", "corner_count",
            "fx_delta", "fy_delta", "cx_delta", "cy_delta",
            "dist_l2_delta", "dist_max_abs_delta",
            "translation_delta_m", "rotation_delta_deg",
            "raw_translation_delta_m", "sim3_center_residual_m",
        ],
    )
    write_tsv(
        output_root / "participation.tsv",
        participation_rows,
        [
            "camera_index", "label", "tag_count", "corner_count",
            "verified_pairs", "verified_inliers",
            "before_triangulated_observations", "after_triangulated_observations",
            "before_rmse_px", "after_rmse_px", "status",
        ],
    )
    write_tsv(output_root / "verified_pairs.tsv", match_pairs, ["label1", "label2", "inlier_matches"])
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_report(output_root, summary, camera_rows, per_camera_before, per_camera_after, match_pairs, inactive_all)

    print(json.dumps({
        "output_root": str(output_root),
        "report": str(output_root / "report.html"),
        "summary": str(output_root / "summary.json"),
        "frame": summary["frame"],
        "reprojection_before": repro_before,
        "reprojection_after": repro_after,
        "error_drop": summary["error_drop"],
        "inactive_cameras": inactive_all,
    }, indent=2))


if __name__ == "__main__":
    main()
