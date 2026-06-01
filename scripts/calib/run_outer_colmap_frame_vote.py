#!/usr/bin/env python3
"""Run many single-frame COLMAP trials and robust-vote an outer camera rig."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

import numpy as np


def resolve_colmap_bin(value):
    if value and value != "colmap":
        return str(value)
    found = shutil.which("colmap")
    if found:
        return found
    for candidate in (
            "/home/ubuntu/miniconda3/envs/colmap4/bin/colmap",
            "/home/ubuntu/miniconda3/pkgs/colmap-4.0.4-cuda_129h7d026d0_1/bin/colmap",
            "/home/ubuntu/miniconda3/pkgs/colmap-3.13.0-cpu_h5de0465_4/bin/colmap",
            "/usr/local/bin/colmap",
            "/usr/bin/colmap"):
        if Path(candidate).is_file():
            return candidate
    return str(value or "colmap")


def read_manifest(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            row["camera_index"] = int(row["camera_index"])
            row["frame_count"] = int(row.get("frame_count", "0") or 0)
            rows.append(row)
    rows.sort(key=lambda row: row["camera_index"])
    return rows


def attach_source_files(manifest):
    for row in manifest:
        source_dir = Path(row["source_dir"])
        files = [
            path for path in sorted(source_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not files:
            raise FileNotFoundError(f"No images found in {source_dir}")
        row["files"] = [str(path) for path in files]
        row["frame_count"] = min(row["frame_count"], len(files)) if row.get("frame_count") else len(files)


def parse_label_pose_indices(text):
    result = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        label, index = item.split(":", 1)
        result[label.strip()] = int(index)
    return result


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


def load_pose_yaml(path):
    pose_count = None
    poses = {}
    current = None

    def flush():
        if current is None:
            return
        index = int(current["index"])
        poses[index] = pose_matrix(
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
        raise ValueError(f"Missing pose_count in {path}")
    return [poses.get(i) for i in range(pose_count)]


def write_pose_yaml(path, poses):
    lines = [
        "# Each pose gives the B_tr_A transformation (i.e. A to B with right-multiplication).",
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


def load_anchor_centers(path, label_to_pose_index):
    poses = load_pose_yaml(path)
    centers = {}
    for label, index in label_to_pose_index.items():
        if index >= len(poses) or poses[index] is None:
            raise ValueError(f"Anchor pose {label}:{index} missing in {path}")
        centers[label] = invert_pose(poses[index])[:3, 3]
    return centers


def available_frames(manifest):
    return list(range(min(len(row["files"]) for row in manifest)))


def select_frames(manifest, args):
    if args.frames:
        frames = []
        for chunk in args.frames.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                a, b = chunk.split("-", 1)
                frames.extend(range(int(a), int(b) + 1))
            else:
                frames.append(int(chunk))
        return list(dict.fromkeys(frames))
    frames = available_frames(manifest)
    rng = random.Random(args.seed)
    if args.sample_count > len(frames):
        raise ValueError(f"Requested {args.sample_count} frames but only {len(frames)} are available")
    return sorted(rng.sample(frames, args.sample_count))


def run_command(cmd, cwd, log_path, env):
    with Path(log_path).open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env)
    return proc.returncode


def link_or_copy(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def prepare_frame_images(run_dir, manifest, frame, mode):
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in manifest:
        src = Path(row["files"][frame])
        if not src.exists():
            raise FileNotFoundError(src)
        name = f"cam{row['camera_index']:02d}_{row['camera_id']}_f{frame:04d}.jpg"
        link_or_copy(src, image_dir / name, mode)
        records.append({
            "camera_index": row["camera_index"],
            "camera_id": row["camera_id"],
            "image_name": name,
            "source": str(src),
        })
    return image_dir, records


def parse_colmap_label(name):
    match = re.search(r"cam\d+_([^_]+)_f\d+", name)
    return match.group(1) if match else Path(name).stem


def read_points3d_count(path):
    count = 0
    if not Path(path).exists():
        return 0
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            count += 1
    return count


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
        if i < len(lines):
            i += 1
        point_ids = point_line.split()[2::3]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_tr_world = pose_matrix(quat_wxyz_to_matrix(qw, qx, qy, qz), [tx, ty, tz])
        world_tr_camera = invert_pose(camera_tr_world)
        label = parse_colmap_label(parts[9])
        images[label] = {
            "name": parts[9],
            "camera_tr_world": camera_tr_world,
            "world_tr_camera": world_tr_camera,
            "center_world": world_tr_camera[:3, 3],
            "point2d_count": len(point_ids),
            "triangulated_point_count": sum(1 for p in point_ids if p != "-1"),
        }
    return images


def read_colmap_camera_params(path):
    rows = []
    if not Path(path).exists():
        return rows
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append({
            "camera_id": int(parts[0]),
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(v) for v in parts[4:]],
        })
    return rows


def run_one_frame(task):
    args, manifest_payload, frame = task
    manifest = manifest_payload
    run_dir = Path(args["output_root"]) / "runs" / f"frame_{frame:04d}"
    if run_dir.exists() and args["overwrite"]:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "frame": frame,
        "run_dir": str(run_dir),
        "status": "started",
        "return_codes": {},
        "registered_count": 0,
        "points3d_count": 0,
        "best_model": "",
        "elapsed_sec": None,
    }
    start = time.time()
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    colmap_threads = int(args.get("colmap_threads", 0) or 0)
    if colmap_threads > 0:
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = str(colmap_threads)

    try:
        image_dir, _records = prepare_frame_images(run_dir, manifest, frame, args["image_mode"])
    except Exception as exc:
        summary["status"] = "prepare_failed"
        summary["error"] = str(exc)
        summary["elapsed_sec"] = time.time() - start
        return summary

    database_path = run_dir / "database.db"
    sparse_dir = run_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    camera_params = f"{args['focal']},{args['cx']},{args['cy']},{args['radial']}"
    feature_cmd = [
        args["colmap_bin"],
        "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.camera_params", camera_params,
        "--FeatureExtraction.use_gpu", "0",
        "--FeatureExtraction.num_threads", str(colmap_threads),
        "--FeatureExtraction.max_image_size", str(args["max_image_size"]),
        "--SiftExtraction.max_num_features", str(args["max_num_features"]),
    ]
    rc = run_command(feature_cmd, run_dir, run_dir / "feature_extractor.log", env)
    summary["return_codes"]["feature_extractor"] = rc
    if rc != 0:
        summary["status"] = "feature_failed"
        summary["elapsed_sec"] = time.time() - start
        return summary

    matcher_cmd = [
        args["colmap_bin"],
        "exhaustive_matcher",
        "--database_path", database_path,
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.num_threads", str(colmap_threads),
        "--FeatureMatching.guided_matching", "1",
        "--FeatureMatching.max_num_matches", str(args["max_num_matches"]),
    ]
    rc = run_command(matcher_cmd, run_dir, run_dir / "exhaustive_matcher.log", env)
    summary["return_codes"]["exhaustive_matcher"] = rc
    if rc != 0:
        summary["status"] = "matcher_failed"
        summary["elapsed_sec"] = time.time() - start
        return summary

    mapper_cmd = [
        args["colmap_bin"],
        "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
        "--Mapper.min_num_matches", str(args["mapper_min_matches"]),
        "--Mapper.init_min_num_inliers", str(args["mapper_min_inliers"]),
        "--Mapper.abs_pose_min_num_inliers", str(args["mapper_min_inliers"]),
        "--Mapper.num_threads", str(colmap_threads),
        "--Mapper.ba_refine_focal_length", "1" if args["refine_focal"] else "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1" if args["refine_radial"] else "0",
        "--Mapper.ba_global_max_num_iterations", str(args["ba_iterations"]),
    ]
    rc = run_command(mapper_cmd, run_dir, run_dir / "mapper.log", env)
    summary["return_codes"]["mapper"] = rc
    if rc != 0:
        summary["status"] = "mapper_failed"
        summary["elapsed_sec"] = time.time() - start
        return summary

    candidates = []
    for model_dir in sorted(p for p in sparse_dir.iterdir() if p.is_dir()):
        txt_dir = run_dir / "sparse_txt" / model_dir.name
        txt_dir.mkdir(parents=True, exist_ok=True)
        convert_cmd = [
            args["colmap_bin"],
            "model_converter",
            "--input_path", model_dir,
            "--output_path", txt_dir,
            "--output_type", "TXT",
        ]
        rc = run_command(convert_cmd, run_dir, run_dir / f"model_converter_{model_dir.name}.log", env)
        if rc != 0:
            continue
        images = load_colmap_images(txt_dir / "images.txt")
        candidates.append({
            "model": model_dir.name,
            "txt_dir": str(txt_dir),
            "registered_count": len(images),
            "points3d_count": read_points3d_count(txt_dir / "points3D.txt"),
            "images": images,
            "cameras": read_colmap_camera_params(txt_dir / "cameras.txt"),
        })
    if not candidates:
        summary["status"] = "no_model"
        summary["elapsed_sec"] = time.time() - start
        return summary
    best = max(candidates, key=lambda row: (row["registered_count"], row["points3d_count"]))
    summary.update({
        "status": "mapped",
        "best_model": best["model"],
        "best_txt_dir": best["txt_dir"],
        "registered_count": best["registered_count"],
        "registered_labels": sorted(best["images"]),
        "points3d_count": best["points3d_count"],
        "camera_params": best["cameras"],
        "elapsed_sec": time.time() - start,
    })
    return summary


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
    return scale, rotation, translation, singular_values, residuals


def rotation_angle_deg(rotation):
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(value)))


def average_rotations(rotations):
    if not rotations:
        return np.eye(3, dtype=np.float64)
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


def robust_median_pose(votes, center_gate_m, min_votes):
    if len(votes) < min_votes:
        return None
    centers = np.asarray([vote["center"] for vote in votes], dtype=np.float64)
    median_center = np.median(centers, axis=0)
    residuals = np.linalg.norm(centers - median_center[None, :], axis=1)
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    gate = max(center_gate_m, med + 3.0 * 1.4826 * mad)
    inliers = [vote for vote, residual in zip(votes, residuals) if residual <= gate]
    if len(inliers) < min_votes:
        return None
    inlier_centers = np.asarray([vote["center"] for vote in inliers], dtype=np.float64)
    center = np.median(inlier_centers, axis=0)
    rotation = average_rotations([vote["rig_R_camera"] for vote in inliers])
    center_residuals = np.linalg.norm(inlier_centers - center[None, :], axis=1)
    rotation_residuals = [
        rotation_angle_deg(vote["rig_R_camera"] @ rotation.T)
        for vote in inliers
    ]
    rig_tr_camera = pose_matrix(rotation, center)
    return {
        "camera_tr_rig": invert_pose(rig_tr_camera),
        "center": center,
        "raw_votes": len(votes),
        "inlier_votes": len(inliers),
        "center_median_residual_m": float(np.median(center_residuals)),
        "center_p90_residual_m": float(np.percentile(center_residuals, 90)),
        "rotation_median_residual_deg": float(np.median(rotation_residuals)),
        "rotation_p90_residual_deg": float(np.percentile(rotation_residuals, 90)),
        "track_median": float(np.median([vote["tracks"] for vote in inliers])),
    }


def build_votes(args, manifest, frame_summaries, anchor_centers):
    labels = [row["camera_id"] for row in manifest]
    accepted_runs = []
    votes_by_label = {label: [] for label in labels}
    anchor_labels = list(anchor_centers)
    target = np.asarray([anchor_centers[label] for label in anchor_labels], dtype=np.float64)
    for summary in frame_summaries:
        if summary.get("status") != "mapped":
            continue
        images = load_colmap_images(Path(summary["best_txt_dir"]) / "images.txt")
        missing = [label for label in anchor_labels if label not in images]
        if missing:
            summary["vote_status"] = "missing_anchors"
            summary["missing_anchors"] = missing
            continue
        source = np.asarray([images[label]["center_world"] for label in anchor_labels], dtype=np.float64)
        scale, rotation, translation, singular_values, residuals = umeyama_similarity(source, target)
        anchor_rms = float(np.sqrt(np.mean(residuals ** 2)))
        summary["anchor_rms_m"] = anchor_rms
        summary["sim3_scale"] = scale
        summary["sim3_singular_values"] = singular_values.tolist()
        if not math.isfinite(scale) or scale <= 0 or anchor_rms > args.max_anchor_rms_m:
            summary["vote_status"] = "bad_anchor_alignment"
            continue
        accepted_runs.append(summary)
        summary["vote_status"] = "accepted"
        for label, image in images.items():
            center = scale * rotation @ image["center_world"] + translation
            if np.linalg.norm(center) > args.max_center_norm_m:
                continue
            if image["triangulated_point_count"] < args.min_tracks_per_vote:
                continue
            rig_R_camera = rotation @ image["world_tr_camera"][:3, :3]
            votes_by_label[label].append({
                "frame": summary["frame"],
                "center": center,
                "rig_R_camera": rig_R_camera,
                "tracks": image["triangulated_point_count"],
                "point2d_count": image["point2d_count"],
            })
    voted = {}
    for label, votes in votes_by_label.items():
        result = robust_median_pose(votes, args.center_vote_gate_m, args.min_votes_per_camera)
        if result is not None:
            voted[label] = result
    return accepted_runs, votes_by_label, voted


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(args, manifest, frames, frame_summaries, accepted_runs, votes_by_label, voted):
    output_root = Path(args.output_root)
    labels = [row["camera_id"] for row in manifest]
    poses = []
    camera_rows = []
    for row in manifest:
        label = row["camera_id"]
        result = voted.get(label)
        poses.append(result["camera_tr_rig"] if result else None)
        if result:
            center = result["center"]
            camera_rows.append({
                "camera_index": row["camera_index"],
                "camera_id": label,
                "status": "voted",
                "raw_votes": result["raw_votes"],
                "inlier_votes": result["inlier_votes"],
                "center_x_m": f"{center[0]:.8g}",
                "center_y_m": f"{center[1]:.8g}",
                "center_z_m": f"{center[2]:.8g}",
                "center_median_residual_m": f"{result['center_median_residual_m']:.8g}",
                "center_p90_residual_m": f"{result['center_p90_residual_m']:.8g}",
                "rotation_median_residual_deg": f"{result['rotation_median_residual_deg']:.8g}",
                "rotation_p90_residual_deg": f"{result['rotation_p90_residual_deg']:.8g}",
                "track_median": f"{result['track_median']:.8g}",
            })
        else:
            camera_rows.append({
                "camera_index": row["camera_index"],
                "camera_id": label,
                "status": "insufficient_votes",
                "raw_votes": len(votes_by_label[label]),
                "inlier_votes": 0,
                "center_x_m": "",
                "center_y_m": "",
                "center_z_m": "",
                "center_median_residual_m": "",
                "center_p90_residual_m": "",
                "rotation_median_residual_deg": "",
                "rotation_p90_residual_deg": "",
                "track_median": "",
            })
    write_pose_yaml(output_root / "camera_tr_rig_voted.yaml", poses)
    write_tsv(output_root / "camera_vote_summary.tsv", camera_rows, [
        "camera_index", "camera_id", "status", "raw_votes", "inlier_votes",
        "center_x_m", "center_y_m", "center_z_m",
        "center_median_residual_m", "center_p90_residual_m",
        "rotation_median_residual_deg", "rotation_p90_residual_deg", "track_median",
    ])
    write_tsv(output_root / "frame_run_summary.tsv", [
        {
            "frame": row.get("frame"),
            "status": row.get("status"),
            "vote_status": row.get("vote_status", ""),
            "registered_count": row.get("registered_count", 0),
            "points3d_count": row.get("points3d_count", 0),
            "anchor_rms_m": row.get("anchor_rms_m", ""),
            "sim3_scale": row.get("sim3_scale", ""),
            "elapsed_sec": row.get("elapsed_sec", ""),
            "run_dir": row.get("run_dir", ""),
        }
        for row in frame_summaries
    ], [
        "frame", "status", "vote_status", "registered_count", "points3d_count",
        "anchor_rms_m", "sim3_scale", "elapsed_sec", "run_dir",
    ])
    summary = {
        "frames_requested": frames,
        "frame_count": len(frames),
        "mapped_count": sum(1 for row in frame_summaries if row.get("status") == "mapped"),
        "accepted_run_count": len(accepted_runs),
        "voted_camera_count": len(voted),
        "camera_count": len(manifest),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_html(output_root / "index.html", summary, camera_rows, frame_summaries)


def fmt(value, digits=3):
    if value == "" or value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def write_html(path, summary, camera_rows, frame_rows):
    camera_html = []
    for row in camera_rows:
        camera_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['camera_index']))}</td>"
            f"<td>{html.escape(row['camera_id'])}</td>"
            f"<td>{html.escape(row['status'])}</td>"
            f"<td>{row['raw_votes']}</td>"
            f"<td>{row['inlier_votes']}</td>"
            f"<td>{fmt(row['center_x_m'])}</td>"
            f"<td>{fmt(row['center_y_m'])}</td>"
            f"<td>{fmt(row['center_z_m'])}</td>"
            f"<td>{fmt(row['center_median_residual_m'])}</td>"
            f"<td>{fmt(row['rotation_median_residual_deg'])}</td>"
            f"<td>{fmt(row['track_median'], 1)}</td>"
            "</tr>"
        )
    frame_html = []
    for row in frame_rows:
        frame_html.append(
            "<tr>"
            f"<td>{row.get('frame')}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('vote_status', '')))}</td>"
            f"<td>{row.get('registered_count', 0)}</td>"
            f"<td>{row.get('points3d_count', 0)}</td>"
            f"<td>{fmt(row.get('anchor_rms_m', ''))}</td>"
            f"<td>{fmt(row.get('sim3_scale', ''))}</td>"
            f"<td>{fmt(row.get('elapsed_sec', ''), 1)}</td>"
            "</tr>"
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Outer COLMAP Frame Vote</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2328; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin-top: 28px; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
    th {{ background: #f6f8fa; }}
    td:nth-child(2), td:nth-child(3), th:nth-child(2), th:nth-child(3) {{ text-align: left; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 18px 0; }}
    .metric {{ border: 1px solid #d0d7de; border-radius: 6px; padding: 10px; }}
    .metric strong {{ display: block; font-size: 22px; }}
    .metric span {{ color: #57606a; font-size: 12px; }}
    .note {{ color: #57606a; line-height: 1.5; max-width: 980px; }}
  </style>
</head>
<body>
  <h1>Outer COLMAP Frame Vote</h1>
  <p class="note">Each synchronized whole frame is reconstructed independently with shared/similar intrinsics. Per-frame COLMAP gauges are aligned with the 4-1/4-2/4-3 bridge anchors, then camera poses are robust-voted across frames. This is a rough initializer, not final AprilTag BA.</p>
  <div class="grid">
    <div class="metric"><strong>{summary['frame_count']}</strong><span>sampled frames</span></div>
    <div class="metric"><strong>{summary['mapped_count']}</strong><span>mapped runs</span></div>
    <div class="metric"><strong>{summary['accepted_run_count']}</strong><span>accepted aligned runs</span></div>
    <div class="metric"><strong>{summary['voted_camera_count']}/{summary['camera_count']}</strong><span>voted cameras</span></div>
  </div>
  <h2>Camera Vote Summary</h2>
  <table>
    <thead><tr><th>Index</th><th>Camera</th><th>Status</th><th>Raw votes</th><th>Inliers</th><th>X m</th><th>Y m</th><th>Z m</th><th>Center med m</th><th>Rot med deg</th><th>Median tracks</th></tr></thead>
    <tbody>{''.join(camera_html)}</tbody>
  </table>
  <h2>Frame Runs</h2>
  <table>
    <thead><tr><th>Frame</th><th>Status</th><th>Vote status</th><th>Registered</th><th>Points3D</th><th>Anchor RMS m</th><th>Sim3 scale</th><th>Sec</th></tr></thead>
    <tbody>{''.join(frame_html)}</tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--anchor-pose-yaml", required=True, type=Path)
    parser.add_argument("--anchor-label-to-pose-index", default="4-1:8,4-2:9,4-3:10")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--frames", default="")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--image-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--focal", type=float, default=3800.0)
    parser.add_argument("--cx", type=float, default=2048.0)
    parser.add_argument("--cy", type=float, default=1500.0)
    parser.add_argument("--radial", type=float, default=0.0)
    parser.add_argument("--refine-focal", action="store_true")
    parser.add_argument("--refine-radial", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=2000)
    parser.add_argument("--max-num-features", type=int, default=12000)
    parser.add_argument("--max-num-matches", type=int, default=32768)
    parser.add_argument(
        "--colmap-threads",
        type=int,
        default=0,
        help="Threads per COLMAP child process. 0 auto-sizes from CPU count and --jobs.",
    )
    parser.add_argument("--mapper-min-matches", type=int, default=8)
    parser.add_argument("--mapper-min-inliers", type=int, default=15)
    parser.add_argument("--ba-iterations", type=int, default=30)
    parser.add_argument("--max-anchor-rms-m", type=float, default=0.35)
    parser.add_argument("--max-center-norm-m", type=float, default=8.0)
    parser.add_argument("--min-tracks-per-vote", type=int, default=10)
    parser.add_argument("--min-votes-per-camera", type=int, default=4)
    parser.add_argument("--center-vote-gate-m", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.colmap_bin = resolve_colmap_bin(args.colmap_bin)

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.colmap_threads <= 0:
        cpu_count = os.cpu_count() or 8
        args.colmap_threads = max(1, min(8, cpu_count // max(1, args.jobs)))
    manifest = read_manifest(args.manifest)
    attach_source_files(manifest)
    frames = select_frames(manifest, args)
    (args.output_root / "selected_frames.txt").write_text(
        "\n".join(str(frame) for frame in frames) + "\n", encoding="utf-8")

    label_to_pose_index = parse_label_pose_indices(args.anchor_label_to_pose_index)
    anchor_centers = load_anchor_centers(args.anchor_pose_yaml, label_to_pose_index)
    payload_args = vars(args).copy()
    payload_args["manifest"] = str(args.manifest)
    payload_args["output_root"] = str(args.output_root)
    payload_args["colmap_bin"] = str(args.colmap_bin)
    tasks = [(payload_args, manifest, frame) for frame in frames]
    frame_summaries = []
    if args.jobs <= 1:
        for task in tasks:
            summary = run_one_frame(task)
            frame_summaries.append(summary)
            print(json.dumps({
                "frame": summary.get("frame"),
                "status": summary.get("status"),
                "error": summary.get("error", ""),
                "registered": summary.get("registered_count"),
                "points3D": summary.get("points3d_count"),
                "elapsed_sec": summary.get("elapsed_sec"),
            }, sort_keys=True))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one_frame, task) for task in tasks]
            for future in as_completed(futures):
                summary = future.result()
                frame_summaries.append(summary)
                print(json.dumps({
                    "frame": summary.get("frame"),
                    "status": summary.get("status"),
                    "error": summary.get("error", ""),
                    "registered": summary.get("registered_count"),
                    "points3D": summary.get("points3d_count"),
                    "elapsed_sec": summary.get("elapsed_sec"),
                }, sort_keys=True))
    frame_summaries.sort(key=lambda row: row["frame"])
    accepted_runs, votes_by_label, voted = build_votes(args, manifest, frame_summaries, anchor_centers)
    write_outputs(args, manifest, frames, frame_summaries, accepted_runs, votes_by_label, voted)
    print(json.dumps({
        "output_root": str(args.output_root),
        "frames": frames,
        "mapped_count": sum(1 for row in frame_summaries if row.get("status") == "mapped"),
        "accepted_run_count": len(accepted_runs),
        "voted_camera_count": len(voted),
        "camera_tr_rig_voted": str(args.output_root / "camera_tr_rig_voted.yaml"),
        "report": str(args.output_root / "index.html"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
