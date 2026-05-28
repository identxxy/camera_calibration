#!/usr/bin/env python3
"""Triangulate a COLMAP sparse cloud from fixed calibrated camera poses."""

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path

import yaml


FULL_OPENCV_MODEL_ID = 6


def read_manifest(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["camera_index"] = int(row["camera_index"])
            rows[row["camera_index"]] = row
    return rows


def read_intrinsics(path):
    text = Path(path).read_text(encoding="utf-8")
    width = int(re.search(r"image_width:\s*(\d+)", text).group(1))
    height = int(re.search(r"image_height:\s*(\d+)", text).group(1))
    data = re.findall(r"data:\s*\[([^\]]+)\]", text, flags=re.S)
    if len(data) < 2:
        raise ValueError(f"Cannot parse OpenCV YAML intrinsics: {path}")
    k = [float(x) for x in re.split(r"[,\s]+", data[0].strip()) if x]
    d = [float(x) for x in re.split(r"[,\s]+", data[1].strip()) if x]
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    k1, k2, p1, p2, k3 = (d + [0.0] * 5)[:5]
    return {
        "width": width,
        "height": height,
        "params": [fx, fy, cx, cy, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0],
    }


def collect_intrinsics(intrinsics_dir, manifest, camera_indices):
    result = {}
    for cam in camera_indices:
        camera_id = manifest[cam]["camera_id"]
        root = Path(intrinsics_dir)
        candidates = sorted(root.glob(f"opencv_intrinsics{cam}_{camera_id}.yaml"))
        if not candidates:
            candidates = sorted(root.glob(f"intrinsics{cam}_{camera_id}.yaml"))
        if not candidates:
            raise FileNotFoundError(f"Missing intrinsics for camera {cam} / {camera_id}")
        result[cam] = read_intrinsics(candidates[0])
    return result


def load_camera_tr_world(path):
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pose_count = int(node["pose_count"])
    poses = {}
    for pose in node.get("poses", []):
        idx = int(pose["index"])
        poses[idx] = {
            "qvec": [
                float(pose["qw"]),
                float(pose["qx"]),
                float(pose["qy"]),
                float(pose["qz"]),
            ],
            "tvec": [
                float(pose["tx"]),
                float(pose["ty"]),
                float(pose["tz"]),
            ],
        }
    for idx in range(pose_count):
        if idx not in poses:
            raise ValueError(f"Missing pose index {idx} in {path}")
    return poses


def parse_camera_list(text):
    return [int(v) for v in text.split(",") if v.strip()]


def safe_copy(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def prepare_images(output_dir, manifest, camera_indices, frame):
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for cam in camera_indices:
        row = manifest[cam]
        camera_id = row["camera_id"]
        src = Path(row["source_dir"]) / f"{camera_id}_{frame:04d}.jpg"
        if not src.exists():
            raise FileNotFoundError(src)
        name = f"cam{cam:02d}_{camera_id}_f{frame:04d}.jpg"
        safe_copy(src, image_dir / name)
        records.append({
            "camera_index": cam,
            "camera_id": camera_id,
            "image_name": name,
            "source": str(src),
        })
    return image_dir, records


def run_command(cmd, cwd, log_path, env=None):
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def update_database_cameras(database_path, image_records, intrinsics):
    conn = sqlite3.connect(database_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM cameras")
        for rec in image_records:
            cam = rec["camera_index"]
            db_camera_id = cam + 1
            intr = intrinsics[cam]
            params_blob = struct.pack("<" + "d" * len(intr["params"]), *intr["params"])
            cur.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (db_camera_id, FULL_OPENCV_MODEL_ID, intr["width"], intr["height"], params_blob, 1),
            )
            cur.execute(
                "UPDATE images SET camera_id = ? WHERE name = ?",
                (db_camera_id, rec["image_name"]),
            )
        conn.commit()
    finally:
        conn.close()


def read_database_images(database_path):
    conn = sqlite3.connect(database_path)
    try:
        cur = conn.cursor()
        rows = cur.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        return [{"image_id": int(r[0]), "name": r[1], "camera_id": int(r[2])} for r in rows]
    finally:
        conn.close()


def write_known_pose_model(model_dir, db_images, intrinsics, poses):
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in sorted(intrinsics):
            intr = intrinsics[cam]
            params = " ".join(f"{v:.17g}" for v in intr["params"])
            f.write(f"{cam + 1} FULL_OPENCV {intr['width']} {intr['height']} {params}\n")

    name_to_cam = {}
    for row in db_images:
        m = re.match(r"cam(\d+)_", row["name"])
        if not m:
            raise ValueError(f"Cannot parse camera index from image name {row['name']}")
        name_to_cam[row["name"]] = int(m.group(1))

    with (model_dir / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for row in sorted(db_images, key=lambda x: x["image_id"]):
            cam = name_to_cam[row["name"]]
            pose = poses[cam]
            q = " ".join(f"{v:.17g}" for v in pose["qvec"])
            t = " ".join(f"{v:.17g}" for v in pose["tvec"])
            f.write(f"{row['image_id']} {q} {t} {cam + 1} {row['name']}\n\n")

    (model_dir / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        encoding="utf-8",
    )


def parse_points3d(path, max_error, min_track_len, max_points):
    points = []
    if not Path(path).exists():
        return points
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            error = float(parts[7])
            track_len = max(0, (len(parts) - 8) // 2)
            if max_error > 0 and error > max_error:
                continue
            if track_len < min_track_len:
                continue
            points.append({
                "id": int(parts[0]),
                "xyz": [float(parts[1]), float(parts[2]), float(parts[3])],
                "rgb": [int(parts[4]), int(parts[5]), int(parts[6])],
                "error": error,
                "track_len": track_len,
            })
    points.sort(key=lambda p: (p["error"], -p["track_len"], p["id"]))
    if max_points > 0:
        points = points[:max_points]
    return points


def write_point_cloud_json(path, args, points, image_records):
    payload = {
        "coordinate_frame": "camera0_opencv",
        "source_points3D": str(path.with_name("points3D.txt")),
        "frame": args.frame,
        "camera_indices": parse_camera_list(args.cameras),
        "image_count": len(image_records),
        "point_count": len(points),
        "points": points,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_report(path, args, points, image_records):
    lines = [
        "# Fixed-Pose COLMAP Sparse Cloud",
        "",
        f"- Frame: `{args.frame:04d}`",
        f"- Cameras: `{args.cameras}`",
        f"- COLMAP: `{args.colmap_bin}`",
        f"- Known poses: `{args.pose_yaml}`",
        f"- Intrinsics: `{args.intrinsics_dir}`",
        f"- Points after filtering: `{len(points)}`",
        "",
        "COLMAP was used for feature extraction, exhaustive matching, and point triangulation only.",
        "The input camera poses are the current calibrated `camera_tr_camera0` transforms, so the sparse cloud is already in camera0/OpenCV coordinates.",
        "",
        "## Images",
        "",
        "| Camera | Image | Source |",
        "| ---: | --- | --- |",
    ]
    for rec in image_records:
        lines.append(f"| {rec['camera_index']} | `{rec['image_name']}` | `{rec['source']}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--intrinsics-dir", required=True)
    parser.add_argument("--pose-yaml", required=True)
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--cameras", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-image-size", type=int, default=2400)
    parser.add_argument("--max-num-features", type=int, default=12000)
    parser.add_argument("--min-num-matches", type=int, default=4)
    parser.add_argument("--max-reproj-error", type=float, default=8.0)
    parser.add_argument("--min-track-len", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=50000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    camera_indices = parse_camera_list(args.cameras)
    manifest = read_manifest(args.manifest)
    intrinsics = collect_intrinsics(args.intrinsics_dir, manifest, camera_indices)
    poses = load_camera_tr_world(args.pose_yaml)
    image_dir, image_records = prepare_images(output_dir, manifest, camera_indices, args.frame)

    database_path = output_dir / "database.db"
    input_model_dir = output_dir / "known_pose_model"
    sparse_dir = output_dir / "sparse"
    sparse_txt_dir = output_dir / "sparse_txt"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    first_intr = intrinsics[camera_indices[0]]
    dummy_params = ",".join(f"{v:.17g}" for v in first_intr["params"][:4])
    feature_cmd = [
        args.colmap_bin,
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "0",
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", dummy_params,
        "--FeatureExtraction.use_gpu", "0",
        "--SiftExtraction.max_image_size", str(args.max_image_size),
        "--SiftExtraction.max_num_features", str(args.max_num_features),
    ]
    run_command(feature_cmd, output_dir, output_dir / "feature_extractor.log", env)
    update_database_cameras(database_path, image_records, intrinsics)

    matcher_cmd = [
        args.colmap_bin,
        "exhaustive_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.guided_matching", "1",
        "--FeatureMatching.max_num_matches", "32768",
    ]
    run_command(matcher_cmd, output_dir, output_dir / "exhaustive_matcher.log", env)

    db_images = read_database_images(database_path)
    write_known_pose_model(input_model_dir, db_images, intrinsics, poses)

    sparse_dir.mkdir(parents=True, exist_ok=True)
    triangulator_cmd = [
        args.colmap_bin,
        "point_triangulator",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--input_path", str(input_model_dir),
        "--output_path", str(sparse_dir),
        "--clear_points", "1",
        "--refine_intrinsics", "0",
        "--Mapper.fix_existing_frames", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.min_num_matches", str(args.min_num_matches),
        "--Mapper.tri_ignore_two_view_tracks", "0",
        "--Mapper.tri_min_angle", "0.1",
        "--Mapper.filter_max_reproj_error", str(args.max_reproj_error),
    ]
    run_command(triangulator_cmd, output_dir, output_dir / "point_triangulator.log", env)

    sparse_txt_dir.mkdir(parents=True, exist_ok=True)
    convert_cmd = [
        args.colmap_bin,
        "model_converter",
        "--input_path", str(sparse_dir),
        "--output_path", str(sparse_txt_dir),
        "--output_type", "TXT",
    ]
    run_command(convert_cmd, output_dir, output_dir / "model_converter.log", env)

    points = parse_points3d(
        sparse_txt_dir / "points3D.txt",
        args.max_reproj_error,
        args.min_track_len,
        args.max_points,
    )
    write_point_cloud_json(output_dir / "sparse_points_camera0.json", args, points, image_records)
    write_report(output_dir / "README.md", args, points, image_records)
    print(json.dumps({
        "output_dir": str(output_dir),
        "point_count": len(points),
        "point_json": str(output_dir / "sparse_points_camera0.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
