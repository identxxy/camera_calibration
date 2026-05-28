#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path


FULL_OPENCV_MODEL_ID = 6
PAIR_ID_MOD = 2147483647


def parse_camera_list(text):
    return [int(v) for v in text.split(",") if v.strip()]


def parse_frame_list(text):
    frames = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            begin, end = chunk.split("-", 1)
            frames.extend(range(int(begin), int(end) + 1))
        else:
            frames.append(int(chunk))
    return list(dict.fromkeys(frames))


def read_manifest(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["camera_index"] = int(row["camera_index"])
            rows[row["camera_index"]] = row
    return rows


def read_intrinsics(path):
    text = Path(path).read_text()
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
        candidates = sorted(Path(intrinsics_dir).glob(f"opencv_intrinsics{cam}_{camera_id}.yaml"))
        if not candidates:
            candidates = sorted(Path(intrinsics_dir).glob(f"intrinsics{cam}_{camera_id}.yaml"))
        if not candidates:
            raise FileNotFoundError(f"Missing intrinsics for camera {cam} / {camera_id}")
        result[cam] = read_intrinsics(candidates[0])
    return result


def safe_copy(src, dst):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def prepare_images(run_dir, manifest, camera_indices, frame):
    image_dir = run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_records = []
    missing = []
    for cam in camera_indices:
        row = manifest[cam]
        camera_id = row["camera_id"]
        src = Path(row["source_dir"]) / f"{camera_id}_{frame:04d}.jpg"
        name = f"cam{cam:02d}_{camera_id}_f{frame:04d}.jpg"
        dst = image_dir / name
        if not src.exists():
            missing.append(str(src))
            continue
        safe_copy(src, dst)
        image_records.append({
            "camera_index": cam,
            "camera_id": camera_id,
            "stage_name": row.get("stage_name", ""),
            "image_name": name,
            "source": str(src),
        })
    return image_dir, image_records, missing


def run_command(cmd, cwd, log_path, env=None):
    with open(log_path, "w") as log:
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
    return proc.returncode


def update_database_cameras(database_path, image_records, intrinsics):
    conn = sqlite3.connect(database_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM cameras")
        cam_to_db_id = {}
        for rec in image_records:
            cam = rec["camera_index"]
            if cam not in cam_to_db_id:
                cam_to_db_id[cam] = len(cam_to_db_id) + 1
        for cam, db_camera_id in cam_to_db_id.items():
            intr = intrinsics[cam]
            params_blob = struct.pack("<" + "d" * len(intr["params"]), *intr["params"])
            cur.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (db_camera_id, FULL_OPENCV_MODEL_ID, intr["width"], intr["height"], params_blob, 1),
            )
        for rec in image_records:
            cur.execute(
                "UPDATE images SET camera_id = ? WHERE name = ?",
                (cam_to_db_id[rec["camera_index"]], rec["image_name"]),
            )
        conn.commit()
    finally:
        conn.close()


def decode_pair_id(pair_id):
    image_id2 = pair_id % PAIR_ID_MOD
    image_id1 = (pair_id - image_id2) // PAIR_ID_MOD
    return image_id1, image_id2


def read_pairwise_geometries(database_path):
    if not Path(database_path).exists():
        return []
    conn = sqlite3.connect(database_path)
    try:
        cur = conn.cursor()
        names = {}
        for image_id, name in cur.execute("SELECT image_id, name FROM images"):
            names[image_id] = name
        pairs = []
        for pair_id, rows in cur.execute("SELECT pair_id, rows FROM two_view_geometries"):
            image_id1, image_id2 = decode_pair_id(pair_id)
            pairs.append({
                "image1": names.get(image_id1, str(image_id1)),
                "image2": names.get(image_id2, str(image_id2)),
                "inliers": int(rows),
            })
        return sorted(pairs, key=lambda x: x["inliers"], reverse=True)
    finally:
        conn.close()


def qvec_to_rot(q):
    qw, qx, qy, qz = q
    return [
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ]


def mat_t_vec_mul_transpose(r, t):
    return [
        -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
        -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
        -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
    ]


def parse_images_txt(path):
    centers = {}
    if not Path(path).exists():
        return centers
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            q = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            t = [float(parts[5]), float(parts[6]), float(parts[7])]
            name = parts[9]
            cam_match = re.match(r"cam(\d+)_", name)
            if cam_match:
                r = qvec_to_rot(q)
                centers[int(cam_match.group(1))] = mat_t_vec_mul_transpose(r, t)
        i += 2
    return centers


def read_current_centers(tsv_path):
    centers = {}
    with open(tsv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cam = int(row["camera_index"])
            centers[cam] = [float(row["center_x"]), float(row["center_y"]), float(row["center_z"])]
    return centers


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def distance_matrix_score(colmap_centers, current_centers, swap_1_4=False):
    cams = sorted(set(colmap_centers) & set(current_centers))
    if len(cams) < 4:
        return None
    colmap_d = []
    current_d = []
    for i, cam_i in enumerate(cams):
        for cam_j in cams[i + 1:]:
            mapped_i = 4 if swap_1_4 and cam_i == 1 else 1 if swap_1_4 and cam_i == 4 else cam_i
            mapped_j = 4 if swap_1_4 and cam_j == 1 else 1 if swap_1_4 and cam_j == 4 else cam_j
            if mapped_i not in current_centers or mapped_j not in current_centers:
                continue
            colmap_d.append(dist(colmap_centers[cam_i], colmap_centers[cam_j]))
            current_d.append(dist(current_centers[mapped_i], current_centers[mapped_j]))
    denom = sum(v * v for v in colmap_d)
    if denom <= 0:
        return None
    scale = sum(a * b for a, b in zip(colmap_d, current_d)) / denom
    residuals = [scale * a - b for a, b in zip(colmap_d, current_d)]
    rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return {"camera_indices": cams, "scale": scale, "rmse": rmse, "pair_count": len(residuals)}


def count_points3d(path):
    if not Path(path).exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def run_colmap_for_frame(args, manifest, current_centers, camera_indices, frame, label):
    run_dir = Path(args.output_root) / label / f"frame_{frame:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    image_dir, image_records, missing = prepare_images(run_dir, manifest, camera_indices, frame)
    summary = {
        "label": label,
        "frame": frame,
        "run_dir": str(run_dir),
        "camera_indices": camera_indices,
        "images": image_records,
        "missing": missing,
        "commands": [],
        "return_codes": {},
    }
    if missing or len(image_records) != len(camera_indices):
        summary["status"] = "missing_images"
        return summary

    intrinsics = collect_intrinsics(args.intrinsics_dir, manifest, camera_indices)
    avg_fx = sum(v["params"][0] for v in intrinsics.values()) / len(intrinsics)
    avg_fy = sum(v["params"][1] for v in intrinsics.values()) / len(intrinsics)
    avg_cx = sum(v["params"][2] for v in intrinsics.values()) / len(intrinsics)
    avg_cy = sum(v["params"][3] for v in intrinsics.values()) / len(intrinsics)

    database_path = run_dir / "database.db"
    if database_path.exists():
        database_path.unlink()
    sparse_dir = run_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    txt_dir = run_dir / "sparse_txt"
    txt_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    feature_cmd = [
        args.colmap_bin,
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "0",
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", f"{avg_fx},{avg_fy},{avg_cx},{avg_cy}",
        "--FeatureExtraction.use_gpu", "0",
        "--SiftExtraction.max_image_size", str(args.max_image_size),
    ]
    summary["commands"].append(feature_cmd)
    rc = run_command(feature_cmd, run_dir, run_dir / "feature_extractor.log", env)
    summary["return_codes"]["feature_extractor"] = rc
    if rc != 0:
        summary["status"] = "feature_failed"
        return summary

    update_database_cameras(database_path, image_records, intrinsics)

    matcher_cmd = [
        args.colmap_bin,
        "exhaustive_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.guided_matching", "1",
        "--FeatureMatching.max_num_matches", "32768",
    ]
    summary["commands"].append(matcher_cmd)
    rc = run_command(matcher_cmd, run_dir, run_dir / "exhaustive_matcher.log", env)
    summary["return_codes"]["exhaustive_matcher"] = rc
    summary["pairwise_geometries"] = read_pairwise_geometries(database_path)
    if rc != 0:
        summary["status"] = "matcher_failed"
        return summary

    mapper_cmd = [
        args.colmap_bin,
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.min_num_matches", str(args.mapper_min_matches),
        "--Mapper.init_min_num_inliers", str(args.mapper_min_inliers),
        "--Mapper.abs_pose_min_num_inliers", str(args.mapper_min_inliers),
        "--Mapper.ba_global_max_num_iterations", "20",
    ]
    summary["commands"].append(mapper_cmd)
    rc = run_command(mapper_cmd, run_dir, run_dir / "mapper.log", env)
    summary["return_codes"]["mapper"] = rc

    model_dir = sparse_dir / "0"
    if model_dir.exists():
        convert_cmd = [
            args.colmap_bin,
            "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ]
        summary["commands"].append(convert_cmd)
        rc_convert = run_command(convert_cmd, run_dir, run_dir / "model_converter.log", env)
        summary["return_codes"]["model_converter"] = rc_convert
        centers = parse_images_txt(txt_dir / "images.txt")
        summary["registered_camera_indices"] = sorted(centers)
        summary["registered_count"] = len(centers)
        summary["points3d_count"] = count_points3d(txt_dir / "points3D.txt")
        summary["distance_score_current"] = distance_matrix_score(centers, current_centers, False)
        summary["distance_score_swap_1_4"] = distance_matrix_score(centers, current_centers, True)
        summary["status"] = "mapped" if centers else "no_registered_images"
    else:
        summary["registered_camera_indices"] = []
        summary["registered_count"] = 0
        summary["points3d_count"] = 0
        summary["status"] = "no_model"
    return summary


def write_markdown_report(path, args, all_results):
    lines = [
        "# COLMAP Sanity Check: cam1/cam4 and cam3/cam7",
        "",
        f"- COLMAP binary: `{args.colmap_bin}`",
        f"- Output root: `{args.output_root}`",
        f"- Manifest: `{args.manifest}`",
        f"- Intrinsics: `{args.intrinsics_dir}`",
        f"- Current rig TSV: `{args.current_rig_tsv}`",
        f"- Core cameras: `{args.core_cameras}`",
        f"- Attach cameras: `{args.attach_cameras}`",
        f"- Frames tried: `{args.frames}`",
        "",
        "## Frame Results",
        "",
        "| set | frame | status | registered | points3D | top verified pair inliers | current RMSE | swap cam1/cam4 RMSE |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for res in all_results:
        top_inliers = ""
        if res.get("pairwise_geometries"):
            top_inliers = str(res["pairwise_geometries"][0]["inliers"])
        score = res.get("distance_score_current") or {}
        swap = res.get("distance_score_swap_1_4") or {}
        lines.append(
            f"| {res['label']} | {res['frame']} | {res.get('status','')} | "
            f"{','.join(str(c) for c in res.get('registered_camera_indices', []))} | "
            f"{res.get('points3d_count', 0)} | {top_inliers} | "
            f"{score.get('rmse', '')} | {swap.get('rmse', '')} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- `current RMSE` compares COLMAP pairwise camera-center distances to the current rig labels after one global scale.")
    lines.append("- `swap cam1/cam4 RMSE` repeats the same distance-only comparison after swapping labels 1 and 4 in the current rig.")
    lines.append("- A lower swap RMSE would support a cam1/cam4 label swap; a lower current RMSE supports the existing label assignment.")
    lines.append("- If no model is produced, use `pairwise_geometries` in `summary.json` and per-frame COLMAP logs as failure evidence rather than inferring layout.")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--intrinsics-dir", required=True)
    parser.add_argument("--current-rig-tsv", required=True)
    parser.add_argument("--colmap-bin", required=True)
    parser.add_argument("--frames", default="0-30")
    parser.add_argument("--core-cameras", default="0,1,2,4,5,6")
    parser.add_argument("--attach-cameras", default="3,7")
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--mapper-min-matches", type=int, default=8)
    parser.add_argument("--mapper-min-inliers", type=int, default=15)
    parser.add_argument("--stop-after-core-registered", type=int, default=4)
    parser.add_argument("--max-attach-runs", type=int, default=8)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    current_centers = read_current_centers(args.current_rig_tsv)
    core_cameras = parse_camera_list(args.core_cameras)
    attach_cameras = parse_camera_list(args.attach_cameras)
    frames = parse_frame_list(args.frames)

    all_results = []
    selected_core_frames = []
    for frame in frames:
        res = run_colmap_for_frame(args, manifest, current_centers, core_cameras, frame, "core6")
        all_results.append(res)
        if res.get("registered_count", 0) >= args.stop_after_core_registered:
            selected_core_frames.append(frame)
            break

    attach_frames = selected_core_frames + [f for f in frames if f not in selected_core_frames]
    for frame in attach_frames[:args.max_attach_runs]:
        res = run_colmap_for_frame(
            args,
            manifest,
            current_centers,
            core_cameras + attach_cameras,
            frame,
            "all8_attach",
        )
        all_results.append(res)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    write_markdown_report(output_root / "README.md", args, all_results)


if __name__ == "__main__":
    main()
