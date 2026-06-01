#!/usr/bin/env python3
"""Vote a stable outer rig from already completed single-frame COLMAP runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path

import numpy as np

import run_outer_colmap_frame_vote as base


def parse_frames(text):
    if not text:
        return None
    frames = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            frames.extend(range(int(start), int(end) + 1))
        else:
            frames.append(int(chunk))
    return set(frames)


def parse_frame_from_run_dir(path):
    name = Path(path).name
    if not name.startswith("frame_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except ValueError:
        return None


def discover_completed_runs(runs_root, frames=None, max_runs=0):
    runs_root = Path(runs_root)
    summaries = []
    for run_dir in sorted((runs_root / "runs").glob("frame_*")):
        frame = parse_frame_from_run_dir(run_dir)
        if frame is None:
            continue
        if frames is not None and frame not in frames:
            continue

        candidates = []
        sparse_txt_root = run_dir / "sparse_txt"
        if not sparse_txt_root.exists():
            continue
        for txt_dir in sorted(path for path in sparse_txt_root.iterdir() if path.is_dir()):
            images_path = txt_dir / "images.txt"
            if not images_path.exists():
                continue
            try:
                images = base.load_colmap_images(images_path)
            except Exception:
                continue
            candidates.append({
                "txt_dir": str(txt_dir),
                "registered_count": len(images),
                "points3d_count": base.read_points3d_count(txt_dir / "points3D.txt"),
                "model": txt_dir.name,
            })
        if not candidates:
            continue

        best = max(candidates, key=lambda row: (row["registered_count"], row["points3d_count"]))
        summaries.append({
            "frame": frame,
            "run_dir": str(run_dir),
            "status": "mapped",
            "best_model": best["model"],
            "best_txt_dir": best["txt_dir"],
            "registered_count": best["registered_count"],
            "points3d_count": best["points3d_count"],
        })

    summaries.sort(key=lambda row: row["frame"])
    if max_runs and max_runs > 0:
        summaries = summaries[:max_runs]
    return summaries


def rotation_residual_deg(rotation_a, rotation_b):
    return base.rotation_angle_deg(rotation_a @ rotation_b.T)


def ransac_pose(votes, min_votes, center_threshold_m, rotation_threshold_deg):
    if len(votes) < min_votes:
        return None

    best = None
    centers = np.asarray([vote["center"] for vote in votes], dtype=np.float64)
    for hypothesis_index, hypothesis in enumerate(votes):
        center_residuals = np.linalg.norm(centers - hypothesis["center"][None, :], axis=1)
        rotation_residuals = np.asarray([
            rotation_residual_deg(vote["rig_R_camera"], hypothesis["rig_R_camera"])
            for vote in votes
        ], dtype=np.float64)
        inlier_indices = [
            index for index, (center_error, rotation_error) in enumerate(zip(center_residuals, rotation_residuals))
            if center_error <= center_threshold_m and rotation_error <= rotation_threshold_deg
        ]
        if len(inlier_indices) < min_votes:
            continue
        score = (
            len(inlier_indices),
            -float(np.median(center_residuals[inlier_indices])),
            -float(np.median(rotation_residuals[inlier_indices])),
            float(sum(votes[index]["tracks"] for index in inlier_indices)),
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "hypothesis_index": hypothesis_index,
                "inlier_indices": inlier_indices,
            }

    if best is None:
        return None

    inliers = [votes[index] for index in best["inlier_indices"]]
    inlier_centers = np.asarray([vote["center"] for vote in inliers], dtype=np.float64)
    center = np.median(inlier_centers, axis=0)
    rotation = base.average_rotations([vote["rig_R_camera"] for vote in inliers])

    center_residuals = np.linalg.norm(inlier_centers - center[None, :], axis=1)
    rotation_residuals = [
        rotation_residual_deg(vote["rig_R_camera"], rotation)
        for vote in inliers
    ]
    rig_tr_camera = base.pose_matrix(rotation, center)
    return {
        "camera_tr_rig": base.invert_pose(rig_tr_camera),
        "center": center,
        "raw_votes": len(votes),
        "inlier_votes": len(inliers),
        "inlier_frames": [vote["frame"] for vote in inliers],
        "hypothesis_frame": votes[best["hypothesis_index"]]["frame"],
        "center_median_residual_m": float(np.median(center_residuals)),
        "center_p90_residual_m": float(np.percentile(center_residuals, 90)),
        "center_max_residual_m": float(np.max(center_residuals)),
        "rotation_median_residual_deg": float(np.median(rotation_residuals)),
        "rotation_p90_residual_deg": float(np.percentile(rotation_residuals, 90)),
        "rotation_max_residual_deg": float(np.max(rotation_residuals)),
        "track_median": float(np.median([vote["tracks"] for vote in inliers])),
    }


def run_ransac(manifest, votes_by_label, args):
    voted = {}
    for row in manifest:
        label = row["camera_id"]
        result = ransac_pose(
            votes_by_label.get(label, []),
            args.min_votes_per_camera,
            args.ransac_center_threshold_m,
            args.ransac_rotation_threshold_deg,
        )
        if result is not None:
            voted[label] = result
    return voted


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def link_or_copy_image(src, dst):
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(Path(src).resolve(), dst)


def export_camera_images(output_root, first_summary, manifest):
    image_dir = Path(output_root) / "viewer" / "camera_images"
    source_dir = Path(first_summary["run_dir"]) / "images"
    for row in manifest:
        pattern = f"cam{row['camera_index']:02d}_{row['camera_id']}_*.jpg"
        matches = sorted(source_dir.glob(pattern))
        if not matches:
            continue
        link_or_copy_image(matches[0], image_dir / matches[0].name)
    return image_dir


def write_outputs(args, manifest, frame_summaries, accepted_runs, votes_by_label, voted):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
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
                "status": "ransac_voted",
                "raw_votes": result["raw_votes"],
                "inlier_votes": result["inlier_votes"],
                "inlier_fraction": f"{result['inlier_votes'] / result['raw_votes']:.8g}",
                "hypothesis_frame": result["hypothesis_frame"],
                "inlier_frames": ",".join(str(frame) for frame in result["inlier_frames"]),
                "center_x_m": f"{center[0]:.8g}",
                "center_y_m": f"{center[1]:.8g}",
                "center_z_m": f"{center[2]:.8g}",
                "center_median_residual_m": f"{result['center_median_residual_m']:.8g}",
                "center_p90_residual_m": f"{result['center_p90_residual_m']:.8g}",
                "center_max_residual_m": f"{result['center_max_residual_m']:.8g}",
                "rotation_median_residual_deg": f"{result['rotation_median_residual_deg']:.8g}",
                "rotation_p90_residual_deg": f"{result['rotation_p90_residual_deg']:.8g}",
                "rotation_max_residual_deg": f"{result['rotation_max_residual_deg']:.8g}",
                "track_median": f"{result['track_median']:.8g}",
            })
        else:
            camera_rows.append({
                "camera_index": row["camera_index"],
                "camera_id": label,
                "status": "insufficient_or_unstable_votes",
                "raw_votes": len(votes_by_label[label]),
                "inlier_votes": 0,
                "inlier_fraction": "",
                "hypothesis_frame": "",
                "inlier_frames": "",
                "center_x_m": "",
                "center_y_m": "",
                "center_z_m": "",
                "center_median_residual_m": "",
                "center_p90_residual_m": "",
                "center_max_residual_m": "",
                "rotation_median_residual_deg": "",
                "rotation_p90_residual_deg": "",
                "rotation_max_residual_deg": "",
                "track_median": "",
            })

    base.write_pose_yaml(output_root / "camera_tr_rig_ransac.yaml", poses)
    camera_fields = [
        "camera_index", "camera_id", "status", "raw_votes", "inlier_votes", "inlier_fraction",
        "hypothesis_frame", "inlier_frames",
        "center_x_m", "center_y_m", "center_z_m",
        "center_median_residual_m", "center_p90_residual_m", "center_max_residual_m",
        "rotation_median_residual_deg", "rotation_p90_residual_deg", "rotation_max_residual_deg",
        "track_median",
    ]
    write_tsv(output_root / "camera_ransac_summary.tsv", camera_rows, camera_fields)
    write_tsv(output_root / "run_alignment_summary.tsv", [
        {
            "frame": row.get("frame"),
            "status": row.get("status"),
            "vote_status": row.get("vote_status", ""),
            "registered_count": row.get("registered_count", 0),
            "points3d_count": row.get("points3d_count", 0),
            "anchor_rms_m": row.get("anchor_rms_m", ""),
            "sim3_scale": row.get("sim3_scale", ""),
            "run_dir": row.get("run_dir", ""),
            "best_txt_dir": row.get("best_txt_dir", ""),
        }
        for row in frame_summaries
    ], [
        "frame", "status", "vote_status", "registered_count", "points3d_count",
        "anchor_rms_m", "sim3_scale", "run_dir", "best_txt_dir",
    ])
    raw_vote_rows = []
    for row in manifest:
        label = row["camera_id"]
        for vote in votes_by_label.get(label, []):
            center = vote["center"]
            raw_vote_rows.append({
                "camera_index": row["camera_index"],
                "camera_id": label,
                "frame": vote["frame"],
                "tracks": vote["tracks"],
                "point2d_count": vote["point2d_count"],
                "center_x_m": f"{center[0]:.8g}",
                "center_y_m": f"{center[1]:.8g}",
                "center_z_m": f"{center[2]:.8g}",
                "center_norm_m": f"{np.linalg.norm(center):.8g}",
            })
    write_tsv(output_root / "raw_vote_detail.tsv", raw_vote_rows, [
        "camera_index", "camera_id", "frame", "tracks", "point2d_count",
        "center_x_m", "center_y_m", "center_z_m", "center_norm_m",
    ])

    voted_rows = [row for row in camera_rows if row["status"] == "ransac_voted"]
    summary = {
        "source_runs_root": str(Path(args.runs_root).resolve()),
        "frame_count": len(frame_summaries),
        "frames_used": [row["frame"] for row in frame_summaries],
        "mapped_count": sum(1 for row in frame_summaries if row.get("status") == "mapped"),
        "accepted_run_count": len(accepted_runs),
        "voted_camera_count": len(voted),
        "camera_count": len(manifest),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    if voted_rows:
        summary.update({
            "median_inlier_fraction": float(np.median([float(row["inlier_fraction"]) for row in voted_rows])),
            "median_center_residual_m": float(np.median([float(row["center_median_residual_m"]) for row in voted_rows])),
            "median_rotation_residual_deg": float(np.median([float(row["rotation_median_residual_deg"]) for row in voted_rows])),
        })
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_html(output_root / "index.html", summary, camera_rows, frame_summaries)
    return camera_rows, summary


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
            f"<td>{fmt(row['inlier_fraction'], 2)}</td>"
            f"<td>{html.escape(str(row['hypothesis_frame']))}</td>"
            f"<td>{fmt(row['center_x_m'])}</td>"
            f"<td>{fmt(row['center_y_m'])}</td>"
            f"<td>{fmt(row['center_z_m'])}</td>"
            f"<td>{fmt(row['center_median_residual_m'])}</td>"
            f"<td>{fmt(row['center_p90_residual_m'])}</td>"
            f"<td>{fmt(row['rotation_median_residual_deg'])}</td>"
            f"<td>{fmt(row['rotation_p90_residual_deg'])}</td>"
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
            "</tr>"
        )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Outer COLMAP RANSAC Vote</title>
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
  <h1>Outer COLMAP RANSAC Vote</h1>
  <p class="note">Each single-frame COLMAP model is first aligned to the metric bridge anchors 4-1/4-2/4-3. For each camera, every aligned pose vote is tested as a one-sample RANSAC hypothesis; inliers must agree in both center distance and rotation angle. This is a rough outer-rig initializer.</p>
  <div class="grid">
    <div class="metric"><strong>{summary['frame_count']}</strong><span>candidate frames</span></div>
    <div class="metric"><strong>{summary['accepted_run_count']}</strong><span>accepted aligned runs</span></div>
    <div class="metric"><strong>{summary['voted_camera_count']}/{summary['camera_count']}</strong><span>RANSAC voted cameras</span></div>
    <div class="metric"><strong>{fmt(summary.get('median_inlier_fraction', ''), 2)}</strong><span>median inlier fraction</span></div>
  </div>
  <h2>Camera RANSAC Summary</h2>
  <table>
    <thead><tr><th>Index</th><th>Camera</th><th>Status</th><th>Raw</th><th>Inliers</th><th>Frac</th><th>Hyp frame</th><th>X m</th><th>Y m</th><th>Z m</th><th>Center med m</th><th>Center p90 m</th><th>Rot med deg</th><th>Rot p90 deg</th><th>Tracks</th></tr></thead>
    <tbody>{''.join(camera_html)}</tbody>
  </table>
  <h2>Frame Alignment</h2>
  <table>
    <thead><tr><th>Frame</th><th>Status</th><th>Vote status</th><th>Registered</th><th>Points3D</th><th>Anchor RMS m</th><th>Sim3 scale</th></tr></thead>
    <tbody>{''.join(frame_html)}</tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--anchor-pose-yaml", required=True, type=Path)
    parser.add_argument("--anchor-label-to-pose-index", default="4-1:8,4-2:9,4-3:10")
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-runs", type=int, default=16)
    parser.add_argument("--max-anchor-rms-m", type=float, default=0.35)
    parser.add_argument("--max-center-norm-m", type=float, default=8.0)
    parser.add_argument("--min-tracks-per-vote", type=int, default=10)
    parser.add_argument("--min-votes-per-camera", type=int, default=4)
    parser.add_argument("--center-vote-gate-m", type=float, default=0.35)
    parser.add_argument("--ransac-center-threshold-m", type=float, default=0.50)
    parser.add_argument("--ransac-rotation-threshold-deg", type=float, default=15.0)
    parser.add_argument("--export-camera-images", action="store_true")
    args = parser.parse_args()

    manifest = base.read_manifest(args.manifest)
    frames = parse_frames(args.frames)
    frame_summaries = discover_completed_runs(args.runs_root, frames=frames, max_runs=args.max_runs)
    if not frame_summaries:
        raise RuntimeError(f"No completed COLMAP text models found under {args.runs_root}/runs")

    label_to_pose_index = base.parse_label_pose_indices(args.anchor_label_to_pose_index)
    anchor_centers = base.load_anchor_centers(args.anchor_pose_yaml, label_to_pose_index)
    accepted_runs, votes_by_label, _median_voted = base.build_votes(
        args, manifest, frame_summaries, anchor_centers)
    voted = run_ransac(manifest, votes_by_label, args)
    camera_rows, summary = write_outputs(
        args, manifest, frame_summaries, accepted_runs, votes_by_label, voted)
    if args.export_camera_images:
        image_dir = export_camera_images(args.output_root, frame_summaries[0], manifest)
        summary["viewer_camera_image_dir"] = str(image_dir.resolve())
        (Path(args.output_root) / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "output_root": str(args.output_root),
        "frames_used": summary["frames_used"],
        "accepted_run_count": summary["accepted_run_count"],
        "voted_camera_count": summary["voted_camera_count"],
        "median_inlier_fraction": summary.get("median_inlier_fraction"),
        "median_center_residual_m": summary.get("median_center_residual_m"),
        "median_rotation_residual_deg": summary.get("median_rotation_residual_deg"),
        "pose_yaml": str(Path(args.output_root) / "camera_tr_rig_ransac.yaml"),
        "report": str(Path(args.output_root) / "index.html"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
