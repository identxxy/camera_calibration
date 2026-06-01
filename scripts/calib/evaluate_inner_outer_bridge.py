#!/usr/bin/env python3
"""Evaluate a large-board bridge between an inner rig and outer rig poses."""

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path

import numpy as np


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
    if trace > 0.0:
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
        q *= -1.0
    return q


def matrix_to_quat_xyzw(rotation):
    q = matrix_to_quat_wxyz(rotation)
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_wxyz_to_matrix_direct(q):
    return quat_wxyz_to_matrix(q[0], q[1], q[2], q[3])


def average_rotations(rotations):
    if not rotations:
        return np.eye(3, dtype=np.float64)
    accumulator = np.zeros((4, 4), dtype=np.float64)
    reference = None
    for rotation in rotations:
        q = matrix_to_quat_wxyz(rotation)
        if reference is None:
            reference = q
        elif np.dot(reference, q) < 0:
            q *= -1.0
        accumulator += np.outer(q, q)
    _, vectors = np.linalg.eigh(accumulator)
    q = vectors[:, -1]
    if q[0] < 0:
        q *= -1.0
    return quat_wxyz_to_matrix_direct(q)


def pose_matrix(rotation, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def pose_from_xyzw(qx, qy, qz, qw, tx, ty, tz):
    return pose_matrix(quat_xyzw_to_matrix(qx, qy, qz, qw), np.asarray([tx, ty, tz], dtype=np.float64))


def invert_pose(matrix):
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv


def rotation_angle_deg(rotation):
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(value)))


def load_pose_yaml(path):
    text = Path(path).read_text(encoding="utf-8").splitlines()
    pose_count = None
    poses = {}
    current = None
    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pose_count:"):
            pose_count = int(line.split(":", 1)[1].strip())
        elif line.startswith("- index:"):
            if current is not None:
                index = int(current["index"])
                poses[index] = pose_from_xyzw(
                    float(current["qx"]),
                    float(current["qy"]),
                    float(current["qz"]),
                    float(current["qw"]),
                    float(current["tx"]),
                    float(current["ty"]),
                    float(current["tz"]),
                )
            current = {"index": line.split(":", 1)[1].strip()}
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    if current is not None:
        index = int(current["index"])
        poses[index] = pose_from_xyzw(
            float(current["qx"]),
            float(current["qy"]),
            float(current["qz"]),
            float(current["qw"]),
            float(current["tx"]),
            float(current["ty"]),
            float(current["tz"]),
        )
    if pose_count is None:
        raise ValueError(f"Could not find pose_count in {path}")
    return [poses.get(index) for index in range(pose_count)]


def write_pose_yaml(path, poses):
    path = Path(path)
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def map_inner_poses_to_bridge_indices(inner_camera_tr_rig, inner_indices):
    if not inner_indices:
        raise ValueError("--inner_indices must not be empty")
    max_inner_index = max(inner_indices)
    if max_inner_index < len(inner_camera_tr_rig):
        return {
            index: inner_camera_tr_rig[index]
            for index in inner_indices
        }, "direct_bridge_indices"
    if len(inner_camera_tr_rig) == len(inner_indices):
        return {
            bridge_index: inner_camera_tr_rig[source_index]
            for source_index, bridge_index in enumerate(inner_indices)
        }, "compact_inner_rig_remapped_to_bridge_indices"
    raise ValueError(
        "Inner pose file must either contain the requested bridge indices or "
        "be a compact inner rig with the same count as --inner_indices"
    )


def parse_error_value(value):
    if value is None or value == "":
        return float("inf")
    return float(value)


def load_pnp_views(path, max_median_error_px):
    views = []
    with Path(path).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["status"] != "solved":
                continue
            mean_error = parse_error_value(row["mean_error_px"])
            median_error = parse_error_value(row["median_error_px"])
            if not math.isfinite(median_error) or median_error > max_median_error_px:
                continue
            views.append({
                "camera_index": int(row["camera_index"]),
                "imageset_index": int(row["imageset_index"]),
                "user_id": row["user_id"],
                "points": int(row["points"]),
                "inliers": int(row["inliers"]),
                "mean_error_px": mean_error,
                "median_error_px": median_error,
                "camera_tr_board": pose_from_xyzw(
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                ),
            })
    return views


def parse_colmap_label(name):
    match = re.search(r"cam\d+_([^_]+)_f\d+", name)
    if match:
        return match.group(1)
    return Path(name).stem


def load_colmap_images(path):
    images = {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        point_line = lines[index].strip() if index < len(lines) else ""
        if index < len(lines):
            index += 1
        point_parts = point_line.split()
        point_ids = point_parts[2::3]
        triangulated_count = sum(1 for point_id in point_ids if point_id != "-1")
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = parts[9]
        rotation = quat_wxyz_to_matrix(qw, qx, qy, qz)
        camera_tr_world = pose_matrix(rotation, np.asarray([tx, ty, tz], dtype=np.float64))
        label = parse_colmap_label(name)
        images[label] = {
            "name": name,
            "camera_tr_world": camera_tr_world,
            "world_tr_camera": invert_pose(camera_tr_world),
            "center_world": invert_pose(camera_tr_world)[:3, 3],
            "point2d_count": len(point_ids),
            "triangulated_point_count": triangulated_count,
        }
    return images


def mean_pose(poses):
    translations = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    rotations = [pose[:3, :3] for pose in poses]
    return pose_matrix(average_rotations(rotations), translations.mean(axis=0))


def average_poses(poses):
    if not poses:
        raise ValueError("Cannot average an empty pose list")
    return mean_pose(poses)


def median_camera_pose(camera_tr_rig_votes):
    centers = np.asarray([invert_pose(pose)[:3, 3] for pose in camera_tr_rig_votes], dtype=np.float64)
    median_center = np.median(centers, axis=0)
    rotation = average_rotations([pose[:3, :3] for pose in camera_tr_rig_votes])
    translation = -rotation @ median_center
    return pose_matrix(rotation, translation), centers


def summarize_outer_votes(label, votes):
    camera_tr_rig, centers = median_camera_pose(votes)
    median_center = invert_pose(camera_tr_rig)[:3, 3]
    center_residuals = np.linalg.norm(centers - median_center[None, :], axis=1)
    rotation_residuals = np.asarray([
        rotation_angle_deg(vote[:3, :3] @ camera_tr_rig[:3, :3].T)
        for vote in votes
    ], dtype=np.float64)
    return {
        "label": label,
        "vote_count": int(len(votes)),
        "center_inner_rig_m": median_center.tolist(),
        "center_residual_median_m": float(np.median(center_residuals)),
        "center_residual_p90_m": float(np.percentile(center_residuals, 90)),
        "rotation_residual_median_deg": float(np.median(rotation_residuals)),
        "rotation_residual_p90_deg": float(np.percentile(rotation_residuals, 90)),
        "camera_tr_inner_rig": camera_tr_rig.tolist(),
    }


def build_outer_final_alignment(outer_summaries, outer_camera_tr_rig, outer_indices):
    votes = []
    per_anchor = {}
    for row, outer_index in zip(outer_summaries, outer_indices):
        if outer_index >= len(outer_camera_tr_rig) or outer_camera_tr_rig[outer_index] is None:
            raise ValueError(f"Outer final pose YAML missing anchor pose index {outer_index}")
        camera_tr_inner_rig = np.asarray(row["camera_tr_inner_rig"], dtype=np.float64)
        camera_tr_outer_rig = np.asarray(outer_camera_tr_rig[outer_index], dtype=np.float64)
        outer_rig_tr_inner_rig = invert_pose(camera_tr_outer_rig) @ camera_tr_inner_rig
        votes.append(outer_rig_tr_inner_rig)
        per_anchor[row["label"]] = {
            "outer_index": int(outer_index),
            "outer_rig_tr_inner_rig_vote": outer_rig_tr_inner_rig.tolist(),
        }

    outer_rig_tr_inner_rig = average_poses(votes)
    center_residuals = []
    rotation_residuals = []
    for row, outer_index in zip(outer_summaries, outer_indices):
        camera_tr_inner_rig_observed = np.asarray(row["camera_tr_inner_rig"], dtype=np.float64)
        camera_tr_inner_rig_predicted = (
            np.asarray(outer_camera_tr_rig[outer_index], dtype=np.float64)
            @ outer_rig_tr_inner_rig
        )
        observed_center = invert_pose(camera_tr_inner_rig_observed)[:3, 3]
        predicted_center = invert_pose(camera_tr_inner_rig_predicted)[:3, 3]
        center_residual = float(np.linalg.norm(predicted_center - observed_center))
        rotation_residual = rotation_angle_deg(
            camera_tr_inner_rig_predicted[:3, :3]
            @ camera_tr_inner_rig_observed[:3, :3].T
        )
        center_residuals.append(center_residual)
        rotation_residuals.append(rotation_residual)
        per_anchor[row["label"]].update({
            "center_residual_m": center_residual,
            "rotation_residual_deg": float(rotation_residual),
        })

    return {
        "status": "ready",
        "anchor_count": int(len(votes)),
        "outer_rig_tr_inner_rig": outer_rig_tr_inner_rig.tolist(),
        "inner_rig_tr_outer_rig": invert_pose(outer_rig_tr_inner_rig).tolist(),
        "center_residual_median_m": float(np.median(center_residuals)),
        "center_residual_max_m": float(np.max(center_residuals)),
        "rotation_residual_median_deg": float(np.median(rotation_residuals)),
        "rotation_residual_max_deg": float(np.max(rotation_residuals)),
        "per_anchor": per_anchor,
    }


def umeyama_similarity(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    variance = np.sum(source_centered ** 2) / source.shape[0]
    scale = float(np.sum(singular_values * np.diag(sign)) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation, singular_values


def triangle_area(points):
    if len(points) != 3:
        return float("nan")
    a, b, c = np.asarray(points, dtype=np.float64)
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))


def pairwise_distance_summary(labels, inner_centers, colmap_centers):
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            inner_distance = float(np.linalg.norm(inner_centers[i] - inner_centers[j]))
            colmap_distance = float(np.linalg.norm(colmap_centers[i] - colmap_centers[j]))
            rows.append({
                "camera_a": labels[i],
                "camera_b": labels[j],
                "inner_distance_m": inner_distance,
                "colmap_distance_units": colmap_distance,
                "distance_ratio_colmap_per_meter": colmap_distance / inner_distance if inner_distance > 0 else float("nan"),
            })
    return rows


def gate_check(name, value, op, threshold, units=""):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        passed = False
    elif op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported gate op: {op}")
    return {
        "name": name,
        "value": value,
        "op": op,
        "threshold": threshold,
        "units": units,
        "pass": bool(passed),
    }


def summarize_gate(checks, pass_status="pass", fail_status="fail"):
    failed = [check for check in checks if not check["pass"]]
    return {
        "status": pass_status if not failed else fail_status,
        "passed": not failed,
        "failed_checks": [check["name"] for check in failed],
        "checks": checks,
    }


def build_quality_gates(summary, args):
    outer_rows = summary["outer_camera_summaries"]
    max_center_p90 = max(row["center_residual_p90_m"] for row in outer_rows)
    max_rotation_p90 = max(row["rotation_residual_p90_deg"] for row in outer_rows)
    min_votes = min(row["vote_count"] for row in outer_rows)
    inner_summary = summary["inner_board_pose_summary"]
    alignment = summary["colmap_alignment"]
    pairwise_ratios = [
        row["distance_ratio_colmap_per_meter"]
        for row in alignment["pairwise_distances"]
        if row["distance_ratio_colmap_per_meter"] > 0
    ]
    pairwise_ratio_spread = (
        max(pairwise_ratios) / min(pairwise_ratios)
        if pairwise_ratios else float("inf")
    )
    min_colmap_tracks = min(
        row["triangulated_point_count"]
        for row in alignment["per_camera"].values()
    )

    metric_checks = [
        gate_check(
            "inner_board_frame_count",
            inner_summary["frame_count"],
            ">=",
            args.bridge_gate_min_inner_frames,
            "frames",
        ),
        gate_check(
            "inner_support_median",
            inner_summary["inner_support_median"],
            ">=",
            args.bridge_gate_min_inner_support_median,
            "cameras",
        ),
        gate_check(
            "outer_vote_count_min",
            min_votes,
            ">=",
            args.min_outer_votes,
            "votes",
        ),
        gate_check(
            "outer_center_residual_p90_max",
            max_center_p90,
            "<=",
            args.bridge_gate_max_center_p90_m,
            "m",
        ),
        gate_check(
            "outer_rotation_residual_p90_max",
            max_rotation_p90,
            "<=",
            args.bridge_gate_max_rotation_p90_deg,
            "deg",
        ),
        gate_check(
            "topdown_triangle_area",
            alignment["inner_triangle_area_m2"],
            ">=",
            args.bridge_gate_min_triangle_area_m2,
            "m^2",
        ),
    ]
    colmap_checks = [
        gate_check(
            "min_colmap_triangulated_tracks",
            min_colmap_tracks,
            ">=",
            args.colmap_gate_min_triangulated_tracks,
            "tracks",
        ),
        gate_check(
            "colmap_pairwise_ratio_spread",
            pairwise_ratio_spread,
            "<=",
            args.colmap_gate_max_pairwise_ratio_spread,
            "ratio",
        ),
        gate_check(
            "colmap_rotation_residual_median",
            alignment["rotation_residual_median_deg"],
            "<=",
            args.colmap_gate_max_rotation_residual_median_deg,
            "deg",
        ),
    ]
    return {
        "metric_bridge": summarize_gate(metric_checks),
        "colmap_prior_diagnostic": summarize_gate(
            colmap_checks,
            pass_status="consistent",
            fail_status="weak_or_inconsistent",
        ),
        "metric_summary": {
            "max_outer_center_residual_p90_m": max_center_p90,
            "max_outer_rotation_residual_p90_deg": max_rotation_p90,
            "min_outer_votes": min_votes,
            "pairwise_ratio_spread": pairwise_ratio_spread,
            "min_colmap_triangulated_tracks": min_colmap_tracks,
        },
    }


def make_html(summary):
    def fmt(value, digits=4):
        if value is None:
            return "n/a"
        if isinstance(value, float) and not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    outer_rows = []
    for row in summary["outer_camera_summaries"]:
        center = row["center_inner_rig_m"]
        col = summary["colmap_alignment"]["per_camera"].get(row["label"], {})
        outer_rows.append(
            "<tr>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{row['vote_count']}</td>"
            f"<td>{fmt(row['center_residual_median_m'])}</td>"
            f"<td>{fmt(row['center_residual_p90_m'])}</td>"
            f"<td>{fmt(row['rotation_residual_median_deg'])}</td>"
            f"<td>{fmt(row['rotation_residual_p90_deg'])}</td>"
            f"<td>{fmt(center[0])}, {fmt(center[1])}, {fmt(center[2])}</td>"
            f"<td>{col.get('triangulated_point_count', 'n/a')}</td>"
            f"<td>{fmt(col.get('rotation_residual_deg'))}</td>"
            f"<td>{fmt(col.get('center_residual_colmap_units'))}</td>"
            "</tr>"
        )
    pairwise_rows = []
    for row in summary["colmap_alignment"]["pairwise_distances"]:
        pairwise_rows.append(
            "<tr>"
            f"<td>{html.escape(row['camera_a'])}-{html.escape(row['camera_b'])}</td>"
            f"<td>{fmt(row['inner_distance_m'])}</td>"
            f"<td>{fmt(row['colmap_distance_units'])}</td>"
            f"<td>{fmt(row['distance_ratio_colmap_per_meter'])}</td>"
            "</tr>"
        )
    inner = summary["inner_board_pose_summary"]
    align = summary["colmap_alignment"]
    gates = summary.get("quality_gates", {})
    metric_gate = gates.get("metric_bridge", {})
    colmap_gate = gates.get("colmap_prior_diagnostic", {})
    outer_final = summary.get("outer_final_alignment") or {}
    outer_final_block = ""
    if outer_final:
        outer_final_block = f"""
  <h2>Outer Final Rig Alignment</h2>
  <div class="grid">
    <div class="metric"><div class="label">Anchors</div><div class="value">{outer_final.get("anchor_count", "n/a")}</div></div>
    <div class="metric"><div class="label">Center med m</div><div class="value">{fmt(outer_final.get("center_residual_median_m"))}</div></div>
    <div class="metric"><div class="label">Center max m</div><div class="value">{fmt(outer_final.get("center_residual_max_m"))}</div></div>
    <div class="metric"><div class="label">Rot med deg</div><div class="value">{fmt(outer_final.get("rotation_residual_median_deg"))}</div></div>
    <div class="metric"><div class="label">Rot max deg</div><div class="value">{fmt(outer_final.get("rotation_residual_max_deg"))}</div></div>
  </div>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Inner/Outer Bridge Check</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2328; }}
    h1 {{ font-size: 26px; margin: 0 0 12px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f6f8fa; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 16px; }}
    .metric {{ border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }}
    .metric .label {{ color: #57606a; font-size: 12px; }}
    .metric .value {{ font-size: 22px; margin-top: 4px; }}
    .note {{ color: #57606a; line-height: 1.5; max-width: 900px; }}
  </style>
</head>
<body>
  <h1>Inner/Outer Bridge Check</h1>
  <p class="note">{html.escape(summary["conclusion"])}</p>
  <div class="grid">
    <div class="metric"><div class="label">Inner board poses</div><div class="value">{inner["frame_count"]}</div></div>
    <div class="metric"><div class="label">Median inner support</div><div class="value">{fmt(inner["inner_support_median"], 1)} cams</div></div>
    <div class="metric"><div class="label">Sim3 scale</div><div class="value">{fmt(align["scale"], 5)}</div></div>
    <div class="metric"><div class="label">Outer triangle area</div><div class="value">{fmt(align["inner_triangle_area_m2"], 4)} m²</div></div>
    <div class="metric"><div class="label">Metric bridge gate</div><div class="value">{html.escape(str(metric_gate.get("status", "n/a")))}</div></div>
    <div class="metric"><div class="label">COLMAP prior diagnostic</div><div class="value">{html.escape(str(colmap_gate.get("status", "n/a")))}</div></div>
  </div>
  <h2>Outer Top-Down Bridge Cameras</h2>
  <table>
    <thead>
      <tr>
        <th>Camera</th><th>Votes</th><th>Center med m</th><th>Center p90 m</th>
        <th>Rot med deg</th><th>Rot p90 deg</th><th>Center in inner rig m</th><th>COLMAP tracks</th>
        <th>COLMAP rot residual deg</th><th>COLMAP center residual</th>
      </tr>
    </thead>
    <tbody>{''.join(outer_rows)}</tbody>
  </table>
	  <h2>COLMAP Pairwise Scale Check</h2>
	  <table>
	    <thead><tr><th>Pair</th><th>Bridge distance m</th><th>COLMAP distance</th><th>Ratio</th></tr></thead>
	    <tbody>{''.join(pairwise_rows)}</tbody>
	  </table>
  {outer_final_block}
	  <h2>Inputs</h2>
	  <p class="note">
	    PnP views: <code>{html.escape(summary["inputs"]["pnp_views"])}</code><br>
	    Inner refined camera_tr_rig: <code>{html.escape(summary["inputs"]["inner_camera_tr_rig"])}</code><br>
	    Outer final camera_tr_rig: <code>{html.escape(summary["inputs"].get("outer_camera_tr_rig", ""))}</code><br>
	    COLMAP images: <code>{html.escape(summary["inputs"]["colmap_images"])}</code><br>
	    Bridge pose output: <code>{html.escape(summary["outputs"].get("bridge_camera_tr_rig", "n/a"))}</code>
	  </p>
  <h2>Interpretation</h2>
  <p class="note">
    Center residuals are measured from per-frame large-board PnP votes after anchoring the board
    with the refined inner rig. COLMAP center residuals are after a three-camera Sim(3) alignment,
    so they are mainly a sanity check of non-degeneracy; rotation residuals are the stronger check
    for whether the COLMAP outer rough orientation is consistent with the metric bridge. Low COLMAP
    track counts or inconsistent pairwise ratios mean the first-frame COLMAP outer model should not
	    be treated as a reliable metric bridge prior for these selected cameras. When an outer final rig
	    is supplied, the output pose YAML contains the full outer rig transformed into the refined inner
	    rig frame using the top-down large-marker anchors.
	  </p>
</body>
</html>
"""


def evaluate(args):
    inner_camera_tr_rig = load_pose_yaml(args.inner_camera_tr_rig)
    outer_camera_tr_rig = load_pose_yaml(args.outer_camera_tr_rig) if args.outer_camera_tr_rig else None
    inner_pose_by_bridge_index, inner_pose_source_mode = map_inner_poses_to_bridge_indices(
        inner_camera_tr_rig,
        args.inner_indices,
    )
    pnp_views = load_pnp_views(args.pnp_views, args.max_median_error_px)
    colmap_images = load_colmap_images(args.colmap_images)

    inner_by_frame = {}
    outer_by_label = {label: [] for label in args.outer_labels}
    outer_index_to_label = dict(zip(args.outer_indices, args.outer_labels))

    for view in pnp_views:
        camera_index = view["camera_index"]
        frame = view["imageset_index"]
        if camera_index in args.inner_indices:
            camera_tr_rig = inner_pose_by_bridge_index[camera_index]
            if camera_tr_rig is None:
                continue
            rig_tr_board = invert_pose(camera_tr_rig) @ view["camera_tr_board"]
            inner_by_frame.setdefault(frame, []).append(rig_tr_board)

    rig_tr_board_by_frame = {}
    support_counts = []
    for frame, poses in inner_by_frame.items():
        if len(poses) < args.min_inner_support:
            continue
        rig_tr_board_by_frame[frame] = mean_pose(poses)
        support_counts.append(len(poses))

    for view in pnp_views:
        camera_index = view["camera_index"]
        if camera_index not in outer_index_to_label:
            continue
        frame = view["imageset_index"]
        if frame not in rig_tr_board_by_frame:
            continue
        camera_tr_board = view["camera_tr_board"]
        rig_tr_board = rig_tr_board_by_frame[frame]
        camera_tr_rig = camera_tr_board @ invert_pose(rig_tr_board)
        outer_by_label[outer_index_to_label[camera_index]].append(camera_tr_rig)

    outer_summaries = []
    for label in args.outer_labels:
        votes = outer_by_label[label]
        if len(votes) < args.min_outer_votes:
            raise ValueError(f"Camera {label} only has {len(votes)} bridge votes")
        outer_summaries.append(summarize_outer_votes(label, votes))

    outer_final_alignment = None
    if outer_camera_tr_rig is not None:
        outer_final_alignment = build_outer_final_alignment(
            outer_summaries,
            outer_camera_tr_rig,
            args.outer_indices,
        )

    missing_colmap = [label for label in args.outer_labels if label not in colmap_images]
    if missing_colmap:
        raise ValueError(f"COLMAP model missing outer labels: {missing_colmap}")

    inner_centers = np.asarray([row["center_inner_rig_m"] for row in outer_summaries], dtype=np.float64)
    colmap_centers = np.asarray([colmap_images[row["label"]]["center_world"] for row in outer_summaries], dtype=np.float64)
    pairwise_distances = pairwise_distance_summary(args.outer_labels, inner_centers, colmap_centers)
    scale, sim_rotation, sim_translation, singular_values = umeyama_similarity(inner_centers, colmap_centers)
    predicted_centers = scale * (sim_rotation @ inner_centers.T).T + sim_translation[None, :]
    center_residuals = np.linalg.norm(predicted_centers - colmap_centers, axis=1)

    per_camera_alignment = {}
    for row, center_residual in zip(outer_summaries, center_residuals):
        label = row["label"]
        camera_tr_inner = np.asarray(row["camera_tr_inner_rig"], dtype=np.float64)
        colmap_camera_tr_world = colmap_images[label]["camera_tr_world"]
        predicted_camera_tr_colmap = camera_tr_inner[:3, :3] @ sim_rotation.T
        rotation_residual = rotation_angle_deg(predicted_camera_tr_colmap @ colmap_camera_tr_world[:3, :3].T)
        per_camera_alignment[label] = {
            "colmap_image_name": colmap_images[label]["name"],
            "point2d_count": int(colmap_images[label]["point2d_count"]),
            "triangulated_point_count": int(colmap_images[label]["triangulated_point_count"]),
            "center_residual_colmap_units": float(center_residual),
            "rotation_residual_deg": float(rotation_residual),
            "colmap_center": colmap_images[label]["center_world"].tolist(),
            "predicted_colmap_center": predicted_centers[args.outer_labels.index(label)].tolist(),
        }

    rotation_residual_values = [v["rotation_residual_deg"] for v in per_camera_alignment.values()]
    triangulated_counts = [v["triangulated_point_count"] for v in per_camera_alignment.values()]
    pairwise_ratios = [row["distance_ratio_colmap_per_meter"] for row in pairwise_distances]
    support_array = np.asarray(support_counts, dtype=np.float64)
    conclusion = (
        "Large-marker bridge is geometrically connected for the three top-down outer cameras "
        f"({', '.join(args.outer_labels)}). Per-frame metric PnP consistency is the primary quality signal; "
        "the three-camera COLMAP Sim(3) alignment is under-constrained for translation/scale but still useful "
        "for checking orientation consistency."
    )
    if min(triangulated_counts) < 30 or (max(pairwise_ratios) / min(pairwise_ratios)) > 2.0:
        conclusion += (
            " The selected COLMAP outer poses are weak for bridge validation: at least one camera has very few "
            "triangulated tracks and the pairwise COLMAP distances are not close to a single global scale."
        )

    summary = {
        "inputs": {
            "pnp_views": str(args.pnp_views),
            "inner_camera_tr_rig": str(args.inner_camera_tr_rig),
            "inner_pose_source_mode": inner_pose_source_mode,
            "inner_indices": args.inner_indices,
            "outer_indices": args.outer_indices,
            "outer_labels": args.outer_labels,
            "colmap_images": str(args.colmap_images),
            "outer_camera_tr_rig": str(args.outer_camera_tr_rig) if args.outer_camera_tr_rig else "",
        },
        "outputs": {},
        "inner_board_pose_summary": {
            "frame_count": int(len(rig_tr_board_by_frame)),
            "inner_support_min": int(np.min(support_array)) if len(support_array) else 0,
            "inner_support_median": float(np.median(support_array)) if len(support_array) else 0.0,
            "inner_support_max": int(np.max(support_array)) if len(support_array) else 0,
            "min_inner_support": int(args.min_inner_support),
            "max_median_error_px": float(args.max_median_error_px),
        },
        "outer_camera_summaries": outer_summaries,
        "colmap_alignment": {
            "scale": float(scale),
            "rotation": sim_rotation.tolist(),
            "translation": sim_translation.tolist(),
            "singular_values": singular_values.tolist(),
            "inner_triangle_area_m2": triangle_area(inner_centers),
            "colmap_triangle_area_units2": triangle_area(colmap_centers),
            "center_residual_median_colmap_units": float(np.median(center_residuals)),
            "rotation_residual_median_deg": float(np.median(rotation_residual_values)),
            "rotation_residual_max_deg": float(np.max(rotation_residual_values)),
            "pairwise_distances": pairwise_distances,
            "per_camera": per_camera_alignment,
        },
        "conclusion": conclusion,
    }
    if outer_final_alignment is not None:
        summary["outer_final_alignment"] = outer_final_alignment
        summary["conclusion"] += (
            " The supplied outer final rig has been rigidly aligned into the inner rig frame "
            f"using {outer_final_alignment['anchor_count']} large-marker top-down anchors."
        )
    summary["quality_gates"] = build_quality_gates(summary, args)
    metric_gate_status = summary["quality_gates"]["metric_bridge"]["status"]
    colmap_gate_status = summary["quality_gates"]["colmap_prior_diagnostic"]["status"]
    summary["conclusion"] += (
        f" Metric bridge gate: {metric_gate_status}. "
        f"COLMAP prior diagnostic: {colmap_gate_status}."
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnp_views", required=True, type=Path)
    parser.add_argument("--inner_camera_tr_rig", required=True, type=Path)
    parser.add_argument("--colmap_images", required=True, type=Path)
    parser.add_argument("--outer_camera_tr_rig", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--inner_indices", default="24,25,26,27,28,29,30,31")
    parser.add_argument("--outer_indices", default="9,10,11")
    parser.add_argument("--outer_labels", default="4-1,4-2,4-3")
    parser.add_argument("--min_inner_support", type=int, default=2)
    parser.add_argument("--min_outer_votes", type=int, default=10)
    parser.add_argument("--max_median_error_px", type=float, default=10.0)
    parser.add_argument("--bridge_gate_min_inner_frames", type=int, default=50)
    parser.add_argument("--bridge_gate_min_inner_support_median", type=float, default=3.0)
    parser.add_argument("--bridge_gate_max_center_p90_m", type=float, default=0.25)
    parser.add_argument("--bridge_gate_max_rotation_p90_deg", type=float, default=5.0)
    parser.add_argument("--bridge_gate_min_triangle_area_m2", type=float, default=0.02)
    parser.add_argument("--colmap_gate_min_triangulated_tracks", type=int, default=30)
    parser.add_argument("--colmap_gate_max_pairwise_ratio_spread", type=float, default=2.0)
    parser.add_argument("--colmap_gate_max_rotation_residual_median_deg", type=float, default=15.0)
    args = parser.parse_args()
    args.inner_indices = [int(x) for x in args.inner_indices.split(",") if x]
    args.outer_indices = [int(x) for x in args.outer_indices.split(",") if x]
    args.outer_labels = [x for x in args.outer_labels.split(",") if x]
    if len(args.outer_indices) != len(args.outer_labels):
        raise ValueError("--outer_indices and --outer_labels must have the same length")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate(args)
    bridge_pose_path = args.output_dir / "camera_tr_inner_refined_plus_outer_topdown.yaml"
    inner_camera_tr_rig = load_pose_yaml(args.inner_camera_tr_rig)
    outer_camera_tr_rig = load_pose_yaml(args.outer_camera_tr_rig) if args.outer_camera_tr_rig else None
    inner_pose_by_bridge_index, inner_pose_source_mode = map_inner_poses_to_bridge_indices(
        inner_camera_tr_rig,
        args.inner_indices,
    )
    outer_pose_count = len(outer_camera_tr_rig) if outer_camera_tr_rig is not None else 0
    target_count = max(max(args.outer_indices), max(args.inner_indices), outer_pose_count - 1) + 1
    if inner_pose_source_mode == "direct_bridge_indices":
        bridge_poses = list(inner_camera_tr_rig)
        while len(bridge_poses) < target_count:
            bridge_poses.append(None)
    else:
        bridge_poses = [None for _ in range(target_count)]
    for index in args.inner_indices:
        bridge_poses[index] = inner_pose_by_bridge_index[index]
    if outer_camera_tr_rig is not None and "outer_final_alignment" in summary:
        outer_rig_tr_inner_rig = np.asarray(
            summary["outer_final_alignment"]["outer_rig_tr_inner_rig"],
            dtype=np.float64,
        )
        for index, camera_tr_outer_rig in enumerate(outer_camera_tr_rig):
            if camera_tr_outer_rig is not None:
                bridge_poses[index] = np.asarray(camera_tr_outer_rig, dtype=np.float64) @ outer_rig_tr_inner_rig
    else:
        for index, label in zip(args.outer_indices, args.outer_labels):
            for row in summary["outer_camera_summaries"]:
                if row["label"] == label:
                    bridge_poses[index] = np.asarray(row["camera_tr_inner_rig"], dtype=np.float64)
                    break
    write_pose_yaml(bridge_pose_path, bridge_poses)
    summary["outputs"]["bridge_camera_tr_rig"] = str(bridge_pose_path)
    if outer_camera_tr_rig is not None:
        summary["outputs"]["bridge_camera_tr_rig_semantics"] = (
            "indices 0..outer_count-1 are outer final poses transformed into the inner rig frame; "
            "inner bridge indices keep the refined inner rig poses"
        )
    summary_path = args.output_dir / "bridge_summary.json"
    html_path = args.output_dir / "index.html"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(make_html(summary), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {html_path}")
    print(json.dumps({
        "inner_board_frames": summary["inner_board_pose_summary"]["frame_count"],
        "outer_votes": {row["label"]: row["vote_count"] for row in summary["outer_camera_summaries"]},
        "median_colmap_rotation_residual_deg": summary["colmap_alignment"]["rotation_residual_median_deg"],
        "max_colmap_rotation_residual_deg": summary["colmap_alignment"]["rotation_residual_max_deg"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
