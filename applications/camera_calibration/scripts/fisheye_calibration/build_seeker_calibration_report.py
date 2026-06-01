#!/usr/bin/env python3
"""Build a static visual report for the Seeker four-fisheye calibration run."""

import argparse
import html
import json
import math
import os
import re
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


CAMERAS = ["cam0", "cam1", "cam2", "cam3"]
CAM_LABELS = {
    "cam0": "cam0 / LU",
    "cam1": "cam1 / LD",
    "cam2": "cam2 / RD",
    "cam3": "cam3 / RU",
}
COLORS = {
    "cam0": "#2f6df6",
    "cam1": "#16a34a",
    "cam2": "#d97706",
    "cam3": "#dc2626",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def q(values, percentile):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * percentile / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def parse_feature_log(path):
    counts = []
    by_src = {}
    if not Path(path).exists():
        return counts, by_src
    pattern = re.compile(r"/(?P<name>\d+)_src(?P<src>\d+)_[^/]+\.png:\s+(?P<count>\d+)\s+features")
    fallback = re.compile(r":\s+(?P<count>\d+)\s+features\b")
    with Path(path).open(errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                count = int(match.group("count"))
                src = int(match.group("src"))
                counts.append(count)
                by_src[src] = count
                continue
            match = fallback.search(line)
            if match:
                counts.append(int(match.group("count")))
    return counts, by_src


def read_u32(data, offset):
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def read_i32(data, offset):
    return struct.unpack_from(">i", data, offset)[0], offset + 4


def read_f32(data, offset):
    # Dataset serialization writes floats in host order, while integer fields use network order.
    return struct.unpack_from("<f", data, offset)[0], offset + 4


def read_dataset_feature_points(path):
    path = Path(path)
    if not path.exists():
        return {"points": [], "imagesets": 0, "features": 0, "image_size": None}
    data = path.read_bytes()
    offset = 0
    if data[:10] != b"calib_data":
        raise ValueError(f"{path} is not a calib_data dataset")
    offset += 10
    version, offset = read_u32(data, offset)
    if version not in (0, 1):
        raise ValueError(f"{path} has unsupported dataset version {version}")
    num_cameras, offset = read_u32(data, offset)
    image_sizes = []
    for _ in range(num_cameras):
        width, offset = read_u32(data, offset)
        height, offset = read_u32(data, offset)
        image_sizes.append((width, height))

    num_imagesets, offset = read_u32(data, offset)
    points = []
    observations = []
    for image_index in range(num_imagesets):
        filename_len, offset = read_u32(data, offset)
        offset += filename_len
        for camera_index in range(num_cameras):
            num_features, offset = read_u32(data, offset)
            for _ in range(num_features):
                x, offset = read_f32(data, offset)
                y, offset = read_f32(data, offset)
                feature_id, offset = read_i32(data, offset)
                if camera_index == 0 and math.isfinite(x) and math.isfinite(y):
                    points.append((x, y))
                    observations.append({
                        "image_index": image_index,
                        "x": x,
                        "y": y,
                        "feature_id": feature_id,
                    })
    return {
        "points": points,
        "observations": observations,
        "imagesets": num_imagesets,
        "features": len(points),
        "image_size": image_sizes[0] if image_sizes else None,
    }


def load_dataset_feature_points(root):
    dataset_dir = Path(root) / "datasets_all_patterns"
    out = {}
    for cam in CAMERAS:
        out[cam] = read_dataset_feature_points(dataset_dir / f"{cam}_features.bin")
    return out


def pose_to_matrix(pose):
    tx, ty, tz = float(pose["tx"]), float(pose["ty"]), float(pose["tz"])
    qx, qy, qz, qw = (float(pose["qx"]), float(pose["qy"]), float(pose["qz"]), float(pose["qw"]))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rot = np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rot
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def project_central_thin_prism_fisheye(point, params, use_equidistant_projection=True):
    x, y, z = [float(v) for v in point]
    if z <= 0:
        return None

    undistorted_x = x / z
    undistorted_y = y / z
    radius = math.hypot(undistorted_x, undistorted_y)
    if use_equidistant_projection and radius > 1e-6:
        theta_by_r = math.atan(radius) / radius
        fisheye_x = theta_by_r * undistorted_x
        fisheye_y = theta_by_r * undistorted_y
    else:
        fisheye_x = undistorted_x
        fisheye_y = undistorted_y

    x2 = fisheye_x * fisheye_x
    y2 = fisheye_y * fisheye_y
    xy = fisheye_x * fisheye_y
    r2 = x2 + y2
    r4 = r2 * r2
    r6 = r4 * r2
    r8 = r6 * r2

    fx, fy, cx, cy, k1, k2, k3, k4, p1, p2, sx1, sy1 = [float(v) for v in params]
    radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8
    dx = 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2) + sx1 * r2
    dy = 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2) + sy1 * r2

    distorted_x = fisheye_x + radial * fisheye_x + dx
    distorted_y = fisheye_y + radial * fisheye_y + dy
    return np.array([fx * distorted_x + cx, fy * distorted_y + cy], dtype=np.float64)


def read_report_camera_info(path):
    info = {}
    path = Path(path)
    if not path.exists():
        return info
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if not value:
            continue
        try:
            if re.fullmatch(r"[-+]?\d+", value):
                info[key] = int(value)
            else:
                info[key] = float(value)
        except ValueError:
            info[key] = value
    return info


def error_stats(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return {"count": 0, "median": None, "average": None, "maximum": None, "p90": None, "p95": None}
    return {
        "count": int(errors.size),
        "median": float(np.median(errors)),
        "average": float(np.mean(errors)),
        "maximum": float(np.max(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
    }


def compute_camera_reprojection_errors(large_root, cam):
    calib_dir = Path(large_root) / "calibration_kb8_firstpass" / f"{cam}_thin_prism_nofinal"
    dataset = read_dataset_feature_points(calib_dir / "dataset.bin")
    points_doc = yaml.safe_load((calib_dir / "points.yaml").read_text())
    points_3d = np.asarray(points_doc["points"], dtype=np.float64).reshape(-1, 3)
    feature_to_point = {
        int(row["feature_id"]): int(row["point_index"])
        for row in points_doc["feature_id_to_point_index"]
    }

    rig_poses = [pose_to_matrix(pose) for pose in yaml.safe_load((calib_dir / "rig_tr_global.yaml").read_text())["poses"]]
    camera_pose_doc = yaml.safe_load((calib_dir / "camera_tr_rig.yaml").read_text())
    camera_tr_rig = pose_to_matrix(camera_pose_doc["poses"][0])
    intrinsics_doc = yaml.safe_load((calib_dir / "intrinsics0.yaml").read_text())
    params = [float(v) for v in intrinsics_doc["parameters"]]
    use_equidistant = bool(intrinsics_doc.get("use_equidistant_projection", True))

    observed_points = []
    errors = []
    skipped = 0
    for obs in dataset["observations"]:
        point_index = feature_to_point.get(int(obs["feature_id"]))
        image_index = int(obs["image_index"])
        if point_index is None or image_index >= len(rig_poses):
            skipped += 1
            continue
        point_global_h = np.append(points_3d[point_index], 1.0)
        point_camera = (camera_tr_rig @ rig_poses[image_index] @ point_global_h)[:3]
        projection = project_central_thin_prism_fisheye(point_camera, params, use_equidistant)
        if projection is None:
            skipped += 1
            continue
        observed = np.array([float(obs["x"]), float(obs["y"])], dtype=np.float64)
        observed_points.append(observed)
        errors.append(float(np.linalg.norm(observed - projection)))

    observed_points = np.asarray(observed_points, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "points": observed_points,
        "errors": errors,
        "stats": error_stats(errors),
        "report_info": read_report_camera_info(calib_dir / "report_camera0_info.txt"),
        "imagesets": dataset["imagesets"],
        "image_size": dataset["image_size"],
        "skipped": int(skipped),
        "source_dir": str(calib_dir),
    }


def load_reprojection_data(large_root):
    return {cam: compute_camera_reprojection_errors(large_root, cam) for cam in CAMERAS}


def feature_log_for(root, cam, preferred="all"):
    logs = Path(root) / "logs"
    if preferred == "pattern3":
        path = logs / f"{cam}_features_pattern3_win10.log"
        if path.exists():
            return path
    return logs / f"{cam}_features_all_patterns.log"


def load_capture(root, feature_preference="all"):
    root = Path(root)
    summary = read_json(root / "metadata" / "summary.json")
    frames = {}
    selected = {}
    selected_tag_side = {}
    selected_sharpness = {}
    selected_area = {}
    selected_centers = {}
    feature_counts = {}
    feature_by_src = {}

    for cam in CAMERAS:
        rows = read_jsonl(root / "metadata" / f"{cam}_frames.jsonl")
        frames[cam] = rows
        chosen = [r for r in rows if r.get("selected")]
        selected[cam] = chosen
        selected_tag_side[cam] = [math.sqrt(max(0.0, r.get("board_area") or 0.0)) for r in chosen]
        selected_sharpness[cam] = [r.get("sharpness") or 0.0 for r in chosen]
        selected_area[cam] = [r.get("board_area") or 0.0 for r in chosen]
        selected_centers[cam] = [
            (r.get("board_cx"), r.get("board_cy"), math.sqrt(max(0.0, r.get("board_area") or 0.0)))
            for r in chosen
            if r.get("board_cx") is not None and r.get("board_cy") is not None
        ]
        counts, by_src = parse_feature_log(feature_log_for(root, cam, feature_preference))
        feature_counts[cam] = counts
        feature_by_src[cam] = by_src

    return {
        "root": str(root),
        "summary": summary,
        "frames": frames,
        "selected": selected,
        "selected_tag_side": selected_tag_side,
        "selected_sharpness": selected_sharpness,
        "selected_area": selected_area,
        "selected_centers": selected_centers,
        "feature_counts": feature_counts,
        "feature_by_src": feature_by_src,
    }


def summarize_capture(capture):
    out = {}
    summary_cams = capture["summary"].get("cameras", [])
    for idx, cam in enumerate(CAMERAS):
        decision_counts = {}
        if idx < len(summary_cams):
            decision_counts = summary_cams[idx].get("decision_counts", {})
        features = capture["feature_counts"][cam]
        tag_sides = capture["selected_tag_side"][cam]
        sharpness = capture["selected_sharpness"][cam]
        out[cam] = {
            "processed": summary_cams[idx].get("processed") if idx < len(summary_cams) else len(capture["frames"][cam]),
            "selected": len(capture["selected"][cam]),
            "no_board": int(decision_counts.get("no_board", 0)),
            "near_duplicate": int(decision_counts.get("near_duplicate", 0)),
            "feature_frames": len(features),
            "nonzero_feature_frames": sum(1 for v in features if v > 0),
            "total_features": int(sum(features)),
            "median_tag_side_px": q(tag_sides, 50),
            "p10_tag_side_px": q(tag_sides, 10),
            "p90_tag_side_px": q(tag_sides, 90),
            "median_sharpness": q(sharpness, 50),
        }
    return out


def compute_pair_overlaps(feature_by_src):
    pairs = {}
    for i, a in enumerate(CAMERAS):
        for b in CAMERAS[i + 1:]:
            common = sorted(set(feature_by_src[a]) & set(feature_by_src[b]))
            both_nonzero = [src for src in common if feature_by_src[a][src] > 0 and feature_by_src[b][src] > 0]
            feature_total = sum(feature_by_src[a][src] + feature_by_src[b][src] for src in both_nonzero)
            pairs[f"{a}-{b}"] = {
                "common_selected_frames": len(common),
                "both_nonzero_frames": len(both_nonzero),
                "pair_feature_total": int(feature_total),
            }
    return pairs


def plot_bar(path, title, labels, series, ylabel):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    x = np.arange(len(labels))
    width = 0.82 / max(1, len(series))
    for idx, (name, values, color) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width
        ax.bar(x + offset, values, width, label=name, color=color)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False, ncols=min(3, len(series)))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_box(path, title, values_by_group, ylabel):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    labels = list(values_by_group)
    values = [values_by_group[k] if values_by_group[k] else [0] for k in labels]
    ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, label in zip(ax.artists, labels):
        patch.set_facecolor(COLORS.get(label.split()[0], "#64748b"))
        patch.set_alpha(0.25)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(path, capture, reprojection_data, width=1088, height=1280):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 9.0), dpi=160)
    error_vmin = 0.05
    error_vmax = 8.0
    for ax, cam in zip(axes.ravel(), CAMERAS):
        data = reprojection_data.get(cam, {})
        all_points = data.get("points", np.empty((0, 2)))
        all_errors = data.get("errors", np.empty((0,), dtype=np.float64))
        if len(all_points):
            color_errors = np.clip(all_errors, error_vmin, error_vmax)
            sc = ax.scatter(
                all_points[:, 0],
                all_points[:, 1],
                c=color_errors,
                s=2.4,
                cmap="magma",
                norm=LogNorm(vmin=error_vmin, vmax=error_vmax, clip=True),
                alpha=0.76,
                linewidth=0,
                rasterized=True,
                label="board corners",
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="reproj error [px, log]")
            cbar.set_ticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
            cbar.set_ticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4", "8+"])

        centers = capture["selected_centers"][cam]
        if centers:
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            sizes = [max(10, min(120, c[2] * 0.5)) for c in centers]
            ax.scatter(xs, ys, s=sizes, facecolors="none", edgecolors="#111827", alpha=0.42, linewidth=0.7, label="screening centroids")
        ax.set_title(f"{CAM_LABELS[cam]} reprojection error coverage", loc="left", fontweight="bold")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x px")
        ax.set_ylabel("y px")
        ax.grid(color="#e5e7eb", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def kb8_theta_distorted(theta, k):
    t2 = theta * theta
    return theta * (1.0 + k[0] * t2 + k[1] * t2 * t2 + k[2] * t2 * t2 * t2 + k[3] * t2 * t2 * t2 * t2)


def plot_kb8_curves(path, kb8):
    theta = np.linspace(0, 2.35, 300)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    for cam in CAMERAS:
        params = kb8["cameras"][cam]["params"]
        radial = kb8_theta_distorted(theta, params[4:8])
        ax.plot(theta, radial, label=CAM_LABELS[cam], color=COLORS[cam], linewidth=2)
    ax.plot(theta, theta, "--", color="#64748b", linewidth=1, label="ideal theta")
    ax.set_title("KB8 radial projection curves", loc="left", fontweight="bold")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("theta_d [normalized radius]")
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_omni_fit_error(path, kb8, omni_yaml):
    theta = np.linspace(0.001, 2.35, 300)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    for cam in CAMERAS:
        params = kb8["cameras"][cam]["params"]
        fx = params[0]
        target = kb8_theta_distorted(theta, params[4:8])
        entry = omni_yaml[cam]
        xi, omni_fx = entry["intrinsics"][0], entry["intrinsics"][1]
        d = entry["distortion_coeffs"]
        s = np.sin(theta) / (np.cos(theta) + xi)
        pred = (omni_fx / fx) * s * (1.0 + d[0] * s**2 + d[1] * s**4 + d[4] * s**6)
        error_px = (pred - target) * fx
        ax.plot(theta, error_px, label=CAM_LABELS[cam], color=COLORS[cam], linewidth=2)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Post-hoc omni+radtan fit error against KB8", loc="left", fontweight="bold")
    ax.set_xlabel("theta [rad]")
    ax.set_ylabel("radial error [px]")
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_intrinsics(path, kb8):
    labels = CAMERAS
    fx = [kb8["cameras"][cam]["params"][0] for cam in labels]
    fy = [kb8["cameras"][cam]["params"][1] for cam in labels]
    cx_offset = [kb8["cameras"][cam]["params"][2] - kb8["cameras"][cam]["width"] / 2.0 for cam in labels]
    cy_offset = [kb8["cameras"][cam]["params"][3] - kb8["cameras"][cam]["height"] / 2.0 for cam in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=160)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, fx, width=0.36, label="fx", color="#2f6df6")
    axes[0].bar(x + 0.18, fy, width=0.36, label="fy", color="#16a34a")
    axes[0].set_title("Focal length", loc="left", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("px")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, cx_offset, width=0.36, label="cx - w/2", color="#d97706")
    axes[1].bar(x + 0.18, cy_offset, width=0.36, label="cy - h/2", color="#dc2626")
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_title("Principal point offset", loc="left", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("px")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_extrinsics(path, extrinsics, pair_overlaps):
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=160)
    layout = {
        "cam0": (0.0, 1.0),
        "cam3": (1.8, 1.0),
        "cam1": (0.0, 0.0),
        "cam2": (1.8, 0.0),
    }
    edges = [
        ("cam0", "cam3", "observed top", True),
        ("cam1", "cam2", "observed bottom", True),
        ("cam0", "cam1", "assumed 180 deg", False),
        ("cam3", "cam2", "assumed 180 deg", False),
    ]
    for a, b, label, observed in edges:
        xa, ya = layout[a]
        xb, yb = layout[b]
        style = "-" if observed else "--"
        color = "#111827" if observed else "#64748b"
        ax.plot([xa, xb], [ya, yb], style, color=color, linewidth=2.2 if observed else 1.6)
        mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
        if observed:
            key = "cam0-cam3" if {a, b} == {"cam0", "cam3"} else "cam1-cam2"
            count = pair_overlaps.get(key, {}).get("both_nonzero_frames", 0)
            ax.text(mx, my + 0.08, f"{label}: {count} frames", ha="center", va="bottom", fontsize=9, color=color)
        else:
            ax.text(mx - 0.03, my, label, ha="right", va="center", fontsize=9, color=color, rotation=90)
    poses = extrinsics["poses_from_top_left_reference"]
    for cam, (x, y) in layout.items():
        ax.scatter([x], [y], s=520, color=COLORS[cam], edgecolor="white", linewidth=1.5, zorder=3)
        pose = poses[cam]
        t = pose["matrix"][0][3], pose["matrix"][1][3], pose["matrix"][2][3]
        ax.text(x, y, cam, color="white", ha="center", va="center", fontweight="bold", zorder=4)
        ax.text(x, y - 0.18, f"{CAM_LABELS[cam]}\nt=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]m", ha="center", va="top", fontsize=8)
    avg = extrinsics["averaged"]["T_right_left"]
    ax.text(
        0.9,
        1.34,
        f"averaged horizontal baseline: {avg['tx']:+.4f}, {avg['ty']:+.4f}, {avg['tz']:+.4f} m",
        ha="center",
        fontsize=9,
        color="#111827",
    )
    ax.text(0.9, -0.36, "vertical/back-to-back translation is assumed zero until CAD/measured baseline is supplied", ha="center", fontsize=9, color="#b45309")
    ax.set_title("Observed and assumed four-fisheye rig graph", loc="left", fontweight="bold")
    ax.set_xlim(-0.45, 2.25)
    ax.set_ylim(-0.55, 1.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(path, image_dir, title, limit=12):
    files = sorted(Path(image_dir).glob("*.png"))
    if not files:
        return False
    if len(files) > limit:
        picks = [files[int(round(i * (len(files) - 1) / (limit - 1)))] for i in range(limit)]
    else:
        picks = files
    thumb_w, thumb_h = 210, 248
    cols = 4
    rows = int(math.ceil(len(picks) / cols))
    title_h = 34
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h + title_h), "#f8fafc")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 9), title, fill="#111827")
    for idx, file in enumerate(picks):
        img = Image.open(file).convert("RGB")
        img.thumbnail((thumb_w - 12, thumb_h - 38), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = title_h + (idx // cols) * thumb_h
        draw.rectangle([x + 4, y + 4, x + thumb_w - 4, y + thumb_h - 4], outline="#d1d5db", width=1)
        sheet.paste(img, (x + (thumb_w - img.width) // 2, y + 8))
        label = file.name.split("_")[0] + "_" + file.name.split("_")[1]
        draw.text((x + 8, y + thumb_h - 24), label, fill="#334155")
    sheet.save(path, quality=88)
    return True


def table(headers, rows):
    out = ["<table>", "<thead><tr>"]
    for h in headers:
        out.append(f"<th>{html.escape(str(h))}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{html.escape(str(cell))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def img(src, alt):
    return f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def build_html(report_title, summary, assets, output_root):
    large = summary["large_capture"]
    small = summary.get("small_capture") or None
    capture_label = summary.get("capture_label", "Large capture")
    intrinsics = summary["intrinsics"]
    fit = summary["omni_fit_quality"]
    pair_overlaps = summary["pair_overlaps"]
    reprojection = summary["feature_reprojection_errors"]

    capture_rows = []
    for cam in CAMERAS:
        l = large[cam]
        if small:
            s = small.get(cam, {})
            capture_rows.append([
                CAM_LABELS[cam],
                l["selected"],
                l["nonzero_feature_frames"],
                l["total_features"],
                fmt(l["median_tag_side_px"], 1),
                s.get("selected", "n/a"),
                fmt(s.get("median_tag_side_px"), 1),
            ])
        else:
            capture_rows.append([
                CAM_LABELS[cam],
                l["selected"],
                l["nonzero_feature_frames"],
                l["total_features"],
                fmt(l["median_tag_side_px"], 1),
                fmt(l["median_sharpness"], 1),
            ])

    intr_rows = []
    for cam in CAMERAS:
        p = intrinsics[cam]["params"]
        intr_rows.append([CAM_LABELS[cam], fmt(p[0], 3), fmt(p[1], 3), fmt(p[2], 3), fmt(p[3], 3), fmt(p[4], 6), fmt(p[5], 6), fmt(p[6], 6), fmt(p[7], 6)])

    fit_rows = []
    for cam in CAMERAS:
        f = fit.get(cam, {})
        fit_rows.append([CAM_LABELS[cam], fmt(f.get("rms_px_fx"), 2), fmt(f.get("max_px_fx"), 2), fmt(f.get("theta_max_rad"), 2)])

    reproj_rows = []
    for cam in CAMERAS:
        item = reprojection[cam]
        computed = item["computed"]
        source = item.get("source_report", {})
        reproj_rows.append([
            CAM_LABELS[cam],
            computed["count"],
            fmt(computed["median"], 3),
            fmt(computed["average"], 3),
            fmt(computed["maximum"], 3),
            fmt(source.get("reprojection_error_median"), 3),
            fmt(source.get("reprojection_error_average"), 3),
            fmt(source.get("reprojection_error_maximum"), 3),
        ])

    pair_rows = []
    for key, value in pair_overlaps.items():
        pair_rows.append([key, value["common_selected_frames"], value["both_nonzero_frames"], value["pair_feature_total"]])

    cards = [
        (capture_label, "usable first-pass", "Balanced selected frames and feature extraction across all four cameras."),
        ("Intrinsics", "KB8 source", "BA source model is central_thin_prism_fisheye; KB8 keeps the first eight equidistant terms."),
        ("Reprojection", "BA verified", "Per-feature reprojection errors are recomputed from the saved BA state."),
        ("Extrinsics", "assumed rig", "Two horizontal edges observed; back-to-back vertical relation is assumed 180 degrees with zero translation."),
    ]

    card_html = "\n".join(
        f'<div class="card"><div class="card-k">{html.escape(k)}</div><div class="card-v">{html.escape(v)}</div><p>{html.escape(desc)}</p></div>'
        for k, v, desc in cards
    )
    contact_html = "\n".join(img(f"assets/{name}", alt) for name, alt in assets["contact_sheets"])
    if small:
        capture_table = table(["Camera", "large selected", "large nonzero feature frames", "large total features", "large median tag side px", "small selected", "small median tag side px"], capture_rows)
        comparison_plot = img("assets/large_vs_small_selected.png", "Large versus small selected frames")
        executive_text = "The large recording is suitable for a first-pass four-camera fisheye intrinsic calibration. The small recording is a separate negative-control capture and should be reviewed in its own QA report. The native calibration source is KB8-like equidistant parameters exported from <code>central_thin_prism_fisheye</code>; <code>omni+radtan</code> is a compatibility approximation."
        meta_text = "Generated from the provided MCAP calibration artifacts on 2026-05-27."
    else:
        capture_table = table(["Camera", "selected", "nonzero feature frames", "total features", "median tag side px", "median sharpness"], capture_rows)
        comparison_plot = ""
        executive_text = "This capture is suitable for a first-pass four-camera fisheye intrinsic calibration. The native calibration source is KB8-like equidistant parameters exported from <code>central_thin_prism_fisheye</code>; <code>omni+radtan</code> is a compatibility approximation."
        meta_text = f"Generated from {capture_label} MCAP calibration artifacts on 2026-05-27."

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report_title)}</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --ink: #111827;
      --muted: #64748b;
      --line: #d9dee7;
      --surface: #ffffff;
      --accent: #2f6df6;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{
      padding: 42px 48px 28px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 38px 0 14px; font-size: 22px; }}
    h3 {{ margin: 20px 0 10px; font-size: 16px; }}
    p {{ color: #334155; max-width: 980px; }}
    main {{ padding: 26px 48px 56px; max-width: 1320px; margin: 0 auto; }}
    .meta {{ color: var(--muted); margin-top: 8px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 24px; }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px; min-height: 150px; }}
    .card-k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .card-v {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    .card p {{ margin: 10px 0 0; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
    figcaption {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf0f5; text-align: right; font-size: 13px; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; color: #334155; font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    .note {{ border-left: 4px solid var(--warn); background: #fff7ed; padding: 12px 14px; border-radius: 6px; }}
    .paths {{ color: var(--muted); font-size: 13px; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 20px; padding-right: 20px; }}
      .cards, .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 27px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(report_title)}</h1>
    <div class="meta">{html.escape(meta_text)}</div>
    <div class="cards">{card_html}</div>
  </header>
  <main>
    <section>
      <h2>Executive Summary</h2>
      <p>{executive_text}</p>
      <p class="note">The final full BA stage on the full large dataset was skipped because the local run hit an OOM path. Treat these as first-pass calibration results for preview/plumbing, not final production calibration.</p>
    </section>

    <section>
      <h2>Capture QA</h2>
      {capture_table}
      <div class="grid">
        {img("assets/capture_decisions.png", f"{capture_label} screening decisions")}
        {comparison_plot}
        {img("assets/tag_side_boxplot.png", "Projected tag side distributions")}
        {img("assets/feature_counts.png", "Calibration feature counts")}
      </div>
    </section>

    <section>
      <h2>Board Coverage</h2>
      <p>The dense clouds are the actual board corner feature observations used by BA, loaded from each camera's first-pass <code>dataset.bin</code>. Point color encodes per-observation reprojection error computed from <code>points.yaml</code>, <code>rig_tr_global.yaml</code>, <code>camera_tr_rig.yaml</code>, and <code>intrinsics0.yaml</code>. The colorbar uses a log scale with <code>8 px</code> as the clipped maximum. The larger gray hollow markers are screening-stage board centroids only; frame order is no longer encoded.</p>
      {table(["Camera", "features", "computed median px", "computed mean px", "computed max px", "report median px", "report mean px", "report max px"], reproj_rows)}
      {img("assets/board_coverage.png", "Detected board corner reprojection error per camera")}
    </section>

    <section>
      <h2>Intrinsics</h2>
      {table(["Camera", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"], intr_rows)}
      <div class="grid">
        {img("assets/intrinsics_overview.png", "Focal length and principal point overview")}
        {img("assets/kb8_radial_curves.png", "KB8 radial projection curves")}
      </div>
    </section>

    <section>
      <h2>KB8 And Omni Export Comparison</h2>
      <p>The KB8 camchain directly copies the native first eight equidistant terms. The Seeker-compatible <code>omni+radtan</code> camchain is a post-hoc radial fit and should be treated as a preview compatibility file.</p>
      {table(["Camera", "omni fit RMS px", "omni fit max px", "theta max rad"], fit_rows)}
      {img("assets/omni_fit_error.png", "Radial error of fitted omni+radtan against KB8")}
    </section>

    <section>
      <h2>Extrinsics And Assumed Rig</h2>
      {table(["Pair", "common selected frames", "both nonzero frames", "pair feature total"], pair_rows)}
      {img("assets/extrinsics_graph.png", "Observed and assumed rig graph")}
      <p class="note">Only <code>cam0-cam3</code> and <code>cam1-cam2</code> are observed pairwise. The upper/lower back-to-back relation is a mechanical assumption: 180 degrees about camera X with zero translation until CAD/measured baselines are supplied.</p>
    </section>

    <section>
      <h2>Frame Contact Sheets</h2>
      <div class="grid">{contact_html}</div>
    </section>

    <section>
      <h2>Operator Feedback For Next Capture</h2>
      <p>Use the large board. Target median AprilTag side length at least <code>80 px</code>, preferably <code>100-150 px</code>. Target at least <code>60</code> nonzero calibration-feature frames per camera, preferably <code>80+</code>, and at least <code>20000</code> total calibration features per camera. For full four-camera extrinsics, add bridge observations that connect the two current components.</p>
      <p class="paths">Report directory: {html.escape(str(output_root))}</p>
    </section>
  </main>
</body>
</html>
"""
    return body


def build_capture_qa_html(report_title, capture_label, capture_summary, assets, output_root, root_path):
    capture_rows = []
    for cam in CAMERAS:
        item = capture_summary[cam]
        capture_rows.append([
            CAM_LABELS[cam],
            item["selected"],
            item["nonzero_feature_frames"],
            item["total_features"],
            fmt(item["median_tag_side_px"], 1),
            fmt(item["median_sharpness"], 1),
        ])

    cards = [
        (capture_label, "QA only", "This report describes one capture artifact and does not mix comparison captures."),
        ("Calibration", "not accepted", "The projected pattern is too small for robust calibration feature extraction."),
        ("Operator target", "larger board", "Use a larger projected tag size and more non-duplicate board poses."),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="card-k">{html.escape(k)}</div><div class="card-v">{html.escape(v)}</div><p>{html.escape(desc)}</p></div>'
        for k, v, desc in cards
    )
    contact_html = "\n".join(img(f"assets/{name}", alt) for name, alt in assets["contact_sheets"])

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report_title)}</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --ink: #111827;
      --muted: #64748b;
      --line: #d9dee7;
      --surface: #ffffff;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{ padding: 42px 48px 28px; background: #ffffff; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 38px 0 14px; font-size: 22px; }}
    p {{ color: #334155; max-width: 980px; }}
    main {{ padding: 26px 48px 56px; max-width: 1320px; margin: 0 auto; }}
    .meta {{ color: var(--muted); margin-top: 8px; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 24px; }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px; min-height: 140px; }}
    .card-k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .card-v {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    .card p {{ margin: 10px 0 0; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
    figcaption {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf0f5; text-align: right; font-size: 13px; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; color: #334155; font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    .note {{ border-left: 4px solid var(--warn); background: #fff7ed; padding: 12px 14px; border-radius: 6px; }}
    .paths {{ color: var(--muted); font-size: 13px; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 20px; padding-right: 20px; }}
      .cards, .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 27px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(report_title)}</h1>
    <div class="meta">Generated from {html.escape(capture_label)} MCAP artifacts on 2026-05-27.</div>
    <div class="cards">{card_html}</div>
  </header>
  <main>
    <section>
      <h2>Executive Summary</h2>
      <p>This is a standalone QA report for {html.escape(capture_label)}. It is not merged into the large-capture calibration report, because capture quality and calibration acceptance should be judged per capture.</p>
      <p class="note">This capture is useful as a diagnostic negative control, but should not be used as the first-pass four-fisheye intrinsic calibration source: the projected tag size is too small for robust calibration feature extraction.</p>
    </section>

    <section>
      <h2>Capture QA</h2>
      {table(["Camera", "selected", "nonzero feature frames", "total features", "median tag side px", "median sharpness"], capture_rows)}
      <div class="grid">
        {img("assets/capture_decisions.png", f"{capture_label} screening decisions")}
        {img("assets/tag_side_boxplot.png", "Projected tag side distributions")}
        {img("assets/feature_counts.png", "Calibration feature counts")}
      </div>
    </section>

    <section>
      <h2>Frame Contact Sheets</h2>
      <div class="grid">{contact_html}</div>
    </section>

    <section>
      <h2>Operator Feedback For Next Capture</h2>
      <p>Use the large board or move the board closer so median AprilTag side length is at least <code>80 px</code>, preferably <code>100-150 px</code>. Keep enough non-duplicate board poses to reach at least <code>60</code> nonzero calibration-feature frames per camera.</p>
      <p class="paths">Source root: {html.escape(str(root_path))}</p>
      <p class="paths">Report directory: {html.escape(str(output_root))}</p>
    </section>
  </main>
</body>
</html>
"""
    return body


def build_report_index(index_root, large_report, small_report):
    index_root = Path(index_root).resolve()
    large_href = os.path.relpath(Path(large_report).resolve(), index_root)
    small_href = os.path.relpath(Path(small_report).resolve(), index_root)
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Seeker Calibration Report Index</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f7f8fa; color: #111827; line-height: 1.45; }}
    main {{ max-width: 900px; margin: 0 auto; padding: 48px; }}
    h1 {{ margin: 0 0 12px; font-size: 34px; }}
    p {{ color: #334155; }}
    .links {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 24px; }}
    a {{ display: block; padding: 18px; border: 1px solid #d9dee7; border-radius: 8px; background: #fff; color: #111827; text-decoration: none; font-weight: 700; }}
    span {{ display: block; margin-top: 8px; color: #64748b; font-weight: 400; font-size: 14px; }}
  </style>
</head>
<body>
  <main>
    <h1>Seeker Calibration Report Index</h1>
    <p>Each report below is scoped to one capture. The large and small captures are intentionally not merged into a single calibration report.</p>
    <div class="links">
      <a href="{html.escape(large_href)}">Seeker Large Capture Calibration Report<span>First-pass four-fisheye calibration and BA reprojection QA.</span></a>
      <a href="{html.escape(small_href)}">Seeker Small Capture QA Report<span>Standalone capture-quality diagnostic report.</span></a>
    </div>
  </main>
</body>
</html>
"""
    ensure_dir(index_root).joinpath("index.html").write_text(body)


def write_capture_qa_assets(output_root, capture_root, capture, capture_summary, capture_label):
    assets_dir = ensure_dir(Path(output_root) / "assets")
    plot_bar(
        assets_dir / "capture_decisions.png",
        f"{capture_label} screening decisions",
        CAMERAS,
        [
            ("selected", [capture_summary[c]["selected"] for c in CAMERAS], "#16a34a"),
            ("no_board", [capture_summary[c]["no_board"] for c in CAMERAS], "#64748b"),
            ("near_duplicate", [capture_summary[c]["near_duplicate"] for c in CAMERAS], "#d97706"),
        ],
        "frames",
    )
    plot_box(
        assets_dir / "tag_side_boxplot.png",
        "Projected tag side length",
        {cam: capture["selected_tag_side"][cam] for cam in CAMERAS},
        "sqrt(board area) [px]",
    )
    plot_bar(
        assets_dir / "feature_counts.png",
        "Calibration feature extraction",
        CAMERAS,
        [
            ("nonzero frames", [capture_summary[c]["nonzero_feature_frames"] for c in CAMERAS], "#2f6df6"),
            ("total features / 300", [capture_summary[c]["total_features"] / 300.0 for c in CAMERAS], "#16a34a"),
        ],
        "count",
    )

    contact_sheets = []
    for cam in CAMERAS:
        name = f"contact_{cam}.jpg"
        ok = make_contact_sheet(assets_dir / name, Path(capture_root) / "images" / cam, f"{CAM_LABELS[cam]} selected frames")
        if ok:
            contact_sheets.append((name, f"{CAM_LABELS[cam]} selected-frame contact sheet"))
    return contact_sheets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large-root", default="/tmp/camera_calibration_mcap/seeker_large_intrinsics_full")
    parser.add_argument("--small-root", default="/tmp/camera_calibration_mcap/seeker_small_intrinsics_full")
    parser.add_argument("--large-output-root", default="reports/seeker_large_capture_calibration_20260526")
    parser.add_argument("--small-output-root", default="reports/seeker_small_capture_qa_20260526")
    parser.add_argument("--output-root", default="reports/seeker_calibration_20260526", help="Index page linking the per-capture reports.")
    args = parser.parse_args()

    large = load_capture(args.large_root, feature_preference="all")
    large_summary = summarize_capture(large)
    large_output_root = ensure_dir(args.large_output_root)
    large_assets_dir = ensure_dir(large_output_root / "assets")
    large_contact_sheets = write_capture_qa_assets(large_output_root, args.large_root, large, large_summary, "Large capture")

    kb8 = read_json(Path(args.large_root) / "kb8_intrinsics_firstpass.json")
    omni_yaml = yaml.safe_load((Path(args.large_root) / "kalibr_cam_chain_seeker_driver.yaml").read_text())
    omni_meta = read_json(Path(args.large_root) / "kalibr_cam_chain_seeker_driver.metadata.json")
    extrinsics = read_json(Path(args.large_root) / "assumed_four_fisheye_extrinsics_seeker1_rows.json")
    reprojection_data = load_reprojection_data(args.large_root)
    pair_overlaps = compute_pair_overlaps(large["feature_by_src"])

    plot_coverage(large_assets_dir / "board_coverage.png", large, reprojection_data)
    plot_intrinsics(large_assets_dir / "intrinsics_overview.png", kb8)
    plot_kb8_curves(large_assets_dir / "kb8_radial_curves.png", kb8)
    plot_omni_fit_error(large_assets_dir / "omni_fit_error.png", kb8, omni_yaml)
    plot_extrinsics(large_assets_dir / "extrinsics_graph.png", extrinsics, pair_overlaps)

    large_report_summary = {
        "capture_label": "Large capture",
        "large_root": str(Path(args.large_root)),
        "large_capture": large_summary,
        "feature_point_coverage": {
            cam: {
                "imagesets": reprojection_data[cam]["imagesets"],
                "features": reprojection_data[cam]["stats"]["count"],
                "image_size": reprojection_data[cam]["image_size"],
                "skipped": reprojection_data[cam]["skipped"],
                "source_dir": reprojection_data[cam]["source_dir"],
            }
            for cam in CAMERAS
        },
        "feature_reprojection_errors": {
            cam: {
                "computed": reprojection_data[cam]["stats"],
                "source_report": {
                    key: reprojection_data[cam]["report_info"].get(key)
                    for key in [
                        "reprojection_error_count",
                        "reprojection_error_median",
                        "reprojection_error_average",
                        "reprojection_error_maximum",
                    ]
                },
                "skipped": reprojection_data[cam]["skipped"],
            }
            for cam in CAMERAS
        },
        "pair_overlaps": pair_overlaps,
        "intrinsics": {cam: kb8["cameras"][cam] for cam in CAMERAS},
        "omni_fit_quality": omni_meta.get("fit_quality", {}),
        "extrinsics_source": str(Path(args.large_root) / "assumed_four_fisheye_extrinsics_seeker1_rows.json"),
        "output_root": str(large_output_root.resolve()),
    }
    (large_output_root / "summary.json").write_text(json.dumps(large_report_summary, indent=2, sort_keys=True) + "\n")

    large_report_title = "Seeker Large Capture Calibration Report"
    large_html = build_html(large_report_title, large_report_summary, {"contact_sheets": large_contact_sheets}, large_output_root.resolve())
    large_report_path = large_output_root.resolve() / "index.html"
    large_report_path.write_text(large_html)

    report_paths = [large_report_path]
    small_report_path = None
    if Path(args.small_root).exists():
        small = load_capture(args.small_root, feature_preference="pattern3")
        small_summary = summarize_capture(small)
        small_output_root = ensure_dir(args.small_output_root)
        small_contact_sheets = write_capture_qa_assets(small_output_root, args.small_root, small, small_summary, "Small capture")
        small_report_summary = {
            "capture_label": "Small capture",
            "capture_root": str(Path(args.small_root)),
            "capture": small_summary,
            "output_root": str(small_output_root.resolve()),
        }
        (small_output_root / "summary.json").write_text(json.dumps(small_report_summary, indent=2, sort_keys=True) + "\n")
        small_html = build_capture_qa_html(
            "Seeker Small Capture QA Report",
            "Small capture",
            small_summary,
            {"contact_sheets": small_contact_sheets},
            small_output_root.resolve(),
            Path(args.small_root),
        )
        small_report_path = small_output_root.resolve() / "index.html"
        small_report_path.write_text(small_html)
        report_paths.append(small_report_path)

    if small_report_path:
        build_report_index(args.output_root, large_report_path, small_report_path)
        report_paths.append(Path(args.output_root).resolve() / "index.html")

    for path in report_paths:
        print(path)


if __name__ == "__main__":
    main()
