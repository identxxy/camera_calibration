#!/usr/bin/env python3
"""Generate an HTML calibration report with reprojection-arrow plots."""

import argparse
import html
import json
import math
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm


def read_dataset(path):
    path = Path(path)
    with path.open("rb") as f:
        if f.read(10) != b"calib_data":
            raise ValueError(f"invalid dataset header: {path}")
        version = struct.unpack(">I", f.read(4))[0]
        if version not in (0, 1):
            raise ValueError(f"unsupported dataset version {version}: {path}")

        camera_count = struct.unpack(">I", f.read(4))[0]
        image_sizes = []
        for _ in range(camera_count):
            width = struct.unpack(">I", f.read(4))[0]
            height = struct.unpack(">I", f.read(4))[0]
            image_sizes.append((width, height))

        imagesets = []
        imageset_count = struct.unpack(">I", f.read(4))[0]
        for _ in range(imageset_count):
            filename_len = struct.unpack(">I", f.read(4))[0]
            filename = f.read(filename_len).decode("utf-8", errors="replace")
            features_by_camera = []
            for _camera in range(camera_count):
                feature_count = struct.unpack(">I", f.read(4))[0]
                if feature_count:
                    raw = f.read(feature_count * 12)
                    values = np.frombuffer(raw, dtype=np.dtype([
                        ("x", "<f4"),
                        ("y", "<f4"),
                        ("id", ">i4"),
                    ])).copy()
                else:
                    values = np.zeros(0, dtype=[("x", "<f4"), ("y", "<f4"), ("id", ">i4")])
                features_by_camera.append(values)
            imagesets.append({"filename": filename, "features": features_by_camera})

        known_geometries = []
        known_geometry_count = struct.unpack(">I", f.read(4))[0]
        for _ in range(known_geometry_count):
            cell_length = struct.unpack("<f", f.read(4))[0]
            feature_id_to_position = {}
            count_2d = struct.unpack(">I", f.read(4))[0]
            for _ in range(count_2d):
                feature_id, x, y = struct.unpack(">iii", f.read(12))
                feature_id_to_position[feature_id] = (x, y)
            feature_id_to_position3d = {}
            if version >= 1:
                count_3d = struct.unpack(">I", f.read(4))[0]
                for _ in range(count_3d):
                    feature_id = struct.unpack(">i", f.read(4))[0]
                    xyz = struct.unpack("<fff", f.read(12))
                    feature_id_to_position3d[feature_id] = xyz
            known_geometries.append({
                "cell_length": cell_length,
                "feature_id_to_position": feature_id_to_position,
                "feature_id_to_position3d": feature_id_to_position3d,
            })

    return {
        "camera_count": camera_count,
        "image_sizes": image_sizes,
        "imagesets": imagesets,
        "known_geometries": known_geometries,
    }


def load_pose_file(path):
    node = yaml.safe_load(Path(path).read_text())
    pose_count = int(node["pose_count"])
    used = np.zeros(pose_count, dtype=bool)
    rotations = np.repeat(np.eye(3)[None, :, :], pose_count, axis=0)
    translations = np.zeros((pose_count, 3), dtype=np.float64)
    for pose in node.get("poses", []):
        index = int(pose["index"])
        used[index] = True
        translations[index] = [float(pose["tx"]), float(pose["ty"]), float(pose["tz"])]
        rotations[index] = quat_to_matrix(
            float(pose["qw"]),
            float(pose["qx"]),
            float(pose["qy"]),
            float(pose["qz"]),
        )
    return used, rotations, translations


def quat_to_matrix(w, x, y, z):
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_intrinsics(path):
    node = yaml.safe_load(Path(path).read_text())
    if node["type"] != "CentralOpenCVModel":
        raise ValueError(f"unsupported camera model for Python reprojection report: {node['type']}")
    return {
        "type": node["type"],
        "width": int(node["width"]),
        "height": int(node["height"]),
        "parameters": np.asarray(node["parameters"], dtype=np.float64),
    }


def load_points(path):
    node = yaml.safe_load(Path(path).read_text())
    flat_points = np.asarray(node["points"], dtype=np.float64)
    points = flat_points.reshape((-1, 3))
    feature_to_point = {
        int(item["feature_id"]): int(item["point_index"])
        for item in node["feature_id_to_point_index"]
    }
    return points, feature_to_point


def load_manifest(path):
    if not path:
        return {}
    result = {}
    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            row = dict(zip(header, fields))
            camera_index = int(row.get("camera_index", row.get("camera", -1)))
            result[camera_index] = row
    return result


def project_central_opencv(points_camera, intrinsics):
    params = intrinsics["parameters"]
    fx, fy, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2 = params
    z = points_camera[:, 2]
    valid = z > 0
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    x[valid] = points_camera[valid, 0] / z[valid]
    y[valid] = points_camera[valid, 1] / z[valid]
    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
    dx = 2 * p1 * xy + p2 * (r2 + 2 * x2)
    dy = 2 * p2 * xy + p1 * (r2 + 2 * y2)
    u = fx * (x * radial + dx) + cx
    v = fy * (y * radial + dy) + cy
    valid &= (u >= 0) & (v >= 0) & (u < intrinsics["width"]) & (v < intrinsics["height"])
    return np.stack([u, v], axis=1), valid


def compose_pose(rotation_a, translation_a, rotation_b, translation_b):
    rotation = rotation_a @ rotation_b
    translation = rotation_a @ translation_b + translation_a
    return rotation, translation


def compute_camera_residuals(dataset, state, camera_index):
    intrinsics = state["intrinsics"][camera_index]
    points = state["points"]
    feature_to_point = state["feature_to_point"]
    camera_r = state["camera_rotations"][camera_index]
    camera_t = state["camera_translations"][camera_index]

    observed_chunks = []
    error_chunks = []
    image_index_chunks = []
    skipped_missing_point = 0
    skipped_projection = 0

    for imageset_index, imageset in enumerate(dataset["imagesets"]):
        if imageset_index >= len(state["image_used"]) or not state["image_used"][imageset_index]:
            continue
        features = imageset["features"][camera_index]
        if len(features) == 0:
            continue

        point_indices = []
        keep_indices = []
        for feature_index, feature_id in enumerate(features["id"]):
            point_index = feature_to_point.get(int(feature_id))
            if point_index is None:
                skipped_missing_point += 1
                continue
            point_indices.append(point_index)
            keep_indices.append(feature_index)
        if not point_indices:
            continue

        observed = np.stack([features["x"][keep_indices], features["y"][keep_indices]], axis=1).astype(np.float64)
        global_points = points[np.asarray(point_indices, dtype=np.int64)]
        rig_r = state["rig_rotations"][imageset_index]
        rig_t = state["rig_translations"][imageset_index]
        image_r, image_t = compose_pose(camera_r, camera_t, rig_r, rig_t)
        camera_points = (image_r @ global_points.T).T + image_t
        projected, valid = project_central_opencv(camera_points, intrinsics)
        skipped_projection += int((~valid).sum())
        if not np.any(valid):
            continue

        observed_chunks.append(observed[valid])
        error_chunks.append(projected[valid] - observed[valid])
        image_index_chunks.append(np.full(int(valid.sum()), imageset_index, dtype=np.int32))

    if not observed_chunks:
        return {
            "observed": np.zeros((0, 2), dtype=np.float64),
            "errors": np.zeros((0, 2), dtype=np.float64),
            "image_indices": np.zeros(0, dtype=np.int32),
            "skipped_missing_point": skipped_missing_point,
            "skipped_projection": skipped_projection,
        }

    return {
        "observed": np.concatenate(observed_chunks, axis=0),
        "errors": np.concatenate(error_chunks, axis=0),
        "image_indices": np.concatenate(image_index_chunks, axis=0),
        "skipped_missing_point": skipped_missing_point,
        "skipped_projection": skipped_projection,
    }


def load_state(state_dir, camera_count):
    state_dir = Path(state_dir)
    image_used, rig_rotations, rig_translations = load_pose_file(state_dir / "rig_tr_global.yaml")
    identity_rotation = np.eye(3)[None, :, :]
    identity_pose = (
        np.linalg.norm(rig_translations, axis=1) < 1e-12
    ) & (
        np.linalg.norm(rig_rotations - identity_rotation, axis=(1, 2)) < 1e-12
    )
    image_used &= ~identity_pose
    _camera_used, camera_rotations, camera_translations = load_pose_file(state_dir / "camera_tr_rig.yaml")
    intrinsics = [load_intrinsics(state_dir / f"intrinsics{idx}.yaml") for idx in range(camera_count)]
    points, feature_to_point = load_points(state_dir / "points.yaml")
    return {
        "image_used": image_used,
        "rig_rotations": rig_rotations,
        "rig_translations": rig_translations,
        "camera_rotations": camera_rotations,
        "camera_translations": camera_translations,
        "intrinsics": intrinsics,
        "points": points,
        "feature_to_point": feature_to_point,
    }


def sample_indices(count, max_count):
    if count <= max_count:
        return np.arange(count)
    rng = np.random.default_rng(20260527)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def plot_reprojection_arrows(camera_index, camera_label, intrinsics, residuals, output_path, max_arrows):
    observed = residuals["observed"]
    errors = residuals["errors"]
    width = intrinsics["width"]
    height = intrinsics["height"]
    magnitudes = np.linalg.norm(errors, axis=1)

    fig_width = 12
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.set_facecolor("#101214")
    ax.set_title(f"Camera {camera_index}: {camera_label}", fontsize=11)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.grid(color="#333333", linewidth=0.4, alpha=0.4)

    if len(observed):
        heat = ax.hexbin(
            observed[:, 0],
            observed[:, 1],
            gridsize=80,
            extent=(0, width, 0, height),
            mincnt=1,
            bins="log",
            cmap="Greys",
            alpha=0.35,
            linewidths=0,
        )
        heat.set_zorder(0)

        indices = sample_indices(len(observed), max_arrows)
        obs = observed[indices]
        err = errors[indices]
        mag = np.clip(magnitudes[indices], 1e-3, None)
        norm = LogNorm(vmin=0.01, vmax=max(0.5, float(np.percentile(np.clip(magnitudes, 1e-3, None), 99))))
        display_scale = 45.0
        lengths = np.linalg.norm(err, axis=1)
        scale_factor = np.ones_like(lengths)
        nonzero = lengths > 1e-9
        scaled_lengths = np.clip(lengths[nonzero] * display_scale, 2.0, 45.0)
        scale_factor[nonzero] = scaled_lengths / lengths[nonzero]
        err_display = err * scale_factor[:, None]
        quiver = ax.quiver(
            obs[:, 0],
            obs[:, 1],
            err_display[:, 0],
            err_display[:, 1],
            mag,
            cmap="turbo",
            norm=norm,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0013,
            headwidth=3.4,
            headlength=4.4,
            headaxislength=3.8,
            alpha=0.92,
        )
        cbar = fig.colorbar(quiver, ax=ax, shrink=0.82)
        cbar.set_label("reprojection error [px], log scale")
        ax.text(
            0.01,
            0.99,
            f"{len(observed):,} residuals, {len(indices):,} arrows shown",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="#e8e8e8",
            fontsize=9,
            bbox={"facecolor": "#111111", "alpha": 0.65, "edgecolor": "none", "pad": 4},
        )
    else:
        ax.text(0.5, 0.5, "no valid residuals", transform=ax.transAxes, ha="center", va="center")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_html(output_path, summary, camera_rows):
    rows = []
    for row in camera_rows:
        rows.append(
            "<tr>"
            f"<td>{row['camera_index']}</td>"
            f"<td>{html.escape(row['camera_label'])}</td>"
            f"<td>{row['residual_count']:,}</td>"
            f"<td>{row['median_error_px']:.4f}</td>"
            f"<td>{row['mean_error_px']:.4f}</td>"
            f"<td>{row['p90_error_px']:.4f}</td>"
            f"<td>{row['max_error_px']:.4f}</td>"
            "</tr>"
        )

    figures = []
    for row in camera_rows:
        rel = Path(row["plot_path"]).name
        figures.append(
            "<section class='camera'>"
            f"<h2>Camera {row['camera_index']} <span>{html.escape(row['camera_label'])}</span></h2>"
            f"<img src='{html.escape(rel)}' alt='Camera {row['camera_index']} reprojection arrows'>"
            "</section>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Inner Camera Calibration Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f7; color: #1f2328; }}
    header {{ padding: 28px 36px 18px; background: #20242a; color: white; }}
    header h1 {{ margin: 0 0 10px; font-size: 26px; }}
    header p {{ margin: 4px 0; color: #c9d1d9; }}
    main {{ padding: 24px 36px 40px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin-bottom: 28px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 9px 10px; text-align: right; font-size: 13px; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #eef1f4; font-weight: 650; }}
    .camera {{ margin: 0 0 32px; background: white; padding: 14px 14px 18px; border: 1px solid #d8dee4; }}
    .camera h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .camera h2 span {{ color: #57606a; font-weight: 500; }}
    .camera img {{ display: block; width: 100%; height: auto; border: 1px solid #d8dee4; }}
    code {{ background: #eef1f4; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Inner Camera Calibration Report</h1>
    <p>Dataset: <code>{html.escape(summary['dataset'])}</code></p>
    <p>State: <code>{html.escape(summary['state_dir'])}</code></p>
    <p>Projection: CentralOpenCVModel; arrows show projected minus observed residual, color uses log-level reprojection error.</p>
  </header>
  <main>
    <table>
      <thead><tr><th>Camera</th><th>Label</th><th>Residuals</th><th>Median px</th><th>Mean px</th><th>P90 px</th><th>Max px</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    {''.join(figures)}
  </main>
</body>
</html>
"""
    Path(output_path).write_text(html_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--max-arrows-per-camera", type=int, default=60000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = read_dataset(args.dataset)
    state = load_state(args.state_dir, dataset["camera_count"])
    manifest = load_manifest(args.manifest)

    camera_rows = []
    for camera_index in range(dataset["camera_count"]):
        label_parts = []
        row = manifest.get(camera_index, {})
        for key in ("camera_name", "staged_camera", "camera_id", "user_id", "sn"):
            if row.get(key):
                label_parts.append(row[key])
        camera_label = " / ".join(label_parts) if label_parts else f"camera{camera_index}"

        residuals = compute_camera_residuals(dataset, state, camera_index)
        magnitudes = np.linalg.norm(residuals["errors"], axis=1)
        plot_path = output_dir / f"camera{camera_index:02d}_reprojection_arrows_log.png"
        plot_reprojection_arrows(
            camera_index,
            camera_label,
            state["intrinsics"][camera_index],
            residuals,
            plot_path,
            args.max_arrows_per_camera,
        )

        if len(magnitudes):
            metrics = {
                "median_error_px": float(np.median(magnitudes)),
                "mean_error_px": float(np.mean(magnitudes)),
                "p90_error_px": float(np.percentile(magnitudes, 90)),
                "max_error_px": float(np.max(magnitudes)),
            }
        else:
            metrics = {
                "median_error_px": float("nan"),
                "mean_error_px": float("nan"),
                "p90_error_px": float("nan"),
                "max_error_px": float("nan"),
            }

        camera_rows.append({
            "camera_index": camera_index,
            "camera_label": camera_label,
            "residual_count": int(len(magnitudes)),
            "skipped_missing_point": int(residuals["skipped_missing_point"]),
            "skipped_projection": int(residuals["skipped_projection"]),
            "plot_path": str(plot_path),
            **metrics,
        })

    summary = {
        "dataset": str(Path(args.dataset).resolve()),
        "state_dir": str(Path(args.state_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "camera_count": dataset["camera_count"],
        "imageset_count": len(dataset["imagesets"]),
    }
    (output_dir / "summary.json").write_text(json.dumps({
        "summary": summary,
        "cameras": camera_rows,
    }, indent=2), encoding="utf-8")

    with (output_dir / "camera_metrics.tsv").open("w", encoding="utf-8") as f:
        fields = [
            "camera_index",
            "camera_label",
            "residual_count",
            "median_error_px",
            "mean_error_px",
            "p90_error_px",
            "max_error_px",
            "skipped_missing_point",
            "skipped_projection",
            "plot_path",
        ]
        f.write("\t".join(fields) + "\n")
        for row in camera_rows:
            f.write("\t".join(str(row[field]) for field in fields) + "\n")

    write_html(output_dir / "index.html", summary, camera_rows)
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
