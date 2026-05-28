#!/usr/bin/env python3
"""Generate a camera-rig extrinsics report with camera 0 as the gauge."""

import argparse
import csv
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def quat_to_matrix(qx, qy, qz, qw):
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat(rotation):
    trace = np.trace(rotation)
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


def pose_to_matrix(pose):
    rotation = quat_to_matrix(
        float(pose["qx"]),
        float(pose["qy"]),
        float(pose["qz"]),
        float(pose["qw"]),
    )
    translation = np.array([
        float(pose["tx"]),
        float(pose["ty"]),
        float(pose["tz"]),
    ], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def matrix_to_pose(index, matrix):
    qx, qy, qz, qw = matrix_to_quat(matrix[:3, :3])
    tx, ty, tz = matrix[:3, 3]
    return {
        "index": int(index),
        "tx": float(tx),
        "ty": float(ty),
        "tz": float(tz),
        "qx": float(qx),
        "qy": float(qy),
        "qz": float(qz),
        "qw": float(qw),
    }


def load_poses(path):
    node = yaml.safe_load(Path(path).read_text())
    pose_count = int(node["pose_count"])
    used = np.zeros(pose_count, dtype=bool)
    poses = [np.eye(4, dtype=np.float64) for _ in range(pose_count)]
    for pose in node.get("poses", []):
        index = int(pose["index"])
        used[index] = True
        poses[index] = pose_to_matrix(pose)
    return used, poses


def write_poses(path, used, poses):
    path = Path(path)
    lines = [
        "# Each pose gives the B_tr_A transformation (i.e., A to B with right-multiplication), where the spaces A and B are defined by the filename. Quaternions are written as used by the Eigen library.",
        f"pose_count: {len(poses)}",
        "poses:",
    ]
    for index, matrix in enumerate(poses):
        if not used[index]:
            continue
        pose = matrix_to_pose(index, matrix)
        lines.extend([
            f"  - index: {index}",
            f"    tx: {pose['tx']:.14g}",
            f"    ty: {pose['ty']:.14g}",
            f"    tz: {pose['tz']:.14g}",
            f"    qx: {pose['qx']:.14g}",
            f"    qy: {pose['qy']:.14g}",
            f"    qz: {pose['qz']:.14g}",
            f"    qw: {pose['qw']:.14g}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rotation_angle_deg(matrix):
    trace = np.trace(matrix[:3, :3])
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def normalize_camera_poses(camera_used, camera_poses, reference_camera):
    reference_inv = np.linalg.inv(camera_poses[reference_camera])
    normalized = [pose @ reference_inv for pose in camera_poses]
    return camera_used.copy(), normalized


def normalize_rig_poses(image_used, rig_poses, reference_camera_pose):
    return image_used.copy(), [reference_camera_pose @ pose for pose in rig_poses]


def camera_center_in_reference(camera_tr_reference):
    return np.linalg.inv(camera_tr_reference)[:3, 3]


def read_tsv(path):
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_rows(camera_used, normalized_poses, baseline_poses=None):
    rows = []
    for index, pose in enumerate(normalized_poses):
        center = camera_center_in_reference(pose)
        translation = pose[:3, 3]
        row = {
            "camera_index": index,
            "used": bool(camera_used[index]),
            "tx": float(translation[0]),
            "ty": float(translation[1]),
            "tz": float(translation[2]),
            "distance_m": float(np.linalg.norm(translation)),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "center_z": float(center[2]),
            "rotation_deg": float(rotation_angle_deg(pose)),
            "delta_translation_m": float("nan"),
            "delta_rotation_deg": float("nan"),
        }
        if baseline_poses is not None:
            delta = pose @ np.linalg.inv(baseline_poses[index])
            row["delta_translation_m"] = float(np.linalg.norm(delta[:3, 3]))
            row["delta_rotation_deg"] = float(rotation_angle_deg(delta))
        rows.append(row)
    return rows


def plot_layout(rows, output_path):
    used_rows = [row for row in rows if row["used"]]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d8dee4", linewidth=0.7)
    ax.set_xlabel("camera center x in camera0 frame [m]")
    ax.set_ylabel("camera center z in camera0 frame [m]")
    ax.set_title("Rig Layout, Top-Down Approximation")
    for row in used_rows:
        x = row["center_x"]
        z = row["center_z"]
        ax.scatter([x], [z], s=50, color="#0969da")
        ax.text(x, z, f"  c{row['camera_index']}", va="center", fontsize=9)
    ax.scatter([0], [0], marker="+", s=130, color="#d1242f")
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def set_axes_equal(ax, points):
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    radius = max(0.35, float(np.max(np.linalg.norm(points - center, axis=1))))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_layout_3d(rows, normalized_poses, output_path):
    used_indices = [row["camera_index"] for row in rows if row["used"]]
    centers = []
    for index in used_indices:
        centers.append(camera_center_in_reference(normalized_poses[index]))
    if not centers:
        centers = [np.zeros(3, dtype=np.float64)]

    fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Rig Layout in Camera0 Frame")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    axis_len = 0.13
    frustum_depth = 0.22
    frustum_half_w = 0.11
    frustum_half_h = 0.08
    for row in rows:
        if not row["used"]:
            continue
        index = row["camera_index"]
        camera_tr_ref = normalized_poses[index]
        ref_tr_camera = np.linalg.inv(camera_tr_ref)
        center = ref_tr_camera[:3, 3]
        rotation = ref_tr_camera[:3, :3]

        ax.scatter(center[0], center[1], center[2], s=44, color="#0969da")
        ax.text(center[0], center[1], center[2], f" c{index}", fontsize=9)

        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        z_axis = rotation[:, 2]
        ax.quiver(*center, *(axis_len * x_axis), color="#d1242f", arrow_length_ratio=0.25, linewidth=1.2)
        ax.quiver(*center, *(axis_len * y_axis), color="#1a7f37", arrow_length_ratio=0.25, linewidth=1.2)
        ax.quiver(*center, *(axis_len * z_axis), color="#0969da", arrow_length_ratio=0.25, linewidth=1.2)

        corners = [
            center + frustum_depth * z_axis + frustum_half_w * x_axis + frustum_half_h * y_axis,
            center + frustum_depth * z_axis - frustum_half_w * x_axis + frustum_half_h * y_axis,
            center + frustum_depth * z_axis - frustum_half_w * x_axis - frustum_half_h * y_axis,
            center + frustum_depth * z_axis + frustum_half_w * x_axis - frustum_half_h * y_axis,
        ]
        for corner in corners:
            ax.plot([center[0], corner[0]], [center[1], corner[1]], [center[2], corner[2]], color="#57606a", linewidth=0.7)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#57606a", linewidth=0.7)

    ax.scatter(0, 0, 0, marker="+", s=120, color="#d1242f")
    set_axes_equal(ax, centers)
    ax.view_init(elev=24, azim=-58)
    fig.savefig(output_path, dpi=135)
    plt.close(fig)


def write_frustum_obj(path, rows, normalized_poses):
    vertices = []
    lines = []
    frustum_depth = 0.22
    frustum_half_w = 0.11
    frustum_half_h = 0.08

    for row in rows:
        if not row["used"]:
            continue
        index = row["camera_index"]
        ref_tr_camera = np.linalg.inv(normalized_poses[index])
        center = ref_tr_camera[:3, 3]
        rotation = ref_tr_camera[:3, :3]
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        z_axis = rotation[:, 2]
        corners = [
            center + frustum_depth * z_axis + frustum_half_w * x_axis + frustum_half_h * y_axis,
            center + frustum_depth * z_axis - frustum_half_w * x_axis + frustum_half_h * y_axis,
            center + frustum_depth * z_axis - frustum_half_w * x_axis - frustum_half_h * y_axis,
            center + frustum_depth * z_axis + frustum_half_w * x_axis - frustum_half_h * y_axis,
        ]
        base = len(vertices) + 1
        vertices.extend([center] + corners)
        lines.extend([
            (base, base + 1),
            (base, base + 2),
            (base, base + 3),
            (base, base + 4),
            (base + 1, base + 2),
            (base + 2, base + 3),
            (base + 3, base + 4),
            (base + 4, base + 1),
        ])

    with Path(path).open("w", encoding="utf-8") as f:
        f.write("# Camera frustums in camera0 frame\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for a, b in lines:
            f.write(f"l {a} {b}\n")


def write_table_tsv(path, rows):
    fields = [
        "camera_index",
        "used",
        "tx",
        "ty",
        "tz",
        "distance_m",
        "center_x",
        "center_y",
        "center_z",
        "rotation_deg",
        "delta_translation_m",
        "delta_rotation_deg",
    ]
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for row in rows:
            f.write("\t".join(str(row[field]) for field in fields) + "\n")


def write_html(path, summary, rows, pnp_rows, edge_rows):
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['camera_index']}</td>"
            f"<td>{'yes' if row['used'] else 'no'}</td>"
            f"<td>{row['tx']:.4f}</td>"
            f"<td>{row['ty']:.4f}</td>"
            f"<td>{row['tz']:.4f}</td>"
            f"<td>{row['distance_m']:.4f}</td>"
            f"<td>{row['center_x']:.4f}</td>"
            f"<td>{row['center_y']:.4f}</td>"
            f"<td>{row['center_z']:.4f}</td>"
            f"<td>{row['delta_translation_m']:.5f}</td>"
            f"<td>{row['delta_rotation_deg']:.5f}</td>"
            "</tr>"
        )

    pnp_table = []
    for row in pnp_rows:
        pnp_table.append(
            "<tr>"
            f"<td>{html.escape(row.get('camera_index', ''))}</td>"
            f"<td>{html.escape(row.get('stage_name', ''))}</td>"
            f"<td>{html.escape(row.get('connected', ''))}</td>"
            f"<td>{html.escape(row.get('positive_views', ''))}</td>"
            f"<td>{html.escape(row.get('solved_views', ''))}</td>"
            f"<td>{html.escape(row.get('median_view_error_px', ''))}</td>"
            "</tr>"
        )

    edge_table = []
    for row in edge_rows[:24]:
        edge_table.append(
            "<tr>"
            f"<td>{html.escape(row.get('camera_a', ''))}</td>"
            f"<td>{html.escape(row.get('camera_b', ''))}</td>"
            f"<td>{html.escape(row.get('shared_frames', ''))}</td>"
            f"<td>{html.escape(row.get('median_translation_residual_m', ''))}</td>"
            f"<td>{html.escape(row.get('median_rotation_residual_deg', ''))}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Rig Extrinsics Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f7; color: #1f2328; }}
    header {{ padding: 26px 34px 16px; background: #20242a; color: white; }}
    header h1 {{ margin: 0 0 10px; font-size: 25px; }}
    header p {{ margin: 4px 0; color: #c9d1d9; }}
    main {{ padding: 22px 34px 38px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 0 0 24px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 8px 9px; text-align: right; font-size: 13px; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #eef1f4; font-weight: 650; }}
    img {{ display: block; max-width: 680px; width: 100%; background: white; border: 1px solid #d8dee4; margin-bottom: 24px; }}
    code {{ background: #eef1f4; padding: 2px 5px; border-radius: 4px; }}
    h2 {{ font-size: 18px; margin: 26px 0 10px; }}
  </style>
</head>
<body>
  <header>
    <h1>Rig Extrinsics Report</h1>
    <p>State: <code>{html.escape(summary['state_dir'])}</code></p>
    <p>Reference camera: <code>{summary['reference_camera']}</code>; all exported poses are re-gauged so camera0 is identity.</p>
  </header>
  <main>
    <img src="rig_layout_3d.png" alt="Rig 3D layout">
    <img src="rig_layout_topdown.png" alt="Rig top-down layout">
    <h2>Normalized Camera Extrinsics</h2>
    <table>
      <thead><tr><th>Camera</th><th>Used</th><th>tx</th><th>ty</th><th>tz</th><th>|t| m</th><th>center x</th><th>center y</th><th>center z</th><th>delta t m</th><th>delta rot deg</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
    <h2>PnP Summary</h2>
    <table>
      <thead><tr><th>Camera</th><th>Stage</th><th>Connected</th><th>Positive Views</th><th>Solved Views</th><th>Median Error px</th></tr></thead>
      <tbody>{''.join(pnp_table)}</tbody>
    </table>
    <h2>Top Pairwise Edges</h2>
    <table>
      <thead><tr><th>Camera A</th><th>Camera B</th><th>Shared Frames</th><th>Median Trans Residual m</th><th>Median Rot Residual deg</th></tr></thead>
      <tbody>{''.join(edge_table)}</tbody>
    </table>
  </main>
</body>
</html>
"""
    Path(path).write_text(html_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-camera", type=int, default=0)
    parser.add_argument("--baseline-state-dir", default="")
    parser.add_argument("--pnp-summary", default="")
    parser.add_argument("--pairwise-edges", default="")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_used, camera_poses = load_poses(state_dir / "camera_tr_rig.yaml")
    normalized_used, normalized_camera_poses = normalize_camera_poses(
        camera_used,
        camera_poses,
        args.reference_camera,
    )
    write_poses(output_dir / "camera_tr_camera0.yaml", normalized_used, normalized_camera_poses)

    rig_path = state_dir / "rig_tr_global.yaml"
    if rig_path.exists():
        image_used, rig_poses = load_poses(rig_path)
        normalized_image_used, normalized_rig_poses = normalize_rig_poses(
            image_used,
            rig_poses,
            camera_poses[args.reference_camera],
        )
        write_poses(output_dir / "camera0_tr_global.yaml", normalized_image_used, normalized_rig_poses)

    baseline_poses = None
    if args.baseline_state_dir:
        baseline_used, baseline_camera_poses = load_poses(Path(args.baseline_state_dir) / "camera_tr_rig.yaml")
        _baseline_used, baseline_poses = normalize_camera_poses(
            baseline_used,
            baseline_camera_poses,
            args.reference_camera,
        )

    rows = build_rows(normalized_used, normalized_camera_poses, baseline_poses)
    write_table_tsv(output_dir / "camera_tr_camera0.tsv", rows)
    plot_layout(rows, output_dir / "rig_layout_topdown.png")
    plot_layout_3d(rows, normalized_camera_poses, output_dir / "rig_layout_3d.png")
    write_frustum_obj(output_dir / "camera_frustums_camera0.obj", rows, normalized_camera_poses)

    pnp_rows = read_tsv(args.pnp_summary)
    edge_rows = read_tsv(args.pairwise_edges)
    summary = {
        "state_dir": str(state_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "reference_camera": args.reference_camera,
    }
    (output_dir / "summary.json").write_text(json.dumps({
        "summary": summary,
        "cameras": rows,
        "pnp_summary": pnp_rows,
        "pairwise_edges": edge_rows,
    }, indent=2), encoding="utf-8")
    write_html(output_dir / "index.html", summary, rows, pnp_rows, edge_rows)
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
