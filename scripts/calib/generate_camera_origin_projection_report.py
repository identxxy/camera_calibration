#!/usr/bin/env python3
"""Project rig camera origins into sample images for visual extrinsic checks."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from pathlib import Path

import cv2
import numpy as np
import yaml


POSE_FIELD_RE = re.compile(r"^\s*(?:-\s*)?(index|tx|ty|tz|qx|qy|qz|qw)\s*:\s*(.+?)\s*$")
PARAM_RE = re.compile(r"parameters\s*:\s*\[(.*?)\]", re.S)
WIDTH_RE = re.compile(r"width\s*:\s*(\d+)")
HEIGHT_RE = re.compile(r"height\s*:\s*(\d+)")


COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default="Camera Origin Projection Diagnostic")
    parser.add_argument("--pose-yaml", default="")
    parser.add_argument("--intrinsics-dir", default="")
    parser.add_argument("--studio-yaml", default="")
    parser.add_argument("--camera-group", choices=["all", "outer", "inner"], default="all")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-mount-root", default="")
    parser.add_argument(
        "--capture-kind",
        default="large_marker",
        help="Capture directory name used only when manifest source_dir paths are unavailable.",
    )
    parser.add_argument("--capture-time", default="")
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument("--max-image-width", type=int, default=1600)
    parser.add_argument(
        "--transform-convention",
        choices=["camera_tr_rig", "rig_tr_camera"],
        default="camera_tr_rig",
        help=(
            "camera_tr_rig means each pose maps rig/world points into camera "
            "coordinates. rig_tr_camera inverts each pose before projection."
        ),
    )
    return parser.parse_args()


def parse_float(value):
    return float(str(value).strip().rstrip(","))


def quat_to_rotation(qx, qy, qz, qw):
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0 or not np.all(np.isfinite(q)):
        raise ValueError("Invalid quaternion")
    qx, qy, qz, qw = q / norm
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(row):
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_rotation(row["qx"], row["qy"], row["qz"], row["qw"])
    mat[:3, 3] = [row["tx"], row["ty"], row["tz"]]
    return mat


def studio_pose_to_matrix(row):
    return pose_to_matrix({
        "tx": float(row["tx"]),
        "ty": float(row["ty"]),
        "tz": float(row["tz"]),
        "qx": float(row["qx"]),
        "qy": float(row["qy"]),
        "qz": float(row["qz"]),
        "qw": float(row["qw"]),
    })


def read_pose_yaml(path):
    poses = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = POSE_FIELD_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        if key == "index":
            if current is not None:
                poses[int(current["index"])] = pose_to_matrix(current)
            current = {"index": int(value)}
        elif current is not None:
            current[key] = parse_float(value)
    if current is not None:
        poses[int(current["index"])] = pose_to_matrix(current)
    missing = [idx for idx, pose in poses.items() if pose.shape != (4, 4)]
    if missing:
        raise ValueError(f"Invalid pose rows for indices: {missing}")
    return poses


def read_generic_intrinsics(path):
    text = Path(path).read_text(encoding="utf-8")
    params_match = PARAM_RE.search(text)
    if not params_match:
        raise ValueError(f"Missing parameters in {path}")
    values = [float(item.strip()) for item in params_match.group(1).split(",") if item.strip()]
    if len(values) < 4:
        raise ValueError(f"Too few intrinsic parameters in {path}")
    width_match = WIDTH_RE.search(text)
    height_match = HEIGHT_RE.search(text)
    return {
        "path": str(path),
        "width": int(width_match.group(1)) if width_match else 0,
        "height": int(height_match.group(1)) if height_match else 0,
        "params": values,
    }


def find_intrinsics_file(intrinsics_dir, camera_index, camera_id):
    intrinsics_dir = Path(intrinsics_dir)
    candidates = [
        intrinsics_dir / f"intrinsics{camera_index}_{camera_id}.yaml",
        intrinsics_dir / f"intrinsics{camera_index}.yaml",
        intrinsics_dir / f"opencv_intrinsics{camera_index}_{camera_id}.yaml",
    ]
    for path in candidates:
        if path.is_file() and path.name.startswith("intrinsics"):
            return path
    matches = sorted(intrinsics_dir.glob(f"intrinsics{camera_index}_*.yaml"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No generic intrinsics YAML for camera {camera_index} ({camera_id})")


def read_manifest(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status", "usable") != "usable":
                continue
            row["camera_index"] = int(row["camera_index"])
            row["camera_id"] = str(row["camera_id"])
            row["frame_count"] = int(row.get("frame_count") or 0)
            rows.append(row)
    rows.sort(key=lambda item: item["camera_index"])
    return rows


def read_studio_yaml(path, camera_group):
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cameras = {}
    matched_count = 0
    for camera in doc.get("cameras", []):
        group = str(camera.get("group", ""))
        if camera_group != "all" and group != camera_group:
            continue
        index = int(camera["index"])
        label = str(camera.get("label") or camera.get("camera_id") or index)
        camera_id = str(camera.get("camera_id") or "")
        intr = camera["intrinsics"]
        entry = {
            "index": index,
            "label": label,
            "camera_id": camera_id,
            "group": group,
            "intrinsics": {
                "path": str(path),
                "width": int(intr.get("width") or 0),
                "height": int(intr.get("height") or 0),
                "params": [float(v) for v in intr["parameters"]],
            },
            "pose": studio_pose_to_matrix(camera["camera_tr_studio_rig"]),
        }
        matched_count += 1
        for key in [label, camera_id, str(index), f"camera{index}"]:
            if key:
                cameras.setdefault(key, entry)
        if group == "inner" and label.startswith("inner"):
            cameras.setdefault(label.removeprefix("inner"), entry)
    if matched_count == 0:
        raise ValueError(f"No cameras matched group {camera_group!r} in {path}")
    return cameras


def manifest_camera_match_keys(row):
    keys = [
        row.get("camera_id"),
        row.get("stage_name"),
        row.get("user_id"),
        row.get("label"),
    ]
    try:
        camera_index = int(row["camera_index"])
    except Exception:
        camera_index = None
    if camera_index is not None:
        keys.append(str(camera_index))
        if camera_index >= 24:
            keys.append(f"inner{camera_index - 24}")
    return [str(key) for key in keys if key not in (None, "")]


def attach_studio_camera_rows(rows, studio_cameras):
    output = []
    for row in rows:
        camera = None
        for key in manifest_camera_match_keys(row):
            camera = studio_cameras.get(key)
            if camera is not None:
                break
        if camera is None:
            continue
        row = dict(row)
        row["pose_index"] = camera["index"]
        row["display_label"] = camera["label"]
        row["studio_camera_id"] = camera.get("camera_id", "")
        row["group"] = camera.get("group", "")
        row["intrinsics"] = camera["intrinsics"]
        row["pose"] = camera["pose"]
        output.append(row)
    output.sort(key=lambda item: int(item["pose_index"]))
    return output


def derive_image_dir(row, raw_mount_root, capture_kind, capture_time):
    source_dir = Path(row.get("source_dir", ""))
    if source_dir.is_dir():
        return source_dir
    if raw_mount_root and capture_time:
        candidate = (
            Path(raw_mount_root)
            / row["machine"]
            / f"output/calib/{capture_kind}"
            / capture_time
            / row["camera_id"]
        )
        if candidate.is_dir():
            return candidate
    if raw_mount_root:
        root = Path(raw_mount_root) / row["machine"] / f"output/calib/{capture_kind}"
        matches = sorted(root.glob(f"*/{row['camera_id']}"))
        matches = [path for path in matches if path.is_dir()]
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No image directory for camera {row['camera_index']} ({row['camera_id']})")


def image_for_frame(image_dir, camera_id, frame_index):
    patterns = [
        f"{camera_id}_{frame_index:04d}.jpg",
        f"{camera_id}_{frame_index:05d}.jpg",
        f"*_{frame_index:04d}.jpg",
        f"*_{frame_index:05d}.jpg",
    ]
    for pattern in patterns:
        matches = sorted(Path(image_dir).glob(pattern))
        if matches:
            return matches[0]
    images = sorted(Path(image_dir).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No jpg images in {image_dir}")
    frame_index = max(0, min(frame_index, len(images) - 1))
    return images[frame_index]


def project_point(point_cam, intrinsics):
    x, y, z = [float(v) for v in point_cam[:3]]
    if z <= 1e-9 or not np.all(np.isfinite(point_cam[:3])):
        return None
    params = intrinsics["params"]
    fx, fy, cx, cy = params[:4]
    k1 = params[4] if len(params) > 4 else 0.0
    k2 = params[5] if len(params) > 5 else 0.0
    k3 = params[6] if len(params) > 6 else 0.0
    k4 = params[7] if len(params) > 7 else 0.0
    k5 = params[8] if len(params) > 8 else 0.0
    k6 = params[9] if len(params) > 9 else 0.0
    p1 = params[10] if len(params) > 10 else 0.0
    p2 = params[11] if len(params) > 11 else 0.0
    xn = x / z
    yn = y / z
    r2 = xn * xn + yn * yn
    r4 = r2 * r2
    r6 = r4 * r2
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    radial = numerator / denominator if abs(denominator) > 1e-12 else numerator
    x_dist = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    y_dist = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return np.asarray([fx * x_dist + cx, fy * y_dist + cy, z], dtype=np.float64)


def clip_to_image(point, width, height):
    x, y = float(point[0]), float(point[1])
    cx, cy = width * 0.5, height * 0.5
    dx, dy = x - cx, y - cy
    scales = []
    if abs(dx) > 1e-9:
        scales.extend([(0 - cx) / dx, ((width - 1) - cx) / dx])
    if abs(dy) > 1e-9:
        scales.extend([(0 - cy) / dy, ((height - 1) - cy) / dy])
    valid = [s for s in scales if s > 0]
    if not valid:
        return np.asarray([min(max(x, 0), width - 1), min(max(y, 0), height - 1)])
    s = min(valid)
    return np.asarray([min(max(cx + s * dx, 0), width - 1), min(max(cy + s * dy, 0), height - 1)])


def draw_marker(image, uv, label, color, scale, status):
    x = int(round(float(uv[0]) * scale))
    y = int(round(float(uv[1]) * scale))
    color_bgr = (int(color[2]), int(color[1]), int(color[0]))
    radius = 9 if status == "inside" else 7
    thickness = -1 if status == "inside" else 2
    cv2.circle(image, (x, y), radius, color_bgr, thickness, lineType=cv2.LINE_AA)
    cv2.circle(image, (x, y), radius + 3, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    text = label if status == "inside" else f"{label} {status}"
    cv2.putText(
        image,
        text,
        (x + 12, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x + 12, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_bgr,
        2,
        cv2.LINE_AA,
    )


def build_report(args):
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    poses = {}
    intrinsics = {}
    if args.studio_yaml:
        rows = attach_studio_camera_rows(rows, read_studio_yaml(args.studio_yaml, args.camera_group))
        if not rows:
            raise SystemExit("No manifest rows matched cameras from --studio-yaml.")
        for row in rows:
            index = int(row["pose_index"])
            pose = row["pose"]
            poses[index] = pose if args.transform_convention == "camera_tr_rig" else np.linalg.inv(pose)
            intrinsics[index] = row["intrinsics"]
    else:
        if not args.pose_yaml or not args.intrinsics_dir:
            raise SystemExit("--pose-yaml and --intrinsics-dir are required unless --studio-yaml is used.")
        poses_raw = read_pose_yaml(args.pose_yaml)
        for index, pose in poses_raw.items():
            poses[index] = pose if args.transform_convention == "camera_tr_rig" else np.linalg.inv(pose)
        for row in rows:
            index = row["camera_index"]
            row["pose_index"] = index
            row["display_label"] = row.get("stage_name") or str(index)
            row["group"] = row.get("kind", "")
            intrinsics[index] = read_generic_intrinsics(
                find_intrinsics_file(args.intrinsics_dir, index, row["camera_id"])
            )
    source_dirs = {}
    for row in rows:
        source_dirs[int(row["pose_index"])] = derive_image_dir(
            row,
            args.raw_mount_root,
            args.capture_kind,
            args.capture_time,
        )

    frame_index = args.frame_index
    if frame_index < 0:
        frame_count = min(row["frame_count"] for row in rows if row["frame_count"] > 0)
        frame_index = frame_count // 2

    centers = {}
    for index, pose in poses.items():
        centers[index] = np.linalg.inv(pose)[:3, 3]

    camera_panels = []
    all_projection_rows = []
    for row in rows:
        view_index = int(row["pose_index"])
        view_label = row.get("display_label") or row.get("stage_name") or str(view_index)
        image_path = image_for_frame(source_dirs[view_index], row["camera_id"], frame_index)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]

        view_pose = poses[view_index]
        per_view_rows = []
        for target_row in rows:
            target_index = int(target_row["pose_index"])
            if target_index == view_index:
                continue
            target_label = target_row.get("display_label") or target_row.get("stage_name") or str(target_index)
            point_rig = np.append(centers[target_index], 1.0)
            point_cam = view_pose @ point_rig
            projection = project_point(point_cam, intrinsics[view_index])
            if projection is None:
                status = "behind"
                uv = np.asarray([float("nan"), float("nan")])
                draw_uv = None
            else:
                uv = projection[:2]
                inside = 0 <= uv[0] < width and 0 <= uv[1] < height
                status = "inside" if inside else "outside"
                draw_uv = uv if inside else clip_to_image(uv, width, height)
            result = {
                "view_index": view_index,
                "view_label": view_label,
                "view_group": row.get("group", ""),
                "target_index": target_index,
                "target_label": target_label,
                "target_group": target_row.get("group", ""),
                "u": float(uv[0]) if projection is not None else "",
                "v": float(uv[1]) if projection is not None else "",
                "depth_m": float(point_cam[2]),
                "status": status,
                "draw_u": float(draw_uv[0]) if draw_uv is not None else "",
                "draw_v": float(draw_uv[1]) if draw_uv is not None else "",
                "color_index": target_index,
            }
            per_view_rows.append(result)
            all_projection_rows.append(result)

        out_name = f"camera{view_index:02d}_{row['camera_id']}_origin_projection.jpg"
        mode_images = write_mode_images(
            image,
            image_dir,
            out_name,
            width,
            args.max_image_width,
            per_view_rows,
        )
        camera_panels.append({
            "index": view_index,
            "label": view_label,
            "camera_id": row["camera_id"],
            "image_path": str(image_path),
            "report_image": f"images/{out_name}",
            "report_images": {key: f"images/{value}" for key, value in mode_images.items()},
            "rows": per_view_rows,
            "inside_count": sum(1 for item in per_view_rows if item["status"] == "inside"),
            "outside_count": sum(1 for item in per_view_rows if item["status"] == "outside"),
            "behind_count": sum(1 for item in per_view_rows if item["status"] == "behind"),
        })

    write_projection_tsv(output_dir / "camera_origin_projections.tsv", all_projection_rows)
    write_html(output_dir / "index.html", args, frame_index, rows, camera_panels, all_projection_rows)


def row_matches_mode(row, mode):
    if mode == "all":
        return True
    if mode == "outer":
        return row.get("target_group") == "outer"
    if mode == "inner":
        return row.get("target_group") == "inner"
    raise ValueError(f"unknown projection mode: {mode}")


def write_mode_images(image, image_dir, base_name, width, max_image_width, rows):
    scale = min(1.0, float(max_image_width) / float(width))
    modes = [
        ("all", base_name),
        ("outer", base_name.replace("_origin_projection.jpg", "_origin_projection_outer_targets.jpg")),
        ("inner", base_name.replace("_origin_projection.jpg", "_origin_projection_inner_targets.jpg")),
    ]
    outputs = {}
    for mode, out_name in modes:
        if scale < 1.0:
            canvas = cv2.resize(
                image,
                (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            canvas = image.copy()
        for row in rows:
            if not row_matches_mode(row, mode):
                continue
            if row.get("draw_u", "") == "":
                continue
            color = COLORS[int(row["color_index"]) % len(COLORS)]
            draw_marker(
                canvas,
                np.asarray([float(row["draw_u"]), float(row["draw_v"])]),
                row["target_label"],
                color,
                scale,
                row["status"],
            )
        cv2.imwrite(str(image_dir / out_name), canvas)
        outputs[mode] = out_name
    return outputs


def write_projection_tsv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "view_index",
            "view_label",
            "view_group",
            "target_index",
            "target_label",
            "target_group",
            "u",
            "v",
            "depth_m",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt_float(value):
    if value == "":
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    if abs(value) >= 100000:
        return f"{value:.2e}"
    return f"{value:.1f}"


def write_html(path, args, frame_index, rows, panels, projection_rows):
    css = """
body { margin: 0; font-family: Inter, system-ui, sans-serif; background: #111827; color: #e5e7eb; }
main { max-width: 1720px; margin: 0 auto; padding: 24px; }
h1 { font-size: 24px; margin: 0 0 10px; }
.meta { color: #9ca3af; font-size: 13px; line-height: 1.6; margin-bottom: 18px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 18px; }
.controls button { border: 1px solid #4b5563; background: #1f2937; color: #e5e7eb; border-radius: 6px; padding: 7px 10px; cursor: pointer; }
.controls button.active { background: #2563eb; border-color: #3b82f6; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.card { background: #1f2937; border: 1px solid #374151; border-radius: 8px; overflow: hidden; }
.card h2 { font-size: 16px; margin: 0; padding: 12px 14px; border-bottom: 1px solid #374151; }
.card img { width: 100%; display: block; background: #000; }
.caption { color: #9ca3af; font-size: 12px; padding: 10px 14px; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 5px 7px; border-top: 1px solid #374151; text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
.inside { color: #86efac; }
.outside { color: #fde68a; }
.behind { color: #fca5a5; }
@media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
"""
    cards = []
    status_counts = {
        status: sum(1 for row in projection_rows if row["status"] == status)
        for status in ["inside", "outside", "behind"]
    }
    for panel in panels:
        rows_html = []
        for row in panel["rows"]:
            status = html.escape(row["status"])
            rows_html.append(
                "<tr>"
                f"<td>{row['target_index']}</td>"
                f"<td>{html.escape(row['target_label'])}</td>"
                f"<td>{fmt_float(row['u'])}</td>"
                f"<td>{fmt_float(row['v'])}</td>"
                f"<td>{float(row['depth_m']):.3f}</td>"
                f"<td class=\"{status}\">{status}</td>"
                "</tr>"
            )
        cards.append(
            "<section class=\"card\">"
            f"<h2>cam{panel['index']:02d} · {html.escape(panel['label'])} · "
            f"inside/outside/behind = {panel['inside_count']}/{panel['outside_count']}/{panel['behind_count']}</h2>"
            f"<img class=\"projection-image\" "
            f"src=\"{html.escape(panel['report_images']['all'])}\" "
            f"data-all=\"{html.escape(panel['report_images']['all'])}\" "
            f"data-outer=\"{html.escape(panel['report_images']['outer'])}\" "
            f"data-inner=\"{html.escape(panel['report_images']['inner'])}\" "
            f"alt=\"camera {panel['index']} projection overlay\">"
            f"<div class=\"caption\">source: {html.escape(panel['image_path'])}</div>"
            "<table><thead><tr><th>target</th><th>label</th><th>u</th><th>v</th><th>z m</th><th>status</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
            "</section>"
        )
    body = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(args.title)}</title>"
        f"<style>{css}</style></head><body><main>"
        f"<h1>{html.escape(args.title)}</h1>"
        "<div class=\"meta\">"
        "Projected points are estimated optical centers from the calibration YAML, not visible camera body centers or bracket centers.<br>"
        "<code>inside</code> means in front of the view camera and inside the image bounds only; this diagnostic does not test occlusion by the tower, rig, cables, or camera bodies.<br>"
        f"studio YAML: {html.escape(str(args.studio_yaml))}<br>"
        f"pose YAML: {html.escape(str(args.pose_yaml))}<br>"
        f"intrinsics dir: {html.escape(str(args.intrinsics_dir))}<br>"
        f"manifest: {html.escape(str(args.manifest))}<br>"
        f"capture kind: {html.escape(str(args.capture_kind))}<br>"
        f"frame index: {frame_index}<br>"
        f"view cameras: {len(panels)}; projected target-camera origins: {len(projection_rows)} "
        f"(inside/outside/behind = {status_counts['inside']}/{status_counts['outside']}/{status_counts['behind']})<br>"
        "Default image mode is <code>all 32 targets</code>. Use the buttons below to filter to outer-only or inner-only targets. "
        "The projected points are optical centers only; they are not camera body silhouettes and are not occlusion-aware.<br>"
        f"transform convention: {html.escape(args.transform_convention)}"
        "</div>"
        "<div class=\"controls\" id=\"mode-controls\">"
        "<button type=\"button\" data-mode=\"all\" class=\"active\">All 32 targets</button>"
        "<button type=\"button\" data-mode=\"outer\">Outer targets</button>"
        "<button type=\"button\" data-mode=\"inner\">Inner targets</button>"
        "</div>"
        f"<div class=\"grid\">{''.join(cards)}</div>"
        "<script>"
        "const buttons=[...document.querySelectorAll('#mode-controls button')];"
        "function setMode(mode){"
        "document.querySelectorAll('.projection-image').forEach(img=>{img.src=img.dataset[mode];});"
        "buttons.forEach(btn=>btn.classList.toggle('active', btn.dataset.mode===mode));"
        "}"
        "buttons.forEach(btn=>btn.addEventListener('click',()=>setMode(btn.dataset.mode)));"
        "</script>"
        "</main></body></html>"
    )
    Path(path).write_text(body, encoding="utf-8")


def main():
    build_report(parse_args())


if __name__ == "__main__":
    main()
