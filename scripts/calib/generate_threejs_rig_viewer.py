#!/usr/bin/env python3
"""Generate an interactive Three.js camera rig viewer."""

import argparse
import base64
import csv
import html
import json
import math
import os
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def quat_to_matrix(qx, qy, qz, qw):
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


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


def load_poses(path):
    if yaml is None:
        return load_poses_simple(path)
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pose_count = int(node["pose_count"])
    used = np.zeros(pose_count, dtype=bool)
    poses = [np.eye(4, dtype=np.float64) for _ in range(pose_count)]
    for pose in node.get("poses", []):
        index = int(pose["index"])
        used[index] = True
        poses[index] = pose_to_matrix(pose)
    return used, poses


def load_poses_simple(path):
    pose_count = None
    current = None
    pose_nodes = []

    def flush():
        if current is not None:
            pose_nodes.append(current.copy())

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
        raise RuntimeError(f"Missing pose_count in {path}")
    used = np.zeros(pose_count, dtype=bool)
    poses = [np.eye(4, dtype=np.float64) for _ in range(pose_count)]
    for pose in pose_nodes:
        index = int(pose["index"])
        used[index] = True
        poses[index] = pose_to_matrix(pose)
    return used, poses


def read_metrics(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    metrics = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            index = int(row["camera_index"])
            parsed = {}
            for key, value in row.items():
                if key == "camera_index":
                    continue
                if value in ("True", "False"):
                    parsed[key] = value == "True"
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            metrics[index] = parsed
    return metrics


def parse_optional_float(value):
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def stable_float(value):
    parsed = parse_optional_float(value)
    return parsed


def html_fmt(value, digits=4):
    parsed = parse_optional_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.{digits}f}"


def html_int(value):
    parsed = parse_optional_int(value)
    if parsed is None:
        return "-"
    return f"{parsed:,}"


def relative_asset_url(output_dir, asset_path):
    if not asset_path:
        return ""
    asset = Path(asset_path)
    if not asset.is_absolute():
        candidates = [
            (Path.cwd() / asset),
            asset,
        ]
        for candidate in candidates:
            if candidate.exists():
                asset = candidate
                break
    try:
        rel = os.path.relpath(asset.resolve(), Path(output_dir).resolve())
        return Path(rel).as_posix()
    except OSError:
        return str(asset)


def parse_reprojection_report_spec(spec):
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(spec)
    return path.name, path


def load_reprojection_report(spec, output_dir):
    name, report_dir = parse_reprojection_report_spec(spec)
    metrics_path = report_dir / "camera_metrics.tsv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for raw_row in csv.DictReader(f, delimiter="\t"):
            camera_index = parse_optional_int(raw_row.get("camera_index"))
            if camera_index is None:
                continue
            plot_path = raw_row.get("plot_path", "")
            if plot_path:
                raw_plot = Path(plot_path)
                if not raw_plot.is_absolute() and not raw_plot.exists():
                    raw_plot = report_dir / raw_plot.name
                plot_url = relative_asset_url(output_dir, raw_plot)
            else:
                plot_url = ""

            row = {
                "camera_index": camera_index,
                "camera_label": raw_row.get("camera_label") or raw_row.get("user_id") or raw_row.get("stage_name") or f"camera{camera_index}",
                "stage_name": raw_row.get("stage_name", ""),
                "machine": raw_row.get("machine", ""),
                "user_id": raw_row.get("user_id", ""),
                "usable_views": parse_optional_int(raw_row.get("usable_views")),
                "usable_points": parse_optional_int(raw_row.get("usable_points")),
                "residual_count": parse_optional_int(raw_row.get("residual_count")),
                "skipped_missing_point": parse_optional_int(raw_row.get("skipped_missing_point")),
                "skipped_projection": parse_optional_int(raw_row.get("skipped_projection")),
                "rms": stable_float(raw_row.get("rms")),
                "median_error_px": stable_float(raw_row.get("median_error_px")),
                "mean_error_px": stable_float(raw_row.get("mean_error_px")),
                "p90_error_px": stable_float(raw_row.get("p90_error_px")),
                "max_error_px": stable_float(raw_row.get("max_error_px")),
                "fx": stable_float(raw_row.get("fx")),
                "fy": stable_float(raw_row.get("fy")),
                "cx": stable_float(raw_row.get("cx")),
                "cy": stable_float(raw_row.get("cy")),
                "plot_url": plot_url,
                "plot_path": str(Path(plot_path)) if plot_path else "",
            }
            rows.append(row)

    residual_counts = [row["residual_count"] or 0 for row in rows]
    means = [
        (row["mean_error_px"], row["residual_count"])
        for row in rows
        if row["mean_error_px"] is not None and row["residual_count"]
    ]
    weighted_mean = None
    if means:
        total = sum(count for _value, count in means)
        if total > 0:
            weighted_mean = sum(value * count for value, count in means) / total

    finite_medians = [row["median_error_px"] for row in rows if row["median_error_px"] is not None]
    finite_p90 = [row["p90_error_px"] for row in rows if row["p90_error_px"] is not None]
    finite_max = [row["max_error_px"] for row in rows if row["max_error_px"] is not None]
    worst = None
    if finite_max:
        worst = max(
            (row for row in rows if row["max_error_px"] is not None),
            key=lambda row: row["max_error_px"],
        )

    return {
        "name": name,
        "source_dir": str(report_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "camera_count": len(rows),
        "total_residuals": int(sum(residual_counts)),
        "weighted_mean_error_px": weighted_mean,
        "median_of_camera_medians_px": float(np.median(finite_medians)) if finite_medians else None,
        "max_camera_p90_px": max(finite_p90) if finite_p90 else None,
        "max_error_px": max(finite_max) if finite_max else None,
        "worst_camera_index": worst["camera_index"] if worst else None,
        "worst_camera_label": worst["camera_label"] if worst else "",
        "rows": sorted(rows, key=lambda row: row["camera_index"]),
    }


def load_reprojection_reports(specs, output_dir):
    return [load_reprojection_report(spec, output_dir) for spec in specs]


def render_reprojection_sections(reports):
    if not reports:
        return ""

    summary_cards = []
    for report in reports:
        summary_cards.append(
            "<div class='metric reproj-metric'>"
            f"<strong>{html.escape(html_fmt(report['weighted_mean_error_px'], 4))}</strong>"
            f"<span>{html.escape(report['name'])}<br>weighted mean px</span>"
            "</div>"
        )

    camera_indices = sorted({
        row["camera_index"]
        for report in reports
        for row in report["rows"]
    })
    labels = {}
    for report in reports:
        for row in report["rows"]:
            labels.setdefault(row["camera_index"], row["camera_label"])

    comparison_rows = []
    for camera_index in camera_indices:
        cells = [
            f"<td>cam{camera_index}</td>",
            f"<td>{html.escape(str(labels.get(camera_index, '')))}</td>",
        ]
        for report in reports:
            row = next((item for item in report["rows"] if item["camera_index"] == camera_index), None)
            if row is None:
                cells.append("<td class='missing'>-</td>")
                continue
            cells.append(
                "<td>"
                f"<div><strong>{html.escape(html_fmt(row['median_error_px'], 4))}</strong> med</div>"
                f"<div>{html.escape(html_fmt(row['mean_error_px'], 4))} mean</div>"
                f"<div>{html.escape(html_fmt(row['p90_error_px'], 4))} p90</div>"
                f"<div>{html.escape(html_fmt(row['max_error_px'], 4))} max</div>"
                f"<div>{html.escape(html_int(row['residual_count']))} residuals</div>"
                "</td>"
            )
        comparison_rows.append("<tr>" + "".join(cells) + "</tr>")

    comparison_header = (
        "<tr><th>Camera</th><th>Label</th>"
        + "".join(f"<th>{html.escape(report['name'])}</th>" for report in reports)
        + "</tr>"
    )

    report_sections = []
    for report in reports:
        metric_rows = []
        for row in report["rows"]:
            metric_rows.append(
                "<tr>"
                f"<td>cam{row['camera_index']}</td>"
                f"<td>{html.escape(str(row['camera_label']))}</td>"
                f"<td>{html.escape(str(row['stage_name']))}</td>"
                f"<td>{html.escape(str(row['machine']))}</td>"
                f"<td>{html_int(row['usable_views'])}</td>"
                f"<td>{html_int(row['usable_points'])}</td>"
                f"<td>{html_int(row['residual_count'])}</td>"
                f"<td>{html_int(row['skipped_missing_point'])}</td>"
                f"<td>{html_int(row['skipped_projection'])}</td>"
                f"<td>{html_fmt(row['rms'], 4)}</td>"
                f"<td>{html_fmt(row['median_error_px'], 4)}</td>"
                f"<td>{html_fmt(row['mean_error_px'], 4)}</td>"
                f"<td>{html_fmt(row['p90_error_px'], 4)}</td>"
                f"<td>{html_fmt(row['max_error_px'], 4)}</td>"
                f"<td>{html_fmt(row['fx'], 2)}</td>"
                f"<td>{html_fmt(row['fy'], 2)}</td>"
                f"<td>{html_fmt(row['cx'], 2)}</td>"
                f"<td>{html_fmt(row['cy'], 2)}</td>"
                "</tr>"
            )

        plot_cards = []
        for row in report["rows"]:
            if row["plot_url"]:
                media = (
                    f"<a href='{html.escape(row['plot_url'])}'>"
                    f"<img class='reproj-plot' src='{html.escape(row['plot_url'])}' loading='lazy' "
                    f"alt='{html.escape(report['name'])} cam{row['camera_index']} reprojection plot'>"
                    "</a>"
                )
            else:
                media = "<div class='plot-missing'>missing plot</div>"
            plot_cards.append(
                "<article class='plot-card'>"
                f"<h4>cam{row['camera_index']} <span>{html.escape(str(row['camera_label']))}</span></h4>"
                "<div class='plot-stats'>"
                f"<span>median {html_fmt(row['median_error_px'], 4)} px</span>"
                f"<span>p90 {html_fmt(row['p90_error_px'], 4)} px</span>"
                f"<span>max {html_fmt(row['max_error_px'], 4)} px</span>"
                f"<span>{html_int(row['residual_count'])} residuals</span>"
                "</div>"
                f"{media}"
                "</article>"
            )

        report_sections.append(
            "<section class='reproj-stage'>"
            f"<h3>{html.escape(report['name'])}</h3>"
            "<div class='stage-summary'>"
            f"<span>source: <code>{html.escape(report['source_dir'])}</code></span>"
            f"<span>cameras: {report['camera_count']}</span>"
            f"<span>residuals: {html_int(report['total_residuals'])}</span>"
            f"<span>weighted mean: {html_fmt(report['weighted_mean_error_px'], 4)} px</span>"
            f"<span>median(camera medians): {html_fmt(report['median_of_camera_medians_px'], 4)} px</span>"
            f"<span>worst: cam{html.escape(str(report['worst_camera_index']))} / {html_fmt(report['max_error_px'], 4)} px</span>"
            "</div>"
            "<div class='table-wrap reproj-table-wrap'><table class='reproj-table'>"
            "<thead><tr><th>Cam</th><th>Label</th><th>Stage</th><th>Machine</th><th>Views</th><th>Points</th>"
            "<th>Residuals</th><th>Missing 3D</th><th>Projection skip</th><th>RMS</th>"
            "<th>Median</th><th>Mean</th><th>P90</th><th>Max</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th></tr></thead>"
            f"<tbody>{''.join(metric_rows)}</tbody></table></div>"
            f"<div class='plot-grid'>{''.join(plot_cards)}</div>"
            "</section>"
        )

    return (
        "<h2 class='section-title'>Reprojection Error Report</h2>"
        "<section class='reprojection-block'>"
        f"<div class='summary-grid reproj-summary'>{''.join(summary_cards)}</div>"
        "<h3>Cross-Stage Per-Camera Comparison</h3>"
        "<div class='table-wrap reproj-table-wrap'><table class='reproj-compare'>"
        f"<thead>{comparison_header}</thead><tbody>{''.join(comparison_rows)}</tbody></table></div>"
        f"{''.join(report_sections)}"
        "</section>"
    )


def render_board_orientation_section(rig_data):
    options = rig_data.get("viewer_options", {}) or {}
    alignment = options.get("board_orientation_alignment") or {}
    sources = alignment.get("sources") or {}
    if not sources:
        return ""
    rows = []
    for name, stats in sources.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html_int(stats.get('sample_count'))}</td>"
            f"<td>{html_fmt(stats.get('median_angle_from_horizontal_deg'), 3)}</td>"
            f"<td>{html_fmt(stats.get('p90_angle_from_horizontal_deg'), 3)}</td>"
            f"<td>{html_fmt(stats.get('max_angle_from_horizontal_deg'), 3)}</td>"
            f"<td>{html.escape(str(stats.get('description') or ''))}</td>"
            "</tr>"
        )
    aggregate = alignment.get("aggregate") or {}
    gravity = alignment.get("gravity_display_up_vector")
    gravity_text = ", ".join(html_fmt(value, 4) for value in gravity) if gravity else "-"
    return (
        "<h2 class='section-title'>Board Orientation / Gravity</h2>"
        "<section class='orientation-block'>"
        "<div class='stage-summary'>"
        f"<span>method: {html.escape(str(alignment.get('method') or '-'))}</span>"
        f"<span>display up: {html.escape(gravity_text)}</span>"
        f"<span>all normals p90 from horizontal: {html_fmt(aggregate.get('p90_angle_from_horizontal_deg'), 3)} deg</span>"
        f"<span>source: <code>{html.escape(str(alignment.get('gravity_source') or ''))}</code></span>"
        "</div>"
        "<div class='table-wrap reproj-table-wrap'><table class='orientation-table'>"
        "<thead><tr><th>Board set</th><th>Normals</th><th>Median horizontal err deg</th>"
        "<th>P90 horizontal err deg</th><th>Max horizontal err deg</th><th>Definition</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</section>"
    )


def camera_center_in_reference(camera_tr_reference):
    return np.linalg.inv(camera_tr_reference)[:3, 3]


def to_three(point):
    point = np.asarray(point, dtype=np.float64)
    return [float(point[0]), float(-point[1]), float(-point[2])]


def vector_to_three(vector):
    vector = np.asarray(vector, dtype=np.float64)
    mapped = np.array([vector[0], -vector[1], -vector[2]], dtype=np.float64)
    norm = np.linalg.norm(mapped)
    if norm > 0:
        mapped /= norm
    return [float(mapped[0]), float(mapped[1]), float(mapped[2])]


def line_to_three(a, b):
    return [to_three(a), to_three(b)]


def build_camera_geometry(camera_tr_reference, frustum_depth, frustum_half_width, frustum_half_height, axis_length):
    reference_tr_camera = np.linalg.inv(camera_tr_reference)
    center = reference_tr_camera[:3, 3]
    rotation = reference_tr_camera[:3, :3]

    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    z_axis = rotation[:, 2]

    corners = [
        center + frustum_depth * z_axis + frustum_half_width * x_axis + frustum_half_height * y_axis,
        center + frustum_depth * z_axis - frustum_half_width * x_axis + frustum_half_height * y_axis,
        center + frustum_depth * z_axis - frustum_half_width * x_axis - frustum_half_height * y_axis,
        center + frustum_depth * z_axis + frustum_half_width * x_axis - frustum_half_height * y_axis,
    ]

    frustum_lines = [
        line_to_three(center, corners[0]),
        line_to_three(center, corners[1]),
        line_to_three(center, corners[2]),
        line_to_three(center, corners[3]),
        line_to_three(corners[0], corners[1]),
        line_to_three(corners[1], corners[2]),
        line_to_three(corners[2], corners[3]),
        line_to_three(corners[3], corners[0]),
    ]

    axes = {
        "x": line_to_three(center, center + axis_length * x_axis),
        "y": line_to_three(center, center + axis_length * y_axis),
        "z": line_to_three(center, center + axis_length * z_axis),
    }

    return {
        "center": to_three(center),
        "basis": {
            "x": vector_to_three(x_axis),
            "y": vector_to_three(y_axis),
            "z": vector_to_three(z_axis),
        },
        "frustum_lines": frustum_lines,
        "axes": axes,
    }


def compute_bounds(points):
    if not points:
        return {"center": [0.0, 0.0, 0.0], "radius": 1.0}
    arr = np.asarray(points, dtype=np.float64)
    center = arr.mean(axis=0)
    radius = float(np.max(np.linalg.norm(arr - center, axis=1)))
    return {
        "center": [float(v) for v in center],
        "radius": max(0.6, radius * 1.25),
    }


def find_camera_image(camera_image_dir, camera_index):
    if not camera_image_dir:
        return None
    image_dir = Path(camera_image_dir)
    patterns = [
        f"cam{camera_index:02d}_*.jpg",
        f"cam{camera_index:02d}_*.jpeg",
        f"cam{camera_index:02d}_*.png",
        f"cam{camera_index}_*.jpg",
        f"camera{camera_index:02d}_*.jpg",
        f"camera{camera_index}_*.jpg",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(sorted(image_dir.glob(pattern)))
    if not matches:
        return None
    return matches[0].resolve()


def camera_image_url(output_dir, image_path):
    if not image_path:
        return ""
    output_root = Path(output_dir).resolve()
    try:
        return image_path.relative_to(output_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Camera image {image_path} must be inside output directory {output_root}"
        ) from exc


def camera_texture_data_url(image_path, max_width, quality):
    if not image_path:
        return ""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return ""

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max_width > 0:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image.thumbnail((max_width, max_width), resampling)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def load_sparse_point_cloud(path):
    if not path:
        return {
            "source": "",
            "coordinate_frame": "camera0_opencv",
            "point_count": 0,
            "positions": [],
            "colors": [],
        }
    point_path = Path(path)
    if not point_path.exists():
        raise FileNotFoundError(point_path)
    payload = json.loads(point_path.read_text(encoding="utf-8"))
    positions = []
    colors = []
    for point in payload.get("points", []):
        positions.extend(to_three(point["xyz"]))
        rgb = point.get("rgb", [255, 255, 255])
        colors.extend([max(0.0, min(1.0, float(v) / 255.0)) for v in rgb[:3]])
    return {
        "source": str(point_path.resolve()),
        "coordinate_frame": payload.get("coordinate_frame", "camera0_opencv"),
        "frame": payload.get("frame"),
        "point_count": len(payload.get("points", [])),
        "positions": positions,
        "colors": colors,
    }


def build_rig_data(pose_yaml, metrics_tsv, args, output_dir):
    used, poses = load_poses(pose_yaml)
    metrics = read_metrics(metrics_tsv)
    cameras = []
    bound_points = []

    for index, pose in enumerate(poses):
        if args.used_only and not used[index]:
            continue
        geometry = build_camera_geometry(
            pose,
            args.frustum_depth,
            args.frustum_half_width,
            args.frustum_half_height,
            args.axis_length,
        )
        image_path = find_camera_image(args.camera_image_dir, index)
        camera = {
            "index": index,
            "used": bool(used[index]),
            "center": geometry["center"],
            "basis": geometry["basis"],
            "frustum_lines": geometry["frustum_lines"],
            "axes": geometry["axes"],
            "metrics": metrics.get(index, {}),
            "image_url": camera_image_url(output_dir, image_path),
            "image_texture_url": camera_texture_data_url(
                image_path,
                args.camera_image_texture_max_width,
                args.camera_image_texture_quality,
            ),
        }
        cameras.append(camera)
        bound_points.append(geometry["center"])
        for line in geometry["frustum_lines"]:
            bound_points.append(line[0])
            bound_points.append(line[1])

    sparse_point_cloud = load_sparse_point_cloud(args.sparse_point_cloud_json)
    reprojection_reports = load_reprojection_reports(args.reprojection_report, output_dir)
    return {
        "title": args.title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_pose_yaml": str(Path(pose_yaml).resolve()),
        "source_metrics_tsv": str(Path(metrics_tsv).resolve()) if metrics_tsv else "",
        "camera_image_dir": str(Path(args.camera_image_dir).resolve()) if args.camera_image_dir else "",
        "coordinate_note": "Scene coordinates map camera0 OpenCV coordinates as x -> x, y -> -y, z -> -z so OpenCV/COLMAP +Z forward becomes CG/Three -Z forward.",
        "frustum": {
            "default_near": args.default_near,
            "default_far": args.default_far,
            "half_width_over_depth": args.frustum_half_width / args.frustum_depth,
            "half_height_over_depth": args.frustum_half_height / args.frustum_depth,
            "fill_opacity": args.frustum_fill_alpha,
        },
        "sparse_point_cloud": sparse_point_cloud,
        "reprojection_reports": reprojection_reports,
        "cameras": cameras,
        "bounds": compute_bounds(bound_points),
    }


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive 3D Camera Rig</title>
  <style>
    :root {
      --bg: #141414;
      --panel: #f4f4f4;
      --panel-2: #e7e7e7;
      --ink: #202124;
      --muted: #5f6368;
      --line: #c9c9c9;
      --focus: #0b57d0;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f7;
      color: var(--ink);
    }
    header {
      padding: 24px 32px 16px;
      background: #202124;
      color: #ffffff;
    }
    header h1 {
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.25;
      font-weight: 680;
    }
    header p {
      margin: 4px 0;
      color: #c7cbd1;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    main {
      padding: 22px 32px 36px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric {
      background: #ffffff;
      border: 1px solid var(--line);
      padding: 12px 14px;
    }
    .metric strong {
      display: block;
      font-size: 20px;
      line-height: 1.2;
      margin-bottom: 3px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    .reproj-metric strong {
      font-size: 18px;
    }
    .section-title {
      margin: 24px 0 10px;
      font-size: 18px;
    }
    .sim-stage {
      position: relative;
      height: min(74vh, 760px);
      min-height: 540px;
      border: 1px solid #262626;
      background: var(--bg);
      overflow: hidden;
    }
    #viewport {
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 30% 20%, #252525 0, #141414 58%);
    }
    .webgl-error {
      position: absolute;
      left: 24px;
      right: 380px;
      top: 24px;
      max-width: 620px;
      padding: 14px 16px;
      background: #fff4ef;
      border: 1px solid #e5a088;
      color: #422318;
      font-size: 13px;
      line-height: 1.45;
      z-index: 3;
    }
    #panel {
      position: absolute;
      top: 12px;
      right: 12px;
      width: 340px;
      max-height: calc(100% - 24px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 18px 52px rgba(0, 0, 0, 0.24);
      padding: 14px;
    }
    #panel h2 {
      margin: 0 0 5px;
      font-size: 18px;
      line-height: 1.25;
      font-weight: 680;
    }
    .sub {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .button-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 7px;
      margin: 10px 0 12px;
    }
    .control-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 0 0 12px;
    }
    .control-grid label,
    .range-row label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .control-grid select {
      width: 100%;
      min-height: 31px;
      border: 1px solid #b8b8b8;
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      padding: 0 7px;
      font-size: 12px;
    }
    .range-row {
      display: grid;
      gap: 7px;
      margin: 0 0 12px;
    }
    .range-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--ink);
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--focus);
    }
    button {
      border: 1px solid #b8b8b8;
      background: #ffffff;
      color: var(--ink);
      min-height: 32px;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
    }
    button:hover { border-color: #70757a; }
    button.active {
      background: #dfe9ff;
      border-color: var(--focus);
      color: #174ea6;
    }
    .segmented {
      display: grid;
      gap: 6px;
      margin: 0 0 10px;
    }
    .segmented-title {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
      text-transform: uppercase;
    }
    .segmented-buttons {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 5px;
    }
    .segmented-buttons button {
      min-width: 0;
      padding: 0 6px;
    }
    .control-section {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }
    .control-section:first-of-type {
      border-top: 0;
      padding-top: 0;
      margin-top: 0;
    }
    .control-section-title {
      color: var(--ink);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 7px;
    }
    .button-row.view-row {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .legend {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
      margin: 6px 0 13px;
      font-size: 12px;
    }
    .legend span {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      min-width: 0;
    }
    .swatch {
      width: 13px;
      height: 3px;
      display: inline-block;
    }
    .correspondence-controls {
      display: none;
      gap: 8px;
      margin-top: 8px;
    }
    .correspondence-controls.active {
      display: grid;
    }
    .correspondence-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .correspondence-controls label {
      display: grid;
      gap: 4px;
      font-size: 11px;
      color: var(--muted);
      min-width: 0;
    }
    .correspondence-controls select,
    .correspondence-controls input {
      width: 100%;
      min-width: 0;
      box-sizing: border-box;
    }
    .correspondence-controls select {
      height: 28px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      font-size: 12px;
    }
    .correspondence-controls .checkbox-row {
      display: inline-flex;
      grid-template-columns: none;
      align-items: center;
      gap: 7px;
      color: var(--ink);
    }
    .correspondence-controls .checkbox-row input {
      width: 14px;
      height: 14px;
      margin: 0;
    }
    .correspondence-value {
      color: var(--ink);
      font-variant-numeric: tabular-nums;
    }
    #selected {
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 10px 0;
      margin-bottom: 12px;
      font-size: 12px;
      line-height: 1.55;
    }
    #selected strong { font-size: 13px; }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      background: #ffffff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      min-width: 760px;
    }
    th, td {
      padding: 7px 8px;
      border-bottom: 1px solid #dddddd;
      text-align: right;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: var(--panel-2);
      font-weight: 650;
    }
    td:first-child, th:first-child { text-align: left; }
    tr { cursor: pointer; }
    tr:hover { background: #f0f5ff; }
    tr.selected { background: #dfe9ff; }
    tr.coverage-inactive {
      color: var(--muted);
      background: #f4f4f1;
    }
    .reprojection-block {
      display: grid;
      gap: 18px;
      margin-bottom: 28px;
    }
    .reprojection-block h3 {
      margin: 4px 0 0;
      font-size: 16px;
    }
    .reproj-summary {
      margin-bottom: 0;
    }
    .stage-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .stage-summary span {
      background: #ffffff;
      border: 1px solid var(--line);
      padding: 5px 7px;
      overflow-wrap: anywhere;
    }
    .stage-summary code {
      background: transparent;
      padding: 0;
    }
    .reproj-stage {
      display: grid;
      gap: 10px;
    }
    .reproj-table-wrap {
      margin-bottom: 4px;
    }
    .reproj-table,
    .reproj-compare {
      min-width: 980px;
    }
    .reproj-table td,
    .reproj-table th,
    .reproj-compare td,
    .reproj-compare th {
      vertical-align: top;
      font-size: 11px;
    }
    .reproj-compare td div {
      line-height: 1.45;
      color: var(--muted);
    }
    .reproj-compare td strong {
      color: var(--ink);
    }
    .missing,
    .plot-missing {
      color: var(--muted);
    }
    .plot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      align-items: start;
    }
    .plot-card {
      background: #ffffff;
      border: 1px solid var(--line);
      padding: 10px;
    }
    .plot-card h4 {
      margin: 0 0 7px;
      font-size: 13px;
      line-height: 1.25;
    }
    .plot-card h4 span {
      color: var(--muted);
      font-weight: 500;
    }
    .plot-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 4px 8px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
    }
    .reproj-plot {
      display: block;
      width: 100%;
      max-height: 320px;
      object-fit: contain;
      border: 1px solid #dddddd;
      background: #111111;
    }
    canvas { display: block; outline: none; }
    @media (max-width: 760px) {
      header { padding: 18px 18px 12px; }
      main { padding: 16px 12px 24px; }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .sim-stage { height: 860px; min-height: 860px; }
      #panel {
        left: 10px;
        right: 10px;
        top: auto;
        bottom: 10px;
        width: auto;
        max-height: 44%;
        padding: 11px;
      }
      #panel h2 { font-size: 16px; }
      .button-row { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      button { min-height: 30px; }
    }
  </style>
</head>
<body>
  <header>
    <h1 id="report-title">Inner 8-Camera Rig Report</h1>
    <p id="source"></p>
  </header>
  <main>
    <section class="summary-grid">
      <div class="metric"><strong id="metric-camera-count">-</strong><span>visible / total cameras</span></div>
      <div class="metric"><strong id="metric-near-far">0.30-0.70 m</strong><span>default frustum range</span></div>
      <div class="metric"><strong id="metric-max-delta">-</strong><span>max small refine delta</span></div>
      <div class="metric"><strong id="metric-sparse-points">0</strong><span>COLMAP sparse points</span></div>
      <div class="metric"><strong id="metric-board-normal">-</strong><span>board normal p90 horizontal err</span></div>
      <div class="metric"><strong id="metric-overlap">off</strong><span>strict all-camera overlap</span></div>
    </section>

    <h2 class="section-title">3D Rig Sim</h2>
    <section class="sim-stage">
      <div id="viewport"></div>
      <aside id="panel">
        <h2>3D Rig Controls</h2>
        <div class="control-section">
          <div class="control-section-title">Rig Scope</div>
          <div class="segmented">
            <div class="segmented-title">Camera Set</div>
            <div class="segmented-buttons">
              <button id="scope-all" class="active">All</button>
              <button id="scope-inner">Inner</button>
              <button id="scope-outer">Outer</button>
            </div>
          </div>
          <div class="segmented">
            <div class="segmented-title">Dataset Coverage</div>
            <div class="segmented-buttons">
              <button id="coverage-whole" class="active">Whole</button>
              <button id="coverage-large-marker">Large</button>
              <button id="coverage-small-marker">Small</button>
            </div>
          </div>
        </div>
        <div class="control-section">
          <div class="control-section-title">View</div>
          <div class="button-row view-row">
            <button id="top">Top</button>
            <button id="front">Front</button>
            <button id="toggle-gizmo" class="active">Gizmo</button>
          </div>
        </div>
        <div class="control-section">
          <div class="control-section-title">Layers</div>
          <div class="button-row">
            <button id="toggle-frustum" class="active">Frustums</button>
            <button id="toggle-axes" class="active">Axes</button>
            <button id="toggle-labels" class="active">Labels</button>
            <button id="toggle-grid" class="active">Grid</button>
            <button id="toggle-images" class="active">Images</button>
            <button id="toggle-points" class="active">Points</button>
            <button id="toggle-outer-topdown" class="active">Topdown</button>
            <button id="toggle-outer-colmap" class="active">Rough</button>
            <button id="toggle-overlap">Overlap</button>
          </div>
        </div>
        <div class="control-section" id="correspondence-section">
          <div class="control-section-title">Correspondence</div>
          <div class="button-row">
            <button id="load-correspondence">Load Corr</button>
          </div>
          <div class="correspondence-controls" id="correspondence-controls">
            <label class="checkbox-row">
              <input id="correspondence-all-frames" type="checkbox">
              <span>All frames</span>
            </label>
            <label>Frame <span class="correspondence-value" id="correspondence-frame-value"></span>
              <input id="correspondence-frame-slider" type="range" min="0" max="0" step="1" value="0">
            </label>
            <label>Point group
              <select id="correspondence-point-group"></select>
            </label>
            <div class="correspondence-grid">
              <label>Max shown <span class="correspondence-value" id="correspondence-max-value"></span>
                <input id="correspondence-max" type="range" min="100" max="30000" step="100" value="8000">
              </label>
              <label>Residual <= <span class="correspondence-value" id="correspondence-residual-max-value"></span>
                <input id="correspondence-residual-max" type="range" min="1" max="200" step="1" value="200">
              </label>
            </div>
          </div>
          <p class="sub" id="correspondence-status"></p>
        </div>
        <div class="control-section">
          <div class="control-section-title">Frustum Range</div>
          <div class="range-row">
            <label><span class="range-head"><span>Near</span><span id="near-value"></span></span>
              <input id="near-slider" type="range" min="0.05" max="2.00" step="0.01">
            </label>
            <label><span class="range-head"><span>Far</span><span id="far-value"></span></span>
              <input id="far-slider" type="range" min="0.10" max="3.00" step="0.01">
            </label>
          </div>
        </div>
        <p class="sub" id="overlap-status"></p>
        <div class="legend">
          <span><i class="swatch" style="background:#e53935"></i>x</span>
          <span><i class="swatch" style="background:#1e8e3e"></i>y</span>
          <span><i class="swatch" style="background:#1a73e8"></i>z</span>
          <span><i class="swatch" style="background:#ffffff"></i>points</span>
        </div>
        <div id="selected"></div>
      </aside>
    </section>

    __REPROJECTION_HTML__
    __BOARD_ORIENTATION_HTML__

    <h2 class="section-title">Camera Intrinsics / Final Residuals</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Cam</th><th>Set</th><th>Coverage</th><th>Raw obs</th><th>Solve obs</th><th>Median px</th><th>P90 px</th><th>Max px</th><th>fx</th><th>fy</th><th>Decision</th></tr>
        </thead>
        <tbody id="camera-table"></tbody>
      </table>
    </div>
  </main>
  <script src="./three.min.js"></script>
  <script src="./OrbitControls.js"></script>
  <script src="./TransformControls.js"></script>
  <script>
const RIG_DATA = __RIG_DATA__;

const viewport = document.getElementById("viewport");
const renderCanvas = document.createElement("canvas");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({
    canvas: renderCanvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
} catch (error) {
  viewport.innerHTML = '<div class="webgl-error"><strong>WebGL renderer failed.</strong><br>Try enabling browser hardware acceleration, or reload after closing other GPU-heavy tabs.</div>';
  throw error;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x141414, 1);
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
let controls = null;
function configuredWorldUp() {
  const options = RIG_DATA.viewer_options || {};
  const values = options.world_up_three;
  if (Array.isArray(values) && values.length === 3) {
    const up = new THREE.Vector3(values[0], values[1], values[2]);
    if (up.lengthSq() > 0) {
      return up.normalize();
    }
  }
  return new THREE.Vector3(0, 1, 0);
}
const WORLD_UP = configuredWorldUp();

// The Three.js scene frame is the fixed display world.
// Calibrated geometry stays in CAM0 gauge below this root, and this root stores
// the explicit relative rotation R_world_cam0.
const cam0ToWorldRoot = new THREE.Group();
scene.add(cam0ToWorldRoot);

const frustumGroup = new THREE.Group();
const imagePlaneGroup = new THREE.Group();
const pointCloudGroup = new THREE.Group();
const axisGroup = new THREE.Group();
const labelGroup = new THREE.Group();
const markerGroup = new THREE.Group();
const correspondenceGroup = new THREE.Group();
const overlapGroup = new THREE.Group();
const pivotFrameGroup = new THREE.Group();
cam0ToWorldRoot.add(frustumGroup);
cam0ToWorldRoot.add(imagePlaneGroup);
cam0ToWorldRoot.add(pointCloudGroup);
cam0ToWorldRoot.add(axisGroup);
cam0ToWorldRoot.add(labelGroup);
cam0ToWorldRoot.add(markerGroup);
cam0ToWorldRoot.add(correspondenceGroup);
cam0ToWorldRoot.add(overlapGroup);
overlapGroup.visible = false;
correspondenceGroup.visible = false;
scene.add(pivotFrameGroup);

const bounds = RIG_DATA.bounds;
const radius = Math.max(0.7, bounds.radius);
scene.fog = new THREE.Fog(
  0x141414,
  Math.max(10.0, radius * 4.0),
  Math.max(24.0, radius * 12.0)
);
const grid = new THREE.GridHelper(radius * 3.2, 16, 0x9a9a9a, 0x404040);
grid.material.transparent = true;
grid.material.opacity = 0.45;
scene.add(grid);

const originAxes = new THREE.AxesHelper(radius * 0.32);
scene.add(originAxes);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const pickables = [];
const cameraMeshes = new Map();
const frustumObjects = new Map();
const cameraSceneObjects = new Map();
const tableRows = new Map();
let selectedIndex = RIG_DATA.cameras.length ? RIG_DATA.cameras[0].index : -1;
let nearDistance = RIG_DATA.frustum.default_near;
let farDistance = RIG_DATA.frustum.default_far;
const cameraVisibility = Object.assign(
  {inner: true, outer: true, outer_topdown: true, outer_colmap: true},
  (RIG_DATA.viewer_options || {}).default_visibility || {}
);
let cameraScope = "all";
let coverageMode = ((RIG_DATA.dataset_coverage || {}).default_mode) || "whole";
let currentFrameMode = "iso";
let worldFromCam0Quat = initialWorldFromReferenceQuaternion();
let transformControl = null;
let worldGizmoPointerActive = false;
let worldGizmoDragging = false;
let correspondenceData = null;
let correspondenceLoaded = false;
let correspondenceVisible = false;
let correspondenceSelectedFrameByDataset = {};
let correspondenceAllFramesByDataset = {};
let correspondenceSelectedPointGroupByDataset = {};
let correspondenceMaxShown = 8000;
let correspondenceResidualMax = 200;
const worldGizmoDragAnchor = new THREE.Vector3();

function createOrbitControls(target) {
  const nextControls = new THREE.OrbitControls(camera, renderer.domElement);
  nextControls.enableDamping = true;
  nextControls.dampingFactor = 0.08;
  nextControls.screenSpacePanning = true;
  if (target) {
    nextControls.target.copy(target);
  }
  nextControls.update();
  return nextControls;
}

function rebuildOrbitControlsForCurrentUp(enabled, preferredUp) {
  const position = camera.position.clone();
  const target = controls ? controls.target.clone() : new THREE.Vector3();
  if (controls) {
    controls.dispose();
  }
  const nextUp = preferredUp ? preferredUp.clone().normalize() : WORLD_UP.clone();
  camera.up.copy(nextUp);
  // This OrbitControls build captures camera.up when constructed.
  controls = createOrbitControls(target);
  controls.enabled = enabled;
  camera.position.copy(position);
  controls.target.copy(target);
  camera.up.copy(nextUp);
  camera.lookAt(controls.target);
}

function fmt(value, digits) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function cameraLabel(cam) {
  return cam.label || ("cam" + cam.index);
}

function cameraCategory(cam) {
  const kind = String(cam.kind || "").toLowerCase();
  const label = cameraLabel(cam).toLowerCase();
  if (kind.startsWith("inner") || label.startsWith("inner")) return "inner";
  if (kind.startsWith("outer_topdown")) return "outer_topdown";
  if (kind.startsWith("outer_colmap")) return "outer_colmap";
  if (kind.startsWith("outer") || /^\\d+-\\d+/.test(label)) return "outer";
  return "other";
}

function cameraIsVisible(cam) {
  const category = cameraCategory(cam);
  if (cameraScope === "inner" && category !== "inner") return false;
  if (cameraScope === "outer" && !category.startsWith("outer")) return false;
  if (category === "inner") return cameraVisibility.inner;
  if (category.startsWith("outer")) return cameraVisibility.outer && cameraVisibility[category] !== false;
  return true;
}

function coverageForCamera(cam) {
  const coverage = cam.coverage || {};
  return coverage[coverageMode] || {active: true, status: "unknown", quality: "unknown", observation_count: null, detail: "No dataset coverage metadata in this viewer."};
}

function calibrationQuality(cam) {
  return cam.calibration_quality || {
    source: "missing",
    decision: "missing",
    observation_count: null,
    median_error_px: null,
    p90_error_px: null,
    max_error_px: null,
    fx: null,
    fy: null,
  };
}

function displayResidualQuality(cam) {
  const q = calibrationQuality(cam);
  const c = coverageForCamera(cam);
  if ((coverageMode === "large_marker" || coverageMode === "small_marker") && c.active !== false) {
    return {
      source: coverageMode + "_pnp",
      decision: c.status || c.quality || coverageMode,
      observation_count: c.total_inliers ?? c.observation_count ?? null,
      median_error_px: c.median_view_error_px ?? null,
      p90_error_px: null,
      max_error_px: null,
    };
  }
  return q;
}

function cameraCoverageActive(cam) {
  return coverageForCamera(cam).active !== false;
}

function cameraCategoryExists(category) {
  if (category === "outer") {
    return RIG_DATA.cameras.some((cam) => cameraCategory(cam).startsWith("outer"));
  }
  return RIG_DATA.cameras.some((cam) => cameraCategory(cam) === category);
}

function viewerOption(name, fallback) {
  const options = RIG_DATA.viewer_options || {};
  if (Object.prototype.hasOwnProperty.call(options, name)) {
    return options[name];
  }
  return fallback;
}

function flattenLines(lines) {
  const values = [];
  lines.forEach((line) => {
    values.push(line[0][0], line[0][1], line[0][2], line[1][0], line[1][1], line[1][2]);
  });
  return values;
}

function makeLineSegments(lines, color, opacity) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(flattenLines(lines), 3));
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
  });
  return new THREE.LineSegments(geometry, material);
}

function v3(values) {
  return new THREE.Vector3(values[0], values[1], values[2]);
}

function initialWorldFromReferenceQuaternion() {
  const options = RIG_DATA.viewer_options || {};
  const quat = options.default_world_from_reference_quaternion_xyzw;
  if (Array.isArray(quat) && quat.length === 4) {
    return new THREE.Quaternion(quat[0], quat[1], quat[2], quat[3]).normalize();
  }
  const up = options.default_reference_up_vector_three;
  if (Array.isArray(up) && up.length === 3) {
    const from = v3(up);
    if (from.lengthSq() > 0) {
      return new THREE.Quaternion().setFromUnitVectors(from.normalize(), WORLD_UP.clone());
    }
  }
  return new THREE.Quaternion();
}

function vecToArray(vector) {
  return [vector.x, vector.y, vector.z];
}

function addScaled(base, vectors) {
  const out = base.clone();
  vectors.forEach(([vector, scale]) => out.addScaledVector(vector, scale));
  return out;
}

function makeGeometryFromLines(lines) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(flattenLines(lines), 3));
  return geometry;
}

function calcFrustum(cam, nearValue, farValue) {
  const center = v3(cam.center);
  const xAxis = v3(cam.basis.x).normalize();
  const yAxis = v3(cam.basis.y).normalize();
  const zAxis = v3(cam.basis.z).normalize();
  const sx = RIG_DATA.frustum.half_width_over_depth;
  const sy = RIG_DATA.frustum.half_height_over_depth;

  function planeCorners(depth) {
    const halfW = sx * depth;
    const halfH = sy * depth;
    return [
      addScaled(center, [[zAxis, depth], [xAxis, halfW], [yAxis, halfH]]),
      addScaled(center, [[zAxis, depth], [xAxis, -halfW], [yAxis, halfH]]),
      addScaled(center, [[zAxis, depth], [xAxis, -halfW], [yAxis, -halfH]]),
      addScaled(center, [[zAxis, depth], [xAxis, halfW], [yAxis, -halfH]]),
    ];
  }

  const nearCorners = planeCorners(nearValue);
  const farCorners = planeCorners(farValue);
  const lines = [];
  for (let i = 0; i < 4; ++i) {
    const next = (i + 1) % 4;
    lines.push([vecToArray(nearCorners[i]), vecToArray(nearCorners[next])]);
    lines.push([vecToArray(farCorners[i]), vecToArray(farCorners[next])]);
    lines.push([vecToArray(nearCorners[i]), vecToArray(farCorners[i])]);
  }

  return {
    center,
    insidePoint: addScaled(center, [[zAxis, (nearValue + farValue) * 0.5]]),
    nearCorners,
    farCorners,
    lines,
  };
}

function makeFrustumMeshGeometry(frustum) {
  const vertices = frustum.nearCorners.concat(frustum.farCorners);
  const positions = [];
  const indices = [
    [0, 1, 2], [0, 2, 3],
    [4, 6, 5], [4, 7, 6],
    [0, 4, 5], [0, 5, 1],
    [1, 5, 6], [1, 6, 2],
    [2, 6, 7], [2, 7, 3],
    [3, 7, 4], [3, 4, 0],
  ];
  indices.forEach((tri) => {
    tri.forEach((idx) => {
      const p = vertices[idx];
      positions.push(p.x, p.y, p.z);
    });
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function makeImagePlaneGeometry(cam, frustum) {
  const zAxis = v3(cam.basis.z).normalize();
  const corners = frustum.farCorners.map((p) => p.clone().addScaledVector(zAxis, 0.004));
  const imageCorners = [
    corners[2], corners[1], corners[0],
    corners[2], corners[0], corners[3],
  ];
  const positions = [];
  imageCorners.forEach((p) => positions.push(p.x, p.y, p.z));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute([
    0, 1,
    0, 0,
    1, 0,
    0, 1,
    1, 0,
    1, 1,
  ], 2));
  geometry.computeVertexNormals();
  return geometry;
}

function loadLocalTexture(url, onLoad, onError) {
  const image = new Image();
  image.onload = () => {
    const texture = new THREE.Texture(image);
    texture.needsUpdate = true;
    onLoad(texture);
  };
  image.onerror = onError;
  image.src = url;
}

function imagePlaneFacesCamera(item) {
  if (!item.imagePlane || !item.imagePlane.geometry || !item.imagePlane.geometry.attributes.position) {
    return false;
  }
  const pos = item.imagePlane.geometry.attributes.position;
  if (pos.count < 3) {
    return false;
  }
  const a = new THREE.Vector3().fromBufferAttribute(pos, 0);
  const b = new THREE.Vector3().fromBufferAttribute(pos, 1);
  const c = new THREE.Vector3().fromBufferAttribute(pos, 2);
  const normal = b.clone().sub(a).cross(c.clone().sub(a)).normalize();
  return normal.dot(v3(item.cam.basis.z).normalize()) < -0.99;
}

function orientPlane(a, b, c, insidePoint) {
  const normal = b.clone().sub(a).cross(c.clone().sub(a)).normalize();
  let plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, a);
  if (plane.distanceToPoint(insidePoint) < 0) {
    plane = new THREE.Plane(plane.normal.clone().negate(), -plane.constant);
  }
  return plane;
}

function frustumPlanes(frustum) {
  const n = frustum.nearCorners;
  const f = frustum.farCorners;
  const inside = frustum.insidePoint;
  return [
    orientPlane(n[0], n[1], n[2], inside),
    orientPlane(f[0], f[2], f[1], inside),
    orientPlane(n[0], f[0], f[1], inside),
    orientPlane(n[1], f[1], f[2], inside),
    orientPlane(n[2], f[2], f[3], inside),
    orientPlane(n[3], f[3], f[0], inside),
  ];
}

function intersectThreePlanes(a, b, c) {
  const n1 = a.normal;
  const n2 = b.normal;
  const n3 = c.normal;
  const n2xn3 = new THREE.Vector3().crossVectors(n2, n3);
  const denom = n1.dot(n2xn3);
  if (Math.abs(denom) < 1e-8) {
    return null;
  }
  const term1 = n2xn3.multiplyScalar(-a.constant);
  const term2 = new THREE.Vector3().crossVectors(n3, n1).multiplyScalar(-b.constant);
  const term3 = new THREE.Vector3().crossVectors(n1, n2).multiplyScalar(-c.constant);
  return term1.add(term2).add(term3).multiplyScalar(1.0 / denom);
}

function makeOverlapGeometry(planes) {
  const eps = 1e-5;
  const vertices = [];
  const keys = new Set();
  for (let i = 0; i < planes.length; ++i) {
    for (let j = i + 1; j < planes.length; ++j) {
      for (let k = j + 1; k < planes.length; ++k) {
        const p = intersectThreePlanes(planes[i], planes[j], planes[k]);
        if (!p) continue;
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) continue;
        const inside = planes.every((plane) => plane.distanceToPoint(p) >= -eps);
        if (!inside) continue;
        const key = [p.x, p.y, p.z].map((v) => Math.round(v * 10000)).join(",");
        if (keys.has(key)) continue;
        keys.add(key);
        vertices.push(p);
      }
    }
  }
  if (vertices.length < 4) {
    return null;
  }

  const positions = [];
  planes.forEach((plane) => {
    const face = vertices.filter((p) => Math.abs(plane.distanceToPoint(p)) < 2e-4);
    if (face.length < 3) return;
    const center = face.reduce((acc, p) => acc.add(p), new THREE.Vector3()).multiplyScalar(1 / face.length);
    let u = face[0].clone().sub(center).normalize();
    if (u.lengthSq() < 1e-10) return;
    const v = new THREE.Vector3().crossVectors(plane.normal, u).normalize();
    face.sort((a, b) => {
      const da = a.clone().sub(center);
      const db = b.clone().sub(center);
      return Math.atan2(da.dot(v), da.dot(u)) - Math.atan2(db.dot(v), db.dot(u));
    });
    for (let i = 1; i + 1 < face.length; ++i) {
      [face[0], face[i], face[i + 1]].forEach((p) => positions.push(p.x, p.y, p.z));
    }
  });
  if (!positions.length) {
    return null;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function makeLabel(text, color) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const font = "600 26px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.font = font;
  const metrics = ctx.measureText(text);
  const width = Math.ceil(metrics.width + 28);
  const height = 42;
  canvas.width = width * 2;
  canvas.height = height * 2;
  ctx.scale(2, 2);
  ctx.clearRect(0, 0, width, height);
  ctx.font = font;
  ctx.textBaseline = "middle";
  ctx.fillStyle = "rgba(244,244,244,0.92)";
  ctx.strokeStyle = "rgba(0,0,0,0.22)";
  ctx.lineWidth = 1;
  roundRect(ctx, 1, 1, width - 2, height - 2, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(text, 14, height / 2 + 1);
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    alphaTest: 0.02,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(width / 430, height / 430, 1);
  return sprite;
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function cameraColor(index) {
  const color = new THREE.Color();
  color.setHSL((index * 0.137) % 1.0, 0.72, 0.56);
  return color;
}

function buildSparsePointCloud() {
  const cloud = RIG_DATA.sparse_point_cloud || {};
  if (!cloud.positions || !cloud.positions.length) {
    pointCloudGroup.visible = false;
    const button = document.getElementById("toggle-points");
    if (button) button.classList.remove("active");
    return;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(cloud.positions, 3));
  if (cloud.colors && cloud.colors.length === cloud.positions.length) {
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(cloud.colors, 3));
  }
  const material = new THREE.PointsMaterial({
    size: Math.max(0.008, radius * 0.012),
    sizeAttenuation: true,
    vertexColors: Boolean(cloud.colors && cloud.colors.length === cloud.positions.length),
    color: 0xffffff,
    transparent: true,
    opacity: 0.95,
    depthWrite: true,
  });
  const points = new THREE.Points(geometry, material);
  pointCloudGroup.add(points);
}

function correspondenceUrl() {
  return viewerOption("correspondence_data_url", "");
}

function correspondenceDatasetName() {
  if (coverageMode === "large_marker") return "large";
  if (coverageMode === "small_marker") return "small";
  return "whole";
}

function correspondenceDataset() {
  if (!correspondenceData) return null;
  const name = correspondenceDatasetName();
  if (name === "whole") return correspondenceData.outer || null;
  return ((correspondenceData.datasets || {})[name]) || null;
}

function correspondenceFrameStats(dataset) {
  const counts = new Map();
  const observations = dataset && dataset.observations ? dataset.observations : [];
  observations.forEach((obs) => {
    const key = String(obs.frame_index);
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  if (!counts.size) {
    const frames = dataset && dataset.frames ? dataset.frames : [];
    frames.forEach((frame) => counts.set(String(frame), 0));
  }
  return Array.from(counts.entries())
    .map(([frame, count]) => ({frame, count}))
    .sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      const av = Number(a.frame);
      const bv = Number(b.frame);
      if (Number.isFinite(av) && Number.isFinite(bv)) return av - bv;
      return String(a.frame).localeCompare(String(b.frame));
    });
}

function correspondenceFrameNumbers(dataset) {
  const frames = new Set();
  const observations = dataset && dataset.observations ? dataset.observations : [];
  observations.forEach((obs) => {
    const value = Number(obs.frame_index);
    if (Number.isFinite(value)) frames.add(value);
  });
  const listedFrames = dataset && dataset.frames ? dataset.frames : [];
  listedFrames.forEach((frame) => {
    const value = Number(frame);
    if (Number.isFinite(value)) frames.add(value);
  });
  return Array.from(frames).sort((a, b) => a - b);
}

function correspondenceFrameObservationCount(dataset, frame) {
  const observations = dataset && dataset.observations ? dataset.observations : [];
  return observations.filter((obs) => String(obs.frame_index) === String(frame)).length;
}

function correspondenceUsesAllFrames(name) {
  return correspondenceAllFramesByDataset[name] === true;
}

function correspondenceDefaultFrame(name, dataset) {
  if (correspondenceSelectedFrameByDataset[name] !== undefined
      && correspondenceSelectedFrameByDataset[name] !== "__all__") {
    return correspondenceSelectedFrameByDataset[name];
  }
  const defaults = (correspondenceData && correspondenceData.defaults && correspondenceData.defaults.frame_by_dataset) || {};
  if (defaults[name] !== undefined && defaults[name] !== null) {
    return defaults[name];
  }
  const top = dataset && dataset.top_frames && dataset.top_frames.length ? dataset.top_frames[0] : null;
  if (top && top.frame_index !== undefined && top.frame_index !== null) {
    return top.frame_index;
  }
  const stats = correspondenceFrameStats(dataset);
  return stats.length ? stats[0].frame : null;
}

function correspondencePointKey(obs, includeFrame) {
  const framePrefix = includeFrame ? ("frame:" + String(obs.frame_index) + "|") : "";
  if (obs.point_index !== undefined && obs.point_index !== null) {
    return framePrefix + "point:" + String(obs.point_index);
  }
  if (obs.face_id !== undefined || obs.tag_id !== undefined || obs.corner_id !== undefined) {
    return framePrefix + "face:" + String(obs.face_id)
      + "|tag:" + String(obs.tag_id)
      + "|corner:" + String(obs.corner_id);
  }
  return framePrefix + "feature:" + String(obs.feature_id);
}

function correspondencePointLabel(obs, includeFrame) {
  const prefix = includeFrame ? ("F" + String(obs.frame_index) + " ") : "";
  if (obs.point_index !== undefined && obs.point_index !== null) {
    return prefix + "point " + String(obs.point_index);
  }
  if (obs.face_id !== undefined || obs.tag_id !== undefined || obs.corner_id !== undefined) {
    return prefix + "face " + String(obs.face_id)
      + " tag " + String(obs.tag_id)
      + " c" + String(obs.corner_id);
  }
  return prefix + "feature " + String(obs.feature_id);
}

function selectedCorrespondencePointGroupKey() {
  const name = correspondenceDatasetName();
  const key = correspondenceSelectedPointGroupByDataset[name];
  return key === undefined ? "__all__" : key;
}

function syncCorrespondenceControlValues() {
  const maxInput = document.getElementById("correspondence-max");
  const residualInput = document.getElementById("correspondence-residual-max");
  const maxValue = document.getElementById("correspondence-max-value");
  const residualValue = document.getElementById("correspondence-residual-max-value");
  if (maxInput) {
    correspondenceMaxShown = Math.max(1, Number(maxInput.value || correspondenceMaxShown));
  }
  if (residualInput) {
    correspondenceResidualMax = Math.max(0, Number(residualInput.value || correspondenceResidualMax));
  }
  if (maxValue) maxValue.textContent = String(correspondenceMaxShown);
  if (residualValue) residualValue.textContent = correspondenceResidualMax >= 200 ? "200+ px" : correspondenceResidualMax + " px";
}

function populateCorrespondenceFrameControl() {
  const controlsEl = document.getElementById("correspondence-controls");
  const slider = document.getElementById("correspondence-frame-slider");
  const allFramesCheckbox = document.getElementById("correspondence-all-frames");
  const frameValue = document.getElementById("correspondence-frame-value");
  if (!controlsEl || !slider || !allFramesCheckbox || !frameValue) return;
  if (!correspondenceLoaded || !correspondenceData) {
    controlsEl.classList.remove("active");
    return;
  }
  const name = correspondenceDatasetName();
  const dataset = correspondenceDataset();
  const frames = correspondenceFrameNumbers(dataset);
  controlsEl.classList.add("active");
  const minFrame = frames.length ? frames[0] : 0;
  const maxFrame = frames.length ? frames[frames.length - 1] : 0;
  slider.min = String(minFrame);
  slider.max = String(maxFrame);
  slider.step = "1";
  let selected = Number(correspondenceDefaultFrame(name, dataset));
  if (!Number.isFinite(selected)) selected = minFrame;
  selected = Math.max(minFrame, Math.min(maxFrame, selected));
  slider.value = String(Math.round(selected));
  correspondenceSelectedFrameByDataset[name] = slider.value;
  allFramesCheckbox.checked = correspondenceUsesAllFrames(name);
  slider.disabled = allFramesCheckbox.checked || !frames.length;
  const obsCount = correspondenceFrameObservationCount(dataset, slider.value);
  frameValue.textContent = allFramesCheckbox.checked
    ? "all"
    : String(slider.value) + " (" + obsCount + " obs)";
  if (!frames.length) {
    frameValue.textContent = "none";
  }
  syncCorrespondenceControlValues();
  populateCorrespondencePointGroupControl();
}

function populateCorrespondencePointGroupControl() {
  const select = document.getElementById("correspondence-point-group");
  if (!select || !correspondenceLoaded || !correspondenceData) return;
  const name = correspondenceDatasetName();
  const dataset = correspondenceDataset();
  const frame = correspondenceDefaultFrame(name, dataset);
  const allFrames = correspondenceUsesAllFrames(name);
  const observations = dataset && dataset.observations ? dataset.observations : [];
  const groups = new Map();
  let frameObservationCount = 0;
  for (const obs of observations) {
    if (!allFrames && frame !== null && frame !== undefined && String(obs.frame_index) !== String(frame)) continue;
    frameObservationCount += 1;
    const key = correspondencePointKey(obs, allFrames);
    let item = groups.get(key);
    if (!item) {
      item = {key, label: correspondencePointLabel(obs, allFrames), count: 0};
      groups.set(key, item);
    }
    item.count += 1;
  }
  const previous = selectedCorrespondencePointGroupKey();
  select.innerHTML = "";
  const allOption = document.createElement("option");
  allOption.value = "__all__";
  allOption.textContent = "All points (" + frameObservationCount + " obs)";
  select.appendChild(allOption);
  Array.from(groups.values())
    .sort((a, b) => {
      if (b.count !== a.count) return b.count - a.count;
      return a.label.localeCompare(b.label);
    })
    .slice(0, 512)
    .forEach((item) => {
      const option = document.createElement("option");
      option.value = item.key;
      option.textContent = item.label + " (" + item.count + " obs)";
      select.appendChild(option);
    });
  select.value = Array.from(select.options).some((option) => option.value === previous) ? previous : "__all__";
  correspondenceSelectedPointGroupByDataset[name] = select.value;
}

function residualColor(residualPx) {
  const value = Number(residualPx || 0);
  const t = Math.max(0, Math.min(1, Math.log1p(value) / Math.log1p(50)));
  return new THREE.Color().setHSL((1 - t) * 0.33, 0.9, 0.48);
}

function clearCorrespondenceOverlay() {
  while (correspondenceGroup.children.length) {
    const child = correspondenceGroup.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  }
}

function updateCorrespondenceOverlay() {
  const status = document.getElementById("correspondence-status");
  if (!status) return;
  clearCorrespondenceOverlay();
  if (!correspondenceLoaded || !correspondenceVisible || !correspondenceData) {
    correspondenceGroup.visible = false;
    status.textContent = correspondenceLoaded ? "Correspondence hidden." : "Correspondence data is loaded on demand.";
    return;
  }
  const name = correspondenceDatasetName();
  const dataset = correspondenceDataset();
  const observations = dataset && dataset.observations ? dataset.observations : [];
  if (!observations.length) {
    correspondenceGroup.visible = false;
    status.textContent = "No feature-level correspondence rows for " + name + ".";
    return;
  }
  const frame = correspondenceDefaultFrame(name, dataset);
  const allFrames = correspondenceUsesAllFrames(name);
  const pointGroupKey = selectedCorrespondencePointGroupKey();
  const cameraLabelById = new Map(RIG_DATA.cameras.map((cam) => [String(cam.label), cam]));
  const cameraIdByIndex = new Map(RIG_DATA.cameras.map((cam) => [Number(cam.index), cam]));
  const selected = [];
  let available = 0;
  for (const obs of observations) {
    if (!allFrames && frame !== null && frame !== undefined && String(obs.frame_index) !== String(frame)) continue;
    if (pointGroupKey !== "__all__" && correspondencePointKey(obs, allFrames) !== pointGroupKey) continue;
    if (Number(obs.residual_px || 0) > correspondenceResidualMax) continue;
    const cam = cameraLabelById.get(String(obs.viewer_camera_label || ""))
      || cameraLabelById.get(String(obs.camera_id))
      || cameraIdByIndex.get(Number(obs.camera_index));
    if (cam && !cameraIsVisible(cam)) continue;
    available += 1;
    if (selected.length < correspondenceMaxShown) selected.push({obs, cam});
  }
  if (!selected.length) {
    correspondenceGroup.visible = false;
    status.textContent = name + " " + (allFrames ? "all frames" : "frame " + frame) + ": no visible correspondences.";
    return;
  }

  const pointPositions = [];
  const pointColors = [];
  const linePositions = [];
  const lineColors = [];
  for (const item of selected) {
    const obs = item.obs;
    const cam = item.cam;
    const point = obs.three;
    const line = obs.line_three;
    if (!point || point.length !== 3) continue;
    const color = residualColor(obs.residual_px);
    pointPositions.push(point[0], point[1], point[2]);
    pointColors.push(color.r, color.g, color.b);
    const lineStart = cam && cam.center ? cam.center : (line && line[0]);
    const lineEnd = point || (line && line[1]);
    if (lineStart && lineEnd && lineStart.length === 3 && lineEnd.length === 3) {
      linePositions.push(lineStart[0], lineStart[1], lineStart[2], lineEnd[0], lineEnd[1], lineEnd[2]);
      lineColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    }
  }

  if (pointPositions.length) {
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.Float32BufferAttribute(pointPositions, 3));
    pointGeometry.setAttribute("color", new THREE.Float32BufferAttribute(pointColors, 3));
    const points = new THREE.Points(pointGeometry, new THREE.PointsMaterial({
      size: Math.max(0.008, radius * 0.008),
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.96,
      depthWrite: true,
    }));
    correspondenceGroup.add(points);
  }
  if (linePositions.length) {
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    lineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(lineColors, 3));
    const lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.18,
      depthWrite: false,
    }));
    correspondenceGroup.add(lines);
  }
  correspondenceGroup.visible = true;
  status.textContent = name + " " + (allFrames ? "all frames" : "frame " + frame) + ": "
    + selected.length + "/" + available + " correspondences shown; residual <= "
    + (correspondenceResidualMax >= 200 ? "200+" : correspondenceResidualMax)
    + " px; point group = " + (pointGroupKey === "__all__" ? "all" : pointGroupKey)
    + "; color = log residual px.";
}

async function loadOrToggleCorrespondence() {
  const button = document.getElementById("load-correspondence");
  const status = document.getElementById("correspondence-status");
  const url = correspondenceUrl();
  if (!url) {
    if (button) button.style.display = "none";
    if (status) status.textContent = "No correspondence data URL configured.";
    return;
  }
  if (!correspondenceLoaded) {
    if (button) button.textContent = "Loading...";
    if (status) status.textContent = "Fetching correspondence JSON...";
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(response.status + " " + response.statusText);
      correspondenceData = await response.json();
      correspondenceLoaded = true;
      correspondenceVisible = true;
      populateCorrespondenceFrameControl();
    } catch (error) {
      correspondenceLoaded = false;
      correspondenceVisible = false;
      if (button) button.textContent = "Load Corr";
      if (status) status.textContent = "Load failed: " + error.message;
      return;
    }
  } else {
    correspondenceVisible = !correspondenceVisible;
  }
  if (button) {
    button.textContent = correspondenceVisible ? "Hide Corr" : "Show Corr";
    button.classList.toggle("active", correspondenceVisible);
  }
  populateCorrespondenceFrameControl();
  updateCorrespondenceOverlay();
}

function buildScene() {
  const sphereRadius = 0.03;
  const sphereGeometry = new THREE.SphereGeometry(sphereRadius, 28, 16);

  RIG_DATA.cameras.forEach((cam) => {
    const color = cameraColor(cam.index);
    const line = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.95})
    );
    line.userData.index = cam.index;
    const fill = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: RIG_DATA.frustum.fill_opacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    fill.userData.index = cam.index;
    frustumGroup.add(fill, line);

    let imagePlane = null;
    if (cam.image_url) {
      const imageMaterial = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.92,
        side: THREE.FrontSide,
        depthWrite: false,
      });
      imagePlane = new THREE.Mesh(new THREE.BufferGeometry(), imageMaterial);
      imagePlane.userData.index = cam.index;
      imagePlane.renderOrder = 2;
      imagePlaneGroup.add(imagePlane);
      loadLocalTexture(
        cam.image_texture_url || cam.image_url,
        (texture) => {
          if (THREE.SRGBColorSpace) {
            texture.colorSpace = THREE.SRGBColorSpace;
          }
          texture.generateMipmaps = false;
          texture.minFilter = THREE.LinearFilter;
          texture.magFilter = THREE.LinearFilter;
          texture.wrapS = THREE.ClampToEdgeWrapping;
          texture.wrapT = THREE.ClampToEdgeWrapping;
          texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
          imageMaterial.map = texture;
          imageMaterial.needsUpdate = true;
        },
        undefined,
        () => {
          imagePlane.visible = false;
        }
      );
    }
    frustumObjects.set(cam.index, {cam, line, fill, imagePlane});

    const xAxis = makeLineSegments([cam.axes.x], 0xe53935, 1);
    const yAxis = makeLineSegments([cam.axes.y], 0x1e8e3e, 1);
    const zAxis = makeLineSegments([cam.axes.z], 0x1a73e8, 1);
    axisGroup.add(xAxis, yAxis, zAxis);

    const material = new THREE.MeshBasicMaterial({color});
    const mesh = new THREE.Mesh(sphereGeometry, material);
    mesh.position.set(cam.center[0], cam.center[1], cam.center[2]);
    mesh.userData.index = cam.index;
    markerGroup.add(mesh);
    pickables.push(mesh);
    cameraMeshes.set(cam.index, mesh);

    const label = makeLabel(cameraLabel(cam), "#" + color.getHexString());
    label.position.set(cam.center[0], cam.center[1] + sphereRadius * 3.2, cam.center[2]);
    label.userData.index = cam.index;
    labelGroup.add(label);
    cameraSceneObjects.set(cam.index, {
      cam,
      axes: [xAxis, yAxis, zAxis],
      marker: mesh,
      label,
    });
  });
  (RIG_DATA.reference_frames || []).forEach((frame) => {
    const center = frame.center || [0, 0, 0];
    const axes = frame.axes || {};
    if (axes.x) axisGroup.add(makeLineSegments([axes.x], 0xe53935, 1));
    if (axes.y) axisGroup.add(makeLineSegments([axes.y], 0x1e8e3e, 1));
    if (axes.z) axisGroup.add(makeLineSegments([axes.z], 0x1a73e8, 1));
    const markerRadius = Number(frame.marker_radius || sphereRadius * 1.35);
    const markerGeometry = new THREE.SphereGeometry(markerRadius, 28, 16);
    const markerColor = new THREE.Color(frame.color || "#fbbc04");
    const marker = new THREE.Mesh(markerGeometry, new THREE.MeshBasicMaterial({color: markerColor}));
    marker.position.set(center[0], center[1], center[2]);
    markerGroup.add(marker);
    const label = makeLabel(frame.label || "reference", frame.label_color || "#fbbc04");
    label.position.set(center[0], center[1] + markerRadius * 3.2, center[2]);
    labelGroup.add(label);
  });
  updateFrustums();
  buildSparsePointCloud();
}

function updateFrustums() {
  const allPlanes = [];
  frustumObjects.forEach(({cam, line, fill, imagePlane}) => {
    const frustum = calcFrustum(cam, nearDistance, farDistance);
    if (line.geometry) line.geometry.dispose();
    if (fill.geometry) fill.geometry.dispose();
    line.geometry = makeGeometryFromLines(frustum.lines);
    fill.geometry = makeFrustumMeshGeometry(frustum);
    if (imagePlane) {
      if (imagePlane.geometry) imagePlane.geometry.dispose();
      imagePlane.geometry = makeImagePlaneGeometry(cam, frustum);
    }
    allPlanes.push(...frustumPlanes(frustum));
  });
  if (viewerOption("enable_overlap", true)) {
    updateOverlap(allPlanes);
  } else {
    overlapGroup.clear();
    const status = document.getElementById("overlap-status");
    if (status) status.textContent = "Overlap: disabled for this viewer";
  }
}

function updateOverlap(planes) {
  overlapGroup.clear();
  const status = document.getElementById("overlap-status");
  if (!planes.length) {
    status.textContent = "Overlap: n/a";
    return;
  }
  const geometry = makeOverlapGeometry(planes);
  if (!geometry) {
    status.textContent = "Overlap: no common volume";
    return;
  }
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
      depthWrite: false,
    })
  );
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.85})
  );
  overlapGroup.add(mesh, edges);
  status.textContent = "Overlap: mesh ready";
}

function buildPivotFrameGizmo() {
  const axisLength = Math.max(0.22, radius * 0.22);
  const axes = [
    {name: "x", color: 0xe53935, dir: new THREE.Vector3(1, 0, 0)},
    {name: "y", color: 0x1e8e3e, dir: new THREE.Vector3(0, 1, 0)},
    {name: "z", color: 0x1a73e8, dir: new THREE.Vector3(0, 0, 1)},
  ];
  axes.forEach((axis) => {
    const line = makeLineSegments([[[0, 0, 0], vecToArray(axis.dir.clone().multiplyScalar(axisLength))]], axis.color, 1);
    pivotFrameGroup.add(line);
    const tip = new THREE.Mesh(
      new THREE.SphereGeometry(axisLength * 0.055, 20, 12),
      new THREE.MeshBasicMaterial({color: axis.color})
    );
    tip.position.copy(axis.dir.clone().multiplyScalar(axisLength));
    pivotFrameGroup.add(tip);
    const label = makeLabel(axis.name, "#" + new THREE.Color(axis.color).getHexString());
    label.scale.multiplyScalar(0.62);
    label.position.copy(axis.dir.clone().multiplyScalar(axisLength * 1.12));
    pivotFrameGroup.add(label);
  });
  pivotFrameGroup.position.copy(cam0ToWorldPoint(bounds.center));
  pivotFrameGroup.quaternion.copy(worldFromCam0Quat);
  transformControl = new THREE.TransformControls(camera, renderer.domElement);
  transformControl.setMode("rotate");
  transformControl.setSpace("local");
  transformControl.setSize(0.75);
  transformControl.attach(pivotFrameGroup);
  transformControl.addEventListener("mouseDown", () => {
    worldGizmoPointerActive = true;
    worldGizmoDragAnchor.copy(pivotFrameGroup.position);
    if (controls) {
      controls.enabled = false;
    }
  });
  transformControl.addEventListener("mouseUp", () => {
    worldFromCam0Quat.copy(pivotFrameGroup.quaternion).normalize();
  });
  transformControl.addEventListener("dragging-changed", (event) => {
    worldGizmoDragging = event.value;
    if (event.value) {
      worldGizmoPointerActive = true;
      worldGizmoDragAnchor.copy(pivotFrameGroup.position);
      if (controls) {
        controls.enabled = false;
      }
    } else {
      commitWorldFromCam0GizmoOrientation();
      if (controls) {
        controls.enabled = true;
      }
      worldGizmoPointerActive = false;
    }
  });
  transformControl.addEventListener("objectChange", () => {
    worldFromCam0Quat.copy(pivotFrameGroup.quaternion).normalize();
    applyWorldFromCam0(false, !worldGizmoDragging);
  });
  scene.add(transformControl);
}

function buildTable() {
  const body = document.getElementById("camera-table");
  body.innerHTML = "";
  tableRows.clear();
  RIG_DATA.cameras.forEach((cam) => {
    const row = document.createElement("tr");
    row.dataset.index = cam.index;
    const c = coverageForCamera(cam);
    const q = calibrationQuality(cam);
    const residual = displayResidualQuality(cam);
    row.innerHTML = "<td>" + cameraLabel(cam) + "</td>"
      + "<td>" + cameraCategory(cam).replace("outer_final", "outer") + "</td>"
      + "<td>" + (c.active === false ? "inactive" : "active") + "</td>"
      + "<td>" + fmt(c.observation_count, 0) + "</td>"
      + "<td>" + fmt(residual.observation_count ?? residual.residual_count, 0) + "</td>"
      + "<td>" + fmt(residual.median_error_px, 3) + "</td>"
      + "<td>" + fmt(residual.p90_error_px, 3) + "</td>"
      + "<td>" + fmt(residual.max_error_px, 3) + "</td>"
      + "<td>" + fmt(q.fx, 1) + "</td>"
      + "<td>" + fmt(q.fy, 1) + "</td>"
      + "<td>" + (residual.decision || residual.source || q.decision || q.source || "-") + "</td>";
    row.addEventListener("click", () => selectCamera(cam.index, true));
    body.appendChild(row);
    tableRows.set(cam.index, row);
  });
}

function buildSummary() {
  const visible = visibleCameraCount();
  const activeInDataset = RIG_DATA.cameras.filter((cam) => cameraCoverageActive(cam)).length;
  const deltas = RIG_DATA.cameras
    .map((cam) => cam.metrics.delta_translation_m)
    .filter((value) => value !== undefined && Number.isFinite(Number(value)))
    .map((value) => Number(value) * 1000.0);
  const maxDelta = deltas.length ? Math.max(...deltas) : NaN;
  document.getElementById("metric-camera-count").textContent =
    visible + " / " + RIG_DATA.cameras.length + " (" + activeInDataset + " active)";
  document.getElementById("metric-near-far").textContent =
    RIG_DATA.frustum.default_near.toFixed(2) + "-" + RIG_DATA.frustum.default_far.toFixed(2) + " m";
  document.getElementById("metric-max-delta").textContent = fmt(maxDelta, 2) + " mm";
  document.getElementById("metric-sparse-points").textContent =
    String((RIG_DATA.sparse_point_cloud || {}).point_count || 0);
  const boardAggregate = (((RIG_DATA.viewer_options || {}).board_orientation_alignment || {}).aggregate || {});
  document.getElementById("metric-board-normal").textContent =
    fmt(boardAggregate.p90_angle_from_horizontal_deg, 2) + " deg";
}

function firstFrameImageCount() {
  return RIG_DATA.cameras.filter((cam) => cam.image_url).length;
}

function visibleCameraCount() {
  return RIG_DATA.cameras.filter((cam) => cameraIsVisible(cam)).length;
}

function selectedCameraData() {
  return RIG_DATA.cameras.find((cam) => cam.index === selectedIndex) || null;
}

function updateSelectedPanel() {
  const cam = selectedCameraData();
  const el = document.getElementById("selected");
  tableRows.forEach((row, index) => row.classList.toggle("selected", index === selectedIndex));
  cameraMeshes.forEach((mesh, index) => {
    mesh.scale.setScalar(1.0);
  });
  if (!cam) {
    el.textContent = "No camera selected.";
    return;
  }
  const m = cam.metrics || {};
  const c = coverageForCamera(cam);
  const q = calibrationQuality(cam);
  const residual = displayResidualQuality(cam);
  const coverageLabel = ((RIG_DATA.dataset_coverage || {}).modes || {})[coverageMode] || {label: coverageMode};
  el.innerHTML = "<strong>" + cameraLabel(cam) + "</strong><br>"
    + "dataset: " + coverageLabel.label + " / " + (c.active === false ? "inactive" : "active")
    + " (" + (c.detail || c.status || "-") + ")<br>"
    + "raw coverage obs: " + fmt(c.observation_count, 0) + "<br>"
    + "final residual: "
    + fmt(residual.median_error_px, 3) + " med, "
    + fmt(residual.p90_error_px, 3) + " p90, "
    + fmt(residual.max_error_px, 3) + " max px; "
    + fmt(residual.observation_count ?? residual.residual_count, 0) + " accepted obs<br>"
    + "intrinsics: fx " + fmt(q.fx, 1) + ", fy " + fmt(q.fy, 1)
    + "; decision: " + (residual.decision || residual.source || q.decision || q.source || "-") + "<br>"
    + "delta refine: "
    + fmt((m.delta_translation_m || 0) * 1000.0, 3) + " mm, "
    + fmt(m.delta_rotation_deg, 4) + " deg<br>"
    + "rotation from cam0: " + fmt(m.rotation_deg, 3) + " deg<br>"
    + "first frame image: " + (cam.image_url ? cam.image_url : "missing");
}

function selectCamera(index, focus) {
  selectedIndex = index;
  updateSelectedPanel();
  if (focus) {
    const cam = selectedCameraData();
    if (cam) {
      controls.target.copy(cam0ToWorldPoint(cam.center));
    }
  }
}

function setObjectOpacity(object, opacity) {
  if (!object || !object.material) return;
  object.material.transparent = opacity < 0.999;
  object.material.opacity = opacity;
  object.material.needsUpdate = true;
}

function applyCoverageStyle(item, visible) {
  const active = cameraCoverageActive(item.cam);
  const mutedColor = 0x7f858a;
  const baseColor = cameraColor(item.cam.index);
  const lineOpacity = active ? 0.95 : 0.20;
  const fillOpacity = active ? RIG_DATA.frustum.fill_opacity : Math.min(0.035, RIG_DATA.frustum.fill_opacity);
  const markerOpacity = active ? 1.0 : 0.28;
  const labelOpacity = active ? 1.0 : 0.34;
  item.line.material.color.set(active ? baseColor : mutedColor);
  item.fill.material.color.set(active ? baseColor : mutedColor);
  setObjectOpacity(item.line, lineOpacity);
  setObjectOpacity(item.fill, fillOpacity);
  if (item.imagePlane) {
    setObjectOpacity(item.imagePlane, active ? 0.92 : 0.12);
  }
  const sceneItem = cameraSceneObjects.get(item.cam.index);
  if (sceneItem) {
    sceneItem.marker.material.color.set(active ? baseColor : mutedColor);
    setObjectOpacity(sceneItem.marker, markerOpacity);
    sceneItem.axes.forEach((axis) => setObjectOpacity(axis, active ? 1.0 : 0.18));
    setObjectOpacity(sceneItem.label, labelOpacity);
  }
  if (!visible) {
    item.line.visible = false;
    item.fill.visible = false;
    if (item.imagePlane) item.imagePlane.visible = false;
  }
}

function applyCameraVisibility() {
  frustumObjects.forEach((item) => {
    const visible = cameraIsVisible(item.cam);
    item.line.visible = visible;
    item.fill.visible = visible;
    if (item.imagePlane) item.imagePlane.visible = visible;
    applyCoverageStyle(item, visible);
  });
  cameraSceneObjects.forEach((item) => {
    const visible = cameraIsVisible(item.cam);
    item.axes.forEach((axis) => { axis.visible = visible; });
    item.marker.visible = visible;
    item.label.visible = visible;
  });
  tableRows.forEach((row, index) => {
    const cam = RIG_DATA.cameras.find((item) => item.index === index);
    const visible = cam && cameraIsVisible(cam);
    row.style.display = visible ? "" : "none";
    row.classList.toggle("coverage-inactive", Boolean(cam && !cameraCoverageActive(cam)));
  });
  const selected = selectedCameraData();
  if (selected && !cameraIsVisible(selected)) {
    const next = RIG_DATA.cameras.find((cam) => cameraIsVisible(cam));
    selectedIndex = next ? next.index : -1;
  }
  updateSelectedPanel();
  if (correspondenceLoaded && correspondenceVisible) {
    updateCorrespondenceOverlay();
  }
}

function cam0UpInWorldVector() {
  return new THREE.Vector3(0, 1, 0).applyQuaternion(worldFromCam0Quat).normalize();
}

function syncCameraUpToWorldFrame() {
  const position = camera.position.clone();
  const target = controls ? controls.target.clone() : null;
  camera.up.copy(WORLD_UP);
  if (target) {
    camera.position.copy(position);
    camera.lookAt(target);
    controls.target.copy(target);
  }
}

function commitWorldFromCam0GizmoOrientation() {
  if (!pivotFrameGroup) return;
  worldFromCam0Quat.copy(pivotFrameGroup.quaternion).normalize();
  applyWorldFromCam0(false, true);
}

function applyWorldFromCam0(refit, syncPivotPosition = true) {
  syncCameraUpToWorldFrame();
  cam0ToWorldRoot.quaternion.copy(worldFromCam0Quat);
  if (!worldGizmoDragging) {
    pivotFrameGroup.quaternion.copy(worldFromCam0Quat);
  }
  if (syncPivotPosition) {
    pivotFrameGroup.position.copy(cam0ToWorldPoint(bounds.center));
  } else {
    pivotFrameGroup.position.copy(worldGizmoDragAnchor);
  }
  if (refit) {
    frameRig(currentFrameMode);
  }
}

function cam0ToWorldPoint(point) {
  return v3(point).applyQuaternion(worldFromCam0Quat);
}

function gravityAlignedAxis(localAxis) {
  return localAxis.clone().applyQuaternion(worldFromCam0Quat).normalize();
}

function horizontalFrameAxis(localAxis) {
  const axis = gravityAlignedAxis(localAxis);
  axis.addScaledVector(WORLD_UP, -axis.dot(WORLD_UP));
  if (axis.lengthSq() > 1e-8) {
    return axis.normalize();
  }
  return null;
}

function horizontalRigForward() {
  return horizontalFrameAxis(new THREE.Vector3(0, 0, 1))
    || horizontalFrameAxis(new THREE.Vector3(1, 0, 0))
    || new THREE.Vector3(1, 0, 0);
}

function projectedCameraUp(preferredUp, viewDirection) {
  const up = preferredUp.clone().normalize();
  const view = viewDirection.clone().normalize();
  up.addScaledVector(view, -up.dot(view));
  if (up.lengthSq() > 1e-8) {
    return up.normalize();
  }
  for (const fallback of [
    gravityAlignedAxis(new THREE.Vector3(0, 0, 1)),
    gravityAlignedAxis(new THREE.Vector3(1, 0, 0)),
    WORLD_UP.clone(),
  ]) {
    fallback.addScaledVector(view, -fallback.dot(view));
    if (fallback.lengthSq() > 1e-8) {
      return fallback.normalize();
    }
  }
  return WORLD_UP.clone();
}

function frameRig(mode) {
  currentFrameMode = mode;
  const c = bounds.center;
  const mobile = viewport.clientWidth <= 760;
  const targetY = c[1] + (mobile ? -radius * 0.36 : 0);
  const zoom = mobile ? 1.45 : 1.0;
  const localTarget = new THREE.Vector3(c[0], targetY, c[2]);
  // The world/root quaternion already maps the estimated rig gravity to display WORLD_UP.
  // Top/front view buttons are therefore defined in the display gravity frame, not in
  // the original rig local-Y frame.
  const gravityUp = WORLD_UP.clone();
  const controlsUp = gravityUp.clone().negate();
  const rigForward = horizontalRigForward();
  let offset;
  if (mode === "top") {
    offset = gravityUp.clone().multiplyScalar(-radius * 2.85 * zoom)
      .addScaledVector(rigForward, radius * 0.002);
  } else if (mode === "front") {
    offset = rigForward.clone().multiplyScalar(radius * 2.65 * zoom);
  } else {
    offset = new THREE.Vector3(radius * 1.55 * zoom, radius * 1.15 * zoom, radius * 1.95 * zoom)
      .applyQuaternion(worldFromCam0Quat);
  }
  const target = cam0ToWorldPoint(localTarget.toArray());
  camera.position.copy(target.clone().add(offset));
  controls.target.copy(target);
  camera.near = 0.005;
  camera.far = Math.max(20, radius * 20);
  camera.updateProjectionMatrix();
  rebuildOrbitControlsForCurrentUp(true, controlsUp);
}

function setToggle(buttonId, object) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  button.addEventListener("click", () => {
    object.visible = !object.visible;
    button.classList.toggle("active", object.visible);
    if (buttonId === "toggle-overlap") {
      document.getElementById("metric-overlap").textContent = object.visible ? "on" : "off";
    }
  });
}

function setGizmoToggle(buttonId) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  button.addEventListener("click", () => {
    const visible = !pivotFrameGroup.visible;
    pivotFrameGroup.visible = visible;
    if (transformControl) {
      transformControl.visible = visible;
      transformControl.enabled = visible;
    }
    worldGizmoDragging = false;
    worldGizmoPointerActive = false;
    if (controls) {
      controls.enabled = true;
    }
    button.classList.toggle("active", visible);
  });
}

function syncDistanceSliders(changed) {
  const nearSlider = document.getElementById("near-slider");
  const farSlider = document.getElementById("far-slider");
  let nextNear = Number(nearSlider.value);
  let nextFar = Number(farSlider.value);
  if (nextNear >= nextFar) {
    if (changed === "near") {
      nextNear = Math.max(Number(nearSlider.min), nextFar - 0.01);
    } else {
      nextFar = Math.min(Number(farSlider.max), nextNear + 0.01);
    }
  }
  nearDistance = nextNear;
  farDistance = nextFar;
  nearSlider.value = nearDistance.toFixed(2);
  farSlider.value = farDistance.toFixed(2);
  document.getElementById("near-value").textContent = nearDistance.toFixed(2) + " m";
  document.getElementById("far-value").textContent = farDistance.toFixed(2) + " m";
  updateFrustums();
}

function initDistanceSliders() {
  const nearSlider = document.getElementById("near-slider");
  const farSlider = document.getElementById("far-slider");
  nearSlider.value = nearDistance.toFixed(2);
  farSlider.value = farDistance.toFixed(2);
  nearSlider.addEventListener("input", () => syncDistanceSliders("near"));
  farSlider.addEventListener("input", () => syncDistanceSliders("far"));
  syncDistanceSliders("");
}

function onPointerDown(event) {
  if (transformControl && (
    worldGizmoPointerActive || worldGizmoDragging || transformControl.dragging || transformControl.axis
  )) {
    event.preventDefault();
    event.stopImmediatePropagation();
    return;
  }
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(pickables.filter((object) => object.visible), false);
  if (hits.length) {
    selectCamera(hits[0].object.userData.index, true);
  }
}

function onResize() {
  const width = Math.max(320, viewport.clientWidth);
  const height = Math.max(320, viewport.clientHeight);
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  if (controls && !worldGizmoDragging) {
    controls.update();
  }
  renderer.render(scene, camera);
}

function setupCameraCategoryToggles() {
  [
    ["toggle-outer-topdown", "outer_topdown"],
    ["toggle-outer-colmap", "outer_colmap"],
  ].forEach(([buttonId, category]) => {
    const button = document.getElementById(buttonId);
    if (!button) return;
    if (!cameraCategoryExists(category)) {
      button.style.display = "none";
      return;
    }
    button.classList.toggle("active", cameraVisibility[category] !== false);
    button.addEventListener("click", () => {
      cameraVisibility[category] = !cameraVisibility[category];
      button.classList.toggle("active", cameraVisibility[category]);
      applyCameraVisibility();
    });
  });
}

function setupCorrespondenceControls() {
  const section = document.getElementById("correspondence-section");
  const button = document.getElementById("load-correspondence");
  const status = document.getElementById("correspondence-status");
  const frameSlider = document.getElementById("correspondence-frame-slider");
  const allFramesCheckbox = document.getElementById("correspondence-all-frames");
  const pointSelect = document.getElementById("correspondence-point-group");
  const maxInput = document.getElementById("correspondence-max");
  const residualInput = document.getElementById("correspondence-residual-max");
  if (!correspondenceUrl()) {
    if (section) section.style.display = "none";
    return;
  }
  if (status) status.textContent = "Click Load Corr to fetch feature correspondences.";
  if (button) button.addEventListener("click", loadOrToggleCorrespondence);
  if (frameSlider) {
    frameSlider.addEventListener("input", () => {
      const name = correspondenceDatasetName();
      correspondenceSelectedFrameByDataset[name] = frameSlider.value;
      correspondenceSelectedPointGroupByDataset[name] = "__all__";
      correspondenceAllFramesByDataset[name] = false;
      if (allFramesCheckbox) allFramesCheckbox.checked = false;
      populateCorrespondenceFrameControl();
      updateCorrespondenceOverlay();
    });
  }
  if (allFramesCheckbox) {
    allFramesCheckbox.addEventListener("change", () => {
      const name = correspondenceDatasetName();
      correspondenceAllFramesByDataset[name] = allFramesCheckbox.checked;
      correspondenceSelectedPointGroupByDataset[correspondenceDatasetName()] = "__all__";
      populateCorrespondenceFrameControl();
      populateCorrespondencePointGroupControl();
      updateCorrespondenceOverlay();
    });
  }
  if (pointSelect) {
    pointSelect.addEventListener("change", () => {
      correspondenceSelectedPointGroupByDataset[correspondenceDatasetName()] = pointSelect.value;
      updateCorrespondenceOverlay();
    });
  }
  [maxInput, residualInput].forEach((input) => {
    if (!input) return;
    input.addEventListener("input", () => {
      syncCorrespondenceControlValues();
      updateCorrespondenceOverlay();
    });
  });
  syncCorrespondenceControlValues();
}

function setButtonGroupActive(prefix, activeId) {
  document.querySelectorAll("[id^='" + prefix + "']").forEach((button) => {
    button.classList.toggle("active", button.id === activeId);
  });
}

function refreshDatasetCoverageUi() {
  const modes = (RIG_DATA.dataset_coverage || {}).modes || {};
  const mode = modes[coverageMode] || {};
  document.getElementById("source").textContent =
    RIG_DATA.cameras.length + " cameras; "
    + firstFrameImageCount() + " first-frame textures; generated " + RIG_DATA.generated_at
    + "; dataset coverage: " + (mode.label || coverageMode)
    + " (" + (mode.active_camera_count ?? "-") + " active)";
  buildTable();
  buildSummary();
  if (correspondenceLoaded) {
    populateCorrespondenceFrameControl();
  }
  applyCameraVisibility();
}

function setupScopeControls() {
  [
    ["scope-all", "all"],
    ["scope-inner", "inner"],
    ["scope-outer", "outer"],
  ].forEach(([buttonId, scope]) => {
    const button = document.getElementById(buttonId);
    if (!button) return;
    button.addEventListener("click", () => {
      cameraScope = scope;
      setButtonGroupActive("scope-", buttonId);
      buildSummary();
      applyCameraVisibility();
    });
  });
}

function setupCoverageControls() {
  [
    ["coverage-whole", "whole"],
    ["coverage-large-marker", "large_marker"],
    ["coverage-small-marker", "small_marker"],
  ].forEach(([buttonId, mode]) => {
    const button = document.getElementById(buttonId);
    if (!button) return;
    const modeInfo = ((RIG_DATA.dataset_coverage || {}).modes || {})[mode];
    if (!modeInfo) {
      button.style.display = "none";
      return;
    }
    button.addEventListener("click", () => {
      coverageMode = mode;
      setButtonGroupActive("coverage-", buttonId);
      refreshDatasetCoverageUi();
    });
  });
  setButtonGroupActive("coverage-", coverageMode === "large_marker"
    ? "coverage-large-marker"
    : coverageMode === "small_marker"
      ? "coverage-small-marker"
      : "coverage-whole");
}

document.getElementById("report-title").textContent = RIG_DATA.title || "Interactive 3D Camera Rig";
document.getElementById("top").addEventListener("click", () => frameRig("top"));
document.getElementById("front").addEventListener("click", () => frameRig("front"));
setToggle("toggle-frustum", frustumGroup);
setToggle("toggle-axes", axisGroup);
setToggle("toggle-labels", labelGroup);
setToggle("toggle-grid", grid);
setToggle("toggle-images", imagePlaneGroup);
setToggle("toggle-points", pointCloudGroup);
setupScopeControls();
setupCoverageControls();
setupCameraCategoryToggles();
setupCorrespondenceControls();
if (viewerOption("enable_overlap", true)) {
  setToggle("toggle-overlap", overlapGroup);
} else {
  const overlapButton = document.getElementById("toggle-overlap");
  if (overlapButton) overlapButton.style.display = "none";
  document.getElementById("metric-overlap").textContent = "disabled";
}
window.addEventListener("resize", onResize);
new ResizeObserver(onResize).observe(viewport);

initDistanceSliders();
buildScene();
buildPivotFrameGizmo();
setGizmoToggle("toggle-gizmo");
renderer.domElement.addEventListener("pointerdown", onPointerDown);
controls = createOrbitControls();
buildTable();
buildSummary();
refreshDatasetCoverageUi();
if (viewport.clientWidth <= 760) {
  labelGroup.visible = false;
  document.getElementById("toggle-labels").classList.remove("active");
}
onResize();
applyWorldFromCam0(false);
applyCameraVisibility();
frameRig("iso");
selectCamera(selectedIndex, false);
animate();

window.__rigViewer = {
  camera,
  get controls() { return controls; },
  renderer,
  scene,
  cam0ToWorldRoot,
  root: cam0ToWorldRoot,
  frustumGroup,
  imagePlaneGroup,
  pointCloudGroup,
  axisGroup,
  labelGroup,
  markerGroup,
  transformControl,
  pivotFrameGroup,
  grid,
  frameRig,
  updateFrustums,
  getFrustumState: () => ({
    near: nearDistance,
    far: farDistance,
    frustumMeshCount: frustumObjects.size,
    imagePlaneCount: Array.from(frustumObjects.values()).filter((item) => item.imagePlane).length,
    imagePlaneFrontSideCount: Array.from(frustumObjects.values()).filter((item) =>
      item.imagePlane && item.imagePlane.material && item.imagePlane.material.side === THREE.FrontSide
    ).length,
    imagePlaneFacingCameraCount: Array.from(frustumObjects.values()).filter((item) => imagePlaneFacesCamera(item)).length,
    imageTextureReadyCount: Array.from(frustumObjects.values()).filter((item) =>
      item.imagePlane && item.imagePlane.material && item.imagePlane.material.map && item.imagePlane.material.map.image
    ).length,
    visibleCameraCount: visibleCameraCount(),
    referenceFrameCount: (RIG_DATA.reference_frames || []).length,
    cameraScope,
    coverageMode,
    datasetActiveVisibleCount: RIG_DATA.cameras.filter((cam) => cameraIsVisible(cam) && cameraCoverageActive(cam)).length,
    innerVisible: cameraVisibility.inner,
    outerVisible: cameraVisibility.outer,
    outerTopdownVisible: cameraVisibility.outer_topdown,
    outerColmapVisible: cameraVisibility.outer_colmap,
    imageGroupVisible: imagePlaneGroup.visible,
    sparsePointCount: (RIG_DATA.sparse_point_cloud || {}).point_count || 0,
    pointGroupVisible: pointCloudGroup.visible,
    gizmoVisible: pivotFrameGroup.visible && (!transformControl || transformControl.visible),
    gizmoEnabled: !transformControl || transformControl.enabled,
    overlapVisible: overlapGroup.visible,
    overlapMeshCount: overlapGroup.children.length,
    correspondenceLoaded,
    correspondenceVisible: correspondenceGroup.visible,
    correspondenceObjectCount: correspondenceGroup.children.length,
    displayUp: WORLD_UP.toArray(),
    cam0UpInWorld: cam0UpInWorldVector().toArray(),
    worldFromCam0Quaternion: worldFromCam0Quat.toArray(),
    cam0ToWorldRootQuaternion: cam0ToWorldRoot.quaternion.toArray(),
    worldGridQuaternion: grid.quaternion.toArray(),
  }),
  setWorldFromCam0ForTest: (x, y, z) => {
    worldFromCam0Quat.setFromEuler(new THREE.Euler(x, y, z, "XYZ")).normalize();
    pivotFrameGroup.quaternion.copy(worldFromCam0Quat);
    applyWorldFromCam0(false);
  },
  getSelectedIndex: () => selectedIndex,
};
  </script>
</body>
</html>
"""


def write_html(path, rig_data):
    html_text = HTML_TEMPLATE.replace(
        "__RIG_DATA__",
        json.dumps(rig_data, indent=2),
    )
    html_text = html_text.replace(
        "__REPROJECTION_HTML__",
        render_reprojection_sections(rig_data.get("reprojection_reports", [])),
    )
    html_text = html_text.replace(
        "__BOARD_ORIENTATION_HTML__",
        render_board_orientation_section(rig_data),
    )
    Path(path).write_text(html_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-yaml", required=True)
    parser.add_argument("--metrics-tsv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Interactive 3D Camera Rig")
    parser.add_argument("--frustum-depth", type=float, default=0.25)
    parser.add_argument("--frustum-half-width", type=float, default=0.14)
    parser.add_argument("--frustum-half-height", type=float, default=0.09)
    parser.add_argument("--default-near", type=float, default=0.3)
    parser.add_argument("--default-far", type=float, default=0.7)
    parser.add_argument("--frustum-fill-alpha", type=float, default=0.1)
    parser.add_argument("--axis-length", type=float, default=0.16)
    parser.add_argument("--camera-image-dir", default="")
    parser.add_argument("--camera-image-texture-max-width", type=int, default=768)
    parser.add_argument("--camera-image-texture-quality", type=int, default=82)
    parser.add_argument("--sparse-point-cloud-json", default="")
    parser.add_argument(
        "--reprojection-report",
        action="append",
        default=[],
        help="Reprojection report directory with camera_metrics.tsv. Use name=path to set the displayed stage name.",
    )
    parser.add_argument("--used-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rig_data = build_rig_data(args.pose_yaml, args.metrics_tsv, args, output_dir)
    write_html(output_dir / "index.html", rig_data)
    (output_dir / "rig_data.json").write_text(json.dumps(rig_data, indent=2), encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
