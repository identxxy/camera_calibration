#!/usr/bin/env python3
"""Distributed AprilTag tower quality filter.

Worker mode scans local Windows-style capture folders and writes lightweight
per-image AprilTag counts. Aggregation mode runs on t0, merges those worker
outputs, and creates a filtered whole-dir layout with manifest.tsv and
image_directories.txt compatible with the existing calibration scripts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_ID_RE = re.compile(r"_(\d+)\.[^.]+$", re.IGNORECASE)
TRAILING_ID_RE = re.compile(r"(\d+)$")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOWER_CONFIG = (
    REPO_ROOT
    / "applications/camera_calibration/patterns/apriltag_tower_8faces_2x16_8cm.yaml"
)

OUTER_CAMERA_MACHINE = {}
for _machine, _labels in [
    ("w4_D", ["1-1", "1-2", "1-3", "2-1", "2-2", "2-3", "3-1", "3-2", "3-3", "4-1", "4-2", "4-3"]),
    ("w3_D", ["5-1", "5-2", "5-3", "6-1", "6-2", "6-3", "7-1", "7-2", "7-3", "8-1", "8-2", "8-3"]),
    ("w1_D", ["22463688", "22463690", "22587611", "22587616"]),
    ("w2_D", ["22463689", "22463691", "22463702", "22587614"]),
]:
    for _label in _labels:
        OUTER_CAMERA_MACHINE[_label] = _machine


def split_values(text):
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,\r\n]+", str(text)) if item.strip()]


def natural_key(value):
    parts = re.split(r"(\d+)", str(value))
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def frame_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def frame_text(value):
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def frame_item_sort_key(item):
    return (natural_key(item["time"]), frame_key(item["frame_id"]))


def frame_item_id(item):
    return f"{item['time']}::{item['frame_id']}"


def parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool_text(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}


def parse_simple_yaml(path):
    data = {}
    path = Path(path)
    if not path.is_file():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value.strip("\"'")
        data[key.strip()] = parsed
    return data


def tower_valid_tag_ids(config):
    try:
        faces = int(config.get("faces", 0))
        columns = int(config.get("tag_columns", 0))
        rows = int(config.get("tag_rows", 0))
        first_tag_id = int(config.get("first_tag_id", 0))
        stride = int(config.get("face_id_stride", columns * rows))
    except (TypeError, ValueError):
        return set()
    valid = set()
    for face in range(max(0, faces)):
        start = first_tag_id + face * stride
        for offset in range(max(0, columns * rows)):
            valid.add(start + offset)
    return valid


def dictionary_from_config(config, explicit):
    if explicit:
        return explicit
    family = str(config.get("tag_family", "tag36h11")).lower()
    if family == "tag36h11":
        return "DICT_APRILTAG_36h11"
    if family == "tag25h9":
        return "DICT_APRILTAG_25h9"
    if family == "tag16h5":
        return "DICT_APRILTAG_16h5"
    return "DICT_APRILTAG_36h11"


def load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "OpenCV Python import failed. Install opencv-contrib-python on the worker "
            f"or use detect --dry-run for path validation only. Error: {exc}")
    if not hasattr(cv2, "aruco"):
        raise SystemExit("cv2.aruco is unavailable. Install opencv-contrib-python.")
    return cv2


def corner_refinement_method(cv2, name):
    methods = {
        "none": "CORNER_REFINE_NONE",
        "subpix": "CORNER_REFINE_SUBPIX",
        "contour": "CORNER_REFINE_CONTOUR",
        "apriltag": "CORNER_REFINE_APRILTAG",
    }
    attr = methods.get(str(name).lower())
    if attr is None:
        raise SystemExit(f"Unsupported corner refinement method: {name}")
    return getattr(cv2.aruco, attr, getattr(cv2.aruco, "CORNER_REFINE_NONE", 0))


def create_detector(
        cv2,
        dictionary_name,
        detect_inverted,
        error_correction_rate,
        corner_refinement="subpix",
        corner_refinement_window_size=5,
        corner_refinement_max_iterations=30,
        corner_refinement_min_accuracy=0.01):
    if not hasattr(cv2.aruco, dictionary_name):
        raise SystemExit(f"OpenCV has no dictionary: cv2.aruco.{dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(parameters, "detectInvertedMarker"):
        parameters.detectInvertedMarker = bool(detect_inverted)
    if hasattr(parameters, "errorCorrectionRate"):
        parameters.errorCorrectionRate = float(error_correction_rate)
    if hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = corner_refinement_method(cv2, corner_refinement)
    if hasattr(parameters, "cornerRefinementWinSize"):
        parameters.cornerRefinementWinSize = int(corner_refinement_window_size)
    if hasattr(parameters, "cornerRefinementMaxIterations"):
        parameters.cornerRefinementMaxIterations = int(corner_refinement_max_iterations)
    if hasattr(parameters, "cornerRefinementMinAccuracy"):
        parameters.cornerRefinementMinAccuracy = float(corner_refinement_min_accuracy)
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def detect_markers(cv2, detector, image):
    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=parameters)
    detections = []
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
            pts = marker_corners.reshape(-1, 2).tolist()
            detections.append({
                "tag_id": int(marker_id),
                "corners": [[float(x), float(y)] for x, y in pts],
            })
    return detections, len(rejected) if rejected is not None else 0


def resize_for_detection(cv2, image, resize_factor):
    if resize_factor >= 0.999:
        return image, 1.0
    if resize_factor <= 0:
        raise SystemExit("--resize-factor must be positive.")
    resized = cv2.resize(
        image,
        None,
        fx=resize_factor,
        fy=resize_factor,
        interpolation=cv2.INTER_AREA)
    return resized, 1.0 / resize_factor


def scale_detections(detections, scale):
    if abs(scale - 1.0) < 1e-9:
        return detections
    scaled = []
    for detection in detections:
        scaled.append({
            "tag_id": detection["tag_id"],
            "corners": [
                [float(x) * scale, float(y) * scale]
                for x, y in detection["corners"]
            ],
        })
    return scaled


def refine_detections_subpixel(
        cv2,
        image,
        detections,
        window_size=5,
        max_iterations=30,
        epsilon=0.01):
    if not detections or window_size <= 0:
        return detections
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required for subpixel corner refinement.") from exc

    corners = []
    lengths = []
    for detection in detections:
        pts = detection.get("corners", [])
        lengths.append(len(pts))
        for x, y in pts:
            corners.append([float(x), float(y)])
    if not corners:
        return detections

    corner_array = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        int(max_iterations),
        float(epsilon),
    )
    refined = cv2.cornerSubPix(
        image,
        corner_array,
        (int(window_size), int(window_size)),
        (-1, -1),
        criteria)
    refined = refined.reshape(-1, 2)
    output = []
    cursor = 0
    for detection, length in zip(detections, lengths):
        item = dict(detection)
        item["corners"] = [
            [float(x), float(y)]
            for x, y in refined[cursor:cursor + length]
        ]
        item["subpixel_refined"] = True
        output.append(item)
        cursor += length
    return output


def _bilinear_sample(image, x, y):
    import numpy as np

    gray = np.asarray(image)
    if gray.ndim != 2:
        raise ValueError("edge-line refinement expects a grayscale image")
    height, width = gray.shape[:2]
    if x < 0 or y < 0 or x >= width - 1 or y >= height - 1:
        return None
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    dx = float(x - x0)
    dy = float(y - y0)
    v00 = float(gray[y0, x0])
    v10 = float(gray[y0, x0 + 1])
    v01 = float(gray[y0 + 1, x0])
    v11 = float(gray[y0 + 1, x0 + 1])
    return (
        (1.0 - dx) * (1.0 - dy) * v00
        + dx * (1.0 - dy) * v10
        + (1.0 - dx) * dy * v01
        + dx * dy * v11)


def _fit_line_orthogonal(points):
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    center = pts.mean(axis=0)
    centered = pts - center
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    direction = direction / norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        return None
    normal = normal / normal_norm
    offset = -float(normal @ center)
    residuals = np.abs(pts @ normal + offset)
    return {
        "normal": normal,
        "offset": offset,
        "point_count": int(pts.shape[0]),
        "mean_residual": float(residuals.mean()) if residuals.size else 0.0,
    }


def _intersect_lines(line_a, line_b):
    import numpy as np

    matrix = np.vstack([line_a["normal"], line_b["normal"]])
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-4:
        return None
    rhs = -np.asarray([line_a["offset"], line_b["offset"]], dtype=np.float64)
    point = np.linalg.solve(matrix, rhs)
    if not np.all(np.isfinite(point)):
        return None
    return point


def _sample_edge_line_points(
        image,
        p0,
        p1,
        inside_normal,
        search_radius_px,
        sample_spacing_px,
        gradient_step_px,
        min_gradient,
        polarity):
    import numpy as np

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    inside_normal = np.asarray(inside_normal, dtype=np.float64)
    edge = p1 - p0
    length = float(np.linalg.norm(edge))
    if length < max(4.0, sample_spacing_px * 2.0):
        return []
    direction = edge / length
    normal_norm = float(np.linalg.norm(inside_normal))
    if normal_norm <= 1e-12:
        return []
    inside_normal = inside_normal / normal_norm
    sample_count = max(4, min(80, int(length / max(0.5, sample_spacing_px))))
    search_step = max(0.25, min(1.0, gradient_step_px * 0.5))
    offsets = np.arange(-search_radius_px, search_radius_px + 0.5 * search_step, search_step)
    points = []

    # Avoid corner neighborhoods; adjacent tag edges create ambiguous gradients there.
    for sample_index in range(sample_count):
        alpha = (sample_index + 0.5) / sample_count
        center = p0 * (1.0 - alpha) + p1 * alpha
        best_point = None
        best_score = -1.0
        for offset in offsets:
            candidate = center + inside_normal * offset
            outside = candidate - inside_normal * gradient_step_px
            inside = candidate + inside_normal * gradient_step_px
            value_outside = _bilinear_sample(image, outside[0], outside[1])
            value_inside = _bilinear_sample(image, inside[0], inside[1])
            if value_outside is None or value_inside is None:
                continue
            if polarity == "outside_black_inside_white":
                score = (value_inside - value_outside) / max(1e-6, 2.0 * gradient_step_px)
            elif polarity == "absolute":
                score = abs(value_outside - value_inside) / max(1e-6, 2.0 * gradient_step_px)
            else:
                score = (value_outside - value_inside) / max(1e-6, 2.0 * gradient_step_px)
            if score > best_score:
                best_score = score
                best_point = candidate
        if best_point is not None and best_score >= min_gradient:
            points.append(best_point)
    return points


def refine_detection_edge_lines(
        image,
        corners,
        search_radius_px=5.0,
        sample_spacing_px=2.0,
        gradient_step_px=1.0,
        min_gradient=2.0,
        min_edge_points=8,
        max_shift_px=4.0,
        polarity="outside_white_inside_black"):
    import numpy as np

    original = np.asarray(corners, dtype=np.float64)
    if original.shape != (4, 2):
        return corners, False, 0
    polygon_center = original.mean(axis=0)
    lines = []
    valid_edges = 0
    for edge_index in range(4):
        p0 = original[edge_index]
        p1 = original[(edge_index + 1) % 4]
        edge = p1 - p0
        edge_midpoint = 0.5 * (p0 + p1)
        normal = np.asarray([-edge[1], edge[0]], dtype=np.float64)
        if float(normal @ (polygon_center - edge_midpoint)) < 0:
            normal = -normal
        points = _sample_edge_line_points(
            image,
            p0,
            p1,
            normal,
            float(search_radius_px),
            float(sample_spacing_px),
            float(gradient_step_px),
            float(min_gradient),
            str(polarity))
        if len(points) < int(min_edge_points):
            lines.append(None)
            continue
        line = _fit_line_orthogonal(points)
        lines.append(line)
        if line is not None:
            valid_edges += 1

    refined = original.copy()
    accepted_corners = 0
    for corner_index in range(4):
        previous_line = lines[(corner_index - 1) % 4]
        next_line = lines[corner_index]
        if previous_line is None or next_line is None:
            continue
        point = _intersect_lines(previous_line, next_line)
        if point is None:
            continue
        shift = float(np.linalg.norm(point - original[corner_index]))
        if shift <= float(max_shift_px):
            refined[corner_index] = point
            accepted_corners += 1

    return (
        [[float(x), float(y)] for x, y in refined],
        accepted_corners > 0,
        valid_edges,
    )


def refine_detections_edge_lines(
        image,
        detections,
        search_radius_px=5.0,
        sample_spacing_px=2.0,
        gradient_step_px=1.0,
        min_gradient=2.0,
        min_edge_points=8,
        max_shift_px=4.0,
        polarity="outside_white_inside_black"):
    if not detections:
        return detections
    output = []
    for detection in detections:
        item = dict(detection)
        refined, changed, valid_edges = refine_detection_edge_lines(
            image,
            item.get("corners", []),
            search_radius_px,
            sample_spacing_px,
            gradient_step_px,
            min_gradient,
            min_edge_points,
            max_shift_px,
            polarity)
        item["corners"] = refined
        item["edge_line_refined"] = bool(changed)
        item["edge_line_valid_edges"] = int(valid_edges)
        output.append(item)
    return output


def write_tsv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=fieldnames,
            extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def html_table(rows, columns=None, limit=300):
    rows = rows[:limit]
    if not rows:
        return '<p class="note">No rows.</p>'
    if columns is None:
        columns = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>"
            for column in columns) + "</tr>")
    suffix = ""
    if limit and len(rows) == limit:
        suffix = f'<p class="note">Showing first {limit} rows.</p>'
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
        + suffix)


def write_html(path, title, summary, sections):
    cards = "".join(
        f"<div class=\"metric\"><strong>{html.escape(str(value))}</strong><span>{html.escape(str(label))}</span></div>"
        for label, value in sections.get("metrics", [])
    )
    tables = []
    for section in sections.get("tables", []):
        tables.append(
            f"<h2>{html.escape(section['title'])}</h2>"
            + html_table(section.get("rows", []), section.get("columns"), section.get("limit", 300)))
    outputs = summary.get("outputs", {})
    output_rows = [
        {"name": key, "path": value}
        for key, value in outputs.items()
    ]
    path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #17202a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 26px 0 10px; font-size: 18px; }}
    .note {{ color: #667085; line-height: 1.45; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 25px; }}
    .metric span {{ color: #667085; font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; background: #fff; font-size: 13px; }}
    th, td {{ border: 1px solid #d9dee7; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef1f5; }}
    code {{ word-break: break-word; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(title)}</h1>
  <p class="note">Generated at <code>{html.escape(str(summary.get("generated_at", "")))}</code>. This is a data-quality/filtering report only; it does not run bundle adjustment.</p>
  <div class="metrics">{cards}</div>
  <h2>Outputs</h2>
  {html_table(output_rows, ["name", "path"], 100)}
  {''.join(tables)}
</div>
</body>
</html>
""", encoding="utf-8")


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def list_images(image_dir, max_frames=0, stride=1):
    files = [
        path for path in sorted(Path(image_dir).iterdir(), key=lambda p: natural_key(p.name))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if stride > 1:
        files = files[::stride]
    if max_frames and max_frames > 0:
        files = files[:max_frames]
    return files


def parse_frame_id(filename, fallback_index):
    match = IMAGE_ID_RE.search(filename)
    if match:
        return str(int(match.group(1))), "filename_suffix"
    stem = Path(filename).stem
    match = TRAILING_ID_RE.search(stem)
    if match:
        return str(int(match.group(1))), "stem_suffix"
    return str(fallback_index), "sorted_index"


def discover_camera_dirs(input_root, times, camera_ids):
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {input_root}")
    time_names = split_values(times)
    if not time_names:
        time_names = [
            path.name for path in sorted(input_root.iterdir(), key=lambda p: natural_key(p.name))
            if path.is_dir() and not path.name.startswith("_")
        ]
    camera_filter = set(split_values(camera_ids))
    result = []
    for time_name in time_names:
        time_dir = input_root / time_name
        if not time_dir.is_dir():
            raise SystemExit(f"Time directory does not exist: {time_dir}")
        camera_names = [
            path.name for path in sorted(time_dir.iterdir(), key=lambda p: natural_key(p.name))
            if path.is_dir()
        ]
        if camera_filter:
            camera_names = [name for name in camera_names if name in camera_filter]
        for camera_id in camera_names:
            result.append({
                "time": time_name,
                "camera_id": camera_id,
                "image_dir": time_dir / camera_id,
            })
    if not result:
        raise SystemExit("No camera directories matched the detect input.")
    return result


def init_stats(worker_id, time_id, camera_id, image_dir):
    return {
        "worker_id": worker_id,
        "time": time_id,
        "camera_id": camera_id,
        "image_dir": str(image_dir),
        "total_images": 0,
        "decoded_images": 0,
        "failed_images": 0,
        "passing_images": 0,
        "passing_ratio": 0.0,
        "total_tags": 0,
        "total_corners": 0,
        "max_tags": 0,
        "first_passing_frame_id": "",
        "last_passing_frame_id": "",
        "width": 0,
        "height": 0,
    }


def update_stats(stats, row, min_tags):
    stats["total_images"] += 1
    if parse_bool_text(row.get("decode_ok")):
        stats["decoded_images"] += 1
    else:
        stats["failed_images"] += 1
    tag_count = parse_int(row.get("tag_count"))
    corner_count = parse_int(row.get("corner_count"))
    stats["total_tags"] += tag_count
    stats["total_corners"] += corner_count
    stats["max_tags"] = max(stats["max_tags"], tag_count)
    stats["width"] = parse_int(row.get("width"), stats["width"])
    stats["height"] = parse_int(row.get("height"), stats["height"])
    if tag_count >= min_tags and parse_bool_text(row.get("decode_ok")):
        stats["passing_images"] += 1
        if not stats["first_passing_frame_id"]:
            stats["first_passing_frame_id"] = row.get("frame_id", "")
        stats["last_passing_frame_id"] = row.get("frame_id", "")


def finalize_stats(stats):
    total = stats["total_images"]
    stats["passing_ratio"] = stats["passing_images"] / total if total else 0.0
    return stats


def aggregate_time_stats(camera_stats):
    by_time = {}
    for row in camera_stats:
        item = by_time.setdefault(row["time"], {
            "time": row["time"],
            "camera_count": 0,
            "total_images": 0,
            "decoded_images": 0,
            "failed_images": 0,
            "passing_images": 0,
            "max_tags": 0,
        })
        item["camera_count"] += 1
        for key in ["total_images", "decoded_images", "failed_images", "passing_images"]:
            item[key] += parse_int(row.get(key))
        item["max_tags"] = max(item["max_tags"], parse_int(row.get("max_tags")))
    for item in by_time.values():
        total = item["total_images"]
        item["passing_ratio"] = item["passing_images"] / total if total else 0.0
    return [by_time[key] for key in sorted(by_time, key=natural_key)]


def run_detect(args):
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dirs = discover_camera_dirs(args.input_root, args.time, args.camera_id)
    worker_id = args.worker_id or os.environ.get("COMPUTERNAME", "")

    if args.dry_run:
        rows = []
        for item in camera_dirs:
            files = list_images(item["image_dir"], args.max_frames, args.stride)
            rows.append({
                "worker_id": worker_id,
                "time": item["time"],
                "camera_id": item["camera_id"],
                "image_dir": str(item["image_dir"]),
                "image_count": len(files),
            })
        write_tsv(
            output_dir / "dry_run_images.tsv",
            rows,
            ["worker_id", "time", "camera_id", "image_dir", "image_count"])
        summary = {
            "mode": "detect_dry_run",
            "worker_id": worker_id,
            "camera_time_count": len(rows),
            "image_count": sum(parse_int(row["image_count"]) for row in rows),
            "output_dir": str(output_dir),
        }
        (output_dir / "worker_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    tower_config = parse_simple_yaml(args.tower_config)
    valid_tag_ids = tower_valid_tag_ids(tower_config)
    if args.filter_tower_ids and not valid_tag_ids:
        raise SystemExit(
            "--filter-tower-ids is enabled, but no valid tower tag IDs could be "
            f"read from --tower-config: {args.tower_config}")
    dictionary = dictionary_from_config(tower_config, args.dictionary)
    cv2 = load_cv2()
    detector = create_detector(
        cv2,
        dictionary,
        args.detect_inverted,
        args.error_correction_rate,
        args.corner_refinement,
        args.corner_refinement_window_size,
        args.corner_refinement_max_iterations,
        args.corner_refinement_min_accuracy)

    metric_fields = [
        "worker_id", "time", "camera_id", "image_dir", "filename",
        "frame_id", "frame_id_source", "image_index", "image_path",
        "decode_ok", "width", "height", "tag_count", "corner_count",
        "raw_tag_count", "raw_corner_count", "rejected_count", "tag_ids",
        "raw_tag_ids", "passes_min_tags", "error",
    ]
    camera_stats = {}
    passing_rows = []
    total_images = 0
    total_tags = 0
    start = time.time()

    metrics_path = output_dir / "per_image_metrics.tsv"
    detections_path = output_dir / "detections.jsonl"
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_stream, (
        detections_path.open("w", encoding="utf-8")
        if not args.skip_detections_jsonl else open(os.devnull, "w", encoding="utf-8")
    ) as detections_stream:
        writer = csv.DictWriter(metrics_stream, delimiter="\t", fieldnames=metric_fields)
        writer.writeheader()
        for item in camera_dirs:
            key = (item["time"], item["camera_id"])
            stats = camera_stats.setdefault(
                key,
                init_stats(worker_id, item["time"], item["camera_id"], item["image_dir"]))
            files = list_images(item["image_dir"], args.max_frames, args.stride)
            for image_index, image_path in enumerate(files):
                total_images += 1
                frame_id, frame_source = parse_frame_id(image_path.name, image_index)
                row = {
                    "worker_id": worker_id,
                    "time": item["time"],
                    "camera_id": item["camera_id"],
                    "image_dir": str(item["image_dir"]),
                    "filename": image_path.name,
                    "frame_id": frame_id,
                    "frame_id_source": frame_source,
                    "image_index": image_index,
                    "image_path": str(image_path),
                    "decode_ok": 0,
                    "width": 0,
                    "height": 0,
                    "tag_count": 0,
                    "corner_count": 0,
                    "raw_tag_count": 0,
                    "raw_corner_count": 0,
                    "rejected_count": 0,
                    "tag_ids": "",
                    "raw_tag_ids": "",
                    "passes_min_tags": 0,
                    "error": "",
                }
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    row["error"] = "decode_failed"
                    writer.writerow(row)
                    update_stats(stats, row, args.min_tags)
                    continue

                height, width = image.shape[:2]
                detect_image, scale = resize_for_detection(cv2, image, args.resize_factor)
                detections, rejected_count = detect_markers(cv2, detector, detect_image)
                detections = scale_detections(detections, scale)
                raw_ids = [det["tag_id"] for det in detections]
                if valid_tag_ids and args.filter_tower_ids:
                    detections = [det for det in detections if det["tag_id"] in valid_tag_ids]
                if args.subpixel_refine_original:
                    detections = refine_detections_subpixel(
                        cv2,
                        image,
                        detections,
                        args.subpixel_window_size,
                        args.subpixel_max_iterations,
                        args.subpixel_epsilon)
                if args.edge_line_refine_original:
                    detections = refine_detections_edge_lines(
                        image,
                        detections,
                        args.edge_line_search_radius_px,
                        args.edge_line_sample_spacing_px,
                        args.edge_line_gradient_step_px,
                        args.edge_line_min_gradient,
                        args.edge_line_min_edge_points,
                        args.edge_line_max_shift_px,
                        args.edge_line_polarity)
                tag_ids = [det["tag_id"] for det in detections]

                row.update({
                    "decode_ok": 1,
                    "width": width,
                    "height": height,
                    "tag_count": len(tag_ids),
                    "corner_count": len(tag_ids) * 4,
                    "raw_tag_count": len(raw_ids),
                    "raw_corner_count": len(raw_ids) * 4,
                    "rejected_count": rejected_count,
                    "tag_ids": ",".join(str(tag_id) for tag_id in sorted(tag_ids)),
                    "raw_tag_ids": ",".join(str(tag_id) for tag_id in sorted(raw_ids)),
                    "passes_min_tags": 1 if len(tag_ids) >= args.min_tags else 0,
                })
                writer.writerow(row)
                if row["passes_min_tags"]:
                    passing_rows.append(row.copy())
                total_tags += len(tag_ids)
                update_stats(stats, row, args.min_tags)
                if not args.skip_detections_jsonl:
                    detections_stream.write(json.dumps({
                        "worker_id": worker_id,
                        "time": item["time"],
                        "camera_id": item["camera_id"],
                        "filename": image_path.name,
                        "frame_id": frame_id,
                        "image_path": str(image_path),
                        "width": width,
                        "height": height,
                        "detections": detections,
                        "raw_tag_ids": raw_ids,
                        "rejected_count": rejected_count,
                    }, sort_keys=True) + "\n")

    camera_rows = [
        finalize_stats(camera_stats[key])
        for key in sorted(camera_stats, key=lambda k: (natural_key(k[0]), natural_key(k[1])))
    ]
    camera_fields = [
        "worker_id", "time", "camera_id", "image_dir", "total_images",
        "decoded_images", "failed_images", "passing_images", "passing_ratio",
        "total_tags", "total_corners", "max_tags", "first_passing_frame_id",
        "last_passing_frame_id", "width", "height",
    ]
    write_tsv(output_dir / "per_camera_stats.tsv", camera_rows, camera_fields)
    write_tsv(
        output_dir / "per_time_stats.tsv",
        aggregate_time_stats(camera_rows),
        [
            "time", "camera_count", "total_images", "decoded_images",
            "failed_images", "passing_images", "passing_ratio", "max_tags",
        ])
    write_tsv(output_dir / f"images_min{args.min_tags}.tsv", passing_rows, metric_fields)
    (output_dir / f"images_min{args.min_tags}.txt").write_text(
        "\n".join(row["image_path"] for row in passing_rows) + ("\n" if passing_rows else ""),
        encoding="utf-8")

    summary = {
        "mode": "worker_detect",
        "quality_validation_only": True,
        "worker_id": worker_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start,
        "input_root": str(args.input_root),
        "output_dir": str(output_dir),
        "tower_config": str(args.tower_config),
        "dictionary": dictionary,
        "corner_refinement": args.corner_refinement,
        "subpixel_refine_original": bool(args.subpixel_refine_original),
        "edge_line_refine_original": bool(args.edge_line_refine_original),
        "edge_line_search_radius_px": args.edge_line_search_radius_px,
        "edge_line_sample_spacing_px": args.edge_line_sample_spacing_px,
        "edge_line_gradient_step_px": args.edge_line_gradient_step_px,
        "edge_line_min_gradient": args.edge_line_min_gradient,
        "edge_line_min_edge_points": args.edge_line_min_edge_points,
        "edge_line_max_shift_px": args.edge_line_max_shift_px,
        "edge_line_polarity": args.edge_line_polarity,
        "filter_tower_ids": bool(args.filter_tower_ids),
        "valid_tower_tag_count": len(valid_tag_ids),
        "min_tags": args.min_tags,
        "camera_time_count": len(camera_rows),
        "total_images": total_images,
        "passing_images": len(passing_rows),
        "total_tags": total_tags,
        "outputs": {
            "per_image_metrics": str(metrics_path),
            "per_camera_stats": str(output_dir / "per_camera_stats.tsv"),
            "per_time_stats": str(output_dir / "per_time_stats.tsv"),
            "passing_images_tsv": str(output_dir / f"images_min{args.min_tags}.tsv"),
            "passing_images_txt": str(output_dir / f"images_min{args.min_tags}.txt"),
            "detections_jsonl": str(detections_path) if not args.skip_detections_jsonl else "",
            "index_html": str(output_dir / "index.html"),
        },
    }
    (output_dir / "worker_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_html(
        output_dir / "index.html",
        "Distributed AprilTag Worker Quality Report",
        summary,
        {
            "metrics": [
                ("camera/time dirs", len(camera_rows)),
                ("images", total_images),
                (f">= {args.min_tags} tags", len(passing_rows)),
                ("total tags", total_tags),
            ],
            "tables": [
                {
                    "title": "Per-Camera Stats",
                    "rows": camera_rows,
                    "columns": [
                        "time", "camera_id", "total_images", "decoded_images",
                        "failed_images", "passing_images", "passing_ratio",
                        "max_tags", "first_passing_frame_id", "last_passing_frame_id",
                    ],
                    "limit": 200,
                },
                {
                    "title": f"Images With >= {args.min_tags} Tags",
                    "rows": passing_rows,
                    "columns": [
                        "time", "camera_id", "frame_id", "filename",
                        "tag_count", "corner_count", "image_path",
                    ],
                    "limit": 300,
                },
            ],
        })
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def find_metrics_files(worker_outputs):
    paths = []
    for item in worker_outputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
            continue
        direct = path / "per_image_metrics.tsv"
        if direct.is_file():
            paths.append(direct)
            continue
        paths.extend(sorted(path.glob("**/per_image_metrics.tsv")))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise SystemExit("No per_image_metrics.tsv files found in --worker-output.")
    return unique


def load_metric_rows(worker_outputs):
    rows = []
    for path in find_metrics_files(worker_outputs):
        for row in read_tsv(path):
            row["_metrics_file"] = str(path)
            row["tag_count_int"] = parse_int(row.get("tag_count"))
            row["corner_count_int"] = parse_int(row.get("corner_count"))
            row["decode_ok_bool"] = parse_bool_text(row.get("decode_ok"))
            row["frame_id_text"] = frame_text(row.get("frame_id", ""))
            rows.append(row)
    if not rows:
        raise SystemExit("Worker metrics exist, but contain no image rows.")
    return rows


def read_manifest(path):
    rows = read_tsv(path)
    for index, row in enumerate(rows):
        row["_original_row_index"] = index
        row["_original_camera_index"] = row.get("camera_index", str(index))
    rows.sort(key=lambda row: parse_int(row.get("camera_index"), row["_original_row_index"]))
    return rows


def machine_for_metric(row, camera_id):
    worker = (row.get("worker_id") or "").strip()
    if worker:
        if worker.endswith("_D"):
            return worker
        for machine in ("w1", "w2", "w3", "w4"):
            if worker == machine or worker.startswith(f"{machine}_"):
                return f"{machine}_D"
        return f"{worker}_D"
    return OUTER_CAMERA_MACHINE.get(camera_id, "")


def infer_source_dir(args, row, time_id, camera_id):
    image_dir = row.get("image_dir", "")
    if image_dir and Path(image_dir).is_dir():
        return image_dir
    machine = machine_for_metric(row, camera_id)
    if args.mount_root and machine:
        return str(Path(args.mount_root) / machine / "output/calib" / args.marker / time_id / camera_id)
    return str(Path(row.get("image_path", "")).parent) if row.get("image_path") else ""


def resolve_aggregate_times(args, metrics):
    available_times = sorted(
        {row.get("time", "") for row in metrics if row.get("time", "")},
        key=natural_key,
    )
    time_ids = split_values(args.time)
    if not time_ids:
        time_ids = available_times
    missing = [time_id for time_id in time_ids if time_id not in available_times]
    if missing:
        raise SystemExit(
            "Requested --time values are missing from worker metrics: "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(available_times))
    if not time_ids:
        raise SystemExit("Worker metrics do not contain any time values.")
    return time_ids, available_times


def build_manifest_rows(args, metrics, time_ids):
    by_camera = {}
    time_set = set(time_ids)
    for row in metrics:
        if row.get("time") not in time_set:
            continue
        by_camera.setdefault(row.get("camera_id", ""), row)
    if args.base_manifest:
        base_rows = read_manifest(args.base_manifest)
        rows = []
        missing = []
        for row in base_rows:
            camera_id = row.get("camera_id", row.get("user_id", ""))
            if camera_id in by_camera:
                rows.append(dict(row))
            else:
                missing.append(camera_id)
        if not rows:
            raise SystemExit("No base manifest cameras are present in worker metrics.")
    else:
        rows = []
        missing = []
        for index, camera_id in enumerate(sorted(by_camera, key=natural_key)):
            metric = by_camera[camera_id]
            machine = machine_for_metric(metric, camera_id)
            machine_label = machine.replace("_D", "") if machine else metric.get("worker_id", "")
            rows.append({
                "camera_index": str(index),
                "stage_name": f"cam{index:02d}_{machine_label}_{camera_id}".replace("__", "_"),
                "machine": machine,
                "camera_id": camera_id,
                "source_dir": infer_source_dir(args, metric, metric.get("time", ""), camera_id),
                "frame_count": "0",
            })
    return rows, missing


def choose_metric(existing, candidate):
    if existing is None:
        return candidate
    if candidate["tag_count_int"] > existing["tag_count_int"]:
        return candidate
    if candidate["tag_count_int"] == existing["tag_count_int"]:
        if candidate.get("decode_ok_bool") and not existing.get("decode_ok_bool"):
            return candidate
    return existing


def group_metrics(metrics, time_ids, camera_ids):
    camera_ids = set(camera_ids)
    time_set = set(time_ids)
    by_camera_frame = {}
    for row in metrics:
        time_id = row.get("time")
        if time_id not in time_set:
            continue
        camera_id = row.get("camera_id", "")
        if camera_id not in camera_ids:
            continue
        frame_id = row.get("frame_id_text", frame_text(row.get("frame_id", "")))
        key = (camera_id, time_id, frame_id)
        by_camera_frame[key] = choose_metric(by_camera_frame.get(key), row)
    return by_camera_frame


def select_frames(by_camera_frame, camera_ids, min_tags, min_cameras_per_frame, require_all_metrics):
    frame_items = [
        {"time": time_id, "frame_id": frame_id}
        for time_id, frame_id in sorted(
            {(time_id, frame_id) for _camera_id, time_id, frame_id in by_camera_frame},
            key=lambda item: (natural_key(item[0]), frame_key(item[1])),
        )
    ]
    selected = []
    frame_rows = []
    for item in frame_items:
        time_id = item["time"]
        frame_id = item["frame_id"]
        observed = 0
        passing = 0
        for camera_id in camera_ids:
            row = by_camera_frame.get((camera_id, time_id, frame_id))
            if not row:
                continue
            observed += 1
            if row.get("decode_ok_bool") and row.get("tag_count_int", 0) >= min_tags:
                passing += 1
        if require_all_metrics and observed != len(camera_ids):
            status = "missing_camera_metrics"
            accepted = False
        elif passing < min_cameras_per_frame:
            status = "below_min_cameras_per_frame"
            accepted = False
        else:
            status = "selected"
            accepted = True
            selected.append(item)
        frame_rows.append({
            "time": time_id,
            "frame_id": frame_id,
            "frame_key": frame_item_id(item),
            "observed_camera_count": observed,
            "passing_camera_count": passing,
            "active_camera_count": len(camera_ids),
            "status": status,
            "selected": 1 if accepted else 0,
        })
    return selected, frame_rows


def source_candidates(args, manifest_row, metric_row):
    filename = metric_row.get("filename", "")
    candidates = []
    image_path = metric_row.get("image_path", "")
    if image_path:
        candidates.append(Path(image_path))
    machine = machine_for_metric(metric_row, manifest_row.get("camera_id", ""))
    if args.mount_root and machine and filename:
        candidates.append(
            Path(args.mount_root)
            / machine
            / "output/calib"
            / args.marker
            / metric_row.get("time", "")
            / manifest_row.get("camera_id", "")
            / filename)
    original = manifest_row.get("source_dir", "")
    if original and filename:
        candidates.append(Path(original) / filename)
    return candidates


def resolve_source(args, manifest_row, metric_row):
    candidates = source_candidates(args, manifest_row, metric_row)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if args.no_check_source_exists and candidates:
        return candidates[0]
    return None


def link_image(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise SystemExit(f"Unsupported link mode: {mode}")


def prepare_output_dir(path, overwrite):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"Refusing to overwrite non-empty output dir: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_filtered_dataset(args, output_dir, manifest_rows, by_camera_frame, selected_frames):
    image_dirs = []
    output_manifest = []
    selected_rows = []
    missing_sources = []
    images_root = output_dir / "images"
    for new_index, row in enumerate(manifest_rows):
        camera_id = row.get("camera_id", row.get("user_id", ""))
        stage_name = row.get("stage_name") or f"cam{new_index:02d}_{camera_id}"
        dst_dir = images_root / stage_name
        image_dirs.append(str(dst_dir))
        original_source_dir = row.get("source_dir", "")
        for out_frame, frame in enumerate(selected_frames):
            time_id = frame["time"]
            frame_id = frame["frame_id"]
            metric = by_camera_frame.get((camera_id, time_id, frame_id))
            if not metric:
                missing_sources.append({
                    "camera_id": camera_id,
                    "time": time_id,
                    "frame_id": frame_id,
                    "reason": "missing_metric_row",
                })
                continue
            src = resolve_source(args, row, metric)
            if not src:
                missing_sources.append({
                    "camera_id": camera_id,
                    "time": time_id,
                    "frame_id": frame_id,
                    "filename": metric.get("filename", ""),
                    "reason": "source_not_found",
                    "candidates": ";".join(str(path) for path in source_candidates(args, row, metric)),
                })
                continue
            dst = dst_dir / f"{out_frame:06d}{args.output_extension}"
            link_image(src, dst, args.link_mode)
            selected_rows.append({
                "out_frame": out_frame,
                "time": time_id,
                "frame_id": frame_id,
                "frame_key": frame_item_id(frame),
                "camera_index": new_index,
                "camera_id": camera_id,
                "tag_count": metric.get("tag_count", ""),
                "corner_count": metric.get("corner_count", ""),
                "source": str(src),
                "filtered_image": str(dst),
            })
        output_row = dict(row)
        output_row["camera_index"] = new_index
        output_row["stage_name"] = stage_name
        output_row["source_dir"] = str(dst_dir)
        output_row["frame_count"] = len(selected_frames)
        output_row["original_camera_index"] = row.get("_original_camera_index", row.get("camera_index", new_index))
        output_row["original_source_dir"] = original_source_dir
        output_manifest.append(output_row)
    if missing_sources and not args.allow_missing_source:
        write_tsv(
            output_dir / "missing_sources.tsv",
            missing_sources,
            sorted({key for row in missing_sources for key in row.keys()}))
        raise SystemExit(
            f"Missing {len(missing_sources)} source images. See {output_dir / 'missing_sources.tsv'}")
    return image_dirs, output_manifest, selected_rows, missing_sources


def metric_passes(row, min_tags):
    return row.get("decode_ok_bool") and row.get("tag_count_int", 0) >= min_tags


def write_passing_image_dataset(args, output_dir, manifest_rows, chosen_metrics, time_ids):
    image_dirs = []
    output_manifest = []
    selected_rows = []
    missing_sources = []
    images_root = output_dir / "images"
    time_set = set(time_ids)
    metrics_by_camera = {}
    for row in chosen_metrics:
        if row.get("time") not in time_set:
            continue
        if not metric_passes(row, args.min_tags):
            continue
        metrics_by_camera.setdefault(row.get("camera_id", ""), []).append(row)

    for new_index, row in enumerate(manifest_rows):
        camera_id = row.get("camera_id", row.get("user_id", ""))
        stage_name = row.get("stage_name") or f"cam{new_index:02d}_{camera_id}"
        dst_dir = images_root / stage_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        image_dirs.append(str(dst_dir))
        original_source_dir = row.get("source_dir", "")
        camera_metrics = sorted(
            metrics_by_camera.get(camera_id, []),
            key=lambda item: (
                natural_key(item.get("time", "")),
                frame_key(item.get("frame_id_text", item.get("frame_id", ""))),
            ))
        staged_count = 0
        for metric in camera_metrics:
            src = resolve_source(args, row, metric)
            if not src:
                missing_sources.append({
                    "camera_id": camera_id,
                    "time": metric.get("time", ""),
                    "frame_id": metric.get("frame_id_text", metric.get("frame_id", "")),
                    "filename": metric.get("filename", ""),
                    "reason": "source_not_found",
                    "candidates": ";".join(str(path) for path in source_candidates(args, row, metric)),
                })
                continue
            dst = dst_dir / f"{staged_count:06d}{args.output_extension}"
            link_image(src, dst, args.link_mode)
            selected_rows.append({
                "out_frame": staged_count,
                "time": metric.get("time", ""),
                "frame_id": metric.get("frame_id_text", metric.get("frame_id", "")),
                "frame_key": frame_item_id({
                    "time": metric.get("time", ""),
                    "frame_id": metric.get("frame_id_text", metric.get("frame_id", "")),
                }),
                "camera_index": new_index,
                "camera_id": camera_id,
                "tag_count": metric.get("tag_count", ""),
                "corner_count": metric.get("corner_count", ""),
                "source": str(src),
                "filtered_image": str(dst),
            })
            staged_count += 1
        output_row = dict(row)
        output_row["camera_index"] = new_index
        output_row["stage_name"] = stage_name
        output_row["source_dir"] = str(dst_dir)
        output_row["frame_count"] = staged_count
        output_row["original_camera_index"] = row.get("_original_camera_index", row.get("camera_index", new_index))
        output_row["original_source_dir"] = original_source_dir
        output_manifest.append(output_row)

    if missing_sources and not args.allow_missing_source:
        write_tsv(
            output_dir / "missing_sources.tsv",
            missing_sources,
            sorted({key for row in missing_sources for key in row.keys()}))
        raise SystemExit(
            f"Missing {len(missing_sources)} source images. See {output_dir / 'missing_sources.tsv'}")
    return image_dirs, output_manifest, selected_rows, missing_sources


def chosen_metric_rows(by_camera_frame, camera_ids, time_ids):
    camera_set = set(camera_ids)
    time_set = set(time_ids)
    rows = []
    for key in sorted(
            by_camera_frame,
            key=lambda item: (natural_key(item[1]), natural_key(item[0]), frame_key(item[2]))):
        camera_id, time_id, _frame_id = key
        if camera_id in camera_set and time_id in time_set:
            rows.append(by_camera_frame[key])
    return rows


def aggregate_camera_stats(chosen_metrics, manifest_rows, time_ids, min_tags, selected_frames):
    time_set = set(time_ids)
    selected_set = {frame_item_id(frame) for frame in selected_frames}
    rows = []
    for row in manifest_rows:
        camera_id = row.get("camera_id", row.get("user_id", ""))
        stats = init_stats("", ",".join(time_ids), camera_id, row.get("source_dir", ""))
        selected_passing = 0
        selected_observed = 0
        for metric in chosen_metrics:
            if metric.get("time") not in time_set or metric.get("camera_id") != camera_id:
                continue
            stat_row = dict(metric)
            stat_row["decode_ok"] = "1" if metric.get("decode_ok_bool") else "0"
            update_stats(stats, stat_row, min_tags)
            key = frame_item_id({"time": metric.get("time", ""), "frame_id": metric.get("frame_id_text", "")})
            if key in selected_set:
                selected_observed += 1
                if metric.get("decode_ok_bool") and metric.get("tag_count_int", 0) >= min_tags:
                    selected_passing += 1
        finalize_stats(stats)
        stats["selected_observed_frames"] = selected_observed
        stats["selected_passing_frames"] = selected_passing
        rows.append(stats)
    return rows


def run_aggregate(args):
    worker_outputs = []
    for item in args.worker_output:
        worker_outputs.extend(split_values(item))
    if not worker_outputs:
        raise SystemExit("Provide at least one --worker-output.")
    metrics = load_metric_rows(worker_outputs)
    time_ids, available_times = resolve_aggregate_times(args, metrics)

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    manifest_rows, missing_manifest_cameras = build_manifest_rows(args, metrics, time_ids)
    if args.require_all_base_manifest_cameras and missing_manifest_cameras:
        raise SystemExit(
            "Base manifest cameras are missing from worker metrics: "
            + ", ".join(missing_manifest_cameras))
    camera_ids = [row.get("camera_id", row.get("user_id", "")) for row in manifest_rows]
    by_camera_frame = group_metrics(metrics, time_ids, camera_ids)
    chosen_metrics = chosen_metric_rows(by_camera_frame, camera_ids, time_ids)
    selected_frames, frame_rows = select_frames(
        by_camera_frame,
        camera_ids,
        args.min_tags,
        args.min_cameras_per_frame,
        not args.allow_missing_camera_metrics)
    if args.stride > 1:
        selected_frames = selected_frames[::args.stride]
    if args.max_frames > 0:
        selected_frames = selected_frames[:args.max_frames]
    if not selected_frames and not args.allow_empty:
        write_tsv(
            output_dir / "frame_selection.tsv",
            frame_rows,
            ["time", "frame_id", "frame_key", "observed_camera_count", "passing_camera_count", "active_camera_count", "status", "selected"])
        raise SystemExit(
            f"No frames passed the aggregate gate. See {output_dir / 'frame_selection.tsv'}")

    if args.stage_mode == "passing-images":
        image_dirs, output_manifest, selected_image_rows, missing_sources = write_passing_image_dataset(
            args,
            output_dir,
            manifest_rows,
            chosen_metrics,
            time_ids)
    else:
        image_dirs, output_manifest, selected_image_rows, missing_sources = write_filtered_dataset(
            args,
            output_dir,
            manifest_rows,
            by_camera_frame,
            selected_frames)

    manifest_fields = [
        "camera_index", "stage_name", "machine", "camera_id", "source_dir",
        "frame_count", "original_camera_index", "original_source_dir",
    ]
    extra_fields = [
        key for key in sorted({key for row in output_manifest for key in row.keys()})
        if key not in manifest_fields and not key.startswith("_")
    ]
    write_tsv(output_dir / "manifest.tsv", output_manifest, manifest_fields + extra_fields)
    (output_dir / "image_directories.txt").write_text(
        ",".join(image_dirs) + "\n",
        encoding="utf-8")

    selected_frame_set = {frame_item_id(frame) for frame in selected_frames}
    selected_index_by_key = {
        frame_item_id(frame): index
        for index, frame in enumerate(selected_frames)
    }
    filtered_frame_rows = []
    for row in frame_rows:
        item = dict(row)
        item["selected"] = 1 if item["frame_key"] in selected_frame_set else 0
        if item["selected"]:
            item["out_frame"] = selected_index_by_key[item["frame_key"]]
            item["selected_filename"] = f"{item['out_frame']:06d}{args.output_extension}"
        else:
            item["out_frame"] = ""
            item["selected_filename"] = ""
        filtered_frame_rows.append(item)
    write_tsv(
        output_dir / "frame_selection.tsv",
        filtered_frame_rows,
        [
            "out_frame", "time", "frame_id", "frame_key", "selected_filename", "observed_camera_count",
            "passing_camera_count", "active_camera_count", "status", "selected",
        ])
    write_tsv(
        output_dir / "selected_frames.tsv",
        [row for row in filtered_frame_rows if parse_bool_text(row.get("selected"))],
        [
            "out_frame", "time", "frame_id", "frame_key", "selected_filename", "observed_camera_count",
            "passing_camera_count", "active_camera_count", "status", "selected",
        ])
    write_tsv(
        output_dir / "selected_images.tsv",
        selected_image_rows,
        [
            "out_frame", "time", "frame_id", "frame_key", "camera_index", "camera_id",
            "tag_count", "corner_count", "source", "filtered_image",
        ])

    passing_rows = []
    time_set = set(time_ids)
    for row in chosen_metrics:
        if row.get("time") not in time_set or row.get("camera_id") not in camera_ids:
            continue
        key = frame_item_id({"time": row.get("time", ""), "frame_id": row.get("frame_id_text", "")})
        if row.get("decode_ok_bool") and row.get("tag_count_int", 0) >= args.min_tags:
            passing_rows.append({
                "worker_id": row.get("worker_id", ""),
                "time": row.get("time", ""),
                "camera_id": row.get("camera_id", ""),
                "frame_id": row.get("frame_id_text", ""),
                "filename": row.get("filename", ""),
                "image_path": row.get("image_path", ""),
                "tag_count": row.get("tag_count", ""),
                "corner_count": row.get("corner_count", ""),
                "selected_frame": 1 if key in selected_frame_set else 0,
            })
    write_tsv(
        output_dir / f"images_min{args.min_tags}.tsv",
        passing_rows,
        [
            "worker_id", "time", "camera_id", "frame_id", "filename",
            "image_path", "tag_count", "corner_count", "selected_frame",
        ])

    camera_stats = aggregate_camera_stats(chosen_metrics, manifest_rows, time_ids, args.min_tags, selected_frames)
    write_tsv(
        output_dir / "per_camera_stats.tsv",
        camera_stats,
        [
            "worker_id", "time", "camera_id", "image_dir", "total_images",
            "decoded_images", "failed_images", "passing_images", "passing_ratio",
            "total_tags", "total_corners", "max_tags", "first_passing_frame_id",
            "last_passing_frame_id", "width", "height",
            "selected_observed_frames", "selected_passing_frames",
        ])

    candidate_passing_histogram = {}
    selected_passing_histogram = {}
    for row in filtered_frame_rows:
        passing_count = str(row.get("passing_camera_count", "0"))
        candidate_passing_histogram[passing_count] = candidate_passing_histogram.get(passing_count, 0) + 1
        if parse_bool_text(row.get("selected")):
            selected_passing_histogram[passing_count] = selected_passing_histogram.get(passing_count, 0) + 1

    summary = {
        "mode": "t0_aggregate_filtered_whole",
        "quality_validation_only": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "worker_outputs": worker_outputs,
        "times": time_ids,
        "available_times": available_times,
        "output_dir": str(output_dir),
        "camera_count": len(manifest_rows),
        "metric_row_count": len(metrics),
        "chosen_metric_row_count": len(chosen_metrics),
        "selected_frame_count": len(selected_frames),
        "candidate_frame_count": len(frame_rows),
        "passing_image_count": len(passing_rows),
        "candidate_passing_camera_count_histogram": candidate_passing_histogram,
        "selected_passing_camera_count_histogram": selected_passing_histogram,
        "missing_base_manifest_cameras": missing_manifest_cameras,
        "missing_source_count": len(missing_sources),
        "selection": {
            "stage_mode": args.stage_mode,
            "min_tags": args.min_tags,
            "min_cameras_per_frame": args.min_cameras_per_frame,
            "require_all_camera_metrics": not args.allow_missing_camera_metrics,
            "stride": args.stride,
            "max_frames": args.max_frames,
        },
        "outputs": {
            "manifest": str(output_dir / "manifest.tsv"),
            "image_directories": str(output_dir / "image_directories.txt"),
            "frame_selection": str(output_dir / "frame_selection.tsv"),
            "selected_frames": str(output_dir / "selected_frames.tsv"),
            "selected_images": str(output_dir / "selected_images.tsv"),
            "passing_images": str(output_dir / f"images_min{args.min_tags}.tsv"),
            "per_camera_stats": str(output_dir / "per_camera_stats.tsv"),
            "index_html": str(output_dir / "index.html"),
        },
        "next_step_hint": (
            "Use image_directories.txt with parallel_extract_features.py. "
            "For repo calibration-board data, pass --pattern-files. "
            "For tower data, pass --apriltag-tower-config. "
            "Use --stage-mode passing-images for per-camera intrinsic initialization, "
            "and the default synchronized frame staging for multi-camera rig calibration."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_html(
        output_dir / "index.html",
        "Distributed AprilTag Filtered Whole Dataset",
        summary,
        {
            "metrics": [
                ("cameras", len(manifest_rows)),
                ("stage mode", args.stage_mode),
                ("selected frames", len(selected_frames)),
                ("staged images", len(selected_image_rows)),
                (f"images >= {args.min_tags} tags", len(passing_rows)),
                ("candidate frames", len(frame_rows)),
            ],
            "tables": [
                {
                    "title": "Per-Camera Aggregate Stats",
                    "rows": camera_stats,
                    "columns": [
                        "time", "camera_id", "total_images", "passing_images",
                        "passing_ratio", "max_tags", "selected_observed_frames",
                        "selected_passing_frames",
                    ],
                    "limit": 200,
                },
                {
                    "title": "Selected Frames",
                    "rows": [row for row in filtered_frame_rows if parse_bool_text(row.get("selected"))],
                    "columns": [
                        "out_frame", "frame_id", "selected_filename",
                        "time", "observed_camera_count", "passing_camera_count",
                        "active_camera_count",
                    ],
                    "limit": 300,
                },
            ],
        })
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def add_detect_parser(subparsers):
    parser = subparsers.add_parser(
        "detect",
        help="Run on a Windows worker to count AprilTags under D:/output/calib/whole/<time>/<camera_id>.")
    parser.add_argument("--input-root", type=Path, default=Path("D:/output/calib/whole"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker-id", default="")
    parser.add_argument("--time", default="", help="Comma/newline separated time directory filter.")
    parser.add_argument("--camera-id", default="", help="Comma/newline separated camera-id filter.")
    parser.add_argument("--tower-config", type=Path, default=DEFAULT_TOWER_CONFIG)
    parser.add_argument("--dictionary", default="")
    parser.add_argument("--min-tags", type=int, default=4)
    parser.add_argument("--filter-tower-ids", action="store_true", default=True)
    parser.add_argument("--no-filter-tower-ids", dest="filter_tower_ids", action="store_false")
    parser.add_argument("--detect-inverted", action="store_true", default=True)
    parser.add_argument("--no-detect-inverted", dest="detect_inverted", action="store_false")
    parser.add_argument("--error-correction-rate", type=float, default=0.6)
    parser.add_argument(
        "--corner-refinement",
        choices=["none", "subpix", "contour", "apriltag"],
        default="subpix")
    parser.add_argument("--corner-refinement-window-size", type=int, default=5)
    parser.add_argument("--corner-refinement-max-iterations", type=int, default=30)
    parser.add_argument("--corner-refinement-min-accuracy", type=float, default=0.01)
    parser.add_argument("--subpixel-refine-original", action="store_true", default=True)
    parser.add_argument("--no-subpixel-refine-original", dest="subpixel_refine_original", action="store_false")
    parser.add_argument("--subpixel-window-size", type=int, default=5)
    parser.add_argument("--subpixel-max-iterations", type=int, default=30)
    parser.add_argument("--subpixel-epsilon", type=float, default=0.01)
    parser.add_argument("--edge-line-refine-original", action="store_true")
    parser.add_argument("--edge-line-search-radius-px", type=float, default=5.0)
    parser.add_argument("--edge-line-sample-spacing-px", type=float, default=2.0)
    parser.add_argument("--edge-line-gradient-step-px", type=float, default=1.0)
    parser.add_argument("--edge-line-min-gradient", type=float, default=2.0)
    parser.add_argument("--edge-line-min-edge-points", type=int, default=8)
    parser.add_argument("--edge-line-max-shift-px", type=float, default=4.0)
    parser.add_argument(
        "--edge-line-polarity",
        choices=["outside_white_inside_black", "outside_black_inside_white", "absolute"],
        default="outside_white_inside_black")
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip-detections-jsonl", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(func=run_detect)


def add_aggregate_parser(subparsers):
    parser = subparsers.add_parser(
        "aggregate",
        help="Run on t0 to merge worker metrics into a filtered whole-dir manifest/image_directories layout.")
    parser.add_argument("--worker-output", action="append", default=[])
    parser.add_argument("--base-manifest", type=Path, default=None)
    parser.add_argument("--require-all-base-manifest-cameras", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--time", default="")
    parser.add_argument("--mount-root", type=Path, default=Path("/home/ubuntu/cameras_mount"))
    parser.add_argument("--marker", default="whole")
    parser.add_argument("--min-tags", type=int, default=4)
    parser.add_argument("--min-cameras-per-frame", type=int, default=1)
    parser.add_argument("--allow-missing-camera-metrics", action="store_true")
    parser.add_argument("--allow-missing-source", action="store_true")
    parser.add_argument("--no-check-source-exists", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--stage-mode",
        choices=["frames", "passing-images"],
        default="frames",
        help=(
            "frames keeps synchronized selected frames across all cameras. "
            "passing-images stages only per-camera images that pass --min-tags; "
            "use it for fast per-camera intrinsic initialization."
        ))
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--output-extension", default=".jpg")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=run_aggregate)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Distributed AprilTag tower quality filtering for worker/t0 calibration workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_detect_parser(subparsers)
    add_aggregate_parser(subparsers)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if hasattr(args, "output_extension") and not args.output_extension.startswith("."):
        args.output_extension = "." + args.output_extension
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
