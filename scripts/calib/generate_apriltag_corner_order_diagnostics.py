#!/usr/bin/env python3
"""Render AprilTag corner-index overlays for tower model debugging."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
import sys

import distributed_apriltag_quality_filter as qc


DEFAULT_HTTP_ROOT = Path("/home/ubuntu/calib_data")
DEFAULT_HTTP_BASE = "http://192.168.2.0:9899"


def report_url(path, http_root, http_base):
    path = Path(path).resolve()
    try:
        rel = path.relative_to(Path(http_root).resolve())
        return f"{http_base.rstrip('/')}/{rel.as_posix()}"
    except ValueError:
        return path.as_uri()


def read_selected_images(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def choose_images(rows, camera_ids, limit_per_camera):
    selected = []
    for camera_id in camera_ids:
        candidates = [
            row for row in rows
            if row.get("camera_id") == camera_id and row.get("filtered_image")
        ]
        candidates.sort(
            key=lambda row: (
                -int(row.get("tag_count") or 0),
                row.get("time", ""),
                int(row.get("frame_id") or 0),
            ))
        selected.extend(candidates[:limit_per_camera])
    return selected


def draw_overlay(cv2, image, detections):
    colors = [
        (0, 0, 255),      # red
        (0, 180, 255),    # orange
        (0, 255, 0),      # green
        (255, 0, 0),      # blue
    ]
    canvas = image.copy()
    for det in detections:
        pts = [tuple(int(round(v)) for v in point) for point in det["corners"]]
        for corner_index, point in enumerate(pts):
            cv2.circle(canvas, point, 10, colors[corner_index], -1)
            cv2.putText(
                canvas,
                str(corner_index),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colors[corner_index],
                2,
                cv2.LINE_AA,
            )
        for corner_index in range(4):
            cv2.line(canvas, pts[corner_index], pts[(corner_index + 1) % 4], (255, 255, 255), 2)
        center_x = int(round(sum(point[0] for point in pts) / 4.0))
        center_y = int(round(sum(point[1] for point in pts) / 4.0))
        cv2.putText(
            canvas,
            str(det["tag_id"]),
            (center_x - 18, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def run(args):
    cv2 = qc.load_cv2()
    config = qc.parse_simple_yaml(args.tower_config)
    valid_ids = qc.tower_valid_tag_ids(config)
    dictionary_name = qc.dictionary_from_config(config, args.dictionary)
    detector = qc.create_detector(cv2, dictionary_name, args.detect_inverted, args.error_correction_rate)
    rows = read_selected_images(args.selected_images_tsv)
    selected = choose_images(rows, args.camera_id, args.limit_per_camera)
    if not selected:
        raise SystemExit("No selected images matched requested camera ids.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for row in selected:
        image_path = Path(row["filtered_image"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rendered.append({**row, "status": "missing_or_undecodable", "overlay": ""})
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections, rejected_count = qc.detect_markers(cv2, detector, gray)
        detections = [det for det in detections if det["tag_id"] in valid_ids]
        canvas = draw_overlay(cv2, image, detections)
        scale = args.max_display_width / max(1, canvas.shape[1])
        if scale < 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        output_name = (
            f"{row['camera_id'].replace('-', '_')}_"
            f"{row.get('time', '').replace(':', '_')}_"
            f"{int(row.get('frame_id') or 0):04d}.jpg"
        )
        output_path = args.output_dir / output_name
        cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        rendered.append({
            **row,
            "status": "ok",
            "overlay": output_name,
            "detected_tag_count": str(len(detections)),
            "rejected_count": str(rejected_count),
        })

    index = args.output_dir / "index.html"
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>AprilTag Tower Corner Order Diagnostics</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}",
        "img{max-width:100%;border:1px solid #444;margin:12px 0 28px}",
        "code{color:#9cf} .muted{color:#aaa}",
        "</style></head><body>",
        "<h1>AprilTag Tower Corner Order Diagnostics</h1>",
        (
            "<p>Corner colors: <b style='color:#f66'>0 red</b>, "
            "<b style='color:#fb3'>1 orange</b>, "
            "<b style='color:#4f4'>2 green</b>, "
            "<b style='color:#59f'>3 blue</b>. Tag id is drawn near the center.</p>"
        ),
        f"<p class='muted'><code>{html.escape(str(args.tower_config))}</code></p>",
    ]
    for row in rendered:
        title = (
            f"{row.get('camera_id', '')} | {row.get('time', '')} | "
            f"frame {row.get('frame_id', '')} | tags {row.get('detected_tag_count', row.get('tag_count', ''))}"
        )
        body.append(f"<h2>{html.escape(title)}</h2>")
        body.append(f"<p class='muted'><code>{html.escape(row.get('filtered_image', ''))}</code></p>")
        if row.get("overlay"):
            body.append(f"<img src='{html.escape(row['overlay'])}'>")
        else:
            body.append(f"<p>{html.escape(row.get('status', 'missing'))}</p>")
    body.append("</body></html>")
    index.write_text("\n".join(body), encoding="utf-8")
    print(index)
    print(report_url(index, args.http_root, args.http_base))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-images-tsv", required=True, type=Path)
    parser.add_argument("--tower-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-id", action="append", required=True)
    parser.add_argument("--limit-per-camera", type=int, default=1)
    parser.add_argument("--dictionary")
    parser.add_argument("--detect-inverted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--error-correction-rate", type=float, default=0.6)
    parser.add_argument("--max-display-width", type=float, default=1200.0)
    parser.add_argument("--http-root", type=Path, default=DEFAULT_HTTP_ROOT)
    parser.add_argument("--http-base", default=DEFAULT_HTTP_BASE)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
