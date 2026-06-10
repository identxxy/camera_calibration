#!/usr/bin/env python3
"""Generate a per-camera AprilTag timeline report for whole/tower captures."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

try:
    from export_calibration_correspondence_residuals import parse_frame_index, read_dataset
except ModuleNotFoundError:
    from scripts.calib.export_calibration_correspondence_residuals import parse_frame_index, read_dataset


def esc(value):
    return html.escape(str(value))


def finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def read_manifest(path):
    path = Path(path) if path else None
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {
            int(row["camera_index"]): row
            for row in csv.DictReader(stream, delimiter="\t")
            if row.get("camera_index", "").strip().isdigit()
        }


def camera_label(index, manifest):
    row = manifest.get(index) or {}
    for key in ("camera_id", "user_id", "stage_name"):
        value = row.get(key)
        if value:
            return str(value)
    return f"cam{index:02d}"


def read_residual_counts(path):
    path = Path(path) if path else None
    if path is None or not path.is_file():
        return {}
    stats = defaultdict(lambda: {"accepted_corners": 0, "frames": set(), "residuals": []})
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if (row.get("projection_status") or "").lower() != "ok":
                continue
            label = row.get("camera_id") or row.get("camera_label") or row.get("camera_index") or ""
            frame = row.get("frame_index")
            bucket = stats[str(label)]
            bucket["accepted_corners"] += 1
            if frame not in (None, ""):
                try:
                    bucket["frames"].add(int(frame))
                except ValueError:
                    pass
            residual = finite_float(row.get("residual_px"))
            if residual is not None:
                bucket["residuals"].append(residual)
    result = {}
    for label, bucket in stats.items():
        residuals = sorted(bucket["residuals"])
        median = residuals[len(residuals) // 2] if residuals else None
        result[label] = {
            "accepted_corners": bucket["accepted_corners"],
            "accepted_frames": len(bucket["frames"]),
            "accepted_median_px": median,
        }
    return result


def color_for_frame(frame, min_frame, max_frame):
    span = max(1, max_frame - min_frame)
    t = max(0.0, min(1.0, (frame - min_frame) / span))
    hue = 220.0 - 210.0 * t
    return f"hsl({hue:.1f} 85% 48%)"


def decimate(items, max_items):
    if len(items) <= max_items:
        return items
    stride = max(1, int(math.ceil(len(items) / max_items)))
    return items[::stride]


def summarize_dataset(dataset, manifest, residual_counts):
    cameras = []
    global_frames = set()
    frame_covis = defaultdict(lambda: {
        "cameras": set(),
        "tag_to_cameras": defaultdict(set),
    })
    min_frame = None
    max_frame = None
    for index in range(dataset["camera_count"]):
        width, height = dataset["image_sizes"][index]
        cameras.append({
            "index": index,
            "label": camera_label(index, manifest),
            "width": width,
            "height": height,
            "frames": {},
            "tag_center_samples": [],
            "mean_samples": [],
            "total_tags": 0,
            "total_corners": 0,
            "max_tags": 0,
        })

    for imageset in dataset["imagesets"]:
        frame = int(parse_frame_index(imageset["filename"], imageset["index"]))
        global_frames.add(frame)
        min_frame = frame if min_frame is None else min(min_frame, frame)
        max_frame = frame if max_frame is None else max(max_frame, frame)
        for camera_index, features in enumerate(imageset["features"]):
            if not features:
                continue
            camera = cameras[camera_index]
            tags = defaultdict(list)
            for feature in features:
                feature_id = int(feature["feature_id"])
                tag_id = feature_id // 4
                tags[tag_id].append((float(feature["x"]), float(feature["y"])))
            tag_centers = []
            for tag_id, corners in tags.items():
                x = sum(point[0] for point in corners) / len(corners)
                y = sum(point[1] for point in corners) / len(corners)
                tag_centers.append((tag_id, x, y, len(corners)))
                camera["tag_center_samples"].append({
                    "frame": frame,
                    "tag_id": tag_id,
                    "x": x,
                    "y": y,
                    "corner_count": len(corners),
                })
                frame_covis[frame]["tag_to_cameras"][tag_id].add(camera["label"])
            mean_x = sum(item[1] for item in tag_centers) / len(tag_centers)
            mean_y = sum(item[2] for item in tag_centers) / len(tag_centers)
            camera["frames"][frame] = {
                "tag_count": len(tags),
                "corner_count": len(features),
                "mean_x": mean_x,
                "mean_y": mean_y,
                "tag_ids": sorted(tags),
            }
            camera["mean_samples"].append((frame, mean_x, mean_y, len(tags), len(features)))
            camera["total_tags"] += len(tags)
            camera["total_corners"] += len(features)
            camera["max_tags"] = max(camera["max_tags"], len(tags))
            frame_covis[frame]["cameras"].add(camera["label"])

    if min_frame is None:
        min_frame = 0
        max_frame = 0
    all_frames = list(range(min_frame, max_frame + 1))
    covis_rows = []
    for frame in all_frames:
        item = frame_covis.get(frame)
        if not item:
            covis_rows.append({
                "frame": frame,
                "camera_count": 0,
                "tag_count": 0,
                "shared_tag_count": 0,
                "cameras": [],
            })
            continue
        tag_to_cameras = item["tag_to_cameras"]
        covis_rows.append({
            "frame": frame,
            "camera_count": len(item["cameras"]),
            "tag_count": len(tag_to_cameras),
            "shared_tag_count": sum(1 for cams in tag_to_cameras.values() if len(cams) >= 2),
            "cameras": sorted(item["cameras"]),
        })

    for camera in cameras:
        frames = sorted(camera["frames"])
        camera["frames_with_tags"] = len(frames)
        camera["first_frame"] = frames[0] if frames else None
        camera["last_frame"] = frames[-1] if frames else None
        camera["residual"] = residual_counts.get(camera["label"], {})

    return {
        "camera_count": len(cameras),
        "frame_range": [min_frame, max_frame],
        "frame_count": len(all_frames),
        "frames_with_any_tag": sum(1 for row in covis_rows if row["camera_count"] > 0),
        "max_cameras_per_frame": max((row["camera_count"] for row in covis_rows), default=0),
        "max_shared_tags_per_frame": max((row["shared_tag_count"] for row in covis_rows), default=0),
        "cameras": cameras,
        "covis_rows": covis_rows,
        "top_covis_frames": sorted(
            covis_rows,
            key=lambda row: (-row["camera_count"], -row["shared_tag_count"], row["frame"]),
        )[:24],
    }


def svg_global_covis(covis_rows, min_frame, max_frame):
    width = 1040
    height = 170
    pad_l, pad_r, pad_t, pad_b = 44, 12, 18, 26
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_camera = max(1, max(row["camera_count"] for row in covis_rows))
    max_shared = max(1, max(row["shared_tag_count"] for row in covis_rows))

    def sx(frame):
        return pad_l + (frame - min_frame) / max(1, max_frame - min_frame) * plot_w

    def sy_count(value, max_value):
        return pad_t + plot_h - value / max_value * plot_h

    shared_points = []
    bars = []
    for row in covis_rows:
        x = sx(row["frame"])
        if row["camera_count"]:
            bars.append(
                f'<rect x="{x:.2f}" y="{sy_count(row["camera_count"], max_camera):.2f}" '
                f'width="1.4" height="{plot_h - (sy_count(row["camera_count"], max_camera) - pad_t):.2f}" '
                'fill="#2563eb" opacity="0.42"/>')
        shared_points.append(f'{x:.2f},{sy_count(row["shared_tag_count"], max_shared):.2f}')
    return f"""
<svg viewBox="0 0 {width} {height}" class="wide-plot" role="img" aria-label="Global co-visibility timeline">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="#94a3b8"/>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#94a3b8"/>
  {''.join(bars)}
  <polyline points="{' '.join(shared_points)}" fill="none" stroke="#16a34a" stroke-width="2"/>
  <text x="{pad_l}" y="14" class="svg-label">blue bars: cameras with tags per frame; green line: tag IDs seen by >=2 cameras</text>
  <text x="{pad_l}" y="{height - 6}" class="svg-label">frame {min_frame}</text>
  <text x="{width - 92}" y="{height - 6}" class="svg-label">frame {max_frame}</text>
</svg>"""


def svg_count_timeline(camera, min_frame, max_frame):
    width = 440
    height = 76
    pad_l, pad_r, pad_t, pad_b = 28, 8, 10, 16
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_tags = max(1, camera["max_tags"])

    def sx(frame):
        return pad_l + (frame - min_frame) / max(1, max_frame - min_frame) * plot_w

    bars = []
    for frame, item in camera["frames"].items():
        h = item["tag_count"] / max_tags * plot_h
        x = sx(frame)
        bars.append(
            f'<rect x="{x:.2f}" y="{pad_t + plot_h - h:.2f}" width="1.4" height="{h:.2f}" '
            'fill="#2563eb" opacity="0.58"/>')
    return f"""
<svg viewBox="0 0 {width} {height}" class="mini-plot" role="img" aria-label="tag count timeline">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="#cbd5e1"/>
  {''.join(bars)}
  <text x="{pad_l}" y="9" class="svg-label">tag count over original frame index</text>
</svg>"""


def svg_xy_over_time(camera, min_frame, max_frame, value_index):
    width = 440
    height = 86
    pad_l, pad_r, pad_t, pad_b = 28, 8, 10, 16
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    image_limit = camera["width"] if value_index == 1 else camera["height"]
    label = "mean tag image x over time" if value_index == 1 else "mean tag image y over time"
    points = []
    for frame, mean_x, mean_y, _tags, _corners in camera["mean_samples"]:
        value = mean_x if value_index == 1 else mean_y
        x = pad_l + (frame - min_frame) / max(1, max_frame - min_frame) * plot_w
        y = pad_t + value / max(1, image_limit) * plot_h
        points.append(f"{x:.2f},{y:.2f}")
    polyline = f'<polyline points="{" ".join(points)}" fill="none" stroke="#dc2626" stroke-width="1.5" opacity="0.78"/>' if points else ""
    return f"""
<svg viewBox="0 0 {width} {height}" class="mini-plot" role="img" aria-label="{label}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#cbd5e1"/>
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="#cbd5e1"/>
  {polyline}
  <text x="{pad_l}" y="9" class="svg-label">{label}</text>
</svg>"""


def svg_image_footprint(camera, min_frame, max_frame):
    width = 220
    height = 170
    pad = 10
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    samples = decimate(camera["tag_center_samples"], 1600)
    points = []
    for sample in samples:
        x = pad + sample["x"] / max(1, camera["width"]) * plot_w
        y = pad + sample["y"] / max(1, camera["height"]) * plot_h
        color = color_for_frame(sample["frame"], min_frame, max_frame)
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.35" fill="{color}" opacity="0.58"/>')
    return f"""
<svg viewBox="0 0 {width} {height}" class="image-plot" role="img" aria-label="image footprint">
  <rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#94a3b8"/>
  {''.join(points)}
  <text x="{pad}" y="{height - 2}" class="svg-label">image footprint, color=time</text>
</svg>"""


def fmt(value, digits=2):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def render_report(data, output_dir, title, dataset_path, manifest_path, residual_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    min_frame, max_frame = data["frame_range"]
    rows = []
    for camera in data["cameras"]:
        residual = camera.get("residual") or {}
        rows.append(f"""
<section class="camera-card">
  <div class="camera-head">
    <h2>{esc(camera["label"])}</h2>
    <div class="meta">cam {camera["index"]} · {camera["width"]}x{camera["height"]} · frames {esc(camera["first_frame"])}..{esc(camera["last_frame"])}</div>
  </div>
  <div class="stats">
    <span><b>{camera["frames_with_tags"]}</b> frames</span>
    <span><b>{camera["total_tags"]}</b> tag views</span>
    <span><b>{camera["total_corners"]}</b> corners</span>
    <span><b>{camera["max_tags"]}</b> max tags/frame</span>
    <span><b>{residual.get("accepted_corners", 0)}</b> BA corners</span>
    <span><b>{fmt(residual.get("accepted_median_px"))}</b> BA med px</span>
  </div>
  <div class="plots">
    <div>{svg_count_timeline(camera, min_frame, max_frame)}{svg_xy_over_time(camera, min_frame, max_frame, 1)}{svg_xy_over_time(camera, min_frame, max_frame, 2)}</div>
    {svg_image_footprint(camera, min_frame, max_frame)}
  </div>
</section>""")

    camera_table = []
    for camera in data["cameras"]:
        residual = camera.get("residual") or {}
        camera_table.append(f"""
<tr>
  <td>{esc(camera["label"])}</td>
  <td>{camera["index"]}</td>
  <td>{camera["frames_with_tags"]}</td>
  <td>{camera["total_tags"]}</td>
  <td>{camera["max_tags"]}</td>
  <td>{esc(camera["first_frame"])}</td>
  <td>{esc(camera["last_frame"])}</td>
  <td>{residual.get("accepted_corners", 0)}</td>
  <td>{fmt(residual.get("accepted_median_px"))}</td>
</tr>""")

    top_rows = []
    for item in data["top_covis_frames"]:
        top_rows.append(f"""
<tr>
  <td>{item["frame"]}</td>
  <td>{item["camera_count"]}</td>
  <td>{item["tag_count"]}</td>
  <td>{item["shared_tag_count"]}</td>
  <td>{esc(", ".join(item["cameras"][:16]))}</td>
</tr>""")

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #0f172a; background: #f8fafc; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    code {{ background: #e2e8f0; padding: 1px 4px; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; margin: 14px 0 24px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 7px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #e2e8f0; font-weight: 700; }}
    .muted {{ color: #475569; font-size: 13px; line-height: 1.45; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; margin: 16px 0; }}
    .metric {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; font-size: 23px; }}
    .camera-grid {{ display: grid; grid-template-columns: repeat(2, minmax(560px, 1fr)); gap: 14px; }}
    .camera-card {{ background: #ffffff; border: 1px solid #dbe3ef; border-radius: 8px; padding: 12px; }}
    .camera-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }}
    .meta {{ color: #64748b; font-size: 12px; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }}
    .stats span {{ background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 999px; padding: 3px 8px; font-size: 12px; }}
    .plots {{ display: grid; grid-template-columns: minmax(420px, 1fr) 220px; gap: 10px; align-items: start; }}
    .mini-plot {{ display: block; width: 100%; height: 74px; margin-bottom: 4px; border: 1px solid #e2e8f0; }}
    .image-plot {{ width: 220px; height: 170px; border: 1px solid #e2e8f0; }}
    .wide-plot {{ width: 100%; max-width: 1120px; border: 1px solid #dbe3ef; background: #ffffff; }}
    .svg-label {{ font: 10px system-ui, sans-serif; fill: #475569; }}
    @media (max-width: 1180px) {{ .camera-grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <p class="muted">Source dataset: <code>{esc(dataset_path)}</code><br>
  Manifest: <code>{esc(manifest_path or "")}</code><br>
  BA residual TSV: <code>{esc(residual_path or "")}</code></p>
  <p class="muted">Semantics: raw AprilTag detections are per-camera observations. A multi-view tag correspondence exists only when the same original frame and tag id is seen by two or more cameras. The per-camera plots below use raw detected tag centers from the calib_data binary; BA columns use the final accepted/reprojected residual TSV when available.</p>

  <div class="metrics">
    <div class="metric"><b>{data["camera_count"]}</b><span>cameras in dataset</span></div>
    <div class="metric"><b>{data["frame_range"][0]}..{data["frame_range"][1]}</b><span>original frame range</span></div>
    <div class="metric"><b>{data["frames_with_any_tag"]}</b><span>frames with any tag</span></div>
    <div class="metric"><b>{data["max_cameras_per_frame"]}</b><span>max cameras with tags/frame</span></div>
    <div class="metric"><b>{data["max_shared_tags_per_frame"]}</b><span>max shared tag ids/frame</span></div>
  </div>

  <h2>Global Co-Visibility Timeline</h2>
  {svg_global_covis(data["covis_rows"], min_frame, max_frame)}

  <h2>Top Co-Visible Frames</h2>
  <table>
    <thead><tr><th>frame</th><th>cameras with tags</th><th>unique tag ids</th><th>tag ids seen by >=2 cameras</th><th>cameras</th></tr></thead>
    <tbody>{''.join(top_rows)}</tbody>
  </table>

  <h2>Camera Summary</h2>
  <table>
    <thead><tr><th>camera</th><th>index</th><th>raw frames</th><th>raw tag views</th><th>max tags/frame</th><th>first</th><th>last</th><th>BA corners</th><th>BA median px</th></tr></thead>
    <tbody>{''.join(camera_table)}</tbody>
  </table>

  <h2>Per-Camera Tag Timeline And Image Position</h2>
  <div class="camera-grid">
    {''.join(rows)}
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    summary = {
        key: value
        for key, value in data.items()
        if key not in ("cameras", "covis_rows")
    }
    summary["camera_summary"] = [
        {
            "index": camera["index"],
            "label": camera["label"],
            "frames_with_tags": camera["frames_with_tags"],
            "total_tags": camera["total_tags"],
            "total_corners": camera["total_corners"],
            "max_tags": camera["max_tags"],
            "first_frame": camera["first_frame"],
            "last_frame": camera["last_frame"],
            "residual": camera.get("residual", {}),
        }
        for camera in data["cameras"]
    ]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir / "index.html"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot whole AprilTag raw detections and tag image positions over original frame index.")
    parser.add_argument("--dataset", required=True, help="opencv_tower_dataset_fullres.bin or compatible calib_data file.")
    parser.add_argument("--manifest", default="", help="Staged whole manifest.tsv for camera labels.")
    parser.add_argument("--observation-residuals-tsv", default="", help="Optional final BA observation_residuals.tsv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Whole AprilTag Per-Camera Timeline Report")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = read_dataset(args.dataset)
    manifest = read_manifest(args.manifest)
    residual_counts = read_residual_counts(args.observation_residuals_tsv)
    data = summarize_dataset(dataset, manifest, residual_counts)
    index = render_report(data, args.output_dir, args.title, args.dataset, args.manifest, args.observation_residuals_tsv)
    print(index)


if __name__ == "__main__":
    main()
