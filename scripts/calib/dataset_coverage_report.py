#!/usr/bin/env python3
"""Generate an early coverage gate report from a camera_calibration dataset."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import struct


def read_exact(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise EOFError("Unexpected end of dataset file")
    return data


def read_u32(stream):
    return struct.unpack(">I", read_exact(stream, 4))[0]


def read_i32(stream):
    return struct.unpack(">i", read_exact(stream, 4))[0]


def read_f32(stream):
    return struct.unpack("<f", read_exact(stream, 4))[0]


def read_dataset(path):
    with Path(path).open("rb") as stream:
        header = read_exact(stream, 10)
        if header != b"calib_data":
            raise ValueError(f"Invalid dataset header: {path}")
        version = read_u32(stream)
        if version not in (0, 1):
            raise ValueError(f"Unsupported dataset version {version}: {path}")

        camera_count = read_u32(stream)
        image_sizes = [(read_u32(stream), read_u32(stream)) for _ in range(camera_count)]

        imagesets = []
        imageset_count = read_u32(stream)
        for _ in range(imageset_count):
            name_len = read_u32(stream)
            filename = read_exact(stream, name_len).decode("utf-8")
            camera_features = []
            for _camera in range(camera_count):
                features = []
                feature_count = read_u32(stream)
                for _feature in range(feature_count):
                    x = read_f32(stream)
                    y = read_f32(stream)
                    feature_id = read_i32(stream)
                    features.append((x, y, feature_id))
                camera_features.append(features)
            imagesets.append({"filename": filename, "features": camera_features})

        known_geometries = []
        known_geometry_count = read_u32(stream)
        for _ in range(known_geometry_count):
            cell_length = read_f32(stream)
            pos2d_count = read_u32(stream)
            for _item in range(pos2d_count):
                read_i32(stream)
                read_i32(stream)
                read_i32(stream)
            pos3d_count = 0
            if version >= 1:
                pos3d_count = read_u32(stream)
                for _item in range(pos3d_count):
                    read_i32(stream)
                    read_f32(stream)
                    read_f32(stream)
                    read_f32(stream)
            known_geometries.append({
                "cell_length_m": cell_length,
                "feature_count_2d": pos2d_count,
                "feature_count_3d": pos3d_count,
            })

    return {
        "camera_count": camera_count,
        "image_sizes": image_sizes,
        "imagesets": imagesets,
        "known_geometries": known_geometries,
    }


def read_manifest(path, camera_count):
    entries = []
    if not path:
        for index in range(camera_count):
            entries.append({
                "camera_index": str(index),
                "stage_name": f"camera_{index:02d}",
                "machine": "",
                "user_id": str(index),
            })
        return entries

    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            entries.append({
                "camera_index": row.get("camera_index", row.get("index", "")),
                "stage_name": row.get("stage_name", ""),
                "machine": row.get("machine", ""),
                "user_id": row.get("user_id", row.get("camera_id", "")),
            })
    if len(entries) != camera_count:
        raise ValueError(
            f"Manifest has {len(entries)} rows, but dataset has {camera_count} cameras")
    return entries


def group_count(features, group_size):
    return len({feature_id // group_size for _x, _y, feature_id in features})


def summarize_camera(dataset, manifest, args, camera_index):
    imagesets = dataset["imagesets"]
    width, height = dataset["image_sizes"][camera_index]
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    positive_views = 0
    usable_views = 0
    total_features = 0
    usable_features = 0
    max_features = 0
    max_groups = 0
    positive_runs = 0
    current_run = 0
    max_positive_run = 0
    first_positive = ""
    last_positive = ""

    for frame_index, imageset in enumerate(imagesets):
        features = imageset["features"][camera_index]
        feature_count = len(features)
        groups = group_count(features, args.group_size)
        total_features += feature_count
        max_features = max(max_features, feature_count)
        max_groups = max(max_groups, groups)

        if feature_count:
            positive_views += 1
            if first_positive == "":
                first_positive = frame_index
            last_positive = frame_index
            current_run += 1
            max_positive_run = max(max_positive_run, current_run)
            for x, y, _feature_id in features:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        else:
            if current_run:
                positive_runs += 1
            current_run = 0

        if feature_count >= args.min_features_per_view and groups >= args.min_tags_per_view:
            usable_views += 1
            usable_features += feature_count

    if current_run:
        positive_runs += 1

    bbox_area_ratio = 0.0
    if math.isfinite(min_x) and width > 0 and height > 0:
        bbox_area_ratio = max(0.0, max_x - min_x) * max(0.0, max_y - min_y) / (width * height)

    reasons = []
    if usable_views < args.min_usable_views:
        reasons.append(f"usable_views<{args.min_usable_views}")
    if usable_features < args.min_usable_points:
        reasons.append(f"usable_points<{args.min_usable_points}")
    status = "red" if reasons else "green"

    if status == "green":
        if positive_views < args.target_positive_frames:
            reasons.append(f"positive_views<{args.target_positive_frames}")
        if max_groups < args.target_max_tags:
            reasons.append(f"max_tags<{args.target_max_tags}")
        if bbox_area_ratio < args.min_bbox_area:
            reasons.append(f"bbox_area_ratio<{args.min_bbox_area}")
        if reasons:
            status = "yellow"

    entry = manifest[camera_index]
    return {
        "camera_index": camera_index,
        "stage_name": entry["stage_name"],
        "machine": entry["machine"],
        "user_id": entry["user_id"],
        "status": status,
        "reason": ";".join(reasons),
        "width": width,
        "height": height,
        "total_views": len(imagesets),
        "positive_views": positive_views,
        "positive_ratio": positive_views / len(imagesets) if imagesets else 0.0,
        "usable_views": usable_views,
        "total_features": total_features,
        "usable_features": usable_features,
        "total_tags": total_features // args.group_size,
        "max_features": max_features,
        "max_tags": max_groups,
        "bbox_area_ratio": bbox_area_ratio,
        "max_positive_run": max_positive_run,
        "positive_runs": positive_runs,
        "first_positive_frame": first_positive,
        "last_positive_frame": last_positive,
    }


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def frame_histograms(dataset, args):
    rows = []
    for min_tags in args.hist_tag_thresholds:
        counts = {}
        for imageset in dataset["imagesets"]:
            visible = 0
            for features in imageset["features"]:
                if group_count(features, args.group_size) >= min_tags:
                    visible += 1
            counts[visible] = counts.get(visible, 0) + 1
        for camera_count, frame_count in sorted(counts.items()):
            rows.append({
                "min_tags_per_camera": min_tags,
                "visible_camera_count": camera_count,
                "frame_count": frame_count,
            })
    return rows


def pairwise_covisibility(dataset, manifest, args):
    camera_count = dataset["camera_count"]
    counts = [[0 for _ in range(camera_count)] for _ in range(camera_count)]
    for imageset in dataset["imagesets"]:
        visible = [
            group_count(features, args.group_size) >= args.pair_min_tags
            for features in imageset["features"]
        ]
        for i in range(camera_count):
            if not visible[i]:
                continue
            for j in range(i + 1, camera_count):
                if visible[j]:
                    counts[i][j] += 1

    rows = []
    for i in range(camera_count):
        for j in range(i + 1, camera_count):
            rows.append({
                "camera_i": i,
                "user_i": manifest[i]["user_id"],
                "camera_j": j,
                "user_j": manifest[j]["user_id"],
                "shared_frames": counts[i][j],
            })
    return rows


def fmt(value, digits=4):
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def write_html(path, summary, camera_rows):
    counts = summary["status_counts"]
    cards = []
    for row in camera_rows:
        cards.append(f"""
<div class="card {html.escape(row['status'])}">
  <strong>{html.escape(str(row['user_id']))}</strong>
  <span>{html.escape(str(row['stage_name']))}</span>
  <b>{html.escape(row['status'])}</b>
  <small>views {row['usable_views']} / pts {row['usable_features']} / tags {row['total_tags']}</small>
</div>""")

    table = []
    for row in camera_rows:
        table.append("<tr>" + "".join(
            f"<td>{html.escape(str(fmt(row[key])))}</td>"
            for key in [
                "camera_index", "user_id", "stage_name", "status", "reason",
                "positive_views", "usable_views", "usable_features",
                "max_tags", "bbox_area_ratio",
            ]) + "</tr>")

    Path(path).write_text(f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Calibration Coverage Gate</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #17202a; font-family: Inter, system-ui, sans-serif; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p {{ color: #667085; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
    .metric, .panel {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    .metric strong {{ display: block; font-size: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; }}
    .card {{ border: 1px solid #d9dee7; border-radius: 8px; padding: 10px; min-height: 96px; background: #fff; }}
    .card span, .card small {{ display: block; color: #667085; font-size: 12px; }}
    .card b {{ display: inline-block; margin: 8px 0; font-size: 12px; text-transform: uppercase; }}
    .green {{ background: #e7f5ee; border-color: #9ed7bd; }}
    .yellow {{ background: #fff4db; border-color: #efcc82; }}
    .red {{ background: #fdeaea; border-color: #ecaaa8; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #fff; }}
    th, td {{ border-bottom: 1px solid #d9dee7; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #eef1f5; }}
    @media (max-width: 900px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} .grid {{ grid-template-columns: repeat(3, 1fr); }} }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Calibration Coverage Gate</h1>
  <p>{html.escape(summary['dataset'])}</p>
  <div class="metrics">
    <div class="metric"><strong>{summary['camera_count']}</strong> cameras</div>
    <div class="metric"><strong>{summary['imageset_count']}</strong> non-empty frames</div>
    <div class="metric green"><strong>{counts.get('green', 0)}</strong> green</div>
    <div class="metric red"><strong>{counts.get('red', 0)}</strong> red</div>
  </div>
  <div class="grid">{''.join(cards)}</div>
  <div class="panel">
    <table>
      <thead><tr><th>idx</th><th>user</th><th>stage</th><th>status</th><th>reason</th><th>positive</th><th>usable</th><th>points</th><th>max tags</th><th>bbox</th></tr></thead>
      <tbody>{''.join(table)}</tbody>
    </table>
  </div>
</div>
</body>
</html>
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--min-features-per-view", type=int, default=4)
    parser.add_argument("--min-tags-per-view", type=int, default=1)
    parser.add_argument("--min-usable-views", type=int, default=10)
    parser.add_argument("--min-usable-points", type=int, default=80)
    parser.add_argument("--target-positive-frames", type=int, default=30)
    parser.add_argument("--target-max-tags", type=int, default=3)
    parser.add_argument("--min-bbox-area", type=float, default=0.05)
    parser.add_argument("--pair-min-tags", type=int, default=1)
    parser.add_argument("--hist-tag-thresholds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    dataset = read_dataset(args.dataset)
    manifest = read_manifest(args.manifest, dataset["camera_count"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera_rows = [
        summarize_camera(dataset, manifest, args, camera_index)
        for camera_index in range(dataset["camera_count"])
    ]
    status_counts = {}
    for row in camera_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    camera_fields = [
        "camera_index", "stage_name", "machine", "user_id", "status", "reason",
        "width", "height", "total_views", "positive_views", "positive_ratio",
        "usable_views", "total_features", "usable_features", "total_tags",
        "max_features", "max_tags", "bbox_area_ratio", "max_positive_run",
        "positive_runs", "first_positive_frame", "last_positive_frame",
    ]
    write_tsv(args.output_dir / "coverage.tsv", camera_rows, camera_fields)
    write_tsv(
        args.output_dir / "insufficient_cameras.tsv",
        [row for row in camera_rows if row["status"] == "red"],
        camera_fields)

    hist_rows = frame_histograms(dataset, args)
    write_tsv(
        args.output_dir / "frame_covisibility_histogram.tsv",
        hist_rows,
        ["min_tags_per_camera", "visible_camera_count", "frame_count"])

    pair_rows = pairwise_covisibility(dataset, manifest, args)
    write_tsv(
        args.output_dir / "pair_covisibility.tsv",
        pair_rows,
        ["camera_i", "user_i", "camera_j", "user_j", "shared_frames"])

    summary = {
        "dataset": str(args.dataset),
        "camera_count": dataset["camera_count"],
        "imageset_count": len(dataset["imagesets"]),
        "known_geometries": dataset["known_geometries"],
        "status_counts": status_counts,
        "thresholds": {
            "min_features_per_view": args.min_features_per_view,
            "min_tags_per_view": args.min_tags_per_view,
            "min_usable_views": args.min_usable_views,
            "min_usable_points": args.min_usable_points,
            "target_positive_frames": args.target_positive_frames,
            "target_max_tags": args.target_max_tags,
            "min_bbox_area": args.min_bbox_area,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8")
    write_html(args.output_dir / "coverage_report.html", summary, camera_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
