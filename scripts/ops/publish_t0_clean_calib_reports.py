#!/usr/bin/env python3
"""Publish the canonical t0 calibration report index and remove stale HTML pages."""

import csv
import datetime
import html
import json
import shutil
from pathlib import Path
import statistics
import argparse


ROOT = Path("/home/ubuntu/calib_data")
BASE_URL = "http://192.168.2.0:9899"
RUN_TAG = "recalib_20260610_black_tile_wide200_pipeline_v2"
RUN = ROOT / "studio_calibration_runs" / RUN_TAG
CURRENT = ROOT / "current_calibration"
REPORTS = CURRENT / "reports"
FINAL_YAML = RUN / "calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml"
CURRENT_FINAL_YAML = CURRENT / "artifacts/studio_32_cameras.yaml"
CORRESPONDENCE_JSON = RUN / "advanced_correspondence_viewer_v1/correspondence_data.json"
CURRENT_CORRESPONDENCE_JSON = CURRENT / "advanced_correspondence_viewer_v1/correspondence_data.json"
OUTER_LARGE_INTRINSIC_REPORT = (
    ROOT
    / "calib_2026_06_04_outer_large_marker_v2"
    / "outer_large_marker_20260604_passing_images_only_min1_bycam"
    / "outer24_intrinsic_report_large_marker_v1"
)
OUTER_LARGE_QC_ROOT = (
    ROOT
    / "calib_2026_06_04_outer_large_marker_v2"
    / "outer_large_marker_20260604_distributed_filtered_min1_bycam"
)
WHOLE_QC_ROOT = (
    ROOT
    / "calib_2026_05_31_fullres_probe_v1"
    / "whole_outer24_filtered_min4_fullres_min4cam"
)
OUTER_FRAME_FACE_REPORT_ROOT = RUN / "outer_tower/frame_face_refine_wide200_then_gate6"
BRIDGE_CAMERA_ORIGIN_PROJECTION_REPORT = RUN / "inner_bridge/reports/bridge_all32_camera_origin_projection"

CANONICAL_REPORTS = [
    {
        "number": "1",
        "title": "inner 数据采集报告 (small marker)",
        "relative_index": "02_inner_capture_small_marker/index.html",
        "description": "Small-marker staged data and PnP coverage for inner8.",
    },
    {
        "number": "2",
        "title": "inner 内参报告 (small marker)",
        "relative_index": "03_inner_intrinsics_small_marker/index.html",
        "description": "Inner8 intrinsic feature coverage and residual distribution.",
    },
    {
        "number": "3",
        "title": "inner 外参报告 (small marker)",
        "relative_index": "04_inner_extrinsics_small_marker/index.html",
        "description": "Inner8 small-marker fixed-rig pixel reprojection residual report.",
    },
    {
        "number": "4",
        "title": "outer 数据采集报告",
        "relative_index": "05_outer_capture_outer_large_marker_whole/index.html",
        "description": "Outer-large-marker intrinsic capture plus whole tower extrinsic capture QC.",
    },
    {
        "number": "5",
        "title": "outer 内参报告 (outer large marker)",
        "relative_index": "06_outer_intrinsics_outer_large_marker/index.html",
        "description": "Outer24 large-marker intrinsic residual/coverage report.",
    },
    {
        "number": "6",
        "title": "outer 外参报告 (whole)",
        "relative_index": "07_outer_extrinsics_whole/index.html",
        "description": "Whole/tower outer extrinsic refinement residual report.",
    },
    {
        "number": "7",
        "title": "bridge 结果报告 (large marker)",
        "relative_index": "09_bridge_result_large_marker/index.html",
        "description": "Large-marker inner/outer bridge result plus all32 camera-origin projection diagnostic.",
    },
]


def configure(args):
    global ROOT, BASE_URL, RUN_TAG, RUN, CURRENT, REPORTS
    global FINAL_YAML, CURRENT_FINAL_YAML
    global CORRESPONDENCE_JSON, CURRENT_CORRESPONDENCE_JSON
    global OUTER_LARGE_INTRINSIC_REPORT, OUTER_LARGE_QC_ROOT, WHOLE_QC_ROOT
    global OUTER_FRAME_FACE_REPORT_ROOT, BRIDGE_CAMERA_ORIGIN_PROJECTION_REPORT

    ROOT = Path(args.root).resolve()
    BASE_URL = args.base_url.rstrip("/")
    RUN_TAG = args.run_tag
    RUN = ROOT / "studio_calibration_runs" / RUN_TAG
    CURRENT = Path(args.current_dir).resolve() if args.current_dir else ROOT / "current_calibration"
    REPORTS = CURRENT / "reports"
    FINAL_YAML = RUN / "calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml"
    CURRENT_FINAL_YAML = CURRENT / "artifacts/studio_32_cameras.yaml"
    CORRESPONDENCE_JSON = RUN / "advanced_correspondence_viewer_v1/correspondence_data.json"
    CURRENT_CORRESPONDENCE_JSON = CURRENT / "advanced_correspondence_viewer_v1/correspondence_data.json"
    OUTER_LARGE_INTRINSIC_REPORT = Path(args.outer_large_intrinsic_report).resolve()
    OUTER_LARGE_QC_ROOT = Path(args.outer_large_qc_root).resolve()
    WHOLE_QC_ROOT = Path(args.whole_qc_root).resolve()
    OUTER_FRAME_FACE_REPORT_ROOT = (
        Path(args.outer_frame_face_report_root).resolve()
        if args.outer_frame_face_report_root
        else RUN / "outer_tower/frame_face_refine_wide200_then_gate6"
    )
    BRIDGE_CAMERA_ORIGIN_PROJECTION_REPORT = RUN / "inner_bridge/reports/bridge_all32_camera_origin_projection"


def rel_url(path):
    return BASE_URL + "/" + str(Path(path).relative_to(ROOT)).replace("\\", "/")


def canonical_report_path(report):
    return REPORTS / report["relative_index"]


def canonical_report_entries():
    return [
        {
            **report,
            "path": canonical_report_path(report),
        }
        for report in CANONICAL_REPORTS
    ]


def esc(value):
    return html.escape("" if value is None else str(value))


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fnum(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return esc(value)


def stat_sum(rows, field):
    total = 0.0
    for row in rows:
        try:
            total += float(row.get(field, 0) or 0)
        except Exception:
            pass
    return total


def stat_count(rows, predicate):
    return sum(1 for row in rows if predicate(row))


def median_field(rows, field):
    values = []
    for row in rows:
        try:
            values.append(float(row.get(field, "")))
        except Exception:
            pass
    return statistics.median(values) if values else None


def percentile(values, percentile_value):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * float(percentile_value) / 100.0
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def write_camera_metrics_from_correspondence(input_tsv, output_tsv):
    rows = read_tsv(input_tsv)
    by_camera = {}
    for row in rows:
        if row.get("projection_status") != "ok":
            continue
        try:
            residual = float(row.get("residual_px", ""))
        except Exception:
            continue
        camera_index = row.get("camera_index", "")
        bucket = by_camera.setdefault(
            camera_index,
            {
                "camera_index": camera_index,
                "camera_label": row.get("camera_label") or row.get("user_id") or row.get("camera_id") or camera_index,
                "residuals": [],
                "frames": set(),
            },
        )
        bucket["residuals"].append(residual)
        if row.get("frame_index", ""):
            bucket["frames"].add(row.get("frame_index"))
    if not by_camera:
        return None

    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "camera_index",
                "camera_label",
                "residual_count",
                "frame_count",
                "median_error_px",
                "mean_error_px",
                "p90_error_px",
                "max_error_px",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for _camera_index, bucket in sorted(by_camera.items(), key=lambda item: int(float(item[0] or 0))):
            residuals = bucket["residuals"]
            writer.writerow({
                "camera_index": bucket["camera_index"],
                "camera_label": bucket["camera_label"],
                "residual_count": len(residuals),
                "frame_count": len(bucket["frames"]),
                "median_error_px": f"{statistics.median(residuals):.6f}",
                "mean_error_px": f"{statistics.mean(residuals):.6f}",
                "p90_error_px": f"{percentile(residuals, 90):.6f}",
                "max_error_px": f"{max(residuals):.6f}",
            })
    return output_tsv


def table(headers, rows, limit=None):
    shown = rows if limit is None else rows[:limit]
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{esc(header)}</th>" for header in headers)
    parts.append("</tr></thead><tbody>")
    for row in shown:
        parts.append("<tr>")
        parts.extend(f"<td>{esc(row.get(header, ''))}</td>" for header in headers)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if limit is not None and len(rows) > limit:
        parts.append(f"<p class='muted'>Showing {limit} / {len(rows)} rows.</p>")
    return "\n".join(parts)


STYLE = """
:root { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #222; background: #f6f6f2; }
body { margin: 0; }
header { padding: 28px 36px 20px; background: #fff; border-bottom: 1px solid #ddd9cf; }
main { padding: 24px 36px 44px; max-width: 1280px; }
h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
h2 { margin: 28px 0 10px; font-size: 18px; }
section { background: #fff; border: 1px solid #ddd9cf; border-radius: 8px; padding: 16px 18px; margin: 14px 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.metric { background: #f0f0ea; border-radius: 8px; padding: 12px; }
.metric strong { display: block; font-size: 22px; margin-bottom: 3px; }
.metric span, .muted { color: #666a70; font-size: 13px; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th, td { padding: 7px 8px; border-bottom: 1px solid #ebe8df; font-size: 13px; text-align: left; vertical-align: top; }
th { background: #eeece5; }
img.report-image { width: 100%; height: auto; border: 1px solid #ddd9cf; border-radius: 6px; background: #111; }
.image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.image-card { background: #f7f7f4; border: 1px solid #ddd9cf; border-radius: 8px; padding: 10px; }
.image-card strong { display: block; font-size: 13px; margin-bottom: 7px; color: #444; }
code { background: #eeece5; padding: 1px 4px; border-radius: 4px; }
a { color: #235c8f; }
@media (max-width: 760px) { header, main { padding-left: 18px; padding-right: 18px; } }
"""


def write_page(path, title, sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = []
    for heading, content in sections:
        body.append(f"<section><h2>{esc(heading)}</h2>{content}</section>")
    generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{STYLE}</style>
</head>
<body>
<header>
  <h1>{esc(title)}</h1>
  <p class="muted">Generated {esc(generated)} - canonical report wrapper - current run: <code>{esc(RUN_TAG)}</code></p>
</header>
<main>{''.join(body)}</main>
</body>
</html>
"""
    path.write_text(text, encoding="utf-8")


def copy_report(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def require_path(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def publish_final_yaml():
    require_path(FINAL_YAML, "final 32-camera YAML")
    CURRENT_FINAL_YAML.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FINAL_YAML, CURRENT_FINAL_YAML)
    return {
        "source_path": str(FINAL_YAML),
        "path": str(CURRENT_FINAL_YAML),
        "url": rel_url(CURRENT_FINAL_YAML),
    }


def publish_correspondence_data():
    if CORRESPONDENCE_JSON.is_file():
        CURRENT_CORRESPONDENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CORRESPONDENCE_JSON, CURRENT_CORRESPONDENCE_JSON)
    if not CURRENT_CORRESPONDENCE_JSON.is_file():
        return {}
    return {
        "path": str(CURRENT_CORRESPONDENCE_JSON),
        "url": rel_url(CURRENT_CORRESPONDENCE_JSON),
        "size_bytes": CURRENT_CORRESPONDENCE_JSON.stat().st_size,
        "source": str(CORRESPONDENCE_JSON),
    }


def image_grid(images, limit=None):
    images = sorted(Path(image).name for image in images)
    if limit is not None:
        images = images[:limit]
    if not images:
        return "<p class='muted'>No PNG plots found in this report directory.</p>"
    parts = ["<div class='image-grid'>"]
    for image in images:
        parts.append(
            f"<div class='image-card'><strong>{esc(image)}</strong>"
            f"<img class='report-image' src='{esc(image)}' alt='{esc(image)}'></div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def summary_metrics_grid(items):
    parts = ["<div class='grid'>"]
    for label, value, subtitle in items:
        parts.append(
            f"<div class='metric'><strong>{esc(value)}</strong><span>{esc(label)}</span>"
            f"<span>{esc(subtitle)}</span></div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def publish_inner_intrinsic_wrapper(report_dir):
    report_dir = Path(report_dir)
    rows = read_tsv(report_dir / "camera_metrics.tsv")
    summary = load_json(report_dir / "summary.json")
    images = (
        list(report_dir.glob("camera*_feature_coverage_reprojection.png"))
        + list(report_dir.glob("camera*_reprojection_arrows_log.png"))
    )
    write_page(
        report_dir / "index.html",
        "Inner 内参报告 - small_marker",
        [
            (
                "Summary",
                summary_metrics_grid([
                    ("camera count", summary.get("camera_count", len(rows)), "inner cameras in the report"),
                    ("total residuals", int(stat_sum(rows, "residual_count")), "board-corner reprojection residuals"),
                    ("median per-camera median error", f"{fnum(median_field(rows, 'median_error_px'))} px", "intrinsic reprojection median"),
                    ("median per-camera p90 error", f"{fnum(median_field(rows, 'p90_error_px'))} px", "intrinsic reprojection p90"),
                ]),
            ),
            (
                "Color Scale",
                "<p class='muted'>Coverage plots use a fixed log reprojection-error colormap "
                "from <code>10^-1</code> to <code>10^1</code> px, shared across inner and outer intrinsic reports.</p>",
            ),
            (
                "Per-Camera Intrinsic Residuals",
                table(
                    [
                        "camera_index",
                        "frame_count",
                        "residual_count",
                        "median_error_px",
                        "p90_error_px",
                        "max_error_px",
                    ],
                    rows,
                ),
            ),
            ("Accumulated Corner Coverage / Reprojection Arrows", image_grid(images)),
        ],
    )


def publish_outer_intrinsic_wrapper(report_dir):
    report_dir = Path(report_dir)
    rows = read_tsv(report_dir / "camera_metrics.tsv")
    summary = load_json(report_dir / "summary.json")
    images = list(report_dir.glob("camera*_reprojection_arrows_log.png"))
    write_page(
        report_dir / "index.html",
        "Outer 内参报告 - outer_large_marker",
        [
            (
                "Summary",
                summary_metrics_grid([
                    ("camera count", summary.get("camera_count", len(rows)), "outer cameras in the report"),
                    ("usable views", int(stat_sum(rows, "usable_views")), "large-marker frames used for intrinsic calibration"),
                    ("usable points", int(stat_sum(rows, "usable_points") or stat_sum(rows, "residual_count")), "detected board corners"),
                    ("median per-camera median error", f"{fnum(median_field(rows, 'median_error_px'))} px", "intrinsic reprojection median"),
                    ("median per-camera p90 error", f"{fnum(median_field(rows, 'p90_error_px'))} px", "intrinsic reprojection p90"),
                ]),
            ),
            (
                "Color Scale",
                "<p class='muted'>Coverage plots use a fixed log reprojection-error colormap "
                "from <code>10^-1</code> to <code>10^1</code> px, shared across inner and outer intrinsic reports.</p>",
            ),
            (
                "Per-Camera Intrinsic Residuals",
                table(
                    [
                        "camera_index",
                        "user_id",
                        "usable_views",
                        "usable_points",
                        "residual_count",
                        "median_error_px",
                        "p90_error_px",
                        "max_error_px",
                    ],
                    rows,
                    limit=40,
                ),
            ),
            ("Accumulated Corner Coverage / Reprojection Arrows", image_grid(images, limit=40)),
        ],
    )


def index_rows_by_camera(rows):
    result = {}
    for row in rows:
        value = row.get("camera_index", row.get("camera", ""))
        try:
            result[int(value)] = row
        except Exception:
            continue
    return result


def publish_inner_extrinsic_wrapper(
        report_dir,
        residual_metrics_tsv=None,
        pnp_summary_tsv=None,
        source_description="It summarizes small-marker fixed-rig reprojection residuals in pixels."):
    report_dir = Path(report_dir)
    summary = load_json(report_dir / "summary.json")
    residual_rows = read_tsv(residual_metrics_tsv) if residual_metrics_tsv else []
    pnp_rows = index_rows_by_camera(read_tsv(pnp_summary_tsv)) if pnp_summary_tsv else {}
    table_rows_data = []
    for row in residual_rows:
        try:
            camera_index = int(row.get("camera_index", ""))
        except Exception:
            camera_index = None
        pnp = pnp_rows.get(camera_index, {}) if camera_index is not None else {}
        table_rows_data.append({
            "camera_index": row.get("camera_index", ""),
            "camera_label": row.get("camera_label", pnp.get("user_id", "")),
            "connected": pnp.get("connected", ""),
            "positive_views": pnp.get("positive_views", ""),
            "solved_views": pnp.get("solved_views", ""),
            "residual_count": row.get("residual_count", ""),
            "frame_count": row.get("frame_count", ""),
            "median_error_px": row.get("median_error_px", ""),
            "mean_error_px": row.get("mean_error_px", ""),
            "p90_error_px": row.get("p90_error_px", ""),
            "max_error_px": row.get("max_error_px", ""),
        })
    write_page(
        report_dir / "index.html",
        "Inner 外参报告 - small_marker",
        [
            (
                "Summary",
                summary_metrics_grid([
                    ("camera count", summary.get("camera_count", len(table_rows_data)), "inner rig cameras"),
                    ("residual observations", int(stat_sum(table_rows_data, "residual_count")), "small-marker fixed-rig reprojections"),
                    ("median per-camera median error", f"{fnum(median_field(table_rows_data, 'median_error_px'))} px", "small-marker reprojection median"),
                    ("median per-camera p90 error", f"{fnum(median_field(table_rows_data, 'p90_error_px'))} px", "small-marker reprojection p90"),
                ]),
            ),
            (
                "Per-Camera Pixel Reprojection Error",
                table(
                    [
                        "camera_index",
                        "camera_label",
                        "connected",
                        "positive_views",
                        "solved_views",
                        "residual_count",
                        "frame_count",
                        "median_error_px",
                        "mean_error_px",
                        "p90_error_px",
                        "max_error_px",
                    ],
                    table_rows_data,
                    limit=40,
                ),
            ),
            (
                "Definition",
                "<p class='muted'>This extrinsic report intentionally does not include static layout plots. "
                f"{esc(source_description)} Spatial layout inspection belongs to the Overall 3D Viewer.</p>",
            ),
        ],
    )


def publish_reports():
    viewer_source = RUN / "inner_bridge/combined_studio_rig_viewer_v1"
    temp_viewer_source = Path("/tmp/t0_calib_publish_01_3d_viewer")
    if temp_viewer_source.exists():
        shutil.rmtree(temp_viewer_source)
    if not (viewer_source / "index.html").is_file() and (REPORTS / "01_3d_viewer/index.html").is_file():
        shutil.copytree(REPORTS / "01_3d_viewer", temp_viewer_source)
        viewer_source = temp_viewer_source
    if not (viewer_source / "index.html").is_file():
        raise FileNotFoundError(
            "3D viewer source index.html is missing. Regenerate "
            f"{RUN / 'inner_bridge/combined_studio_rig_viewer_v1/index.html'} first."
        )
    require_path(RUN / "inner_bridge/reports/inner_reprojection", "inner intrinsic/reprojection report source")
    require_path(RUN / "inner_bridge/reports/rig_extrinsics", "inner extrinsic report source")
    require_path(OUTER_LARGE_INTRINSIC_REPORT, "outer large-marker intrinsic report source")
    require_path(OUTER_LARGE_QC_ROOT / "per_camera_stats.tsv", "outer large-marker QC stats")
    require_path(WHOLE_QC_ROOT / "per_camera_stats.tsv", "whole QC stats")
    require_path(OUTER_FRAME_FACE_REPORT_ROOT / "summary.json", "outer frame-face report summary")
    require_path(BRIDGE_CAMERA_ORIGIN_PROJECTION_REPORT / "index.html", "bridge all32 camera-origin projection report")
    final_yaml = publish_final_yaml()

    if REPORTS.exists():
        shutil.rmtree(REPORTS)
    REPORTS.mkdir(parents=True, exist_ok=True)

    copy_report(viewer_source, REPORTS / "01_3d_viewer")
    copy_report(
        RUN / "inner_bridge/reports/inner_reprojection",
        REPORTS / "03_inner_intrinsics_small_marker",
    )
    publish_inner_intrinsic_wrapper(REPORTS / "03_inner_intrinsics_small_marker")
    copy_report(RUN / "inner_bridge/reports/rig_extrinsics", REPORTS / "04_inner_extrinsics_small_marker")
    small_quality_correspondence = (
        RUN / "inner_bridge/small_marker_inner8/fixed_intrinsic_small_grid4_quality_probe_v1/correspondence_residuals.tsv"
    )
    small_quality_metrics = write_camera_metrics_from_correspondence(
        small_quality_correspondence,
        REPORTS / "04_inner_extrinsics_small_marker/small_marker_quality_camera_metrics.tsv",
    )
    inner_extrinsic_metrics_tsv = (
        small_quality_metrics
        if small_quality_metrics
        else REPORTS / "03_inner_intrinsics_small_marker/camera_metrics.tsv"
    )
    inner_extrinsic_source_description = (
        "It summarizes small-marker fixed-rig quality-probe reprojection residuals in pixels."
        if small_quality_metrics
        else "Small-marker quality-probe correspondence residuals were unavailable; this table falls back to the published inner baseline reprojection metrics."
    )
    publish_inner_extrinsic_wrapper(
        REPORTS / "04_inner_extrinsics_small_marker",
        inner_extrinsic_metrics_tsv,
        RUN / "inner_bridge/small_marker_inner8/fixed_intrinsic_small_grid4_quality_probe_v1/camera_pnp_summary.tsv",
        inner_extrinsic_source_description,
    )
    copy_report(
        OUTER_LARGE_INTRINSIC_REPORT,
        REPORTS / "06_outer_intrinsics_outer_large_marker",
    )
    publish_outer_intrinsic_wrapper(REPORTS / "06_outer_intrinsics_outer_large_marker")
    correspondence_data = publish_correspondence_data()

    small_manifest = read_tsv(RUN / "inner_bridge/planned_inputs/small_marker_usable_manifest.tsv")
    small_pnp = read_tsv(
        RUN / "inner_bridge/small_marker_inner8/fixed_intrinsic_small_grid4_quality_probe_v1/camera_pnp_summary.tsv"
    )
    write_page(
        REPORTS / "02_inner_capture_small_marker/index.html",
        "Inner 数据采集报告 - small_marker",
        [
            (
                "Summary",
                f"""
<div class="grid">
  <div class="metric"><strong>{len(small_manifest)}</strong><span>inner cameras staged</span></div>
  <div class="metric"><strong>{stat_count(small_manifest, lambda r: r.get('status') == 'usable')}</strong><span>usable camera sequences</span></div>
  <div class="metric"><strong>{int(stat_sum(small_manifest, 'frame_count'))}</strong><span>staged frames across cameras</span></div>
  <div class="metric"><strong>{int(stat_sum(small_pnp, 'total_inliers'))}</strong><span>small-marker inliers used by PnP probe</span></div>
  <div class="metric"><strong>{fnum(median_field(small_pnp, 'median_view_error_px'))} px</strong><span>median per-camera PnP median error</span></div>
</div>
""",
            ),
            (
                "Staged Cameras",
                table(["camera_index", "stage_name", "machine", "camera_id", "frame_count", "status", "reason"], small_manifest),
            ),
            (
                "Small Marker PnP Probe",
                table(
                    [
                        "camera_index",
                        "user_id",
                        "connected",
                        "total_views",
                        "positive_views",
                        "solved_views",
                        "total_inliers",
                        "median_view_error_px",
                    ],
                    small_pnp,
                ),
            ),
        ],
    )

    outer_large_summary = load_json(OUTER_LARGE_QC_ROOT / "summary.json")
    outer_large_stats = read_tsv(OUTER_LARGE_QC_ROOT / "per_camera_stats.tsv")
    whole_stats = read_tsv(WHOLE_QC_ROOT / "per_camera_stats.tsv")
    write_page(
        REPORTS / "05_outer_capture_outer_large_marker_whole/index.html",
        "Outer 数据采集报告 - outer_large_marker + whole",
        [
            (
                "Summary",
                f"""
<div class="grid">
  <div class="metric"><strong>{outer_large_summary.get('camera_count', '-')}</strong><span>outer_large_marker cameras</span></div>
  <div class="metric"><strong>{outer_large_summary.get('candidate_frame_count', '-')}</strong><span>outer_large_marker candidate frames</span></div>
  <div class="metric"><strong>{outer_large_summary.get('passing_image_count', '-')}</strong><span>outer_large_marker passing images</span></div>
  <div class="metric"><strong>{outer_large_summary.get('selected_frame_count', '-')}</strong><span>outer_large_marker selected synchronized frames</span></div>
  <div class="metric"><strong>{stat_count(whole_stats, lambda r: float(r.get('passing_images') or 0) > 0)}</strong><span>whole cameras with raw positive tag views</span></div>
  <div class="metric"><strong>{int(stat_sum(whole_stats, 'selected_passing_frames'))}</strong><span>whole selected passing camera-frames</span></div>
</div>
""",
            ),
            (
                "outer_large_marker Per-Camera QC",
                table(
                    [
                        "camera_id",
                        "total_images",
                        "decoded_images",
                        "failed_images",
                        "passing_images",
                        "passing_ratio",
                        "max_tags",
                        "selected_passing_frames",
                    ],
                    outer_large_stats,
                ),
            ),
            (
                "whole Per-Camera QC",
                table(
                    [
                        "camera_id",
                        "total_images",
                        "decoded_images",
                        "failed_images",
                        "passing_images",
                        "passing_ratio",
                        "total_tags",
                        "max_tags",
                        "selected_passing_frames",
                    ],
                    whole_stats,
                ),
            ),
        ],
    )

    outer_summary = load_json(OUTER_FRAME_FACE_REPORT_ROOT / "summary.json")
    outer_reproj = read_tsv(OUTER_FRAME_FACE_REPORT_ROOT / "diagnostics/camera_reprojection.tsv")
    res_after = outer_summary.get("residual_after", {})
    obs_gate = outer_summary.get("observation_gate", {})
    cameras = outer_summary.get("cameras", {})
    write_page(
        REPORTS / "07_outer_extrinsics_whole/index.html",
        "Outer 外参报告 - whole",
        [
            (
                "Summary",
                f"""
<div class="grid">
  <div class="metric"><strong>{cameras.get('active_delta', '-')} / {cameras.get('total', '-')}</strong><span>active outer cameras optimized</span></div>
  <div class="metric"><strong>{res_after.get('count', '-')}</strong><span>final residual observations</span></div>
  <div class="metric"><strong>{fnum(res_after.get('median_px'))} px</strong><span>final median reprojection error</span></div>
  <div class="metric"><strong>{fnum(res_after.get('p90_px'))} px</strong><span>final p90 reprojection error</span></div>
  <div class="metric"><strong>{outer_summary.get('frame_faces', {}).get('used', '-')}</strong><span>used frame-face poses</span></div>
  <div class="metric"><strong>{obs_gate.get('max_residual_px', '-')} px</strong><span>final observation gate</span></div>
</div>
<p class="muted">Inactive delta cameras: {esc(', '.join(cameras.get('inactive_delta', [])))}</p>
""",
            ),
            (
                "Per-Camera Reprojection",
                table(
                    [
                        "camera_id",
                        "observation_count",
                        "before_median_px",
                        "before_p90_px",
                        "after_median_px",
                        "after_p90_px",
                        "after_max_px",
                    ],
                    outer_reproj,
                ),
            ),
        ],
    )

    large_manifest = read_tsv(RUN / "inner_bridge/planned_inputs/large_marker_usable_manifest.tsv")
    large_pnp = read_tsv(
        RUN / "inner_bridge/large_marker_bridge_all32/fixed_intrinsic_bridge_pnp_stride1_v1/camera_pnp_summary.tsv"
    )
    inner_bridge_summary = load_json(RUN / "inner_bridge/summary.json")
    bridge_corr = inner_bridge_summary.get("bridge_correspondence_quality", {})
    bridge_intrinsics = inner_bridge_summary.get("bridge_intrinsics", {})
    bridge_layout = inner_bridge_summary.get("bridge_layout", {})
    copy_report(
        BRIDGE_CAMERA_ORIGIN_PROJECTION_REPORT,
        REPORTS / "09_bridge_result_large_marker/camera_origin_projection",
    )
    write_page(
        REPORTS / "09_bridge_result_large_marker/index.html",
        "Bridge 结果报告 - large_marker",
        [
            (
                "Summary",
                f"""
<div class="grid">
  <div class="metric"><strong>{bridge_corr.get('status', '-')}</strong><span>all32 bridge BA residual status</span></div>
  <div class="metric"><strong>{bridge_corr.get('ok_count', '-')}</strong><span>post-BA correspondences</span></div>
  <div class="metric"><strong>{fnum(bridge_corr.get('median_residual_px'))} px</strong><span>large-marker dataset median px</span></div>
  <div class="metric"><strong>{fnum(bridge_corr.get('p90_residual_px'))} px</strong><span>large-marker dataset p90 px</span></div>
  <div class="metric"><strong>{fnum(bridge_corr.get('max_residual_px'))} px</strong><span>large-marker dataset max px</span></div>
  <div class="metric"><strong>{bridge_intrinsics.get('ready_count', '-')} / {bridge_intrinsics.get('expected_count', '-')}</strong><span>fixed intrinsics available</span></div>
  <div class="metric"><strong>{bridge_layout.get('observed_camera_count', '-')} / {bridge_layout.get('expected_camera_count', '-')}</strong><span>observed cameras in bridge</span></div>
  <div class="metric"><strong>{stat_count(large_pnp, lambda r: r.get('connected') == 'yes')}</strong><span>PnP initializer connected cameras</span></div>
</div>
<p class="muted">This report is based on the current production all32 fixed-known-point joint BA. The older top-down-anchor bridge summary remains in the run directory as a diagnostic, but it is not the current bridge quality gate.</p>
""",
            ),
            (
                "All32 Camera-Origin Projection",
                "<p class='muted'>This diagnostic projects each of the 32 calibrated camera optical centers "
                "into each of the 32 large-marker view images. It is the bridge sanity check for gross extrinsic "
                "orientation or scale failures.</p>"
                "<p><a href='camera_origin_projection/index.html'>Open all32 camera-origin projection report</a></p>",
            ),
            (
                "Large Marker Staging",
                table(
                    ["camera_index", "stage_name", "machine", "camera_id", "kind", "frame_count", "status", "reason"],
                    large_manifest,
                    limit=40,
                ),
            ),
            (
                "All32 PnP Initializer Coverage",
                table(
                    [
                        "camera_index",
                        "user_id",
                        "connected",
                        "total_views",
                        "positive_views",
                        "solved_views",
                        "total_inliers",
                        "median_view_error_px",
                    ],
                    large_pnp,
                    limit=40,
                ),
            ),
        ],
    )

    reports = canonical_report_entries()

    viewer_card = (
        f"<a class='card viewer-card' href='{esc(rel_url(REPORTS / '01_3d_viewer/index.html'))}'>"
        "<strong>Overall 3D Viewer</strong>"
        "<span>Unified 32-camera interactive viewer with intrinsic residuals, "
        "final dataset/extrinsic residuals, camera filters, and correspondence loading.</span>"
        "</a>"
    )
    cards = []
    for report in reports:
        cards.append(
            f"<a class='card' href='{esc(rel_url(report['path']))}'>"
            f"<strong>{esc(report['number'])}. {esc(report['title'])}</strong>"
            f"<span>{esc(report['description'])}</span></a>"
        )
    final_yaml_url = final_yaml["url"]
    artifact_card = (
        f"<a class='artifact-card' href='{esc(final_yaml_url)}'>"
        "<strong>Final 32-camera YAML</strong>"
        "<span>Machine-readable intrinsics, distortion, and "
        "T_camera_studio extrinsics for all 24 outer + 8 inner cameras.</span>"
        f"<code>{esc(str(CURRENT_FINAL_YAML))}</code>"
        "</a>"
    )
    index_style = """
:root { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #222; background: #f6f6f2; }
body { margin: 0; }
header { padding: 30px 38px 22px; background: #fff; border-bottom: 1px solid #ddd9cf; }
main { padding: 26px 38px 48px; max-width: 1180px; }
h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }
p { color: #62666b; line-height: 1.45; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.viewer-grid { display: grid; grid-template-columns: minmax(280px, 520px); gap: 12px; margin-bottom: 22px; }
.card { display: flex; flex-direction: column; gap: 6px; min-height: 86px; padding: 16px; border: 1px solid #ddd9cf; border-radius: 8px; background: #fff; color: #222; text-decoration: none; }
.card:hover { border-color: #8aa0b7; }
.card strong { font-size: 17px; }
.card span { color: #62666b; font-size: 13px; line-height: 1.35; }
.viewer-card { border-color: #b6c5d2; background: #fbfdff; }
.artifact-card { display: flex; flex-direction: column; gap: 8px; padding: 16px; margin-bottom: 18px; border: 1px solid #b9c6d4; border-radius: 8px; background: #f7fbff; color: #222; text-decoration: none; }
.artifact-card:hover { border-color: #5f83a9; }
.artifact-card strong { font-size: 18px; }
.artifact-card span { color: #46515c; font-size: 13px; line-height: 1.35; }
.artifact-card code { overflow-wrap: anywhere; }
code { background: #eeece5; padding: 1px 4px; border-radius: 4px; }
@media (max-width: 760px) { header, main { padding-left: 18px; padding-right: 18px; } }
"""
    index_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Studio Calibration Reports</title>
<style>{index_style}</style>
</head>
<body>
<header>
<h1>Studio Calibration Reports</h1>
<p>当前 32-camera 标定只保留一个 overall viewer 和七个 canonical report 入口。历史探索报告、advanced/debug/audit 页面不在 9899 首页展示。</p>
<p>Current run: <code>{esc(RUN_TAG)}</code></p>
</header>
<main>{artifact_card}<h2>Overall Viewer</h2><div class="viewer-grid">{viewer_card}</div><h2>Reports</h2><div class="grid">{''.join(cards)}</div></main>
</body>
</html>
"""
    (ROOT / "index.html").write_text(index_html, encoding="utf-8")
    (CURRENT / "index.html").write_text(index_html, encoding="utf-8")

    allowed = {ROOT / "index.html", CURRENT / "index.html"}
    allowed.add(REPORTS / "01_3d_viewer/index.html")
    allowed.update(report["path"] for report in reports)
    allowed_html_dirs = [
        REPORTS / "09_bridge_result_large_marker/camera_origin_projection",
    ]
    removed_current_html = []
    for path in sorted(CURRENT.rglob("*.html")):
        if path in allowed or any(directory in path.parents for directory in allowed_html_dirs):
            continue
        path.unlink()
        removed_current_html.append(str(path))
    for stale in [
        CURRENT / "operations",
        CURRENT / "reports/08_bridge_capture_large_marker",
    ]:
        if stale.exists():
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
    for pattern in [
        "report_cleanup_manifest_*.json",
        "report_refresh_manifest_*.json",
        "report_registry.json",
        "inner_recalib_audit.json",
        "README.md",
    ]:
        for path in CURRENT.glob(pattern):
            path.unlink()
    if CURRENT_CORRESPONDENCE_JSON.parent.is_dir():
        for path in CURRENT_CORRESPONDENCE_JSON.parent.iterdir():
            if path != CURRENT_CORRESPONDENCE_JSON:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    manifest_path = CURRENT / "report_cleanup_manifest_latest.json"
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "root_url": BASE_URL + "/",
        "final_yaml": final_yaml,
        "correspondence_data": correspondence_data,
        "overall_viewer": {
            "path": str(REPORTS / "01_3d_viewer/index.html"),
            "url": rel_url(REPORTS / "01_3d_viewer/index.html"),
        },
        "allowed_reports": [
            {
                "number": report["number"],
                "title": report["title"],
                "path": str(report["path"]),
                "url": rel_url(report["path"]),
            }
            for report in reports
        ],
        "source_paths": {
            "run": str(RUN),
            "outer_frame_face_report_root": str(OUTER_FRAME_FACE_REPORT_ROOT),
            "outer_large_intrinsic_report": str(OUTER_LARGE_INTRINSIC_REPORT),
            "outer_large_qc_root": str(OUTER_LARGE_QC_ROOT),
            "whole_qc_root": str(WHOLE_QC_ROOT),
            "inner_extrinsic_metrics_tsv": str(inner_extrinsic_metrics_tsv),
        },
        "removed_current_html_count": len(removed_current_html),
        "removed_current_html": removed_current_html,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if temp_viewer_source.exists():
        shutil.rmtree(temp_viewer_source)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--current-dir", default="")
    parser.add_argument("--outer-large-intrinsic-report", default=str(OUTER_LARGE_INTRINSIC_REPORT))
    parser.add_argument("--outer-large-qc-root", default=str(OUTER_LARGE_QC_ROOT))
    parser.add_argument("--whole-qc-root", default=str(WHOLE_QC_ROOT))
    parser.add_argument("--outer-frame-face-report-root", default="")
    return parser.parse_args()


def main():
    configure(parse_args())
    manifest = publish_reports()
    print(
        json.dumps(
            {
                "root_url": BASE_URL + "/",
                "final_yaml_url": manifest["final_yaml"]["url"],
                "allowed_report_count": len(manifest["allowed_reports"]),
                "removed_current_html_count": manifest["removed_current_html_count"],
                "manifest": str(CURRENT / "report_cleanup_manifest_latest.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
