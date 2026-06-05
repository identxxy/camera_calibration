#!/usr/bin/env python3
"""Publish the canonical t0 calibration report index and remove stale HTML pages."""

import csv
import datetime
import html
import json
import shutil
from pathlib import Path
import statistics


ROOT = Path("/home/ubuntu/calib_data")
BASE_URL = "http://192.168.2.0:9899"
RUN_TAG = "recalib_20260604_outer_large_intrinsics_v1"
RUN = ROOT / "studio_calibration_runs" / RUN_TAG
CURRENT = ROOT / "current_calibration"
REPORTS = CURRENT / "reports"
FINAL_YAML = RUN / "calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml"
CORRESPONDENCE_JSON = RUN / "advanced_correspondence_viewer_v1/correspondence_data.json"
CURRENT_CORRESPONDENCE_JSON = CURRENT / "advanced_correspondence_viewer_v1/correspondence_data.json"


def rel_url(path):
    return BASE_URL + "/" + str(Path(path).relative_to(ROOT)).replace("\\", "/")


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
    images = list(report_dir.glob("camera*_feature_coverage_reprojection.png"))
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


def publish_inner_extrinsic_wrapper(report_dir):
    report_dir = Path(report_dir)
    summary = load_json(report_dir / "summary.json")
    rows = read_tsv(report_dir / "camera_tr_camera0.tsv")
    images = [image for image in [report_dir / "rig_layout_3d.png", report_dir / "rig_layout_topdown.png"] if image.is_file()]
    write_page(
        report_dir / "index.html",
        "Inner 外参报告 - small_marker",
        [
            (
                "Summary",
                summary_metrics_grid([
                    ("camera count", summary.get("camera_count", len(rows)), "inner rig cameras"),
                    ("reference", summary.get("reference_camera", "camera0"), "relative pose reference"),
                    ("pose table rows", len(rows), "camera_tr_camera0.tsv entries"),
                ]),
            ),
            (
                "Relative Camera Poses",
                table(
                    [
                        "camera_index",
                        "tx_m",
                        "ty_m",
                        "tz_m",
                        "roll_deg",
                        "pitch_deg",
                        "yaw_deg",
                    ],
                    rows,
                    limit=40,
                ),
            ),
            ("Rig Layout Plots", image_grid(images)),
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

    if REPORTS.exists():
        shutil.rmtree(REPORTS)
    REPORTS.mkdir(parents=True, exist_ok=True)

    copy_report(viewer_source, REPORTS / "01_3d_viewer")
    copy_report(
        RUN / "inner_bridge/reports/inner8_intrinsic_feature_coverage_small_marker",
        REPORTS / "03_inner_intrinsics_small_marker",
    )
    publish_inner_intrinsic_wrapper(REPORTS / "03_inner_intrinsics_small_marker")
    copy_report(RUN / "inner_bridge/reports/rig_extrinsics", REPORTS / "04_inner_extrinsics_small_marker")
    publish_inner_extrinsic_wrapper(REPORTS / "04_inner_extrinsics_small_marker")
    copy_report(
        ROOT / "calib_2026_06_04_outer_large_marker_v2"
        / "outer_large_marker_20260604_passing_images_only_min1_bycam"
        / "outer24_intrinsic_report_large_marker_v1",
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

    outer_large_summary = load_json(
        ROOT
        / "calib_2026_06_04_outer_large_marker_v2/outer_large_marker_20260604_distributed_filtered_min1_bycam/summary.json"
    )
    outer_large_stats = read_tsv(
        ROOT
        / "calib_2026_06_04_outer_large_marker_v2/outer_large_marker_20260604_distributed_filtered_min1_bycam/per_camera_stats.tsv"
    )
    whole_stats = read_tsv(ROOT / "calib_2026_05_31_v3/whole_outer24_filtered_min4_hybrid_min4cam/per_camera_stats.tsv")
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

    outer_summary = load_json(RUN / "outer_tower/frame_face_refine_wide50_then_gate6/summary.json")
    outer_reproj = read_tsv(RUN / "outer_tower/frame_face_refine_wide50_then_gate6/diagnostics/camera_reprojection.tsv")
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
    write_page(
        REPORTS / "08_bridge_capture_large_marker/index.html",
        "Bridge 数据采集报告 - large_marker",
        [
            (
                "Summary",
                f"""
<div class="grid">
  <div class="metric"><strong>{len(large_manifest)}</strong><span>staged cameras</span></div>
  <div class="metric"><strong>{stat_count(large_manifest, lambda r: r.get('status') == 'usable')}</strong><span>usable camera sequences</span></div>
  <div class="metric"><strong>{int(stat_sum(large_manifest, 'frame_count'))}</strong><span>staged frames across cameras</span></div>
  <div class="metric"><strong>{stat_count(large_pnp, lambda r: r.get('connected') == 'yes')}</strong><span>large-marker connected cameras</span></div>
  <div class="metric"><strong>{int(stat_sum(large_pnp, 'total_inliers'))}</strong><span>large-marker PnP inliers</span></div>
  <div class="metric"><strong>{fnum(median_field(large_pnp, 'median_view_error_px'))} px</strong><span>median per-camera PnP median error</span></div>
</div>
""",
            ),
            (
                "Staged Cameras",
                table(
                    ["camera_index", "stage_name", "machine", "camera_id", "kind", "frame_count", "status", "reason"],
                    large_manifest,
                    limit=40,
                ),
            ),
            (
                "Large Marker PnP Coverage",
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

    bridge = load_json(RUN / "inner_bridge/bridge_colmap_inner_refined_v1/bridge_summary.json")
    metric = bridge.get("quality_gates", {}).get("metric_summary", {})
    anchor_rows = []
    for item in bridge.get("outer_camera_summaries", []):
        anchor_rows.append(
            {
                "label": item.get("label"),
                "vote_count": item.get("vote_count"),
                "center_residual_median_m": fnum(item.get("center_residual_median_m"), 4),
                "center_residual_p90_m": fnum(item.get("center_residual_p90_m"), 4),
                "rotation_residual_median_deg": fnum(item.get("rotation_residual_median_deg"), 3),
                "rotation_residual_p90_deg": fnum(item.get("rotation_residual_p90_deg"), 3),
            }
        )
    write_page(
        REPORTS / "09_bridge_result_large_marker/index.html",
        "Bridge 结果报告 - large_marker",
        [
            (
                "Summary",
                f"""
<div class="grid">
  <div class="metric"><strong>{bridge.get('quality_gates', {}).get('metric_bridge', {}).get('status', '-')}</strong><span>metric bridge gate</span></div>
  <div class="metric"><strong>{metric.get('min_outer_votes', '-')}</strong><span>minimum top-down anchor votes</span></div>
  <div class="metric"><strong>{fnum(metric.get('max_outer_center_residual_p90_m'), 4)} m</strong><span>max top-down center residual p90</span></div>
  <div class="metric"><strong>{fnum(metric.get('max_outer_rotation_residual_p90_deg'), 3)} deg</strong><span>max top-down rotation residual p90</span></div>
</div>
<p>{esc(bridge.get('conclusion', ''))}</p>
""",
            ),
            (
                "Top-Down Anchor Consistency",
                table(
                    [
                        "label",
                        "vote_count",
                        "center_residual_median_m",
                        "center_residual_p90_m",
                        "rotation_residual_median_deg",
                        "rotation_residual_p90_deg",
                    ],
                    anchor_rows,
                ),
            ),
        ],
    )

    reports = [
        ("1", "3D viewer", REPORTS / "01_3d_viewer/index.html", "Unified 32-camera interactive 3D viewer."),
        (
            "2",
            "inner 数据采集报告 (small marker)",
            REPORTS / "02_inner_capture_small_marker/index.html",
            "Small-marker staged data and PnP coverage for inner8.",
        ),
        (
            "3",
            "inner 内参报告 (small marker)",
            REPORTS / "03_inner_intrinsics_small_marker/index.html",
            "Inner8 intrinsic feature coverage and residual distribution.",
        ),
        (
            "4",
            "inner 外参报告 (small marker)",
            REPORTS / "04_inner_extrinsics_small_marker/index.html",
            "Inner rig extrinsic layout report.",
        ),
        (
            "5",
            "outer 数据采集报告",
            REPORTS / "05_outer_capture_outer_large_marker_whole/index.html",
            "Outer-large-marker intrinsic capture plus whole tower extrinsic capture QC.",
        ),
        (
            "6",
            "outer 内参报告 (outer large marker)",
            REPORTS / "06_outer_intrinsics_outer_large_marker/index.html",
            "Outer24 large-marker intrinsic residual/coverage report.",
        ),
        (
            "7",
            "outer 外参报告 (whole)",
            REPORTS / "07_outer_extrinsics_whole/index.html",
            "Whole/tower outer extrinsic refinement residual report.",
        ),
        (
            "8",
            "bridge 数据采集报告 (large marker)",
            REPORTS / "08_bridge_capture_large_marker/index.html",
            "Large-marker bridge capture/staging/PnP coverage.",
        ),
        (
            "9",
            "bridge 结果报告 (large marker)",
            REPORTS / "09_bridge_result_large_marker/index.html",
            "Large-marker inner/outer bridge result and anchor consistency.",
        ),
    ]

    cards = []
    for number, title, path, description in reports:
        cards.append(
            f"<a class='card' href='{esc(rel_url(path))}'><strong>{esc(number)}. {esc(title)}</strong>"
            f"<span>{esc(description)}</span></a>"
        )
    final_yaml_url = rel_url(FINAL_YAML)
    artifact_card = (
        f"<a class='artifact-card' href='{esc(final_yaml_url)}'>"
        "<strong>Final 32-camera YAML</strong>"
        "<span>Machine-readable intrinsics, distortion, and "
        "T_camera_studio extrinsics for all 24 outer + 8 inner cameras.</span>"
        f"<code>{esc(str(FINAL_YAML))}</code>"
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
.card { display: flex; flex-direction: column; gap: 6px; min-height: 86px; padding: 16px; border: 1px solid #ddd9cf; border-radius: 8px; background: #fff; color: #222; text-decoration: none; }
.card:hover { border-color: #8aa0b7; }
.card strong { font-size: 17px; }
.card span { color: #62666b; font-size: 13px; line-height: 1.35; }
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
<p>当前 32-camera 标定只保留 9 个 canonical report 入口。历史探索报告、advanced/debug/audit 页面已从 9899 公开入口清理。</p>
<p>Current run: <code>{esc(RUN_TAG)}</code></p>
</header>
<main>{artifact_card}<div class="grid">{''.join(cards)}</div></main>
</body>
</html>
"""
    (ROOT / "index.html").write_text(index_html, encoding="utf-8")
    (CURRENT / "index.html").write_text(index_html, encoding="utf-8")

    allowed = {ROOT / "index.html", CURRENT / "index.html"}
    allowed.update(path for _number, _title, path, _description in reports)
    removed = []
    for path in sorted(ROOT.rglob("*.html")):
        if path in allowed:
            continue
        path.unlink()
        removed.append(str(path))
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "root_url": BASE_URL + "/",
        "final_yaml": {"path": str(FINAL_YAML), "url": final_yaml_url},
        "correspondence_data": correspondence_data,
        "allowed_reports": [
            {"number": n, "title": t, "path": str(p), "url": rel_url(p)} for n, t, p, _d in reports
        ],
        "removed_html_count": len(removed),
        "removed_html": removed,
    }
    (CURRENT / "report_cleanup_manifest_20260604.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if temp_viewer_source.exists():
        shutil.rmtree(temp_viewer_source)
    return manifest


def main():
    manifest = publish_reports()
    print(
        json.dumps(
            {
                "root_url": BASE_URL + "/",
                "final_yaml_url": manifest["final_yaml"]["url"],
                "allowed_report_count": len(manifest["allowed_reports"]),
                "removed_html_count": manifest["removed_html_count"],
                "manifest": str(CURRENT / "report_cleanup_manifest_20260604.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
