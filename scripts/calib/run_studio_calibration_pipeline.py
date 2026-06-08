#!/usr/bin/env python3
"""One-command orchestration for the studio calibration pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


DEFAULT_WHOLE_DATA_ROOT = Path("/home/ubuntu/calib_data/calib_2026_05_31_v3")
DEFAULT_INNER_DATA_ROOT = Path("/home/ubuntu/calib_data/calib_2026_05_31_v3")
DEFAULT_HTTP_ROOT = Path("/home/ubuntu/calib_data")
DEFAULT_REPORT_URL_BASE = "http://192.168.2.0:9899"
DEFAULT_PANEL_URL = "http://192.168.2.0:9898/"
DEFAULT_INNER_PRIOR = (
    Path("/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3")
    / "final_inner8_calibration_v1/states/final_small_marker_grid4_refine_v1"
)
DEFAULT_OUTER_COLMAP_PRIOR = (
    Path("/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3")
    / "colmap_outer24_firstframe_colmap404_v3/fixed_intrinsics/sparse_txt_final24_fixedK_ba/images.txt"
)
DEFAULT_OUTER_FRAME_FACE_PRIOR_POSE_YAML = (
    Path("/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50")
    / "outer_tower/frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1/camera_tr_rig_delta_refined.yaml"
)
DEFAULT_OUTER_FRAME_FACE_INTRINSICS_DIR = (
    Path("/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50")
    / "outer_tower/frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1/intrinsics_refined"
)
DEFAULT_OUTER_INTRINSIC_METRICS_TSV = (
    Path("/home/ubuntu/calib_data/calib_2026_06_04_outer_large_marker_v2")
    / "outer_large_marker_20260604_passing_images_only_min1_bycam"
    / "outer24_intrinsic_report_large_marker_v1/camera_metrics.tsv"
)
DEFAULT_OUTER_LARGE_OPENCV_INTRINSICS_DIR = (
    Path("/home/ubuntu/calib_data/calib_2026_06_04_outer_large_marker_v2")
    / "outer_large_marker_20260604_passing_images_only_min1_bycam"
    / "outer24_opencv_intrinsics_large_marker_v1"
)
DEFAULT_OUTER_LARGE_QC_ROOT = (
    Path("/home/ubuntu/calib_data/calib_2026_06_04_outer_large_marker_v2")
    / "outer_large_marker_20260604_distributed_filtered_min1_bycam"
)
def repo_root():
    return Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def timestamp_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def command_string(argv):
    return shlex.join(str(item) for item in argv)


def resolve_path(path, base=None):
    path = Path(path).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve(strict=False)


def rel_for_report(path, http_root):
    path = Path(path).resolve(strict=False)
    http_root = Path(http_root).resolve(strict=False)
    try:
        return path.relative_to(http_root).as_posix()
    except ValueError:
        return path.as_posix()


def report_url(path, http_root, report_url_base):
    path = Path(path).resolve(strict=False)
    try:
        rel = path.relative_to(Path(http_root).resolve(strict=False))
        return f"{report_url_base.rstrip('/')}/{rel.as_posix()}"
    except ValueError:
        return path.as_uri()


def path_status(path):
    path = Path(path)
    if path.is_file():
        kind = "file"
    elif path.is_dir():
        kind = "directory"
    elif path.exists():
        kind = "other"
    else:
        kind = "missing"
    return {"path": str(path), "exists": path.exists(), "kind": kind}


def first_existing(paths):
    for path in paths:
        if path and Path(path).is_file():
            return Path(path).resolve(strict=False)
    return ""


def default_output_root(args):
    run_tag = args.run_tag or timestamp_tag()
    return DEFAULT_HTTP_ROOT / "studio_calibration_runs" / run_tag


def build_paths(args):
    whole_data_root = resolve_path(args.whole_data_root)
    inner_data_root = resolve_path(args.inner_data_root)
    output_root = resolve_path(args.output_root) if args.output_root else default_output_root(args)
    outer_wrapper_root = output_root / "outer_tower_wrapper"
    outer_frame_face_dir = output_root / "outer_tower" / f"frame_face_refine_{args.outer_preset}"
    outer_pose_yaml = (
        resolve_path(args.outer_final_pose_yaml)
        if args.outer_final_pose_yaml
        else outer_frame_face_dir / "camera_tr_rig_delta_refined.yaml"
    )
    outer_intrinsics_dir = (
        resolve_path(args.outer_final_intrinsics_dir)
        if args.outer_final_intrinsics_dir
        else outer_frame_face_dir / "intrinsics_refined"
    )
    outer_large_opencv_intrinsics_dir = resolve_path(args.outer_large_opencv_intrinsics_dir)
    outer_large_intrinsic_report_dir = output_root / "reports" / "outer_intrinsics_outer_large_marker"
    outer_intrinsic_metrics_tsv = (
        resolve_path(args.outer_intrinsic_metrics_tsv)
        if args.outer_intrinsic_metrics_tsv
        else outer_large_intrinsic_report_dir / "camera_metrics.tsv"
        if not args.bridge_only
        else first_existing([
            DEFAULT_OUTER_INTRINSIC_METRICS_TSV,
            DEFAULT_HTTP_ROOT / "current_calibration/reports/06_outer_intrinsics_outer_large_marker/camera_metrics.tsv",
        ])
    )
    bridge_root = output_root / "inner_bridge"
    large_marker_ba_state_dir = (
        bridge_root / args.large_marker_sequence /
        f"fixed_points_joint_ba_stride{args.large_frame_stride}_{args.large_bridge_schur_mode}_v1"
    )
    unified_artifact_dir = output_root / "calibration_artifacts" / "studio_32_cameras_current"
    marker_correspondence_dir = output_root / "marker_correspondences"
    advanced_correspondence_root = output_root / "advanced_correspondence_viewer_v1"
    current_output_dir = resolve_path(args.current_output_dir)
    outer_large_qc_root = resolve_path(args.outer_large_qc_root)
    if args.whole_qc_root:
        whole_qc_root = resolve_path(args.whole_qc_root)
    else:
        whole_qc_stats = first_existing([
            whole_data_root / "whole_outer24_filtered_min4_fullres_min4cam" / "per_camera_stats.tsv",
            whole_data_root / "whole_outer24_filtered_min4_hybrid_min4cam" / "per_camera_stats.tsv",
        ])
        whole_qc_root = Path(whole_qc_stats).parent if whole_qc_stats else ""
    whole_data_report = (
        resolve_path(args.whole_data_report)
        if args.whole_data_report
        else whole_data_root / "whole_outer24_filtered_min4_hybrid_min4cam" / "index.html"
    )
    return {
        "whole_data_root": whole_data_root,
        "inner_data_root": inner_data_root,
        "output_root": output_root,
        "outer_wrapper_root": outer_wrapper_root,
        "outer_frame_face_dir": outer_frame_face_dir,
        "outer_pose_yaml": outer_pose_yaml,
        "outer_intrinsics_dir": outer_intrinsics_dir,
        "outer_intrinsic_metrics_tsv": outer_intrinsic_metrics_tsv,
        "outer_large_opencv_intrinsics_dir": outer_large_opencv_intrinsics_dir,
        "outer_large_intrinsic_report_dir": outer_large_intrinsic_report_dir,
        "outer_large_qc_root": outer_large_qc_root,
        "whole_qc_root": whole_qc_root,
        "whole_data_report": whole_data_report,
        "bridge_root": bridge_root,
        "bridge_viewer": bridge_root / "combined_studio_rig_viewer_v1" / "index.html",
        "bridge_pose_yaml": large_marker_ba_state_dir / "camera_tr_rig.yaml",
        "bridge_intrinsics_dir": bridge_root / "planned_inputs" / "bridge_all32_fixed_intrinsics",
        "unified_artifact_dir": unified_artifact_dir,
        "unified_camera_yaml": unified_artifact_dir / "studio_32_cameras.yaml",
        "large_marker_dataset": bridge_root / args.large_marker_sequence / f"features_parallel_pattern0_bridge_stride{args.large_frame_stride}_v1.bin",
        "large_marker_state_dir": large_marker_ba_state_dir,
        "large_marker_manifest": bridge_root / "planned_inputs" / "large_marker_usable_manifest.tsv",
        "small_marker_dataset": bridge_root / args.small_marker_sequence / f"features_pattern3_grid4_stride{args.small_frame_stride}_fast_v1.bin",
        "small_marker_state_dir": bridge_root / args.small_marker_sequence / "fixed_intrinsic_small_grid4_quality_probe_v1",
        "small_marker_manifest": bridge_root / "planned_inputs" / "small_marker_usable_manifest.tsv",
        "marker_correspondence_dir": marker_correspondence_dir,
        "large_marker_correspondence_tsv": marker_correspondence_dir / "large_marker_correspondences.tsv",
        "large_marker_correspondence_summary": marker_correspondence_dir / "large_marker_correspondences.summary.json",
        "small_marker_correspondence_tsv": marker_correspondence_dir / "small_marker_correspondences.tsv",
        "small_marker_correspondence_summary": marker_correspondence_dir / "small_marker_correspondences.summary.json",
        "outer_observation_residuals_tsv": outer_frame_face_dir / "diagnostics" / "observation_residuals.tsv",
        "outer_frame_face_pose_yaml": outer_frame_face_dir / "rig_tr_frame_face.yaml",
        "large_pnp_dir": (
            bridge_root / args.large_marker_sequence /
            f"fixed_intrinsic_bridge_pnp_stride{args.large_frame_stride}_v1"
        ),
        "large_ba_dir": large_marker_ba_state_dir,
        "small_pnp_dir": bridge_root / args.small_marker_sequence / "fixed_intrinsic_small_grid4_quality_probe_v1",
        "viewer_assets_dir": bridge_root / "combined_studio_rig_viewer_v1",
        "advanced_correspondence_root": advanced_correspondence_root,
        "advanced_correspondence_viewer": advanced_correspondence_root / "index.html",
        "current_output_dir": current_output_dir,
    }


def outer_command(args, paths):
    command = [
        sys.executable,
        repo_root() / "scripts/calib/run_outer_tower_recalib_pipeline.py",
        "--data-root", paths["whole_data_root"],
        "--output-root", paths["outer_wrapper_root"],
        "--frame-face-output-dir", paths["outer_frame_face_dir"],
        "--run-frame-face-refine",
        "--frame-face-refine-preset", args.outer_preset,
        "--run-reports",
        "--run-tag", args.run_tag,
    ]
    if args.outer_frame_face_prior_pose_yaml:
        command.extend(["--frame-face-prior-pose-yaml", resolve_path(args.outer_frame_face_prior_pose_yaml)])
    if args.outer_frame_face_intrinsics_dir:
        command.extend(["--frame-face-intrinsics-dir", resolve_path(args.outer_frame_face_intrinsics_dir)])
    if args.force:
        command.append("--force")
    return command


def bridge_command(args, paths):
    command = [
        sys.executable,
        repo_root() / "scripts/calib/run_inner_bridge_recalib_pipeline.py",
        "--data-root", paths["inner_data_root"],
        "--output-root", paths["bridge_root"],
        "--outer-final-pose-yaml", paths["outer_pose_yaml"],
        "--outer-intrinsics", paths["outer_intrinsics_dir"],
        "--large-marker-sequence", args.large_marker_sequence,
        "--large-inner-marker-sequence", args.large_inner_marker_sequence,
        "--small-marker-sequence", args.small_marker_sequence,
        "--large-frame-stride", str(args.large_frame_stride),
        "--large-bridge-schur-mode", args.large_bridge_schur_mode,
        "--large-bridge-max-ba-iterations", str(args.large_bridge_max_ba_iterations),
        "--large-bridge-model", args.large_bridge_model,
        "--small-frame-stride", str(args.small_frame_stride),
        "--run-large-bridge",
        "--run-reports",
        "--run-tag", args.run_tag,
    ]
    if args.camera_calibration_binary:
        command.extend(["--camera-calibration-binary", resolve_path(args.camera_calibration_binary)])
    if paths["outer_intrinsic_metrics_tsv"]:
        command.extend(["--outer-intrinsic-metrics-tsv", paths["outer_intrinsic_metrics_tsv"]])
    if args.inner_prior:
        command.extend(["--inner-prior", resolve_path(args.inner_prior)])
    if args.outer_prior:
        command.extend(["--outer-prior", resolve_path(args.outer_prior)])
    if args.run_large_inner_init:
        command.append("--run-large-inner-init")
    if args.run_small_quality:
        command.append("--run-small-fixed-rig-quality")
    if args.force:
        command.append("--force")
    return command


def publish_command(args, paths):
    http_root = resolve_path(args.http_root)
    return [
        sys.executable,
        repo_root() / "scripts/ops/publish_t0_clean_calib_reports.py",
        "--root", http_root,
        "--base-url", args.report_url_base,
        "--run-tag", args.run_tag,
        "--current-dir", paths["current_output_dir"],
        "--outer-large-intrinsic-report", paths["outer_large_intrinsic_report_dir"],
        "--outer-large-qc-root", paths["outer_large_qc_root"],
        "--whole-qc-root", paths["whole_qc_root"],
    ]


def export_unified_command(args, paths):
    return [
        sys.executable,
        repo_root() / "scripts/calib/export_combined_studio_extrinsics.py",
        "--inner-bridge-pose-yaml", paths["bridge_pose_yaml"],
        "--outer-final-pose-yaml", paths["outer_pose_yaml"],
        "--intrinsics-dir", paths["bridge_intrinsics_dir"],
        "--output-dir", paths["unified_artifact_dir"],
        "--run-tag", args.run_tag,
        "--viewer-url", report_url(paths["bridge_viewer"], args.http_root, args.report_url_base),
    ]


def advanced_correspondence_command(args, paths):
    return [
        sys.executable,
        repo_root() / "scripts/calib/generate_studio_correspondence_viewer.py",
        "--output-dir", paths["advanced_correspondence_root"],
        "--studio32-yaml", paths["unified_camera_yaml"],
        "--outer-observation-residuals-tsv", paths["outer_observation_residuals_tsv"],
        "--outer-frame-face-pose-yaml", paths["outer_frame_face_pose_yaml"],
        "--large-correspondence-tsv", paths["large_marker_correspondence_tsv"],
        "--small-correspondence-tsv", paths["small_marker_correspondence_tsv"],
        "--large-pnp-dir", paths["large_pnp_dir"],
        "--small-pnp-dir", paths["small_pnp_dir"],
        "--viewer-assets-dir", paths["viewer_assets_dir"],
    ]


def outer_large_intrinsic_report_command(args, paths):
    return [
        sys.executable,
        repo_root() / "scripts/calib/generate_opencv_intrinsics_report.py",
        "--intrinsics-dir", paths["outer_large_opencv_intrinsics_dir"],
        "--output-dir", paths["outer_large_intrinsic_report_dir"],
    ]


def marker_correspondence_command(args, paths, dataset_name):
    if dataset_name == "large":
        dataset = paths["large_marker_dataset"]
        state_dir = paths["large_marker_state_dir"]
        manifest = paths["large_marker_manifest"]
        output_tsv = paths["large_marker_correspondence_tsv"]
        summary = paths["large_marker_correspondence_summary"]
        camera_index_offset = 0
    elif dataset_name == "small":
        dataset = paths["small_marker_dataset"]
        state_dir = paths["small_marker_state_dir"]
        manifest = paths["small_marker_manifest"]
        output_tsv = paths["small_marker_correspondence_tsv"]
        summary = paths["small_marker_correspondence_summary"]
        camera_index_offset = 24
    else:
        raise ValueError(f"unknown correspondence dataset {dataset_name}")
    return [
        sys.executable,
        repo_root() / "scripts/calib/export_calibration_correspondence_residuals.py",
        "--dataset", dataset,
        "--state-dir", state_dir,
        "--output-tsv", output_tsv,
        "--summary-json", summary,
        "--dataset-name", dataset_name,
        "--manifest", manifest,
        "--reference-studio32-yaml", paths["unified_camera_yaml"],
        "--camera-index-offset", str(camera_index_offset),
    ]


def make_stage(name, requested, command):
    return {
        "name": name,
        "requested": bool(requested),
        "status": "planned" if requested else "not_requested",
        "commands": [command_string(command)] if requested and command else [],
        "log_path": "",
        "returncode": None,
        "started_at": "",
        "finished_at": "",
        "duration_s": 0.0,
    }


def build_stages(args, paths):
    run_outer = not args.bridge_only
    run_bridge = not args.outer_only
    run_export = run_bridge
    run_marker_correspondences = run_bridge
    run_advanced = run_bridge
    run_outer_intrinsic_report = run_outer
    run_publish = args.publish_current and run_bridge
    return [
        make_stage("outer_tower", run_outer, outer_command(args, paths)),
        make_stage("generate_outer_intrinsic_report", run_outer_intrinsic_report, outer_large_intrinsic_report_command(args, paths)),
        make_stage("inner_bridge", run_bridge, bridge_command(args, paths)),
        make_stage("export_unified_cameras", run_export, export_unified_command(args, paths)),
        make_stage("export_large_marker_correspondences", run_marker_correspondences, marker_correspondence_command(args, paths, "large")),
        make_stage("export_small_marker_correspondences", run_marker_correspondences, marker_correspondence_command(args, paths, "small")),
        make_stage("generate_advanced_correspondence_viewer", run_advanced, advanced_correspondence_command(args, paths)),
        make_stage("publish_current", run_publish, publish_command(args, paths)),
    ]


def execute_stages(stages, dry_run, output_root=None):
    if dry_run:
        return stages
    logs_dir = Path(output_root) / "logs" if output_root else None
    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        if not stage["requested"]:
            continue
        stage["started_at"] = utc_now()
        start = time.time()
        argv = shlex.split(stage["commands"][0])
        log_path = logs_dir / f"{stage['name']}.log" if logs_dir else None
        if log_path:
            stage["log_path"] = str(log_path)
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + stage["commands"][0] + "\n\n")
                completed = subprocess.run(
                    argv,
                    cwd=repo_root(),
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        else:
            completed = subprocess.run(argv, cwd=repo_root(), text=True)
        stage["returncode"] = completed.returncode
        stage["finished_at"] = utc_now()
        stage["duration_s"] = round(time.time() - start, 3)
        stage["status"] = "complete" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            break
    return stages


def render_index(summary):
    def esc(value):
        return html.escape(str(value))

    rows = "\n".join(
        "<tr>"
        f"<td>{esc(stage['name'])}</td>"
        f"<td>{esc(stage['status'])}</td>"
        f"<td>{esc(stage['duration_s'])}</td>"
        f"<td><code>{esc(stage.get('log_path', ''))}</code></td>"
        f"<td><code>{esc(stage['commands'][0] if stage['commands'] else '')}</code></td>"
        "</tr>"
        for stage in summary["stages"]
    )
    links = "\n".join(
        f"<li><a href='{esc(value)}'>{esc(key)}</a></li>"
        for key, value in summary["report_urls"].items()
        if value
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Studio Calibration Pipeline Run</title>
  <style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 28px; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 7px 9px; text-align: left; vertical-align: top; }}
th {{ background: #f1f1ed; }}
code {{ background: #f1f1ed; padding: 2px 4px; border-radius: 4px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Studio Calibration Pipeline Run</h1>
  <p>Mode: <strong>{esc(summary['mode'])}</strong>. Run tag: <code>{esc(summary['run_tag'])}</code>. Total duration: <strong>{esc(summary['duration_s'])} s</strong>.</p>
  <h2>Reports</h2>
  <ul>{links}</ul>
  <h2>Stages</h2>
  <table><thead><tr><th>Stage</th><th>Status</th><th>Duration s</th><th>Log</th><th>Command</th></tr></thead><tbody>{rows}</tbody></table>
</body>
</html>
"""


def write_outputs(args, paths, stages, started_at, finished_at, duration_s):
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    http_root = resolve_path(args.http_root)
    stage_durations = {stage["name"]: stage["duration_s"] for stage in stages}
    summary = {
        "mode": "dry_run" if args.dry_run else "execute",
        "run_tag": args.run_tag,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_s, 3),
        "run_timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "total_duration_s": round(duration_s, 3),
            "stage_count": len(stages),
            "stage_durations_s": stage_durations,
        },
        "inputs": {
            "whole_data_root": path_status(paths["whole_data_root"]),
            "inner_data_root": path_status(paths["inner_data_root"]),
        },
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key not in {"whole_data_root", "inner_data_root"}
        },
        "report_urls": {
            "pipeline_index": report_url(output_root / "index.html", http_root, args.report_url_base),
            "outer_report": report_url(paths["outer_wrapper_root"] / "index.html", http_root, args.report_url_base),
            "outer_intrinsic_report": report_url(paths["outer_large_intrinsic_report_dir"] / "index.html", http_root, args.report_url_base),
            "bridge_report": report_url(paths["bridge_root"] / "final_report" / "index.html", http_root, args.report_url_base),
            "unified_viewer": report_url(paths["bridge_viewer"], http_root, args.report_url_base),
            "unified_camera_yaml": report_url(paths["unified_camera_yaml"], http_root, args.report_url_base),
            "large_marker_correspondences": report_url(paths["large_marker_correspondence_tsv"], http_root, args.report_url_base),
            "small_marker_correspondences": report_url(paths["small_marker_correspondence_tsv"], http_root, args.report_url_base),
            "advanced_correspondence_viewer": report_url(paths["advanced_correspondence_viewer"], http_root, args.report_url_base),
            "current_entry": report_url(paths["current_output_dir"] / "index.html", http_root, args.report_url_base)
            if args.publish_current else "",
        },
        "stages": stages,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "index.html").write_text(render_index(summary), encoding="utf-8")
    return summary


def safe_print_json(data):
    try:
        print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the studio calibration pipeline as one command: outer whole-tower refine, "
            "inner/outer large-marker bridge, and optional current report publication."
        )
    )
    parser.add_argument("--whole-data-root", type=Path, default=DEFAULT_WHOLE_DATA_ROOT)
    parser.add_argument("--inner-data-root", type=Path, default=DEFAULT_INNER_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-tag", default="latest")
    parser.add_argument("--outer-preset", default="wide50_then_gate6")
    parser.add_argument("--large-marker-sequence", default="large_marker_bridge_all32")
    parser.add_argument("--large-inner-marker-sequence", default="large_marker_inner8")
    parser.add_argument("--small-marker-sequence", default="small_marker_inner8")
    parser.add_argument("--large-frame-stride", type=int, default=1)
    parser.add_argument("--large-bridge-model", default="central_opencv")
    parser.add_argument("--large-bridge-max-ba-iterations", type=int, default=80)
    parser.add_argument(
        "--large-bridge-schur-mode",
        choices=["dense", "dense_cuda", "dense_onthefly", "sparse", "sparse_onthefly"],
        default="dense",
    )
    parser.add_argument("--small-frame-stride", type=int, default=4)
    parser.add_argument("--run-large-inner-init", action="store_true")
    parser.add_argument("--run-small-quality", action="store_true")
    parser.add_argument(
        "--inner-prior",
        type=Path,
        default=DEFAULT_INNER_PRIOR,
        help="Warm-start inner state directory passed to the inner/bridge wrapper.",
    )
    parser.add_argument(
        "--outer-prior",
        type=Path,
        default=DEFAULT_OUTER_COLMAP_PRIOR,
        help="Legacy outer COLMAP images.txt prior passed to the inner/bridge wrapper.",
    )
    parser.add_argument(
        "--outer-frame-face-prior-pose-yaml",
        type=Path,
        default=DEFAULT_OUTER_FRAME_FACE_PRIOR_POSE_YAML,
        help="Previous outer frame-face pose YAML used as the delta prior for outer refinement.",
    )
    parser.add_argument(
        "--outer-frame-face-intrinsics-dir",
        type=Path,
        default=DEFAULT_OUTER_FRAME_FACE_INTRINSICS_DIR,
        help="Previous/refined outer intrinsics directory used by the outer frame-face refinement.",
    )
    parser.add_argument(
        "--outer-final-pose-yaml",
        type=Path,
        default=None,
        help="Override final outer pose YAML consumed by bridge/export stages.",
    )
    parser.add_argument(
        "--outer-final-intrinsics-dir",
        type=Path,
        default=None,
        help="Override final outer intrinsics directory consumed by bridge stages.",
    )
    parser.add_argument(
        "--outer-intrinsic-metrics-tsv",
        type=Path,
        default=None,
        help=(
            "Outer large-marker intrinsic camera_metrics.tsv used by the unified viewer "
            "for intrinsic residual columns."
        ),
    )
    parser.add_argument(
        "--outer-large-opencv-intrinsics-dir",
        type=Path,
        default=DEFAULT_OUTER_LARGE_OPENCV_INTRINSICS_DIR,
        help="Outer large-marker OpenCV intrinsic calibration directory used to regenerate the canonical outer intrinsic report.",
    )
    parser.add_argument(
        "--outer-large-qc-root",
        type=Path,
        default=DEFAULT_OUTER_LARGE_QC_ROOT,
        help="Outer large-marker distributed QC root used by the current clean report publisher.",
    )
    parser.add_argument(
        "--whole-qc-root",
        type=Path,
        default=None,
        help="Whole/tower distributed QC root used by the current clean report publisher.",
    )
    parser.add_argument(
        "--whole-data-report",
        type=Path,
        default=None,
        help="Promoted whole data collection report passed into the current-calibration entry publisher.",
    )
    parser.add_argument(
        "--camera-calibration-binary",
        type=Path,
        default=None,
        help=(
            "Optional camera_calibration C++ binary forwarded to the inner/bridge wrapper. "
            "When omitted, the inner wrapper probes the current checkout and known T0 release builds."
        ),
    )
    parser.add_argument("--outer-only", action="store_true")
    parser.add_argument("--bridge-only", action="store_true")
    parser.add_argument("--publish-current", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--http-root", type=Path, default=DEFAULT_HTTP_ROOT)
    parser.add_argument("--report-url-base", default=DEFAULT_REPORT_URL_BASE)
    parser.add_argument("--panel-url", default=DEFAULT_PANEL_URL)
    parser.add_argument("--current-output-dir", type=Path, default=DEFAULT_HTTP_ROOT / "current_calibration")
    args = parser.parse_args()
    if args.outer_only and args.bridge_only:
        parser.error("--outer-only and --bridge-only are mutually exclusive")
    if args.large_frame_stride < 1:
        parser.error("--large-frame-stride must be >= 1")
    if args.large_bridge_max_ba_iterations < 0:
        parser.error("--large-bridge-max-ba-iterations must be non-negative")
    if args.small_frame_stride < 1:
        parser.error("--small-frame-stride must be >= 1")
    return args


def main():
    args = parse_args()
    started_at = utc_now()
    start = time.time()
    paths = build_paths(args)
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    stages = build_stages(args, paths)
    stages = execute_stages(stages, args.dry_run, paths["output_root"])
    finished_at = utc_now()
    summary = write_outputs(args, paths, stages, started_at, finished_at, time.time() - start)
    safe_print_json({
        "summary_json": str(paths["output_root"] / "summary.json"),
        "index_html": str(paths["output_root"] / "index.html"),
        "pipeline_url": summary["report_urls"]["pipeline_index"],
        "unified_viewer_url": summary["report_urls"]["unified_viewer"],
        "unified_camera_yaml_url": summary["report_urls"]["unified_camera_yaml"],
        "current_entry_url": summary["report_urls"]["current_entry"],
        "stage_status": {stage["name"]: stage["status"] for stage in stages},
    })
    if args.dry_run:
        print("\nPlanned commands:")
        for stage in stages:
            if stage["requested"]:
                print(f"[{stage['name']}] {stage['commands'][0]}")
    if any(stage["status"] == "failed" for stage in stages):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
