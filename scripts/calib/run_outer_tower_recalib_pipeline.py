#!/usr/bin/env python3
"""Pipeline wrapper for outer AprilTag-tower recalibration."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time


T0_DATA_ROOT = Path("/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3")
T0_HTTP_ROOT = Path("/home/ubuntu/calib_data")
T0_HTTP_BASE = "http://192.168.2.0:9899"
DEFAULT_LEGACY_ANCHOR_LABEL_TO_POSE_INDEX = "4-1:8,4-2:9,4-3:10"
DEFAULT_ALL32_ANCHOR_LABEL_TO_POSE_INDEX = "4-1:9,4-2:10,4-3:11"
# Production whole-tower BA should use physical black-tile outer corners:
# 8 cm black tile footprint and 2 cm white gap. The older 0.067104... effective
# detector-square geometry is only valid for legacy datasets that stored raw
# OpenCV AprilTag inner-detector corners.
DEFAULT_TOWER_TAG_CENTER_PITCH_M = 0.10
DEFAULT_TOWER_DETECTOR_TAG_SIZE_M = 0.08
FRAME_FACE_REFINE_PRESETS = {
    "wide200_then_gate6": {
        "pnp_ransac_iterations": 1000,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 5.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": 200.0,
        "observation_residual_gate_px": 6.0,
        "optimizer_residual_clip_px": 20.0,
        "outer_iterations": 12,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 8,
    },
    "wide200_then_gate6_flex_faces": {
        "tower_model": "flex_yaw_offset_tower",
        "optimize_tower_face_width": False,
        "pnp_ransac_iterations": 1000,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 5.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": 200.0,
        "observation_residual_gate_px": 6.0,
        "optimizer_residual_clip_px": 20.0,
        "outer_iterations": 14,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 8,
        "flex_face_yaw_sigma_deg": 3.0,
        "flex_face_yaw_max_deg": 8.0,
        "flex_face_radial_offset_sigma_m": 0.015,
        "flex_face_radial_offset_max_m": 0.05,
        "flex_face_tangent_offset_sigma_m": 0.010,
        "flex_face_tangent_offset_max_m": 0.02,
        "flex_face_adjacent_angle_min_deg": 30.0,
        "flex_face_adjacent_angle_max_deg": 60.0,
        "flex_face_geometry_block_iterations": 6,
    },
    "wide50_then_gate4": {
        "pnp_ransac_iterations": 1000,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 5.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": 50.0,
        "observation_residual_gate_px": 4.0,
        "optimizer_residual_clip_px": 20.0,
        "outer_iterations": 12,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 8,
    },
    "wide50_then_gate6": {
        "pnp_ransac_iterations": 1000,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 5.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": 50.0,
        "observation_residual_gate_px": 6.0,
        "optimizer_residual_clip_px": 20.0,
        "outer_iterations": 12,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 8,
    },
    "wide50_then_gate10": {
        "pnp_ransac_iterations": 1000,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 5.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": 50.0,
        "observation_residual_gate_px": 10.0,
        "optimizer_residual_clip_px": 30.0,
        "outer_iterations": 12,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 8,
    },
    "wide50_then_gate16": {
        "pnp_ransac_iterations": 1000,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 5.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": 50.0,
        "observation_residual_gate_px": 16.0,
        "optimizer_residual_clip_px": 40.0,
        "outer_iterations": 12,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 8,
    },
    "wide50_then_gate16_flex_faces": {
        "tower_model": "flex_yaw_offset_tower",
        "optimize_tower_face_width": False,
        "pnp_ransac_iterations": 1000,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 5.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": 50.0,
        "observation_residual_gate_px": 16.0,
        "optimizer_residual_clip_px": 40.0,
        "outer_iterations": 14,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 8,
        "flex_face_yaw_sigma_deg": 3.0,
        "flex_face_yaw_max_deg": 8.0,
        "flex_face_radial_offset_sigma_m": 0.015,
        "flex_face_radial_offset_max_m": 0.05,
        "flex_face_tangent_offset_sigma_m": 0.010,
        "flex_face_tangent_offset_max_m": 0.02,
        "flex_face_adjacent_angle_min_deg": 30.0,
        "flex_face_adjacent_angle_max_deg": 60.0,
        "flex_face_geometry_block_iterations": 6,
    },
    "gate20": {
        "pnp_ransac_iterations": 100,
        "pnp_ransac_threshold_px": 5.0,
        "max_pnp_median_error_px": 6.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": None,
        "observation_residual_gate_px": 20.0,
        "optimizer_residual_clip_px": 50.0,
        "outer_iterations": 8,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 64,
    },
    "gate10_coverage": {
        "pnp_ransac_iterations": 100,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 5.0,
        "min_frame_face_observations": 8,
        "initial_observation_residual_gate_px": None,
        "observation_residual_gate_px": 10.0,
        "optimizer_residual_clip_px": 30.0,
        "outer_iterations": 10,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 16,
    },
    "gate10": {
        "pnp_ransac_iterations": 100,
        "pnp_ransac_threshold_px": 4.0,
        "max_pnp_median_error_px": 4.0,
        "min_frame_face_observations": 12,
        "initial_observation_residual_gate_px": None,
        "observation_residual_gate_px": 10.0,
        "optimizer_residual_clip_px": 30.0,
        "outer_iterations": 8,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 64,
    },
    "gate6": {
        "pnp_ransac_iterations": 100,
        "pnp_ransac_threshold_px": 3.0,
        "max_pnp_median_error_px": 3.0,
        "min_frame_face_observations": 12,
        "initial_observation_residual_gate_px": None,
        "observation_residual_gate_px": 6.0,
        "optimizer_residual_clip_px": 20.0,
        "outer_iterations": 8,
        "block_iterations": 8,
        "min_camera_observations_for_delta": 64,
    },
}


def repo_root():
    return Path(__file__).resolve().parents[2]


def as_abs(path):
    return Path(path).expanduser().resolve()


def path_status(path):
    if path is None:
        return {"path": "", "exists": False, "kind": "unset"}
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


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def read_tsv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso_timestamp(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def duration_s(value):
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def shell_join(command):
    return shlex.join(str(item) for item in command)


def git_metadata(root):
    root = Path(root)

    def run_git(args):
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            return "", str(exc)
        if completed.returncode != 0:
            return completed.stdout.strip(), completed.stderr.strip()
        return completed.stdout.strip(), ""

    commit, commit_error = run_git(["rev-parse", "HEAD"])
    branch, branch_error = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status, status_error = run_git(["status", "--short"])
    status_lines = status.splitlines()
    return {
        "repo_root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_lines),
        "status_short": status_lines[:200],
        "status_short_truncated": len(status_lines) > 200,
        "errors": [error for error in [commit_error, branch_error, status_error] if error],
    }


def pipeline_provenance():
    root = repo_root()
    return {
        "script": str(Path(__file__).resolve(strict=False)),
        "argv": shell_join(sys.argv),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "git": git_metadata(root),
    }


def report_url(path):
    path = Path(path)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    try:
        rel = resolved.relative_to(T0_HTTP_ROOT)
        return f"{T0_HTTP_BASE}/{rel.as_posix()}"
    except ValueError:
        return resolved.as_uri()


def choose_first_existing(candidates, fallback):
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return Path(fallback)


def default_anchor_pose_yaml(data_root):
    return choose_first_existing(
        [
            data_root
            / "recalib_pipelines"
            / "fast_inner_bridge"
            / "latest"
            / "bridge_colmap_inner_refined_v1"
            / "camera_tr_inner_refined_plus_outer_topdown.yaml",
            data_root
            / "large_marker_bridge_4topdown_v1"
            / "bridge_colmap_inner_refined_v1"
            / "camera_tr_inner_refined_plus_outer_topdown.yaml",
        ],
        data_root
        / "recalib_pipelines"
        / "fast_inner_bridge"
        / "latest"
        / "bridge_colmap_inner_refined_v1"
        / "camera_tr_inner_refined_plus_outer_topdown.yaml",
    )


def default_anchor_label_to_pose_index(anchor_pose_yaml):
    text = str(anchor_pose_yaml)
    if "fast_inner_bridge" in text or "large_marker_bridge_all32" in text:
        return DEFAULT_ALL32_ANCHOR_LABEL_TO_POSE_INDEX
    return DEFAULT_LEGACY_ANCHOR_LABEL_TO_POSE_INDEX


def default_output_root(whole_dir, data_root=None):
    if whole_dir.exists():
        return whole_dir / "outer_tower_recalib_pipeline_v1"
    if data_root and Path(data_root).exists():
        return Path(data_root) / "recalib_pipelines" / "outer_tower" / "outer_tower_recalib_pipeline_v1"
    return Path("/tmp/camera_calibration_outer_tower_recalib_pipeline_dry_run")


def build_paths(args):
    data_root_arg = args.data_root or args.stage_root
    data_root = as_abs(data_root_arg) if data_root_arg else T0_DATA_ROOT
    whole_dir = (
        as_abs(args.whole_dir)
        if args.whole_dir
        else data_root / args.whole_sequence
    )
    output_root = as_abs(args.output_root) if args.output_root else default_output_root(whole_dir, data_root)

    dataset = choose_first_existing(
        [
            whole_dir / "tower_features_parallel_fixed_geometry_v1.bin",
            whole_dir / "tower_features_rot180.bin",
        ],
        whole_dir / "tower_features_parallel_fixed_geometry_v1.bin",
    )
    manifest = whole_dir / "manifest.tsv"
    anchor_pose_yaml = as_abs(args.anchor_pose_yaml) if args.anchor_pose_yaml else default_anchor_pose_yaml(data_root)
    anchor_label_to_pose_index = args.anchor_label_to_pose_index or default_anchor_label_to_pose_index(anchor_pose_yaml)
    pnp_views = whole_dir / "fixed_intrinsic_pnp_colmap_fallback_v1" / "pnp_views.tsv"
    outer_colmap_images_txt = (
        data_root
        / "colmap_outer24_firstframe_colmap404_v3"
        / "fixed_intrinsics"
        / "sparse_txt_final24_fixedK_ba"
        / "images.txt"
    )

    existing_frame_runs = whole_dir / "outer_colmap_frame_vote_32_v1"
    existing_ransac_dir = whole_dir / "outer_colmap_frame_vote_32_ransac_v1"
    existing_side_dir = whole_dir / "outer_colmap_frame_vote_32_side_prior_v1"
    existing_tag_dir = choose_first_existing(
        [
            output_root / "tag_refine_robust",
            data_root / "recalib_pipelines" / "outer_tower" / "latest" / "tag_refine_robust",
            whole_dir / "outer_tower_side_prior_tag_refine_clip500_loose_v1",
            whole_dir / "outer_tower_side_prior_tag_refine_robust_v2",
        ],
        output_root / "tag_refine_robust",
    )

    previous_outer_rig = (
        as_abs(args.previous_outer_rig)
        if args.previous_outer_rig
        else existing_side_dir / "camera_tr_rig_side_prior.yaml"
    )
    previous_intrinsics_dir = (
        as_abs(args.previous_intrinsics_dir)
        if args.previous_intrinsics_dir
        else choose_first_existing(
            [
                whole_dir / "intrinsics_tower_safe_rms5_v1",
                whole_dir / "intrinsics_tower_fixed_geometry_v1",
            ],
            whole_dir / "intrinsics_tower_safe_rms5_v1",
        )
    )
    frame_face_stage_roots = [
        data_root / "whole_outer24_filtered_min4_fullres_min4cam",
        data_root / "whole_outer24_filtered_min4_hybrid_min4cam",
        whole_dir,
    ]
    default_frame_face_stage_root = choose_first_existing(
        [
            candidate
            for root in frame_face_stage_roots
            for candidate in [
                root / "opencv_tower_dataset_black_tile_red_scale_edge.bin",
            ]
        ],
        frame_face_stage_roots[0] / "opencv_tower_dataset_black_tile_red_scale_edge.bin",
    ).parent
    default_frame_face_root = choose_first_existing(
        [root / "pnp_inlier_filter_facewidth025_optwidth_v1" for root in frame_face_stage_roots],
        default_frame_face_stage_root / "pnp_inlier_filter_facewidth025_optwidth_v1",
    )
    frame_face_dataset = (
        as_abs(args.frame_face_dataset)
        if args.frame_face_dataset
        else choose_first_existing(
            [
                whole_dir / "opencv_tower_dataset_black_tile_red_scale_edge.bin",
                *[root / "opencv_tower_dataset_black_tile_red_scale_edge.bin" for root in frame_face_stage_roots],
            ],
            default_frame_face_stage_root / "opencv_tower_dataset_black_tile_red_scale_edge.bin",
        )
    )
    frame_face_manifest = (
        as_abs(args.frame_face_manifest)
        if args.frame_face_manifest
        else choose_first_existing(
            [
                frame_face_dataset.parent / "manifest.tsv",
                frame_face_dataset.parent.parent / "manifest.tsv",
                whole_dir / "manifest.tsv",
                manifest,
            ],
            frame_face_dataset.parent / "manifest.tsv",
        )
    )
    frame_face_prior_pose_yaml = (
        as_abs(args.frame_face_prior_pose_yaml)
        if args.frame_face_prior_pose_yaml
        else choose_first_existing(
            [
                default_frame_face_root / "tag_refine_safe5coeff_percam_fxfycxcy_optwidth_v1" / "camera_tr_rig_prior.yaml",
                default_frame_face_root / "tag_refine_safe5coeff_percam_fxfycxcy_optwidth_v1" / "camera_tr_rig_delta_refined_accepted.yaml",
                default_frame_face_root / "selected_outer_frame_face_current" / "camera_tr_rig_delta_refined.yaml",
                default_frame_face_root / "frame_face_planes_all616_fixed_gate20_v1" / "camera_tr_rig_delta_refined.yaml",
                default_frame_face_root / "frame_face_planes_all616_weakK_then_fixed_gate20_v1" / "camera_tr_rig_delta_refined.yaml",
            ],
            previous_outer_rig,
        )
    )
    frame_face_intrinsics_dir = (
        as_abs(args.frame_face_intrinsics_dir)
        if args.frame_face_intrinsics_dir
        else choose_first_existing(
            [
                default_frame_face_root / "frame_face_planes_all616_weakK_then_fixed_gate20_v1" / "intrinsics_refined",
                default_frame_face_root / "selected_outer_frame_face_current" / "intrinsics_refined",
                default_frame_face_root / "frame_face_planes_all616_fixed_gate20_v1" / "intrinsics_refined",
            ],
            previous_intrinsics_dir,
        )
    )
    frame_face_refine_dir = (
        as_abs(args.frame_face_output_dir)
        if args.frame_face_output_dir
        else output_root / f"frame_face_refine_{args.frame_face_refine_preset}"
    )

    work = {
        "coverage_dir": output_root / "coverage_gate",
        "colmap_frame_dir": output_root / "colmap_frame_vote_runs",
        "colmap_ransac_dir": output_root / "colmap_ransac_vote",
        "side_prior_dir": output_root / "side_prior",
        "pnp_consensus_dir": output_root / "pnp_pose_consensus",
        "tag_refine_dir": output_root / "tag_refine_robust",
        "residual_tail_dir": output_root / "residual_tail_report",
        "intrinsic_feature_coverage_dir": output_root / "intrinsic_feature_coverage_report",
        "viewer_dir": output_root / "viewer",
        "quality_report_dir": output_root / "quality_report",
        "final_report_dir": output_root / "final_report",
        "logs_dir": output_root / "logs",
    }

    return {
        "data_root": data_root,
        "whole_dir": whole_dir,
        "output_root": output_root,
        "whole_sequence": args.whole_sequence,
        "dataset": dataset,
        "manifest": manifest,
        "anchor_pose_yaml": anchor_pose_yaml,
        "bridge_summary_json": anchor_pose_yaml.parent / "bridge_summary.json",
        "anchor_label_to_pose_index": anchor_label_to_pose_index,
        "pnp_views": pnp_views,
        "pnp_consensus_views": output_root / "pnp_pose_consensus" / "pnp_views_consensus.tsv",
        "outer_colmap_images_txt": outer_colmap_images_txt,
        "existing_frame_runs": existing_frame_runs,
        "existing_ransac_dir": existing_ransac_dir,
        "existing_side_dir": existing_side_dir,
        "existing_tag_dir": existing_tag_dir,
        "previous_outer_rig": previous_outer_rig,
        "previous_intrinsics_dir": previous_intrinsics_dir,
        "frame_face_dataset": frame_face_dataset,
        "frame_face_manifest": frame_face_manifest,
        "frame_face_prior_pose_yaml": frame_face_prior_pose_yaml,
        "frame_face_intrinsics_dir": frame_face_intrinsics_dir,
        "frame_face_refine_dir": frame_face_refine_dir,
        **work,
    }


def summarize_manifest(path):
    path = Path(path)
    if not path.is_file():
        return {
            "status": "missing",
            "manifest": str(path),
            "camera_count": 0,
            "frame_count_min": None,
            "frame_count_max": None,
        }

    rows = read_tsv(path)
    frame_counts = []
    for row in rows:
        value = row.get("frame_count", "")
        try:
            frame_counts.append(int(value))
        except ValueError:
            pass

    result = {
        "status": "present",
        "manifest": str(path),
        "camera_count": len(rows),
        "frame_count_min": min(frame_counts) if frame_counts else None,
        "frame_count_max": max(frame_counts) if frame_counts else None,
        "frame_count_unique": sorted(set(frame_counts)) if frame_counts else [],
    }
    if frame_counts and max(frame_counts) - min(frame_counts) <= 2:
        result["frame_alignment"] = "frame_count_only_common_prefix_tail_trim_assumed"
        result["frame_id_contiguity_checked"] = False
    elif frame_counts:
        result["frame_alignment"] = "large_frame_count_spread_check_camera_drop"
        result["frame_id_contiguity_checked"] = False
    else:
        result["frame_alignment"] = "frame_count_unavailable"
        result["frame_id_contiguity_checked"] = False
    return result


def summarize_coverage(coverage_dir):
    summary_path = Path(coverage_dir) / "summary.json"
    summary = read_json(summary_path)
    if not summary:
        return {
            "status": "missing",
            "summary_json": str(summary_path),
            "report_html": str(Path(coverage_dir) / "coverage_report.html"),
        }
    counts = summary.get("status_counts", {})
    red = int(counts.get("red", 0) or 0)
    yellow = int(counts.get("yellow", 0) or 0)
    if red:
        gate = "has_red_cameras"
    elif yellow:
        gate = "has_yellow_cameras"
    else:
        gate = "ok"
    return {
        "status": "present",
        "gate": gate,
        "summary_json": str(summary_path),
        "report_html": str(Path(coverage_dir) / "coverage_report.html"),
        "camera_count": summary.get("camera_count"),
        "imageset_count": summary.get("imageset_count"),
        "status_counts": counts,
    }


def summarize_colmap_ransac(ransac_dir):
    ransac_dir = Path(ransac_dir)
    summary = read_json(ransac_dir / "summary.json")
    pose_yaml = ransac_dir / "camera_tr_rig_ransac.yaml"
    if not summary:
        return {
            "status": "missing",
            "summary_json": str(ransac_dir / "summary.json"),
            "pose_yaml": str(pose_yaml),
            "report_html": str(ransac_dir / "index.html"),
        }
    voted = int(summary.get("voted_camera_count", 0) or 0)
    total = int(summary.get("camera_count", 0) or 0)
    return {
        "status": "present" if pose_yaml.is_file() else "summary_only",
        "pose_yaml": str(pose_yaml),
        "report_html": str(ransac_dir / "index.html"),
        "voted_camera_count": voted,
        "camera_count": total,
        "accepted_run_count": summary.get("accepted_run_count"),
        "median_inlier_fraction": summary.get("median_inlier_fraction"),
        "median_center_residual_m": summary.get("median_center_residual_m"),
        "median_rotation_residual_deg": summary.get("median_rotation_residual_deg"),
    }


def summarize_side_prior(side_dir):
    side_dir = Path(side_dir)
    summary = read_json(side_dir / "summary.json")
    pose_yaml = side_dir / "camera_tr_rig_side_prior.yaml"
    if not summary:
        return {
            "status": "missing",
            "summary_json": str(side_dir / "summary.json"),
            "pose_yaml": str(pose_yaml),
        }
    completed = int(summary.get("completed_pose_count", 0) or 0)
    total = int(summary.get("camera_count", 0) or 0)
    status = "complete" if total and completed == total else "partial"
    if not pose_yaml.is_file():
        status = "summary_only"
    return {
        "status": status,
        "pose_yaml": str(pose_yaml),
        "completed_pose_count": completed,
        "camera_count": total,
        "side_prior_completed_count": summary.get("side_prior_completed_count"),
        "relative_pair_success_count": summary.get("relative_pair_success_count"),
        "relative_pair_count": summary.get("relative_pair_count"),
        "bridge_pose_override_count": summary.get("bridge_pose_override_count", 0),
        "bridge_pose_overrides": summary.get("bridge_pose_overrides", []),
    }


def summarize_tag_refine(tag_dir, default_intrinsics_refine_mode="fixed"):
    tag_dir = Path(tag_dir)
    summary = read_json(tag_dir / "summary.json")
    pose_yaml = tag_dir / "camera_tr_rig_delta_refined_accepted.yaml"
    diagnostics_dir = tag_dir / "diagnostics"
    if not summary:
        return {
            "status": "missing",
            "summary_json": str(tag_dir / "summary.json"),
            "accepted_pose_yaml": str(pose_yaml),
        }
    cameras = summary.get("cameras", {})
    accepted = cameras.get("accepted_refined", [])
    prior_only = cameras.get("prior_only", [])
    status = "partial_refine" if accepted else "prior_only_or_no_acceptance"
    if not pose_yaml.is_file():
        status = "summary_only"
    intrinsics = summary.get("intrinsics", {}) or {}
    if not intrinsics.get("refine_mode"):
        intrinsics = dict(intrinsics)
        intrinsics["refine_mode"] = default_intrinsics_refine_mode
        intrinsics["source"] = "wrapper_default_or_legacy_summary"
    acceptance_rows = read_tsv(diagnostics_dir / "camera_acceptance.tsv")
    reprojection_rows = read_tsv(diagnostics_dir / "camera_reprojection.tsv")
    delta_rows = read_tsv(diagnostics_dir / "camera_delta.tsv")
    intrinsic_rows = read_tsv(diagnostics_dir / "camera_intrinsics.tsv")
    reprojection_by_id = {row.get("camera_id", ""): row for row in reprojection_rows}
    delta_by_id = {row.get("camera_id", ""): row for row in delta_rows}
    intrinsic_by_id = {row.get("camera_id", ""): row for row in intrinsic_rows}
    camera_report_rows = []
    for row in acceptance_rows:
        camera_id = row.get("camera_id", "")
        reproj = reprojection_by_id.get(camera_id, {})
        delta = delta_by_id.get(camera_id, {})
        intrinsic = intrinsic_by_id.get(camera_id, {})
        camera_report_rows.append({
            "camera_id": camera_id,
            "decision": row.get("decision", ""),
            "output_pose": row.get("output_pose", ""),
            "reason": row.get("reason", ""),
            "intrinsic_decision": intrinsic.get("decision", ""),
            "output_intrinsics": intrinsic.get("output_intrinsics", ""),
            "max_abs_focal_delta_frac": intrinsic.get("max_abs_focal_delta_frac", ""),
            "principal_delta_px": intrinsic.get("principal_delta_px", ""),
            "max_abs_distortion_delta": intrinsic.get("max_abs_distortion_delta", ""),
            "used_observation_count": delta.get("used_observation_count", row.get("observation_count", "")),
            "after_median_px": reproj.get("after_median_px", row.get("after_median_px", "")),
            "after_p90_px": reproj.get("after_p90_px", ""),
            "after_max_px": reproj.get("after_max_px", ""),
            "after_under_300_fraction": reproj.get(
                "after_under_300_fraction",
                row.get("after_under_300_fraction", ""),
            ),
            "delta_rotation_deg": delta.get("delta_rotation_deg", ""),
            "delta_translation_m": delta.get("delta_translation_m", ""),
        })
    return {
        "status": status,
        "accepted_pose_yaml": str(pose_yaml),
        "accepted_refined": accepted,
        "accepted_refined_count": cameras.get("accepted_refined_count", len(accepted)),
        "prior_only": prior_only,
        "prior_only_count": len(prior_only),
        "settings": summary.get("settings", {}),
        "observation_gate": summary.get("observation_gate", {}),
        "post_refine_observation_gate": summary.get("post_refine_observation_gate", {}),
        "residual_before": summary.get("residual_before", {}),
        "residual_after": summary.get("residual_after", {}),
        "residual_after_output_accepted": summary.get("residual_after_output_accepted", {}),
        "raw_residual_before": summary.get("raw_residual_before", {}),
        "raw_residual_after": summary.get("raw_residual_after", {}),
        "raw_residual_after_output_accepted": summary.get("raw_residual_after_output_accepted", {}),
        "intrinsics": intrinsics,
        "bridge_prior_overrides": summary.get("bridge_prior_overrides", []),
        "camera_report_rows": camera_report_rows,
        "diagnostics": {
            "camera_acceptance_tsv": str(diagnostics_dir / "camera_acceptance.tsv"),
            "camera_reprojection_tsv": str(diagnostics_dir / "camera_reprojection.tsv"),
            "camera_reprojection_accepted_tsv": str(diagnostics_dir / "camera_reprojection_accepted.tsv"),
            "camera_delta_tsv": str(diagnostics_dir / "camera_delta.tsv"),
            "camera_intrinsics_tsv": str(diagnostics_dir / "camera_intrinsics.tsv"),
        },
    }


def summarize_residual_tail(report_dir):
    report_dir = Path(report_dir)
    summary_path = report_dir / "residual_tail_summary.json"
    report_html = report_dir / "residual_tail_report.html"
    summary = read_json(summary_path) or {}
    diagnostics = summary.get("observation_diagnostics", {}) if isinstance(summary, dict) else {}
    return {
        "status": "present" if summary_path.is_file() and report_html.is_file() else "missing",
        "summary_json": str(summary_path),
        "summary_json_exists": summary_path.is_file(),
        "report_html": str(report_html),
        "report_html_exists": report_html.is_file(),
        "report_url": report_url(report_html),
        "camera_count": summary.get("camera_count") if isinstance(summary, dict) else None,
        "observation_diagnostics_available": diagnostics.get("available"),
        "observation_diagnostics_message": diagnostics.get("message", ""),
    }


def summarize_intrinsic_feature_coverage(report_dir):
    report_dir = Path(report_dir)
    summary_path = report_dir / "summary.json"
    index_html = report_dir / "index.html"
    metrics_tsv = report_dir / "camera_metrics.tsv"
    data = read_json(summary_path) or {}
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    cameras = data.get("cameras", []) if isinstance(data, dict) else []
    plot_count = sum(1 for row in cameras if Path(row.get("plot_path", "")).is_file())
    return {
        "status": "present" if summary_path.is_file() and index_html.is_file() else "missing",
        "index_html": str(index_html),
        "index_html_exists": index_html.is_file(),
        "index_url": report_url(index_html),
        "summary_json": str(summary_path),
        "summary_json_exists": summary_path.is_file(),
        "metrics_tsv": str(metrics_tsv),
        "metrics_tsv_exists": metrics_tsv.is_file(),
        "source_type": summary.get("source_type", ""),
        "source": summary.get("source", ""),
        "camera_count": summary.get("camera_count", len(cameras)),
        "plot_count": plot_count,
    }


def summarize_frame_face_refine(refine_dir):
    refine_dir = Path(refine_dir)
    summary_path = refine_dir / "summary.json"
    pose_yaml = refine_dir / "camera_tr_rig_delta_refined.yaml"
    metrics_tsv = refine_dir / "diagnostics" / "camera_reprojection.tsv"
    summary = read_json(summary_path)
    if not summary:
        return {
            "status": "missing",
            "summary_json": str(summary_path),
            "pose_yaml": str(pose_yaml),
            "metrics_tsv": str(metrics_tsv),
        }
    cameras = summary.get("cameras", {}) or {}
    residual_after = summary.get("residual_after", {}) or {}
    return {
        "status": "present",
        "summary_json": str(summary_path),
        "pose_yaml": str(pose_yaml),
        "pose_yaml_exists": pose_yaml.is_file(),
        "metrics_tsv": str(metrics_tsv),
        "metrics_tsv_exists": metrics_tsv.is_file(),
        "settings": summary.get("settings", {}),
        "observation_gate": summary.get("observation_gate", {}),
        "residual_before": summary.get("residual_before", {}),
        "residual_after": residual_after,
        "active_delta": cameras.get("active_delta"),
        "inactive_delta": cameras.get("inactive_delta", []),
        "camera_count": cameras.get("total"),
        "used_observations": (summary.get("observations", {}) or {}).get("used"),
        "median_px": residual_after.get("median_px"),
        "p90_px": residual_after.get("p90_px"),
    }


def stage_completed(stage_results, name):
    for stage in stage_results or []:
        if stage.get("name") == name:
            return stage.get("status") == "completed"
    return None


def tag_refine_summary_intrinsics_mode(tag_dir):
    summary = read_json(Path(tag_dir) / "summary.json")
    if not summary:
        return "fixed"
    settings = summary.get("settings", {}) or {}
    intrinsics = summary.get("intrinsics", {}) or {}
    return (
        settings.get("intrinsics_refine_mode")
        or intrinsics.get("refine_mode")
        or "fixed"
    )


def tag_refine_final_eligible(tag_dir, promote_diagnostic=False):
    mode = tag_refine_summary_intrinsics_mode(tag_dir)
    return mode == "fixed" or bool(promote_diagnostic)


def final_pose_candidate(paths, args, stage_results=None):
    run_colmap_vote = args.run_colmap_vote or args.run_all
    run_side_prior = args.run_side_prior or args.run_all
    run_tag_refine = args.run_tag_refine or args.run_all
    run_frame_face_refine = getattr(args, "run_frame_face_refine", False)
    diagnostic_tag_refine = args.tag_intrinsics_refine_mode != "fixed"
    promote_tag_refine = (not diagnostic_tag_refine) or args.promote_diagnostic_tag_refine
    frame_face_stage_ok = stage_completed(stage_results, "frame_face_refine")
    tag_stage_ok = stage_completed(stage_results, "tag_refine_robust")
    side_stage_ok = stage_completed(stage_results, "side_prior_completion")
    ransac_stage_ok = stage_completed(stage_results, "colmap_frame_vote")
    frame_face_dir = paths.get("frame_face_refine_dir", Path("__missing_frame_face_refine__"))
    frame_face_pose = frame_face_dir / "camera_tr_rig_delta_refined.yaml"
    tag_pose = paths["tag_refine_dir"] / "camera_tr_rig_delta_refined_accepted.yaml"
    existing_tag_pose = paths["existing_tag_dir"] / "camera_tr_rig_delta_refined_accepted.yaml"
    side_pose = paths["side_prior_dir"] / "camera_tr_rig_side_prior.yaml"
    existing_side_pose = paths["existing_side_dir"] / "camera_tr_rig_side_prior.yaml"
    ransac_pose = paths["colmap_ransac_dir"] / "camera_tr_rig_ransac.yaml"
    existing_ransac_pose = paths["existing_ransac_dir"] / "camera_tr_rig_ransac.yaml"
    tag_stage_failed = run_tag_refine and tag_stage_ok is False

    if run_frame_face_refine and (getattr(args, "dry_run", False) or frame_face_stage_ok is not False):
        return frame_face_pose, "frame_face_refine_expected"
    if run_tag_refine and promote_tag_refine and tag_stage_ok is not False:
        return tag_pose, "tag_refine_expected"
    if (
        existing_tag_pose.is_file()
        and not (tag_stage_failed and paths["existing_tag_dir"] == paths["tag_refine_dir"])
        and not (run_tag_refine and diagnostic_tag_refine and not args.promote_diagnostic_tag_refine)
        and tag_refine_final_eligible(
        paths["existing_tag_dir"],
        args.promote_diagnostic_tag_refine,
        )
    ):
        return existing_tag_pose, "existing_tag_refine"
    if run_side_prior and side_stage_ok is not False:
        return side_pose, "side_prior_expected"
    if paths["previous_outer_rig"].is_file():
        return paths["previous_outer_rig"], "previous_outer_rig"
    if existing_side_pose.is_file():
        return existing_side_pose, "existing_side_prior"
    if run_colmap_vote and ransac_stage_ok is not False:
        return ransac_pose, "colmap_ransac_expected"
    return existing_ransac_pose, "existing_colmap_ransac"


def final_metrics_candidate(paths, final_source):
    if final_source == "frame_face_refine_expected":
        return paths["frame_face_refine_dir"] / "diagnostics" / "camera_reprojection.tsv"
    if final_source == "tag_refine_expected":
        return paths["tag_refine_dir"] / "diagnostics" / "camera_reprojection.tsv"
    if final_source == "side_prior_expected":
        return paths["side_prior_dir"] / "camera_side_prior_summary.tsv"
    if final_source == "colmap_ransac_expected":
        return paths["colmap_ransac_dir"] / "camera_ransac_summary.tsv"
    if "tag_refine" in final_source:
        tag_metrics = paths["tag_refine_dir"] / "diagnostics" / "camera_reprojection.tsv"
        existing_tag_metrics = paths["existing_tag_dir"] / "diagnostics" / "camera_reprojection.tsv"
        return tag_metrics if tag_metrics.exists() else existing_tag_metrics
    if "side" in final_source or "previous" in final_source:
        metrics = paths["side_prior_dir"] / "camera_side_prior_summary.tsv"
        existing_metrics = paths["existing_side_dir"] / "camera_side_prior_summary.tsv"
        return metrics if metrics.exists() else existing_metrics
    metrics = paths["colmap_ransac_dir"] / "camera_ransac_summary.tsv"
    existing_metrics = paths["existing_ransac_dir"] / "camera_ransac_summary.tsv"
    return metrics if metrics.exists() else existing_metrics


def final_observation_residuals_candidate(paths, final_source):
    if final_source == "frame_face_refine_expected":
        return paths["frame_face_refine_dir"] / "diagnostics" / "observation_residuals.tsv"
    if final_source == "tag_refine_expected":
        return paths["tag_refine_dir"] / "diagnostics" / "observation_residuals.tsv"
    if "tag_refine" in final_source:
        tag_residuals = paths["tag_refine_dir"] / "diagnostics" / "observation_residuals.tsv"
        existing_tag_residuals = paths["existing_tag_dir"] / "diagnostics" / "observation_residuals.tsv"
        return tag_residuals if tag_residuals.exists() else existing_tag_residuals
    return Path("__missing_observation_residuals__.tsv")


def final_intrinsics_dir_candidate(paths, final_source):
    if final_source == "frame_face_refine_expected":
        return paths["frame_face_refine_dir"] / "intrinsics_refined"
    if final_source == "tag_refine_expected":
        return paths["tag_refine_dir"] / "intrinsics_refined_accepted"
    if "tag_refine" in final_source:
        tag_intrinsics = paths["tag_refine_dir"] / "intrinsics_refined_accepted"
        existing_tag_intrinsics = paths["existing_tag_dir"] / "intrinsics_refined_accepted"
        return tag_intrinsics if tag_intrinsics.exists() else existing_tag_intrinsics
    if "frame_face" in final_source:
        return paths["frame_face_refine_dir"] / "intrinsics_refined"
    return paths["previous_intrinsics_dir"]


def prefer_existing_path(primary, fallback):
    primary = Path(primary)
    return primary if primary.exists() else Path(fallback)


def make_command(script_name, *args):
    return [sys.executable, str(repo_root() / "scripts" / "calib" / script_name), *[str(x) for x in args]]


def tower_detector_tag_spacing_m(args):
    return float(args.tower_tag_center_pitch_m) - float(args.tower_detector_tag_size_m)


def bridge_prior_override_decision(args, paths):
    requested = (args.bridge_prior_override_labels or "").strip()
    summary_path = paths["bridge_summary_json"]
    summary = read_json(summary_path)
    metric_gate = ((summary or {}).get("quality_gates") or {}).get("metric_bridge") or {}
    metric_passed = metric_gate.get("passed") is True or metric_gate.get("status") == "pass"
    decision = {
        "policy": args.bridge_prior_override_policy,
        "requested_labels": requested,
        "effective_labels": "",
        "bridge_summary_json": str(summary_path),
        "metric_gate_status": metric_gate.get("status", "missing"),
        "metric_gate_passed": bool(metric_passed),
    }
    if not requested or args.bridge_prior_override_policy == "never":
        decision["reason"] = "disabled"
        return decision
    if args.bridge_prior_override_policy == "always":
        decision["effective_labels"] = requested
        decision["reason"] = "forced_by_policy"
        return decision
    if metric_passed:
        decision["effective_labels"] = requested
        decision["reason"] = "metric_bridge_gate_passed"
    else:
        decision["reason"] = "metric_bridge_gate_not_passed"
    return decision


def build_stage_plan(args, paths):
    run_quality = args.run_quality or args.run_reports or args.run_all
    run_viewer = args.run_viewer or args.run_reports or args.run_all
    run_colmap_vote = args.run_colmap_vote or args.run_all
    run_side_prior = args.run_side_prior or args.run_all
    run_tag_refine = args.run_tag_refine or args.run_all
    run_frame_face_refine = args.run_frame_face_refine

    coverage_dataset = paths["dataset"]
    coverage_manifest = paths["manifest"]
    if run_frame_face_refine and (not coverage_dataset.exists()) and paths["frame_face_dataset"].exists():
        coverage_dataset = paths["frame_face_dataset"]
        coverage_manifest = paths["frame_face_manifest"]

    coverage_cmd = make_command(
        "dataset_coverage_report.py",
        "--dataset", coverage_dataset,
        "--manifest", coverage_manifest,
        "--output-dir", paths["coverage_dir"],
    )

    colmap_frame_cmd = make_command(
        "run_outer_colmap_frame_vote.py",
        "--manifest", paths["manifest"],
        "--output-root", paths["colmap_frame_dir"],
        "--anchor-pose-yaml", paths["anchor_pose_yaml"],
        "--anchor-label-to-pose-index", paths["anchor_label_to_pose_index"],
        "--sample-count", args.sample_count,
        "--jobs", args.colmap_jobs,
        "--colmap-bin", args.colmap_bin,
        "--max-anchor-rms-m", 0.35,
        "--max-center-norm-m", 8.0,
        "--min-tracks-per-vote", 10,
        "--min-votes-per-camera", 4,
        "--center-vote-gate-m", 0.35,
    )
    colmap_ransac_cmd = make_command(
        "vote_outer_colmap_runs.py",
        "--manifest", paths["manifest"],
        "--runs-root", paths["colmap_frame_dir"],
        "--output-root", paths["colmap_ransac_dir"],
        "--anchor-pose-yaml", paths["anchor_pose_yaml"],
        "--anchor-label-to-pose-index", paths["anchor_label_to_pose_index"],
        "--max-runs", args.sample_count,
        "--max-anchor-rms-m", 0.35,
        "--max-center-norm-m", 8.0,
        "--min-tracks-per-vote", 10,
        "--min-votes-per-camera", 4,
        "--ransac-center-threshold-m", 0.50,
        "--ransac-rotation-threshold-deg", 15.0,
        "--export-camera-images",
    )

    runs_root_for_side = (
        paths["colmap_frame_dir"]
        if run_colmap_vote
        else prefer_existing_path(paths["colmap_frame_dir"], paths["existing_frame_runs"])
    )
    base_ransac_dir = (
        paths["colmap_ransac_dir"]
        if run_colmap_vote
        else prefer_existing_path(paths["colmap_ransac_dir"], paths["existing_ransac_dir"])
    )
    bridge_override = bridge_prior_override_decision(args, paths)
    side_cmd = make_command(
        "complete_outer_rig_side_prior.py",
        "--manifest", paths["manifest"],
        "--runs-root", runs_root_for_side,
        "--anchor-pose-yaml", paths["anchor_pose_yaml"],
        "--anchor-label-to-pose-index", paths["anchor_label_to_pose_index"],
        "--base-pose-yaml", base_ransac_dir / "camera_tr_rig_ransac.yaml",
        "--base-metrics-tsv", base_ransac_dir / "camera_ransac_summary.tsv",
        "--output-root", paths["side_prior_dir"],
        "--max-runs", args.sample_count,
        "--max-anchor-rms-m", 0.35,
        "--max-center-norm-m", 8.0,
        "--min-tracks-per-vote", 10,
        "--min-relative-votes", 4,
        "--relative-translation-threshold-m", 0.65,
        "--relative-rotation-threshold-deg", 20.0,
    )
    if bridge_override["effective_labels"]:
        side_cmd.extend([
            "--bridge-pose-override-labels",
            bridge_override["effective_labels"],
        ])

    tag_prior = (
        paths["side_prior_dir"] / "camera_tr_rig_side_prior.yaml"
        if run_side_prior or (paths["side_prior_dir"] / "camera_tr_rig_side_prior.yaml").is_file()
        else paths["previous_outer_rig"]
    )
    pnp_views_for_tag_refine = (
        paths["pnp_consensus_views"]
        if args.tag_pnp_pose_consensus
        else paths["pnp_views"]
    )
    pnp_consensus_cmd = make_command(
        "filter_pnp_views_by_pose_consensus.py",
        "--pnp-views", paths["pnp_views"],
        "--camera-prior-pose-yaml", tag_prior,
        "--output-pnp-views", paths["pnp_consensus_views"],
        "--summary-json", paths["pnp_consensus_dir"] / "summary.json",
        "--per-frame-tsv", paths["pnp_consensus_dir"] / "per_frame_consensus.tsv",
        "--center-threshold-m", args.tag_pnp_consensus_center_threshold_m,
        "--rotation-threshold-deg", args.tag_pnp_consensus_rotation_threshold_deg,
        "--max-median-error-px", args.tag_max_pnp_median_error_px,
        "--min-points", args.tag_pnp_consensus_min_points,
        "--min-inliers", args.tag_pnp_consensus_min_inliers,
        "--min-consensus-votes", args.tag_pnp_consensus_min_votes,
    )
    tag_cmd = make_command(
        "refine_outer_tower_delta_prior.py",
        "--dataset", paths["dataset"],
        "--manifest", paths["manifest"],
        "--pnp_views", pnp_views_for_tag_refine,
        "--bridge_pose_yaml", paths["anchor_pose_yaml"],
        "--anchor_label_to_pose_index", paths["anchor_label_to_pose_index"],
        "--camera_prior_pose_yaml", tag_prior,
        "--output_dir", paths["tag_refine_dir"],
        "--min_camera_observations_for_use", args.tag_min_camera_observations_for_use,
        "--min_camera_observations_for_delta", args.tag_min_camera_observations_for_delta,
        "--observation_residual_gate_px", args.tag_observation_residual_gate_px,
        "--post_refine_observation_residual_gate_px", args.tag_post_refine_observation_residual_gate_px,
        "--post_refine_outer_iterations", args.tag_post_refine_outer_iterations,
        "--optimizer_residual_clip_px", args.tag_residual_clip_px,
        "--accept_camera_median_px", args.tag_accept_camera_median_px,
        "--accept_camera_p90_px", args.tag_accept_camera_p90_px,
        "--accept_camera_min_under_300_fraction", args.tag_accept_under_300_fraction,
        "--accept_camera_max_delta_translation_m", args.tag_accept_max_delta_translation_m,
        "--accept_camera_max_delta_rotation_deg", args.tag_accept_max_delta_rotation_deg,
        "--outer_iterations", args.tag_outer_iterations,
        "--block_iterations", args.tag_block_iterations,
        "--delta_translation_sigma_m", args.delta_translation_sigma_m,
        "--delta_rotation_sigma_deg", args.delta_rotation_sigma_deg,
        "--intrinsics_refine_mode", args.tag_intrinsics_refine_mode,
        "--intrinsics_focal_sigma_frac", args.tag_intrinsics_focal_sigma_frac,
        "--intrinsics_principal_sigma_px", args.tag_intrinsics_principal_sigma_px,
        "--intrinsics_distortion_sigma", args.tag_intrinsics_distortion_sigma,
        "--intrinsics_max_focal_step_frac", args.tag_intrinsics_max_focal_step_frac,
        "--intrinsics_max_principal_step_px", args.tag_intrinsics_max_principal_step_px,
        "--intrinsics_max_distortion_step", args.tag_intrinsics_max_distortion_step,
        "--intrinsics_max_total_focal_delta_frac", args.tag_intrinsics_max_total_focal_delta_frac,
        "--intrinsics_max_total_principal_delta_px", args.tag_intrinsics_max_total_principal_delta_px,
        "--intrinsics_max_total_distortion_delta", args.tag_intrinsics_max_total_distortion_delta,
        "--intrinsics_block_iterations", args.tag_intrinsics_block_iterations,
        "--accept_camera_max_intrinsic_focal_delta_frac", args.tag_accept_max_intrinsic_focal_delta_frac,
        "--accept_camera_max_intrinsic_principal_delta_px", args.tag_accept_max_intrinsic_principal_delta_px,
        "--accept_camera_max_intrinsic_distortion_delta", args.tag_accept_max_intrinsic_distortion_delta,
        "--tower_face_width_initial_m", args.tag_tower_face_width_initial_m,
        "--tower_face_width_sigma_m", args.tag_tower_face_width_sigma_m,
        "--tower_face_width_min_m", args.tag_tower_face_width_min_m,
        "--tower_face_width_max_m", args.tag_tower_face_width_max_m,
        "--tower_face_width_max_step_m", args.tag_tower_face_width_max_step_m,
        "--tower_tag_size_m", args.tower_detector_tag_size_m,
        "--tower_tag_spacing_m", tower_detector_tag_spacing_m(args),
        "--max_pnp_median_error_px", args.tag_max_pnp_median_error_px,
    )
    if args.tag_optimize_tower_face_width:
        tag_cmd.append("--optimize_tower_face_width")
    if bridge_override["effective_labels"]:
        tag_cmd.extend([
            "--bridge_prior_override_labels",
            bridge_override["effective_labels"],
        ])
    if args.tag_intrinsics_mode == "central_opencv" and paths["previous_intrinsics_dir"].is_dir():
        tag_cmd.extend([
            "--intrinsics_dir", str(paths["previous_intrinsics_dir"]),
            "--intrinsics_mode", "central_opencv",
        ])
    else:
        tag_cmd.extend(["--intrinsics_mode", "colmap_fixed"])

    frame_face_preset = FRAME_FACE_REFINE_PRESETS[args.frame_face_refine_preset]
    frame_face_cmd = make_command(
        "refine_outer_tower_frame_face_planes.py",
        "--dataset", paths["frame_face_dataset"],
        "--manifest", paths["frame_face_manifest"],
        "--camera_prior_pose_yaml", paths["frame_face_prior_pose_yaml"],
        "--intrinsics_dir", paths["frame_face_intrinsics_dir"],
        "--intrinsics_mode", "central_opencv",
        "--intrinsics_refine_mode", "fixed",
        "--tower_model", frame_face_preset.get("tower_model", "rigid_yaw45_tower"),
        "--tower_face_count", 8,
        "--tower_face0_angle_degrees", 0.0,
        "--tower_face_width_initial_m", 0.25,
        "--tower_face_width_sigma_m", 0.03,
        "--tower_face_width_min_m", 0.18,
        "--tower_face_width_max_m", 0.32,
        "--tower_tag_size_m", args.tower_detector_tag_size_m,
        "--tower_tag_spacing_m", tower_detector_tag_spacing_m(args),
        "--output_dir", paths["frame_face_refine_dir"],
        "--outer_iterations", frame_face_preset["outer_iterations"],
        "--block_iterations", frame_face_preset["block_iterations"],
        "--min_pnp_points", 8,
        "--pnp_ransac",
        "--pnp_ransac_iterations", frame_face_preset["pnp_ransac_iterations"],
        "--pnp_ransac_threshold_px", frame_face_preset["pnp_ransac_threshold_px"],
        "--max_pnp_median_error_px", frame_face_preset["max_pnp_median_error_px"],
        "--min_frame_face_observations", frame_face_preset["min_frame_face_observations"],
        "--min_camera_observations_for_delta", frame_face_preset["min_camera_observations_for_delta"],
    )
    if frame_face_preset.get("optimize_tower_face_width", True):
        frame_face_cmd.append("--optimize_tower_face_width")
    else:
        frame_face_cmd.append("--no-optimize_tower_face_width")
    if frame_face_preset.get("tower_model") == "flex_yaw_offset_tower":
        frame_face_cmd.extend([
            "--flex_face_yaw_sigma_deg", frame_face_preset["flex_face_yaw_sigma_deg"],
            "--flex_face_yaw_max_deg", frame_face_preset["flex_face_yaw_max_deg"],
            "--flex_face_radial_offset_sigma_m", frame_face_preset["flex_face_radial_offset_sigma_m"],
            "--flex_face_radial_offset_max_m", frame_face_preset["flex_face_radial_offset_max_m"],
            "--flex_face_tangent_offset_sigma_m", frame_face_preset["flex_face_tangent_offset_sigma_m"],
            "--flex_face_tangent_offset_max_m", frame_face_preset["flex_face_tangent_offset_max_m"],
            "--flex_face_adjacent_angle_min_deg", frame_face_preset["flex_face_adjacent_angle_min_deg"],
            "--flex_face_adjacent_angle_max_deg", frame_face_preset["flex_face_adjacent_angle_max_deg"],
            "--flex_face_geometry_block_iterations", frame_face_preset["flex_face_geometry_block_iterations"],
        ])
    if frame_face_preset.get("initial_observation_residual_gate_px") is not None:
        frame_face_cmd.extend([
            "--initial_observation_residual_gate_px",
            frame_face_preset["initial_observation_residual_gate_px"],
        ])
    frame_face_cmd.extend([
        "--observation_residual_gate_px", frame_face_preset["observation_residual_gate_px"],
        "--optimizer_residual_clip_px", frame_face_preset["optimizer_residual_clip_px"],
    ])

    final_pose, final_source = final_pose_candidate(paths, args)
    final_metrics = final_metrics_candidate(paths, final_source)
    final_observation_residuals = final_observation_residuals_candidate(paths, final_source)
    final_intrinsics_dir = final_intrinsics_dir_candidate(paths, final_source)
    runs_root_for_viewer = (
        paths["colmap_frame_dir"]
        if run_colmap_vote
        else prefer_existing_path(paths["colmap_frame_dir"], paths["existing_frame_runs"])
    )
    viewer_has_colmap_context = Path(runs_root_for_viewer).exists() and Path(paths["manifest"]).exists()
    viewer_requested = run_viewer and (
        args.run_viewer
        or args.run_all
        or run_colmap_vote
        or viewer_has_colmap_context
    )
    viewer_cmd = make_command(
        "generate_outer_colmap_scene_viewer.py",
        "--manifest", paths["manifest"],
        "--runs-root", runs_root_for_viewer,
        "--anchor-pose-yaml", paths["anchor_pose_yaml"],
        "--anchor-label-to-pose-index", paths["anchor_label_to_pose_index"],
        "--final-pose-yaml", final_pose,
        "--final-metrics-tsv", final_metrics,
        "--final-rig-label", final_source.replace("_", " "),
        "--final-rig-source", final_source,
        "--tower-pose-yaml", paths["tag_refine_dir"] / "rig_tr_global.yaml",
        "--output-dir", paths["viewer_dir"],
        "--max-runs", args.sample_count,
        "--max-anchor-rms-m", 0.35,
        "--title", "Outer Tower Recalibration Pipeline Viewer",
    )
    residual_tail_cmd = make_command(
        "analyze_outer_tag_residual_tail.py",
        paths["tag_refine_dir"],
        "--output-dir", paths["residual_tail_dir"],
        "--limit", 30,
    )
    residual_tail_requested = (
        run_tag_refine
        or ((run_quality or args.run_reports) and not run_frame_face_refine)
        or (paths["tag_refine_dir"] / "diagnostics" / "camera_reprojection.tsv").is_file()
    )
    intrinsic_feature_coverage_cmd = make_command(
        "generate_intrinsic_feature_coverage_report.py",
        "--residuals-tsv", final_observation_residuals,
        "--intrinsics-dir", final_intrinsics_dir,
        "--output-dir", paths["intrinsic_feature_coverage_dir"],
        "--title", "Outer Intrinsic Feature Coverage Report",
    )
    intrinsic_feature_coverage_requested = (
        run_quality
        or run_frame_face_refine
        or run_tag_refine
        or final_observation_residuals.is_file()
    )

    return [
        {
            "name": "coverage_report",
            "requested": run_quality,
            "commands": [coverage_cmd],
            "inputs": [coverage_dataset, coverage_manifest],
            "outputs": [paths["coverage_dir"] / "summary.json", paths["coverage_dir"] / "coverage_report.html"],
        },
        {
            "name": "colmap_frame_vote",
            "requested": run_colmap_vote,
            "commands": [colmap_frame_cmd, colmap_ransac_cmd],
            "inputs": [paths["manifest"], paths["anchor_pose_yaml"]],
            "outputs": [
                paths["colmap_ransac_dir"] / "camera_tr_rig_ransac.yaml",
                paths["colmap_ransac_dir"] / "summary.json",
            ],
        },
        {
            "name": "side_prior_completion",
            "requested": run_side_prior,
            "commands": [side_cmd],
            "inputs": [
                paths["manifest"],
                runs_root_for_side,
                paths["anchor_pose_yaml"],
                base_ransac_dir / "camera_tr_rig_ransac.yaml",
            ],
            "outputs": [paths["side_prior_dir"] / "camera_tr_rig_side_prior.yaml"],
        },
        {
            "name": "pnp_pose_consensus",
            "requested": run_tag_refine and args.tag_pnp_pose_consensus,
            "commands": [pnp_consensus_cmd],
            "inputs": [
                paths["pnp_views"],
                tag_prior,
            ],
            "outputs": [
                paths["pnp_consensus_views"],
                paths["pnp_consensus_dir"] / "summary.json",
                paths["pnp_consensus_dir"] / "per_frame_consensus.tsv",
            ],
        },
        {
            "name": "tag_refine_robust",
            "requested": run_tag_refine,
            "commands": [tag_cmd],
            "inputs": [
                paths["dataset"],
                paths["manifest"],
                pnp_views_for_tag_refine,
                paths["anchor_pose_yaml"],
                tag_prior,
            ],
            "outputs": [
                paths["tag_refine_dir"] / "camera_tr_rig_delta_refined_accepted.yaml",
                paths["tag_refine_dir"] / "intrinsics_refined_accepted",
                paths["tag_refine_dir"] / "summary.json",
                paths["tag_refine_dir"] / "diagnostics",
            ],
        },
        {
            "name": "frame_face_refine",
            "requested": run_frame_face_refine,
            "commands": [frame_face_cmd],
            "inputs": [
                paths["frame_face_dataset"],
                paths["frame_face_manifest"],
                paths["frame_face_prior_pose_yaml"],
                paths["frame_face_intrinsics_dir"],
            ],
            "outputs": [
                paths["frame_face_refine_dir"] / "camera_tr_rig_delta_refined.yaml",
                paths["frame_face_refine_dir"] / "intrinsics_refined",
                paths["frame_face_refine_dir"] / "summary.json",
                paths["frame_face_refine_dir"] / "diagnostics",
            ],
        },
        {
            "name": "residual_tail_report",
            "requested": residual_tail_requested,
            "commands": [residual_tail_cmd],
            "inputs": [paths["tag_refine_dir"] / "diagnostics" / "camera_reprojection.tsv"],
            "outputs": [
                paths["residual_tail_dir"] / "residual_tail_summary.json",
                paths["residual_tail_dir"] / "residual_tail_report.html",
            ],
        },
        {
            "name": "intrinsic_feature_coverage_report",
            "requested": intrinsic_feature_coverage_requested,
            "commands": [intrinsic_feature_coverage_cmd],
            "inputs": [final_observation_residuals, final_intrinsics_dir],
            "outputs": [
                paths["intrinsic_feature_coverage_dir"] / "index.html",
                paths["intrinsic_feature_coverage_dir"] / "summary.json",
                paths["intrinsic_feature_coverage_dir"] / "camera_metrics.tsv",
            ],
        },
        {
            "name": "viewer_generation",
            "requested": viewer_requested,
            "commands": [viewer_cmd],
            "inputs": [paths["manifest"], runs_root_for_viewer, paths["anchor_pose_yaml"], final_pose],
            "outputs": [paths["viewer_dir"] / "index.html", paths["viewer_dir"] / "scene_data.json"],
        },
    ]


def missing_inputs(stage):
    missing = []
    for item in stage["inputs"]:
        if not Path(item).exists():
            missing.append(str(item))
    return missing


def clear_stage_outputs(stage):
    for output in stage.get("outputs", []):
        path = Path(output)
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            pass


def run_command(command, cwd, log):
    log.write("$ " + shell_join(command) + "\n\n")
    log.flush()
    proc = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.write(f"\n[returncode] {proc.returncode}\n")
    log.flush()
    return proc.returncode


def execute_stages(stages, paths, dry_run, force=False):
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)

    results = []
    for stage in stages:
        result = {
            "name": stage["name"],
            "requested": bool(stage["requested"]),
            "commands": [shell_join(command) for command in stage["commands"]],
            "outputs": [str(path) for path in stage["outputs"]],
            "started_at": "",
            "finished_at": "",
            "duration_s": 0.0,
            "log": "",
            "return_codes": [],
            "missing_inputs": [],
        }
        if not stage["requested"]:
            result["status"] = "not_requested"
            results.append(result)
            continue

        missing = missing_inputs(stage)
        result["missing_inputs"] = missing
        if missing and not dry_run:
            result["status"] = "missing_inputs"
            results.append(result)
            continue

        if dry_run:
            result["status"] = "planned_with_missing_inputs" if missing else "planned"
            results.append(result)
            continue

        result["started_at"] = utc_now()
        started = time.time()
        if force:
            clear_stage_outputs(stage)

        log_path = paths["logs_dir"] / f"{stage['name']}.log"
        return_codes = []
        with log_path.open("w", encoding="utf-8") as log:
            for command in stage["commands"]:
                rc = run_command(command, repo_root(), log)
                return_codes.append(rc)
                if rc != 0:
                    break
        result["log"] = str(log_path)
        result["return_codes"] = return_codes
        result["duration_s"] = duration_s(time.time() - started)
        result["finished_at"] = utc_now()
        result["status"] = "completed" if return_codes and return_codes[-1] == 0 else "failed"
        results.append(result)
    return results


def collect_summary(args, paths, stage_results, run_started_at="", run_finished_at="", total_duration=0.0):
    final_pose, final_source = final_pose_candidate(paths, args, stage_results)
    final_metrics = final_metrics_candidate(paths, final_source)
    viewer_index = paths["viewer_dir"] / "index.html"
    manifest_path = paths["output_root"] / "run_manifest.json"
    stage_durations = {
        stage.get("name", ""): duration_s(stage.get("duration_s", stage.get("duration", 0.0)))
        for stage in stage_results
    }
    run_colmap_vote = args.run_colmap_vote or args.run_all
    run_side_prior = args.run_side_prior or args.run_all
    run_tag_refine = args.run_tag_refine or args.run_all
    colmap_summary_dir = (
        paths["colmap_ransac_dir"]
        if run_colmap_vote
        else prefer_existing_path(paths["colmap_ransac_dir"], paths["existing_ransac_dir"])
    )
    if run_side_prior:
        side_summary_dir = paths["side_prior_dir"]
    elif args.previous_outer_rig:
        side_summary_dir = paths["previous_outer_rig"].parent
    else:
        side_summary_dir = prefer_existing_path(paths["side_prior_dir"], paths["existing_side_dir"])
    tag_summary_dir = paths["tag_refine_dir"] if run_tag_refine else paths["existing_tag_dir"]
    tag_refine_failed = run_tag_refine and stage_completed(stage_results, "tag_refine_robust") is False
    if tag_refine_failed:
        tag_refine_summary = {
            "status": "requested_stage_failed",
            "summary_json": str(paths["tag_refine_dir"] / "summary.json"),
            "accepted_pose_yaml": str(paths["tag_refine_dir"] / "camera_tr_rig_delta_refined_accepted.yaml"),
            "accepted_refined": [],
            "accepted_refined_count": 0,
            "prior_only": [],
            "prior_only_count": 0,
            "intrinsics": {"refine_mode": args.tag_intrinsics_refine_mode},
            "camera_report_rows": [],
            "diagnostics": {},
        }
    else:
        tag_refine_summary = summarize_tag_refine(tag_summary_dir, args.tag_intrinsics_refine_mode)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "run_tag": args.run_tag,
        "provenance": pipeline_provenance(),
        "sample_count": args.sample_count,
        "whole_sequence": paths["whole_sequence"],
        "run_manifest": str(manifest_path),
        "run_manifest_path": str(manifest_path),
        "run_manifest_url": report_url(manifest_path),
        "run_timing": {
            "started_at": run_started_at,
            "finished_at": run_finished_at,
            "total_duration_s": duration_s(total_duration),
            "stage_count": len(stage_results),
            "stage_durations_s": stage_durations,
        },
        "inputs": {
            "data_root": path_status(paths["data_root"]),
            "whole_dir": path_status(paths["whole_dir"]),
            "dataset": path_status(paths["dataset"]),
            "manifest": path_status(paths["manifest"]),
            "anchor_pose_yaml": path_status(paths["anchor_pose_yaml"]),
            "bridge_summary_json": path_status(paths["bridge_summary_json"]),
            "anchor_label_to_pose_index": paths["anchor_label_to_pose_index"],
            "pnp_views": path_status(paths["pnp_views"]),
            "previous_outer_rig": path_status(paths["previous_outer_rig"]),
            "previous_intrinsics_dir": path_status(paths["previous_intrinsics_dir"]),
            "frame_face_dataset": path_status(paths["frame_face_dataset"]),
            "frame_face_manifest": path_status(paths["frame_face_manifest"]),
            "frame_face_prior_pose_yaml": path_status(paths["frame_face_prior_pose_yaml"]),
            "frame_face_intrinsics_dir": path_status(paths["frame_face_intrinsics_dir"]),
            "outer_colmap_images_txt": path_status(paths["outer_colmap_images_txt"]),
        },
        "output_root": str(paths["output_root"]),
        "stages": stage_results,
        "capture_quality": summarize_manifest(paths["manifest"]),
        "coverage_gate": summarize_coverage(paths["coverage_dir"]),
        "colmap_ransac_vote": summarize_colmap_ransac(colmap_summary_dir),
        "side_prior": summarize_side_prior(side_summary_dir),
        "bridge_prior_override": bridge_prior_override_decision(args, paths),
        "tag_refine": tag_refine_summary,
        "frame_face_refine": summarize_frame_face_refine(paths["frame_face_refine_dir"]),
        "residual_tail": summarize_residual_tail(paths["residual_tail_dir"]),
        "intrinsic_feature_coverage": summarize_intrinsic_feature_coverage(
            paths["intrinsic_feature_coverage_dir"]
        ),
        "final": {
            "pose_yaml": str(final_pose),
            "pose_yaml_exists": final_pose.is_file(),
            "source": final_source,
            "metrics_tsv": str(final_metrics),
            "metrics_tsv_exists": final_metrics.is_file(),
            "viewer_index": str(viewer_index),
            "viewer_index_exists": viewer_index.is_file(),
            "viewer_url": report_url(viewer_index),
            "intrinsic_feature_coverage_index": str(paths["intrinsic_feature_coverage_dir"] / "index.html"),
            "intrinsic_feature_coverage_url": report_url(paths["intrinsic_feature_coverage_dir"] / "index.html"),
            "quality_report_index": str(paths["quality_report_dir"] / "index.html"),
            "quality_report_url": report_url(paths["quality_report_dir"] / "index.html"),
            "final_report_index": str(paths["final_report_dir"] / "index.html"),
            "final_report_url": report_url(paths["final_report_dir"] / "index.html"),
        },
    }
    return summary


def stage_manifest_entry(stage):
    commands = stage.get("commands", [])
    duration = duration_s(stage.get("duration_s", stage.get("duration", 0.0)))
    return {
        "name": stage.get("name", ""),
        "status": stage.get("status", ""),
        "requested": bool(stage.get("requested")),
        "started_at": stage.get("started_at", ""),
        "finished_at": stage.get("finished_at", ""),
        "duration": duration,
        "duration_s": duration,
        "command": "\n".join(commands),
        "commands": commands,
        "log": stage.get("log", ""),
        "return_codes": stage.get("return_codes", []),
        "missing_inputs": stage.get("missing_inputs", []),
    }


def build_run_manifest(summary):
    run_timing = summary.get("run_timing", {})
    inputs = summary.get("inputs", {})
    final = summary.get("final", {})
    return {
        "created_at": utc_now(),
        "started_at": run_timing.get("started_at", ""),
        "finished_at": run_timing.get("finished_at", ""),
        "total_duration_s": duration_s(run_timing.get("total_duration_s", 0.0)),
        "summary_json": str(Path(summary["output_root"]) / "summary.json"),
        "data_root": inputs.get("data_root", {}).get("path", ""),
        "whole_dir": inputs.get("whole_dir", {}).get("path", ""),
        "output_root": summary.get("output_root", ""),
        "whole_sequence": summary.get("whole_sequence", ""),
        "final_pose_yaml": final.get("pose_yaml", ""),
        "final_source": final.get("source", ""),
        "stages": [stage_manifest_entry(stage) for stage in summary.get("stages", [])],
    }


def write_run_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status_class(status):
    if status in {"completed", "present", "complete", "partial_refine", "ok"}:
        return "ok"
    if status in {"planned", "not_requested", "has_yellow_cameras", "partial"}:
        return "warn"
    return "bad"


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def html_path(path, exists=None):
    if not path:
        return ""
    exists = Path(path).exists() if exists is None else exists
    escaped = html.escape(str(path))
    if exists:
        return f'<a href="{html.escape(report_url(path))}">{escaped}</a>'
    return f"<code>{escaped}</code>"


def html_table(rows, columns):
    if not rows:
        return '<p class="note">No rows available.</p>'
    header = "".join(f"<th>{html.escape(label)}</th>" for _key, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            cells.append(f"<td>{html.escape(fmt(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def write_placeholder_viewer(path, summary):
    final = summary["final"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Outer Tower Viewer Placeholder</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2328; }}
    code {{ word-break: break-word; }}
  </style>
</head>
<body>
  <h1>Outer Tower Viewer Placeholder</h1>
  <p>The interactive viewer was not regenerated in this run. Enable <code>--run-viewer</code>, <code>--run-reports</code>, or <code>--run-all</code> to rebuild it.</p>
  <p>Final pose candidate: <code>{html.escape(final["pose_yaml"])}</code></p>
  <p>Metrics candidate: <code>{html.escape(final["metrics_tsv"])}</code></p>
  <p><a href="../final_report/index.html">Final report</a> · <a href="../quality_report/index.html">Quality report</a> · <a href="../summary.json">summary.json</a></p>
</body>
</html>
""", encoding="utf-8")


def write_index(summary, path, report_kind="final"):
    stage_rows = []
    for stage in summary["stages"]:
        commands = "<br>".join(
            f"<code>{html.escape(command)}</code>" for command in stage.get("commands", [])
        )
        missing = "<br>".join(html.escape(item) for item in stage.get("missing_inputs", []))
        stage_rows.append(
            "<tr>"
            f"<td>{html.escape(stage['name'])}</td>"
            f"<td class=\"{status_class(stage.get('status'))}\">{html.escape(stage.get('status', ''))}</td>"
            f"<td>{commands}</td>"
            f"<td>{missing}</td>"
            "</tr>"
        )

    coverage = summary["coverage_gate"]
    side = summary["side_prior"]
    tag = summary["tag_refine"]
    frame_face = summary.get("frame_face_refine", {})
    residual_tail = summary.get("residual_tail", {})
    intrinsic_feature_coverage = summary.get("intrinsic_feature_coverage", {})
    bridge_override = summary.get("bridge_prior_override", {})
    bridge_override_rows = tag.get("bridge_prior_overrides", [])
    bridge_override_text = "; ".join(
        (
            f"{row.get('camera_id')}: "
            f"{fmt(row.get('center_delta_m'))}m, "
            f"{fmt(row.get('rotation_delta_deg'))}deg"
        )
        for row in bridge_override_rows
    )
    gate = tag.get("observation_gate", {})
    post_gate = tag.get("post_refine_observation_gate", {})
    tag_intrinsics = tag.get("intrinsics", {})
    camera_report_columns = [
        ("camera_id", "camera"),
        ("decision", "decision"),
        ("output_pose", "output"),
        ("reason", "reason"),
        ("output_intrinsics", "intrinsics"),
        ("intrinsic_decision", "intrinsic decision"),
        ("max_abs_focal_delta_frac", "max focal delta"),
        ("principal_delta_px", "principal delta px"),
        ("used_observation_count", "used obs"),
        ("after_median_px", "median px"),
        ("after_p90_px", "p90 px"),
        ("after_max_px", "max px"),
        ("after_under_300_fraction", "<300 frac"),
        ("delta_rotation_deg", "delta rot deg"),
        ("delta_translation_m", "delta trans m"),
    ]
    camera_report_table = html_table(
        tag.get("camera_report_rows", []),
        camera_report_columns,
    )
    final = summary["final"]
    capture = summary["capture_quality"]
    provenance = summary.get("provenance", {})
    git_info = provenance.get("git", {})
    git_dirty = "dirty" if git_info.get("dirty") else "clean"
    run_timing = summary.get("run_timing", {})
    manifest_path = summary.get("run_manifest_path") or summary.get("run_manifest", "")
    manifest_url = summary.get("run_manifest_url") or (report_url(manifest_path) if manifest_path else "")
    inputs = summary.get("inputs", {})
    input_rows = [
        ("data_root", inputs.get("data_root", {}).get("path", "")),
        ("whole_dir", inputs.get("whole_dir", {}).get("path", "")),
        ("output_root", summary.get("output_root", "")),
        ("whole_sequence", summary.get("whole_sequence", "")),
        ("dataset", inputs.get("dataset", {}).get("path", "")),
        ("manifest", inputs.get("manifest", {}).get("path", "")),
        ("anchor_pose_yaml", inputs.get("anchor_pose_yaml", {}).get("path", "")),
        ("frame_face_dataset", inputs.get("frame_face_dataset", {}).get("path", "")),
        ("frame_face_prior_pose_yaml", inputs.get("frame_face_prior_pose_yaml", {}).get("path", "")),
        ("final_pose_yaml", final.get("pose_yaml", "")),
        ("final_source", final.get("source", "")),
    ]
    input_table_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for label, value in input_rows
    )
    stage_timing_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(stage.get('name', ''))}</code></td>"
        f"<td class=\"{status_class(stage.get('status'))}\">{html.escape(stage.get('status', ''))}</td>"
        f"<td>{html.escape(str(stage.get('requested', False)))}</td>"
        f"<td>{html.escape(str(duration_s(stage.get('duration_s', stage.get('duration', 0.0)))))}</td>"
        f"<td>{html.escape(stage.get('started_at', ''))}</td>"
        f"<td>{html.escape(stage.get('finished_at', ''))}</td>"
        f"<td><code>{html.escape(stage.get('log', ''))}</code></td>"
        "</tr>"
        for stage in summary["stages"]
    )
    timing_section = ""
    if report_kind != "quality":
        timing_section = f"""
  <h2>Run Timing / Recalib Inputs</h2>
  <p>Total runtime: <strong>{html.escape(str(run_timing.get("total_duration_s", 0.0)))}</strong> s. Started: <code>{html.escape(str(run_timing.get("started_at", "")))}</code>. Finished: <code>{html.escape(str(run_timing.get("finished_at", "")))}</code>. Manifest: <a href="{html.escape(manifest_url)}">run_manifest.json</a> <code>{html.escape(str(manifest_path))}</code>.</p>
  <table>{input_table_rows}</table>
  <table>
    <thead><tr><th>Stage</th><th>Status</th><th>Requested</th><th>Duration s</th><th>Started</th><th>Finished</th><th>Log</th></tr></thead>
    <tbody>{stage_timing_rows}</tbody>
  </table>
"""
    if report_kind == "quality":
        title = "Outer Tower Data Quality Report"
        subtitle = "Capture coverage, frame alignment, COLMAP vote, side-prior, and tag-refine gate status."
    else:
        title = "Outer Tower Recalibration Pipeline"
        subtitle = "Final outer rig artifacts, stage commands, and report links."
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2328; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 7px 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f6f8fa; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin: 18px 0; }}
    .metric {{ border: 1px solid #d0d7de; border-radius: 6px; padding: 10px; }}
    .metric strong {{ display: block; font-size: 20px; }}
    .metric span {{ color: #57606a; font-size: 12px; }}
    .ok {{ color: #116329; font-weight: 600; }}
    .warn {{ color: #9a6700; font-weight: 600; }}
    .bad {{ color: #cf222e; font-weight: 600; }}
    .note {{ color: #57606a; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="note">{html.escape(subtitle)} Generated at {html.escape(summary['generated_at'])}. Dry run: {summary['dry_run']}.</p>
  <p class="note">Provenance: <code>{html.escape(str(git_info.get("branch", "")))}</code> / <code>{html.escape(str(git_info.get("commit", ""))[:12])}</code> ({html.escape(git_dirty)}); command <code>{html.escape(str(provenance.get("argv", "")))}</code>.</p>
  <div class="grid">
    <div class="metric"><strong>{html.escape(capture.get('status', ''))}</strong><span>capture quality status</span></div>
    <div class="metric"><strong>{html.escape(coverage.get('gate', coverage.get('status', '')))}</strong><span>coverage gate</span></div>
    <div class="metric"><strong>{fmt(side.get('completed_pose_count'))}/{fmt(side.get('camera_count'))}</strong><span>side-prior cameras</span></div>
    <div class="metric"><strong>{fmt(tag.get('accepted_refined_count'))}</strong><span>tag-refined accepted cameras</span></div>
  </div>

  <h2>Final Outputs</h2>
  <table>
    <tr><th>Final YAML</th><td>{html_path(final['pose_yaml'], final['pose_yaml_exists'])}</td></tr>
    <tr><th>Final source</th><td>{html.escape(final['source'])}</td></tr>
    <tr><th>Metrics TSV</th><td>{html_path(final['metrics_tsv'], final['metrics_tsv_exists'])}</td></tr>
    <tr><th>Viewer</th><td>{html_path(final['viewer_index'], final['viewer_index_exists'])}<br><code>{html.escape(final['viewer_url'])}</code></td></tr>
    <tr><th>Intrinsic feature coverage report</th><td>{html_path(intrinsic_feature_coverage.get('index_html'), intrinsic_feature_coverage.get('index_html_exists'))}<br><code>{html.escape(intrinsic_feature_coverage.get('index_url', ''))}</code></td></tr>
    <tr><th>Residual-tail report</th><td>{html_path(residual_tail.get('report_html'), residual_tail.get('report_html_exists'))}<br><code>{html.escape(residual_tail.get('report_url', ''))}</code></td></tr>
    <tr><th>Summary JSON</th><td>{html_path(Path(summary['output_root']) / 'summary.json', True)}</td></tr>
  </table>
  {timing_section}

  <h2>Quality Summary</h2>
  <table>
    <tr><th>Manifest cameras</th><td>{fmt(capture.get('camera_count'))}</td></tr>
    <tr><th>Frame counts</th><td>{fmt(capture.get('frame_count_min'))} - {fmt(capture.get('frame_count_max'))} ({html.escape(capture.get('frame_alignment', ''))})</td></tr>
    <tr><th>Coverage counts</th><td><code>{html.escape(json.dumps(coverage.get('status_counts', {}), sort_keys=True))}</code></td></tr>
    <tr><th>Coverage report</th><td>{html_path(coverage.get('report_html'), Path(coverage.get('report_html', '')).is_file() if coverage.get('report_html') else False)}</td></tr>
    <tr><th>Side-prior status</th><td>{html.escape(side.get('status', ''))}; side completed {fmt(side.get('side_prior_completed_count'))}; bridge overrides {fmt(side.get('bridge_pose_override_count'))}; relative pairs {fmt(side.get('relative_pair_success_count'))}/{fmt(side.get('relative_pair_count'))}</td></tr>
    <tr><th>Bridge prior override</th><td>policy {html.escape(bridge_override.get('policy', ''))}; metric gate {html.escape(str(bridge_override.get('metric_gate_status', '')))}; effective labels {html.escape(bridge_override.get('effective_labels', ''))}; deltas {html.escape(bridge_override_text)}</td></tr>
    <tr><th>Tag-refine status</th><td>{html.escape(tag.get('status', ''))}; accepted {html.escape(', '.join(tag.get('accepted_refined', [])))}; prior-only {html.escape(', '.join(tag.get('prior_only', [])))}</td></tr>
    <tr><th>Tag observation gate</th><td>enabled {html.escape(str(gate.get('enabled', False)))}; threshold {fmt(gate.get('max_residual_px'))} px; kept {fmt(gate.get('kept_observations'))}/{fmt(gate.get('input_observations'))}; removed {fmt(gate.get('removed_observations'))}</td></tr>
    <tr><th>Post-refine observation trim</th><td>enabled {html.escape(str(post_gate.get('enabled', False)))}; threshold {fmt(post_gate.get('threshold_px'))} px; kept {fmt(post_gate.get('kept_observations'))}/{fmt(post_gate.get('input_observations'))}; removed {fmt(post_gate.get('removed_observations'))}; second-pass iterations {fmt(post_gate.get('outer_iterations'))}</td></tr>
    <tr><th>Tag intrinsic refine</th><td>mode {html.escape(tag_intrinsics.get('refine_mode', ''))}; accepted {fmt(tag_intrinsics.get('accepted_refined_count'))}; max focal delta {fmt(tag_intrinsics.get('max_abs_focal_delta_frac'))}; max principal delta {fmt(tag_intrinsics.get('max_principal_delta_px'))} px</td></tr>
    <tr><th>Frame-face high-quality refine</th><td>{html.escape(frame_face.get('status', ''))}; active {fmt(frame_face.get('active_delta'))}/{fmt(frame_face.get('camera_count'))}; observations {fmt(frame_face.get('used_observations'))}; median/p90 {fmt(frame_face.get('median_px'))}/{fmt(frame_face.get('p90_px'))} px; inactive {html.escape(', '.join(frame_face.get('inactive_delta', [])))}</td></tr>
    <tr><th>Intrinsic feature coverage</th><td>{html.escape(intrinsic_feature_coverage.get('status', ''))}; cameras {fmt(intrinsic_feature_coverage.get('camera_count'))}; plots {fmt(intrinsic_feature_coverage.get('plot_count'))}; source {html.escape(intrinsic_feature_coverage.get('source_type', ''))}</td></tr>
    <tr><th>Gated residuals</th><td>before median/p90 {fmt(tag.get('residual_before', {}).get('median_px'))}/{fmt(tag.get('residual_before', {}).get('p90_px'))} px; after median/p90 {fmt(tag.get('residual_after', {}).get('median_px'))}/{fmt(tag.get('residual_after', {}).get('p90_px'))} px</td></tr>
    <tr><th>Raw residuals</th><td>before median/p90 {fmt(tag.get('raw_residual_before', {}).get('median_px'))}/{fmt(tag.get('raw_residual_before', {}).get('p90_px'))} px; after median/p90 {fmt(tag.get('raw_residual_after', {}).get('median_px'))}/{fmt(tag.get('raw_residual_after', {}).get('p90_px'))} px</td></tr>
    <tr><th>Final accepted-output residuals</th><td>gated median/p90 {fmt(tag.get('residual_after_output_accepted', {}).get('median_px'))}/{fmt(tag.get('residual_after_output_accepted', {}).get('p90_px'))} px; raw median/p90 {fmt(tag.get('raw_residual_after_output_accepted', {}).get('median_px'))}/{fmt(tag.get('raw_residual_after_output_accepted', {}).get('p90_px'))} px</td></tr>
    <tr><th>Residual-tail diagnostics</th><td>{html.escape(residual_tail.get('status', ''))}; observation-level {html.escape(str(residual_tail.get('observation_diagnostics_available', '')))}; {html.escape(residual_tail.get('observation_diagnostics_message', ''))}</td></tr>
  </table>

  <h2>Per-Camera Tag Acceptance</h2>
  <p class="note">This table joins <code>camera_acceptance.tsv</code>, <code>camera_reprojection.tsv</code>, <code>camera_delta.tsv</code>, and <code>camera_intrinsics.tsv</code>. It is the quickest place to see whether a camera used the refined delta or fell back to its prior pose or prior intrinsics.</p>
  {camera_report_table}

  <h2>Stage Commands</h2>
  <table>
    <thead><tr><th>Stage</th><th>Status</th><th>Commands</th><th>Missing inputs</th></tr></thead>
    <tbody>{''.join(stage_rows)}</tbody>
  </table>
</body>
</html>
"""
    Path(path).write_text(text, encoding="utf-8")


def write_summary_and_index(args, paths, stage_results, run_started_at="", run_finished_at="", total_duration=0.0):
    summary = collect_summary(args, paths, stage_results, run_started_at, run_finished_at, total_duration)
    summary_path = paths["output_root"] / "summary.json"
    manifest_path = paths["output_root"] / "run_manifest.json"
    index_path = paths["output_root"] / "index.html"
    quality_index = paths["quality_report_dir"] / "index.html"
    final_index = paths["final_report_dir"] / "index.html"
    viewer_index = paths["viewer_dir"] / "index.html"
    paths["quality_report_dir"].mkdir(parents=True, exist_ok=True)
    paths["final_report_dir"].mkdir(parents=True, exist_ok=True)
    write_run_manifest(manifest_path, build_run_manifest(summary))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_index(summary, index_path, "final")
    write_index(summary, quality_index, "quality")
    write_index(summary, final_index, "final")
    if not viewer_index.is_file():
        write_placeholder_viewer(viewer_index, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper for the outer tower recalibration path: coverage report, "
            "COLMAP frame vote, side-prior completion, robust tag refinement, and viewer generation."
        )
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--stage-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--whole-dir", type=Path, default=None)
    parser.add_argument("--whole-sequence", default="whole_outer_tower")
    parser.add_argument("--anchor-pose-yaml", type=Path, default=None)
    parser.add_argument("--anchor-label-to-pose-index", default=None)
    parser.add_argument("--previous-outer-rig", type=Path, default=None)
    parser.add_argument("--previous-intrinsics-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--run-colmap-vote", action="store_true")
    parser.add_argument("--run-side-prior", action="store_true")
    parser.add_argument("--run-tag-refine", action="store_true")
    parser.add_argument(
        "--run-frame-face-refine",
        action="store_true",
        help=(
            "Run the independent frame/face AprilTag tower refine path. "
            "This is the current high-quality subset outer-cage candidate path."
        ),
    )
    parser.add_argument("--run-quality", action="store_true")
    parser.add_argument("--run-viewer", action="store_true")
    parser.add_argument("--run-reports", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument(
        "--colmap-bin",
        default="colmap",
        help=(
            "COLMAP executable for frame-vote runs. The frame-vote script auto-detects "
            "the t0 colmap4 conda env when this is left as 'colmap'."
        ),
    )
    parser.add_argument(
        "--colmap-jobs",
        type=int,
        default=1,
        help="Parallel single-frame COLMAP jobs for --run-colmap-vote.",
    )
    parser.add_argument(
        "--bridge-prior-override-policy",
        choices=["gate", "always", "never"],
        default="gate",
        help=(
            "Whether to inject full bridge poses into the outer tag-refine prior. "
            "'gate' enables it only when the bridge metric gate passes."
        ),
    )
    parser.add_argument(
        "--bridge-prior-override-labels",
        default="4-1,4-2,4-3",
        help="Comma-separated top-down outer cameras to override from the bridge full-pose YAML.",
    )
    parser.add_argument(
        "--promote-diagnostic-tag-refine",
        action="store_true",
        help=(
            "Allow non-fixed tag intrinsic-refine diagnostic outputs to become the "
            "final outer rig. Default keeps them diagnostic and falls back to a stable prior."
        ),
    )
    parser.add_argument(
        "--frame-face-refine-preset",
        choices=sorted(FRAME_FACE_REFINE_PRESETS),
        default="wide200_then_gate6",
        help=(
            "Preset for refine_outer_tower_frame_face_planes.py. wide200_then_gate6 "
            "is the production default for black-tile physical-corner datasets: "
            "loose 200px initialization support, robust BA, then strict 6px "
            "final re-gating. wide50_* presets are legacy diagnostics for older "
            "initialization behavior. *_flex_faces presets relax the ideal "
            "octagonal tower with bounded per-face yaw/radial/tangent offsets."
        ),
    )
    parser.add_argument("--frame-face-dataset", type=Path, default=None)
    parser.add_argument("--frame-face-manifest", type=Path, default=None)
    parser.add_argument("--frame-face-prior-pose-yaml", type=Path, default=None)
    parser.add_argument("--frame-face-intrinsics-dir", type=Path, default=None)
    parser.add_argument("--frame-face-output-dir", type=Path, default=None)
    parser.add_argument(
        "--tower-detector-tag-size-m",
        type=float,
        default=DEFAULT_TOWER_DETECTOR_TAG_SIZE_M,
        help=(
            "Physical side length represented by the 2D tower corner observations. "
            "Production black-tile datasets use the 8 cm printed black tile footprint. "
            "Pass 0.06710408594834662 only for legacy raw OpenCV inner-detector-corner datasets."
        ),
    )
    parser.add_argument(
        "--tower-tag-center-pitch-m",
        type=float,
        default=DEFAULT_TOWER_TAG_CENTER_PITCH_M,
        help=(
            "Center-to-center pitch between adjacent tags on one tower face. "
            "The studio tower uses 8 cm printed tiles with 2 cm tile gaps, so this is 10 cm."
        ),
    )
    parser.add_argument("--tag-observation-residual-gate-px", type=float, default=600.0)
    parser.add_argument("--tag-post-refine-observation-residual-gate-px", type=float, default=190.0)
    parser.add_argument("--tag-post-refine-outer-iterations", type=int, default=2)
    parser.add_argument("--tag-residual-clip-px", type=float, default=500.0)
    parser.add_argument("--tag-max-pnp-median-error-px", type=float, default=8.0)
    parser.add_argument(
        "--tag-pnp-pose-consensus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter per-view PnP poses to the dominant synchronized-frame "
            "pose cluster before initializing tower poses."
        ),
    )
    parser.add_argument("--tag-pnp-consensus-center-threshold-m", type=float, default=0.35)
    parser.add_argument("--tag-pnp-consensus-rotation-threshold-deg", type=float, default=15.0)
    parser.add_argument("--tag-pnp-consensus-min-points", type=int, default=16)
    parser.add_argument("--tag-pnp-consensus-min-inliers", type=int, default=16)
    parser.add_argument("--tag-pnp-consensus-min-votes", type=int, default=1)
    parser.add_argument("--tag-accept-camera-median-px", type=float, default=350.0)
    parser.add_argument("--tag-accept-camera-p90-px", type=float, default=450.0)
    parser.add_argument("--tag-accept-under-300-fraction", type=float, default=0.45)
    parser.add_argument("--tag-accept-max-delta-translation-m", type=float, default=0.35)
    parser.add_argument("--tag-accept-max-delta-rotation-deg", type=float, default=6.5)
    parser.add_argument("--tag-min-camera-observations-for-use", type=int, default=16)
    parser.add_argument("--tag-min-camera-observations-for-delta", type=int, default=10)
    parser.add_argument("--tag-outer-iterations", type=int, default=5)
    parser.add_argument("--tag-block-iterations", type=int, default=8)
    parser.add_argument(
        "--tag-intrinsics-mode",
        choices=["colmap_fixed", "central_opencv"],
        default="colmap_fixed",
        help=(
            "Intrinsics model for outer tower tag delta refine. colmap_fixed is the current "
            "robust default for sparse whole captures; central_opencv uses --previous-intrinsics-dir."
        ),
    )
    parser.add_argument(
        "--tag-intrinsics-refine-mode",
        choices=["fixed", "shared_fxfy", "per_camera_fxfy", "per_camera_fxfycxcy", "per_camera_opencv5"],
        default="fixed",
        help=(
            "Opt-in intrinsic refinement inside tag refine. per_camera_opencv5 also "
            "refines OpenCV k1/k2/p1/p2/k3 deltas."
        ),
    )
    parser.add_argument("--tag-intrinsics-focal-sigma-frac", type=float, default=0.01)
    parser.add_argument("--tag-intrinsics-principal-sigma-px", type=float, default=8.0)
    parser.add_argument("--tag-intrinsics-distortion-sigma", type=float, default=0.05)
    parser.add_argument("--tag-intrinsics-max-focal-step-frac", type=float, default=0.002)
    parser.add_argument("--tag-intrinsics-max-principal-step-px", type=float, default=1.0)
    parser.add_argument("--tag-intrinsics-max-distortion-step", type=float, default=0.01)
    parser.add_argument("--tag-intrinsics-max-total-focal-delta-frac", type=float, default=0.02)
    parser.add_argument("--tag-intrinsics-max-total-principal-delta-px", type=float, default=16.0)
    parser.add_argument("--tag-intrinsics-max-total-distortion-delta", type=float, default=0.0)
    parser.add_argument("--tag-intrinsics-block-iterations", type=int, default=4)
    parser.add_argument("--tag-accept-max-intrinsic-focal-delta-frac", type=float, default=0.02)
    parser.add_argument("--tag-accept-max-intrinsic-principal-delta-px", type=float, default=16.0)
    parser.add_argument("--tag-accept-max-intrinsic-distortion-delta", type=float, default=0.15)
    parser.add_argument(
        "--tag-optimize-tower-face-width",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine one global AprilTag tower face-width delta inside tag BA.",
    )
    parser.add_argument("--tag-tower-face-width-initial-m", type=float, default=0.25)
    parser.add_argument("--tag-tower-face-width-sigma-m", type=float, default=0.03)
    parser.add_argument("--tag-tower-face-width-min-m", type=float, default=0.18)
    parser.add_argument("--tag-tower-face-width-max-m", type=float, default=0.32)
    parser.add_argument("--tag-tower-face-width-max-step-m", type=float, default=0.005)
    parser.add_argument("--delta-translation-sigma-m", type=float, default=0.12)
    parser.add_argument("--delta-rotation-sigma-deg", type=float, default=3.0)
    args = parser.parse_args()
    if args.tower_detector_tag_size_m <= 0:
        parser.error("--tower-detector-tag-size-m must be positive")
    if args.tower_tag_center_pitch_m <= args.tower_detector_tag_size_m:
        parser.error("--tower-tag-center-pitch-m must be larger than --tower-detector-tag-size-m")
    return args


def main():
    run_started_at = utc_now()
    run_started_perf = time.time()
    args = parse_args()
    paths = build_paths(args)
    stages = build_stage_plan(args, paths)
    stage_results = execute_stages(stages, paths, args.dry_run, args.force)
    run_finished_at = utc_now()
    total_duration = time.time() - run_started_perf
    summary = write_summary_and_index(
        args,
        paths,
        stage_results,
        run_started_at,
        run_finished_at,
        total_duration,
    )

    print(json.dumps({
        "summary_json": str(paths["output_root"] / "summary.json"),
        "run_manifest_json": str(paths["output_root"] / "run_manifest.json"),
        "index_html": str(paths["output_root"] / "index.html"),
        "final_pose_yaml": summary["final"]["pose_yaml"],
        "viewer_url": summary["final"]["viewer_url"],
        "stage_status": {stage["name"]: stage["status"] for stage in stage_results},
    }, indent=2, sort_keys=True))
    if args.dry_run:
        print("\nPlanned commands:")
        for stage in stage_results:
            if stage["requested"]:
                print(f"[{stage['name']}] {stage['status']}")
                for command in stage["commands"]:
                    print(command)


if __name__ == "__main__":
    raise SystemExit(main())
