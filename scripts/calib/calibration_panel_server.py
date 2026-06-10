#!/usr/bin/env python3
"""Localhost operation panel for t0 calibration workflows.

The server intentionally exposes only named run modes. Browser requests never
provide shell commands; each mode expands to argv lists owned by this file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlparse
import uuid


DEFAULT_T0_PYTHON = "/home/ubuntu/miniconda3/bin/python"
DEFAULT_T0_REPO = "/home/ubuntu/camera_calibration"
DEFAULT_T0_STAGE_ROOT = "/home/ubuntu/calib_data/calib_2026_05_31_v3"
DEFAULT_T0_WHOLE_ROOT = "/home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1"
DEFAULT_T0_WHOLE_OUTER24_DIR = (
    DEFAULT_T0_WHOLE_ROOT + "/whole_outer24_filtered_min4_fullres_min4cam"
)
DEFAULT_T0_BINARY = (
    "/home/ubuntu/camera_calibration/build_t0_current/applications/"
    "camera_calibration/camera_calibration"
)
DEFAULT_TRUSTED_INNER_ROOT = "/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3"
DEFAULT_INNER_STATE = (
    DEFAULT_TRUSTED_INNER_ROOT
    + "/final_inner8_calibration_v1/states/final_small_marker_grid4_refine_v1"
)
DEFAULT_THREEJS_ASSETS = (
    DEFAULT_TRUSTED_INNER_ROOT
    + "/final_inner8_calibration_v1/reports/interactive_rig_viewer_v1"
)
SMALL_MARKER_PATTERN = (
    "applications/camera_calibration/patterns/"
    "pattern_resolution_50x72_segments_16_apriltag_3.yaml"
)
DEFAULT_CURRENT_STUDIO_RUN = (
    "/home/ubuntu/calib_data/studio_calibration_runs/"
    "recalib_20260610_black_tile_wide200_pipeline_v2"
)
DEFAULT_CURRENT_OUTER_FRAME_FACE_DIR = (
    DEFAULT_CURRENT_STUDIO_RUN
    + "/outer_tower/frame_face_refine_wide200_then_gate6"
)
DEFAULT_CURRENT_OUTER_POSE_YAML = (
    DEFAULT_CURRENT_OUTER_FRAME_FACE_DIR + "/camera_tr_rig_delta_refined.yaml"
)
DEFAULT_CURRENT_OUTER_INTRINSICS_DIR = (
    DEFAULT_CURRENT_OUTER_FRAME_FACE_DIR + "/intrinsics_refined"
)
DEFAULT_STUDIO_PIPELINE_OUTPUT = (
    "/home/ubuntu/calib_data/studio_calibration_runs/latest"
)
DEFAULT_FAST_INNER_BRIDGE_OUTPUT = (
    "/home/ubuntu/calib_data/studio_calibration_runs/latest_inner_bridge"
)
DEFAULT_SMALL_MARKER_OPERATION_OUTPUT = (
    "/home/ubuntu/calib_data/studio_calibration_runs/latest_small_marker_inner"
)
DEFAULT_LARGE_MARKER_OPERATION_OUTPUT = (
    "/home/ubuntu/calib_data/studio_calibration_runs/latest_large_marker_bridge"
)
DEFAULT_OUTER_TOWER_OUTPUT = (
    "/home/ubuntu/calib_data/studio_calibration_runs/latest_outer_tower"
)
DEFAULT_WHOLE_OPERATION_OUTPUT = (
    "/home/ubuntu/calib_data/studio_calibration_runs/latest_whole_outer_cage"
)
DEFAULT_REPORT_HTTP_ROOT = "/home/ubuntu/calib_data"
DEFAULT_REPORT_URL_BASE = "http://192.168.2.0:9899"


MODE_DEFINITIONS = {
    "run_studio_calibration_pipeline": {
        "title": "Studio 32-Camera Production Pipeline",
        "operator_summary": (
            "Run the current reproducible production wrapper: whole outer frame-face "
            "refine, large-marker bridge, unified 32-camera export, and optional "
            "current report publication. Pipeline --dry-run is enabled by default."
        ),
        "params": [
            {
                "name": "script",
                "label": "Pipeline script",
                "type": "text",
                "default": "scripts/calib/run_studio_calibration_pipeline.py",
            },
            {
                "name": "whole_data_root",
                "label": "Whole data root",
                "type": "text",
                "default": DEFAULT_T0_WHOLE_ROOT,
            },
            {
                "name": "inner_data_root",
                "label": "Inner/bridge data root",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT,
            },
            {
                "name": "output_root",
                "label": "Pipeline output root",
                "type": "text",
                "default": DEFAULT_STUDIO_PIPELINE_OUTPUT,
            },
            {"name": "run_tag", "label": "Run tag", "type": "text", "default": "latest"},
            {"name": "outer_preset", "label": "Outer frame-face preset", "type": "text", "default": "wide200_then_gate6"},
            {"name": "outer_only", "label": "Outer only", "type": "checkbox", "default": False},
            {"name": "bridge_only", "label": "Bridge/export only", "type": "checkbox", "default": False},
            {"name": "run_large_inner_init", "label": "Run large-inner init", "type": "checkbox", "default": False},
            {"name": "run_small_quality", "label": "Run small-marker quality", "type": "checkbox", "default": True},
            {"name": "publish_current", "label": "Publish current 9899 entry", "type": "checkbox", "default": False},
            {
                "name": "outer_frame_face_prior_pose_yaml",
                "label": "Outer frame-face prior pose YAML",
                "type": "text",
                "default": DEFAULT_CURRENT_OUTER_POSE_YAML,
            },
            {
                "name": "outer_frame_face_intrinsics_dir",
                "label": "Outer frame-face prior intrinsics",
                "type": "text",
                "default": DEFAULT_CURRENT_OUTER_INTRINSICS_DIR,
            },
            {
                "name": "pipeline_dry_run",
                "label": "Pass --dry-run to pipeline",
                "type": "checkbox",
                "default": True,
            },
            {
                "name": "force_pipeline_outputs",
                "label": "Force recompute requested stages",
                "type": "checkbox",
                "default": True,
            },
        ],
    },
    "operate_whole_outer_cage": {
        "title": "Whole Operation: Outer Cage",
        "operator_summary": (
            "Process whole capture into the outer24 studio cage calibration. "
            "This is the user-facing operation for Whole data."
        ),
        "params": [
            {
                "name": "script",
                "label": "Backend script",
                "type": "text",
                "default": "scripts/calib/run_outer_tower_recalib_pipeline.py",
            },
            {
                "name": "data_root",
                "label": "Whole data root",
                "type": "text",
                "default": DEFAULT_T0_WHOLE_ROOT,
            },
            {
                "name": "whole_dir",
                "label": "Whole capture directory",
                "type": "text",
                "default": DEFAULT_T0_WHOLE_OUTER24_DIR,
            },
            {
                "name": "output_root",
                "label": "Operation output root",
                "type": "text",
                "default": DEFAULT_WHOLE_OPERATION_OUTPUT,
            },
            {"name": "sample_count", "label": "COLMAP sample count", "type": "number", "default": 32},
            {"name": "colmap_jobs", "label": "COLMAP frame jobs", "type": "number", "default": 4},
            {"name": "run_colmap_vote", "label": "Run COLMAP vote", "type": "checkbox", "default": False},
            {"name": "run_side_prior", "label": "Run side prior", "type": "checkbox", "default": False},
            {"name": "run_tag_refine", "label": "Run tag refine", "type": "checkbox", "default": False},
            {"name": "run_frame_face_refine", "label": "Run frame-face refine", "type": "checkbox", "default": True},
            {"name": "frame_face_refine_preset", "label": "Frame-face preset", "type": "text", "default": "wide200_then_gate6"},
            {
                "name": "frame_face_prior_pose_yaml",
                "label": "Frame-face prior pose YAML",
                "type": "text",
                "default": DEFAULT_CURRENT_OUTER_POSE_YAML,
            },
            {
                "name": "frame_face_intrinsics_dir",
                "label": "Frame-face prior intrinsics",
                "type": "text",
                "default": DEFAULT_CURRENT_OUTER_INTRINSICS_DIR,
            },
            {"name": "run_quality", "label": "Run quality report", "type": "checkbox", "default": True},
            {"name": "run_viewer", "label": "Run viewer", "type": "checkbox", "default": False},
            {"name": "run_reports", "label": "Run final reports", "type": "checkbox", "default": True},
            {
                "name": "force_pipeline_outputs",
                "label": "Force recompute requested stages",
                "type": "checkbox",
                "default": True,
            },
            {
                "name": "pipeline_dry_run",
                "label": "Pass --dry-run to pipeline",
                "type": "checkbox",
                "default": True,
            },
        ],
    },
    "operate_large_marker_bridge": {
        "title": "Large Marker Operation: Inner/Outer Bridge",
        "operator_summary": (
            "Process large-marker captures into the bridge between inner cameras "
            "and outer cameras."
        ),
        "params": [
            {
                "name": "script",
                "label": "Backend script",
                "type": "text",
                "default": "scripts/calib/run_inner_bridge_recalib_pipeline.py",
            },
            {
                "name": "data_root",
                "label": "Data root",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT,
            },
            {
                "name": "large_inner_marker",
                "label": "Large-marker inner init path",
                "type": "text",
                "default": "large_marker_inner8",
            },
            {
                "name": "large_marker",
                "label": "Large-marker bridge path",
                "type": "text",
                "default": "large_marker_bridge_all32",
            },
            {
                "name": "output_root",
                "label": "Operation output root",
                "type": "text",
                "default": DEFAULT_LARGE_MARKER_OPERATION_OUTPUT,
            },
            {"name": "large_inner_frame_stride", "label": "Large-inner frame stride", "type": "number", "default": 1},
            {"name": "large_frame_stride", "label": "Large bridge frame stride", "type": "number", "default": 1},
            {"name": "run_large_inner_init", "label": "Run large-inner init", "type": "checkbox", "default": True},
            {"name": "run_large_bridge", "label": "Run bridge solve", "type": "checkbox", "default": True},
            {"name": "run_reports", "label": "Run reports", "type": "checkbox", "default": True},
            {
                "name": "force_pipeline_outputs",
                "label": "Force recompute requested stages",
                "type": "checkbox",
                "default": True,
            },
            {
                "name": "pipeline_dry_run",
                "label": "Pass --dry-run to pipeline",
                "type": "checkbox",
                "default": True,
            },
        ],
    },
    "operate_small_marker_inner": {
        "title": "Small Marker Operation: Inner Cameras",
        "operator_summary": (
            "Process small-marker capture into inner-camera calibration reports "
            "and quality gates."
        ),
        "params": [
            {
                "name": "script",
                "label": "Backend script",
                "type": "text",
                "default": "scripts/calib/run_inner_bridge_recalib_pipeline.py",
            },
            {
                "name": "data_root",
                "label": "Data root",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT,
            },
            {
                "name": "small_marker",
                "label": "Small-marker path",
                "type": "text",
                "default": "small_marker_inner8",
            },
            {
                "name": "output_root",
                "label": "Operation output root",
                "type": "text",
                "default": DEFAULT_SMALL_MARKER_OPERATION_OUTPUT,
            },
            {
                "name": "inner_refine_mode",
                "label": "Inner refine mode",
                "type": "text",
                "default": "fixed_then_joint",
                "placeholder": "fixed, joint, or fixed_then_joint",
            },
            {"name": "small_frame_stride", "label": "Small frame stride", "type": "number", "default": 4},
            {"name": "inner_fixed_max_ba_iterations", "label": "Fixed BA iterations", "type": "number", "default": 3},
            {"name": "inner_joint_max_ba_iterations", "label": "Joint BA iterations", "type": "number", "default": 3},
            {"name": "run_small_fixed_rig_quality", "label": "Run fixed-rig quality probe", "type": "checkbox", "default": True},
            {"name": "run_small_refine", "label": "Run small-marker inner refine", "type": "checkbox", "default": True},
            {"name": "run_reports", "label": "Run reports", "type": "checkbox", "default": True},
            {
                "name": "force_pipeline_outputs",
                "label": "Force recompute requested stages",
                "type": "checkbox",
                "default": True,
            },
            {
                "name": "pipeline_dry_run",
                "label": "Pass --dry-run to pipeline",
                "type": "checkbox",
                "default": True,
            },
        ],
    },
    "run_inner_bridge_recalib_pipeline": {
        "title": "Diagnostic: Inner/Bridge Wrapper",
        "operator_summary": (
            "Advanced wrapper for large-marker, small-marker, and bridge "
            "diagnostics. For routine production runs prefer the semantic "
            "Large Marker and Small Marker operation modes. Pipeline --dry-run "
            "is enabled by default."
        ),
        "params": [
            {
                "name": "script",
                "label": "Pipeline script",
                "type": "text",
                "default": "scripts/calib/run_inner_bridge_recalib_pipeline.py",
            },
            {
                "name": "data_root",
                "label": "Data root",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT,
            },
            {
                "name": "output_root",
                "label": "Latest output root",
                "type": "text",
                "default": DEFAULT_FAST_INNER_BRIDGE_OUTPUT,
            },
            {
                "name": "small_marker",
                "label": "Small-marker path",
                "type": "text",
                "default": "small_marker_inner8",
                "placeholder": "relative paths resolve under data root",
            },
            {
                "name": "large_marker",
                "label": "Large-marker bridge path",
                "type": "text",
                "default": "large_marker_bridge_all32",
                "placeholder": "relative paths resolve under data root",
            },
            {
                "name": "large_inner_marker",
                "label": "Large-marker inner init path",
                "type": "text",
                "default": "large_marker_inner8",
                "placeholder": "relative paths resolve under data root",
            },
            {
                "name": "inner_prior",
                "label": "Inner prior state",
                "type": "text",
                "default": "",
                "placeholder": "wrapper default",
            },
            {
                "name": "outer_prior",
                "label": "Outer prior images.txt",
                "type": "text",
                "default": "",
                "placeholder": "wrapper default",
            },
            {
                "name": "pipeline_dry_run",
                "label": "Pass --dry-run to pipeline",
                "type": "checkbox",
                "default": True,
            },
            {
                "name": "force_pipeline_outputs",
                "label": "Force recompute requested stages",
                "type": "checkbox",
                "default": True,
            },
            {
                "name": "inner_refine_mode",
                "label": "Inner refine mode",
                "type": "text",
                "default": "fixed_rig",
                "placeholder": "fixed_rig, fixed, joint, or fixed_then_joint",
            },
            {"name": "inner_fixed_max_ba_iterations", "label": "Diagnostic fixed BA iterations", "type": "number", "default": 3},
            {"name": "inner_joint_max_ba_iterations", "label": "Joint BA iterations", "type": "number", "default": 3},
            {
                "name": "inner_schur_mode",
                "label": "Inner Schur mode",
                "type": "text",
                "default": "sparse_onthefly",
                "placeholder": "sparse_onthefly",
            },
            {"name": "small_frame_stride", "label": "Small frame stride", "type": "number", "default": 4},
            {"name": "large_inner_frame_stride", "label": "Large-inner frame stride", "type": "number", "default": 1},
            {"name": "large_frame_stride", "label": "Large frame stride", "type": "number", "default": 1},
            {"name": "run_large_inner_init", "label": "Run large-inner init", "type": "checkbox", "default": True},
            {"name": "run_small_fixed_rig_quality", "label": "Run small fixed-rig quality", "type": "checkbox", "default": True},
            {"name": "run_small_refine", "label": "Run small localize/joint diagnostic", "type": "checkbox", "default": False},
            {"name": "run_large_bridge", "label": "Run bridge solve", "type": "checkbox", "default": True},
            {"name": "run_reports", "label": "Run reports", "type": "checkbox", "default": True},
        ],
    },
    "run_outer_tower_recalib_pipeline": {
        "title": "Diagnostic: Outer Tower Wrapper",
        "operator_summary": (
            "Advanced whole/tower wrapper that can re-enable bootstrap and "
            "diagnostic stages. For routine production outer refresh prefer "
            "the Whole Operation mode. Pipeline --dry-run is enabled by default."
        ),
        "params": [
            {
                "name": "script",
                "label": "Pipeline script",
                "type": "text",
                "default": "scripts/calib/run_outer_tower_recalib_pipeline.py",
            },
            {
                "name": "data_root",
                "label": "Data root",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT,
            },
            {
                "name": "output_root",
                "label": "Latest output root",
                "type": "text",
                "default": DEFAULT_OUTER_TOWER_OUTPUT,
            },
            {
                "name": "whole_dir",
                "label": "Whole tower dir",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT + "/whole_outer_tower",
            },
            {
                "name": "previous_outer_rig",
                "label": "Previous outer rig YAML",
                "type": "text",
                "default": "",
                "placeholder": "wrapper default",
            },
            {
                "name": "anchor_pose_yaml",
                "label": "Bridge anchor pose YAML",
                "type": "text",
                "default": "",
                "placeholder": "wrapper default",
            },
            {
                "name": "anchor_label_to_pose_index",
                "label": "Anchor label:index map",
                "type": "text",
                "default": "",
                "placeholder": "4-1:9,4-2:10,4-3:11 for all32 bridge",
            },
            {
                "name": "previous_intrinsics_dir",
                "label": "Previous intrinsics dir",
                "type": "text",
                "default": "",
                "placeholder": "wrapper default",
            },
            {"name": "sample_count", "label": "COLMAP sample count", "type": "number", "default": 32},
            {"name": "colmap_jobs", "label": "COLMAP frame jobs", "type": "number", "default": 4},
            {"name": "run_colmap_vote", "label": "Run COLMAP vote", "type": "checkbox", "default": True},
            {"name": "run_side_prior", "label": "Run side prior", "type": "checkbox", "default": True},
            {"name": "run_tag_refine", "label": "Run tag refine", "type": "checkbox", "default": True},
            {"name": "run_frame_face_refine", "label": "Run frame-face refine", "type": "checkbox", "default": True},
            {"name": "frame_face_refine_preset", "label": "Frame-face preset", "type": "text", "default": "wide200_then_gate6"},
            {
                "name": "frame_face_prior_pose_yaml",
                "label": "Frame-face prior pose YAML",
                "type": "text",
                "default": DEFAULT_CURRENT_OUTER_POSE_YAML,
            },
            {
                "name": "frame_face_intrinsics_dir",
                "label": "Frame-face prior intrinsics",
                "type": "text",
                "default": DEFAULT_CURRENT_OUTER_INTRINSICS_DIR,
            },
            {"name": "run_quality", "label": "Run quality report", "type": "checkbox", "default": True},
            {"name": "run_viewer", "label": "Run viewer", "type": "checkbox", "default": True},
            {"name": "run_reports", "label": "Run final reports", "type": "checkbox", "default": True},
            {
                "name": "force_pipeline_outputs",
                "label": "Force recompute requested stages",
                "type": "checkbox",
                "default": True,
            },
            {
                "name": "tag_intrinsics_mode",
                "label": "Tag intrinsics mode",
                "type": "text",
                "default": "colmap_fixed",
                "placeholder": "colmap_fixed or central_opencv",
            },
            {
                "name": "tag_intrinsics_refine_mode",
                "label": "Tag intrinsic refine mode",
                "type": "text",
                "default": "fixed",
                "placeholder": "fixed, shared_fxfy, per_camera_fxfy, or per_camera_fxfycxcy",
            },
            {"name": "tag_intrinsics_focal_sigma_frac", "label": "Tag focal sigma frac", "type": "number", "default": 0.01},
            {"name": "tag_intrinsics_max_focal_step_frac", "label": "Tag focal max step frac", "type": "number", "default": 0.002},
            {"name": "tag_intrinsics_block_iterations", "label": "Tag intrinsic block iterations", "type": "number", "default": 4},
            {"name": "tag_min_camera_observations_for_use", "label": "Tag min obs for use", "type": "number", "default": 16},
            {"name": "tag_min_camera_observations_for_delta", "label": "Tag min obs for delta", "type": "number", "default": 10},
            {"name": "tag_post_refine_observation_residual_gate_px", "label": "Post-refine trim px", "type": "number", "default": 190},
            {"name": "tag_post_refine_outer_iterations", "label": "Post-refine iterations", "type": "number", "default": 2},
            {
                "name": "pipeline_dry_run",
                "label": "Pass --dry-run to pipeline",
                "type": "checkbox",
                "default": True,
            },
        ],
    },
    "stage_data": {
        "title": "Stage Current Capture Data",
        "operator_summary": "Build normalized symlink datasets from mounted Windows D shares.",
        "params": [
            {
                "name": "mount_root",
                "label": "Mount root",
                "type": "text",
                "default": "/home/ubuntu/cameras_mount",
            },
            {
                "name": "output_root",
                "label": "New staging output root",
                "type": "text",
                "default": "",
                "placeholder": "defaults to this job run dir",
            },
            {
                "name": "max_tail_trim",
                "label": "Max tail trim",
                "type": "number",
                "default": 2,
            },
        ],
    },
    "distributed_qc": {
        "title": "Distributed Windows QC",
        "operator_summary": "Run or aggregate loose SSH OpenCV AprilTag QC on w1-w4.",
        "params": [
            {
                "name": "config",
                "label": "Client config JSON",
                "type": "text",
                "default": "configs/distributed_calib_clients.example.json",
            },
            {
                "name": "output_dir",
                "label": "QC output dir",
                "type": "text",
                "default": "",
                "placeholder": "defaults to this job run dir",
            },
            {"name": "run", "label": "Run clients", "type": "checkbox", "default": True},
            {"name": "collect", "label": "Collect reports", "type": "checkbox", "default": True},
            {
                "name": "aggregate_only",
                "label": "Aggregate only",
                "type": "checkbox",
                "default": False,
            },
            {"name": "jobs", "label": "Parallel SSH jobs", "type": "number", "default": 4},
            {"name": "timeout_sec", "label": "Client timeout sec", "type": "number", "default": 0},
        ],
    },
    "inner_warm_start_refine": {
        "title": "Inner Warm-Start Refine",
        "operator_summary": (
            "Extract small-marker features, grid-subsample, refine from the saved "
            "inner snapshot, then build reprojection and interactive rig reports."
        ),
        "params": [
            {"name": "binary", "label": "camera_calibration binary", "type": "text", "default": DEFAULT_T0_BINARY},
            {
                "name": "small_image_dirs_file",
                "label": "Small-marker image dirs file",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/image_directories.txt",
            },
            {"name": "small_pattern_file", "label": "Small-marker pattern YAML", "type": "text", "default": SMALL_MARKER_PATTERN},
            {
                "name": "manifest",
                "label": "Inner manifest TSV",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/manifest.tsv",
            },
            {"name": "warm_start_state", "label": "Warm-start state dir", "type": "text", "default": DEFAULT_INNER_STATE},
            {
                "name": "output_root",
                "label": "Mode output root",
                "type": "text",
                "default": "",
                "placeholder": "defaults to this job run dir",
            },
            {"name": "feature_jobs", "label": "Feature extraction jobs", "type": "number", "default": 8},
            {"name": "grid_stride", "label": "Pattern grid stride", "type": "number", "default": 4},
            {"name": "min_features_per_camera_view", "label": "Min features per camera view", "type": "number", "default": 20},
            {"name": "max_ba_iterations", "label": "Max BA iterations", "type": "number", "default": 6},
            {"name": "overwrite_features", "label": "Overwrite feature shards", "type": "checkbox", "default": False},
            {
                "name": "camera_image_dirs_file",
                "label": "Camera image dirs for viewer",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/image_directories.txt",
            },
            {"name": "threejs_assets_dir", "label": "Three.js assets dir", "type": "text", "default": DEFAULT_THREEJS_ASSETS},
            {"name": "sparse_point_cloud_json", "label": "Optional sparse point cloud JSON", "type": "text", "default": ""},
        ],
    },
    "report_only": {
        "title": "Report Only",
        "operator_summary": "Generate reprojection, rig extrinsics, and Three.js reports from an existing dataset and state.",
        "params": [
            {
                "name": "dataset_path",
                "label": "Dataset bin",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/features_pattern3_grid4_v1.bin",
            },
            {"name": "state_dir", "label": "State dir for reprojection", "type": "text", "default": DEFAULT_INNER_STATE},
            {
                "name": "rig_state_dir",
                "label": "State dir for rig report",
                "type": "text",
                "default": "",
                "placeholder": "defaults to state dir",
            },
            {
                "name": "manifest",
                "label": "Manifest TSV",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/manifest.tsv",
            },
            {
                "name": "output_dir",
                "label": "Report output dir",
                "type": "text",
                "default": "",
                "placeholder": "defaults to this job run dir",
            },
            {
                "name": "camera_image_dirs_file",
                "label": "Camera image dirs for viewer",
                "type": "text",
                "default": DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/image_directories.txt",
            },
            {"name": "threejs_assets_dir", "label": "Three.js assets dir", "type": "text", "default": DEFAULT_THREEJS_ASSETS},
            {"name": "sparse_point_cloud_json", "label": "Optional sparse point cloud JSON", "type": "text", "default": ""},
            {"name": "title", "label": "Interactive report title", "type": "text", "default": "Inner Camera Calibration Report"},
        ],
    },
}


class JobStep:
    def __init__(self, name, argv=None, cwd=None, env=None, internal=None, kwargs=None):
        self.name = name
        self.argv = list(argv) if argv else []
        self.cwd = str(cwd) if cwd else ""
        self.env = dict(env or {})
        self.internal = internal or ""
        self.kwargs = dict(kwargs or {})

    def to_json(self):
        item = {
            "name": self.name,
            "status": "pending",
            "returncode": None,
        }
        if self.argv:
            item["argv"] = self.argv
            item["command"] = shlex.join(str(part) for part in self.argv)
        if self.cwd:
            item["cwd"] = self.cwd
        if self.internal:
            item["internal"] = self.internal
            item["kwargs"] = self.kwargs
        return item


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_python_bin():
    if Path(DEFAULT_T0_PYTHON).exists():
        return DEFAULT_T0_PYTHON
    return sys.executable


def file_url(path):
    try:
        return Path(path).resolve().as_uri()
    except ValueError:
        return ""


def public_url_for_path(path):
    path = Path(path)
    try:
        rel = path.expanduser().resolve(strict=False).relative_to(
            Path(DEFAULT_REPORT_HTTP_ROOT).resolve(strict=False))
        return DEFAULT_REPORT_URL_BASE.rstrip("/") + "/" + "/".join(rel.parts)
    except ValueError:
        return file_url(path)


def as_path(value, base=None):
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path


def string_param(params, name, default="", required=False):
    value = params.get(name, default)
    if value is None:
        value = ""
    value = str(value).strip()
    if required and not value:
        raise ValueError(f"Missing required parameter: {name}")
    return value


def int_param(params, name, default=0, minimum=None):
    value = params.get(name, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter {name} must be an integer.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"Parameter {name} must be >= {minimum}.")
    return parsed


def float_param(params, name, default=0.0, minimum=None):
    value = params.get(name, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter {name} must be a number.") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"Parameter {name} must be >= {minimum}.")
    return parsed


def bool_param(params, name, default=False):
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def report_item(label, path, primary=False):
    path = Path(path)
    return {
        "label": label,
        "path": str(path),
        "url": public_url_for_path(path),
        "primary": bool(primary),
        "exists": path.exists(),
    }


def command_env(extra=None):
    env = {"QT_QPA_PLATFORM": "offscreen"}
    if extra:
        env.update(extra)
    return env


class PlanContext:
    def __init__(self, repo_root, python_bin):
        self.repo_root = Path(repo_root).resolve()
        self.python_bin = str(python_bin)

    def script(self, relative):
        return str((self.repo_root / relative).resolve())


def build_stage_data_plan(params, context, job_dir):
    mount_root = string_param(params, "mount_root", "/home/ubuntu/cameras_mount", required=True)
    output_root = string_param(params, "output_root", "") or str(job_dir / "staged_data")
    max_tail_trim = int_param(params, "max_tail_trim", 2, minimum=0)
    argv = [
        context.python_bin,
        context.script("scripts/ops/t0_stage_current_calib_data.py"),
        "--mount-root",
        mount_root,
        "--output-root",
        output_root,
        "--max-tail-trim",
        str(max_tail_trim),
    ]
    return {
        "steps": [JobStep("Stage data from mounted camera shares", argv=argv, cwd=context.repo_root)],
        "reports": [
            report_item("Staging summary JSON", Path(output_root) / "summary.json"),
        ],
    }


def build_distributed_qc_plan(params, context, job_dir):
    config = as_path(
        string_param(params, "config", "configs/distributed_calib_clients.example.json", required=True),
        context.repo_root,
    )
    output_dir_value = string_param(params, "output_dir", "")
    output_dir = as_path(output_dir_value, context.repo_root) if output_dir_value else job_dir / "distributed_qc"
    argv = [
        context.python_bin,
        context.script("scripts/calib/server_run_distributed_clients.py"),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--jobs",
        str(int_param(params, "jobs", 4, minimum=1)),
        "--timeout-sec",
        str(int_param(params, "timeout_sec", 0, minimum=0)),
    ]
    for flag_name in ("run", "collect", "aggregate_only"):
        if bool_param(params, flag_name, flag_name in ("run", "collect")):
            argv.append("--" + flag_name.replace("_", "-"))
    return {
        "steps": [JobStep("Run distributed QC orchestration", argv=argv, cwd=context.repo_root)],
        "reports": [
            report_item("Distributed QC report", output_dir / "index.html", primary=True),
            report_item("Distributed QC summary", output_dir / "distributed_summary.json"),
        ],
    }


def inner_bridge_pipeline_reports(output_root):
    output_root = Path(output_root)
    return [
        report_item("1. Inner capture quality: small/large calib board", output_root / "quality_report" / "index.html", primary=True),
        report_item("2. Inner solve 3D viewer", output_root / "reports" / "interactive_inner_viewer" / "index.html"),
        report_item("5. Bridge/all-rig 3D viewer", output_root / "combined_studio_rig_viewer_v1" / "index.html"),
        report_item("Fast inner/bridge final report", output_root / "final_report" / "index.html"),
        report_item("Latest summary.json", output_root / "summary.json"),
    ]


def outer_tower_pipeline_reports(output_root):
    output_root = Path(output_root)
    return [
        report_item("3. Outer capture quality: whole tower AprilTag", output_root / "quality_report" / "index.html", primary=True),
        report_item("4. Outer solve 3D viewer", output_root / "viewer" / "index.html"),
        report_item("4. Outer frame-face refine report", output_root / "frame_face_refine_wide200_then_gate6" / "index.html"),
        report_item("Outer tower final report", output_root / "final_report" / "index.html"),
        report_item("Latest summary.json", output_root / "summary.json"),
    ]


def studio_pipeline_reports(output_root):
    output_root = Path(output_root)
    return [
        report_item("Studio pipeline index", output_root / "index.html", primary=True),
        report_item("4. Outer frame-face report", output_root / "outer_tower" / "frame_face_refine_wide200_then_gate6" / "index.html"),
        report_item("5. Unified 32-camera viewer", output_root / "inner_bridge" / "combined_studio_rig_viewer_v1" / "index.html"),
        report_item("Unified studio_32_cameras.yaml", output_root / "calibration_artifacts" / "studio_32_cameras_current" / "studio_32_cameras.yaml"),
        report_item("Latest summary.json", output_root / "summary.json"),
    ]


def pipeline_output_root(params, param_name, default_latest, job_dir, preview_name):
    requested = string_param(params, param_name, default_latest)
    pipeline_dry_run = bool_param(params, "pipeline_dry_run", True)
    requested_path = Path(requested or default_latest).expanduser().resolve(strict=False)
    default_path = Path(default_latest).expanduser().resolve(strict=False)
    if pipeline_dry_run and requested_path == default_path:
        return str(job_dir / preview_name)
    return requested or str(job_dir / preview_name)


def build_studio_calibration_pipeline_plan(params, context, job_dir):
    script = as_path(
        string_param(params, "script", "scripts/calib/run_studio_calibration_pipeline.py", required=True),
        context.repo_root,
    )
    output_root = pipeline_output_root(
        params, "output_root", DEFAULT_STUDIO_PIPELINE_OUTPUT, job_dir, "studio_pipeline")
    outer_only = bool_param(params, "outer_only", False)
    bridge_only = bool_param(params, "bridge_only", False)
    if outer_only and bridge_only:
        raise ValueError("outer_only and bridge_only are mutually exclusive.")
    argv = [
        context.python_bin,
        str(script),
        "--whole-data-root",
        string_param(params, "whole_data_root", DEFAULT_T0_WHOLE_ROOT, required=True),
        "--inner-data-root",
        string_param(params, "inner_data_root", DEFAULT_T0_STAGE_ROOT, required=True),
        "--output-root",
        output_root,
        "--run-tag",
        string_param(params, "run_tag", "latest") or "latest",
        "--outer-preset",
        string_param(params, "outer_preset", "wide200_then_gate6") or "wide200_then_gate6",
        "--outer-frame-face-prior-pose-yaml",
        string_param(params, "outer_frame_face_prior_pose_yaml", DEFAULT_CURRENT_OUTER_POSE_YAML, required=True),
        "--outer-frame-face-intrinsics-dir",
        string_param(params, "outer_frame_face_intrinsics_dir", DEFAULT_CURRENT_OUTER_INTRINSICS_DIR, required=True),
    ]
    if outer_only:
        argv.append("--outer-only")
    if bridge_only:
        argv.append("--bridge-only")
        argv.extend([
            "--outer-final-pose-yaml",
            string_param(params, "outer_frame_face_prior_pose_yaml", DEFAULT_CURRENT_OUTER_POSE_YAML, required=True),
            "--outer-final-intrinsics-dir",
            string_param(params, "outer_frame_face_intrinsics_dir", DEFAULT_CURRENT_OUTER_INTRINSICS_DIR, required=True),
        ])
    if bool_param(params, "run_large_inner_init", False):
        argv.append("--run-large-inner-init")
    if bool_param(params, "run_small_quality", True):
        argv.append("--run-small-quality")
    if bool_param(params, "publish_current", False):
        argv.append("--publish-current")
    if bool_param(params, "force_pipeline_outputs", True):
        argv.append("--force")
    if bool_param(params, "pipeline_dry_run", True):
        argv.append("--dry-run")
    return {
        "steps": [JobStep("Run studio 32-camera production pipeline", argv=argv, cwd=context.repo_root)],
        "reports": studio_pipeline_reports(output_root),
    }


def build_inner_bridge_pipeline_plan(params, context, job_dir):
    script = as_path(
        string_param(params, "script", "scripts/calib/run_inner_bridge_recalib_pipeline.py", required=True),
        context.repo_root,
    )
    data_root = string_param(params, "data_root", DEFAULT_T0_STAGE_ROOT, required=True)
    default_output_root = string_param(
        params, "_default_output_root", DEFAULT_FAST_INNER_BRIDGE_OUTPUT)
    output_root = pipeline_output_root(
        params, "output_root", default_output_root, job_dir, "fast_inner_bridge")
    small_marker = string_param(params, "small_marker", "small_marker_inner8", required=True)
    large_marker = string_param(params, "large_marker", "large_marker_bridge_all32", required=True)
    large_inner_marker = string_param(params, "large_inner_marker", "large_marker_inner8", required=True)
    inner_prior = string_param(params, "inner_prior", "")
    outer_prior = string_param(params, "outer_prior", "")
    inner_refine_mode = string_param(params, "inner_refine_mode", "fixed_rig") or "fixed_rig"
    if inner_refine_mode not in {"fixed_rig", "fixed", "joint", "fixed_then_joint"}:
        raise ValueError("inner_refine_mode must be one of fixed_rig, fixed, joint, or fixed_then_joint.")
    argv = [
        context.python_bin,
        str(script),
        "--data-root",
        data_root,
        "--output-root",
        output_root,
        "--small-marker",
        small_marker,
        "--large-marker",
        large_marker,
        "--large-inner-marker",
        large_inner_marker,
        "--inner-refine-mode",
        inner_refine_mode,
        "--inner-fixed-max-ba-iterations",
        str(int_param(params, "inner_fixed_max_ba_iterations", 3, minimum=0)),
        "--inner-joint-max-ba-iterations",
        str(int_param(params, "inner_joint_max_ba_iterations", 3, minimum=0)),
        "--inner-schur-mode",
        string_param(params, "inner_schur_mode", "sparse_onthefly") or "sparse_onthefly",
        "--small-frame-stride",
        str(int_param(params, "small_frame_stride", 4, minimum=1)),
        "--large-inner-frame-stride",
        str(int_param(params, "large_inner_frame_stride", 1, minimum=1)),
        "--large-frame-stride",
        str(int_param(params, "large_frame_stride", 1, minimum=1)),
    ]
    if bool_param(params, "run_large_inner_init", True):
        argv.append("--run-large-inner-init")
    if bool_param(params, "run_small_fixed_rig_quality", inner_refine_mode == "fixed_rig"):
        argv.append("--run-small-fixed-rig-quality")
    if bool_param(params, "run_small_refine", False):
        argv.append("--run-small-refine")
    if bool_param(params, "run_large_bridge", True):
        argv.append("--run-large-bridge")
    if bool_param(params, "run_reports", True):
        argv.append("--run-reports")
    if bool_param(params, "force_pipeline_outputs", True):
        argv.append("--force")
    if inner_prior:
        argv.extend(["--inner-prior", inner_prior])
    if outer_prior:
        argv.extend(["--outer-prior", outer_prior])
    if bool_param(params, "pipeline_dry_run", True):
        argv.append("--dry-run")
    return {
        "steps": [JobStep("Run fast inner/bridge recalibration pipeline", argv=argv, cwd=context.repo_root)],
        "reports": inner_bridge_pipeline_reports(output_root),
    }


def build_outer_tower_pipeline_plan(params, context, job_dir):
    script = as_path(
        string_param(params, "script", "scripts/calib/run_outer_tower_recalib_pipeline.py", required=True),
        context.repo_root,
    )
    data_root = string_param(params, "data_root", DEFAULT_T0_STAGE_ROOT, required=True)
    default_output_root = string_param(
        params, "_default_output_root", DEFAULT_OUTER_TOWER_OUTPUT)
    output_root = pipeline_output_root(
        params, "output_root", default_output_root, job_dir, "outer_tower")
    whole_dir = string_param(params, "whole_dir", DEFAULT_T0_STAGE_ROOT + "/whole_outer_tower", required=True)
    previous_outer_rig = string_param(params, "previous_outer_rig", "")
    previous_intrinsics_dir = string_param(params, "previous_intrinsics_dir", "")
    anchor_pose_yaml = string_param(params, "anchor_pose_yaml", "")
    anchor_label_to_pose_index = string_param(params, "anchor_label_to_pose_index", "")
    frame_face_refine_preset = string_param(params, "frame_face_refine_preset", "wide200_then_gate6") or "wide200_then_gate6"
    frame_face_prior_pose_yaml = string_param(params, "frame_face_prior_pose_yaml", "")
    frame_face_intrinsics_dir = string_param(params, "frame_face_intrinsics_dir", "")
    frame_face_output_dir = string_param(params, "frame_face_output_dir", "")
    argv = [
        context.python_bin,
        str(script),
        "--data-root",
        data_root,
        "--output-root",
        output_root,
        "--whole-dir",
        whole_dir,
        "--sample-count",
        str(int_param(params, "sample_count", 32, minimum=1)),
        "--colmap-jobs",
        str(int_param(params, "colmap_jobs", 4, minimum=1)),
    ]
    if previous_outer_rig:
        argv.extend(["--previous-outer-rig", previous_outer_rig])
    if previous_intrinsics_dir:
        argv.extend(["--previous-intrinsics-dir", previous_intrinsics_dir])
    if anchor_pose_yaml:
        argv.extend(["--anchor-pose-yaml", anchor_pose_yaml])
    if anchor_label_to_pose_index:
        argv.extend(["--anchor-label-to-pose-index", anchor_label_to_pose_index])
    if bool_param(params, "run_colmap_vote", True):
        argv.append("--run-colmap-vote")
    if bool_param(params, "run_side_prior", True):
        argv.append("--run-side-prior")
    if bool_param(params, "run_tag_refine", True):
        argv.append("--run-tag-refine")
    if bool_param(params, "run_frame_face_refine", True):
        argv.extend(["--run-frame-face-refine", "--frame-face-refine-preset", frame_face_refine_preset])
    if frame_face_prior_pose_yaml:
        argv.extend(["--frame-face-prior-pose-yaml", frame_face_prior_pose_yaml])
    if frame_face_intrinsics_dir:
        argv.extend(["--frame-face-intrinsics-dir", frame_face_intrinsics_dir])
    if frame_face_output_dir:
        argv.extend(["--frame-face-output-dir", frame_face_output_dir])
    if bool_param(params, "run_quality", True):
        argv.append("--run-quality")
    if bool_param(params, "run_viewer", True):
        argv.append("--run-viewer")
    if bool_param(params, "run_reports", True):
        argv.append("--run-reports")
    if bool_param(params, "force_pipeline_outputs", True):
        argv.append("--force")
    tag_intrinsics_mode = string_param(params, "tag_intrinsics_mode", "colmap_fixed") or "colmap_fixed"
    if tag_intrinsics_mode not in {"colmap_fixed", "central_opencv"}:
        raise ValueError("tag_intrinsics_mode must be colmap_fixed or central_opencv.")
    argv.extend(["--tag-intrinsics-mode", tag_intrinsics_mode])
    tag_intrinsics_refine_mode = (
        string_param(params, "tag_intrinsics_refine_mode", "fixed") or "fixed")
    if tag_intrinsics_refine_mode not in {
        "fixed", "shared_fxfy", "per_camera_fxfy", "per_camera_fxfycxcy",
    }:
        raise ValueError(
            "tag_intrinsics_refine_mode must be fixed, shared_fxfy, "
            "per_camera_fxfy, or per_camera_fxfycxcy.")
    argv.extend(["--tag-intrinsics-refine-mode", tag_intrinsics_refine_mode])
    argv.extend([
        "--tag-intrinsics-focal-sigma-frac",
        str(float_param(params, "tag_intrinsics_focal_sigma_frac", 0.01, minimum=0.0)),
        "--tag-intrinsics-max-focal-step-frac",
        str(float_param(params, "tag_intrinsics_max_focal_step_frac", 0.002, minimum=0.0)),
        "--tag-intrinsics-block-iterations",
        str(int_param(params, "tag_intrinsics_block_iterations", 4, minimum=0)),
        "--tag-min-camera-observations-for-use",
        str(int_param(params, "tag_min_camera_observations_for_use", 16, minimum=0)),
        "--tag-min-camera-observations-for-delta",
        str(int_param(params, "tag_min_camera_observations_for_delta", 10, minimum=0)),
        "--tag-post-refine-observation-residual-gate-px",
        str(float_param(params, "tag_post_refine_observation_residual_gate_px", 190.0, minimum=0.0)),
        "--tag-post-refine-outer-iterations",
        str(int_param(params, "tag_post_refine_outer_iterations", 2, minimum=0)),
    ])
    if bool_param(params, "pipeline_dry_run", True):
        argv.append("--dry-run")
    return {
        "steps": [JobStep("Run outer tower recalibration pipeline", argv=argv, cwd=context.repo_root)],
        "reports": outer_tower_pipeline_reports(output_root),
    }


def report_steps(params, context, output_dir, dataset_path, state_dir, manifest, rig_state_dir=None):
    rig_state = Path(rig_state_dir) if rig_state_dir else Path(state_dir)
    inner_report_dir = Path(output_dir) / "inner_reprojection"
    rig_report_dir = Path(output_dir) / "rig_extrinsics"
    viewer_dir = Path(output_dir) / "interactive_report"
    frames_dir = viewer_dir / "camera_frames"
    title = string_param(params, "title", "Inner Camera Calibration Report")
    sparse_json = string_param(params, "sparse_point_cloud_json", "")
    camera_dirs_file = string_param(params, "camera_image_dirs_file", "")
    threejs_assets_dir = string_param(params, "threejs_assets_dir", DEFAULT_THREEJS_ASSETS)

    steps = [
        JobStep(
            "Generate inner reprojection report",
            argv=[
                context.python_bin,
                context.script("scripts/calib/generate_inner_calibration_report.py"),
                "--dataset",
                str(dataset_path),
                "--state-dir",
                str(state_dir),
                "--output-dir",
                str(inner_report_dir),
                "--manifest",
                str(manifest),
            ],
            cwd=context.repo_root,
        ),
        JobStep(
            "Generate rig extrinsics report",
            argv=[
                context.python_bin,
                context.script("scripts/calib/generate_rig_extrinsics_report.py"),
                "--state-dir",
                str(rig_state),
                "--output-dir",
                str(rig_report_dir),
            ],
            cwd=context.repo_root,
        ),
    ]
    if camera_dirs_file:
        steps.append(
            JobStep(
                "Prepare first-frame images for interactive viewer",
                internal="prepare_camera_frames",
                kwargs={
                    "image_dirs_file": str(camera_dirs_file),
                    "output_dir": str(frames_dir),
                },
            )
        )
    steps.append(
        JobStep(
            "Copy Three.js viewer assets",
            internal="prepare_threejs_assets",
            kwargs={
                "output_dir": str(viewer_dir),
                "candidates": [
                    threejs_assets_dir,
                    DEFAULT_THREEJS_ASSETS,
                    str(context.repo_root / "studio/exp/inner_marker_2026_05_26_processing/interactive_rig_viewer_v1"),
                ],
            },
        )
    )

    viewer_argv = [
        context.python_bin,
        context.script("scripts/calib/generate_threejs_rig_viewer.py"),
        "--pose-yaml",
        str(rig_report_dir / "camera_tr_camera0.yaml"),
        "--metrics-tsv",
        str(rig_report_dir / "camera_tr_camera0.tsv"),
        "--output-dir",
        str(viewer_dir),
        "--title",
        title,
        "--reprojection-report",
        "refined=" + str(inner_report_dir),
    ]
    if camera_dirs_file:
        viewer_argv.extend(["--camera-image-dir", str(frames_dir)])
    if sparse_json:
        viewer_argv.extend(["--sparse-point-cloud-json", sparse_json])
    steps.append(
        JobStep(
            "Generate interactive Three.js report",
            argv=viewer_argv,
            cwd=context.repo_root,
        )
    )
    return steps, [
        report_item("Interactive report", viewer_dir / "index.html", primary=True),
        report_item("Inner reprojection report", inner_report_dir / "index.html"),
        report_item("Rig extrinsics report", rig_report_dir / "index.html"),
    ]


def build_report_only_plan(params, context, job_dir):
    dataset_path = string_param(
        params,
        "dataset_path",
        DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/features_pattern3_grid4_v1.bin",
        required=True,
    )
    state_dir = string_param(params, "state_dir", DEFAULT_INNER_STATE, required=True)
    manifest = string_param(
        params,
        "manifest",
        DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/manifest.tsv",
    )
    rig_state_dir = string_param(params, "rig_state_dir", "") or state_dir
    output_value = string_param(params, "output_dir", "")
    output_dir = as_path(output_value, context.repo_root) if output_value else job_dir / "report_only"
    steps, reports = report_steps(
        params,
        context,
        output_dir,
        dataset_path,
        state_dir,
        manifest,
        rig_state_dir=rig_state_dir,
    )
    return {"steps": steps, "reports": reports}


def build_inner_warm_start_plan(params, context, job_dir):
    output_value = string_param(params, "output_root", "")
    output_root = as_path(output_value, context.repo_root) if output_value else job_dir / "inner_warm_start_refine"
    binary = string_param(params, "binary", DEFAULT_T0_BINARY, required=True)
    small_dirs = string_param(
        params,
        "small_image_dirs_file",
        DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/image_directories.txt",
        required=True,
    )
    pattern_file = string_param(params, "small_pattern_file", SMALL_MARKER_PATTERN, required=True)
    manifest = string_param(
        params,
        "manifest",
        DEFAULT_T0_STAGE_ROOT + "/small_marker_inner8/manifest.tsv",
        required=True,
    )
    warm_start_state = string_param(params, "warm_start_state", DEFAULT_INNER_STATE, required=True)
    feature_jobs = int_param(params, "feature_jobs", 8, minimum=1)
    grid_stride = int_param(params, "grid_stride", 4, minimum=1)
    min_features = int_param(params, "min_features_per_camera_view", 20, minimum=0)
    max_ba_iterations = int_param(params, "max_ba_iterations", 6, minimum=0)

    full_dataset = output_root / "features_parallel_pattern3_panel.bin"
    shard_dir = output_root / "parallel_shards_pattern3_panel"
    grid_dataset = output_root / f"features_pattern3_grid{grid_stride}_panel.bin"
    refine_dir = output_root / "fixed_intrinsic_small_grid_warm_start_panel"
    reports_dir = output_root / "reports"

    feature_argv = [
        context.python_bin,
        context.script("scripts/calib/parallel_extract_features.py"),
        "--binary",
        binary,
        "--repo-root",
        str(context.repo_root),
        "--image-directories-file",
        small_dirs,
        "--pattern-files",
        pattern_file,
        "--output-dataset",
        str(full_dataset),
        "--work-dir",
        str(shard_dir),
        "--jobs",
        str(feature_jobs),
    ]
    if bool_param(params, "overwrite_features", False):
        feature_argv.append("--overwrite")

    steps = [
        JobStep("Extract small-marker features in parallel", argv=feature_argv, cwd=context.repo_root),
        JobStep(
            "Create grid-subsampled small-marker dataset",
            argv=[
                binary,
                "--subsample_dataset",
                "--dataset_files",
                str(full_dataset),
                "--dataset_output_path",
                str(grid_dataset),
                "--subsample_pattern_grid_stride",
                str(grid_stride),
                "--subsample_min_features_per_camera_view",
                str(min_features),
            ],
            cwd=context.repo_root,
            env=command_env(),
        ),
        JobStep(
            "Refine inner rig from warm-start state",
            argv=[
                binary,
                "--dataset_files",
                str(grid_dataset),
                "--state_directory",
                warm_start_state,
                "--output_directory",
                str(refine_dir),
                "--localize_only",
                "--num_pyramid_levels",
                "1",
                "--outlier_removal_factor",
                "0",
                "--max_ba_iterations",
                str(max_ba_iterations),
                "--skip_calibration_report",
            ],
            cwd=context.repo_root,
            env=command_env(),
        ),
    ]

    report_params = dict(params)
    report_params.setdefault("title", "Inner Warm-Start Calibration Report")
    report_steps_list, reports = report_steps(
        report_params,
        context,
        reports_dir,
        grid_dataset,
        refine_dir,
        manifest,
        rig_state_dir=refine_dir,
    )
    steps.extend(report_steps_list)
    return {"steps": steps, "reports": reports}


def merge_operation_defaults(params, defaults):
    merged = dict(defaults)
    merged.update(params)
    return merged


def build_whole_outer_cage_operation_plan(params, context, job_dir):
    defaults = {
        "_default_output_root": DEFAULT_WHOLE_OPERATION_OUTPUT,
        "data_root": DEFAULT_T0_WHOLE_ROOT,
        "whole_dir": DEFAULT_T0_WHOLE_OUTER24_DIR,
        "output_root": DEFAULT_WHOLE_OPERATION_OUTPUT,
        "run_colmap_vote": False,
        "run_side_prior": False,
        "run_tag_refine": False,
        "run_frame_face_refine": True,
        "frame_face_refine_preset": "wide200_then_gate6",
        "frame_face_prior_pose_yaml": DEFAULT_CURRENT_OUTER_POSE_YAML,
        "frame_face_intrinsics_dir": DEFAULT_CURRENT_OUTER_INTRINSICS_DIR,
        "run_quality": True,
        "run_viewer": False,
        "run_reports": True,
        "force_pipeline_outputs": True,
        "pipeline_dry_run": True,
    }
    return build_outer_tower_pipeline_plan(
        merge_operation_defaults(params, defaults), context, job_dir)


def build_large_marker_bridge_operation_plan(params, context, job_dir):
    defaults = {
        "_default_output_root": DEFAULT_LARGE_MARKER_OPERATION_OUTPUT,
        "data_root": DEFAULT_T0_STAGE_ROOT,
        "output_root": DEFAULT_LARGE_MARKER_OPERATION_OUTPUT,
        "small_marker": "small_marker_inner8",
        "large_inner_marker": "large_marker_inner8",
        "large_marker": "large_marker_bridge_all32",
        "inner_refine_mode": "fixed_rig",
        "small_frame_stride": 4,
        "large_inner_frame_stride": 1,
        "large_frame_stride": 1,
        "run_large_inner_init": True,
        "run_small_fixed_rig_quality": False,
        "run_small_refine": False,
        "run_large_bridge": True,
        "run_reports": True,
        "force_pipeline_outputs": True,
        "pipeline_dry_run": True,
    }
    return build_inner_bridge_pipeline_plan(
        merge_operation_defaults(params, defaults), context, job_dir)


def build_small_marker_inner_operation_plan(params, context, job_dir):
    defaults = {
        "_default_output_root": DEFAULT_SMALL_MARKER_OPERATION_OUTPUT,
        "data_root": DEFAULT_T0_STAGE_ROOT,
        "output_root": DEFAULT_SMALL_MARKER_OPERATION_OUTPUT,
        "small_marker": "small_marker_inner8",
        "large_inner_marker": "large_marker_inner8",
        "large_marker": "large_marker_bridge_all32",
        "inner_refine_mode": "fixed_then_joint",
        "small_frame_stride": 4,
        "large_inner_frame_stride": 1,
        "large_frame_stride": 1,
        "run_large_inner_init": False,
        "run_small_fixed_rig_quality": True,
        "run_small_refine": True,
        "run_large_bridge": False,
        "run_reports": True,
        "force_pipeline_outputs": True,
        "pipeline_dry_run": True,
    }
    return build_inner_bridge_pipeline_plan(
        merge_operation_defaults(params, defaults), context, job_dir)


PLAN_BUILDERS = {
    "run_studio_calibration_pipeline": build_studio_calibration_pipeline_plan,
    "operate_whole_outer_cage": build_whole_outer_cage_operation_plan,
    "operate_large_marker_bridge": build_large_marker_bridge_operation_plan,
    "operate_small_marker_inner": build_small_marker_inner_operation_plan,
    "run_inner_bridge_recalib_pipeline": build_inner_bridge_pipeline_plan,
    "run_outer_tower_recalib_pipeline": build_outer_tower_pipeline_plan,
    "stage_data": build_stage_data_plan,
    "distributed_qc": build_distributed_qc_plan,
    "inner_warm_start_refine": build_inner_warm_start_plan,
    "report_only": build_report_only_plan,
}


def prepare_camera_frames(kwargs, log):
    image_dirs_file = Path(kwargs["image_dirs_file"]).expanduser()
    output_dir = Path(kwargs["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    text = image_dirs_file.read_text(encoding="utf-8-sig").strip()
    image_dirs = [Path(item.strip()).expanduser() for item in text.replace("\n", ",").split(",") if item.strip()]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    copied = 0
    for camera_index, image_dir in enumerate(image_dirs):
        if not image_dir.is_dir():
            log.write(f"missing image dir for camera {camera_index}: {image_dir}\n")
            continue
        images = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in extensions]
        if not images:
            log.write(f"no images for camera {camera_index}: {image_dir}\n")
            continue
        src = images[0]
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in image_dir.name)
        dst = output_dir / f"cam{camera_index:02d}_{safe_name}_0000{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied += 1
        log.write(f"camera {camera_index:02d}: {src} -> {dst}\n")
    log.write(f"prepared {copied} camera frame images in {output_dir}\n")


def prepare_threejs_assets(kwargs, log):
    output_dir = Path(kwargs["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = ["three.min.js", "OrbitControls.js", "TransformControls.js"]
    candidates = [Path(item).expanduser() for item in kwargs.get("candidates", []) if item]
    source_dir = None
    for candidate in candidates:
        if all((candidate / name).is_file() for name in required):
            source_dir = candidate
            break
    if source_dir is None:
        checked = ", ".join(str(path) for path in candidates) or "(none)"
        raise FileNotFoundError(f"Could not find Three.js viewer assets. Checked: {checked}")
    for name in required:
        src = source_dir / name
        dst = output_dir / name
        shutil.copy2(src, dst)
        log.write(f"{src} -> {dst}\n")


INTERNAL_STEPS = {
    "prepare_camera_frames": prepare_camera_frames,
    "prepare_threejs_assets": prepare_threejs_assets,
}


class JobManager:
    def __init__(self, repo_root, runs_root, python_bin=None):
        self.repo_root = Path(repo_root).resolve()
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.python_bin = str(python_bin or default_python_bin())
        self.context = PlanContext(self.repo_root, self.python_bin)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.cancel_events = {}
        self.processes = {}
        self._load_existing_jobs()

    def _load_existing_jobs(self):
        for path in sorted(self.runs_root.glob("*/job.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            job_id = job.get("id")
            if job_id:
                self.jobs[job_id] = job

    def start_job(self, mode, params, dry_run=False):
        if mode not in PLAN_BUILDERS:
            raise ValueError(f"Unknown run mode: {mode}")
        params = dict(params or {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{timestamp}_{mode}_{uuid.uuid4().hex[:8]}"
        run_dir = self.runs_root / job_id
        run_dir.mkdir(parents=True, exist_ok=False)
        plan = PLAN_BUILDERS[mode](params, self.context, run_dir)
        job = {
            "id": job_id,
            "mode": mode,
            "mode_title": MODE_DEFINITIONS[mode]["title"],
            "params": params,
            "dry_run": bool(dry_run),
            "status": "pending",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "started_at": "",
            "finished_at": "",
            "run_dir": str(run_dir),
            "log_path": str(run_dir / "run.log"),
            "steps": [step.to_json() for step in plan["steps"]],
            "reports": plan.get("reports", []),
            "final_report_url": "",
            "_plan_steps": plan["steps"],
        }
        Path(job["log_path"]).write_text("", encoding="utf-8")
        event = threading.Event()
        with self.lock:
            self.jobs[job_id] = job
            self.cancel_events[job_id] = event
            self._save_job(job)
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return self.get_job(job_id)

    def _save_job(self, job):
        public_job = {key: value for key, value in job.items() if not key.startswith("_")}
        path = Path(job["run_dir"]) / "job.json"
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(public_job, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def _set_job_status(self, job, status):
        job["status"] = status
        job["updated_at"] = utc_now()
        if status in ("completed", "failed", "canceled"):
            job["finished_at"] = utc_now()
            self._refresh_report_status(job)
        self._save_job(job)

    def _refresh_report_status(self, job):
        reports = []
        final_url = ""
        for index, item in enumerate(job.get("reports", [])):
            refreshed = dict(item)
            path = Path(refreshed["path"])
            refreshed["exists"] = path.exists()
            refreshed["url"] = public_url_for_path(path)
            artifact_name = path.name or "index.html"
            refreshed["panel_url"] = (
                f"/api/jobs/{job['id']}/artifact/{index}/{quote(artifact_name)}"
            )
            if refreshed["exists"] and (refreshed.get("primary") or not final_url):
                final_url = refreshed["url"]
            reports.append(refreshed)
        job["reports"] = reports
        job["final_report_url"] = final_url

    def _append_log(self, job, text):
        with Path(job["log_path"]).open("a", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()

    def _run_job(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            job["started_at"] = utc_now()
            self._set_job_status(job, "running")
        self._append_log(job, f"[{utc_now()}] job {job_id} started: {job['mode']}\n")
        if job["dry_run"]:
            self._append_log(job, "DRY RUN: commands are recorded but not executed.\n")

        try:
            for index, step in enumerate(job["_plan_steps"]):
                if self.cancel_events[job_id].is_set():
                    raise KeyboardInterrupt
                with self.lock:
                    job["steps"][index]["status"] = "running"
                    job["steps"][index]["started_at"] = utc_now()
                    self._save_job(job)
                self._append_log(job, f"\n[{utc_now()}] step {index + 1}: {step.name}\n")
                rc = self._run_step(job_id, job, step)
                with self.lock:
                    job["steps"][index]["returncode"] = rc
                    job["steps"][index]["finished_at"] = utc_now()
                    job["steps"][index]["status"] = "completed" if rc == 0 else "failed"
                    self._save_job(job)
                if rc != 0:
                    raise RuntimeError(f"Step failed with return code {rc}: {step.name}")
            with self.lock:
                self._set_job_status(job, "completed")
            self._append_log(job, f"\n[{utc_now()}] job completed\n")
        except KeyboardInterrupt:
            with self.lock:
                self._set_job_status(job, "canceled")
            self._append_log(job, f"\n[{utc_now()}] job canceled\n")
        except Exception as exc:
            with self.lock:
                job["error"] = str(exc)
                self._set_job_status(job, "failed")
            self._append_log(job, f"\n[{utc_now()}] ERROR: {exc}\n")
        finally:
            with self.lock:
                self.processes.pop(job_id, None)
                self._save_job(job)

    def _run_step(self, job_id, job, step):
        if job["dry_run"]:
            if step.argv:
                self._append_log(job, "DRY RUN command: " + shlex.join(str(part) for part in step.argv) + "\n")
            else:
                self._append_log(job, f"DRY RUN internal step: {step.internal} {json.dumps(step.kwargs, ensure_ascii=False)}\n")
            return 0

        if step.internal:
            func = INTERNAL_STEPS.get(step.internal)
            if func is None:
                raise RuntimeError(f"Unknown internal step: {step.internal}")
            with Path(job["log_path"]).open("a", encoding="utf-8") as log:
                func(step.kwargs, log)
            return 0

        env = os.environ.copy()
        env.update(step.env)
        cwd = step.cwd or str(self.repo_root)
        with Path(job["log_path"]).open("a", encoding="utf-8") as log:
            log.write("command: " + shlex.join(str(part) for part in step.argv) + "\n")
            log.write("cwd: " + str(cwd) + "\n\n")
            log.flush()
            proc = subprocess.Popen(
                step.argv,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with self.lock:
                self.processes[job_id] = proc
            while True:
                rc = proc.poll()
                if rc is not None:
                    return rc
                if self.cancel_events[job_id].is_set():
                    self._terminate_process(proc)
                    return 130
                time.sleep(0.25)

    def _terminate_process(self, proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()

    def get_job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            return json.loads(json.dumps({key: value for key, value in job.items() if not key.startswith("_")}))

    def list_jobs(self):
        with self.lock:
            jobs = [self.get_job(job_id) for job_id in sorted(self.jobs.keys(), reverse=True)]
        return [job for job in jobs if job]

    def cancel_job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            if job["status"] not in ("pending", "running"):
                return self.get_job(job_id)
            self.cancel_events[job_id].set()
            proc = self.processes.get(job_id)
            if proc:
                self._terminate_process(proc)
            return self.get_job(job_id)

    def log_chunk(self, job_id, offset=0):
        job = self.get_job(job_id)
        if not job:
            return None
        path = Path(job["log_path"])
        if not path.exists():
            return {"offset": 0, "next_offset": 0, "text": "", "size": 0}
        size = path.stat().st_size
        offset = max(0, min(int(offset), size))
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read()
        return {
            "offset": offset,
            "next_offset": offset + len(data),
            "size": size,
            "text": data.decode("utf-8", errors="replace"),
        }

    def wait_for_job(self, job_id, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get_job(job_id)
            if job and job["status"] not in ("pending", "running"):
                return job
            time.sleep(0.05)
        raise TimeoutError(job_id)


class PanelRequestHandler(SimpleHTTPRequestHandler):
    manager = None
    static_root = None

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json_head(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/modes":
            self.send_json({"modes": MODE_DEFINITIONS})
            return
        if path == "/api/jobs":
            self.send_json({"jobs": self.manager.list_jobs()})
            return
        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 5 and parts[3] == "artifact":
                self.serve_job_artifact(parts[2], parts[4], "/".join(parts[5:]) or "index.html")
                return
            if len(parts) == 3:
                job = self.manager.get_job(parts[2])
                if not job:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Job not found")
                    return
                self.send_json(job)
                return
            if len(parts) == 4 and parts[3] == "log":
                query = parse_qs(parsed.query)
                offset = int(query.get("offset", ["0"])[0] or 0)
                chunk = self.manager.log_chunk(parts[2], offset)
                if chunk is None:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Job not found")
                    return
                self.send_json(chunk)
                return
        self.serve_static(path)

    def do_HEAD(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/modes":
            self.send_json_head({"modes": MODE_DEFINITIONS})
            return
        if path == "/api/jobs":
            self.send_json_head({"jobs": self.manager.list_jobs()})
            return
        self.serve_static(path)

    def serve_job_artifact(self, job_id, report_index_text, rel_path):
        job = self.manager.get_job(job_id)
        if not job:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            report_index = int(report_index_text)
            report = job.get("reports", [])[report_index]
        except (ValueError, IndexError):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        report_path = Path(report["path"]).resolve()
        base_dir = report_path.parent
        if ".." in Path(rel_path).parts:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        path = (base_dir / unquote(rel_path)).resolve()
        try:
            path.relative_to(base_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}")
            return

        if path == "/api/jobs":
            try:
                job = self.manager.start_job(
                    payload.get("mode", ""),
                    payload.get("params", {}),
                    dry_run=bool(payload.get("dry_run", False)),
                )
            except Exception as exc:
                self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self.send_json(job, status=HTTPStatus.CREATED)
            return

        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            parts = path.strip("/").split("/")
            if len(parts) == 4:
                job = self.manager.cancel_job(parts[2])
                if not job:
                    self.send_error_json(HTTPStatus.NOT_FOUND, "Job not found")
                    return
                self.send_json(job)
                return
        self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def serve_static(self, request_path):
        if request_path in ("", "/"):
            rel = "index.html"
        else:
            rel = unquote(request_path.lstrip("/"))
        if ".." in Path(rel).parts:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        path = (self.static_root / rel).resolve()
        try:
            path.relative_to(self.static_root.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=os.environ.get("CALIB_PANEL_REPO_ROOT", DEFAULT_T0_REPO))
    parser.add_argument("--runs-root", default="")
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main():
    args = build_arg_parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve() if args.runs_root else repo_root / "studio/exp/calibration_panel_runs"
    static_root = Path(__file__).resolve().parent / "panel_static"
    manager = JobManager(repo_root=repo_root, runs_root=runs_root, python_bin=args.python_bin)
    PanelRequestHandler.manager = manager
    PanelRequestHandler.static_root = static_root
    server = ThreadingHTTPServer((args.host, args.port), PanelRequestHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Calibration panel listening on {url}")
    print(f"repo_root={repo_root}")
    print(f"runs_root={runs_root}")
    print(f"python_bin={args.python_bin}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down calibration panel.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
