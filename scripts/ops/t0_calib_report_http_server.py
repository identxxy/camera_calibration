#!/usr/bin/env python3
"""Serve t0 calibration reports from calib_data over HTTP."""

from __future__ import annotations

import argparse
from datetime import datetime
import html
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import parse_qs, quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.calib.calibration_panel_server import JobManager, MODE_DEFINITIONS
except Exception as exc:  # pragma: no cover - exercised by deployment health checks.
    JobManager = None
    MODE_DEFINITIONS = {}
    PANEL_IMPORT_ERROR = exc
else:
    PANEL_IMPORT_ERROR = None


DEFAULT_ROOT = "/home/ubuntu/calib_data"
DEFAULT_REPORT_BASE_URL = "http://192.168.2.0:9899"
DEFAULT_REPO_ROOT = "/home/ubuntu/camera_calibration"
DEFAULT_RUNS_ROOT = "/home/ubuntu/calib_data/panel_runs"
DEFAULT_PYTHON_BIN = "/home/ubuntu/miniconda3/bin/python"
DEFAULT_LANGUAGE = "en"
CURRENT_ENTRY_REL = "current_calibration/index.html"
STAGE_ROOT = "calib_2026_05_26_jpg_v3"
PIPELINE_ROOT = f"{STAGE_ROOT}/recalib_pipelines"
CURRENT_RUN_ROOT = "studio_calibration_runs/recalib_20260610_black_tile_wide200_pipeline_v2"
FAST_INNER_BRIDGE_LATEST = f"{CURRENT_RUN_ROOT}/inner_bridge"
OUTER_TOWER_LATEST = f"{CURRENT_RUN_ROOT}/outer_tower/frame_face_refine_wide200_then_gate6"
CURRENT_WHOLE_ROOT = "calib_2026_05_31_fullres_probe_v1"
CURRENT_WHOLE_OUTER24 = f"{CURRENT_WHOLE_ROOT}/whole_outer24_filtered_min4_fullres_min4cam"
CURRENT_WHOLE_ALL32 = CURRENT_WHOLE_OUTER24
CURRENT_OUTER_CANDIDATE = OUTER_TOWER_LATEST
FINAL_STUDIO32_YAML = "current_calibration/artifacts/studio_32_cameras.yaml"
UNIFIED_VIEWER = "current_calibration/reports/01_3d_viewer/index.html"
INNER_CAPTURE_REPORT = "current_calibration/reports/02_inner_capture_small_marker/index.html"
INNER_INTRINSIC_REPORT = "current_calibration/reports/03_inner_intrinsics_small_marker/index.html"
INNER_EXTRINSIC_REPORT = "current_calibration/reports/04_inner_extrinsics_small_marker/index.html"
OUTER_CAPTURE_REPORT = "current_calibration/reports/05_outer_capture_outer_large_marker_whole/index.html"
OUTER_INTRINSIC_REPORT = "current_calibration/reports/06_outer_intrinsics_outer_large_marker/index.html"
OUTER_EXTRINSIC_REPORT = "current_calibration/reports/07_outer_extrinsics_whole/index.html"
BRIDGE_RESULT_REPORT = "current_calibration/reports/09_bridge_result_large_marker/index.html"
INNER_BRIDGE_QUALITY_REPORT = f"{CURRENT_RUN_ROOT}/inner_bridge/quality_report/index.html"
STUDIO32_YAML = (
    f"{CURRENT_RUN_ROOT}/calibration_artifacts/"
    "studio_32_cameras_current/studio_32_cameras.yaml"
)
STABLE_INNER_VIEWER = (
    f"{STAGE_ROOT}/final_inner8_calibration_v1/reports/interactive_rig_viewer_v1/index.html"
)


EXCLUDED_REPORT_PATHS = {
    # This fast-pipeline artifact has camera poses but no first-frame textures or
    # sparse context, so it renders as an effectively blank viewer.
    f"{FAST_INNER_BRIDGE_LATEST}/reports/interactive_inner_viewer/index.html",
}


REPORT_GROUPS = [
    {
        "title": "Final Calibration Artifact",
        "subtitle": "machine-readable 32-camera YAML",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": (
            "最终 32-camera calibration artifact。下游重建、SLAM、3DGS 和 viewer "
            "应消费这个 YAML，而不是 dated scratch outputs。"
        ),
        "items": [
            {
                "label": "studio_32_cameras.yaml",
                "path": FINAL_STUDIO32_YAML,
                "kind": "final YAML",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
    {
        "title": "Overall Viewer",
        "subtitle": "single unified 32-camera viewer",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": (
            "统一查看 24+8 camera rig。viewer 内包含 camera filters、dataset coverage、"
            "correspondence loading、intrinsic residuals 和 final dataset/extrinsic residuals。"
        ),
        "items": [
            {
                "label": "Overall 3D Viewer",
                "path": UNIFIED_VIEWER,
                "kind": "3D viewer",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
    {
        "title": "Inner Capture / QC",
        "subtitle": "small-marker inner8 data quality",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": (
            "内圈采集质量入口。这里只看同步、尾帧裁剪、掉帧排除、角点覆盖和可用相机集合。"
        ),
        "items": [
            {
                "label": "Inner capture report",
                "path": INNER_CAPTURE_REPORT,
                "kind": "data collection quality",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
    {
        "title": "Inner Intrinsic Result",
        "subtitle": "inner8 feature coverage and residuals",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": (
            "Inner8 intrinsic feature accumulation、reprojection residual 和 per-camera intrinsic quality。"
        ),
        "items": [
            {
                "label": "Inner intrinsic report",
                "path": INNER_INTRINSIC_REPORT,
                "kind": "intrinsic report",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
    {
        "title": "Inner Extrinsic Result",
        "subtitle": "inner8 rig layout and consistency",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": "Inner8 extrinsic layout、relative pose sanity checks 和 final inner consistency。",
        "items": [
            {
                "label": "Inner extrinsic report",
                "path": INNER_EXTRINSIC_REPORT,
                "kind": "extrinsic report",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
    {
        "title": "Outer Capture / QC",
        "subtitle": "outer-large-marker + whole capture QC",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": (
            "Outer-large-marker intrinsic capture 和 whole/tower extrinsic capture 的统一采集质量报告。"
        ),
        "items": [
            {
                "label": "Outer capture report",
                "path": OUTER_CAPTURE_REPORT,
                "kind": "data collection quality",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
    {
        "title": "Outer Intrinsic Result",
        "subtitle": "outer24 large-marker intrinsics",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": "Outer24 large-marker feature accumulation、residuals 和 per-camera intrinsic quality。",
        "items": [
            {
                "label": "Outer intrinsic report",
                "path": OUTER_INTRINSIC_REPORT,
                "kind": "intrinsic report",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
    {
        "title": "Outer Extrinsic Result",
        "subtitle": "whole/tower outer24 extrinsics",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": "Whole/tower outer24 extrinsic refinement residuals and accepted observation summary。",
        "items": [
            {
                "label": "Outer extrinsic report",
                "path": OUTER_EXTRINSIC_REPORT,
                "kind": "extrinsic report",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
    {
        "title": "Bridge Result",
        "subtitle": "large-marker all-camera bridge",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "",
        "description": (
            "Large-marker all-camera bridge result、inner/outer consistency 和 final bridge quality gates。"
        ),
        "items": [
            {
                "label": "Bridge result report",
                "path": BRIDGE_RESULT_REPORT,
                "kind": "bridge report",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [],
    },
]


QUICK_ACTIONS = [
    {
        "slug": "full-dry-run",
        "title": "Dry-run Full Pipeline",
        "title_zh": "预演完整流程",
        "title_en": "Dry-run Full Pipeline",
        "mode": "run_studio_calibration_pipeline",
        "dry_run": True,
        "confirm": False,
        "description": (
            "预演完整 32 相机流程，确认将使用哪些数据路径、运行目录和报告发布路径；"
            "适合在正式处理前检查配置。"
        ),
        "description_en": (
            "Preview the full 32-camera workflow before launching a real run. "
            "Use this to check input paths, output directories, and report publication targets."
        ),
        "params": {"publish_current": True, "pipeline_dry_run": True},
    },
    {
        "slug": "full-run",
        "title": "Run Full Pipeline",
        "title_zh": "运行完整流程",
        "title_en": "Run Full Pipeline",
        "mode": "run_studio_calibration_pipeline",
        "dry_run": False,
        "confirm": True,
        "description": (
            "从当前整理好的四类采集数据重新处理完整结果：large 用于 outer intrinsic，"
            "tower 用于 outer extrinsic，bridge 用于 inner/outer 绑定，small 用于 inner intrinsic/pose refinement。"
        ),
        "description_en": (
            "Reprocess the four current capture datasets: large for outer intrinsics, "
            "tower for outer extrinsics, bridge for inner/outer alignment, and small for "
            "inner intrinsics / pose refinement."
        ),
        "params": {"publish_current": True, "pipeline_dry_run": False},
    },
    {
        "slug": "fast-bridge",
        "title": "Run Fast Bridge",
        "title_zh": "快速重跑 Bridge",
        "title_en": "Run Fast Bridge",
        "mode": "run_studio_calibration_pipeline",
        "dry_run": False,
        "confirm": True,
        "description": (
            "当外圈固定不变、只移动了内圈相机时使用。它复用当前 outer 结果，"
            "重跑 bridge 和 small 相关处理，并发布新的 32 相机结果。"
        ),
        "description_en": (
            "Use this when the fixed outer ring has not changed and only the inner cameras moved. "
            "It reuses the current outer solution, reruns the bridge and small-board "
            "processing, then publishes a new 32-camera result."
        ),
        "params": {
            "bridge_only": True,
            "publish_current": True,
            "run_small_quality": True,
            "pipeline_dry_run": False,
        },
    },
]


CAPTURE_DATASETS = [
    {
        "slug": "large",
        "name": "large",
        "name_zh": "large 大标定板",
        "target_zh": "外圈内参",
        "target_en": "outer intrinsics",
        "hardware_zh": "低密度 / 大尺寸 calib board",
        "hardware_en": "low-density large calibration board",
        "capture_zh": "让外圈 24 台相机尽量看到大标定板，并覆盖图像中心和边缘。",
        "capture_en": "Show the large board to the 24 outer cameras with broad image coverage, including center and borders.",
        "quality_zh": [
            "24 台外圈相机应尽量都有有效角点；少数缺失需要在报告里明确标出。",
            "角点应覆盖图像中心、边缘和不同尺度，不要只集中在画面中央。",
            "fx/fy、主点、畸变和 residual 不应有单台相机明显离群。",
        ],
        "quality_en": [
            "All 24 outer cameras should ideally have valid board corners; any missing cameras should be explicit in the report.",
            "Corners should cover image centers, borders, and multiple scales instead of clustering only near the center.",
            "fx/fy, principal point, distortion, and residuals should not show obvious per-camera outliers.",
        ],
        "raw_path": "D:/output/calib/outer_large_marker/<time>/<camera_id>/<sn>_<imageid>.jpg",
        "staged_path": "/home/ubuntu/calib_data/calib_2026_05_31_v3/outer_large_marker_*",
        "qc_step_slug": "outer-large-marker",
        "report": {
            "label": "Large-board outer capture / QC",
            "label_zh": "large 大标定板外圈采集 / QC",
            "label_en": "Large-board outer capture / QC",
            "path": OUTER_CAPTURE_REPORT,
            "kind": "data capture report",
        },
        "used_by": ["outer-large-marker"],
    },
    {
        "slug": "tower",
        "name": "tower",
        "name_zh": "tower 标定塔",
        "target_zh": "外圈外参",
        "target_en": "outer extrinsics",
        "hardware_zh": "AprilTag 八面塔，tag 8cm，间距 2cm",
        "hardware_en": "AprilTag octagonal tower, 8 cm tags, 2 cm gaps",
        "capture_zh": "推动标定塔在场地中移动；主要约束固定外圈 24 相机的相对 pose。",
        "capture_en": "Move the tower through the studio; this mainly constrains the relative poses of the fixed 24-camera outer cage.",
        "quality_zh": [
            "塔应覆盖场地中心和外圈相机视野交叠区域，不要只沿边缘移动。",
            "大多数外圈相机应在多帧看到足够多 physical tag corners。",
            "单帧 correspondence 在 viewer 中应形成竖直塔形，不应躺倒、散开或尺度异常。",
        ],
        "quality_en": [
            "Move the tower through the studio center and shared outer-camera viewing regions, not only along the boundary.",
            "Most outer cameras should see enough physical tag corners across multiple frames.",
            "Single-frame correspondences in the viewer should form an upright tower, not a flat, scattered, or scale-wrong shape.",
        ],
        "raw_path": "D:/output/calib/whole/<time>/<camera_id>/<sn>_<imageid>.jpg",
        "staged_path": "/home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1/whole_outer24_filtered_min4_fullres_min4cam",
        "qc_step_slug": "whole-outer-cage",
        "report": {
            "label": "Tower whole capture / QC",
            "label_zh": "tower whole 采集 / QC",
            "label_en": "Tower whole capture / QC",
            "path": OUTER_CAPTURE_REPORT,
            "kind": "data capture report",
        },
        "used_by": ["whole-outer-cage"],
    },
    {
        "slug": "bridge",
        "name": "bridge",
        "name_zh": "bridge 大标定板",
        "target_zh": "outer 外参收敛 + inner/outer bridge + inner 外参",
        "target_en": "outer pose consistency + inner/outer bridge + inner extrinsics",
        "hardware_zh": "同一块低密度 / 大尺寸 calib board，放在内外圈都能看到的位置",
        "hardware_en": "the same low-density large board, captured where both inner and outer cameras can see it",
        "capture_zh": "在桌面/中心区域移动大标定板，尽量让内圈和外圈同时有共视。",
        "capture_en": "Move the large board around the table / center region to maximize inner/outer co-visibility.",
        "quality_zh": [
            "必须同时覆盖 inner 和 outer；如果只看到一侧，这组数据不能完成 bridge。",
            "板子应在桌面/中心区域多姿态移动，给内外圈建立稳定共同约束。",
            "结果中的 all-camera camera-origin projection 应符合真实空间布局。",
        ],
        "quality_en": [
            "It must cover both inner and outer cameras; if only one side sees the board, this capture cannot bridge.",
            "Move the board around the table / center region with multiple poses to create stable shared constraints.",
            "The all-camera camera-origin projection should match the real physical layout.",
        ],
        "raw_path": "D:/output/calib/large_marker/<time>/<camera_id>/<sn>_<imageid>.jpg",
        "staged_path": "/home/ubuntu/calib_data/calib_2026_05_31_v3/large_marker_bridge_all32",
        "qc_step_slug": "large-marker-bridge",
        "report": {
            "label": "Bridge large-board capture / QC",
            "label_zh": "bridge 大标定板采集 / QC",
            "label_en": "Bridge large-board capture / QC",
            "path": INNER_BRIDGE_QUALITY_REPORT,
            "kind": "run-level capture report",
        },
        "used_by": ["inner-rig-extrinsics", "large-marker-bridge"],
    },
    {
        "slug": "small",
        "name": "small",
        "name_zh": "small 小标定板",
        "target_zh": "内圈内参 + 内圈外参 refine / quality",
        "target_en": "inner intrinsics + inner pose refinement / quality",
        "hardware_zh": "高密度 / 小尺寸 calib board",
        "hardware_en": "high-density small calibration board",
        "capture_zh": "在内圈工作空间内移动小标定板，重点覆盖 8 台内圈相机的图像面。",
        "capture_en": "Move the small board inside the inner working volume, focusing on broad image coverage for the 8 inner cameras.",
        "quality_zh": [
            "8 台内圈相机都应有有效角点；掉帧 sequence 必须整体排除。",
            "小标定板应覆盖每台内圈相机图像面的不同位置，不要只在桌面中心小范围移动。",
            "内参 residual 和 feature accumulation 应稳定一致。",
        ],
        "quality_en": [
            "All 8 inner cameras should have valid corners; dropped-frame sequences must be excluded as a whole.",
            "Move the small board across different image regions for every inner camera, not only around a small table-center area.",
            "Intrinsic residuals and feature accumulation should be stable and consistent.",
        ],
        "raw_path": "D:/output/calib/small_marker/<time>/<camera_id>/<sn>_<imageid>.jpg",
        "staged_path": "/home/ubuntu/calib_data/calib_2026_05_31_v3/small_marker_inner8",
        "qc_step_slug": "small-marker-inner",
        "report": {
            "label": "Small-board inner capture / QC",
            "label_zh": "small 小标定板内圈采集 / QC",
            "label_en": "Small-board inner capture / QC",
            "path": INNER_CAPTURE_REPORT,
            "kind": "data capture report",
        },
        "used_by": ["small-marker-inner"],
    },
]

CAPTURE_DATASET_BY_SLUG = {dataset["slug"]: dataset for dataset in CAPTURE_DATASETS}


WORKFLOW_STEPS = [
    {
        "slug": "outer-large-marker",
        "number": "1",
        "title": "Outer Large Marker Intrinsics",
        "title_zh": "外圈内参",
        "title_en": "Outer Intrinsics",
        "short_title": "Outer Intrinsics",
        "mode": "run_studio_calibration_pipeline",
        "purpose": (
            "使用 large 数据：用大标定板给外圈 24 个 12mm 相机建立稳定内参初值。"
            "这是相机硬件/镜头没有变化时可复用的一次性步骤。"
        ),
        "purpose_en": (
            "Use the large capture: the large board builds stable intrinsic estimates for "
            "the 24 fixed outer cameras. Reuse this when lenses, focus, and "
            "resolution are unchanged."
        ),
        "capture_date": "2026-06-04",
        "required_capture_slugs": ["large"],
        "data_paths": [
            "Windows raw: D:/output/calib/outer_large_marker/<time>/<camera_id>/<sn>_<imageid>.jpg",
            "t0 staged/QC: /home/ubuntu/calib_data/calib_2026_05_31_v3/outer_large_marker_*",
        ],
        "output_paths": [
            "/home/ubuntu/calib_data/current_calibration/reports/06_outer_intrinsics_outer_large_marker/index.html",
            "/home/ubuntu/calib_data/current_calibration/artifacts/studio_32_cameras.yaml",
        ],
        "capture_reports": [
            {
                "label": "Outer large-marker capture / QC",
                "label_zh": "外圈 large marker 采集 / QC",
                "label_en": "Outer large-marker capture / QC",
                "path": OUTER_CAPTURE_REPORT,
                "kind": "data capture report",
            },
        ],
        "result_reports": [
            {
                "label": "Outer intrinsic calibration result",
                "label_zh": "外圈内参标定结果",
                "label_en": "Outer intrinsic calibration result",
                "path": OUTER_INTRINSIC_REPORT,
                "kind": "intrinsic report",
            },
        ],
        "run_params": {"publish_current": True, "pipeline_dry_run": False},
        "notes": (
            "当前没有单独 outer-intrinsic-only mode；production wrapper 会把它纳入"
            "最终报告刷新。"
        ),
        "notes_en": (
            "There is no standalone outer-intrinsic-only operation yet; the "
            "production wrapper refreshes this report as part of publication."
        ),
    },
    {
        "slug": "whole-outer-cage",
        "number": "2",
        "title": "Whole Tower Outer Extrinsics",
        "title_zh": "外圈外参",
        "title_en": "Outer Extrinsics",
        "short_title": "Whole / Outer",
        "mode": "operate_whole_outer_cage",
        "purpose": (
            "使用 tower 数据作为外圈 24 个固定相机外参的主约束；bridge 数据随后用于"
            "内外圈绑定和最终一致性检查。当前 tower 实现使用 black-tile physical-corner "
            "correspondence、tag 8cm、间距 2cm。"
        ),
        "purpose_en": (
            "Use the tower capture as the main constraint for the 24 fixed outer-camera "
            "extrinsics; the bridge capture is then used for inner/outer alignment and final "
            "consistency checks. The current tower pipeline uses physical black-tile corner "
            "correspondences with 8 cm tags and 2 cm gaps."
        ),
        "capture_date": "2026-05-31",
        "required_capture_slugs": ["tower", "bridge"],
        "data_paths": [
            "Windows raw: D:/output/calib/whole/<time>/<camera_id>/<sn>_<imageid>.jpg",
            "t0 staged: /home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1/whole_outer24_filtered_min4_fullres_min4cam",
        ],
        "output_paths": [
            "/home/ubuntu/calib_data/studio_calibration_runs/latest_whole_outer_cage",
            "/home/ubuntu/calib_data/current_calibration/reports/07_outer_extrinsics_whole/index.html",
        ],
        "capture_reports": [
            {
                "label": "Whole tower capture / QC",
                "label_zh": "whole tower 采集 / QC",
                "label_en": "Whole tower capture / QC",
                "path": OUTER_CAPTURE_REPORT,
                "kind": "data capture report",
            },
        ],
        "result_reports": [
            {
                "label": "Outer extrinsic calibration result",
                "label_zh": "外圈外参标定结果",
                "label_en": "Outer extrinsic calibration result",
                "path": OUTER_EXTRINSIC_REPORT,
                "kind": "extrinsic report",
            },
        ],
        "run_params": {"pipeline_dry_run": False},
        "notes": "外圈相机固定后，除非 cage 或镜头改动，通常不需要频繁重跑。",
        "notes_en": "After the outer cage is fixed, rerun this only when the cage or optics change.",
    },
    {
        "slug": "small-marker-inner",
        "number": "3",
        "title": "Small Marker Inner Intrinsics",
        "title_zh": "内圈内参",
        "title_en": "Inner Intrinsics",
        "short_title": "Inner Intrinsics",
        "mode": "operate_small_marker_inner",
        "purpose": (
            "使用 small 数据：用小标定板检查内圈 8 相机内参、角点覆盖和 fixed-rig residual。"
            "这是内圈相机移动后仍可复用的相机内参质量入口。"
        ),
        "purpose_en": (
            "Use the small capture: the small board checks inner-camera intrinsics, feature "
            "coverage, and fixed-rig residuals."
        ),
        "capture_date": "2026-05-31",
        "required_capture_slugs": ["small"],
        "data_paths": [
            "Windows raw: D:/output/calib/small_marker/<time>/<camera_id>/<sn>_<imageid>.jpg",
            "t0 staged: /home/ubuntu/calib_data/calib_2026_05_31_v3/small_marker_inner8",
        ],
        "output_paths": [
            "/home/ubuntu/calib_data/studio_calibration_runs/latest_small_marker_inner",
            "/home/ubuntu/calib_data/current_calibration/reports/03_inner_intrinsics_small_marker/index.html",
            "/home/ubuntu/calib_data/current_calibration/reports/04_inner_extrinsics_small_marker/index.html",
        ],
        "capture_reports": [
            {
                "label": "Small-marker inner capture / QC",
                "label_zh": "内圈 small marker 采集 / QC",
                "label_en": "Small-marker inner capture / QC",
                "path": INNER_CAPTURE_REPORT,
                "kind": "data capture report",
            },
        ],
        "result_reports": [
            {
                "label": "Inner intrinsic calibration result",
                "label_zh": "内圈内参标定结果",
                "label_en": "Inner intrinsic calibration result",
                "path": INNER_INTRINSIC_REPORT,
                "kind": "intrinsic report",
            },
        ],
        "run_params": {"pipeline_dry_run": False},
        "notes": "默认是 fixed-rig quality probe；如果内参异常，应先修正内参再看外参。",
        "notes_en": "This defaults to a fixed-rig quality probe. Fix intrinsic issues before judging poses.",
    },
    {
        "slug": "inner-rig-extrinsics",
        "number": "4",
        "title": "Large Marker Inner Rig Extrinsics",
        "title_zh": "内圈外参",
        "title_en": "Inner Extrinsics",
        "short_title": "Inner Extrinsics",
        "mode": "operate_large_marker_bridge",
        "purpose": (
            "使用 bridge 数据建立或刷新内圈粗外参 baseline；small 数据随后用于内圈"
            "内参和 pose quality/refinement。这一步需要已经有可用的 inner intrinsics。"
        ),
        "purpose_en": (
            "Use the bridge capture to create or refresh the coarse inner-rig pose baseline; "
            "the small capture then checks inner intrinsics and pose quality/refinement. "
            "This requires usable inner intrinsics."
        ),
        "capture_date": "2026-05-31",
        "required_capture_slugs": ["bridge", "small"],
        "data_paths": [
            "Windows raw: D:/output/calib/large_marker/<time>/<camera_id>/<sn>_<imageid>.jpg",
            "t0 staged: /home/ubuntu/calib_data/calib_2026_05_31_v3/large_marker_inner8",
        ],
        "output_paths": [
            "/home/ubuntu/calib_data/studio_calibration_runs/latest_large_marker_bridge",
            "/home/ubuntu/calib_data/current_calibration/reports/04_inner_extrinsics_small_marker/index.html",
        ],
        "capture_reports": [
            {
                "label": "Large-marker inner capture / QC",
                "label_zh": "large marker 内圈采集 / QC",
                "label_en": "Large-marker inner capture / QC",
                "path": INNER_BRIDGE_QUALITY_REPORT,
                "kind": "run-level capture report",
            },
        ],
        "result_reports": [
            {
                "label": "Inner extrinsic calibration result",
                "label_zh": "内圈外参标定结果",
                "label_en": "Inner extrinsic calibration result",
                "path": INNER_EXTRINSIC_REPORT,
                "kind": "extrinsic report",
            },
        ],
        "run_params": {
            "run_large_inner_init": True,
            "run_large_bridge": False,
            "run_reports": True,
            "pipeline_dry_run": False,
        },
        "notes": (
            "这是依赖图里的 inner pose prior 节点；完整 bridge 节点会再次使用 "
            "large_marker_bridge_all32。"
        ),
        "notes_en": (
            "This is the inner-pose prior node in the dependency graph; the full "
            "bridge step uses large_marker_bridge_all32 again."
        ),
    },
    {
        "slug": "large-marker-bridge",
        "number": "5",
        "title": "Large Marker Inner/Outer Bridge",
        "title_zh": "内外圈 Bridge",
        "title_en": "Inner/Outer Bridge",
        "short_title": "Bridge",
        "mode": "operate_large_marker_bridge",
        "purpose": (
            "使用 bridge 数据：用大标定板同时约束 inner 和 outer，相当于把内圈 8 相机"
            "绑定到固定外圈坐标系。"
        ),
        "purpose_en": (
            "Use the bridge capture: the large board constrains inner and outer cameras together, "
            "binding the movable inner rig to the fixed outer coordinate frame."
        ),
        "capture_date": "2026-05-31",
        "required_capture_slugs": ["bridge"],
        "data_paths": [
            "Windows raw: D:/output/calib/large_marker/<time>/<camera_id>/<sn>_<imageid>.jpg",
            "t0 staged: /home/ubuntu/calib_data/calib_2026_05_31_v3/large_marker_bridge_all32",
        ],
        "output_paths": [
            "/home/ubuntu/calib_data/studio_calibration_runs/latest_large_marker_bridge",
            "/home/ubuntu/calib_data/current_calibration/reports/09_bridge_result_large_marker/index.html",
        ],
        "capture_reports": [
            {
                "label": "Large-marker all-camera capture / QC",
                "label_zh": "large marker 全相机采集 / QC",
                "label_en": "Large-marker all-camera capture / QC",
                "path": INNER_BRIDGE_QUALITY_REPORT,
                "kind": "run-level capture report",
            },
        ],
        "result_reports": [
            {
                "label": "Inner/outer bridge calibration result",
                "label_zh": "内外圈 bridge 标定结果",
                "label_en": "Inner/outer bridge calibration result",
                "path": BRIDGE_RESULT_REPORT,
                "kind": "bridge report",
            },
        ],
        "run_params": {"pipeline_dry_run": False},
        "notes": "内圈位置变化后的关键步骤；应使用全部可见 camera correspondence。",
        "notes_en": "This is the key step after moving inner cameras; use all visible camera correspondences.",
    },
    {
        "slug": "publish-current",
        "number": "6",
        "title": "Publish Current Reports",
        "title_zh": "发布当前结果",
        "title_en": "Publish Current Result",
        "short_title": "Publish",
        "mode": "run_studio_calibration_pipeline",
        "purpose": (
            "导出统一 YAML、刷新 3D viewer、intrinsic/extrinsic/bridge reports，"
            "并发布到 9899 的 current_calibration 入口。"
        ),
        "purpose_en": (
            "Export the unified YAML, refresh the 3D viewer and reports, and publish "
            "them under current_calibration on 9899."
        ),
        "capture_date": "mixed inputs",
        "required_capture_slugs": ["large", "tower", "bridge", "small"],
        "data_paths": [
            "Current run root: /home/ubuntu/calib_data/studio_calibration_runs/recalib_20260610_black_tile_wide200_pipeline_v2",
            "Published root: /home/ubuntu/calib_data/current_calibration",
        ],
        "output_paths": [
            "/home/ubuntu/calib_data/current_calibration/artifacts/studio_32_cameras.yaml",
            "/home/ubuntu/calib_data/current_calibration/reports/01_3d_viewer/index.html",
        ],
        "capture_reports": [],
        "result_reports": [
            {
                "label": "Unified 32-camera YAML",
                "label_zh": "统一 32 相机 YAML",
                "label_en": "Unified 32-camera YAML",
                "path": FINAL_STUDIO32_YAML,
                "kind": "final artifact",
            },
            {
                "label": "Unified 3D viewer",
                "label_zh": "统一 3D Viewer",
                "label_en": "Unified 3D viewer",
                "path": UNIFIED_VIEWER,
                "kind": "3D viewer",
            },
        ],
        "run_params": {
            "bridge_only": True,
            "publish_current": True,
            "pipeline_dry_run": False,
        },
        "notes": "发布前请确认上一阶段的 residual 和 camera-origin projection sanity checks 正常。",
        "notes_en": "Before publishing, confirm residuals and camera-origin projection checks look normal.",
    },
]


WORKFLOW_BY_SLUG = {step["slug"]: step for step in WORKFLOW_STEPS}
QUICK_ACTION_BY_SLUG = {action["slug"]: action for action in QUICK_ACTIONS}


WORKFLOW_GRAPH_NODES = [
    {
        "slug": "outer-large-marker",
        "x": 70,
        "y": 90,
        "w": 235,
        "h": 128,
        "lane": "outer",
        "label": ["Outer intrinsics", "outer_large_marker"],
        "label_zh": ["外圈内参", "large"],
        "label_en": ["Outer intrinsics", "large"],
        "badge": "K first",
    },
    {
        "slug": "whole-outer-cage",
        "x": 395,
        "y": 90,
        "w": 235,
        "h": 128,
        "lane": "outer",
        "label": ["Outer extrinsics", "whole / tower"],
        "label_zh": ["外圈外参", "tower + bridge"],
        "label_en": ["Outer extrinsics", "tower + bridge"],
        "badge": "pose",
    },
    {
        "slug": "small-marker-inner",
        "x": 70,
        "y": 330,
        "w": 235,
        "h": 128,
        "lane": "inner",
        "label": ["Inner intrinsics", "small_marker"],
        "label_zh": ["内圈内参", "small"],
        "label_en": ["Inner intrinsics", "small"],
        "badge": "K first",
    },
    {
        "slug": "inner-rig-extrinsics",
        "x": 395,
        "y": 330,
        "w": 235,
        "h": 128,
        "lane": "inner",
        "label": ["Inner extrinsics", "large_marker inner8"],
        "label_zh": ["内圈外参", "bridge + small"],
        "label_en": ["Inner extrinsics", "bridge + small"],
        "badge": "pose",
    },
    {
        "slug": "large-marker-bridge",
        "x": 745,
        "y": 210,
        "w": 245,
        "h": 132,
        "lane": "bridge",
        "label": ["All32 bridge", "large_marker all cameras"],
        "label_zh": ["32 相机 Bridge", "bridge"],
        "label_en": ["All32 bridge", "bridge"],
        "badge": "needs K+pose",
    },
    {
        "slug": "publish-current",
        "x": 1080,
        "y": 210,
        "w": 245,
        "h": 132,
        "lane": "publish",
        "label": ["Overall publish", "viewer + YAML"],
        "label_zh": ["发布总结果", "viewer + YAML"],
        "label_en": ["Overall publish", "viewer + YAML"],
        "badge": "current",
    },
]


WORKFLOW_GRAPH_EDGES = [
    ("outer-large-marker", "whole-outer-cage", "outer K"),
    ("small-marker-inner", "inner-rig-extrinsics", "inner K"),
    ("whole-outer-cage", "large-marker-bridge", "outer pose"),
    ("inner-rig-extrinsics", "large-marker-bridge", "inner pose"),
    ("large-marker-bridge", "publish-current", "final all32"),
]


STEP_GUIDANCE = {
    "outer-large-marker": {
        "overview_zh": (
            "这一步使用外圈相机拍到的粗 pattern 标定板来估计每台外圈相机的 "
            "K 和畸变。外圈相机固定后，这通常是低频维护项；只有分辨率、镜头、"
            "焦距、对焦或相机硬件变化时才需要重新采集。"
        ),
        "overview_en": (
            "This step estimates K and distortion for each fixed outer camera from "
            "low-density board observations. It is usually a low-frequency maintenance "
            "step; rerun it when resolution, lens, focus, or camera hardware changes."
        ),
        "checks_zh": [
            "先打开 large 的 Data Collect / QC：每台外圈相机应有足够多的大标定板角点，并覆盖图像中心和边缘。",
            "再看内参报告：fx/fy、主点、畸变和 residual 不应出现单台相机明显离群。",
            "如果只有某几台相机 coverage 很差，优先补采这些视角，不要用稀疏点强行优化。"
        ],
        "checks_en": [
            "Open the large Data Collect / QC page first: every outer camera should have enough large-board corners across the image center and borders.",
            "Then inspect the intrinsic report: fx/fy, principal point, distortion, and residuals should not have obvious per-camera outliers.",
            "If only a few cameras have weak coverage, recapture those views instead of forcing calibration from sparse points."
        ],
    },
    "whole-outer-cage": {
        "overview_zh": (
            "这一步使用 AprilTag tower 的 whole 数据优化外圈 24 相机的相对外参。"
            "当前流程不再依赖理想八棱柱模型，而是使用 tag ID 建立物理黑色 tile corner "
            "correspondence，并用 8cm tag 与 2cm 间距定义局部几何。"
        ),
        "overview_en": (
            "This step refines relative extrinsics for the 24 fixed outer cameras from "
            "the AprilTag tower whole capture. The current workflow does not rely on "
            "an ideal octagonal-prism model; it uses tag IDs to build physical black-tile "
            "corner correspondences, with 8 cm tags and 2 cm gaps defining local geometry."
        ),
        "checks_zh": [
            "先打开 tower 的 Data Collect / QC：大多数外圈相机应在多帧里看到足够多 tag physical corners。",
            "bridge 的 Data Collect / QC 也要通过，因为最终 all-camera 结果会用 bridge 检查外圈和绑定内圈。",
            "结果报告重点看 per-camera final observation count、median/p90 reprojection error 和是否有 disconnected camera。",
            "在 3D viewer 里加载 correspondence 时，单帧 tower corner 应形成竖直塔形，而不是躺倒或散开。"
        ],
        "checks_en": [
            "Open the tower Data Collect / QC page first: most outer cameras should see enough tag physical corners across many frames.",
            "The bridge Data Collect / QC page should also pass, because the final all-camera result uses bridge data to check the outer solution and bind the inner rig.",
            "In the result report, focus on per-camera final observation count, median/p90 reprojection error, and disconnected cameras.",
            "In the 3D viewer, loaded correspondences for a single frame should form an upright tower, not a flat or scattered shape."
        ],
    },
    "small-marker-inner": {
        "overview_zh": (
            "这一步使用细 pattern 标定板检查内圈 8 相机的内参和局部 rig 质量。"
            "内圈相机可能移动，但只要镜头、焦距和分辨率不变，内参可以复用；"
            "移动后更关键的是重新估计外参和 bridge。"
        ),
        "overview_en": (
            "This step uses the high-density board to check inner-camera intrinsics and "
            "local rig quality. Inner cameras may move, but intrinsics are reusable as "
            "long as optics, focus, and resolution are unchanged; after movement, poses "
            "and bridge alignment are the critical parts to refresh."
        ),
        "checks_zh": [
            "先打开 small 的 Data Collect / QC：8 台内圈相机都应有有效小标定板角点，不要把掉帧 sequence 混进来。",
            "内参报告重点看 feature accumulation 是否覆盖图像面，以及 residual 是否一致。",
            "如果某台 inner camera failed 或 coverage 很偏，应先重采 small marker。"
        ],
        "checks_en": [
            "Open the small Data Collect / QC page first: all 8 inner cameras should have valid small-board corners, and dropped-frame sequences must be excluded.",
            "The intrinsic report should show broad feature accumulation across the image and consistent residuals.",
            "If any inner camera fails or has strongly biased coverage, recapture the small-marker sequence first."
        ],
    },
    "inner-rig-extrinsics": {
        "overview_zh": (
            "这一步用 large marker 中只关注内圈观测的部分建立内圈相机外参初值。"
            "它是后续 all-camera bridge 的 inner pose prior，不是最终单独交付的 32 相机结果。"
        ),
        "overview_en": (
            "This step uses the inner-camera observations in the large-marker dataset to "
            "build an inner-rig pose baseline. It is the inner pose prior for the full "
            "all-camera bridge, not the final delivered 32-camera calibration by itself."
        ),
        "checks_zh": [
            "先确认 bridge 的 Data Collect / QC：大标定板在桌面/中心区域移动时，8 台内圈相机有足够共视。",
            "再确认 small 的 Data Collect / QC：小标定板数据能支撑内圈内参和 pose quality/refinement。",
            "结果报告看内圈 camera-origin projection sanity check，投影位置应符合物理布局。",
            "若内圈相机刚移动过，这一步和 bridge 都应重新跑。"
        ],
        "checks_en": [
            "First check the bridge Data Collect / QC page: the large board should give enough co-visibility for the 8 inner cameras around the table / center region.",
            "Then check the small Data Collect / QC page: the small-board data should support inner intrinsics and pose quality/refinement.",
            "Inspect camera-origin projection sanity checks; projected camera positions should match the physical layout.",
            "If the inner cameras moved, rerun both this step and the bridge step."
        ],
    },
    "large-marker-bridge": {
        "overview_zh": (
            "这一步使用 large marker 的全部可见相机 correspondence，把内圈 rig 绑定到外圈固定坐标系。"
            "它是内圈移动后的关键重标定步骤，也是最终 32 相机统一坐标系是否可信的主要检查点。"
        ),
        "overview_en": (
            "This step uses all visible large-marker correspondences to bind the inner "
            "rig into the fixed outer coordinate frame. It is the key recalibration step "
            "after moving inner cameras and the main confidence check for the unified "
            "32-camera coordinate system."
        ),
        "checks_zh": [
            "先打开 bridge 的 Data Collect / QC：大标定板必须同时覆盖 inner 和 outer；如果只看到一边，相当于不能 bridge。",
            "结果报告应看 all-camera camera-origin projection，内外圈投影关系应符合实际空间布局。",
            "如果 3D viewer 中 inner 明显偏离 outer cage 中心，应优先排查 bridge 数据和相机 ID 对齐。"
        ],
        "checks_en": [
            "Open the bridge Data Collect / QC page first: the large board must cover both inner and outer cameras; if only one side sees it, it cannot bridge.",
            "Inspect the all-camera camera-origin projection report; inner/outer projected positions should match the real layout.",
            "If the 3D viewer shows the inner rig clearly offset from the outer cage center, first check bridge data and camera-ID alignment."
        ],
    },
    "publish-current": {
        "overview_zh": (
            "这一步把当前有效的 inner、outer 和 bridge 结果打包成统一 YAML，并刷新 9899 首页引用的"
            " 3D viewer 与报告。下游重建、SLAM、3DGS 应优先使用这里发布的 YAML。"
        ),
        "overview_en": (
            "This step packages the current validated inner, outer, and bridge results into "
            "one YAML file, then refreshes the 3D viewer and reports linked from the 9899 "
            "home page. Downstream reconstruction, SLAM, and 3DGS should consume this YAML first."
        ),
        "checks_zh": [
            "发布前确认 outer residual、bridge projection、inner intrinsic quality 都没有明显异常。",
            "发布后从首页打开 3D viewer 和下载 YAML，确认路径指向 current_calibration。",
            "如果只是查看结果，不需要运行这一步；只有上游结果更新后才需要发布。"
        ],
        "checks_en": [
            "Before publishing, confirm outer residuals, bridge projection checks, and inner intrinsic quality do not show obvious anomalies.",
            "After publishing, open the 3D viewer and download the YAML from the home page; both should point to current_calibration.",
            "If you only want to inspect results, do not run this step; publish only after upstream results changed."
        ],
    },
}


class ReportHandler(SimpleHTTPRequestHandler):
    server_version = "CameraCalibReportHTTP/1.0"

    def __init__(self, *args, directory=None, **kwargs):
        self.root = Path(directory or DEFAULT_ROOT).resolve()
        super().__init__(*args, directory=str(self.root), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        request_path = parsed.path
        if request_path in ("/", "/index.html"):
            self._serve_console()
            return
        if request_path == "/healthz":
            self._serve_health()
            return
        if request_path.startswith("/api/"):
            self._serve_api_get(parsed)
            return
        if request_path.startswith("/data-collect/"):
            slug = request_path.strip("/").split("/", 1)[1]
            self._serve_data_collect_detail(slug)
            return
        if request_path.startswith("/operation/"):
            slug = request_path.strip("/").split("/", 1)[1]
            self._serve_operation_detail(slug)
            return
        super().do_GET()

    def do_HEAD(self):
        parsed = urlparse(self.path)
        request_path = parsed.path
        if request_path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            return
        if request_path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            return
        super().do_HEAD()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._serve_api_post(parsed)
            return
        self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)

    def _current_entry_path(self):
        path = self.root / CURRENT_ENTRY_REL
        if path.is_file():
            return path
        return None

    def _serve_current_entry(self):
        path = self._current_entry_path()
        if path is None:
            return False
        try:
            data = path.read_bytes()
        except OSError:
            return False
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True

    def _serve_health(self):
        payload = {
            "ok": True,
            "root": str(self.root),
            "service": "camera-calibration-report-http",
            "panel_api_available": bool(getattr(self.server, "job_manager", None)),
        }
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _panel_manager(self):
        manager = getattr(self.server, "job_manager", None)
        if manager is None:
            message = "Panel backend is unavailable"
            if PANEL_IMPORT_ERROR is not None:
                message += f": {PANEL_IMPORT_ERROR}"
            raise RuntimeError(message)
        return manager

    def _serve_api_get(self, parsed):
        path = parsed.path
        try:
            manager = self._panel_manager()
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return

        if path == "/api/modes":
            self._send_json({"modes": MODE_DEFINITIONS})
            return
        if path == "/api/jobs":
            self._send_json({"jobs": manager.list_jobs()})
            return
        if path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 5 and parts[3] == "artifact":
                self._serve_job_artifact(manager, parts[2], parts[4], "/".join(parts[5:]) or "index.html")
                return
            if len(parts) == 3:
                job = manager.get_job(parts[2])
                if not job:
                    self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(job)
                return
            if len(parts) == 4 and parts[3] == "log":
                query = parse_qs(parsed.query)
                offset = int(query.get("offset", ["0"])[0] or 0)
                chunk = manager.log_chunk(parts[2], offset)
                if chunk is None:
                    self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(chunk)
                return
        self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)

    def _serve_api_post(self, parsed):
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            manager = self._panel_manager()
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return

        if parsed.path == "/api/jobs":
            try:
                job = manager.start_job(
                    payload.get("mode", ""),
                    payload.get("params", {}),
                    dry_run=bool(payload.get("dry_run", False)),
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(job, status=HTTPStatus.CREATED)
            return

        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 4:
                job = manager.cancel_job(parts[2])
                if not job:
                    self._send_json({"error": "Job not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(job)
                return
        self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)

    def _serve_job_artifact(self, manager, job_id, report_index_text, rel_path):
        job = manager.get_job(job_id)
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

    def _candidate_reports(self):
        names = set()
        patterns = ("**/index.html", "**/*report*.html")
        for pattern in patterns:
            for path in self.root.glob(pattern):
                if not path.is_file():
                    continue
                try:
                    rel = path.relative_to(self.root)
                except ValueError:
                    continue
                names.add(rel.as_posix())
        return sorted(
            names,
            key=lambda name: (self.root / name).stat().st_mtime,
            reverse=True,
        )[:200]

    def _curated_paths(self):
        paths = set()
        for group in REPORT_GROUPS:
            for item in group.get("items", []):
                if item.get("path"):
                    paths.add(item["path"])
        return paths

    def _path_href(self, rel):
        base_url = getattr(self.server, "report_base_url", DEFAULT_REPORT_BASE_URL)
        return base_url.rstrip("/") + "/" + quote(rel)

    def _pipeline_summary(self, rel):
        for prefix in (FAST_INNER_BRIDGE_LATEST, OUTER_TOWER_LATEST):
            if rel == prefix or rel.startswith(prefix + "/"):
                summary_path = self.root / prefix / "summary.json"
                if not summary_path.is_file():
                    return {}
                try:
                    return json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return {}
        return {}

    def _is_outer_placeholder_viewer(self, path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return True
        return (
            "Outer Tower Viewer Placeholder" in text
            or "data-viewer-placeholder" in text
        )

    def _viewer_ready(self, rel, path):
        if rel.endswith("outer_tower/latest/viewer/index.html"):
            return (
                (path.parent / "scene_data.json").is_file()
                and not self._is_outer_placeholder_viewer(path)
            )
        if rel.endswith("current_calibration/reports/01_3d_viewer/index.html"):
            rig_data_path = path.parent / "rig_data.json"
            if not rig_data_path.is_file():
                return False
            try:
                rig_data = json.loads(rig_data_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if (rig_data.get("metrics") or {}).get("outer_pose_source") == "colmap_sim3_approx":
                return False
            return bool(rig_data.get("cameras"))
        if rel.endswith("combined_studio_rig_viewer_v1/index.html"):
            rig_data_path = path.parent / "rig_data.json"
            if not rig_data_path.is_file():
                return False
            try:
                rig_data = json.loads(rig_data_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            current_outer = (
                self.root
                / OUTER_TOWER_LATEST
                / "camera_tr_rig_delta_refined.yaml"
            )
            legacy_outer = (
                self.root
                / OUTER_TOWER_LATEST
                / "tag_refine_robust/camera_tr_rig_delta_refined_accepted.yaml"
            )
            outer_final = ((rig_data.get("inputs") or {}).get("outer_final_pose_yaml") or "")
            outer_source = ((rig_data.get("metrics") or {}).get("outer_pose_source") or "")
            return (
                outer_source in {"outer_final_pose_yaml", "outer_final_pose_yaml_bridge_aligned"}
                and outer_final in {str(current_outer), str(legacy_outer)}
            )
        if rel.endswith("reports/interactive_inner_viewer/index.html") or rel.endswith("interactive_rig_viewer_v1/index.html"):
            rig_data_path = path.parent / "rig_data.json"
            if not rig_data_path.is_file():
                return False
            try:
                rig_data = json.loads(rig_data_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            return any(camera.get("image_url") for camera in rig_data.get("cameras", []))
        return True

    def _report_item_ready(self, item):
        rel = item.get("path", "")
        if not rel:
            return False
        path = self.root / rel
        if not path.is_file():
            return False
        if "3D viewer" in item.get("kind", ""):
            return self._viewer_ready(rel, path)
        return True

    def _render_report_item(self, item):
        rel = item.get("path", "")
        ready = self._report_item_ready(item)
        status = item["status_if_exists"] if ready else item["status_if_missing"]
        state_class = " ready" if ready else " missing"
        summary = self._pipeline_summary(rel)
        if ready and (
            summary.get("dry_run")
            or summary.get("mode") == "dry_run"
            or summary.get("args", {}).get("dry_run")
        ):
            status = "preview / dry-run latest"
            state_class = " ready diagnostic"
        if item.get("diagnostic"):
            state_class += " diagnostic"
        label = item.get("label_html") or html.escape(item["label"])
        kind = html.escape(item.get("kind", "report"))
        href = self._path_href(rel)
        path_text = html.escape(href)
        title = f"<a href=\"{html.escape(href)}\">{label}</a>"
        return (
            f"<li class=\"report-item{state_class}\">"
            "<div>"
            f"{title}"
            f"<small>{kind} · {html.escape(status)}</small>"
            "</div>"
            f"<code>{path_text}</code>"
            "</li>"
        )

    def _render_report_group(self, group, panel_url):
        notes = "".join(
            f"<li>{html.escape(note)}</li>" for note in group.get("notes", [])
        )
        items = "".join(self._render_report_item(item) for item in group.get("items", []))
        notes_html = f"<ul class=\"notes\">{notes}</ul>" if notes else ""
        return (
            f"<section class=\"report-group {html.escape(group['status'])}\">"
            "<div class=\"group-head\">"
            "<div>"
            f"<h2>{html.escape(group['title'])}</h2>"
            f"<p class=\"subtitle\">{html.escape(group['subtitle'])}</p>"
            "</div>"
            f"<span class=\"status-pill\">{html.escape(group['status_label'])}</span>"
            "</div>"
            f"<p>{html.escape(group['description'])}</p>"
            f"<ul class=\"report-list\">{items}</ul>"
            f"{notes_html}"
            "</section>"
        )

    def _mode_default_params(self, mode):
        definition = MODE_DEFINITIONS.get(mode, {})
        return {
            item["name"]: item.get("default")
            for item in definition.get("params", [])
            if "name" in item and "default" in item
        }

    def _action_payload(self, action_or_step, dry_run=None):
        mode = action_or_step.get("mode", "")
        params = self._mode_default_params(mode)
        params.update(action_or_step.get("run_params", {}))
        params.update(action_or_step.get("params", {}))
        if dry_run is None:
            dry_run = bool(action_or_step.get("dry_run", False))
        return {
            "mode": mode,
            "params": params,
            "dry_run": bool(dry_run),
        }

    def _run_button(self, label, action_or_step, *, dry_run, primary=False, confirm=False, debug=False, label_en=None):
        payload = html.escape(json.dumps(self._action_payload(action_or_step, dry_run), ensure_ascii=False))
        classes = ["run-button"]
        if primary:
            classes.append("primary")
        if debug:
            classes.append("debug-control")
        confirm_text = "true" if confirm else "false"
        return (
            f"<button class=\"{' '.join(classes)}\" data-payload=\"{payload}\" "
            f"data-confirm=\"{confirm_text}\" onclick=\"startCalibrationJob(this)\">"
            f"{self._bi(label, label_en or label)}</button>"
        )

    def _debug_toggle(self):
        return (
            "<label class=\"debug-toggle\">"
            "<input type=\"checkbox\" onchange=\"toggleDebugControls(this.checked)\"> "
            f"{self._bi('调试：显示预演按钮', 'Debug: show dry-run controls')}"
            "</label>"
        )

    def _capture_qc_action(self, step):
        action = dict(step)
        action["run_params"] = dict(step.get("run_params", {}))
        action["run_params"]["pipeline_dry_run"] = True
        return action

    def _page_css(self):
        return """
    :root {
      --ink: #20242a;
      --muted: #5d6673;
      --line: #d7dde5;
      --soft: #f6f8fb;
      --panel: #ffffff;
      --blue: #0b66c3;
      --blue-soft: #eaf3ff;
      --green: #1b7f45;
      --red: #c9362c;
      --amber: #9a6700;
      --violet: #6f42c1;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #fbfcfe;
    }
    header {
      padding: 26px 34px 22px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }
    main { padding: 24px 34px 42px; max-width: 1380px; }
    h1 { margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }
    h2 { margin: 0 0 10px; font-size: 18px; letter-spacing: 0; }
    h3 { margin: 0 0 7px; font-size: 16px; letter-spacing: 0; }
    p { color: var(--muted); line-height: 1.48; }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .i18n-en { display: none; }
    body.lang-en .i18n-zh { display: none; }
    body.lang-en .i18n-en { display: inline; }
    code {
      background: var(--soft);
      border-radius: 5px;
      padding: 2px 5px;
      overflow-wrap: anywhere;
    }
    .hero-grid {
      display: grid;
      grid-template-columns: minmax(260px, 1.1fr) minmax(280px, 1fr);
      gap: 14px;
      margin-bottom: 18px;
    }
    .capture-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }
    .capture-card {
      display: grid;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }
    .capture-card h3 { margin-bottom: 0; }
    .capture-target {
      display: inline-flex;
      width: fit-content;
      max-width: 100%;
      border-radius: 999px;
      padding: 3px 9px;
      background: var(--blue-soft);
      color: var(--blue);
      font-size: 12px;
      font-weight: 700;
      line-height: 1.25;
    }
    .capture-card p { margin: 0; }
    .capture-card code {
      display: block;
      margin-top: 2px;
      font-size: 12px;
    }
    .readiness-grid, .dependency-card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .readiness-card, .dependency-card {
      display: grid;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }
    .readiness-card.ready, .dependency-card.ready {
      border-color: #b7dfc2;
      background: #fbfff9;
    }
    .readiness-card.missing, .dependency-card.missing {
      border-color: #efc0bd;
      background: #fffafa;
    }
    .readiness-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .readiness-pill {
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 750;
      background: var(--soft);
      color: var(--muted);
      white-space: nowrap;
    }
    .ready .readiness-pill {
      background: #e8f7ed;
      color: var(--green);
    }
    .missing .readiness-pill {
      background: #ffeceb;
      color: var(--red);
    }
    .readiness-card p, .dependency-card p {
      margin: 0;
      font-size: 13px;
    }
    .hero-card, .panel-card, .step-detail, .report-group {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .hero-actions, .quick-actions, .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .link-button, .run-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      border: 1px solid var(--blue);
      border-radius: 7px;
      padding: 8px 12px;
      background: #fff;
      color: var(--blue);
      font-weight: 650;
      cursor: pointer;
      text-decoration: none;
    }
    .link-button.primary, .run-button.primary {
      background: var(--blue);
      color: #fff;
    }
    body:not(.debug-enabled) .debug-control { display: none !important; }
    .run-button:disabled { opacity: 0.56; cursor: wait; }
    .debug-toggle {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      opacity: 0.72;
    }
    .debug-toggle input { width: 13px; height: 13px; }
    .debug-toggle:hover { opacity: 1; }
    .header-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
    }
    .lang-toggle {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--soft);
    }
    .lang-toggle button {
      border: 0;
      border-radius: 999px;
      padding: 5px 9px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }
    body.lang-zh .lang-toggle button[data-lang="zh"],
    body.lang-en .lang-toggle button[data-lang="en"] {
      background: #fff;
      color: var(--blue);
      box-shadow: 0 0 0 1px var(--line);
    }
    .quick-actions { margin-top: 12px; }
    .quick-action {
      display: grid;
      gap: 8px;
      min-width: 245px;
      flex: 1 1 250px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      background: #fff;
    }
    .quick-action.debug-control {
      border-style: dashed;
      background: #fffdf6;
    }
    .quick-action p { margin: 0; font-size: 13px; }
    .flow-wrap {
      margin-top: 22px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    .graph-scroll { overflow-x: auto; padding-bottom: 4px; }
    .dependency-graph {
      width: 100%;
      min-width: 980px;
      height: auto;
      display: block;
    }
    .graph-node rect {
      fill: #fbfdff;
      stroke: var(--line);
      stroke-width: 1.5;
      rx: 8;
    }
    .graph-node.outer rect { fill: #f4f9ff; }
    .graph-node.inner rect { fill: #f7fbf5; }
    .graph-node.bridge rect { fill: #fff9ed; }
    .graph-node.publish rect { fill: #f8f5ff; }
    .graph-node:hover rect { stroke: var(--blue); stroke-width: 2; }
    .graph-title { fill: var(--ink); font-size: 17px; font-weight: 750; }
    .graph-subtitle { fill: var(--muted); font-size: 13px; }
    .graph-date { fill: #4c5968; font-size: 12px; }
    .graph-badge { fill: var(--blue); font-size: 12px; font-weight: 700; }
    .graph-lane { fill: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; }
    .graph-edge { fill: none; stroke: #7d8794; stroke-width: 1.8; }
    .graph-edge-label { fill: var(--muted); font-size: 12px; }
    .path-list, .report-list, .notes {
      margin: 10px 0 0;
      padding: 0;
      list-style: none;
    }
    .path-list li, .notes li { margin-top: 7px; color: var(--muted); line-height: 1.4; }
    .path-list code { display: block; margin-top: 4px; }
    .empty-note { margin: 0; color: var(--muted); font-style: italic; }
    .operation-header {
      background: linear-gradient(180deg, #fff 0%, #f8fbff 100%);
    }
    .operation-header .mode-line { margin-bottom: 0; font-size: 13px; }
    .card-eyebrow {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: .02em;
      text-transform: uppercase;
    }
    .step-detail {
      border-color: #c8d8ea;
      background: #fbfdff;
    }
    .panel-card.report-card {
      border-color: #d7e4d3;
      background: #fbfef9;
    }
    .panel-card.paths-card {
      background: #fff;
    }
    .guidance-card {
      margin-bottom: 14px;
      border-color: #d8d1e8;
      background: #fffdfa;
    }
    .guidance-card h3 {
      margin-top: 14px;
      font-size: 15px;
    }
    .guidance-list {
      margin: 8px 0 0 18px;
      padding: 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .guidance-list li { margin-top: 7px; }
    .detail-grid {
      display: grid;
      grid-template-columns: minmax(300px, 1fr) minmax(300px, 1fr);
      gap: 14px;
      align-items: start;
    }
    .report-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      align-items: stretch;
    }
    .report-group { min-height: 0; }
    .group-head { display: flex; gap: 12px; justify-content: space-between; align-items: flex-start; }
    .subtitle { margin: 5px 0 0; font-size: 13px; color: var(--muted); }
    .status-pill {
      flex: 0 0 auto;
      border-radius: 999px;
      padding: 3px 9px;
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .report-item {
      display: grid;
      gap: 6px;
      border-top: 1px solid var(--line);
      padding: 10px 0;
    }
    .report-item:first-child { border-top: 0; padding-top: 0; }
    .report-item small { display: block; margin-top: 3px; color: var(--muted); }
    .report-item.ready small { color: var(--green); }
    .report-item.missing strong { color: var(--red); }
    .report-item code {
      display: block;
      overflow-wrap: anywhere;
      color: var(--muted);
      font-size: 12px;
    }
    .job-status {
      margin-top: 14px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      min-height: 46px;
      color: var(--muted);
      white-space: pre-wrap;
    }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { background: var(--soft); color: var(--muted); font-weight: 650; }
    @media (max-width: 1020px) {
      .hero-grid, .detail-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      header, main { padding-left: 18px; padding-right: 18px; }
      .quick-action { min-width: 0; flex-basis: 100%; }
    }
"""

    def _bi(self, zh, en):
        return (
            f"<span class=\"i18n-zh\">{html.escape(zh)}</span>"
            f"<span class=\"i18n-en\">{html.escape(en)}</span>"
        )

    def _svg_bi_text(self, class_name, x, y, zh, en, *, anchor=None):
        anchor_attr = f" text-anchor=\"{html.escape(anchor)}\"" if anchor else ""
        return (
            f"<text class=\"{html.escape(class_name)} i18n-zh\" x=\"{x}\" y=\"{y}\"{anchor_attr}>"
            f"{html.escape(zh)}</text>"
            f"<text class=\"{html.escape(class_name)} i18n-en\" x=\"{x}\" y=\"{y}\"{anchor_attr}>"
            f"{html.escape(en)}</text>"
        )

    def _step_text(self, step, key):
        zh = step.get(f"{key}_zh") or step.get(key, "")
        en = step.get(f"{key}_en") or step.get(key, zh)
        return self._bi(zh, en)

    def _action_text(self, action, key):
        zh = action.get(f"{key}_zh") or action.get(key, "")
        en = action.get(f"{key}_en") or action.get(key, zh)
        return self._bi(zh, en)

    def _language_toggle(self):
        return (
            "<div class=\"lang-toggle\" aria-label=\"Language\">"
            "<button type=\"button\" data-lang=\"zh\" onclick=\"setLanguage('zh')\">中文</button>"
            "<button type=\"button\" data-lang=\"en\" onclick=\"setLanguage('en')\">EN</button>"
            "</div>"
        )

    def _page_script(self):
        return """
    const LANGUAGE_STORAGE_KEY = "calibConsoleLanguageV2";
    function uiText(zh, en) {
      return document.body && document.body.classList.contains("lang-en") ? en : zh;
    }
    async function startCalibrationJob(button) {
      const payload = JSON.parse(button.dataset.payload || "{}");
      if (button.dataset.confirm === "true") {
        const ok = window.confirm(uiText(
          "这会在 t0 上启动一次真实标定处理。是否继续？",
          "This will start a real calibration job on t0. Continue?"
        ));
        if (!ok) return;
      }
      const status = document.getElementById("job-status");
      button.disabled = true;
      if (status) status.textContent = uiText("正在启动任务...", "Starting job...");
      try {
        const response = await fetch("/api/jobs", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || response.statusText);
        if (status) {
          status.innerHTML =
            uiText("已启动任务", "Started job") + ": <strong>" + body.id + "</strong>\\n" +
            uiText("任务", "Operation") + ": " + body.mode_title + "\\n" +
            uiText("输出目录", "Output dir") + ": " + body.run_dir + "\\n" +
            "<a href=\\"/api/jobs/" + encodeURIComponent(body.id) + "\\">job.json</a> · " +
            "<a href=\\"/api/jobs/" + encodeURIComponent(body.id) + "/log\\">log json</a>";
        }
      } catch (error) {
        if (status) status.textContent = uiText("任务启动失败: ", "Failed to start job: ") + error.message;
      } finally {
        button.disabled = false;
      }
    }
    function toggleDebugControls(enabled) {
      document.body.classList.toggle("debug-enabled", enabled);
      try { localStorage.setItem("calibConsoleDebugDryRun", enabled ? "1" : "0"); } catch (error) {}
      document.querySelectorAll(".debug-toggle input").forEach((input) => { input.checked = enabled; });
    }
    function setLanguage(lang) {
      const normalized = lang === "en" ? "en" : "zh";
      document.body.classList.toggle("lang-en", normalized === "en");
      document.body.classList.toggle("lang-zh", normalized !== "en");
      document.documentElement.lang = normalized === "en" ? "en" : "zh-CN";
      try { localStorage.setItem(LANGUAGE_STORAGE_KEY, normalized); } catch (error) {}
    }
    document.addEventListener("DOMContentLoaded", () => {
      let enabled = false;
      try { enabled = localStorage.getItem("calibConsoleDebugDryRun") === "1"; } catch (error) {}
      toggleDebugControls(enabled);
      let lang = "%s";
      try { lang = localStorage.getItem(LANGUAGE_STORAGE_KEY) || "%s"; } catch (error) {}
      setLanguage(lang);
    });
""" % (DEFAULT_LANGUAGE, DEFAULT_LANGUAGE)

    def _html_page(self, title, body):
        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{self._page_css()}</style>
</head>
<body class="lang-en">
  {body}
  <script>{self._page_script()}</script>
</body>
</html>
"""
        data = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _render_quick_action(self, action):
        button = self._run_button(
            action.get("title_zh", action["title"]),
            action,
            dry_run=action.get("dry_run", False),
            primary=action["slug"] == "full-run",
            confirm=action.get("confirm", False),
            debug=action.get("dry_run", False),
            label_en=action.get("title_en", action["title"]),
        )
        classes = "quick-action debug-control" if action.get("dry_run", False) else "quick-action"
        return (
            f"<section class=\"{classes}\">"
            f"<h3>{self._action_text(action, 'title')}</h3>"
            f"<p>{self._action_text(action, 'description')}</p>"
            f"{button}"
            "</section>"
        )

    def _mtime_date_for_report(self, rel):
        path = self.root / rel
        if not path.is_file():
            return ""
        try:
            return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        except OSError:
            return ""

    def _step_process_date(self, step):
        reports = step.get("result_reports", []) or step.get("reports", [])
        for report in reports:
            rel = report if isinstance(report, str) else report.get("path", "")
            if not rel:
                continue
            date = self._mtime_date_for_report(rel)
            if date:
                return date
        return step.get("process_date", "not produced")

    def _status_text(self, ready):
        return self._bi("就绪", "Ready") if ready else self._bi("缺失", "Missing")

    def _report_status_card(self, title_html, item, *, detail_href=None, description_html=""):
        ready = self._report_item_ready(item)
        state = "ready" if ready else "missing"
        rel = item.get("path", "")
        date = self._mtime_date_for_report(rel)
        date_html = (
            self._bi(f"更新: {date}", f"updated: {date}")
            if date
            else self._bi("未生成报告", "report not found")
        )
        href = detail_href or self._path_href(rel)
        description = f"<p>{description_html}</p>" if description_html else ""
        return (
            f"<article class=\"readiness-card {state}\">"
            "<div class=\"readiness-head\">"
            f"<h3>{title_html}</h3>"
            f"<span class=\"readiness-pill\">{self._status_text(ready)}</span>"
            "</div>"
            f"{description}"
            f"<p>{date_html}</p>"
            f"<a href=\"{html.escape(href)}\">{self._bi('打开', 'Open')}</a>"
            "</article>"
        )

    def _capture_text(self, dataset, key):
        zh = dataset.get(f"{key}_zh") or dataset.get(key, "")
        en = dataset.get(f"{key}_en") or dataset.get(key, zh)
        return self._bi(zh, en)

    def _render_readiness_overview(self):
        cards = []
        for dataset in CAPTURE_DATASETS:
            cards.append(
                self._report_status_card(
                    self._capture_text(dataset, "name"),
                    dataset["report"],
                    detail_href=f"/data-collect/{quote(dataset['slug'])}",
                    description_html=self._capture_text(dataset, "target"),
                )
            )
        final_item = {
            "label": "studio_32_cameras.yaml",
            "path": FINAL_STUDIO32_YAML,
            "kind": "final YAML",
            "status_if_exists": "ready",
            "status_if_missing": "not produced yet",
        }
        cards.append(
            self._report_status_card(
                self._bi("最终 YAML", "Final YAML"),
                final_item,
                detail_href=self._path_href(FINAL_STUDIO32_YAML),
                description_html=self._bi("下游重建 / SLAM / 3DGS 使用", "for reconstruction / SLAM / 3DGS"),
            )
        )
        return (
            "<section class=\"flow-wrap readiness-wrap\">"
            f"<h2>{self._bi('Readiness：当前数据与结果状态', 'Readiness: Current Data and Result Status')}</h2>"
            f"<p>{self._bi('这里直接读取当前报告和最终 YAML 是否已经生成。先让四类 Data Collect 都 Ready，再运行对应计算步骤。', 'This panel reads whether the current reports and final YAML exist. Make the four Data Collect entries Ready before running the matching processing steps.')}</p>"
            f"<div class=\"readiness-grid\">{''.join(cards)}</div>"
            "</section>"
        )

    def _render_capture_quality_list(self, dataset):
        zh_items = dataset.get("quality_zh", [])
        en_items = dataset.get("quality_en", zh_items)
        max_len = max(len(zh_items), len(en_items))
        items = []
        for idx in range(max_len):
            zh = zh_items[idx] if idx < len(zh_items) else en_items[idx]
            en = en_items[idx] if idx < len(en_items) else zh
            items.append(f"<li>{self._bi(zh, en)}</li>")
        if not items:
            return ""
        return f"<ul class=\"guidance-list\">{''.join(items)}</ul>"

    def _render_step_capture_dependencies(self, step):
        slugs = step.get("required_capture_slugs", [])
        if not slugs:
            return ""
        cards = []
        for slug in slugs:
            dataset = CAPTURE_DATASET_BY_SLUG.get(slug)
            if dataset is None:
                continue
            item = dataset["report"]
            ready = self._report_item_ready(item)
            state = "ready" if ready else "missing"
            date = self._mtime_date_for_report(item.get("path", ""))
            date_html = (
                self._bi(f"更新: {date}", f"updated: {date}")
                if date
                else self._bi("未生成 QC 报告", "QC report not found")
            )
            cards.append(
                f"<article class=\"dependency-card {state}\">"
                "<div class=\"readiness-head\">"
                f"<h3>{self._capture_text(dataset, 'name')}</h3>"
                f"<span class=\"readiness-pill\">{self._status_text(ready)}</span>"
                "</div>"
                f"<p><strong>{self._capture_text(dataset, 'target')}</strong></p>"
                f"<p>{self._capture_text(dataset, 'hardware')}</p>"
                f"<p>{date_html}</p>"
                "<div class=\"button-row\">"
                f"<a class=\"link-button\" href=\"/data-collect/{quote(slug)}\">"
                f"{self._bi('采集详情', 'Capture Details')}</a>"
                f"<a class=\"link-button\" href=\"{html.escape(self._path_href(item['path']))}\">"
                f"{self._bi('QC 报告', 'QC Report')}</a>"
                "</div>"
                "</article>"
            )
        return (
            "<section class=\"panel-card guidance-card\">"
            f"<div class=\"card-eyebrow\">{self._bi('采集依赖', 'Capture dependencies')}</div>"
            f"<h2>{self._bi('本步骤需要哪些 Data Collect', 'Required Data Collect')}</h2>"
            f"<p>{self._bi('先确认下面这些采集项 Ready，再运行本计算步骤。每个采集详情页包含数据路径、QC / 聚合按钮和采集报告入口。', 'Confirm the following capture entries are Ready before running this processing step. Each capture detail page includes data paths, QC / aggregate controls, and the capture report link.')}</p>"
            f"<div class=\"dependency-card-grid\">{''.join(cards)}</div>"
            "</section>"
        )

    def _render_data_collect_overview(self):
        cards = []
        for dataset in CAPTURE_DATASETS:
            href = f"/data-collect/{quote(dataset['slug'])}"
            cards.append(
                "<article class=\"capture-card\">"
                f"<span class=\"capture-target\">{self._capture_text(dataset, 'target')}</span>"
                f"<h3>{self._capture_text(dataset, 'name')}</h3>"
                f"<p>{self._capture_text(dataset, 'hardware')}</p>"
                f"<p>{self._capture_text(dataset, 'capture')}</p>"
                f"<code>{html.escape(dataset['raw_path'])}</code>"
                f"<a class=\"link-button\" href=\"{href}\">{self._bi('采集详情 / QC', 'Capture Details / QC')}</a>"
                "</article>"
            )
        return (
            "<section class=\"flow-wrap data-collect-wrap\">"
            f"<h2>{self._bi('Data Collect：4 类必需采集', 'Data Collect: 4 Required Captures')}</h2>"
            f"<p>{self._bi('本系统把采集和计算分开。先完成下面四类物理采集，再进入后面的 calibration steps。large、bridge 都使用大标定板，但用途不同：large 面向外圈内参，bridge 面向内外圈共同约束。', 'Capture and processing are separated. Complete these four physical captures before running the calibration steps. Both large and bridge use the large board, but with different purposes: large is for outer intrinsics, while bridge constrains inner and outer cameras together.')}</p>"
            f"<div class=\"capture-grid\">{''.join(cards)}</div>"
            "</section>"
        )

    def _render_workflow_graph(self):
        nodes = {node["slug"]: node for node in WORKFLOW_GRAPH_NODES}
        edge_labels_zh = {
            "outer K": "外圈 K",
            "inner K": "内圈 K",
            "outer pose": "外圈外参",
            "inner pose": "内圈外参",
            "final all32": "32 相机结果",
        }
        badge_labels_zh = {
            "K first": "先 K",
            "pose": "外参",
            "needs K+pose": "需要 K+外参",
            "current": "当前结果",
        }
        edges = []
        for source_slug, target_slug, label in WORKFLOW_GRAPH_EDGES:
            source = nodes[source_slug]
            target = nodes[target_slug]
            sx = source["x"] + source["w"]
            sy = source["y"] + source["h"] / 2
            tx = target["x"]
            ty = target["y"] + target["h"] / 2
            c1x = sx + max(75, (tx - sx) * 0.45)
            c2x = tx - max(75, (tx - sx) * 0.45)
            label_x = (sx + tx) / 2
            label_y = (sy + ty) / 2 - 8
            edges.append(
                f"<path class=\"graph-edge\" d=\"M {sx:.1f} {sy:.1f} "
                f"C {c1x:.1f} {sy:.1f}, {c2x:.1f} {ty:.1f}, {tx:.1f} {ty:.1f}\" "
                "marker-end=\"url(#arrowhead)\" />"
                f"{self._svg_bi_text('graph-edge-label', f'{label_x:.1f}', f'{label_y:.1f}', edge_labels_zh.get(label, label), label, anchor='middle')}"
            )

        rendered_nodes = []
        for node in WORKFLOW_GRAPH_NODES:
            step = WORKFLOW_BY_SLUG[node["slug"]]
            href = f"/operation/{quote(node['slug'])}"
            x = node["x"]
            y = node["y"]
            zh_lines = node.get("label_zh", node["label"])
            en_lines = node.get("label_en", node["label"])
            capture_date = step.get("capture_date", "unknown")
            process_date = self._step_process_date(step)
            badge_zh = badge_labels_zh.get(node["badge"], node["badge"])
            rendered_nodes.append(
                f"<a class=\"graph-node {html.escape(node['lane'])}\" href=\"{href}\">"
                f"<rect x=\"{x}\" y=\"{y}\" width=\"{node['w']}\" height=\"{node['h']}\" />"
                f"{self._svg_bi_text('graph-badge', x + 18, y + 25, badge_zh, node['badge'])}"
                f"{self._svg_bi_text('graph-title', x + 18, y + 54, zh_lines[0], en_lines[0])}"
                f"{self._svg_bi_text('graph-subtitle', x + 18, y + 78, zh_lines[1], en_lines[1])}"
                f"{self._svg_bi_text('graph-date', x + 18, y + 101, f'采集: {capture_date}', f'capture: {capture_date}')}"
                f"{self._svg_bi_text('graph-date', x + 18, y + 119, f'处理: {process_date}', f'process: {process_date}')}"
                "</a>"
            )

        return (
            "<div class=\"graph-scroll\">"
            "<svg class=\"dependency-graph\" viewBox=\"0 0 1395 565\" role=\"img\" "
            "aria-label=\"Studio calibration dependency graph\">"
            "<defs>"
            "<marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"8\" refX=\"9\" refY=\"4\" "
            "orient=\"auto\" markerUnits=\"strokeWidth\">"
            "<path d=\"M 0 0 L 10 4 L 0 8 z\" fill=\"#7d8794\" />"
            "</marker>"
            "</defs>"
            f"{self._svg_bi_text('graph-lane', 70, 65, '外圈分支', 'Outer branch')}"
            f"{self._svg_bi_text('graph-lane', 70, 300, '内圈分支', 'Inner branch')}"
            f"{self._svg_bi_text('graph-lane', 745, 182, '汇合', 'Merge')}"
            f"{self._svg_bi_text('graph-lane', 1080, 182, '发布', 'Publish')}"
            + "".join(edges)
            + "".join(rendered_nodes)
            + "</svg>"
            "</div>"
        )

    def _serve_console(self):
        viewer_url = self._path_href(UNIFIED_VIEWER)
        yaml_url = self._path_href(FINAL_STUDIO32_YAML)
        quick_actions = "".join(self._render_quick_action(action) for action in QUICK_ACTIONS)
        readiness = self._render_readiness_overview()
        data_collect = self._render_data_collect_overview()
        flow = self._render_workflow_graph()
        body = f"""
<header>
  <h1>{self._bi('Studio 相机标定控制台', 'Studio Camera Calibration Console')}</h1>
  <p>{self._bi('从这里查看当前 32 相机结果、下载 YAML，并按步骤运行标定流程。页面默认展示最新发布结果；重新处理数据前，请先确认对应采集报告和数据路径。', 'Review the current 32-camera result, download the YAML, and run calibration operations from this page. The page shows the latest published result by default; before reprocessing data, check the matching capture report and data paths first.')}</p>
  <div class="header-tools">{self._language_toggle()}</div>
</header>
<main>
  <section class="hero-grid">
    <div class="hero-card">
      <h2>{self._bi('当前结果', 'Current Result')}</h2>
      <p>{self._bi('先从整体 3D Viewer 检查 rig 结构、camera frustum、correspondence、intrinsic / extrinsic residual。确认结果可信后，再把 YAML 交给重建、SLAM 或 3DGS 流程。', 'Start with the 3D viewer to inspect rig layout, camera frustums, correspondences, and intrinsic / extrinsic residuals. After the result looks consistent, use the YAML for reconstruction, SLAM, or 3DGS.')}</p>
      <div class="hero-actions">
        <a class="link-button primary" href="{html.escape(viewer_url)}">{self._bi('打开 3D Viewer', 'Open 3D Viewer')}</a>
        <a class="link-button" href="{html.escape(yaml_url)}" download>{self._bi('下载 YAML', 'Download YAML')}</a>
      </div>
      <p><code>{html.escape(yaml_url)}</code></p>
    </div>
    <div class="hero-card">
      <h2>{self._bi('常用运行入口', 'Common Operations')}</h2>
	      <p>{self._bi('这些按钮会在 t0 上启动对应处理流程。“运行完整流程”用于整套数据重新处理；“快速重跑 Bridge”用于外圈不变、内圈相机移动后的快速更新。运行记录会写入：', 'These buttons start calibration processing on t0. Use Run Full Pipeline when reprocessing the full dataset; use Run Fast Bridge when the outer cage is unchanged and only the inner cameras moved. Job records are written to:')} <code>{html.escape(getattr(self.server, 'runs_root', DEFAULT_RUNS_ROOT))}</code></p>
	      <div class="quick-actions">{quick_actions}</div>
	      {self._debug_toggle()}
	      <div id="job-status" class="job-status">{self._bi('尚未从本页启动任务。', 'No job started from this page yet.')}</div>
    </div>
  </section>
  {readiness}
  {data_collect}
  <section class="flow-wrap">
    <h2>{self._bi('Calibration Steps：计算依赖图', 'Calibration Steps: Processing Dependency Graph')}</h2>
    <p>{self._bi('完成对应 Data Collect 之后，再按计算依赖运行这些步骤。内圈和外圈可以从任一分支开始；每条分支都先建立 intrinsic，再建立 extrinsic。整体 bridge / publish 依赖两边都有粗略可用的 K 和 pose。', 'After the matching Data Collect pages are complete, run these processing steps according to the dependency graph. The inner and outer branches can start independently; each branch builds intrinsics before extrinsics; bridge and publish depend on usable K and pose from both sides.')}</p>
    {flow}
  </section>
</main>
"""
        self._html_page("Studio Camera Calibration Console", body)

    def _render_report_links(self, reports):
        if not reports:
            return f"<p class=\"empty-note\">{self._bi('这个步骤没有单独的数据采集报告。', 'This step has no separate data-capture report.')}</p>"
        items = []
        for report in reports:
            if isinstance(report, str):
                rel = report
                label = Path(rel).name if not rel.endswith("/index.html") else "Open report"
                label_html = self._bi("打开报告", label) if rel.endswith("/index.html") else html.escape(label)
                kind = "report/artifact"
            else:
                rel = report["path"]
                label = report["label"]
                label_html = self._bi(report.get("label_zh", label), report.get("label_en", label))
                kind = report.get("kind", "report/artifact")
            item = {
                "label": label,
                "label_html": label_html,
                "path": rel,
                "kind": kind,
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            }
            items.append(self._render_report_item(item))
        return "<ul class=\"report-list\">" + "".join(items) + "</ul>"

    def _render_default_params_table(self, mode, overrides):
        params = self._mode_default_params(mode)
        params.update(overrides or {})
        rows = []
        for key in sorted(params):
            rows.append(
                "<tr>"
                f"<td><code>{html.escape(str(key))}</code></td>"
                f"<td><code>{html.escape(str(params[key]))}</code></td>"
                "</tr>"
            )
        return (
            "<table><thead><tr>"
            f"<th>{self._bi('参数', 'Parameter')}</th>"
            f"<th>{self._bi('值', 'Value')}</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    def _render_step_guidance(self, step):
        guidance = STEP_GUIDANCE.get(step["slug"])
        if not guidance:
            return ""
        zh_items = guidance.get("checks_zh", [])
        en_items = guidance.get("checks_en", zh_items)
        max_len = max(len(zh_items), len(en_items))
        items = []
        for idx in range(max_len):
            zh = zh_items[idx] if idx < len(zh_items) else en_items[idx]
            en = en_items[idx] if idx < len(en_items) else zh
            items.append(f"<li>{self._bi(zh, en)}</li>")
        return (
            "<section class=\"panel-card guidance-card\">"
            f"<div class=\"card-eyebrow\">{self._bi('操作说明', 'How to use this step')}</div>"
            f"<h2>{self._bi('这一步做什么', 'What this step does')}</h2>"
            f"<p>{self._bi(guidance.get('overview_zh', ''), guidance.get('overview_en', guidance.get('overview_zh', '')))}</p>"
            f"<h3>{self._bi('运行前后要检查什么', 'What to check before and after running')}</h3>"
            f"<ul class=\"guidance-list\">{''.join(items)}</ul>"
            "</section>"
        )

    def _data_collect_detail_body(self, dataset):
        step = WORKFLOW_BY_SLUG[dataset["qc_step_slug"]]
        qc_dry_run_button = self._run_button(
            "预演数据 QC / 聚合",
            self._capture_qc_action(step),
            dry_run=True,
            primary=False,
            confirm=False,
            debug=True,
            label_en="Preview Data QC / Aggregate",
        )
        qc_run_button = self._run_button(
            "运行数据 QC / 聚合",
            step,
            dry_run=False,
            primary=True,
            confirm=True,
            label_en="Run Data QC / Aggregate",
        )
        used_by = []
        for slug in dataset.get("used_by", []):
            used_step = WORKFLOW_BY_SLUG[slug]
            used_by.append(
                f"<li><a href=\"/operation/{quote(slug)}\">{self._step_text(used_step, 'title')}</a></li>"
            )
        data_paths = (
            f"<li><strong>{self._bi('Windows 原始数据', 'Windows raw data')}</strong>"
            f"<code>{html.escape(dataset['raw_path'])}</code></li>"
            f"<li><strong>{self._bi('t0 整理 / QC 数据', 't0 staged / QC data')}</strong>"
            f"<code>{html.escape(dataset['staged_path'])}</code></li>"
        )
        report_links = self._render_report_links([dataset["report"]])
        quality_list = self._render_capture_quality_list(dataset)
        body = f"""
<header class="operation-header">
  <h1>{self._capture_text(dataset, 'name')}</h1>
  <p>{self._capture_text(dataset, 'capture')}</p>
  <div class="header-tools">{self._language_toggle()}</div>
  <p class="mode-line"><a href="/">{self._bi('返回首页', 'Back to console')}</a></p>
</header>
<main>
  <section class="panel-card guidance-card">
    <div class="card-eyebrow">{self._bi('采集说明', 'Capture guide')}</div>
    <h2>{self._bi('这组数据用来做什么', 'What this capture is for')}</h2>
    <p>{self._bi('目标：', 'Target:')} <strong>{self._capture_text(dataset, 'target')}</strong></p>
    <p>{self._bi('使用设备：', 'Hardware:')} <strong>{self._capture_text(dataset, 'hardware')}</strong></p>
    <p>{self._capture_text(dataset, 'capture')}</p>
    <h3>{self._bi('合格标准', 'Acceptance Criteria')}</h3>
    {quality_list}
  </section>
  <section class="detail-grid">
    <div class="panel-card paths-card">
      <div class="card-eyebrow">{self._bi('路径', 'Paths')}</div>
      <h2>{self._bi('数据应放在哪里', 'Where the data should live')}</h2>
      <p>{self._bi('采集完成后，先确认 Windows 原始路径和 t0 staging / QC 路径。', 'After capture, first confirm both the Windows raw path and the t0 staging / QC path.')}</p>
      <ul class="path-list">{data_paths}</ul>
    </div>
    <div class="panel-card paths-card">
      <div class="card-eyebrow">{self._bi('下游步骤', 'Downstream steps')}</div>
      <h2>{self._bi('哪些标定步骤会使用它', 'Calibration steps using this capture')}</h2>
      <ul class="path-list">{''.join(used_by)}</ul>
    </div>
  </section>
  <section class="detail-grid" style="margin-top:14px;">
    <div class="step-detail">
      <div class="card-eyebrow">{self._bi('采集 QC / 聚合', 'Capture QC / Aggregate')}</div>
      <h2>{self._bi('运行数据 QC / 聚合', 'Run Data QC / Aggregate')}</h2>
      <p>{self._bi('先用这里检查同步、掉帧、尾帧裁剪、可用相机/帧和角点覆盖；确认通过后再回到 calibration steps 运行计算。', 'Use this first to check synchronization, dropped frames, tail trimming, usable cameras / frames, and feature coverage. After the capture looks valid, return to the calibration steps for processing.')}</p>
      <div class="button-row">{qc_dry_run_button}{qc_run_button}</div>
      {self._debug_toggle()}
      <div id="job-status" class="job-status">{self._bi('尚未从本页启动任务。', 'No job started from this page yet.')}</div>
    </div>
    <div class="panel-card report-card">
      <div class="card-eyebrow">{self._bi('采集证据', 'Capture evidence')}</div>
      <h2>{self._bi('数据采集 / QC 报告', 'Data Capture / QC Report')}</h2>
      <p>{self._bi('报告用于判断这组采集是否足够支撑后续标定。', 'Use this report to decide whether the capture is strong enough for downstream calibration.')}</p>
      {report_links}
    </div>
  </section>
</main>
"""
        return body

    def _serve_data_collect_detail(self, slug):
        dataset = CAPTURE_DATASET_BY_SLUG.get(slug)
        if dataset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = self._data_collect_detail_body(dataset)
        self._html_page(f"Data Collect: {dataset['name']}", body)

    def _operation_detail_body(self, step):
        dry_run_button = self._run_button(
            "预演本步骤", step, dry_run=True, primary=False, confirm=False, debug=True,
            label_en="Dry-run This Step",
        )
        run_button = self._run_button(
            "运行本步骤", step, dry_run=False, primary=True, confirm=True,
            label_en="Run This Step",
        )
        data_paths = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in step["data_paths"])
        output_paths = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in step["output_paths"])
        result_reports = self._render_report_links(step.get("result_reports", step.get("reports", [])))
        params = self._render_default_params_table(step["mode"], step.get("run_params", {}))
        guidance = self._render_step_guidance(step)
        capture_dependencies = self._render_step_capture_dependencies(step)
        body = f"""
<header class="operation-header">
  <h1>{self._step_text(step, 'title')}</h1>
  <p>{self._step_text(step, 'purpose')}</p>
  <div class="header-tools">{self._language_toggle()}</div>
  <p class="mode-line"><a href="/">{self._bi('返回首页', 'Back to console')}</a></p>
</header>
<main>
  {guidance}
  {capture_dependencies}
  <section class="detail-grid">
    <div class="panel-card paths-card">
      <div class="card-eyebrow">{self._bi('输入', 'Inputs')}</div>
      <h2>{self._bi('数据路径', 'Data Paths')}</h2>
      <p>{self._bi('这些路径描述本计算步骤会读取或更新的数据。采集 QC 已移动到首页 Data Collect 详情页。', 'These paths describe data read or refreshed by this processing step. Capture QC now lives on the Data Collect detail pages from the home page.')}</p>
      <ul class="path-list">{data_paths}</ul>
    </div>
    <div class="panel-card paths-card">
      <div class="card-eyebrow">{self._bi('输出', 'Outputs')}</div>
      <h2>{self._bi('输出路径', 'Output Paths')}</h2>
      <p>{self._bi('这些是本步骤会写入或更新的主要输出目录和 current artifacts。', 'These are the main output roots and current artifacts written or refreshed by this step.')}</p>
      <ul class="path-list">{output_paths}</ul>
    </div>
  </section>
  <section class="detail-grid" style="margin-top:14px;">
    <div class="step-detail">
      <div class="card-eyebrow">{self._bi('处理', 'Process')}</div>
      <h2>{self._bi('运行本步骤', 'Run This Step')}</h2>
      <p>{self._bi('确认对应 Data Collect 页面里的 QC、数据路径和当前报告后再运行。任务会在 t0 上写入新的运行目录。', 'Run this after checking the matching Data Collect QC page, data paths, and current report. The job writes a new run directory on t0.')}</p>
      <div class="button-row">{dry_run_button}{run_button}</div>
      <div id="job-status" class="job-status">{self._bi('尚未从本页启动任务。', 'No job started from this page yet.')}</div>
    </div>
    <div class="panel-card report-card">
      <div class="card-eyebrow">{self._bi('结果证据', 'Result evidence')}</div>
      <h2>{self._bi('标定结果报告', 'Calibration Result Report')}</h2>
      <p>{self._bi('运行后查看本步骤得到的内参、外参、bridge 结果和 residual。', 'After running, inspect this step’s intrinsics, extrinsics, bridge result, and residuals.')}</p>
      {result_reports}
    </div>
  </section>
  <section class="panel-card" style="margin-top:14px;">
    <h2>{self._bi('步骤说明', 'Step Notes')}</h2>
    <p>{self._step_text(step, 'notes')}</p>
  </section>
  <section class="panel-card" style="margin-top:14px;">
    <h2>{self._bi('高级运行参数', 'Advanced Run Parameters')}</h2>
    <p>{self._bi('这里列出本页面按钮会使用的默认运行参数，方便复核本次处理配置。', 'These are the default run parameters used by the page buttons, so the processing configuration can be reviewed before launch.')}</p>
    {params}
  </section>
</main>
"""
        return body

    def _serve_operation_detail(self, slug):
        step = WORKFLOW_BY_SLUG.get(slug)
        if step is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = self._operation_detail_body(step)
        self._html_page(step["title"], body)

    def _serve_index(self):
        # Backward-compatible fallback for tests and old callers.
        report_groups = "".join(self._render_report_group(group, "") for group in REPORT_GROUPS)
        body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Calibration Reports</title>
  <style>
    :root {{
      --ink: #1f2328;
      --muted: #57606a;
      --line: #d0d7de;
      --soft: #f6f8fa;
      --blue: #0969da;
      --green: #1a7f37;
      --amber: #9a6700;
      --violet: #8250df;
      --red: #cf222e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{ border-bottom: 1px solid var(--line); padding: 28px 32px 22px; }}
    main {{ padding: 24px 32px 40px; }}
    h1 {{ font-size: 25px; margin: 0 0 8px; }}
    h2 {{ font-size: 17px; margin: 0; }}
    p {{ color: var(--muted); line-height: 1.45; }}
    .subtitle {{ margin: 5px 0 0; font-size: 13px; color: var(--muted); }}
    .tools {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
    .tool-link {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 4px 14px;
      min-width: 280px;
      max-width: 460px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--ink);
      text-decoration: none;
    }}
    .tool-link:hover {{ border-color: var(--blue); text-decoration: none; }}
    .tool-link span {{ color: var(--blue); font-size: 12px; text-transform: uppercase; }}
    .tool-link small {{ grid-column: 1 / -1; color: var(--muted); line-height: 1.35; }}
    .report-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      align-items: stretch;
    }}
    .report-group {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-height: 300px;
    }}
    .report-group.available {{ border-top: 4px solid var(--green); }}
    .report-group.pipeline {{ border-top: 4px solid var(--violet); }}
    .report-group.partial {{ border-top: 4px solid var(--amber); }}
    .report-group.missing {{ border-top: 4px solid var(--red); }}
    .report-group.diagnostic {{ border-top: 4px solid var(--amber); }}
    .group-head {{ display: flex; gap: 12px; justify-content: space-between; align-items: flex-start; }}
    .status-pill {{
      flex: 0 0 auto;
      border-radius: 999px;
      padding: 3px 9px;
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .available .status-pill {{ color: var(--green); }}
    .pipeline .status-pill {{ color: var(--violet); }}
    .partial .status-pill, .diagnostic .status-pill {{ color: var(--amber); }}
    .missing .status-pill {{ color: var(--red); }}
    .pipeline-action {{
      display: inline-flex;
      margin-top: 4px;
      border: 1px solid var(--blue);
      border-radius: 6px;
      padding: 7px 10px;
      color: var(--blue);
      font-size: 13px;
      font-weight: 650;
    }}
    .pipeline-action:hover {{ background: #ddf4ff; text-decoration: none; }}
    .report-list, .notes {{ margin: 14px 0 0; padding: 0; list-style: none; }}
    .report-item {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 6px;
      border-top: 1px solid var(--line);
      padding: 10px 0;
    }}
    .report-item:first-child {{ border-top: 0; padding-top: 0; }}
    .report-item small {{ display: block; margin-top: 3px; color: var(--muted); }}
    .report-item.ready small {{ color: var(--green); }}
    .report-item.diagnostic small {{ color: var(--amber); }}
    .report-item.missing strong {{ color: var(--red); }}
    .report-item code {{
      display: block;
      overflow-wrap: anywhere;
      background: var(--soft);
      padding: 5px 6px;
      border-radius: 5px;
      color: var(--muted);
      font-size: 12px;
    }}
    .notes li {{ margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.35; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; }}
    th {{ background: var(--soft); }}
    details {{ margin-top: 26px; }}
    summary {{ cursor: pointer; color: var(--muted); }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    header code, main > p code {{ background: var(--soft); padding: 2px 4px; border-radius: 4px; }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .report-grid {{ grid-template-columns: 1fr; }}
      .tool-link {{ min-width: 0; width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Camera Calibration Reports</h1>
    <p>最终报告入口只展示一个 overall viewer 和七个 canonical reports。所有 report href 使用完整 9899 URL。服务根目录: <code>{html.escape(str(self.root))}</code>。</p>
    {tools_html}
  </header>
  <main>
    <div class="report-grid">{report_groups}</div>
  </main>
</body>
</html>
"""
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9899)
    parser.add_argument("--public-url", default=DEFAULT_REPORT_BASE_URL)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    repo_root = Path(args.repo_root).resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()

    def handler(*handler_args, **handler_kwargs):
        return ReportHandler(*handler_args, directory=str(root), **handler_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.report_base_url = args.public_url
    server.runs_root = str(runs_root)
    if JobManager is not None:
        server.job_manager = JobManager(
            repo_root=repo_root,
            runs_root=runs_root,
            python_bin=args.python_bin,
        )
    else:
        server.job_manager = None
    print(
        f"Serving calibration console {root} on http://{args.host}:{args.port}/ "
        f"(repo={repo_root}, runs={runs_root})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
