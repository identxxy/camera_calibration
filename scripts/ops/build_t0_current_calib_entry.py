#!/usr/bin/env python3
"""Build a clean current calibration entry page for t0 reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import time
from urllib.parse import quote


DEFAULT_ROOT = "/home/ubuntu/calib_data"
DEFAULT_BASE_URL = "http://192.168.2.0:9899"
DEFAULT_PANEL_URL = "http://192.168.2.0:9898/"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/calib_data/current_calibration"
INNER_SUMMARY_REL = (
    "studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/"
    "inner_bridge/summary.json"
)
BRIDGE_SUMMARY_REL = (
    "studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/"
    "inner_bridge/summary.json"
)
CURRENT_BRIDGE_RUN_REL = (
    "studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/"
    "inner_bridge"
)
CURRENT_OUTER_RUN_REL = (
    "studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/"
    "outer_tower/frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1"
)
CURRENT_OUTER_REPORT_REL = CURRENT_OUTER_RUN_REL
CURRENT_STUDIO32_YAML_REL = (
    "studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/"
    "calibration_artifacts/"
    "studio_32_cameras_current/studio_32_cameras.yaml"
)


LINKS = {
    "overall_viewer": (
        "Unified 3D viewer",
        f"{CURRENT_BRIDGE_RUN_REL}/combined_studio_rig_viewer_v1/index.html",
    ),
    "studio32_yaml": (
        "studio_32_cameras.yaml",
        CURRENT_STUDIO32_YAML_REL,
    ),
    "inner_viewer": (
        "Inner8 viewer",
        "calib_2026_05_26_jpg_v3/final_inner8_calibration_v1/reports/"
        "interactive_rig_viewer_v1/index.html",
    ),
    "outer_viewer": (
        "Outer24 viewer",
        f"{CURRENT_BRIDGE_RUN_REL}/combined_studio_rig_viewer_v1/index.html",
    ),
    "whole_data_collection": (
        "Whole 数据采集报告",
        "calib_2026_05_31_v3/whole_outer24_filtered_min4_hybrid_min4cam/index.html",
    ),
    "whole_distributed_qc": (
        "Whole 分布式识别日志",
        "calib_2026_05_31_v3/distributed_qc/index.html",
    ),
    "whole_final": (
        "Whole / Outer 最终标定报告",
        f"{CURRENT_OUTER_REPORT_REL}/index.html",
    ),
    "outer_summary": (
        "Outer solve summary.json",
        f"{CURRENT_OUTER_RUN_REL}/summary.json",
    ),
    "large_data_collection": (
        "Large Marker 数据采集报告",
        f"{CURRENT_BRIDGE_RUN_REL}/quality_report/index.html",
    ),
    "large_final": (
        "Large Marker Bridge 最终标定报告",
        f"{CURRENT_BRIDGE_RUN_REL}/final_report/index.html",
    ),
    "inner_final": (
        "Inner/bridge final report",
        f"{CURRENT_BRIDGE_RUN_REL}/final_report/index.html",
    ),
    "bridge_summary": (
        "Bridge summary.json",
        f"{CURRENT_BRIDGE_RUN_REL}/summary.json",
    ),
    "small_data_collection": (
        "Small Marker 数据采集报告",
        "calib_2026_05_26_jpg_v3/small_marker_inner8/"
        "coverage_gate_pattern3_v1/coverage_report.html",
    ),
    "small_final": (
        "Small Marker 最终标定报告",
        "calib_2026_05_26_jpg_v3/final_inner8_calibration_v1/reports/"
        "report_small_grid4_refined_reprojection_v1/index.html",
    ),
    "outer_quality": (
        "2026-05-31 outer24 quality",
        "calib_2026_05_31_v3/whole_outer24_filtered_min4_hybrid_min4cam/index.html",
    ),
    "outer_coverage": (
        "2026-05-31 outer24 coverage",
        "calib_2026_05_31_v3/whole_outer24_filtered_min4_hybrid_min4cam/"
        "opencv_tower_dataset_fullres_coverage/coverage_report.html",
    ),
}


CANONICAL_REPORT_CATEGORIES = [
    {
        "id": "inner_capture_qc",
        "label": "1. Inner Capture QC",
        "purpose": (
            "small_marker 和 large_marker calib board 的内圈采集质量入口。这里回答 "
            "inner8 是否有足够同步帧、角点 / TAG 检出、可用相机集合和 staging 是否可信。"
        ),
        "items": [
            {"link": "small_data_collection", "role": "small_marker calib board 数据采集报告"},
            {"link": "large_data_collection", "role": "large_marker calib board / bridge 输入采集报告"},
        ],
        "notes": [
            "该类只覆盖采集 / staging / 观测质量，不承诺最终 inner solve 已通过。",
            "large_marker QC 同时服务 inner baseline 和 inner/outer bridge 输入检查。",
        ],
    },
    {
        "id": "inner_solve_result",
        "label": "2. Inner Solve Result",
        "purpose": (
            "inner8 的最终可视化和 reprojection 质量入口。这里回答内圈相机内参、外参、"
            "distortion convention 和 per-camera residual 是否可用于后续 bridge / SLAM。"
        ),
        "items": [
            {"link": "small_final", "role": "inner reprojection / refined calibration report"},
            {"link": "inner_final", "role": "inner/bridge wrapper final report"},
        ],
        "notes": [
            "该类是 solve result，不再混入采集入口或 operation 按钮。",
            "source/debug viewers 不提升为首页入口，除非被 registry 标记为 canonical。",
        ],
    },
    {
        "id": "outer_capture_qc",
        "label": "3. Outer Capture QC",
        "purpose": (
            "whole / tower AprilTag 采集质量入口。这里回答 outer cameras 的同步帧、"
            "tag coverage、accepted frame set 和本轮采集是否足以进入 outer solve。"
        ),
        "items": [
            {"link": "whole_data_collection", "role": "whole/tower AprilTag 数据采集报告"},
        ],
        "notes": [
            "分布式日志和 coverage 细节只保留在 run directory / registry diagnostics，不提升为首页主入口。",
            "tail-trim 与单相机掉帧的处理规则在 staging 阶段完成。",
        ],
    },
    {
        "id": "outer_solve_diagnostics_result",
        "label": "4. Outer Solve Diagnostics / Result",
        "purpose": (
            "outer tower frame-face refine 与 COLMAP audit 诊断入口。这里回答 outer24 "
            "优化结果、reprojection residual、异常相机和 frame-face / prior consistency。"
        ),
        "items": [
            {"link": "whole_final", "role": "outer tower frame-face refine / final report"},
            {"link": "outer_summary", "role": "outer solve diagnostics summary"},
        ],
        "notes": [
            "COLMAP / side-prior / frame-face 诊断留在 run directory，不再作为首页单独入口散开。",
            "report inventory 是维护工具，不是 production 标定结果，默认不出现在首页。",
        ],
    },
    {
        "id": "combined_bridge_32_camera_result",
        "label": "5. Combined Bridge / 32-Camera Result",
        "purpose": (
            "inner + outer bridge 和最终 32-camera artifact 的统一入口。这里回答 "
            "combined rig 是否一致，以及算法消费的 studio_32_cameras.yaml 在哪里。"
        ),
        "items": [
            {"link": "overall_viewer", "role": "unified inner+outer 3D viewer"},
            {"link": "studio32_yaml", "role": "machine-readable unified 32-camera YAML"},
            {"link": "bridge_summary", "role": "inner/outer bridge summary"},
        ],
        "notes": [
            "最终 3D 查看只提升一个 unified viewer；inner-only / outer-only 应作为 viewer 内模式。",
            "studio_32_cameras.yaml 是下游算法消费的 canonical camera model artifact。",
        ],
    },
]


OPERATION_GROUPS = [
    {
        "id": "whole",
        "label": "Whole",
        "purpose": (
            "Whole 的主要目的，是标定整体的 studio cage，也就是 outer cameras / "
            "outer24 笼子。数据采集报告看各机器保存图片后本地识别 TAG 的质量和筛选结果；"
            "最终标定报告看基于筛选图片优化出的 outer/studio 标定结果。"
        ),
    },
    {
        "id": "large_marker",
        "label": "Large Marker",
        "purpose": (
            "Large Marker 的主要目的，是 bridge inner cameras 和 outer cameras。当前包含 "
            "large_marker_inner8 作为 inner baseline，以及 large_marker_bridge_all32 "
            "用于 all32 inner/outer bridge。"
        ),
    },
    {
        "id": "small_marker",
        "label": "Small Marker",
        "purpose": (
            "Small Marker 的主要目的，是标定 inner cameras。它只服务 inner cameras，用于 "
            "inner intrinsics、inner reprojection quality，以及 inner-only quality probe。"
        ),
    },
]


OPERATION_DEFINITIONS = {
    "whole": {
        "label": "Whole Operation",
        "panel_mode": "operate_whole_outer_cage",
        "summary": (
            "采集 whole 数据后，处理 outer camera cage：先做分布式 / 本地 TAG 识别质量检查，"
            "再筛选可用观测并运行 outer24 标定优化、viewer 和 final report。"
        ),
        "current_backend": "scripts/calib/run_outer_tower_recalib_pipeline.py",
        "target_cli": (
            "t0-calib operate whole --capture-root <whole_capture_root> "
            "--output-root <run_output_root> --publish-current"
        ),
        "steps": [
            "Stage / validate whole capture and camera manifest.",
            "Run tag detection coverage and accepted-frame selection.",
            "Run outer cage optimization and quality gates.",
            "Generate data collection report, final calibration report, and outer/final viewer artifacts.",
            "Publish promoted artifacts into current_calibration/report_registry.json.",
        ],
        "notes": [
            "这个 operation 的产品目标是 outer24 / studio cage 标定。",
            "当前 panel mode 已经是用户语义 alias；底层仍调用 outer tower pipeline wrapper。",
        ],
    },
    "large_marker": {
        "label": "Large Marker Operation",
        "panel_mode": "operate_large_marker_bridge",
        "summary": (
            "采集 large marker 数据后，处理 inner-to-outer bridge：检查 large_marker_inner8 "
            "和 large_marker_bridge_all32 的可用观测，先做 all32 PnP initializer，再运行 "
            "all32 fixed-known-point joint BA，并生成 combined viewer。"
        ),
        "current_backend": "scripts/calib/run_inner_bridge_recalib_pipeline.py",
        "target_cli": (
            "t0-calib operate large-marker --inner-sequence <large_marker_inner8> "
            "--bridge-sequence <large_marker_bridge_all32> --publish-current"
        ),
        "steps": [
            "Validate large-marker inner8 and all32 bridge capture contracts.",
            "Run large-marker feature extraction / PnP initializer for bridge inputs.",
            "Run all32 joint BA with known board points fixed and export post-BA correspondence residuals.",
            "Generate bridge final report and combined inner+outer viewer artifacts.",
            "Publish promoted bridge artifacts into current_calibration/report_registry.json.",
        ],
        "notes": [
            "这个 operation 的产品目标是 bridge inner cameras 和 outer cameras。",
            "4-* top-down cameras are metadata / diagnostics only; production bridge quality is post-BA reprojection residual.",
            "当前底层 wrapper 还同时包含 small-marker probe；后续 clean CLI 可把 bridge 和 small-inner 分开。",
        ],
    },
    "small_marker": {
        "label": "Small Marker Operation",
        "panel_mode": "operate_small_marker_inner",
        "summary": (
            "采集 small marker 数据后，处理 inner camera calibration：检查 inner8 覆盖率、"
            "运行 inner calibration / quality gates，并生成 inner final report 和 inner-only viewer。"
        ),
        "current_backend": "scripts/calib/run_inner_bridge_recalib_pipeline.py",
        "target_cli": (
            "t0-calib operate small-marker --inner-sequence <small_marker_inner8> "
            "--output-root <run_output_root> --publish-current"
        ),
        "steps": [
            "Validate small-marker inner8 capture and accepted frames.",
            "Run inner camera calibration / fixed-rig quality gates.",
            "Report per-camera intrinsics, extrinsics, distortion, and reprojection quality.",
            "Generate small-marker final report and inner-only viewer artifacts.",
            "Publish promoted inner artifacts into current_calibration/report_registry.json.",
        ],
        "notes": [
            "这个 operation 的产品目标是 inner camera calibration。",
            "当前已有历史 final inner report；但 clean CLI 需要明确 small-marker 是否能作为 production inner baseline。",
        ],
    },
}


STANDARD_REPORT_DRAFT = [
    {
        "label": "Whole / Outer Cage",
        "items": [
            "每台机器、每个 outer camera 的采集帧数、TAG 检出率、可用帧集合。",
            "outer24 优化后的 pose / intrinsics 版本、reprojection residual、被拒绝帧和被拒绝 camera。",
            "outer cage 的几何 sanity check：相机朝向、top-down cameras、环形一致性、最终 outer-only viewer。",
        ],
    },
    {
        "label": "Small Marker / Inner",
        "items": [
            "inner8 每个 camera 的 small marker 覆盖率、角点数量、accepted frames。",
            "inner intrinsics / extrinsics、distortion sanity、per-camera reprojection residual。",
            "弱相机或 disconnected camera 的明确结论，以及最终 inner-only viewer。",
        ],
    },
    {
        "label": "Large Marker / Bridge",
        "items": [
            "bridge input contract：outer / inner camera index order、accepted frames、all32 correspondence count。",
            "all32 PnP initializer、fixed-known-point joint BA、post-BA reprojection residual。",
            "bridge/outer alignment diagnostic、caveats、以及最终 combined viewer。",
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--panel-url", default=DEFAULT_PANEL_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--current-bridge-run-rel",
        default=CURRENT_BRIDGE_RUN_REL,
        help="Report-root-relative path to the promoted fast_inner_bridge run.",
    )
    parser.add_argument(
        "--current-outer-run-rel",
        default=CURRENT_OUTER_RUN_REL,
        help="Report-root-relative path to the promoted outer frame-face solve run.",
    )
    parser.add_argument(
        "--current-outer-report-rel",
        default=None,
        help=(
            "Report-root-relative path to the promoted outer HTML report directory. "
            "Defaults to --current-outer-run-rel for backward compatibility."
        ),
    )
    parser.add_argument(
        "--whole-data-report-rel",
        default=LINKS["whole_data_collection"][1],
        help="Report-root-relative path to the promoted whole data collection report.",
    )
    parser.add_argument(
        "--whole-distributed-qc-rel",
        default=LINKS["whole_distributed_qc"][1],
        help="Report-root-relative path to the promoted whole distributed QC report.",
    )
    parser.add_argument(
        "--studio32-yaml-rel",
        default=None,
        help="Report-root-relative path to the promoted studio_32_cameras.yaml artifact.",
    )
    parser.add_argument(
        "--write-root-index",
        action="store_true",
        help="Also write root/index.html so the report is available at the bare report server URL.",
    )
    return parser.parse_args()


def configure_current_run_paths(
    bridge_run_rel: str,
    outer_run_rel: str,
    outer_report_rel: str | None = None,
    whole_data_report_rel: str | None = None,
    whole_distributed_qc_rel: str | None = None,
    studio32_yaml_rel: str | None = None,
):
    global INNER_SUMMARY_REL
    global BRIDGE_SUMMARY_REL
    global CURRENT_BRIDGE_RUN_REL
    global CURRENT_OUTER_RUN_REL
    global CURRENT_OUTER_REPORT_REL
    global CURRENT_STUDIO32_YAML_REL
    global LINKS

    CURRENT_BRIDGE_RUN_REL = str(bridge_run_rel).strip("/")
    CURRENT_OUTER_RUN_REL = str(outer_run_rel).strip("/")
    CURRENT_OUTER_REPORT_REL = str(outer_report_rel or outer_run_rel).strip("/")
    whole_data_report_rel = str(whole_data_report_rel or LINKS["whole_data_collection"][1]).strip("/")
    whole_distributed_qc_rel = str(whole_distributed_qc_rel or LINKS["whole_distributed_qc"][1]).strip("/")
    if studio32_yaml_rel:
        CURRENT_STUDIO32_YAML_REL = str(studio32_yaml_rel).strip("/")
    elif CURRENT_BRIDGE_RUN_REL.endswith("/inner_bridge"):
        CURRENT_STUDIO32_YAML_REL = (
            f"{str(Path(CURRENT_BRIDGE_RUN_REL).parent).strip('/')}/"
            "calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml"
        )
    else:
        CURRENT_STUDIO32_YAML_REL = (
            f"{CURRENT_BRIDGE_RUN_REL}/calibration_artifacts/"
            "studio_32_cameras_current/studio_32_cameras.yaml"
        )
    INNER_SUMMARY_REL = f"{CURRENT_BRIDGE_RUN_REL}/summary.json"
    BRIDGE_SUMMARY_REL = f"{CURRENT_BRIDGE_RUN_REL}/summary.json"
    LINKS.update({
        "overall_viewer": (
            "Unified 3D viewer",
            f"{CURRENT_BRIDGE_RUN_REL}/combined_studio_rig_viewer_v1/index.html",
        ),
        "studio32_yaml": (
            "studio_32_cameras.yaml",
            CURRENT_STUDIO32_YAML_REL,
        ),
        "outer_viewer": (
            "Outer24 viewer",
            f"{CURRENT_BRIDGE_RUN_REL}/combined_studio_rig_viewer_v1/index.html",
        ),
        "whole_final": (
            "Whole / Outer 最终标定报告",
            f"{CURRENT_OUTER_REPORT_REL}/index.html",
        ),
        "outer_summary": (
            "Outer solve summary.json",
            f"{CURRENT_OUTER_RUN_REL}/summary.json",
        ),
        "whole_data_collection": (
            "Whole 数据采集报告",
            whole_data_report_rel,
        ),
        "whole_distributed_qc": (
            "Whole 分布式识别日志",
            whole_distributed_qc_rel,
        ),
        "large_data_collection": (
            "Large Marker 数据采集报告",
            f"{CURRENT_BRIDGE_RUN_REL}/quality_report/index.html",
        ),
        "large_final": (
            "Large Marker Bridge 最终标定报告",
            f"{CURRENT_BRIDGE_RUN_REL}/final_report/index.html",
        ),
        "inner_final": (
            "Inner/bridge final report",
            f"{CURRENT_BRIDGE_RUN_REL}/final_report/index.html",
        ),
        "bridge_summary": (
            "Bridge summary.json",
            BRIDGE_SUMMARY_REL,
        ),
    })


def url(base_url: str, rel_path: str) -> str:
    quoted = "/".join(quote(part) for part in rel_path.split("/"))
    return f"{base_url.rstrip('/')}/{quoted}"


def panel_mode_url(panel_url: str, mode: str) -> str:
    separator = "&" if "?" in panel_url else "?"
    return f"{panel_url.rstrip('/')}/{separator}mode={quote(mode)}"


def read_json(path: Path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def link_state(root: Path, base_url: str, rel_path: str):
    path = root / rel_path
    return {
        "rel_path": rel_path,
        "url": url(base_url, rel_path),
        "exists": path.is_file(),
    }


def summarize_data_quality(summary):
    rows = []
    for key, label in [
        ("small_marker", "Small marker inner8"),
        ("large_inner_marker", "Large marker inner8"),
        ("large_marker", "Large marker all32 bridge"),
    ]:
        item = (summary.get("data_quality") or {}).get(key, {})
        rows.append({
            "label": label,
            "status": item.get("status", "missing"),
            "usable": item.get("usable_camera_count"),
            "camera_count": item.get("camera_count"),
            "frames": item.get("common_frame_count"),
            "spread": item.get("frame_count_spread"),
            "warning": item.get("drop_frame_warning") or "",
        })
    return rows


def summarize_stages(summary):
    rows = []
    for stage in summary.get("stages", []):
        rows.append({
            "name": stage.get("name", ""),
            "group": stage.get("group", ""),
            "status": stage.get("status", ""),
            "returncode": stage.get("returncode"),
            "allow_failure": bool(stage.get("allow_failure")),
            "notes": stage.get("notes", []),
        })
    return rows


def inner_audit(root: Path, base_url: str):
    summary = read_json(root / INNER_SUMMARY_REL)
    bridge = read_json(root / BRIDGE_SUMMARY_REL)
    final = summary.get("final_yaml_candidates") or {}
    bridge_quality = summary.get("bridge_quality") or {}
    bridge_correspondence_quality = summary.get("bridge_correspondence_quality") or {}
    small_summary_path = Path(final.get("small_fixed_rig_quality_camera_pnp_summary", ""))
    large_inner_summary_path = Path(final.get("large_inner_init_state_dir", "")) / "camera_pnp_summary.tsv"
    large_bridge_summary_path = (
        root / CURRENT_BRIDGE_RUN_REL
        / "large_marker_bridge_all32/fixed_intrinsic_bridge_pnp_stride1_v1/camera_pnp_summary.tsv"
    )
    small_rows = read_tsv(small_summary_path)
    large_inner_rows = read_tsv(large_inner_summary_path)
    large_bridge_rows = read_tsv(large_bridge_summary_path)
    small_disconnected = [row.get("user_id") for row in small_rows if row.get("connected") != "yes"]
    large_bridge_disconnected = [row.get("user_id") for row in large_bridge_rows if row.get("connected") != "yes"]
    stage_rows = summarize_stages(summary)
    stage_failures = [
        row for row in stage_rows
        if row["status"] in {"failed", "failed_allowed"}
    ]
    contracts = (final.get("input_contracts") or {})
    all_contracts_ready = all(
        contracts.get(name, {}).get("ready")
        for name in ["small_marker_inner8", "large_marker_inner8", "large_marker_bridge_all32"]
    )
    large_inner_connected = sum(1 for row in large_inner_rows if row.get("connected") == "yes")
    inner_baseline_state = Path(final.get("inner_final_baseline_state_dir", ""))
    inner_baseline_ready = inner_baseline_state.is_dir()
    bridge_ba_ready = (
        bridge_correspondence_quality.get("status") == "present"
        and int(bridge_correspondence_quality.get("ok_count") or 0) > 0
    )
    status = "usable_with_caveats"
    if not all_contracts_ready or not bridge_ba_ready or not inner_baseline_ready:
        status = "needs_attention"
    return {
        "status": status,
        "summary": summary,
        "bridge": bridge,
        "contracts_ready": all_contracts_ready,
        "inner_baseline_ready": inner_baseline_ready,
        "data_quality": summarize_data_quality(summary),
        "stages": stage_rows,
        "stage_failures": stage_failures,
        "small_disconnected": small_disconnected,
        "large_inner_connected": large_inner_connected,
        "large_bridge_connected": sum(1 for row in large_bridge_rows if row.get("connected") == "yes"),
        "large_bridge_disconnected": large_bridge_disconnected,
        "bridge_ba_ready": bridge_ba_ready,
        "bridge_quality": bridge_quality,
        "bridge_correspondence_quality": bridge_correspondence_quality,
        "final": final,
        "links": {
            name: {"label": label, **link_state(root, base_url, rel)}
            for name, (label, rel) in LINKS.items()
        },
    }


def esc(value):
    return html.escape("" if value is None else str(value))


def render_link_card(label, href, description, status="ready"):
    return (
        f"<a class='card link-card {esc(status)}' href='{esc(href)}'>"
        f"<strong>{esc(label)}</strong><span>{esc(description)}</span></a>"
    )


def render_quality_rows(rows):
    return "\n".join(
        "<tr>"
        f"<td>{esc(row['label'])}</td><td>{esc(row['status'])}</td>"
        f"<td>{esc(row['usable'])}/{esc(row['camera_count'])}</td>"
        f"<td>{esc(row['frames'])}</td><td>{esc(row['spread'])}</td>"
        f"<td>{esc(row['warning'] or '-')}</td>"
        "</tr>"
        for row in rows
    )


def render_stage_rows(rows):
    return "\n".join(
        "<tr>"
        f"<td>{esc(row['group'])}</td><td>{esc(row['name'])}</td>"
        f"<td><span class='badge {esc(row['status'])}'>{esc(row['status'])}</span></td>"
        f"<td>{esc(row['returncode'])}</td><td>{esc(row['allow_failure'])}</td>"
        f"<td>{esc('; '.join(row['notes']) or '-')}</td>"
        "</tr>"
        for row in rows
    )


def render_topdown_rows(rows):
    return "\n".join(
        "<tr>"
        f"<td>{esc(row.get('user_id'))}</td><td>{esc(row.get('connected'))}</td>"
        f"<td>{esc(row.get('positive_views'))}</td><td>{esc(row.get('solved_views'))}</td>"
        f"<td>{esc(row.get('median_view_error_px'))}</td>"
        "</tr>"
        for row in rows
    )


def render_disconnected(labels):
    if not labels:
        return "-"
    return ", ".join(labels)


def render_report_category(category, links):
    cards = []
    for item in category["items"]:
        link = links[item["link"]]
        cards.append(
            render_link_card(
                link["label"],
                link["url"],
                item["role"],
                "ready" if link["exists"] else "missing",
            )
        )
    notes = "".join(f"<li>{esc(note)}</li>" for note in category.get("notes", []))
    return f"""
<section class="report-section">
  <div class="section-copy">
    <h2>{esc(category['label'])}</h2>
    <p>{esc(category['purpose'])}</p>
    <ul class="notes">{notes}</ul>
  </div>
  <div class="slot-column">
    <div class="slot-grid">
      {"".join(cards)}
    </div>
  </div>
</section>
"""


def render_operation_links(base_url, panel_url, output_rel):
    cards = [
        render_link_card(
            "Full Pipeline",
            panel_mode_url(panel_url, "run_studio_calibration_pipeline"),
            "已完成 QC/stage 后，运行当前 32-camera production wrapper",
            "ready",
        )
    ]
    for operation in OPERATION_GROUPS:
        definition = OPERATION_DEFINITIONS[operation["id"]]
        operation_url = f"{base_url.rstrip('/')}/{output_rel}/operations/{operation['id']}.html"
        cards.append(
            render_link_card(
                definition["label"],
                operation_url,
                operation["purpose"],
                "ready",
            )
        )
    return f"""
<section class="operation-section">
  <div class="section-copy">
    <h2>采集后处理入口</h2>
    <p>采完数据后从这里进入受控 9898 panel。Full Pipeline 是当前推荐的可复现整链路入口；单类 Operation 用于只处理 whole、large_marker 或 small_marker。</p>
  </div>
  <div class="slot-grid">
    {"".join(cards)}
  </div>
</section>
"""


CURRENT_REPORT_ENTRIES = [
    {
        "title": "Overall 3D Viewer",
        "href": "reports/01_3d_viewer/index.html",
        "description": (
            "Unified 32-camera viewer. It includes camera-set filters, dataset coverage, "
            "correspondence loading, intrinsic residuals, and final dataset/extrinsic residuals."
        ),
        "kind": "viewer",
    },
    {
        "title": "Inner Capture Report",
        "href": "reports/02_inner_capture_small_marker/index.html",
        "description": "Small-marker inner8 capture/staging quality and usable observation coverage.",
        "kind": "report",
    },
    {
        "title": "Inner Intrinsic Report",
        "href": "reports/03_inner_intrinsics_small_marker/index.html",
        "description": "Inner8 feature accumulation, reprojection residuals, and per-camera intrinsic quality.",
        "kind": "report",
    },
    {
        "title": "Inner Extrinsic Report",
        "href": "reports/04_inner_extrinsics_small_marker/index.html",
        "description": "Inner8 rig layout and final inner extrinsic consistency.",
        "kind": "report",
    },
    {
        "title": "Outer Capture Report",
        "href": "reports/05_outer_capture_outer_large_marker_whole/index.html",
        "description": "Outer-large-marker intrinsic capture plus whole/tower extrinsic capture QC.",
        "kind": "report",
    },
    {
        "title": "Outer Intrinsic Report",
        "href": "reports/06_outer_intrinsics_outer_large_marker/index.html",
        "description": "Outer24 large-marker feature accumulation, residuals, and intrinsic quality.",
        "kind": "report",
    },
    {
        "title": "Outer Extrinsic Report",
        "href": "reports/07_outer_extrinsics_whole/index.html",
        "description": "Whole/tower outer24 extrinsic refinement residuals and accepted observation summary.",
        "kind": "report",
    },
    {
        "title": "Bridge Result Report",
        "href": "reports/09_bridge_result_large_marker/index.html",
        "description": "Large-marker all-camera inner/outer bridge result and final consistency checks.",
        "kind": "report",
    },
]


def current_report_url(base_url, output_rel, rel_path):
    return f"{base_url.rstrip('/')}/{output_rel.strip('/')}/{rel_path.strip('/')}"


def render_current_report_card(entry, base_url, output_rel):
    href = current_report_url(base_url, output_rel, entry["href"])
    return (
        f"<a class='card link-card current-card {esc(entry['kind'])}' href='{esc(href)}'>"
        f"<strong>{esc(entry['title'])}</strong>"
        f"<span>{esc(entry['description'])}</span>"
        "</a>"
    )


def render_final_yaml_card(link):
    status = "ready" if link.get("exists") else "missing"
    return (
        f"<a class='artifact-card {esc(status)}' href='{esc(link['url'])}'>"
        "<strong>Final 32-camera YAML</strong>"
        "<span>Machine-readable intrinsics, distortion, and T_camera_studio "
        "extrinsics for all 24 outer + 8 inner cameras.</span>"
        f"<code>{esc(link.get('rel_path', ''))}</code>"
        "</a>"
    )


def render_standard_report_draft():
    sections = []
    for section in STANDARD_REPORT_DRAFT:
        items = "".join(f"<li>{esc(item)}</li>" for item in section["items"])
        sections.append(
            f"<div class='card'><strong>{esc(section['label'])}</strong>"
            f"<ul class='notes'>{items}</ul></div>"
        )
    return "\n".join(sections)


def render_operation_page(group, operation, base_url, output_rel, registry_url):
    entry_url = f"{base_url.rstrip('/')}/"
    steps = "".join(f"<li>{esc(step)}</li>" for step in operation["steps"])
    notes = "".join(f"<li>{esc(note)}</li>" for note in operation.get("notes", []))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(operation['label'])}</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <h1>{esc(operation['label'])}</h1>
    <p class="muted"><a href="{esc(entry_url)}">current_calibration</a> · <a href="{esc(registry_url)}">report_registry.json</a></p>
  </header>
  <main>
    <p>{esc(operation['summary'])}</p>
    <div class="grid">
      {render_link_card('Open Operation Panel', operation['panel_url'], '在 9898 panel 里选择受控 mode 并启动处理', 'ready')}
    </div>

    <h2>Backend Contract</h2>
    <table>
      <tbody>
        <tr><th>Capture type</th><td>{esc(group['label'])}</td></tr>
        <tr><th>Current backend</th><td><code>{esc(operation['current_backend'])}</code></td></tr>
        <tr><th>Current panel mode</th><td><code>{esc(operation['panel_mode'])}</code></td></tr>
        <tr><th>Target clean CLI</th><td><code>{esc(operation['target_cli'])}</code></td></tr>
      </tbody>
    </table>

    <h2>Operation Steps</h2>
    <ul class="notes">{steps}</ul>

    <h2>Notes</h2>
    <ul class="notes">{notes}</ul>
  </main>
</body>
</html>
"""


def build_registry(root, base_url, panel_url, output_rel, audit):
    links = audit["links"]
    categories = []
    for category in CANONICAL_REPORT_CATEGORIES:
        categories.append({
            "id": category["id"],
            "label": category["label"],
            "purpose": category["purpose"],
            "items": [
                {
                    "role": item["role"],
                    **links[item["link"]],
                }
                for item in category["items"]
            ],
            "notes": category.get("notes", []),
        })
    operations = []
    for group in OPERATION_GROUPS:
        operation = OPERATION_DEFINITIONS[group["id"]]
        operation_rel = f"{output_rel}/operations/{group['id']}.html"
        operations.append({
            "id": group["id"],
            "label": group["label"],
            "purpose": group["purpose"],
            "operation": {
                "label": operation["label"],
                "rel_path": operation_rel,
                "url": f"{base_url.rstrip('/')}/{operation_rel}",
                "panel_mode": operation["panel_mode"],
                "panel_url": panel_mode_url(panel_url, operation["panel_mode"]),
                "current_backend": operation["current_backend"],
                "target_clean_cli": operation["target_cli"],
            },
        })
    return {
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "human_entry": {
            "rel_path": "",
            "url": f"{base_url.rstrip('/')}/",
            "implementation_rel_path": f"{output_rel}/index.html",
            "implementation_url": f"{base_url.rstrip('/')}/{output_rel}/index.html",
        },
        "policy": {
            "primary_entry_directory": str(root / output_rel),
            "primary_entry_files": [
                "index.html",
                "inner_recalib_audit.html",
                "report_registry.json",
                "README.md",
            ],
            "supporting_operation_files": [
                "operations/whole.html",
                "operations/large_marker.html",
                "operations/small_marker.html",
            ],
            "algorithm_output_rule": (
                "Algorithm pipelines may write run artifacts under their own pipeline/run "
                "directories, but human-facing current links must be registered here."
            ),
            "artifact_manifest_required_fields": [
                "pipeline_id",
                "run_id",
                "created_at",
                "input_datasets",
                "artifacts",
                "quality_gates",
                "recommended_for_humans",
            ],
        },
        "canonical_report_categories": categories,
        "report_groups": categories,
        "operation_entries": operations,
        "operation_contract": {
            "panel_url": panel_url,
            "rule": (
                "Every capture type has a human Operation page. Operation pages link "
                "to the controlled 9898 panel; backend execution must be through "
                "whitelisted CLI modes, not arbitrary shell commands embedded in reports."
            ),
            "target_cli_shape": (
                "t0-calib operate {whole|large-marker|small-marker} --capture-root ... "
                "--output-root ... --publish-current"
            ),
            "entries": operations,
        },
        "final_viewer": {
            "target_contract": (
                "One canonical 3D viewport with camera-set mode toggles "
                "combined inner+outer, inner only, outer only, plus dataset "
                "coverage overlays for whole, large marker, and small marker."
            ),
            "canonical_current_url": links["overall_viewer"]["url"],
            "canonical_studio32_yaml_url": links["studio32_yaml"]["url"],
            "implementation_status": (
                "canonical: the current entry is a single combined viewer. "
                "Inner-only and outer-only are UI modes inside the same viewport; "
                "dataset coverage modes gray out cameras not constrained by the selected capture type."
            ),
            "modes": {
                "combined": links["overall_viewer"],
                "inner_only": links["overall_viewer"],
                "outer_only": links["overall_viewer"],
            },
        },
        "viewers": {
            "contract_note": (
                "Compatibility field. All final 3D modes resolve to the single canonical viewer."
            ),
            "combined": links["overall_viewer"],
            "inner": links["overall_viewer"],
            "outer": links["overall_viewer"],
        },
        "standard_report_draft": STANDARD_REPORT_DRAFT,
        "audits": {
            "inner_recalib": {
                "rel_path": f"{output_rel}/inner_recalib_audit.html",
                "url": f"{base_url.rstrip('/')}/{output_rel}/inner_recalib_audit.html",
                "status": audit["status"],
                "small_marker_disconnected": audit["small_disconnected"],
                "large_marker_bridge_metric_gate": audit["bridge_quality"].get("metric_bridge_gate"),
            },
        },
    }


def render_readme(registry):
    lines = [
        "# t0 标定报告入口",
        "",
        "人类用户从这里开始看：",
        "",
        f"- {registry['human_entry']['url']}",
        "",
        "主入口只按 canonical report categories 组织：",
        "",
    ]
    for category in registry["canonical_report_categories"]:
        lines.extend([
            f"## {category['label']}",
            "",
            category["purpose"],
            "",
        ])
        for item in category["items"]:
            lines.append(f"- {item['role']}: {item['url']}")
        lines.append("")
    lines.extend([
        "## Supporting Operation Pages",
        "",
        "Operation 页面是后台处理入口，不作为首页报告分类。需要采集后处理时使用下面的受控 panel mode：",
        "",
    ])
    for operation in registry["operation_entries"]:
        lines.extend([
            f"- {operation['label']}: {operation['operation']['url']} -> {operation['operation']['panel_url']}",
        ])
    lines.extend([
            "",
        "## Operation Contract",
        "",
        "每一类数据都有一个 Operation 页面。用户先进入 Operation 页面，再跳到 9898 panel 启动受控后端 mode。",
        "",
        f"- Panel: {registry['operation_contract']['panel_url']}",
        f"- Target CLI shape: `{registry['operation_contract']['target_cli_shape']}`",
        "",
    ])
    lines.extend([
        "## 最终 3D Viewer",
        "",
        "目标 contract：最终只有一个 canonical 3D viewport。页面内支持 camera set：combined inner+outer、inner only、outer only；dataset coverage：whole、large marker、small marker。",
        "",
        f"- 当前 canonical final viewer: {registry['final_viewer']['canonical_current_url']}",
        f"- 当前 studio_32_cameras.yaml: {registry['final_viewer']['canonical_studio32_yaml_url']}",
        "",
        "## 标准报告需求草案",
        "",
    ])
    for section in registry["standard_report_draft"]:
        lines.extend([
            f"### {section['label']}",
            "",
        ])
        for item in section["items"]:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend([
        "",
        "## Producer Contract",
        "",
        "算法 pipeline 不应该随意把 HTML 链接塞进主入口。",
        "它们应该把产物写在自己的 run directory，再通过 `report_registry.json` / artifact manifest 注册 production artifacts。",
        "",
    ])
    return "\n".join(lines)


def css():
    return """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; color: #222; background: #f6f6f2; }
header { background: #fff; border-bottom: 1px solid #deded7; padding: 28px 36px 22px; }
main { padding: 26px 36px 46px; }
h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }
h2 { margin: 34px 0 12px; font-size: 20px; }
p { line-height: 1.5; max-width: 1080px; }
.muted { color: #686b70; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.viewer-grid { display: grid; grid-template-columns: minmax(280px, 520px); gap: 12px; }
.report-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.card { background: #fff; border: 1px solid #deded7; border-radius: 8px; padding: 14px 16px; }
.link-card { display: flex; flex-direction: column; gap: 6px; text-decoration: none; color: #222; min-height: 72px; }
.link-card strong { font-size: 17px; }
.link-card span { color: #62666b; font-size: 13px; line-height: 1.35; }
.link-card:hover { border-color: #9ba7b5; }
.current-card.viewer { border-color: #b6c5d2; background: #fbfdff; }
.artifact-card { display: flex; flex-direction: column; gap: 7px; max-width: 720px; margin: 0 0 18px; text-decoration: none; color: #222; background: #f7fbff; border: 1px solid #b6c5d2; border-radius: 8px; padding: 14px 16px; }
.artifact-card strong { font-size: 18px; }
.artifact-card span { color: #46515c; font-size: 13px; line-height: 1.35; }
.artifact-card code { overflow-wrap: anywhere; }
.missing { border-color: #d9a49d; background: #fff8f7; }
.status { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 5px 10px; background: #e5eadf; font-size: 13px; font-weight: 650; }
.needs_attention { background: #f2ddc8; }
.usable_with_caveats { background: #e5eadf; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #deded7; }
th, td { padding: 9px 10px; border-bottom: 1px solid #ecece7; text-align: left; vertical-align: top; font-size: 13px; }
th { background: #eeeeea; font-weight: 650; }
code { background: #eeeeea; padding: 1px 4px; border-radius: 4px; }
.badge { border-radius: 5px; padding: 2px 7px; background: #e8e8e2; white-space: nowrap; }
.complete { background: #dcebdd; }
.failed_allowed, .blocked_missing_inputs { background: #f0dfbf; }
.failed { background: #f4d7d3; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }
.fact strong { display: block; font-size: 24px; margin-bottom: 3px; }
.fact span { color: #62666b; font-size: 13px; }
.report-section { display: grid; grid-template-columns: minmax(260px, 0.9fr) minmax(360px, 1.4fr); gap: 18px; background: #fff; border: 1px solid #deded7; border-radius: 8px; padding: 18px; margin: 14px 0; }
.operation-section { display: grid; grid-template-columns: minmax(260px, 0.7fr) minmax(360px, 1.6fr); gap: 18px; background: #fff; border: 1px solid #deded7; border-radius: 8px; padding: 18px; margin: 14px 0 22px; }
.report-section h2 { margin-top: 0; }
.operation-section h2 { margin-top: 0; }
.section-copy p { margin-top: 6px; }
.slot-column { display: flex; flex-direction: column; gap: 10px; }
.slot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; align-content: start; }
.notes, .secondary-links { color: #62666b; font-size: 13px; line-height: 1.45; padding-left: 18px; }
.secondary-links { margin: 0; }
@media (max-width: 820px) {
  header, main { padding-left: 18px; padding-right: 18px; }
  .report-section { grid-template-columns: 1fr; }
  .operation-section { grid-template-columns: 1fr; }
  .slot-grid { grid-template-columns: 1fr; }
  .secondary-links { grid-column: auto; }
}
"""


def render_audit_page(audit, base_url, output_rel):
    generated = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    bq = audit["bridge_quality"]
    bcq = audit["bridge_correspondence_quality"]
    final = audit["final"]
    links = audit["links"]
    status_text = (
        "usable with caveats"
        if audit["status"] == "usable_with_caveats"
        else "needs attention"
    )
    open_question = (
        "当前 production bridge 使用 large_marker_bridge_all32 的 all32 fixed-known-point joint BA。"
        "legacy top-down metric gate 只保留为诊断，不再作为 current calibration 的主质量门槛。"
        "如果 bridge BA 与 whole outer final rig 的 gauge / scale alignment 诊断偏大，需要单独决定"
        "固定 outer rig 还是接受 large-marker BA gauge。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inner Recalib Pipeline Audit</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <h1>Inner Recalib Pipeline Audit</h1>
    <div class="status {esc(audit['status'])}">{esc(status_text)}</div>
    <p class="muted">Generated: {esc(generated)} · <a href="{esc(base_url.rstrip('/') + '/' + output_rel + '/index.html')}">clean entry</a></p>
  </header>
  <main>
    <h2>Verdict</h2>
    <p>Pipeline 的主路径以 all32 large-marker BA 为 bridge 产品：large-inner fixed-intrinsic baseline 成功；large_marker_bridge_all32 先做 PnP initializer，再用固定 board 3D points 的 joint BA 输出最终 residual；small-marker 质量 probe 的弱相机是 <code>{esc(render_disconnected(audit['small_disconnected']))}</code>。</p>
    <p>{esc(open_question)}</p>
    <div class="facts">
      <div class="card fact"><strong>{esc(audit['large_inner_connected'])}/8</strong><span>large-inner connected cameras</span></div>
      <div class="card fact"><strong>{esc(bcq.get('ok_count', 0))}</strong><span>all32 BA residual count</span></div>
      <div class="card fact"><strong>{esc(bcq.get('median_residual_px'))}</strong><span>all32 BA median residual px</span></div>
      <div class="card fact"><strong>{esc(bcq.get('p90_residual_px'))}</strong><span>all32 BA p90 residual px</span></div>
      <div class="card fact"><strong>{esc(bcq.get('max_residual_px'))}</strong><span>all32 BA max residual px</span></div>
      <div class="card fact"><strong>{esc(audit['large_bridge_connected'])}/32</strong><span>all32 PnP initializer connected cameras</span></div>
    </div>

    <h2>Data Quality</h2>
    <table><thead><tr><th>Dataset</th><th>Status</th><th>Usable cameras</th><th>Common frames</th><th>Frame spread</th><th>Warning</th></tr></thead><tbody>{render_quality_rows(audit['data_quality'])}</tbody></table>

    <h2>Large Marker Bridge Contract</h2>
    <p>all32 order: outer cameras <code>0..23</code>, inner cameras <code>24..31</code>. Production bridge pose source is <code>{esc(final.get('bridge_pose_yaml'))}</code>. Inner final baseline source is <code>{esc(final.get('inner_prior_source'))}</code>. The legacy metric bridge gate status is <code>{esc(bq.get('metric_bridge_gate', 'missing'))}</code> and is diagnostic only.</p>
    <p>Disconnected PnP initializer cameras: <code>{esc(render_disconnected(audit['large_bridge_disconnected']))}</code>.</p>

    <h2>Known Caveats</h2>
    <table><thead><tr><th>Group</th><th>Stage</th><th>Status</th><th>Return code</th><th>Allow failure</th><th>Notes</th></tr></thead><tbody>{render_stage_rows(audit['stage_failures'])}</tbody></table>

    <h2>Source Reports</h2>
    <div class="grid">
      {render_link_card(links['large_data_collection']['label'], links['large_data_collection']['url'], 'capture and stage quality')}
      {render_link_card(links['inner_final']['label'], links['inner_final']['url'], 'wrapper final report')}
      {render_link_card(links['large_final']['label'], links['large_final']['url'], 'all32 bridge BA final report')}
      {render_link_card('summary.json', url(base_url, INNER_SUMMARY_REL), 'machine-readable pipeline summary')}
    </div>
  </main>
</body>
</html>
"""


def render_entry_page(audit, base_url, panel_url, output_rel):
    generated = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    status_text = (
        "current calibration reports ready; inner audit has caveats"
        if audit["status"] == "usable_with_caveats"
        else "current calibration reports ready; inner audit needs attention"
    )
    viewer_card = render_current_report_card(CURRENT_REPORT_ENTRIES[0], base_url, output_rel)
    report_cards = "\n".join(
        render_current_report_card(entry, base_url, output_rel)
        for entry in CURRENT_REPORT_ENTRIES[1:]
    )
    final_yaml_card = render_final_yaml_card(audit["links"]["studio32_yaml"])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Studio Calibration Reports</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <h1>Studio Calibration Reports</h1>
    <div class="status {esc(audit['status'])}">{esc(status_text)}</div>
    <p class="muted">Generated: {esc(generated)} · one viewer + seven reports</p>
  </header>
  <main>
    <p>这个入口只保留当前生产标定需要看的内容：一个整体 3D viewer，以及内圈、外圈、bridge 的七个报告。operation panel、registry、历史诊断和探索报告不在首页展示。</p>

    {final_yaml_card}

    <h2>Overall Viewer</h2>
    <div class="viewer-grid">{viewer_card}</div>

    <h2>Reports</h2>
    <div class="report-grid">{report_cards}</div>
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    configure_current_run_paths(
        bridge_run_rel=args.current_bridge_run_rel,
        outer_run_rel=args.current_outer_run_rel,
        outer_report_rel=args.current_outer_report_rel,
        whole_data_report_rel=args.whole_data_report_rel,
        whole_distributed_qc_rel=args.whole_distributed_qc_rel,
        studio32_yaml_rel=args.studio32_yaml_rel,
    )
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    operations_dir = output_dir / "operations"
    operations_dir.mkdir(parents=True, exist_ok=True)
    output_rel = output_dir.relative_to(root).as_posix()
    audit = inner_audit(root, args.base_url)
    registry = build_registry(root, args.base_url, args.panel_url, output_rel, audit)
    registry_url = f"{args.base_url.rstrip('/')}/{output_rel}/report_registry.json"
    (output_dir / "report_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        render_readme(registry),
        encoding="utf-8",
    )
    (output_dir / "inner_recalib_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "inner_recalib_audit.html").write_text(
        render_audit_page(audit, args.base_url, output_rel),
        encoding="utf-8",
    )
    for group in OPERATION_GROUPS:
        operation = dict(OPERATION_DEFINITIONS[group["id"]])
        operation["panel_url"] = panel_mode_url(args.panel_url, operation["panel_mode"])
        (operations_dir / f"{group['id']}.html").write_text(
            render_operation_page(group, operation, args.base_url, output_rel, registry_url),
            encoding="utf-8",
        )
    entry_html = render_entry_page(audit, args.base_url, args.panel_url, output_rel)
    (output_dir / "index.html").write_text(entry_html, encoding="utf-8")
    if args.write_root_index:
        root_index = root / "index.html"
        if root_index.resolve(strict=False) != (output_dir / "index.html").resolve(strict=False):
            root_index.write_text(entry_html, encoding="utf-8")
    print(output_dir / "index.html")
    for group in OPERATION_GROUPS:
        print(operations_dir / f"{group['id']}.html")
    print(output_dir / "inner_recalib_audit.html")
    print(output_dir / "report_registry.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
