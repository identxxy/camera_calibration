#!/usr/bin/env python3
"""Plan and report the fast inner/bridge recalibration pipeline for t0 data."""

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_T0_DATA_ROOT = Path("/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3")
DEFAULT_T0_REPO = Path("/home/ubuntu/camera_calibration")
DEFAULT_T0_BINARY = (
    "/home/ubuntu/camera_calibration/build_t0/applications/"
    "camera_calibration/camera_calibration"
)
DEFAULT_T0_PYTHON = "/home/ubuntu/miniconda3/bin/python"
DEFAULT_OUTER_FINAL_POSE_YAML = (
    DEFAULT_T0_DATA_ROOT
    / "recalib_pipelines/outer_tower/latest/tag_refine_robust/camera_tr_rig_delta_refined_accepted.yaml"
)
DEFAULT_REPORT_HTTP_ROOT = "http://192.168.2.0:9899"
CALIB_DATA_ROOT = Path("/home/ubuntu/calib_data")

SMALL_MARKER_PATTERN = (
    "applications/camera_calibration/patterns/"
    "pattern_resolution_50x72_segments_16_apriltag_3.yaml"
)
LARGE_MARKER_PATTERN = (
    "applications/camera_calibration/patterns/"
    "pattern_resolution_17x24_segments_16_apriltag_0.yaml"
)

OUTER_CAMERAS = [
    ("w4_D", "1-1"), ("w4_D", "1-2"), ("w4_D", "1-3"),
    ("w4_D", "2-1"), ("w4_D", "2-2"), ("w4_D", "2-3"),
    ("w4_D", "3-1"), ("w4_D", "3-2"), ("w4_D", "3-3"),
    ("w4_D", "4-1"), ("w4_D", "4-2"), ("w4_D", "4-3"),
    ("w3_D", "5-1"), ("w3_D", "5-2"), ("w3_D", "5-3"),
    ("w3_D", "6-1"), ("w3_D", "6-2"), ("w3_D", "6-3"),
    ("w3_D", "7-1"), ("w3_D", "7-2"), ("w3_D", "7-3"),
    ("w3_D", "8-1"), ("w3_D", "8-2"), ("w3_D", "8-3"),
]

INNER_CAMERAS = [
    ("w1_D", "22463688"), ("w1_D", "22463690"),
    ("w1_D", "22587611"), ("w1_D", "22587616"),
    ("w2_D", "22463689"), ("w2_D", "22463691"),
    ("w2_D", "22463702"), ("w2_D", "22587614"),
]

INNER_IDS = {camera_id for _machine, camera_id in INNER_CAMERAS}
OUTER_IDS = {camera_id for _machine, camera_id in OUTER_CAMERAS}
TOPDOWN_BRIDGE_LABEL_ORDER = ["4-1", "4-2", "4-3"]
TOPDOWN_BRIDGE_LABELS = set(TOPDOWN_BRIDGE_LABEL_ORDER)
BRIDGE_ALL32_OUTER_COUNT = len(OUTER_CAMERAS)
BRIDGE_ALL32_INNER_COUNT = len(INNER_CAMERAS)
BRIDGE_ALL32_CAMERA_COUNT = BRIDGE_ALL32_OUTER_COUNT + BRIDGE_ALL32_INNER_COUNT
BRIDGE_ALL32_INNER_INDICES = list(range(BRIDGE_ALL32_OUTER_COUNT, BRIDGE_ALL32_CAMERA_COUNT))
BRIDGE_ALL32_TOPDOWN_INDICES = [
    index for index, (_machine, label) in enumerate(OUTER_CAMERAS)
    if label in TOPDOWN_BRIDGE_LABELS
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_ID_RE = re.compile(r"(?:^|_)(\d+)(?=\.[^.]+$)")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def duration_s(value):
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def script_repo_root():
    return Path(__file__).resolve().parents[2]


def resolve_user_path(path, base=None):
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve(strict=False)


def rel_or_abs(path, root):
    path = Path(path).resolve(strict=False)
    root = Path(root).resolve(strict=False)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def path_url(path):
    path = Path(path).expanduser().resolve(strict=False)
    try:
        return path.as_uri()
    except ValueError:
        return ""


def http_report_url(path, http_root=CALIB_DATA_ROOT, http_base=DEFAULT_REPORT_HTTP_ROOT):
    path = Path(path).expanduser().resolve(strict=False)
    try:
        rel = path.relative_to(Path(http_root).expanduser().resolve(strict=False))
    except ValueError:
        return ""
    return str(http_base).rstrip("/") + "/" + "/".join(rel.parts)


def public_report_url(path, http_root=CALIB_DATA_ROOT, http_base=DEFAULT_REPORT_HTTP_ROOT):
    return http_report_url(path, http_root, http_base) or path_url(path)


def read_json_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc), "path": str(path)}


def run_mode(args):
    if args.dry_run:
        return "dry_run"
    if (
        args.run_all
        or args.run_stage
        or args.run_large_inner_init
        or args.run_small_fixed_rig_quality
        or args.run_small_refine
        or args.run_large_bridge
        or args.run_reports
    ):
        return "execute"
    return "reuse_plan"


def command_string(argv):
    if not argv:
        return ""
    return shlex.join(str(item) for item in argv)


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


def pipeline_provenance(repo_root):
    return {
        "script": str(Path(__file__).resolve(strict=False)),
        "argv": command_string(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "git": git_metadata(repo_root),
    }


def latest_child_dir(path):
    path = Path(path)
    if not path.is_dir():
        return path
    children = [item for item in path.iterdir() if item.is_dir()]
    if not children:
        return path
    return sorted(children, key=lambda item: item.name)[-1]


def resolve_marker_path(data_root, override, staged_name, raw_name):
    if override:
        path = resolve_user_path(override, data_root)
        if path.exists() and (path / "image_directories.txt").is_file():
            return path
        if path.exists() and path.name == raw_name:
            return latest_child_dir(path)
        return path

    candidates = [
        data_root / staged_name,
        data_root / raw_name,
        data_root / "output" / "calib" / raw_name,
    ]
    for candidate in candidates:
        if (candidate / "image_directories.txt").is_file():
            return candidate
    for candidate in candidates:
        if candidate.is_dir():
            return latest_child_dir(candidate)
    return candidates[0]


def read_manifest(path):
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def summarize_fixed_rig_quality(path, expected_camera_count, input_scan):
    path = Path(path)
    summary = {
        "path": str(path),
        "exists": path.is_file(),
        "status": "missing",
        "camera_count": 0,
        "expected_camera_count": expected_camera_count,
        "connected_count": 0,
        "disconnected_count": None,
        "disconnected_cameras": [],
        "notes": [],
    }
    if input_scan.get("usable_camera_count") != expected_camera_count:
        summary["notes"].append(
            "input_usable_camera_count_is_"
            + str(input_scan.get("usable_camera_count"))
            + "_of_"
            + str(expected_camera_count)
        )
    if not path.is_file():
        return summary

    rows = read_manifest(path)
    summary["camera_count"] = len(rows)
    disconnected = []
    for row in rows:
        connected = str(row.get("connected", "")).strip().lower() == "yes"
        if connected:
            summary["connected_count"] += 1
        else:
            label = row.get("user_id") or row.get("stage_name") or row.get("camera_index") or "unknown"
            disconnected.append(str(label))

    summary["disconnected_cameras"] = disconnected
    summary["disconnected_count"] = len(disconnected)
    if not rows:
        summary["status"] = "empty_summary"
    elif disconnected:
        summary["status"] = "disconnected_cameras"
        summary["notes"].append("small_marker_probe_is_quality_only_final_large_inner_baseline_is_unchanged")
    elif len(rows) != expected_camera_count:
        summary["status"] = "connected_unexpected_camera_count"
    else:
        summary["status"] = "all_connected"
    return summary


def annotate_small_quality_stage(stages, quality):
    for stage in stages:
        if stage.get("name") != "estimate_small_marker_fixed_rig_quality":
            continue
        notes = stage.setdefault("notes", [])
        if quality.get("exists") and quality.get("disconnected_count"):
            notes.append(
                "disconnected_cameras="
                + ",".join(quality.get("disconnected_cameras") or [])
            )
            notes.append("quality_probe_failure_does_not_replace_final_inner_baseline")
        elif quality.get("exists") and quality.get("status") == "all_connected":
            notes.append("all_expected_cameras_connected")
        elif not quality.get("exists"):
            notes.append("camera_pnp_summary_missing_until_probe_runs")
        return


def summarize_bridge_quality(bridge_summary_json):
    path = Path(bridge_summary_json)
    summary = {
        "path": str(path),
        "exists": path.is_file(),
        "status": "missing",
        "metric_bridge_gate": "missing",
        "colmap_prior_diagnostic": "missing",
        "outer_vote_count_min": None,
        "max_outer_center_residual_p90_m": None,
        "max_outer_rotation_residual_p90_deg": None,
        "inner_board_frame_count": None,
        "inner_support_median": None,
        "prior_output_ready": False,
        "notes": [],
    }
    data = read_json_file(path)
    if data is None:
        return summary
    if data.get("_read_error"):
        summary["status"] = "read_error"
        summary["notes"].append(data["_read_error"])
        return summary

    gates = data.get("quality_gates", {})
    metric_gate = gates.get("metric_bridge", {})
    colmap_gate = gates.get("colmap_prior_diagnostic", {})
    metric_summary = gates.get("metric_summary", {})
    inner = data.get("inner_board_pose_summary", {})
    outer_rows = data.get("outer_camera_summaries", [])

    summary.update({
        "status": "present",
        "metric_bridge_gate": metric_gate.get("status", "legacy_missing_gate"),
        "metric_bridge_passed": metric_gate.get("passed"),
        "metric_bridge_failed_checks": metric_gate.get("failed_checks", []),
        "colmap_prior_diagnostic": colmap_gate.get("status", "legacy_missing_gate"),
        "colmap_prior_failed_checks": colmap_gate.get("failed_checks", []),
        "outer_vote_count_min": metric_summary.get("min_outer_votes"),
        "max_outer_center_residual_p90_m": metric_summary.get("max_outer_center_residual_p90_m"),
        "max_outer_rotation_residual_p90_deg": metric_summary.get("max_outer_rotation_residual_p90_deg"),
        "inner_board_frame_count": inner.get("frame_count"),
        "inner_support_median": inner.get("inner_support_median"),
        "outer_camera_count": len(outer_rows),
        "outer_labels": [row.get("label") for row in outer_rows],
        "bridge_pose_yaml": data.get("outputs", {}).get("bridge_camera_tr_rig", ""),
    })
    summary["prior_output_ready"] = bool(
        summary.get("bridge_pose_yaml")
        and Path(summary["bridge_pose_yaml"]).is_file()
        and summary.get("metric_bridge_gate") == "pass"
    )
    if summary.get("metric_bridge_gate") != "pass":
        summary["notes"].append("metric_bridge_gate_not_passed_do_not_use_as_production_bridge_prior")
    if summary.get("colmap_prior_diagnostic") not in {"consistent", "missing"}:
        summary["notes"].append("colmap_prior_diagnostic_is_weak_use_metric_bridge_gate_as_primary_signal")
    return summary


def parse_intrinsics_yaml(path):
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(r"parameters\s*:\s*\[([^\]]+)\]", text, flags=re.S)
    if not match:
        return None
    return [float(x) for x in re.split(r"[,\s]+", match.group(1).strip()) if x]


def summarize_inner_joint_intrinsics(prior_state, joint_state):
    prior_state = Path(prior_state)
    joint_state = Path(joint_state)
    summary = {
        "prior_state": str(prior_state),
        "joint_state": str(joint_state),
        "exists": (joint_state / "camera_tr_rig.yaml").is_file(),
        "status": "missing_joint_output",
        "camera_count": 0,
        "accepted_by_sanity_gate": False,
        "gate": {
            "max_abs_focal_delta_frac": 0.05,
            "max_principal_delta_px": 80.0,
            "max_abs_distortion": 5.0,
            "max_abs_distortion_delta": 5.0,
        },
        "max_abs_focal_delta_frac": None,
        "max_principal_delta_px": None,
        "max_abs_distortion": None,
        "max_abs_distortion_delta": None,
        "notes": [],
    }
    if not summary["exists"]:
        summary["final_selection"] = "large_inner_fixed_baseline"
        return summary

    rows = []
    for prior_path in sorted(prior_state.glob("intrinsics*.yaml")):
        joint_path = joint_state / prior_path.name
        prior = parse_intrinsics_yaml(prior_path)
        joint = parse_intrinsics_yaml(joint_path)
        if prior is None or joint is None:
            continue
        prior = (prior + [0.0] * 12)[:12]
        joint = (joint + [0.0] * 12)[:12]
        focal_delta = max(
            abs(joint[0] / prior[0] - 1.0) if prior[0] else float("inf"),
            abs(joint[1] / prior[1] - 1.0) if prior[1] else float("inf"),
        )
        principal_delta = max(abs(joint[2] - prior[2]), abs(joint[3] - prior[3]))
        distortion = max(abs(value) for value in joint[4:12])
        distortion_delta = max(abs(joint[i] - prior[i]) for i in range(4, 12))
        rows.append({
            "camera": prior_path.stem.replace("intrinsics", ""),
            "focal_delta_frac": focal_delta,
            "principal_delta_px": principal_delta,
            "max_abs_distortion": distortion,
            "max_abs_distortion_delta": distortion_delta,
        })

    summary["camera_count"] = len(rows)
    if not rows:
        summary["status"] = "missing_intrinsics_yaml"
        summary["final_selection"] = "large_inner_fixed_baseline"
        return summary

    summary["max_abs_focal_delta_frac"] = max(row["focal_delta_frac"] for row in rows)
    summary["max_principal_delta_px"] = max(row["principal_delta_px"] for row in rows)
    summary["max_abs_distortion"] = max(row["max_abs_distortion"] for row in rows)
    summary["max_abs_distortion_delta"] = max(row["max_abs_distortion_delta"] for row in rows)
    gate = summary["gate"]
    accepted = (
        summary["max_abs_focal_delta_frac"] <= gate["max_abs_focal_delta_frac"]
        and summary["max_principal_delta_px"] <= gate["max_principal_delta_px"]
        and summary["max_abs_distortion"] <= gate["max_abs_distortion"]
        and summary["max_abs_distortion_delta"] <= gate["max_abs_distortion_delta"]
    )
    summary["accepted_by_sanity_gate"] = accepted
    summary["status"] = "accepted_diagnostic" if accepted else "rejected_unphysical_intrinsics"
    if not accepted:
        summary["notes"].append(
            "joint output is diagnostic only; final inner baseline remains the large-marker fixed-intrinsic state")
        summary["final_selection"] = "large_inner_fixed_baseline"
    else:
        summary["notes"].append(
            "joint output passed sanity gate but is not auto-promoted in this run; rerun reports with explicit promotion support before using it as final")
        summary["final_selection"] = "joint_passed_gate_not_auto_promoted"
    return summary


def int_or_none(value):
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize_manifest(path, max_tail_trim):
    if not path:
        return {
            "path": "",
            "exists": False,
            "camera_count": 0,
            "frame_count_min": None,
            "frame_count_max": None,
            "frame_count_spread": None,
            "frame_alignment": "manifest_missing",
            "drop_frame_warning": "manifest_missing",
        }

    path = Path(path)
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "camera_count": 0,
            "frame_count_min": None,
            "frame_count_max": None,
            "frame_count_spread": None,
            "frame_alignment": "manifest_missing",
            "drop_frame_warning": "manifest_missing",
        }

    rows = read_manifest(path)
    frame_counts = []
    for row in rows:
        value = None
        for key in ("frame_count", "frames", "image_count"):
            value = int_or_none(row.get(key))
            if value is not None:
                break
        if value is not None:
            frame_counts.append(value)

    spread = None
    if frame_counts:
        spread = max(frame_counts) - min(frame_counts)
    if spread is None:
        alignment = "frame_count_unavailable"
        warning = "manifest_has_no_frame_count_column"
    elif spread <= max_tail_trim:
        alignment = "tail_trim_ok"
        warning = ""
    else:
        alignment = "frame_count_spread_exceeds_tail_trim"
        warning = "camera_drop_or_unstable_connection_suspected"

    return {
        "path": str(path),
        "exists": True,
        "camera_count": len(rows),
        "frame_count_min": min(frame_counts) if frame_counts else None,
        "frame_count_max": max(frame_counts) if frame_counts else None,
        "frame_count_spread": spread,
        "frame_count_unique": sorted(set(frame_counts)) if frame_counts else [],
        "frame_alignment": alignment,
        "drop_frame_warning": warning,
    }


def read_image_directories_file(path):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    return [
        Path(item.strip()).expanduser()
        for item in text.replace("\n", ",").split(",")
        if item.strip()
    ]


def image_id_from_name(path):
    match = IMAGE_ID_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def scan_images(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return {
            "exists": False,
            "frame_ids": [],
            "frame_count": 0,
            "first_frame": None,
            "last_frame": None,
            "contiguous_from_zero": False,
            "missing_interior_count": 0,
        }

    frame_ids = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frame_id = image_id_from_name(path)
        if frame_id is not None:
            frame_ids.append(frame_id)
    frame_ids = sorted(set(frame_ids))
    contiguous = bool(frame_ids) and frame_ids[0] == 0 and frame_ids == list(range(frame_ids[-1] + 1))
    missing_interior = 0
    if frame_ids:
        missing_interior = frame_ids[-1] + 1 - len(frame_ids)
    return {
        "exists": True,
        "frame_ids": frame_ids,
        "frame_count": len(frame_ids),
        "first_frame": frame_ids[0] if frame_ids else None,
        "last_frame": frame_ids[-1] if frame_ids else None,
        "contiguous_from_zero": contiguous,
        "missing_interior_count": missing_interior,
    }


def infer_machine(path):
    for part in Path(path).parts:
        if part in {"w1_D", "w2_D", "w3_D", "w4_D"}:
            return part
    return ""


def camera_kind(machine, camera_id, stage_name):
    camera_id = str(camera_id or "")
    stage_name = str(stage_name or "")
    machine = str(machine or "")
    if camera_id in INNER_IDS or machine in {"w1_D", "w2_D"} or "_w1_" in stage_name or "_w2_" in stage_name:
        return "inner"
    if camera_id in OUTER_IDS or re.match(r"^\d+-\d+$", camera_id):
        return "outer"
    return "unknown"


def build_camera_entry(index, image_dir, manifest_row=None):
    manifest_row = manifest_row or {}
    stage_name = manifest_row.get("stage_name") or f"cam{index:02d}_{Path(image_dir).name}"
    machine = manifest_row.get("machine") or infer_machine(image_dir)
    camera_id = (
        manifest_row.get("camera_id")
        or manifest_row.get("user_id")
        or manifest_row.get("camera")
        or Path(image_dir).name
    )
    return {
        "index": index,
        "stage_name": stage_name,
        "machine": machine,
        "camera_id": str(camera_id),
        "kind": camera_kind(machine, camera_id, stage_name),
        "image_dir": str(Path(image_dir).expanduser()),
    }


def load_camera_dirs(session_path):
    image_dirs_file = session_path / "image_directories.txt"
    manifest_file = session_path / "manifest.tsv"
    if image_dirs_file.is_file():
        image_dirs = read_image_directories_file(image_dirs_file)
        manifest_rows = read_manifest(manifest_file)
        cameras = []
        for index, image_dir in enumerate(image_dirs):
            manifest_row = manifest_rows[index] if index < len(manifest_rows) else {}
            cameras.append(build_camera_entry(index, image_dir, manifest_row))
        return {
            "source_mode": "image_directories_file",
            "image_directories_file": str(image_dirs_file),
            "manifest": str(manifest_file) if manifest_file.is_file() else "",
            "cameras": cameras,
        }

    child_dirs = [item for item in sorted(session_path.iterdir()) if item.is_dir()]
    if (session_path / "images").is_dir():
        child_dirs = [item for item in sorted((session_path / "images").iterdir()) if item.is_dir()]
    cameras = [build_camera_entry(index, image_dir) for index, image_dir in enumerate(child_dirs)]
    return {
        "source_mode": "camera_directory_root",
        "image_directories_file": "",
        "manifest": "",
        "cameras": cameras,
    }


def scan_marker_session(label, session_path, expected_count, max_tail_trim):
    start = time.time()
    session_path = Path(session_path)
    result = {
        "label": label,
        "path": str(session_path),
        "exists": session_path.exists(),
        "status": "missing",
        "source_mode": "missing",
        "expected_camera_count": expected_count,
        "camera_count": 0,
        "usable_camera_count": 0,
        "unusable_camera_count": 0,
        "max_source_frame_count": 0,
        "min_source_frame_count": 0,
        "common_frame_count": 0,
        "max_tail_trim": max_tail_trim,
        "normalized_inputs": False,
        "image_directories_file": "",
        "image_directories_file_exists": False,
        "manifest": "",
        "manifest_exists": False,
        "manifest_summary": summarize_manifest("", max_tail_trim),
        "frame_count_min": None,
        "frame_count_max": None,
        "frame_count_spread": None,
        "tail_trim_policy": f"Tail-only stop offsets <= {max_tail_trim} frames are accepted; interior drops invalidate that camera.",
        "drop_frame_warning": "",
        "cameras": [],
        "errors": [],
        "warnings": [],
        "duration": 0.0,
    }
    if not session_path.exists():
        result["errors"].append("session_path_missing")
        result["duration"] = time.time() - start
        return result
    if not session_path.is_dir():
        result["errors"].append("session_path_is_not_directory")
        result["duration"] = time.time() - start
        return result

    try:
        loaded = load_camera_dirs(session_path)
    except OSError as exc:
        result["errors"].append(str(exc))
        result["duration"] = time.time() - start
        return result

    cameras = loaded["cameras"]
    result["source_mode"] = loaded["source_mode"]
    result["image_directories_file"] = loaded["image_directories_file"]
    result["manifest"] = loaded["manifest"]
    result["image_directories_file_exists"] = bool(loaded["image_directories_file"]) and Path(loaded["image_directories_file"]).is_file()
    result["manifest_exists"] = bool(loaded["manifest"]) and Path(loaded["manifest"]).is_file()
    result["manifest_summary"] = summarize_manifest(loaded["manifest"], max_tail_trim)
    result["camera_count"] = len(cameras)
    if not cameras:
        result["status"] = "empty"
        result["errors"].append("no_camera_directories_found")
        result["duration"] = time.time() - start
        return result

    scanned = []
    counts = []
    for camera in cameras:
        scan = scan_images(camera["image_dir"])
        row = dict(camera)
        row.update({
            "exists": scan["exists"],
            "frame_count": scan["frame_count"],
            "first_frame": scan["first_frame"],
            "last_frame": scan["last_frame"],
            "contiguous_from_zero": scan["contiguous_from_zero"],
            "missing_interior_count": scan["missing_interior_count"],
            "tail_short": None,
            "status": "unusable",
            "reason": "",
        })
        scanned.append(row)
        counts.append(scan["frame_count"])

    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    usable = []
    for row in scanned:
        count = row["frame_count"]
        row["tail_short"] = max_count - count
        if not row["exists"]:
            row["reason"] = "image_dir_missing"
        elif count == 0:
            row["reason"] = "no_images"
        elif not row["contiguous_from_zero"]:
            row["reason"] = "non_contiguous_or_missing_interior_frames"
        elif row["tail_short"] <= max_tail_trim:
            row["status"] = "usable"
            row["reason"] = "ok" if row["tail_short"] == 0 else f"tail_trim_{row['tail_short']}"
            usable.append(row)
        else:
            row["reason"] = "short_sequence_not_tail_trim"

    common_ids = []
    if usable:
        id_sets = []
        for row in usable:
            id_sets.append(set(scan_images(row["image_dir"])["frame_ids"]))
        common_ids = sorted(set.intersection(*id_sets)) if id_sets else []

    result["cameras"] = scanned
    result["usable_camera_count"] = len(usable)
    result["unusable_camera_count"] = len(scanned) - len(usable)
    result["max_source_frame_count"] = max_count
    result["min_source_frame_count"] = min_count
    result["frame_count_min"] = min_count if counts else None
    result["frame_count_max"] = max_count if counts else None
    result["frame_count_spread"] = (max_count - min_count) if counts else None
    result["common_frame_count"] = len(common_ids)
    result["normalized_inputs"] = (
        loaded["source_mode"] == "image_directories_file"
        and bool(usable)
        and len({row["frame_count"] for row in usable}) == 1
        and len(common_ids) == (usable[0]["frame_count"] if usable else 0)
    )
    if result["usable_camera_count"] == 0:
        result["status"] = "no_usable_cameras"
        result["errors"].append("no_usable_cameras")
    elif result["unusable_camera_count"] > 0:
        result["status"] = "ok_with_unusable_cameras"
    elif result["usable_camera_count"] != expected_count:
        result["status"] = "ok_unexpected_camera_count"
        result["warnings"].append(f"expected_{expected_count}_cameras_got_{result['usable_camera_count']}")
    else:
        result["status"] = "ok"
    if not result["normalized_inputs"]:
        result["warnings"].append("feature_extraction_requires_staged_normalized_image_directories")
    if result["frame_count_spread"] is not None and result["frame_count_spread"] > max_tail_trim:
        result["drop_frame_warning"] = "frame_count_spread_exceeds_tail_trim_check_camera_drop"
        result["warnings"].append(result["drop_frame_warning"])
    if any(row["missing_interior_count"] for row in scanned):
        result["warnings"].append("interior_frame_gap_detected_exclude_affected_camera")
    if not result["manifest_exists"]:
        result["warnings"].append("manifest_missing")
    result["duration"] = time.time() - start
    return result


def write_planned_inputs(output_root, scan):
    input_root = Path(output_root) / "planned_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    prefix = scan["label"]
    usable = [row for row in scan["cameras"] if row["status"] == "usable"]
    image_dirs_file = input_root / f"{prefix}_usable_image_directories.txt"
    manifest_file = input_root / f"{prefix}_usable_manifest.tsv"

    image_dirs_file.write_text(",".join(row["image_dir"] for row in usable) + "\n", encoding="utf-8")
    with manifest_file.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "camera_index", "stage_name", "machine", "camera_id", "kind",
            "source_dir", "frame_count", "status", "reason",
        ]
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for new_index, row in enumerate(usable):
            writer.writerow({
                "camera_index": new_index,
                "stage_name": row["stage_name"],
                "machine": row["machine"],
                "camera_id": row["camera_id"],
                "kind": row["kind"],
                "source_dir": row["image_dir"],
                "frame_count": row["frame_count"],
                "status": row["status"],
                "reason": row["reason"],
            })

    return {
        "image_directories_file": str(image_dirs_file),
        "manifest": str(manifest_file),
        "usable_camera_count": len(usable),
    }


def link_record(label, path, kind="file", http_root=CALIB_DATA_ROOT, http_base=DEFAULT_REPORT_HTTP_ROOT):
    path = Path(path).expanduser().resolve(strict=False)
    return {
        "label": label,
        "kind": kind,
        "path": str(path),
        "exists": path.exists(),
        "file_url": path_url(path) if path.exists() else "",
        "http_url": http_report_url(path, http_root, http_base) if path.exists() else "",
    }


def discover_existing_inner_links(data_root, repo_root, http_root=CALIB_DATA_ROOT, http_base=DEFAULT_REPORT_HTTP_ROOT):
    final_root = data_root / "final_inner8_calibration_v1"
    local_inner_root = repo_root / "studio/exp/inner_marker_2026_05_26_processing"
    return [
        link_record(
            "t0 final inner interactive viewer",
            final_root / "reports/interactive_rig_viewer_v1/index.html",
            "viewer",
            http_root,
            http_base,
        ),
        link_record(
            "t0 final inner reprojection report",
            final_root / "reports/report_small_grid4_refined_reprojection_v1/index.html",
            "report",
            http_root,
            http_base,
        ),
        link_record(
            "t0 final inner state",
            final_root / "states/final_small_marker_grid4_refine_v1",
            "state_dir",
            http_root,
            http_base,
        ),
        link_record(
            "t0 final inner intrinsics",
            final_root / "intrinsics/small_marker_opencv_grid4_pattern3_v2",
            "intrinsics_dir",
            http_root,
            http_base,
        ),
        link_record(
            "local pulled inner interactive viewer",
            local_inner_root / "interactive_rig_viewer_v1/index.html",
            "viewer",
            http_root,
            http_base,
        ),
        link_record(
            "local pulled inner reprojection report",
            local_inner_root / "report_small_grid4_refined_reprojection_v1/index.html",
            "report",
            http_root,
            http_base,
        ),
    ]


def discover_existing_bridge_links(data_root, sequence_name, http_root=CALIB_DATA_ROOT, http_base=DEFAULT_REPORT_HTTP_ROOT):
    sequence_names = []
    for item in [sequence_name, "large_marker_bridge_4topdown_v1", "large_marker_bridge_all32"]:
        item = str(item or "")
        if item and item not in sequence_names:
            sequence_names.append(item)

    links = []
    for name in sequence_names:
        bridge_root = data_root / name / "bridge_colmap_inner_refined_v1"
        links.extend([
            link_record(
                f"{name} bridge report",
                bridge_root / "index.html",
                "report",
                http_root,
                http_base,
            ),
            link_record(
                f"{name} bridge summary",
                bridge_root / "bridge_summary.json",
                "summary",
                http_root,
                http_base,
            ),
            link_record(
                f"{name} bridge pose yaml",
                bridge_root / "camera_tr_inner_refined_plus_outer_topdown.yaml",
                "pose_yaml",
                http_root,
                http_base,
            ),
        ])
    return links


def infer_inner_intrinsics(data_root, inner_prior):
    candidates = [
        data_root / "final_inner8_calibration_v1/intrinsics/small_marker_opencv_grid4_pattern3_v2",
    ]
    if inner_prior:
        inner_prior = Path(inner_prior)
        candidates.extend([
            inner_prior / "intrinsics",
            inner_prior.parent.parent / "intrinsics/small_marker_opencv_grid4_pattern3_v2",
            inner_prior.parent / "intrinsics_opencv_grid4_pattern3_v2",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(strict=False)
    return candidates[0].resolve(strict=False)


def infer_outer_intrinsics(data_root):
    candidates = [
        data_root / "whole_outer_tower/fixed_intrinsic_pnp_colmap_fallback_v1",
        data_root / "whole_outer_tower/fixed_intrinsic_pnp_colmap_fallback_v1/intrinsics",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve(strict=False)
    return candidates[0].resolve(strict=False)


def intrinsic_path_candidates(intrinsics_dir, camera_index, camera_id=""):
    intrinsics_dir = Path(intrinsics_dir)
    candidates = [intrinsics_dir / f"intrinsics{camera_index}.yaml"]
    if camera_id:
        candidates.append(intrinsics_dir / f"intrinsics{camera_index}_{camera_id}.yaml")
    if intrinsics_dir.is_dir():
        candidates.extend(sorted(intrinsics_dir.glob(f"intrinsics{camera_index}_*.yaml")))
    seen = set()
    unique = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def find_intrinsic_path(intrinsics_dir, camera_index, camera_id=""):
    for candidate in intrinsic_path_candidates(intrinsics_dir, camera_index, camera_id):
        if candidate.is_file():
            return candidate
    return None


def normalize_fixed_rig_intrinsics_yaml(text):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "type: CentralOpenCV":
            lines[index] = line.replace("CentralOpenCV", "CentralOpenCVModel", 1)
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        if stripped == "type : CentralOpenCV":
            lines[index] = line.replace("CentralOpenCV", "CentralOpenCVModel", 1)
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    return text


def csv_ints(values):
    return ",".join(str(value) for value in values)


def bridge_all32_layout(large_scan):
    cameras = large_scan.get("cameras", [])
    by_index = {row.get("index"): row for row in cameras}
    observed_topdown = {}
    for row in cameras:
        camera_id = row.get("camera_id")
        if camera_id in TOPDOWN_BRIDGE_LABELS:
            observed_topdown[camera_id] = row.get("index")

    expected_topdown = dict(zip(TOPDOWN_BRIDGE_LABEL_ORDER, BRIDGE_ALL32_TOPDOWN_INDICES))
    warnings = []
    if large_scan.get("camera_count") != BRIDGE_ALL32_CAMERA_COUNT:
        warnings.append(
            f"expected_all32_camera_count_{BRIDGE_ALL32_CAMERA_COUNT}_got_{large_scan.get('camera_count')}"
        )
    if large_scan.get("usable_camera_count") != BRIDGE_ALL32_CAMERA_COUNT:
        warnings.append(
            f"expected_all32_usable_camera_count_{BRIDGE_ALL32_CAMERA_COUNT}_got_{large_scan.get('usable_camera_count')}"
        )
    for label, expected_index in expected_topdown.items():
        observed_index = observed_topdown.get(label)
        if observed_index != expected_index:
            warnings.append(f"topdown_{label}_expected_index_{expected_index}_got_{observed_index}")
    inner_rows = [by_index.get(index) for index in BRIDGE_ALL32_INNER_INDICES]
    if any(row is None or row.get("kind") != "inner" for row in inner_rows):
        warnings.append("inner_bridge_indices_24_31_do_not_all_resolve_to_inner_cameras")

    ready = (
        large_scan.get("camera_count") == BRIDGE_ALL32_CAMERA_COUNT
        and large_scan.get("usable_camera_count") == BRIDGE_ALL32_CAMERA_COUNT
        and all(observed_topdown.get(label) == expected_index for label, expected_index in expected_topdown.items())
        and all(row is not None and row.get("kind") == "inner" for row in inner_rows)
    )
    return {
        "name": "outer24_inner8_all32",
        "ready": ready,
        "expected_camera_count": BRIDGE_ALL32_CAMERA_COUNT,
        "outer_count": BRIDGE_ALL32_OUTER_COUNT,
        "inner_count": BRIDGE_ALL32_INNER_COUNT,
        "inner_indices": list(BRIDGE_ALL32_INNER_INDICES),
        "outer_indices": list(BRIDGE_ALL32_TOPDOWN_INDICES),
        "outer_labels": list(TOPDOWN_BRIDGE_LABEL_ORDER),
        "observed_camera_count": large_scan.get("camera_count"),
        "observed_topdown_indices": observed_topdown,
        "warnings": warnings,
        "index_convention": (
            "bridge all32 order is outer cameras 0..23 followed by inner cameras 24..31; "
            "inner0..inner7 map to bridge indices 24..31."
        ),
    }


def canonical_scan_ready(scan, expected_cameras, label):
    expected_ids = [camera_id for _machine, camera_id in expected_cameras]
    usable = [row for row in scan.get("cameras", []) if row.get("status") == "usable"]
    observed_ids = [row.get("camera_id") for row in usable]
    ready = (
        bool(scan.get("normalized_inputs"))
        and len(usable) == len(expected_ids)
        and observed_ids == expected_ids
    )
    notes = []
    if not scan.get("normalized_inputs"):
        notes.append(f"{label}_inputs_are_missing_or_not_normalized")
    if len(usable) != len(expected_ids):
        notes.append(
            f"{label}_requires_{len(expected_ids)}_canonical_cameras_got_{len(usable)}"
        )
    elif observed_ids != expected_ids:
        notes.append(f"{label}_camera_order_mismatch_fixed_intrinsics_would_misindex")
    return {
        "ready": ready,
        "expected_ids": expected_ids,
        "observed_usable_ids": observed_ids,
        "notes": notes,
    }


def prepare_bridge_intrinsics(output_root, outer_intrinsics_dir, inner_intrinsics_dir, bridge_layout):
    output_dir = Path(output_root) / "planned_inputs" / "bridge_all32_fixed_intrinsics"
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    missing = []

    def copy_intrinsic(source, bridge_index, source_index, camera_id, kind):
        target = output_dir / f"intrinsics{bridge_index}.yaml"
        if target.exists() or target.is_symlink():
            target.unlink()
        row = {
            "bridge_index": bridge_index,
            "source_index": source_index,
            "camera_id": camera_id,
            "kind": kind,
            "source": str(source) if source else "",
            "target": str(target),
            "status": "missing",
        }
        if source is None:
            missing.append(row)
            entries.append(row)
            return
        shutil.copy2(source, target)
        normalized = normalize_fixed_rig_intrinsics_yaml(target.read_text(encoding="utf-8"))
        target.write_text(normalized, encoding="utf-8")
        row["status"] = "ready"
        entries.append(row)

    for outer_index, (_machine, camera_id) in enumerate(OUTER_CAMERAS):
        source = find_intrinsic_path(outer_intrinsics_dir, outer_index, camera_id)
        copy_intrinsic(source, outer_index, outer_index, camera_id, "outer")

    for inner_index, bridge_index in enumerate(bridge_layout["inner_indices"]):
        _machine, camera_id = INNER_CAMERAS[inner_index]
        source = find_intrinsic_path(inner_intrinsics_dir, inner_index, camera_id)
        copy_intrinsic(source, bridge_index, inner_index, camera_id, "inner")

    ready_count = sum(1 for row in entries if row["status"] == "ready")
    expected_count = bridge_layout["expected_camera_count"]
    return {
        "output_dir": str(output_dir.resolve(strict=False)),
        "outer_intrinsics_dir": str(Path(outer_intrinsics_dir).resolve(strict=False)),
        "inner_intrinsics_dir": str(Path(inner_intrinsics_dir).resolve(strict=False)),
        "ready": ready_count == expected_count,
        "ready_count": ready_count,
        "expected_count": expected_count,
        "missing_count": len(missing),
        "missing": missing,
        "entries": entries,
    }


def bridge_candidates(large_scan):
    usable = [row for row in large_scan["cameras"] if row["status"] == "usable"]
    usable_inner = [row for row in usable if row["kind"] == "inner"]
    usable_outer = [row for row in usable if row["kind"] == "outer"]
    topdown = [row for row in usable_outer if row["camera_id"] in TOPDOWN_BRIDGE_LABELS]
    candidates = []
    topdown_labels = sorted(row["camera_id"] for row in topdown)
    if len(topdown) == 3 and len(usable_inner) >= 4:
        status = "candidate"
        reason = "topdown_bridge_anchors_and_inner_cameras_present"
    else:
        status = "blocked"
        reason = "need_4-1_4-2_4-3_and_at_least_4_inner_cameras"
    candidates.append({
        "name": "large_marker_topdown_outer_bridge",
        "status": status,
        "reason": reason,
        "usable_inner_count": len(usable_inner),
        "usable_outer_count": len(usable_outer),
        "topdown_labels": topdown_labels,
    })
    candidates.append({
        "name": "large_marker_all32_diagnostic",
        "status": "candidate" if len(usable) >= 16 else "weak",
        "reason": "diagnostic_all_visible_cameras_from_large_marker",
        "usable_camera_count": len(usable),
        "common_frame_count": large_scan["common_frame_count"],
    })
    return candidates


def safe_name(value, fallback):
    name = Path(str(value or "")).name
    return name or fallback


def stage_requested(args, group):
    explicit = set(args.run_stage or [])
    if args.run_large_inner_init:
        explicit.add("large-inner-init")
    if args.run_small_fixed_rig_quality:
        explicit.add("small-fixed-rig-quality")
    if args.run_small_refine:
        if args.inner_refine_mode == "fixed_rig":
            explicit.add("small-fixed-rig-quality")
        else:
            explicit.add("small-refine")
    if args.run_large_bridge:
        explicit.add("large-bridge")
    if args.run_reports:
        explicit.add("reports")
    return bool(args.run_all or group in explicit)


def inner_refine_plan(args, output_root, small_out, inner_prior):
    fixed_rig_quality_state = small_out / "fixed_intrinsic_small_grid4_quality_probe_v1"
    fixed_state = small_out / "fixed_intrinsic_small_grid4_warm_start_fast_v1"
    joint_state = small_out / "joint_intrinsic_small_grid4_warm_start_fast_v1"
    small_refine_requested = stage_requested(args, "small-refine")

    mode = args.inner_refine_mode
    fixed_rig_quality_enabled = mode == "fixed_rig"
    fixed_enabled = mode in {"fixed", "joint", "fixed_then_joint"}
    joint_enabled = mode in {"joint", "fixed_then_joint"}

    if mode == "fixed_rig":
        selected_state = Path(inner_prior)
        joint_input_state = Path(inner_prior)
    elif mode == "fixed":
        selected_state = fixed_state if (small_refine_requested or (fixed_state / "camera_tr_rig.yaml").exists()) else Path(inner_prior)
        joint_input_state = Path(inner_prior)
    elif mode == "joint":
        selected_state = joint_state if (small_refine_requested or (joint_state / "camera_tr_rig.yaml").exists()) else Path(inner_prior)
        joint_input_state = fixed_state if (small_refine_requested or (fixed_state / "camera_tr_rig.yaml").exists()) else Path(inner_prior)
    else:
        if small_refine_requested or (joint_state / "camera_tr_rig.yaml").exists():
            selected_state = joint_state
        elif (fixed_state / "camera_tr_rig.yaml").exists():
            selected_state = fixed_state
        else:
            selected_state = Path(inner_prior)
        joint_input_state = fixed_state if (small_refine_requested or (fixed_state / "camera_tr_rig.yaml").exists()) else Path(inner_prior)

    return {
        "mode": mode,
        "fixed_rig_quality_enabled": fixed_rig_quality_enabled,
        "fixed_rig_quality_state": fixed_rig_quality_state,
        "fixed_enabled": fixed_enabled,
        "joint_enabled": joint_enabled,
        "fixed_state": fixed_state,
        "joint_state": joint_state,
        "joint_input_state": joint_input_state,
        "selected_state": selected_state,
        "selected_camera_tr_rig": selected_state / "camera_tr_rig.yaml",
        "fixed_stage_needed_for_joint": mode in {"joint", "fixed_then_joint"},
        "report_root": output_root / "reports",
    }


def stage_status(args, ready, requested=False, existing_output=None):
    if existing_output and Path(existing_output).exists() and not args.force:
        return "reused_existing"
    if not ready:
        return "blocked_missing_inputs"
    if args.dry_run:
        return "skipped_dry_run"
    if requested:
        return "ready_to_run"
    return "not_requested_reuse_existing"


def make_stage(name, status, inputs, outputs, argv=None, planned_command="", duration=0.0, notes=None, group="internal", allow_failure=False):
    command = planned_command or command_string(argv)
    return {
        "name": name,
        "group": group,
        "status": status,
        "inputs": inputs,
        "outputs": outputs,
        "planned_command": command,
        "requested": status == "ready_to_run",
        "log_path": "",
        "duration": duration,
        "started_at": "",
        "finished_at": "",
        "notes": notes or [],
        "allow_failure": bool(allow_failure),
    }


def set_stage_timing(stages, name, started_at, finished_at, duration):
    for stage in stages:
        if stage.get("name") == name:
            stage["started_at"] = started_at
            stage["finished_at"] = finished_at
            stage["duration"] = duration_s(duration)
            return


def path_signature(value):
    if isinstance(value, dict):
        return {key: path_signature(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [path_signature(item) for item in value]
    if not isinstance(value, str):
        return value
    path = Path(value)
    try:
        exists = path.exists()
    except (OSError, ValueError):
        exists = False
    if not exists:
        return value
    try:
        stat = path.stat()
    except OSError:
        return {"path": value, "exists": True, "stat_error": True}
    if path.is_file():
        return {
            "path": value,
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    if path.is_dir():
        child_count = 0
        newest_mtime_ns = stat.st_mtime_ns
        try:
            for child in path.iterdir():
                child_count += 1
                try:
                    newest_mtime_ns = max(newest_mtime_ns, child.stat().st_mtime_ns)
                except OSError:
                    pass
        except OSError:
            pass
        return {
            "path": value,
            "kind": "directory",
            "mtime_ns": stat.st_mtime_ns,
            "child_count": child_count,
            "newest_child_mtime_ns": newest_mtime_ns,
        }
    return {
        "path": value,
        "kind": "other",
        "mtime_ns": stat.st_mtime_ns,
    }


def stage_fingerprint(stage):
    payload = {
        "name": stage.get("name"),
        "group": stage.get("group"),
        "command": stage.get("planned_command", ""),
        "inputs": path_signature(stage.get("inputs", {})),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def stage_fingerprint_path(output_root, stage):
    return Path(output_root) / "stage_fingerprints" / f"{stage['name']}.json"


def fingerprint_matches(output_root, stage):
    path = stage_fingerprint_path(output_root, stage)
    if not path.is_file():
        return False
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return previous.get("sha256") == stage_fingerprint(stage)["sha256"]


def write_stage_fingerprint(output_root, stage):
    path = stage_fingerprint_path(output_root, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stage_fingerprint(stage)
    payload["written_at"] = utc_now()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_outputs_exist(stage):
    outputs = stage.get("outputs", {})
    if not isinstance(outputs, dict):
        return False
    for value in outputs.values():
        if isinstance(value, str) and Path(value).exists():
            return True
    return False


def collect_path_values(value):
    paths = set()
    if isinstance(value, (str, Path)):
        text = str(value)
        if text:
            paths.add(str(Path(text)))
    elif isinstance(value, dict):
        for item in value.values():
            paths.update(collect_path_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            paths.update(collect_path_values(item))
    return paths


def clear_stage_outputs(stage, output_root):
    outputs = stage.get("outputs", {})
    if not isinstance(outputs, dict):
        return
    input_paths = collect_path_values(stage.get("inputs", {}))
    for value in outputs.values():
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if str(path) in input_paths:
            continue
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    fingerprint = stage_fingerprint_path(output_root, stage)
    if fingerprint.exists():
        fingerprint.unlink()


def build_pipeline_stages(
    args,
    repo_root,
    data_root,
    output_root,
    small_scan,
    large_inner_scan,
    large_scan,
    planned_inputs,
    priors,
    bridge_layout,
    bridge_intrinsics,
):
    small_input = planned_inputs["small_marker"]["image_directories_file"]
    small_manifest = planned_inputs["small_marker"]["manifest"]
    large_inner_input = planned_inputs["large_inner_marker"]["image_directories_file"]
    large_inner_manifest = planned_inputs["large_inner_marker"]["manifest"]
    large_input = planned_inputs["large_marker"]["image_directories_file"]
    large_manifest = planned_inputs["large_marker"]["manifest"]

    small_out = output_root / safe_name(args.small_marker_sequence, "small_marker_inner8")
    large_inner_out = output_root / safe_name(args.large_inner_marker_sequence, "large_marker_inner8")
    large_out = output_root / safe_name(args.large_marker_sequence, "large_marker_bridge_all32")
    reports_out = output_root / "reports"
    bridge_out = output_root / "bridge_colmap_inner_refined_v1"
    combined_viewer_html = output_root / "combined_studio_rig_viewer_v1/index.html"

    small_features = small_out / "features_parallel_pattern3_fast_v1.bin"
    small_grid = small_out / f"features_pattern3_grid4_stride{args.small_frame_stride}_fast_v1.bin"
    large_inner_features = large_inner_out / "features_parallel_pattern0_inner_init_v1.bin"
    large_inner_stride_features = large_inner_out / f"features_parallel_pattern0_inner_init_stride{args.large_inner_frame_stride}_v1.bin"
    large_inner_init_state = large_inner_out / "fixed_intrinsic_large_marker_inner8_init_v1"

    small_canonical = canonical_scan_ready(small_scan, INNER_CAMERAS, "small_marker_inner8")
    large_inner_canonical = canonical_scan_ready(
        large_inner_scan, INNER_CAMERAS, "large_marker_inner8")
    bridge_canonical = canonical_scan_ready(
        large_scan, OUTER_CAMERAS + INNER_CAMERAS, "large_marker_bridge_all32")
    small_ready = small_canonical["ready"]
    large_inner_ready = large_inner_canonical["ready"]
    large_ready = large_scan["normalized_inputs"] and large_scan["usable_camera_count"] > 0
    inner_intrinsics_ready = Path(priors["inner_intrinsics"]).exists()
    bridge_intrinsics_ready = bridge_intrinsics["ready"]
    outer_prior_ready = Path(priors["outer_prior"]).exists()
    outer_final_pose_yaml = priors.get("outer_final_pose_yaml", "")
    outer_final_pose_ready = bool(outer_final_pose_yaml and Path(outer_final_pose_yaml).is_file())
    viewer_outer_pose_ready = outer_final_pose_ready or outer_prior_ready
    viewer_outer_pose_source = "outer_final_pose_yaml" if outer_final_pose_ready else "colmap_sim3_approx"
    bridge_layout_ready = bridge_layout["ready"]
    bridge_input_ready = large_ready and bridge_layout_ready and bridge_canonical["ready"]
    large_inner_init_requested = stage_requested(args, "large-inner-init")
    large_inner_init_complete = (large_inner_init_state / "camera_tr_rig.yaml").exists()
    large_inner_init_dependency_ready = (
        large_inner_init_complete
        or (large_inner_init_requested and large_inner_ready and inner_intrinsics_ready)
    )
    use_large_inner_init_prior = (
        (large_inner_init_requested and large_inner_ready)
        or (large_inner_init_complete and not large_inner_init_requested)
    )
    effective_inner_prior = large_inner_init_state if use_large_inner_init_prior else Path(priors["inner_prior"])
    inner_prior_source = "large_inner_initializer" if use_large_inner_init_prior else "configured_inner_prior"

    inner_plan = inner_refine_plan(args, output_root, small_out, effective_inner_prior)
    small_fixed_rig_quality = inner_plan["fixed_rig_quality_state"]
    inner_fixed_refine = inner_plan["fixed_state"]
    inner_joint_refine = inner_plan["joint_state"]
    selected_inner_state = inner_plan["selected_state"]
    selected_inner_camera_tr_rig = inner_plan["selected_camera_tr_rig"]
    final_inner_state = Path(effective_inner_prior)
    final_inner_camera_tr_rig = final_inner_state / "camera_tr_rig.yaml"
    small_fixed_rig_quality_dataset = small_grid
    fixed_refine_dataset = inner_fixed_refine / "dataset.bin"
    joint_refine_dataset = inner_joint_refine / "dataset.bin"
    joint_input_dataset = fixed_refine_dataset if inner_plan["mode"] in {"joint", "fixed_then_joint"} else small_grid
    selected_inner_dataset = (
        joint_refine_dataset if selected_inner_state == inner_joint_refine else
        fixed_refine_dataset if selected_inner_state == inner_fixed_refine else
        large_inner_stride_features if selected_inner_state == large_inner_init_state else
        Path(selected_inner_state) / "dataset.bin"
    )
    final_inner_dataset = (
        large_inner_stride_features if final_inner_state == large_inner_init_state else
        joint_refine_dataset if final_inner_state == inner_joint_refine else
        fixed_refine_dataset if final_inner_state == inner_fixed_refine else
        Path(final_inner_state) / "dataset.bin"
    )
    selected_inner_manifest = large_inner_manifest if selected_inner_state == large_inner_init_state else small_manifest
    final_inner_manifest = large_inner_manifest if final_inner_state == large_inner_init_state else small_manifest
    inner_reproj = reports_out / "inner_reprojection"
    rig_report = reports_out / "rig_extrinsics"
    inner_viewer = reports_out / "interactive_inner_viewer"
    large_features = large_out / "features_parallel_pattern0_bridge_v1.bin"
    large_stride_features = large_out / f"features_parallel_pattern0_bridge_stride{args.large_frame_stride}_v1.bin"
    large_bridge_pnp = large_out / f"fixed_intrinsic_bridge_pnp_stride{args.large_frame_stride}_v1"

    inner_prior_ready = Path(effective_inner_prior).exists() or large_inner_init_dependency_ready
    small_quality_requested = stage_requested(args, "small-fixed-rig-quality")
    small_refine_requested = stage_requested(args, "small-refine")
    small_processing_requested = small_quality_requested or small_refine_requested
    large_bridge_requested = stage_requested(args, "large-bridge")
    small_fixed_rig_enabled = inner_plan["fixed_rig_quality_enabled"] or small_quality_requested
    small_fixed_rig_ready = small_ready and inner_intrinsics_ready and small_fixed_rig_enabled
    fixed_refine_ready = small_ready and inner_prior_ready and inner_plan["fixed_enabled"]
    if inner_plan["mode"] in {"joint", "fixed_then_joint"}:
        joint_input_ready = inner_prior_ready and (
            small_refine_requested or (inner_fixed_refine / "camera_tr_rig.yaml").exists()
        )
    else:
        joint_input_ready = inner_prior_ready
    joint_refine_ready = small_ready and inner_plan["joint_enabled"] and joint_input_ready
    if final_inner_state == large_inner_init_state:
        final_inner_ready = large_inner_init_dependency_ready
    else:
        final_inner_ready = final_inner_state.exists()
    bridge_summary_json = bridge_out / "bridge_summary.json"
    bridge_pose_yaml = bridge_out / "camera_tr_inner_refined_plus_outer_topdown.yaml"
    bridge_ready = (
        (bridge_summary_json.exists() and bridge_pose_yaml.exists())
        or (large_bridge_requested and outer_prior_ready)
    )
    final_report_dataset_ready = (
        large_inner_ready if final_inner_state == large_inner_init_state else small_ready
    )
    small_quality_notes = [
        "quality_probe_only_does_not_override_large_inner_baseline",
    ]
    if small_scan["usable_camera_count"] != len(INNER_CAMERAS):
        small_quality_notes.append(
            f"input_has_{small_scan['usable_camera_count']}_of_{len(INNER_CAMERAS)}_usable_inner_cameras"
        )
    if small_canonical["notes"]:
        small_quality_notes.extend(small_canonical["notes"])

    combined_viewer_inputs = {
        "inner_bridge_pose_yaml": str(bridge_out / "camera_tr_inner_refined_plus_outer_topdown.yaml"),
        "bridge_summary_json": str(bridge_out / "bridge_summary.json"),
        "outer_colmap_images_txt": priors["outer_prior"],
        "outer_final_pose_yaml": outer_final_pose_yaml,
        "outer_pose_source": viewer_outer_pose_source,
        "combined_image_directories_file": large_input,
        "whole_coverage_tsv": str(data_root / "whole_outer24_filtered_min4_hybrid_min4cam" / "per_camera_stats.tsv"),
        "large_marker_pnp_summary_tsv": str(large_bridge_pnp / "camera_pnp_summary.tsv"),
        "small_marker_pnp_summary_tsv": str(small_fixed_rig_quality / "camera_pnp_summary.tsv"),
        "large_marker_board_pose_yaml": str(large_inner_init_state / "rig_tr_global.yaml"),
        "small_marker_board_pose_yaml": str(small_fixed_rig_quality / "rig_tr_global.yaml"),
        "bridge_marker_board_pose_yaml": str(large_bridge_pnp / "rig_tr_global.yaml"),
        "inner_bridge_indices": csv_ints(bridge_layout["inner_indices"]),
        "topdown_bridge_indices": csv_ints(bridge_layout["outer_indices"]),
    }
    combined_viewer_argv = [
        DEFAULT_T0_PYTHON, str(repo_root / "scripts/calib/generate_combined_studio_rig_viewer.py"),
        "--inner_bridge_pose_yaml", str(bridge_out / "camera_tr_inner_refined_plus_outer_topdown.yaml"),
        "--bridge_summary_json", str(bridge_out / "bridge_summary.json"),
        "--outer_colmap_images_txt", priors["outer_prior"],
        "--combined_image_directories_file", large_input,
        "--inner_bridge_indices", csv_ints(bridge_layout["inner_indices"]),
        "--topdown_bridge_indices", csv_ints(bridge_layout["outer_indices"]),
        "--topdown_labels", ",".join(bridge_layout["outer_labels"]),
        "--output_html", str(combined_viewer_html),
        "--viewer_scope", "combined",
        "--title", "Fast Inner/Outer Bridge Viewer",
        "--inner_reprojection_metrics_tsv", str(inner_reproj / "camera_metrics.tsv"),
        "--whole_coverage_tsv", str(data_root / "whole_outer24_filtered_min4_hybrid_min4cam" / "per_camera_stats.tsv"),
        "--large_marker_pnp_summary_tsv", str(large_bridge_pnp / "camera_pnp_summary.tsv"),
        "--small_marker_pnp_summary_tsv", str(small_fixed_rig_quality / "camera_pnp_summary.tsv"),
        "--large_marker_board_pose_yaml", str(large_inner_init_state / "rig_tr_global.yaml"),
        "--small_marker_board_pose_yaml", str(small_fixed_rig_quality / "rig_tr_global.yaml"),
        "--bridge_marker_board_pose_yaml", str(large_bridge_pnp / "rig_tr_global.yaml"),
    ]
    bridge_intrinsics_dir = Path(bridge_intrinsics["output_dir"])
    if bridge_intrinsics_dir.is_dir():
        combined_viewer_argv.extend([
            "--inner_intrinsics_dir", str(bridge_intrinsics_dir),
            "--inner_intrinsics_index_offset", "24",
        ])
    elif (final_inner_state / "intrinsics0.yaml").is_file():
        combined_viewer_argv.extend(["--inner_intrinsics_dir", str(final_inner_state)])
    elif Path(priors["inner_intrinsics"]).exists():
        combined_viewer_argv.extend(["--inner_intrinsics_dir", priors["inner_intrinsics"]])
    if outer_final_pose_ready:
        combined_viewer_argv.extend(["--outer_final_pose_yaml", outer_final_pose_yaml])
        outer_final_parent = Path(outer_final_pose_yaml).parent
        outer_tower_pose_yaml = outer_final_parent / "rig_tr_global.yaml"
        if outer_tower_pose_yaml.is_file():
            combined_viewer_argv.extend(["--tower_pose_yaml", str(outer_tower_pose_yaml)])
        outer_reprojection_tsv = outer_final_parent / "diagnostics/camera_reprojection.tsv"
        if outer_reprojection_tsv.is_file():
            combined_viewer_argv.extend(["--outer_reprojection_tsv", str(outer_reprojection_tsv)])
        outer_intrinsics_dir = outer_final_parent / "intrinsics_refined"
        if outer_intrinsics_dir.is_dir():
            combined_viewer_argv.extend(["--outer_intrinsics_dir", str(outer_intrinsics_dir)])

    stages = [
        make_stage(
            "data_quality_scan",
            "complete",
            {
                "small_marker": small_scan["path"],
                "large_inner_marker": large_inner_scan["path"],
                "large_marker": large_scan["path"],
                "max_tail_trim": small_scan["max_tail_trim"],
            },
            {
                "summary_json": str(output_root / "summary.json"),
                "index_html": str(output_root / "index.html"),
            },
            planned_command="internal scan",
            duration=small_scan["duration"] + large_inner_scan["duration"] + large_scan["duration"],
            group="quality",
        ),
        make_stage(
            "prepare_bridge_all32_fixed_intrinsics",
            "complete" if bridge_intrinsics_ready else "blocked_missing_inputs",
            {
                "outer_intrinsics_directory": priors["outer_intrinsics"],
                "inner_intrinsics_directory": priors["inner_intrinsics"],
                "index_convention": bridge_layout["index_convention"],
            },
            {
                "fixed_intrinsics_directory": priors["bridge_intrinsics"],
                "ready_count": bridge_intrinsics["ready_count"],
                "expected_count": bridge_intrinsics["expected_count"],
            },
            planned_command="internal copy outer intrinsics0..23 and remap inner intrinsics0..7 to intrinsics24..31",
            notes=[] if bridge_intrinsics_ready else [
                f"missing_{bridge_intrinsics['missing_count']}_bridge_intrinsics_files"
            ],
            group="large-bridge",
        ),
        make_stage(
            "extract_large_inner_marker_features",
            stage_status(args, large_inner_ready, stage_requested(args, "large-inner-init"), large_inner_features),
            {
                "image_directories_file": large_inner_input,
                "pattern": LARGE_MARKER_PATTERN,
            },
            {
                "dataset": str(large_inner_features),
                "work_dir": str(large_inner_out / "parallel_shards_pattern0_inner_init_v1"),
            },
            argv=[
                DEFAULT_T0_PYTHON,
                str(DEFAULT_T0_REPO / "scripts/calib/parallel_extract_features.py"),
                "--binary", DEFAULT_T0_BINARY,
                "--repo-root", str(DEFAULT_T0_REPO),
                "--image-directories-file", large_inner_input,
                "--pattern-files", LARGE_MARKER_PATTERN,
                "--output-dataset", str(large_inner_features),
                "--work-dir", str(large_inner_out / "parallel_shards_pattern0_inner_init_v1"),
                "--jobs", "8",
                "--resume",
            ],
            notes=[] if large_inner_ready else large_inner_canonical["notes"],
            group="large-inner-init",
        ),
        make_stage(
            "subsample_large_inner_marker_frames",
            stage_status(args, large_inner_ready, stage_requested(args, "large-inner-init"), large_inner_stride_features),
            {"dataset": str(large_inner_features), "frame_stride": args.large_inner_frame_stride},
            {"dataset": str(large_inner_stride_features)},
            argv=[
                DEFAULT_T0_BINARY,
                "--subsample_dataset",
                "--dataset_files", str(large_inner_features),
                "--dataset_output_path", str(large_inner_stride_features),
                "--subsample_frame_stride", str(args.large_inner_frame_stride),
                "--subsample_min_features_per_camera_view", "12",
            ],
            group="large-inner-init",
        ),
        make_stage(
            "estimate_large_inner_fixed_intrinsic_rig",
            stage_status(
                args,
                large_inner_ready and inner_intrinsics_ready,
                stage_requested(args, "large-inner-init"),
                large_inner_init_state / "camera_tr_rig.yaml",
            ),
            {
                "dataset": str(large_inner_stride_features),
                "fixed_intrinsics_directory": priors["inner_intrinsics"],
                "manifest": large_inner_manifest,
            },
            {
                "state_dir": str(large_inner_init_state),
                "camera_tr_rig": str(large_inner_init_state / "camera_tr_rig.yaml"),
                "dataset": str(large_inner_stride_features),
            },
            argv=[
                DEFAULT_T0_BINARY,
                "--estimate_fixed_intrinsic_rig",
                "--dataset_files", str(large_inner_stride_features),
                "--fixed_intrinsics_directory", priors["inner_intrinsics"],
                "--camera_manifest", large_inner_manifest,
                "--output_directory", str(large_inner_init_state),
            ],
            notes=(
                [] if (large_inner_ready and inner_intrinsics_ready) else
                ["inner fixed intrinsics directory is missing"] if not inner_intrinsics_ready else
                large_inner_canonical["notes"]
            ),
            group="large-inner-init",
        ),
        make_stage(
            "extract_small_marker_features",
            stage_status(args, small_ready, small_processing_requested, small_features),
            {
                "image_directories_file": small_input,
                "pattern": SMALL_MARKER_PATTERN,
            },
            {
                "dataset": str(small_features),
                "work_dir": str(small_out / "parallel_shards_pattern3_fast_v1"),
            },
            argv=[
                DEFAULT_T0_PYTHON,
                str(DEFAULT_T0_REPO / "scripts/calib/parallel_extract_features.py"),
                "--binary", DEFAULT_T0_BINARY,
                "--repo-root", str(DEFAULT_T0_REPO),
                "--image-directories-file", small_input,
                "--pattern-files", SMALL_MARKER_PATTERN,
                "--output-dataset", str(small_features),
                "--work-dir", str(small_out / "parallel_shards_pattern3_fast_v1"),
                "--jobs", "8",
                "--resume",
            ],
            notes=[] if small_ready else small_canonical["notes"],
            group="small-marker",
        ),
        make_stage(
            "subsample_small_marker_grid4",
            stage_status(args, small_ready, small_processing_requested, small_grid),
            {"dataset": str(small_features)},
            {"dataset": str(small_grid), "frame_stride": args.small_frame_stride},
            argv=[
                DEFAULT_T0_BINARY,
                "--subsample_dataset",
                "--dataset_files", str(small_features),
                "--dataset_output_path", str(small_grid),
                "--subsample_frame_stride", str(args.small_frame_stride),
                "--subsample_pattern_grid_stride", "4",
                "--subsample_min_features_per_camera_view", "20",
            ],
            group="small-marker",
        ),
        make_stage(
            "estimate_small_marker_fixed_rig_quality",
            stage_status(
                args,
                small_fixed_rig_ready,
                small_quality_requested,
                small_fixed_rig_quality / "camera_pnp_summary.tsv",
            ),
            {
                "dataset": str(small_grid),
                "fixed_intrinsics_directory": priors["inner_intrinsics"],
                "manifest": small_manifest,
                "refine_mode": args.inner_refine_mode,
            },
            {
                "state_dir": str(small_fixed_rig_quality),
                "camera_tr_rig": str(small_fixed_rig_quality / "camera_tr_rig.yaml"),
                "camera_tr_rig_used": str(small_fixed_rig_quality / "camera_tr_rig_used.yaml"),
                "dataset": str(small_fixed_rig_quality_dataset),
                "pnp_views": str(small_fixed_rig_quality / "pnp_views.tsv"),
                "camera_pnp_summary": str(small_fixed_rig_quality / "camera_pnp_summary.tsv"),
            },
            argv=[
                DEFAULT_T0_BINARY,
                "--estimate_fixed_intrinsic_rig",
                "--dataset_files", str(small_grid),
                "--fixed_intrinsics_directory", priors["inner_intrinsics"],
                "--camera_manifest", small_manifest,
                "--output_directory", str(small_fixed_rig_quality),
            ],
            notes=(
                small_quality_notes if small_fixed_rig_ready else
                ["small fixed-rig quality probe disabled by --inner-refine-mode"] if not small_fixed_rig_enabled else
                ["inner fixed intrinsics directory is missing"] if not inner_intrinsics_ready else
                small_canonical["notes"]
            ),
            group="small-fixed-rig-quality",
            allow_failure=True,
        ),
        make_stage(
            "refine_inner_from_prior",
            stage_status(
                args,
                fixed_refine_ready,
                small_refine_requested,
                (inner_fixed_refine / "camera_tr_rig.yaml") if inner_plan["fixed_enabled"] else None,
            ),
            {
                "dataset": str(small_grid),
                "inner_prior": str(effective_inner_prior),
                "inner_prior_source": inner_prior_source,
                "configured_inner_prior": priors["inner_prior"],
                "manifest": small_manifest,
                "refine_mode": args.inner_refine_mode,
            },
            {
                "state_dir": str(inner_fixed_refine),
                "camera_tr_rig": str(inner_fixed_refine / "camera_tr_rig.yaml"),
                "dataset": str(fixed_refine_dataset),
            },
            argv=[
                DEFAULT_T0_BINARY,
                "--dataset_files", str(small_grid),
                "--state_directory", str(effective_inner_prior),
                "--output_directory", str(inner_fixed_refine),
                "--localize_only",
                "--num_pyramid_levels", "1",
                "--outlier_removal_factor", "0",
                "--max_ba_iterations", str(args.inner_fixed_max_ba_iterations),
                "--schur_mode", args.inner_schur_mode,
                *(["--skip_bundle_adjustment"] if args.inner_fixed_max_ba_iterations == 0 else []),
                "--skip_calibration_report",
            ] if inner_plan["fixed_enabled"] else [],
            notes=(
                [] if fixed_refine_ready else
                ["inner fixed refine disabled by --inner-refine-mode"] if not inner_plan["fixed_enabled"] else
                small_canonical["notes"] if not small_ready else
                ["large-inner initializer state will be produced earlier in this run"] if use_large_inner_init_prior and large_inner_init_requested else
                ["large-inner initializer output is missing"] if use_large_inner_init_prior else
                ["inner prior state is missing"]
            ),
            group="small-refine",
        ),
        make_stage(
            "joint_refine_inner_intrinsics_extrinsics",
            stage_status(
                args,
                joint_refine_ready,
                small_refine_requested,
                (inner_joint_refine / "camera_tr_rig.yaml") if inner_plan["joint_enabled"] else None,
            ),
            {
                "dataset": str(joint_input_dataset),
                "state_directory": str(inner_plan["joint_input_state"]),
                "manifest": small_manifest,
                "model": args.inner_model,
                "refine_mode": args.inner_refine_mode,
            },
            {
                "state_dir": str(inner_joint_refine),
                "camera_tr_rig": str(inner_joint_refine / "camera_tr_rig.yaml"),
                "dataset": str(joint_refine_dataset),
            },
            argv=[
                DEFAULT_T0_BINARY,
                "--dataset_files", str(joint_input_dataset),
                "--state_directory", str(inner_plan["joint_input_state"]),
                "--output_directory", str(inner_joint_refine),
                "--model", args.inner_model,
                "--num_pyramid_levels", "1",
                "--outlier_removal_factor", str(args.inner_joint_outlier_removal_factor),
                "--max_ba_iterations", str(args.inner_joint_max_ba_iterations),
                "--schur_mode", args.inner_schur_mode,
                "--skip_calibration_report",
            ] if inner_plan["joint_enabled"] else [],
            notes=(
                [] if joint_refine_ready else
                ["inner joint refine disabled by --inner-refine-mode"] if not inner_plan["joint_enabled"] else
                small_canonical["notes"] if not small_ready else
                ["fixed warm-start state will be produced earlier in this run"] if inner_plan["mode"] == "fixed_then_joint" and small_refine_requested else
                ["inner joint input state is missing"]
            ),
            group="small-refine",
        ),
        make_stage(
            "generate_inner_reports",
            stage_status(args, final_report_dataset_ready and final_inner_ready, stage_requested(args, "reports"), inner_reproj / "index.html"),
            {
                "dataset": str(final_inner_dataset),
                "state_dir": str(final_inner_state),
                "manifest": final_inner_manifest,
            },
            {
                "inner_reprojection": str(inner_reproj / "index.html"),
                "rig_extrinsics": str(rig_report / "index.html"),
                "interactive_viewer": str(inner_viewer / "index.html"),
            },
            planned_command=" && ".join([
                command_string([
                    DEFAULT_T0_PYTHON, str(repo_root / "scripts/calib/generate_inner_calibration_report.py"),
                    "--dataset", str(final_inner_dataset),
                    "--state-dir", str(final_inner_state),
                    "--output-dir", str(inner_reproj),
                    "--manifest", final_inner_manifest,
                ]),
                command_string([
                    DEFAULT_T0_PYTHON, str(repo_root / "scripts/calib/generate_rig_extrinsics_report.py"),
                    "--state-dir", str(final_inner_state),
                    "--output-dir", str(rig_report),
                ]),
                command_string([
                    DEFAULT_T0_PYTHON, str(repo_root / "scripts/calib/generate_threejs_rig_viewer.py"),
                    "--pose-yaml", str(rig_report / "camera_tr_camera0.yaml"),
                    "--metrics-tsv", str(rig_report / "camera_tr_camera0.tsv"),
                    "--output-dir", str(inner_viewer),
                    "--title", "Fast Inner Recalibration Report",
                    "--reprojection-report", "refined=" + str(inner_reproj),
                ]),
            ]),
            group="reports",
        ),
        make_stage(
            "extract_large_marker_bridge_features",
            stage_status(args, bridge_input_ready, stage_requested(args, "large-bridge"), large_features),
            {
                "image_directories_file": large_input,
                "pattern": LARGE_MARKER_PATTERN,
            },
            {
                "dataset": str(large_features),
                "work_dir": str(large_out / "parallel_shards_pattern0_bridge_v1"),
            },
            argv=[
                DEFAULT_T0_PYTHON,
                str(DEFAULT_T0_REPO / "scripts/calib/parallel_extract_features.py"),
                "--binary", DEFAULT_T0_BINARY,
                "--repo-root", str(DEFAULT_T0_REPO),
                "--image-directories-file", large_input,
                "--pattern-files", LARGE_MARKER_PATTERN,
                "--output-dataset", str(large_features),
                "--work-dir", str(large_out / "parallel_shards_pattern0_bridge_v1"),
                "--jobs", "8",
                "--resume",
            ],
            notes=(
                [] if bridge_input_ready else
                bridge_canonical["notes"] if large_ready and not bridge_canonical["ready"] else
                bridge_layout["warnings"] if large_ready and not bridge_layout_ready else
                ["large_marker inputs are missing or not normalized"]
            ),
            group="large-bridge",
        ),
        make_stage(
            "subsample_large_marker_bridge_frames",
            stage_status(args, bridge_input_ready, stage_requested(args, "large-bridge"), large_stride_features),
            {"dataset": str(large_features), "frame_stride": args.large_frame_stride},
            {"dataset": str(large_stride_features)},
            argv=[
                DEFAULT_T0_BINARY,
                "--subsample_dataset",
                "--dataset_files", str(large_features),
                "--dataset_output_path", str(large_stride_features),
                "--subsample_frame_stride", str(args.large_frame_stride),
                "--subsample_min_features_per_camera_view", "12",
            ],
            group="large-bridge",
        ),
        make_stage(
            "estimate_large_marker_bridge_pnp",
            stage_status(args, bridge_input_ready and bridge_intrinsics_ready, stage_requested(args, "large-bridge"), large_bridge_pnp / "pnp_views.tsv"),
            {
                "dataset": str(large_stride_features),
                "fixed_intrinsics_directory": priors["bridge_intrinsics"],
                "manifest": large_manifest,
                "index_convention": bridge_layout["index_convention"],
            },
            {
                "state_dir": str(large_bridge_pnp),
                "pnp_views": str(large_bridge_pnp / "pnp_views.tsv"),
                "camera_pnp_summary": str(large_bridge_pnp / "camera_pnp_summary.tsv"),
            },
            argv=[
                DEFAULT_T0_BINARY,
                "--estimate_fixed_intrinsic_rig",
                "--dataset_files", str(large_stride_features),
                "--fixed_intrinsics_directory", priors["bridge_intrinsics"],
                "--camera_manifest", large_manifest,
                "--output_directory", str(large_bridge_pnp),
            ],
            notes=(
                ["all32 bridge uses outer intrinsics0..23 and inner intrinsics remapped to 24..31"]
                if bridge_intrinsics_ready else
                [f"missing_{bridge_intrinsics['missing_count']}_bridge_intrinsics_files"]
            ),
            group="large-bridge",
            allow_failure=True,
        ),
        make_stage(
            "evaluate_topdown_bridge",
            stage_status(args, bridge_input_ready and outer_prior_ready, stage_requested(args, "large-bridge"), bridge_out / "bridge_summary.json"),
            {
                "pnp_views": str(large_bridge_pnp / "pnp_views.tsv"),
                "inner_camera_tr_rig": str(final_inner_camera_tr_rig),
                "outer_prior": priors["outer_prior"],
                "outer_final_pose_yaml": outer_final_pose_yaml,
                "inner_indices": csv_ints(bridge_layout["inner_indices"]),
                "outer_indices": csv_ints(bridge_layout["outer_indices"]),
                "outer_labels": ",".join(bridge_layout["outer_labels"]),
            },
            {
                "bridge_summary": str(bridge_out / "bridge_summary.json"),
                "bridge_pose_yaml": str(bridge_out / "camera_tr_inner_refined_plus_outer_topdown.yaml"),
                "index_html": str(bridge_out / "index.html"),
            },
            argv=[
                DEFAULT_T0_PYTHON, str(repo_root / "scripts/calib/evaluate_inner_outer_bridge.py"),
                "--pnp_views", str(large_bridge_pnp / "pnp_views.tsv"),
                "--inner_camera_tr_rig", str(final_inner_camera_tr_rig),
                "--colmap_images", priors["outer_prior"],
                "--output_dir", str(bridge_out),
                "--inner_indices", csv_ints(bridge_layout["inner_indices"]),
                "--outer_indices", csv_ints(bridge_layout["outer_indices"]),
                "--outer_labels", ",".join(bridge_layout["outer_labels"]),
                *(["--outer_camera_tr_rig", outer_final_pose_yaml] if outer_final_pose_ready else []),
            ],
            notes=[] if outer_prior_ready else ["outer prior / COLMAP images path is missing"],
            group="large-bridge",
        ),
        make_stage(
            "generate_combined_bridge_viewer",
            stage_status(args, bridge_input_ready and viewer_outer_pose_ready and bridge_ready, stage_requested(args, "reports"), combined_viewer_html),
            combined_viewer_inputs,
            {"viewer_html": str(combined_viewer_html)},
            argv=combined_viewer_argv,
            notes=[] if viewer_outer_pose_ready else ["outer final pose YAML and outer prior / COLMAP images path are both missing"],
            group="reports",
        ),
    ]

    final_candidates = {
        "inner_refine_mode": inner_plan["mode"],
        "inner_prior_source": inner_prior_source,
        "input_contracts": {
            "small_marker_inner8": small_canonical,
            "large_marker_inner8": large_inner_canonical,
            "large_marker_bridge_all32": bridge_canonical,
        },
        "configured_inner_prior_state_dir": priors["inner_prior"],
        "effective_inner_prior_state_dir": str(effective_inner_prior),
        "inner_final_baseline_camera_tr_rig_yaml": str(final_inner_camera_tr_rig),
        "inner_final_baseline_state_dir": str(final_inner_state),
        "inner_final_baseline_dataset": str(final_inner_dataset),
        "inner_final_baseline_manifest": final_inner_manifest,
        "large_inner_init_camera_tr_rig_yaml": str(large_inner_init_state / "camera_tr_rig.yaml"),
        "large_inner_init_state_dir": str(large_inner_init_state),
        "large_inner_init_dataset": str(large_inner_stride_features),
        "small_fixed_rig_quality_camera_tr_rig_yaml": str(small_fixed_rig_quality / "camera_tr_rig.yaml"),
        "small_fixed_rig_quality_state_dir": str(small_fixed_rig_quality),
        "small_fixed_rig_quality_dataset": str(small_fixed_rig_quality_dataset),
        "small_fixed_rig_quality_pnp_views": str(small_fixed_rig_quality / "pnp_views.tsv"),
        "small_fixed_rig_quality_camera_pnp_summary": str(small_fixed_rig_quality / "camera_pnp_summary.tsv"),
        "inner_selected_camera_tr_rig_yaml": str(selected_inner_camera_tr_rig),
        "inner_selected_state_dir": str(selected_inner_state),
        "inner_selected_dataset": str(selected_inner_dataset),
        "inner_selected_manifest": selected_inner_manifest,
        "inner_fixed_camera_tr_rig_yaml": str(inner_fixed_refine / "camera_tr_rig.yaml"),
        "inner_fixed_state_dir": str(inner_fixed_refine),
        "inner_joint_camera_tr_rig_yaml": str(inner_joint_refine / "camera_tr_rig.yaml"),
        "inner_joint_state_dir": str(inner_joint_refine),
        "inner_reprojection_report": str(inner_reproj / "index.html"),
        "inner_interactive_viewer": str(inner_viewer / "index.html"),
        "bridge_pose_yaml": str(bridge_out / "camera_tr_inner_refined_plus_outer_topdown.yaml"),
        "bridge_summary_json": str(bridge_out / "bridge_summary.json"),
        "bridge_all32_fixed_intrinsics_dir": priors["bridge_intrinsics"],
        "outer_final_pose_yaml": outer_final_pose_yaml,
        "outer_final_pose_ready": outer_final_pose_ready,
        "combined_bridge_outer_pose_source": viewer_outer_pose_source,
        "combined_bridge_viewer": str(combined_viewer_html),
    }
    return stages, final_candidates


def execute_requested_stages(args, stages, output_root, repo_root):
    if args.dry_run:
        return stages

    logs_dir = Path(output_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    for stage in stages:
        if (
            stage["status"] == "reused_existing"
            and stage_requested(args, stage.get("group", ""))
            and not fingerprint_matches(output_root, stage)
        ):
            stage["status"] = "ready_to_run"
            stage["requested"] = True
            stage.setdefault("notes", []).append("existing output fingerprint missing or changed; recomputing requested stage")
        if stage["status"] != "ready_to_run":
            continue
        if getattr(args, "force", False):
            clear_stage_outputs(stage, output_root)
        command = stage.get("planned_command", "")
        if not command or command == "internal scan":
            stage["started_at"] = utc_now()
            started = time.time()
            stage["status"] = "complete"
            stage["duration"] = duration_s(time.time() - started)
            stage["finished_at"] = utc_now()
            write_stage_fingerprint(output_root, stage)
            continue
        log_path = logs_dir / f"{stage['name']}.log"
        stage["log_path"] = str(log_path)
        stage["started_at"] = utc_now()
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {command}\n\n")
            proc = subprocess.run(
                command,
                cwd=str(repo_root),
                shell=True,
                text=True,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        stage["duration"] = duration_s(time.time() - started)
        stage["finished_at"] = utc_now()
        stage["returncode"] = proc.returncode
        if proc.returncode == 0:
            stage["status"] = "complete"
            write_stage_fingerprint(output_root, stage)
        else:
            stage["status"] = "failed_allowed" if stage.get("allow_failure") else "failed"
            stage.setdefault("notes", []).append(f"command_failed_returncode_{proc.returncode}")
            if stage.get("allow_failure") and stage_outputs_exist(stage):
                stage.setdefault("notes", []).append(
                    "existing outputs were not fingerprinted because the current command failed"
                )
            if not stage.get("allow_failure"):
                break
    return stages


def stage_timing_entry(stage):
    return {
        "name": stage.get("name", ""),
        "group": stage.get("group", ""),
        "status": stage.get("status", ""),
        "requested": bool(stage.get("requested")),
        "started_at": stage.get("started_at", ""),
        "finished_at": stage.get("finished_at", ""),
        "duration_s": duration_s(stage.get("duration", 0.0)),
        "command": stage.get("planned_command", ""),
        "log_path": stage.get("log_path", ""),
        "returncode": stage.get("returncode"),
        "allow_failure": bool(stage.get("allow_failure")),
        "notes": stage.get("notes", []),
    }


def build_run_manifest(summary, run_started_at, run_finished_at, total_duration):
    args = summary["args"]
    priors = summary["priors"]
    final_candidates = summary["final_yaml_candidates"]
    outer_source_kind = final_candidates.get("combined_bridge_outer_pose_source", "")
    outer_source_path = (
        priors.get("outer_final_pose_yaml", "")
        if outer_source_kind == "outer_final_pose_yaml"
        else priors.get("outer_prior", "")
    )
    return {
        "created_at": utc_now(),
        "run_tag": summary["run_tag"],
        "mode": summary["mode"],
        "started_at": run_started_at,
        "finished_at": run_finished_at,
        "total_duration_s": duration_s(total_duration),
        "summary_json": str(Path(summary["output_root"]) / "summary.json"),
        "inputs": {
            "data_root": summary["data_root"],
            "output_root": summary["output_root"],
            "small_marker_sequence": args["small_marker_sequence"],
            "large_inner_marker_sequence": args["large_inner_marker_sequence"],
            "large_marker_sequence": args["large_marker_sequence"],
            "small_marker": args["small_marker"],
            "large_inner_marker": args["large_inner_marker"],
            "large_marker": args["large_marker"],
            "outer_source": priors.get("outer_final_pose_yaml", ""),
            "outer_source_kind": outer_source_kind,
            "outer_source_path": outer_source_path,
            "outer_prior": priors.get("outer_prior", ""),
            "outer_final_pose_yaml": priors.get("outer_final_pose_yaml", ""),
            "inner_prior": priors.get("inner_prior", ""),
            "inner_intrinsics": priors.get("inner_intrinsics", ""),
            "bridge_intrinsics": priors.get("bridge_intrinsics", ""),
        },
        "stages": [stage_timing_entry(stage) for stage in summary["stages"]],
    }


def write_run_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def html_table(rows, columns):
    header = "".join(f"<th>{html.escape(label)}</th>" for _key, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            if value is None:
                value = ""
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def link_html(item):
    label = html.escape(item["label"])
    path = html.escape(item["path"])
    badges = "exists" if item["exists"] else "missing"
    links = []
    if item.get("file_url"):
        links.append(f'<a href="{html.escape(item["file_url"])}">file</a>')
    if item.get("http_url"):
        links.append(f'<a href="{html.escape(item["http_url"])}">http</a>')
    link_text = " / ".join(links) if links else "no link"
    return f"<li><strong>{label}</strong> <span class=\"badge\">{badges}</span><br><code>{path}</code><br>{link_text}</li>"


def write_index_html(path, summary):
    small = summary["data_quality"]["small_marker"]
    large_inner = summary["data_quality"]["large_inner_marker"]
    large = summary["data_quality"]["large_marker"]
    small_quality = summary.get("small_fixed_rig_quality_probe", {})
    joint_quality = summary.get("inner_joint_intrinsics_quality", {})
    bridge_quality = summary.get("bridge_quality", {})
    provenance = summary.get("provenance", {})
    git_info = provenance.get("git", {})
    git_dirty = "dirty" if git_info.get("dirty") else "clean"
    stages = summary["stages"]
    candidates = summary["bridge_candidates"]
    inner_links = summary["existing_inner_reports"]
    bridge_links = summary["existing_bridge_reports"]

    camera_columns = [
        ("index", "idx"),
        ("kind", "kind"),
        ("camera_id", "camera"),
        ("machine", "machine"),
        ("frame_count", "frames"),
        ("tail_short", "tail"),
        ("status", "status"),
        ("reason", "reason"),
    ]
    stage_columns = [
        ("name", "stage"),
        ("status", "status"),
        ("duration", "duration_s"),
        ("notes", "notes"),
        ("planned_command", "planned command"),
    ]
    candidate_columns = [
        ("name", "candidate"),
        ("status", "status"),
        ("reason", "reason"),
        ("usable_inner_count", "inner"),
        ("usable_outer_count", "outer"),
        ("common_frame_count", "common_frames"),
    ]

    path = Path(path)
    path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fast Inner/Bridge Recalibration Pipeline</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2328; background: #f6f8fa; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; }}
    p {{ color: #57606a; line-height: 1.45; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; }}
    .card strong {{ display: block; font-size: 24px; }}
    .badge {{ display: inline-block; border: 1px solid #d0d7de; border-radius: 999px; padding: 1px 8px; font-size: 12px; background: #f6f8fa; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d0d7de; margin: 10px 0 18px; }}
    th, td {{ border-bottom: 1px solid #d0d7de; padding: 7px 8px; text-align: left; font-size: 12px; vertical-align: top; }}
    th {{ background: #eef1f5; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; word-break: break-all; }}
    li {{ margin: 0 0 12px; }}
  </style>
</head>
<body>
<main>
  <h1>Fast Inner/Bridge Recalibration Pipeline</h1>
  <p>Generated at {html.escape(summary["created_at"])}. Mode: <strong>{html.escape(summary["mode"])}</strong>.</p>
  <p>Final report URL: <a href="{html.escape(summary["report_urls"]["final_report"])}">{html.escape(summary["report_urls"]["final_report"])}</a></p>
  <p>Provenance: <code>{html.escape(str(git_info.get("branch", "")))}</code> / <code>{html.escape(str(git_info.get("commit", ""))[:12])}</code> ({html.escape(git_dirty)}); command <code>{html.escape(str(provenance.get("argv", "")))}</code>.</p>
  <div class="grid">
    <div class="card"><span>small marker</span><strong>{html.escape(small["status"])}</strong><p>{small["usable_camera_count"]} usable / {small["camera_count"]} scanned, common frames {small["common_frame_count"]}</p></div>
    <div class="card"><span>large inner marker</span><strong>{html.escape(large_inner["status"])}</strong><p>{large_inner["usable_camera_count"]} usable / {large_inner["camera_count"]} scanned, common frames {large_inner["common_frame_count"]}</p></div>
    <div class="card"><span>small fixed-rig quality</span><strong>{html.escape(str(small_quality.get("status", "missing")))}</strong><p>{html.escape(str(small_quality.get("connected_count", 0)))} connected / {html.escape(str(small_quality.get("camera_count", 0)))} summarized; disconnected {html.escape(",".join(small_quality.get("disconnected_cameras", [])))}</p></div>
    <div class="card"><span>large bridge marker</span><strong>{html.escape(large["status"])}</strong><p>{large["usable_camera_count"]} usable / {large["camera_count"]} scanned, common frames {large["common_frame_count"]}</p></div>
    <div class="card"><span>bridge metric gate</span><strong>{html.escape(str(bridge_quality.get("metric_bridge_gate", "missing")))}</strong><p>{html.escape(str(bridge_quality.get("outer_vote_count_min")))} min votes; p90 center {html.escape(str(bridge_quality.get("max_outer_center_residual_p90_m")))} m; p90 rot {html.escape(str(bridge_quality.get("max_outer_rotation_residual_p90_deg")))} deg</p></div>
    <div class="card"><span>output root</span><code>{html.escape(summary["output_root"])}</code></div>
  </div>

  <h2>Data Quality Rules</h2>
  <p>Manifest presence, camera count, and frame-count spread are checked before feature extraction. A machine/group stopping up to {html.escape(str(small["max_tail_trim"]))} frames early at the sequence tail is treated as a common-prefix trim case. Interior frame gaps or a spread larger than the tail trim budget indicate a likely per-camera drop; that camera/sequence should be excluded rather than partially retained.</p>
  <table>
    <tr><th>Sequence</th><th>Manifest</th><th>Manifest cameras</th><th>Scanned cameras</th><th>Frame min/max</th><th>Spread</th><th>Warning</th></tr>
    <tr><td>small</td><td>{html.escape(str(small["manifest_exists"]))}</td><td>{html.escape(str(small["manifest_summary"]["camera_count"]))}</td><td>{html.escape(str(small["camera_count"]))}</td><td>{html.escape(str(small["frame_count_min"]))} / {html.escape(str(small["frame_count_max"]))}</td><td>{html.escape(str(small["frame_count_spread"]))}</td><td>{html.escape(small.get("drop_frame_warning", ""))}</td></tr>
    <tr><td>large-inner</td><td>{html.escape(str(large_inner["manifest_exists"]))}</td><td>{html.escape(str(large_inner["manifest_summary"]["camera_count"]))}</td><td>{html.escape(str(large_inner["camera_count"]))}</td><td>{html.escape(str(large_inner["frame_count_min"]))} / {html.escape(str(large_inner["frame_count_max"]))}</td><td>{html.escape(str(large_inner["frame_count_spread"]))}</td><td>{html.escape(large_inner.get("drop_frame_warning", ""))}</td></tr>
    <tr><td>large-bridge</td><td>{html.escape(str(large["manifest_exists"]))}</td><td>{html.escape(str(large["manifest_summary"]["camera_count"]))}</td><td>{html.escape(str(large["camera_count"]))}</td><td>{html.escape(str(large["frame_count_min"]))} / {html.escape(str(large["frame_count_max"]))}</td><td>{html.escape(str(large["frame_count_spread"]))}</td><td>{html.escape(large.get("drop_frame_warning", ""))}</td></tr>
  </table>

  <h2>Pipeline Design</h2>
  <ol>
    <li>Use the large-marker fixed-intrinsic inner initializer as the production inner extrinsic baseline when <code>--run-stage large-inner-init</code>, <code>--run-large-inner-init</code>, or <code>--run-all</code> is explicit or when its output already exists. Its default frame stride is <code>1</code>.</li>
    <li>Run the small-marker fixed-intrinsic rig estimate as a quality probe when <code>--run-stage small-fixed-rig-quality</code>, <code>--run-small-fixed-rig-quality</code>, or <code>--run-all</code> is explicit. This probe writes <code>camera_pnp_summary.tsv</code> and never replaces the final large-inner baseline.</li>
    <li>Keep <code>fixed</code>, <code>joint</code>, and <code>fixed_then_joint</code> small-marker refinement as explicit diagnostic modes. <code>joint</code> first builds a fixed-localize warm start on the current small-marker dataset; direct joint BA from a stale state is avoided. Joint output must pass the intrinsics sanity gate below before it is considered trustworthy.</li>
    <li>Optionally run all32 large-marker bridge fixed-intrinsic PnP/evaluation when <code>--run-stage large-bridge</code> or <code>--run-all</code> is explicit. The bridge index convention is outer <code>0..23</code>, inner <code>24..31</code>; top-down anchors <code>4-1/4-2/4-3</code> are indices <code>9/10/11</code>.</li>
    <li>Generate report/viewer links from existing or newly produced outputs.</li>
  </ol>

  <h2>Bridge all32 Convention</h2>
  <p>{html.escape(summary["bridge_layout"]["index_convention"])} The wrapper prepares <code>{html.escape(summary["priors"]["bridge_intrinsics"])}</code> by copying outer intrinsics <code>0..23</code> and remapping compact inner intrinsics <code>0..7</code> to <code>24..31</code>. This stage is fixed-intrinsic PnP plus top-down anchor evaluation; full combined BA/refinement is a later pipeline stage.</p>

  <h2>Bridge Metric Quality</h2>
  <p>Status: <strong>{html.escape(str(bridge_quality.get("metric_bridge_gate", "missing")))}</strong>. COLMAP prior diagnostic: <strong>{html.escape(str(bridge_quality.get("colmap_prior_diagnostic", "missing")))}</strong>. Bridge pose ready: <strong>{html.escape(str(bridge_quality.get("prior_output_ready", False)))}</strong>. Summary JSON: <code>{html.escape(str(bridge_quality.get("path", "")))}</code>.</p>
  <p>Inner board frames: <code>{html.escape(str(bridge_quality.get("inner_board_frame_count")))}</code>; median inner support: <code>{html.escape(str(bridge_quality.get("inner_support_median")))}</code>; min outer votes: <code>{html.escape(str(bridge_quality.get("outer_vote_count_min")))}</code>; max center p90: <code>{html.escape(str(bridge_quality.get("max_outer_center_residual_p90_m")))}</code> m; max rotation p90: <code>{html.escape(str(bridge_quality.get("max_outer_rotation_residual_p90_deg")))}</code> deg.</p>
  <p>Notes: <code>{html.escape(json.dumps(bridge_quality.get("notes", []), ensure_ascii=False))}</code></p>

  <h2>Small Fixed-Rig Quality Probe</h2>
  <p>Status: <strong>{html.escape(str(small_quality.get("status", "missing")))}</strong>. Summary TSV: <code>{html.escape(str(small_quality.get("path", "")))}</code>. Disconnected cameras are flags for quality comparison only; the final inner baseline remains <code>{html.escape(summary["final_yaml_candidates"]["inner_final_baseline_camera_tr_rig_yaml"])}</code>.</p>

    <h2>Small Joint Intrinsics Sanity</h2>
  <p>Status: <strong>{html.escape(str(joint_quality.get("status", "missing_joint_output")))}</strong>. Accepted by sanity gate: <strong>{html.escape(str(joint_quality.get("accepted_by_sanity_gate", False)))}</strong>. Final selection: <strong>{html.escape(str(joint_quality.get("final_selection", "large_inner_fixed_baseline")))}</strong>. Max focal delta: <code>{html.escape(str(joint_quality.get("max_abs_focal_delta_frac")))}</code>; max principal delta: <code>{html.escape(str(joint_quality.get("max_principal_delta_px")))}</code> px; max distortion abs: <code>{html.escape(str(joint_quality.get("max_abs_distortion")))}</code>; max distortion delta: <code>{html.escape(str(joint_quality.get("max_abs_distortion_delta")))}</code>.</p>

  <h2>Existing Inner Reports</h2>
  <ul>{''.join(link_html(item) for item in inner_links)}</ul>

  <h2>Existing Large-Marker Bridge Reports</h2>
  <ul>{''.join(link_html(item) for item in bridge_links)}</ul>

  <h2>Small Marker Cameras</h2>
  {html_table(small["cameras"], camera_columns)}

  <h2>Large Inner Marker Cameras</h2>
  {html_table(large_inner["cameras"], camera_columns)}

  <h2>Large Bridge Marker Cameras</h2>
  {html_table(large["cameras"], camera_columns)}

  <h2>Bridge Candidates</h2>
  {html_table(candidates, candidate_columns)}

  <h2>Pipeline Stages</h2>
  {html_table(stages, stage_columns)}

  <h2>Final YAML Candidates</h2>
  <ul>{''.join(f'<li><code>{html.escape(k)}</code>: <code>{html.escape(str(v))}</code></li>' for k, v in summary["final_yaml_candidates"].items())}</ul>
</main>
</body>
</html>
""", encoding="utf-8")


def write_simple_report(path, title, summary, body_html):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = summary.get("provenance", {})
    git_info = provenance.get("git", {})
    git_dirty = "dirty" if git_info.get("dirty") else "clean"
    path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 28px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2328; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    a {{ color: #0969da; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 18px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; font-size: 12px; }}
    th {{ background: #f6f8fa; }}
    li {{ margin: 0 0 10px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Run tag: <code>{html.escape(summary["run_tag"])}</code>. Generated at {html.escape(summary["created_at"])}.</p>
  <p>Provenance: <code>{html.escape(str(git_info.get("branch", "")))}</code> / <code>{html.escape(str(git_info.get("commit", ""))[:12])}</code> ({html.escape(git_dirty)}); command <code>{html.escape(str(provenance.get("argv", "")))}</code>.</p>
  <p><a href="../index.html">Pipeline index</a> · <a href="../summary.json">summary.json</a></p>
  {body_html}
</body>
</html>
""", encoding="utf-8")


def write_report_entrypoints(output_root, summary):
    output_root = Path(output_root)
    small = summary["data_quality"]["small_marker"]
    large_inner = summary["data_quality"]["large_inner_marker"]
    large = summary["data_quality"]["large_marker"]
    small_quality = summary.get("small_fixed_rig_quality_probe", {})
    joint_quality = summary.get("inner_joint_intrinsics_quality", {})
    bridge_quality = summary.get("bridge_quality", {})
    outer_pose_source = summary["final_yaml_candidates"].get(
        "combined_bridge_outer_pose_source",
        "unknown",
    )
    outer_final_pose = summary["final_yaml_candidates"].get("outer_final_pose_yaml", "")
    final_items = "".join(
        f"<li><code>{html.escape(key)}</code>: <code>{html.escape(str(value))}</code></li>"
        for key, value in summary["final_yaml_candidates"].items()
    )
    bridge_links = "".join(link_html(item) for item in summary["existing_bridge_reports"])
    inner_links = "".join(link_html(item) for item in summary["existing_inner_reports"])
    run_timing = summary.get("run_timing", {})
    stage_timing_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(stage.get('name', ''))}</code></td>"
        f"<td>{html.escape(stage.get('status', ''))}</td>"
        f"<td>{html.escape(str(duration_s(stage.get('duration', 0.0))))}</td>"
        f"<td>{html.escape(stage.get('started_at', ''))}</td>"
        f"<td>{html.escape(stage.get('finished_at', ''))}</td>"
        "</tr>"
        for stage in summary["stages"]
    )
    input_rows = [
        ("data root", summary["data_root"]),
        ("small marker sequence", summary["args"]["small_marker_sequence"]),
        ("large inner marker sequence", summary["args"]["large_inner_marker_sequence"]),
        ("bridge sequence", summary["args"]["large_marker_sequence"]),
        ("small marker path", summary["args"]["small_marker"]),
        ("large inner marker path", summary["args"]["large_inner_marker"]),
        ("bridge marker path", summary["args"]["large_marker"]),
        ("outer source kind", outer_pose_source),
        (
            "outer source path",
            outer_final_pose if outer_pose_source == "outer_final_pose_yaml" else summary["priors"].get("outer_prior", ""),
        ),
        ("outer final pose yaml", outer_final_pose),
        ("outer prior", summary["priors"].get("outer_prior", "")),
    ]
    input_table_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for label, value in input_rows
    )

    write_simple_report(
        output_root / "quality_report/index.html",
        "Fast Inner/Bridge Quality Report",
        summary,
        f"""
  <p>Small marker: {html.escape(small["status"])}; {small["usable_camera_count"]}/{small["camera_count"]} usable cameras; frame spread {html.escape(str(small["frame_count_spread"]))}.</p>
  <p>Small fixed-rig quality probe: {html.escape(str(small_quality.get("status", "missing")))}; {html.escape(str(small_quality.get("connected_count", 0)))} connected / {html.escape(str(small_quality.get("camera_count", 0)))} summarized; disconnected cameras: {html.escape(",".join(small_quality.get("disconnected_cameras", [])))}.</p>
  <p>Large inner marker: {html.escape(large_inner["status"])}; {large_inner["usable_camera_count"]}/{large_inner["camera_count"]} usable cameras; frame spread {html.escape(str(large_inner["frame_count_spread"]))}.</p>
  <p>Large bridge marker: {html.escape(large["status"])}; {large["usable_camera_count"]}/{large["camera_count"]} usable cameras; frame spread {html.escape(str(large["frame_count_spread"]))}.</p>
  <p>Bridge metric gate: {html.escape(str(bridge_quality.get("metric_bridge_gate", "missing")))}; COLMAP prior diagnostic {html.escape(str(bridge_quality.get("colmap_prior_diagnostic", "missing")))}; min votes {html.escape(str(bridge_quality.get("outer_vote_count_min")))}; max center p90 {html.escape(str(bridge_quality.get("max_outer_center_residual_p90_m")))} m; max rotation p90 {html.escape(str(bridge_quality.get("max_outer_rotation_residual_p90_deg")))} deg.</p>
  <p>Tail-only offsets up to {small["max_tail_trim"]} frames are acceptable. Interior gaps or larger spreads should be treated as camera-drop evidence.</p>
""",
    )
    write_simple_report(
        output_root / "final_report/index.html",
        "Fast Inner/Bridge Final Report",
        summary,
        f"""
  <p>This page is a stable entrypoint for the panel. It links existing reused products and planned final outputs.</p>
  <h2>Run Timing / Recalib Inputs</h2>
  <p>Total runtime: <strong>{html.escape(str(run_timing.get("total_duration_s", 0.0)))}</strong> s. Started: <code>{html.escape(str(run_timing.get("started_at", "")))}</code>. Finished: <code>{html.escape(str(run_timing.get("finished_at", "")))}</code>. Manifest: <a href="../run_manifest.json">run_manifest.json</a>.</p>
  <table>{input_table_rows}</table>
  <table>
    <tr><th>stage</th><th>status</th><th>duration_s</th><th>started</th><th>finished</th></tr>
    {stage_timing_rows}
  </table>
  <p>The final inner extrinsic baseline is <code>{html.escape(summary["final_yaml_candidates"]["inner_final_baseline_camera_tr_rig_yaml"])}</code>. Small-marker fixed-rig output is reported only as a quality probe and does not replace this baseline.</p>
  <p>Small-marker joint intrinsic/extrinsic output is diagnostic unless its intrinsics sanity gate passes. Current joint sanity status: <strong>{html.escape(str(joint_quality.get("status", "missing_joint_output")))}</strong>; accepted: <strong>{html.escape(str(joint_quality.get("accepted_by_sanity_gate", False)))}</strong>.</p>
  <p>Bridge all32 uses outer indices <code>0..23</code>, inner indices <code>24..31</code>, and top-down anchors <code>4-1/4-2/4-3</code> at indices <code>9/10/11</code>. The current bridge product is fixed-intrinsic PnP plus top-down evaluation; combined BA/refine remains a follow-up.</p>
  <p>Combined 24+8 viewer outer pose source: <strong>{html.escape(str(outer_pose_source))}</strong>. Outer final pose YAML: <code>{html.escape(str(outer_final_pose))}</code>. If this source is <code>outer_final_pose_yaml</code>, the 24 outer camera frustums come from the latest outer tower accepted rig instead of the old first-frame COLMAP Sim(3) diagnostic.</p>
  <p>Bridge metric gate: <strong>{html.escape(str(bridge_quality.get("metric_bridge_gate", "missing")))}</strong>. Bridge pose ready: <strong>{html.escape(str(bridge_quality.get("prior_output_ready", False)))}</strong>. COLMAP prior diagnostic: <strong>{html.escape(str(bridge_quality.get("colmap_prior_diagnostic", "missing")))}</strong>.</p>
  <h2>Existing inner products</h2>
  <ul>{inner_links}</ul>
  <h2>Existing bridge products</h2>
  <ul>{bridge_links}</ul>
  <h2>Planned final candidates</h2>
  <ul>{final_items}</ul>
""",
    )
    write_simple_report(
        output_root / "viewer/index.html",
        "Fast Inner/Bridge Viewer Links",
        summary,
        f"""
  <p>This is a stable link page for viewer artifacts. The actual generated combined bridge viewer is expected at:</p>
  <p><code>{html.escape(summary["final_yaml_candidates"]["combined_bridge_viewer"])}</code></p>
  <p>The actual generated inner-only viewer is expected at:</p>
  <p><code>{html.escape(summary["final_yaml_candidates"]["inner_interactive_viewer"])}</code></p>
  <h2>Existing bridge/viewer links</h2>
  <ul>{bridge_links}</ul>
""",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan/report the fast inner/bridge recalibration pipeline for t0-style camera calibration data.",
    )
    parser.add_argument(
        "--data-root",
        "--stage-root",
        dest="data_root",
        type=Path,
        default=DEFAULT_T0_DATA_ROOT,
        help="t0 staging root, usually /home/ubuntu/calib_data/calib_2026_05_26_jpg_v3.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("studio/exp/inner_bridge_recalib_pipeline_dry_run"),
        help="Directory where summary.json, index.html, and planned input lists are written.",
    )
    parser.add_argument(
        "--small-marker",
        type=Path,
        default=None,
        help="Override small_marker staged/session path. Relative paths are resolved under --data-root.",
    )
    parser.add_argument(
        "--small-marker-sequence",
        default="small_marker_inner8",
        help="Small-marker sequence directory name under --stage-root.",
    )
    parser.add_argument(
        "--large-marker",
        type=Path,
        default=None,
        help="Override large_marker staged/session path. Relative paths are resolved under --data-root.",
    )
    parser.add_argument(
        "--large-marker-sequence",
        default="large_marker_bridge_all32",
        help="Large-marker bridge sequence directory name under --stage-root.",
    )
    parser.add_argument(
        "--large-inner-marker",
        type=Path,
        default=None,
        help="Override large_marker inner initializer staged/session path. Relative paths are resolved under --data-root.",
    )
    parser.add_argument(
        "--large-inner-marker-sequence",
        default="large_marker_inner8",
        help="Large-marker inner initializer sequence directory name under --stage-root.",
    )
    parser.add_argument(
        "--inner-prior",
        type=Path,
        default=None,
        help="Warm-start inner state directory. Defaults to final_inner8_calibration_v1 state under --data-root.",
    )
    parser.add_argument(
        "--outer-prior",
        type=Path,
        default=None,
        help="Outer prior path for bridge evaluation, currently expected to be COLMAP images.txt.",
    )
    parser.add_argument(
        "--outer-final-pose-yaml",
        type=Path,
        default=DEFAULT_OUTER_FINAL_POSE_YAML,
        help=(
            "Latest outer tower final camera_tr_rig YAML for the combined bridge viewer. "
            "If it does not exist, the viewer falls back to the legacy COLMAP Sim3 approximate outer poses."
        ),
    )
    parser.add_argument(
        "--outer-intrinsics",
        type=Path,
        default=None,
        help="Outer fixed intrinsics directory for all32 bridge PnP. Defaults to whole_outer_tower/fixed_intrinsic_pnp_colmap_fallback_v1 under --data-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan data and write the plan/report; do not run recalibration commands.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not reuse existing stage outputs when an explicit run stage is requested.",
    )
    parser.add_argument(
        "--run-tag",
        default="latest",
        help="Human run tag recorded in summary/report files.",
    )
    parser.add_argument(
        "--run-stage",
        action="append",
        choices=["large-inner-init", "small-fixed-rig-quality", "small-refine", "large-bridge", "reports"],
        default=[],
        help="Opt-in execution group for non-dry-run mode. Repeat for multiple groups.",
    )
    parser.add_argument(
        "--run-large-inner-init",
        action="store_true",
        help="Opt in to large-marker inner fixed-intrinsic initializer before small-marker refinement.",
    )
    parser.add_argument(
        "--run-small-fixed-rig-quality",
        action="store_true",
        help="Opt in to the small-marker fixed-intrinsic rig quality probe in non-dry-run mode.",
    )
    parser.add_argument(
        "--run-small-refine",
        action="store_true",
        help=(
            "Opt in to legacy/diagnostic small-marker localize-only or joint refinement. "
            "With the default fixed_rig mode, this is treated as the quality probe for compatibility."
        ),
    )
    parser.add_argument(
        "--inner-refine-mode",
        choices=["fixed_rig", "fixed", "joint", "fixed_then_joint"],
        default="fixed_rig",
        help=(
            "Small-marker mode. fixed_rig runs a fixed-intrinsic PnP rig quality probe only; "
            "fixed keeps intrinsics fixed with --localize_only and is a diagnostic path that can "
            "stall on LM bad-cost; joint first builds a fixed-localize warm-start on the current "
            "small-marker dataset and then optimizes intrinsics/extrinsics; fixed_then_joint is "
            "kept as an explicit alias for that two-stage path."
        ),
    )
    parser.add_argument(
        "--inner-model",
        default="central_opencv",
        help="Camera model passed to joint inner calibration when --inner-refine-mode includes joint.",
    )
    parser.add_argument(
        "--inner-fixed-max-ba-iterations",
        type=int,
        default=3,
        help="BA iterations for the fixed-intrinsic localize-only inner warm start. 0 runs localization/init only.",
    )
    parser.add_argument(
        "--inner-joint-max-ba-iterations",
        type=int,
        default=3,
        help="BA iterations for joint inner intrinsics/extrinsics refinement.",
    )
    parser.add_argument(
        "--inner-joint-outlier-removal-factor",
        type=float,
        default=0.0,
        help="Outlier removal factor for joint inner refinement. 0 disables deletion for fast recalib.",
    )
    parser.add_argument(
        "--inner-schur-mode",
        choices=["dense", "dense_cuda", "dense_onthefly", "sparse", "sparse_onthefly"],
        default="sparse_onthefly",
        help="Schur complement mode for inner BA. sparse_onthefly avoids the dense default crash seen in warm-start joint BA.",
    )
    parser.add_argument(
        "--small-frame-stride",
        type=int,
        default=4,
        help="Frame stride for small-marker fast recalib after feature extraction. 4 uses about 80 frames from a 320-frame capture.",
    )
    parser.add_argument(
        "--large-frame-stride",
        type=int,
        default=1,
        help="Frame stride for large-marker bridge PnP after feature extraction. Use 1 by default because all32 bridge connectivity is sensitive to camera0 support.",
    )
    parser.add_argument(
        "--large-inner-frame-stride",
        type=int,
        default=1,
        help="Frame stride for large-marker inner initializer after feature extraction.",
    )
    parser.add_argument(
        "--run-large-bridge",
        action="store_true",
        help="Opt in to large-marker bridge feature extraction/PnP/evaluation in non-dry-run mode.",
    )
    parser.add_argument(
        "--run-reports",
        action="store_true",
        help="Opt in to report/viewer regeneration in non-dry-run mode.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Opt in to all execution groups in non-dry-run mode.",
    )
    parser.add_argument(
        "--report-url-base",
        default=DEFAULT_REPORT_HTTP_ROOT,
        help="HTTP base used for report URLs when outputs are under --http-root.",
    )
    parser.add_argument(
        "--http-root",
        type=Path,
        default=CALIB_DATA_ROOT,
        help="Filesystem root served by --report-url-base.",
    )
    args = parser.parse_args()
    if args.inner_fixed_max_ba_iterations < 0:
        parser.error("--inner-fixed-max-ba-iterations must be non-negative")
    if args.inner_joint_max_ba_iterations < 0:
        parser.error("--inner-joint-max-ba-iterations must be non-negative")
    if args.inner_joint_outlier_removal_factor < 0:
        parser.error("--inner-joint-outlier-removal-factor must be non-negative")
    if args.small_frame_stride < 1:
        parser.error("--small-frame-stride must be >= 1")
    if args.large_frame_stride < 1:
        parser.error("--large-frame-stride must be >= 1")
    if args.large_inner_frame_stride < 1:
        parser.error("--large-inner-frame-stride must be >= 1")
    return args


def main():
    args = parse_args()
    run_started_at = utc_now()
    run_started_perf = time.time()
    repo_root = script_repo_root()
    data_root = resolve_user_path(args.data_root, Path.cwd())
    output_root = resolve_user_path(args.output_root, Path.cwd())

    if output_root.exists() and not output_root.is_dir():
        raise SystemExit(f"--output-root exists but is not a directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    http_root = resolve_user_path(args.http_root, Path.cwd())
    small_selector = args.small_marker if args.small_marker else args.small_marker_sequence
    large_inner_selector = args.large_inner_marker if args.large_inner_marker else args.large_inner_marker_sequence
    large_selector = args.large_marker if args.large_marker else args.large_marker_sequence
    small_path = resolve_marker_path(data_root, small_selector, args.small_marker_sequence, "small_marker")
    large_inner_path = resolve_marker_path(data_root, large_inner_selector, args.large_inner_marker_sequence, "large_marker")
    large_path = resolve_marker_path(data_root, large_selector, args.large_marker_sequence, "large_marker")

    scan_started_at = utc_now()
    scan_started_perf = time.time()
    small_scan = scan_marker_session("small_marker", small_path, len(INNER_CAMERAS), 2)
    large_inner_scan = scan_marker_session("large_inner_marker", large_inner_path, len(INNER_CAMERAS), 2)
    large_scan = scan_marker_session("large_marker", large_path, len(INNER_CAMERAS) + len(OUTER_CAMERAS), 2)
    scan_duration = time.time() - scan_started_perf
    scan_finished_at = utc_now()

    planned_inputs = {
        "small_marker": write_planned_inputs(output_root, small_scan),
        "large_inner_marker": write_planned_inputs(output_root, large_inner_scan),
        "large_marker": write_planned_inputs(output_root, large_scan),
    }

    inner_prior = resolve_user_path(
        args.inner_prior,
        data_root,
    ) if args.inner_prior else (data_root / "final_inner8_calibration_v1/states/final_small_marker_grid4_refine_v1")
    outer_prior = resolve_user_path(
        args.outer_prior,
        data_root,
    ) if args.outer_prior else (
        data_root / "colmap_outer24_firstframe_colmap404_v3/fixed_intrinsics/sparse_txt_final24_fixedK_ba/images.txt"
    )
    outer_final_pose_yaml = resolve_user_path(args.outer_final_pose_yaml, data_root) if args.outer_final_pose_yaml else None
    inner_intrinsics = infer_inner_intrinsics(data_root, inner_prior)
    outer_intrinsics = resolve_user_path(
        args.outer_intrinsics,
        data_root,
    ) if args.outer_intrinsics else infer_outer_intrinsics(data_root)
    bridge_layout = bridge_all32_layout(large_scan)
    bridge_intrinsics_started_at = utc_now()
    bridge_intrinsics_started_perf = time.time()
    bridge_intrinsics = prepare_bridge_intrinsics(
        output_root,
        outer_intrinsics,
        inner_intrinsics,
        bridge_layout,
    )
    bridge_intrinsics_duration = time.time() - bridge_intrinsics_started_perf
    bridge_intrinsics_finished_at = utc_now()
    priors = {
        "inner_prior": str(inner_prior.resolve(strict=False)),
        "inner_intrinsics": str(inner_intrinsics),
        "outer_intrinsics": str(outer_intrinsics),
        "bridge_intrinsics": bridge_intrinsics["output_dir"],
        "outer_prior": str(outer_prior.resolve(strict=False)),
        "outer_final_pose_yaml": str(outer_final_pose_yaml.resolve(strict=False)) if outer_final_pose_yaml else "",
    }

    stages, final_candidates = build_pipeline_stages(
        args,
        repo_root,
        data_root,
        output_root,
        small_scan,
        large_inner_scan,
        large_scan,
        planned_inputs,
        priors,
        bridge_layout,
        bridge_intrinsics,
    )
    set_stage_timing(stages, "data_quality_scan", scan_started_at, scan_finished_at, scan_duration)
    set_stage_timing(
        stages,
        "prepare_bridge_all32_fixed_intrinsics",
        bridge_intrinsics_started_at,
        bridge_intrinsics_finished_at,
        bridge_intrinsics_duration,
    )
    stages = execute_requested_stages(args, stages, output_root, repo_root)
    small_fixed_rig_quality = summarize_fixed_rig_quality(
        final_candidates["small_fixed_rig_quality_camera_pnp_summary"],
        len(INNER_CAMERAS),
        small_scan,
    )
    inner_joint_intrinsics_quality = summarize_inner_joint_intrinsics(
        final_candidates["effective_inner_prior_state_dir"],
        final_candidates["inner_joint_state_dir"],
    )
    bridge_quality = summarize_bridge_quality(
        final_candidates["bridge_summary_json"],
    )
    annotate_small_quality_stage(stages, small_fixed_rig_quality)

    index_path = output_root / "index.html"
    quality_path = output_root / "quality_report/index.html"
    final_path = output_root / "final_report/index.html"
    viewer_path = output_root / "viewer/index.html"
    summary_path = output_root / "summary.json"
    manifest_path = output_root / "run_manifest.json"

    report_urls = {
        "index": public_report_url(index_path, http_root, args.report_url_base),
        "summary": public_report_url(summary_path, http_root, args.report_url_base),
        "run_manifest": public_report_url(manifest_path, http_root, args.report_url_base),
        "quality_report": public_report_url(quality_path, http_root, args.report_url_base),
        "final_report": public_report_url(final_path, http_root, args.report_url_base),
        "viewer": public_report_url(viewer_path, http_root, args.report_url_base),
    }

    run_finished_at = utc_now()
    total_duration = time.time() - run_started_perf
    stage_durations = {
        stage.get("name", ""): duration_s(stage.get("duration", 0.0))
        for stage in stages
    }

    summary = {
        "created_at": utc_now(),
        "mode": run_mode(args),
        "run_tag": args.run_tag,
        "provenance": pipeline_provenance(repo_root),
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "report_url": report_urls["index"],
        "final_report_url": report_urls["final_report"],
        "quality_report_url": report_urls["quality_report"],
        "viewer_url": report_urls["viewer"],
        "run_manifest": str(manifest_path),
        "run_manifest_url": report_urls["run_manifest"],
        "run_timing": {
            "started_at": run_started_at,
            "finished_at": run_finished_at,
            "total_duration_s": duration_s(total_duration),
            "stage_count": len(stages),
            "stage_durations_s": stage_durations,
        },
        "args": {
            "data_root": str(data_root),
            "stage_root": str(data_root),
            "output_root": str(output_root),
            "small_marker_sequence": args.small_marker_sequence,
            "large_inner_marker_sequence": args.large_inner_marker_sequence,
            "large_marker_sequence": args.large_marker_sequence,
            "small_marker": str(small_path),
            "large_inner_marker": str(large_inner_path),
            "large_marker": str(large_path),
            "inner_prior": priors["inner_prior"],
            "outer_prior": priors["outer_prior"],
            "outer_intrinsics": priors["outer_intrinsics"],
            "bridge_intrinsics": priors["bridge_intrinsics"],
            "inner_refine_mode": args.inner_refine_mode,
            "inner_model": args.inner_model,
            "inner_fixed_max_ba_iterations": args.inner_fixed_max_ba_iterations,
            "inner_joint_max_ba_iterations": args.inner_joint_max_ba_iterations,
            "inner_joint_outlier_removal_factor": args.inner_joint_outlier_removal_factor,
            "inner_schur_mode": args.inner_schur_mode,
            "small_frame_stride": args.small_frame_stride,
            "large_inner_frame_stride": args.large_inner_frame_stride,
            "large_frame_stride": args.large_frame_stride,
            "dry_run": bool(args.dry_run),
            "force": bool(args.force),
            "run_stage": args.run_stage,
            "run_large_inner_init": bool(args.run_large_inner_init),
            "run_small_fixed_rig_quality": bool(args.run_small_fixed_rig_quality),
            "run_small_refine": bool(args.run_small_refine),
            "run_large_bridge": bool(args.run_large_bridge),
            "run_reports": bool(args.run_reports),
            "run_all": bool(args.run_all),
            "run_tag": args.run_tag,
            "report_url_base": args.report_url_base,
            "http_root": str(http_root),
        },
        "report_urls": report_urls,
        "data_quality": {
            "small_marker": small_scan,
            "large_inner_marker": large_inner_scan,
            "large_marker": large_scan,
        },
        "planned_inputs": planned_inputs,
        "small_fixed_rig_quality_probe": small_fixed_rig_quality,
        "inner_joint_intrinsics_quality": inner_joint_intrinsics_quality,
        "bridge_quality": bridge_quality,
        "existing_inner_reports": discover_existing_inner_links(data_root, repo_root, http_root, args.report_url_base),
        "existing_bridge_reports": discover_existing_bridge_links(data_root, args.large_marker_sequence, http_root, args.report_url_base),
        "bridge_layout": bridge_layout,
        "bridge_intrinsics": bridge_intrinsics,
        "bridge_candidates": bridge_candidates(large_scan),
        "priors": priors,
        "stages": stages,
        "final_yaml_candidates": final_candidates,
    }

    run_manifest = build_run_manifest(summary, run_started_at, run_finished_at, total_duration)
    write_run_manifest(manifest_path, run_manifest)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_index_html(index_path, summary)
    write_report_entrypoints(output_root, summary)

    print(json.dumps({
        "summary_json": str(summary_path),
        "run_manifest_json": str(manifest_path),
        "index_html": str(index_path),
        "final_report_url": summary["report_urls"]["final_report"],
        "small_marker_status": small_scan["status"],
        "large_inner_marker_status": large_inner_scan["status"],
        "large_marker_status": large_scan["status"],
        "mode": summary["mode"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
