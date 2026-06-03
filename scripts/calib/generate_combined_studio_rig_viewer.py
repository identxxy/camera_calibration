#!/usr/bin/env python3
"""Generate a standalone Three.js viewer for the combined inner/outer studio rig."""

import argparse
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from generate_threejs_rig_viewer import write_html as write_rig_viewer_html
    from studio_canonical_frame import (
        estimate_studio_canonical_frame,
        transform_point_to_aligned,
        transform_vector_to_aligned,
    )
except ModuleNotFoundError:
    from scripts.calib.generate_threejs_rig_viewer import write_html as write_rig_viewer_html
    from scripts.calib.studio_canonical_frame import (
        estimate_studio_canonical_frame,
        transform_point_to_aligned,
        transform_vector_to_aligned,
    )


T0_CALIB_ROOT = Path("/home/ubuntu/calib_data")
LEGACY_STAGE_ROOT = T0_CALIB_ROOT / "calib_2026_05_26_jpg_v3"
CURRENT_RUN_ROOT = (
    T0_CALIB_ROOT
    / "studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50"
)
CURRENT_INNER_BRIDGE_ROOT = CURRENT_RUN_ROOT / "inner_bridge"
CURRENT_OUTER_FRAME_FACE_ROOT = (
    CURRENT_RUN_ROOT / "outer_tower/frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1"
)

DEFAULT_BRIDGE_POSE_YAML = (
    CURRENT_INNER_BRIDGE_ROOT
    / "bridge_colmap_inner_refined_v1/camera_tr_inner_refined_plus_outer_topdown.yaml"
)
DEFAULT_BRIDGE_SUMMARY_JSON = (
    CURRENT_INNER_BRIDGE_ROOT
    / "bridge_colmap_inner_refined_v1/bridge_summary.json"
)
DEFAULT_OUTER_IMAGES_TXT = LEGACY_STAGE_ROOT / "colmap_outer24_firstframe_colmap404_v3/fixed_intrinsics/sparse_txt_final24_fixedK_ba/images.txt"
DEFAULT_OUTER_SUMMARY_JSON = LEGACY_STAGE_ROOT / "colmap_outer24_firstframe_colmap404_v3/fixed_intrinsics/summary_final24_fixedK_ba.json"
DEFAULT_OUTER_FINAL_POSE_YAML = CURRENT_OUTER_FRAME_FACE_ROOT / "camera_tr_rig_delta_refined.yaml"
DEFAULT_COMBINED_IMAGE_DIRS = CURRENT_INNER_BRIDGE_ROOT / "planned_inputs/large_marker_usable_image_directories.txt"
DEFAULT_VIEWER_ASSETS_DIR = LEGACY_STAGE_ROOT / "final_inner8_calibration_v1/reports/interactive_rig_viewer_v1"
DEFAULT_OUTPUT_HTML = CURRENT_INNER_BRIDGE_ROOT / "combined_studio_rig_viewer_v1/index.html"
DEFAULT_OUTER_OUTPUT_HTML = CURRENT_OUTER_FRAME_FACE_ROOT / "outer24_rig_viewer_v1/index.html"
DEFAULT_TOWER_POSE_YAML = CURRENT_OUTER_FRAME_FACE_ROOT / "rig_tr_global.yaml"
DEFAULT_WHOLE_COVERAGE_TSV = Path(
    "/home/ubuntu/calib_data/calib_2026_05_31_v3/"
    "whole_outer24_filtered_min4_hybrid_min4cam/per_camera_stats.tsv"
)
DEFAULT_LARGE_MARKER_PNP_SUMMARY_TSV = (
    CURRENT_INNER_BRIDGE_ROOT / "large_marker_bridge_all32/"
    "fixed_intrinsic_bridge_pnp_stride1_v1/camera_pnp_summary.tsv"
)
DEFAULT_SMALL_MARKER_PNP_SUMMARY_TSV = (
    CURRENT_INNER_BRIDGE_ROOT / "small_marker_inner8/"
    "fixed_intrinsic_small_grid4_quality_probe_v1/camera_pnp_summary.tsv"
)
DEFAULT_INNER_REPROJECTION_METRICS_TSV = (
    CURRENT_INNER_BRIDGE_ROOT / "reports/inner_reprojection/"
    "camera_metrics.tsv"
)
DEFAULT_INNER_INTRINSICS_DIR = (
    CURRENT_INNER_BRIDGE_ROOT / "planned_inputs/bridge_all32_fixed_intrinsics"
)
DEFAULT_OUTER_REPROJECTION_TSV = (
    CURRENT_OUTER_FRAME_FACE_ROOT / "diagnostics/camera_reprojection.tsv"
)
DEFAULT_OUTER_INTRINSICS_DIR = (
    CURRENT_OUTER_FRAME_FACE_ROOT / "intrinsics_refined"
)
DEFAULT_LARGE_MARKER_BOARD_POSE_YAML = (
    CURRENT_INNER_BRIDGE_ROOT / "large_marker_inner8/"
    "fixed_intrinsic_large_marker_inner8_init_v1/rig_tr_global.yaml"
)
DEFAULT_SMALL_MARKER_BOARD_POSE_YAML = (
    CURRENT_INNER_BRIDGE_ROOT / "small_marker_inner8/"
    "fixed_intrinsic_small_grid4_quality_probe_v1/rig_tr_global.yaml"
)
DEFAULT_BRIDGE_MARKER_BOARD_POSE_YAML = (
    CURRENT_INNER_BRIDGE_ROOT / "large_marker_bridge_all32/"
    "fixed_intrinsic_bridge_pnp_stride1_v1/rig_tr_global.yaml"
)
DEFAULT_INNER_BRIDGE_INDICES = "24,25,26,27,28,29,30,31"
DEFAULT_TOPDOWN_BRIDGE_INDICES = "9,10,11"
OUTER_CAMERA_LABELS = [
    "1-1", "1-2", "1-3",
    "2-1", "2-2", "2-3",
    "3-1", "3-2", "3-3",
    "4-1", "4-2", "4-3",
    "5-1", "5-2", "5-3",
    "6-1", "6-2", "6-3",
    "7-1", "7-2", "7-3",
    "8-1", "8-2", "8-3",
]
DEFAULT_IMAGE_WIDTH = 4096.0
DEFAULT_IMAGE_HEIGHT = 3000.0
INTRINSICS_CENTER_MARGIN_FRACTION = 0.15
INTRINSICS_RATIO_HARD_RANGE = (0.8, 1.25)
INTRINSICS_FOCAL_SCALE_WARNING_RANGE = (0.4, 2.0)


def quat_xyzw_to_matrix(qx, qy, qz, qw):
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("Quaternion has zero norm")
    q /= norm
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_wxyz_to_matrix(qw, qx, qy, qz):
    return quat_xyzw_to_matrix(qx, qy, qz, qw)


def pose_matrix(rotation, translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def pose_from_yaml_node(node):
    return pose_matrix(
        quat_xyzw_to_matrix(
            float(node["qx"]),
            float(node["qy"]),
            float(node["qz"]),
            float(node["qw"]),
        ),
        [float(node["tx"]), float(node["ty"]), float(node["tz"])],
    )


def invert_pose(matrix):
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inv


def load_pose_yaml(path):
    pose_count = None
    poses_by_index = {}
    current = None

    def flush_current():
        if current is None:
            return
        index = int(current["index"])
        poses_by_index[index] = pose_from_yaml_node(current)

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("pose_count:"):
            pose_count = int(line.split(":", 1)[1].strip())
        elif line.startswith("- index:"):
            flush_current()
            current = {"index": line.split(":", 1)[1].strip()}
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()

    flush_current()
    if pose_count is None:
        raise ValueError(f"Could not find pose_count in {path}")
    poses = [None for _ in range(pose_count)]
    for index, pose in poses_by_index.items():
        poses[index] = pose
    return poses


def parse_colmap_label(name):
    match = re.search(r"cam\d+_([^_]+)_f\d+", name)
    if match:
        return match.group(1)
    return Path(name).stem


def load_colmap_images(path):
    images = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            point_line = f.readline()
            point_parts = point_line.split() if point_line else []
            point_ids = point_parts[2::3]
            triangulated = sum(1 for point_id in point_ids if point_id != "-1")
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]
            camera_tr_world = pose_matrix(
                quat_wxyz_to_matrix(qw, qx, qy, qz),
                [tx, ty, tz],
            )
            world_tr_camera = invert_pose(camera_tr_world)
            label = parse_colmap_label(name)
            images[label] = {
                "image_id": image_id,
                "label": label,
                "name": name,
                "camera_tr_world": camera_tr_world,
                "world_tr_camera": world_tr_camera,
                "center_world": world_tr_camera[:3, 3],
                "point2d_count": len(point_ids),
                "triangulated_point_count": triangulated,
            }
    return images


def load_json_if_present(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path):
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        return []
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def to_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_or_none(value):
    parsed = to_float(value, None)
    if parsed is None or not math.isfinite(parsed):
        return None
    return float(parsed)


def parse_intrinsics_yaml(path):
    path = Path(path)
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"parameters\s*:\s*\[([^\]]+)\]",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return {}
    values = []
    for item in match.group(1).split(","):
        value = finite_or_none(item.strip())
        if value is not None:
            values.append(value)
    if len(values) < 4:
        return {}
    parsed = {
        "fx": values[0],
        "fy": values[1],
        "cx": values[2],
        "cy": values[3],
    }
    image_size = parse_yaml_image_size(text)
    if image_size:
        width, height = image_size
    else:
        width = parse_yaml_scalar(text, "width")
        height = parse_yaml_scalar(text, "height")
    if width is not None and height is not None and width > 0 and height > 0:
        parsed.update({
            "image_width": width,
            "image_height": height,
            "image_size": [width, height],
        })
    return parsed


def parse_yaml_scalar(text, key):
    match = re.search(
        rf"^\s*{re.escape(key)}\s*:\s*([-+0-9.eE]+)",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return None
    return finite_or_none(match.group(1))


def parse_yaml_image_size(text):
    match = re.search(
        r"^\s*(?:image_size|resolution)\s*:\s*\[([^\]]+)\]",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        return None
    values = []
    for item in match.group(1).split(","):
        value = finite_or_none(item.strip())
        if value is not None:
            values.append(value)
    if len(values) < 2 or values[0] <= 0 or values[1] <= 0:
        return None
    return values[0], values[1]


def intrinsic_file_for_index(directory, index):
    directory = Path(directory)
    if not directory.is_dir():
        return None
    exact = directory / f"intrinsics{index}.yaml"
    if exact.is_file():
        return exact
    matches = sorted(directory.glob(f"intrinsics{index}_*.yaml"))
    return matches[0] if matches else None


def auto_inner_intrinsics_index_offset(directory):
    directory = Path(directory)
    has_compact = all((directory / f"intrinsics{i}.yaml").is_file() for i in range(8))
    has_all32_inner = all((directory / f"intrinsics{i}.yaml").is_file() for i in range(24, 32))
    if has_all32_inner:
        return 24
    if has_compact:
        return 0
    return 0


def load_inner_intrinsics(directory, index_offset=None):
    intrinsics = {}
    if not directory:
        return intrinsics
    if index_offset is None or int(index_offset) < 0:
        index_offset = auto_inner_intrinsics_index_offset(directory)
    for index in range(8):
        path = intrinsic_file_for_index(directory, index + int(index_offset))
        if path:
            intrinsics[index] = parse_intrinsics_yaml(path)
    return intrinsics


def load_outer_intrinsics_from_dir(directory):
    intrinsics = {}
    if not directory:
        return intrinsics
    directory = Path(directory)
    if not directory.is_dir():
        return intrinsics
    for index, label in enumerate(OUTER_CAMERA_LABELS):
        path = intrinsic_file_for_index(directory, index)
        if path:
            parsed = parse_intrinsics_yaml(path)
            if parsed:
                intrinsics[label] = parsed
    return intrinsics


def inactive_coverage(label, reason):
    return {
        "active": False,
        "status": "inactive",
        "quality": "inactive",
        "observation_count": 0,
        "detail": reason,
    }


def build_inner_serial_label_map(large_rows, small_rows):
    mapping = {}
    for row in small_rows:
        idx = to_int(row.get("camera_index"), -1)
        user_id = str(row.get("user_id") or "").strip()
        if 0 <= idx <= 7 and user_id:
            mapping[user_id] = f"inner{idx}"
    for row in large_rows:
        idx = to_int(row.get("camera_index"), -1)
        user_id = str(row.get("user_id") or "").strip()
        if idx >= 24 and user_id:
            mapping[user_id] = f"inner{idx - 24}"
    return mapping


def whole_coverage_by_label(path, serial_to_inner):
    coverage = {}
    for row in read_tsv(path):
        raw_label = str(row.get("camera_id") or row.get("user_id") or "").strip()
        label = serial_to_inner.get(raw_label, raw_label)
        if not label:
            continue
        selected_passing = to_int(row.get("selected_passing_frames"), to_int(row.get("passing_images")))
        passing = to_int(row.get("passing_images"))
        total_tags = to_int(row.get("total_tags"))
        max_tags = to_int(row.get("max_tags"))
        active = selected_passing > 0
        if selected_passing >= 50:
            quality = "strong"
        elif selected_passing >= 10:
            quality = "usable"
        elif selected_passing > 0:
            quality = "weak"
        else:
            quality = "inactive"
        coverage[label] = {
            "active": active,
            "status": "observed" if active else "not_observed",
            "quality": quality,
            "observation_count": selected_passing,
            "positive_views": passing,
            "total_tags": total_tags,
            "max_tags": max_tags,
            "detail": (
                f"{selected_passing} selected frames with >=4 tower tags; "
                f"{passing} passing images; {total_tags} tags total"
            ),
        }
    return coverage


def outer_final_whole_coverage_by_label(path):
    coverage = {}
    for row in read_tsv(path):
        label = str(row.get("camera_id") or row.get("user_id") or "").strip()
        if not label:
            continue
        observations = to_int(row.get("observation_count"), to_int(row.get("residual_count")))
        median_px = to_float(
            row.get("after_median_px"),
            to_float(row.get("final_median_px"), to_float(row.get("median_error_px"), None)),
        )
        p90_px = to_float(
            row.get("after_p90_px"),
            to_float(row.get("final_p90_px"), to_float(row.get("p90_error_px"), None)),
        )
        active = observations > 0
        if observations >= 500:
            quality = "strong"
        elif observations >= 50:
            quality = "usable"
        elif observations > 0:
            quality = "weak"
        else:
            quality = "inactive"
        detail = f"{observations} accepted final tag-corner residuals"
        if median_px is not None or p90_px is not None:
            detail += f"; median/p90 {median_px if median_px is not None else '-'} / {p90_px if p90_px is not None else '-'} px"
        coverage[label] = {
            "active": active,
            "status": "accepted_final_residuals" if active else "not_used_by_final_outer_solve",
            "quality": quality,
            "observation_count": observations,
            "final_residual_count": observations,
            "median_error_px": median_px,
            "p90_error_px": p90_px,
            "detail": detail,
        }
    return coverage


def pnp_coverage_by_label(path, serial_to_inner=None, inner_offset=None):
    serial_to_inner = serial_to_inner or {}
    coverage = {}
    for row in read_tsv(path):
        idx = to_int(row.get("camera_index"), -1)
        user_id = str(row.get("user_id") or "").strip()
        if inner_offset is not None:
            label = f"inner{idx - inner_offset}" if idx >= inner_offset else user_id
        else:
            label = serial_to_inner.get(user_id, user_id)
        if not label:
            continue
        connected = str(row.get("connected") or "").strip().lower() == "yes"
        positive = to_int(row.get("positive_views"))
        solved = to_int(row.get("solved_views"))
        inliers = to_int(row.get("total_inliers"))
        error_px = to_float(row.get("median_view_error_px"), None)
        if connected and solved >= 10:
            quality = "strong"
        elif connected and solved > 0:
            quality = "usable"
        elif positive > 0:
            quality = "observed_only"
        else:
            quality = "inactive"
        coverage[label] = {
            "active": connected,
            "status": "connected" if connected else ("observed_only" if positive > 0 else "not_observed"),
            "quality": quality,
            "observation_count": solved,
            "positive_views": positive,
            "total_inliers": inliers,
            "median_view_error_px": error_px,
            "detail": (
                f"{solved} solved views; {positive} positive views; "
                f"connected={str(row.get('connected') or '').strip() or 'unknown'}"
            ),
        }
    return coverage


def attach_dataset_coverage(cameras, args):
    whole_path = getattr(args, "whole_coverage_tsv", None)
    outer_reprojection_path = getattr(args, "outer_reprojection_tsv", None)
    large_path = getattr(args, "large_marker_pnp_summary_tsv", None)
    small_path = getattr(args, "small_marker_pnp_summary_tsv", None)
    large_rows = read_tsv(large_path)
    small_rows = read_tsv(small_path)
    serial_to_inner = build_inner_serial_label_map(large_rows, small_rows)

    whole_raw = whole_coverage_by_label(whole_path, serial_to_inner)
    whole_final = outer_final_whole_coverage_by_label(outer_reprojection_path)
    whole = dict(whole_raw)
    for label, final_item in whole_final.items():
        raw_item = whole_raw.get(label)
        if raw_item:
            final_item = dict(final_item)
            final_item["raw_selected_passing_frames"] = raw_item.get("observation_count")
            final_item["raw_passing_images"] = raw_item.get("positive_views")
            final_item["total_tags"] = raw_item.get("total_tags")
            final_item["max_tags"] = raw_item.get("max_tags")
            final_item["detail"] += (
                f"; raw QC selected frames {raw_item.get('observation_count')}, "
                f"passing images {raw_item.get('positive_views')}"
            )
        whole[label] = final_item
    large = pnp_coverage_by_label(
        large_path,
        serial_to_inner=serial_to_inner,
        inner_offset=24,
    )
    small = pnp_coverage_by_label(
        small_path,
        serial_to_inner=serial_to_inner,
        inner_offset=0,
    )

    sources = {
        "whole": str(Path(whole_path).resolve()) if whole_path else "",
        "whole_final": str(Path(outer_reprojection_path).resolve()) if outer_reprojection_path else "",
        "large_marker": str(Path(large_path).resolve()) if large_path else "",
        "small_marker": str(Path(small_path).resolve()) if small_path else "",
    }
    active_counts = {"whole": 0, "large_marker": 0, "small_marker": 0}
    for camera in cameras:
        label = camera["label"]
        camera["coverage"] = {
            "whole": whole.get(label, inactive_coverage(label, "No >=4-tag whole/tower observation in the selected all32 QC set.")),
            "large_marker": large.get(label, inactive_coverage(label, "No connected large-marker all32 PnP solve for this camera.")),
            "small_marker": small.get(label, inactive_coverage(label, "Small marker is inner-only, or this inner camera was not connected in the small-marker probe.")),
        }
        for key, item in camera["coverage"].items():
            if item.get("active"):
                active_counts[key] += 1
    return {
        "default_mode": "whole",
        "modes": {
            "whole": {
                "label": "Whole",
                "description": "AprilTag tower whole capture/final-solve coverage; active means this camera contributed accepted tag corners to the current outer solve, or has selected raw whole-tower frames when final residual metadata is unavailable.",
                "source": sources["whole"],
                "final_residual_source": sources["whole_final"],
                "active_camera_count": active_counts["whole"],
            },
            "large_marker": {
                "label": "Large Marker",
                "description": "Large-marker all32 bridge PnP connectivity; active means this camera was connected in the fixed-intrinsic bridge solve.",
                "source": sources["large_marker"],
                "active_camera_count": active_counts["large_marker"],
            },
            "small_marker": {
                "label": "Small Marker",
                "description": "Small-marker inner8 fixed-rig probe connectivity; active means this inner camera was connected in the probe.",
                "source": sources["small_marker"],
                "active_camera_count": active_counts["small_marker"],
            },
        },
    }


def load_inner_quality_by_label(metrics_tsv, intrinsics_dir, intrinsics_index_offset=None):
    intrinsics = load_inner_intrinsics(intrinsics_dir, intrinsics_index_offset)
    quality = {}
    for row in read_tsv(metrics_tsv):
        index = to_int(row.get("camera_index"), -1)
        if not 0 <= index <= 7:
            continue
        label = f"inner{index}"
        item = {
            "source": "large_marker_inner_reprojection",
            "stage": row.get("stage_name") or "inner final reprojection",
            "decision": "inner_final",
            "observation_count": to_int(row.get("residual_count")),
            "residual_count": to_int(row.get("residual_count")),
            "median_error_px": finite_or_none(row.get("median_error_px")),
            "mean_error_px": finite_or_none(row.get("mean_error_px")),
            "p90_error_px": finite_or_none(row.get("p90_error_px")),
            "max_error_px": finite_or_none(row.get("max_error_px")),
            "camera_label": row.get("camera_label") or row.get("user_id") or label,
        }
        item.update({key: value for key, value in intrinsics.get(index, {}).items() if value is not None})
        quality[label] = item
    return quality


def load_outer_quality_by_label(residuals_tsv, intrinsics_tsv=None, intrinsics_dir=None):
    intrinsics = load_outer_intrinsics_from_dir(intrinsics_dir)
    for row in read_tsv(intrinsics_tsv):
        label = str(row.get("camera_id") or row.get("user_id") or "").strip()
        if not label:
            continue
        parsed = {
            "fx": finite_or_none(row.get("output_fx") or row.get("fx")),
            "fy": finite_or_none(row.get("output_fy") or row.get("fy")),
            "cx": finite_or_none(row.get("output_cx") or row.get("cx")),
            "cy": finite_or_none(row.get("output_cy") or row.get("cy")),
            "intrinsics_decision": row.get("output_intrinsics") or row.get("decision") or "",
        }
        width = finite_or_none(row.get("output_width") or row.get("image_width") or row.get("width"))
        height = finite_or_none(row.get("output_height") or row.get("image_height") or row.get("height"))
        if width is not None and height is not None and width > 0 and height > 0:
            parsed.update({
                "image_width": width,
                "image_height": height,
                "image_size": [width, height],
            })
        merged = dict(intrinsics.get(label, {}))
        merged.update({key: value for key, value in parsed.items() if value is not None and value != ""})
        intrinsics[label] = merged
    quality = {}
    for row in read_tsv(residuals_tsv):
        label = str(row.get("camera_id") or row.get("camera_label") or row.get("user_id") or "").strip()
        if not label:
            continue
        if "final_median_px" in row:
            median = row.get("final_median_px")
            p90 = row.get("final_p90_px")
            max_px = row.get("final_max_px")
            decision = row.get("decision") or ""
            source = "whole_outer_tag_refine_accepted"
        else:
            median = row.get("after_median_px")
            p90 = row.get("after_p90_px")
            max_px = row.get("after_max_px")
            decision = "accepted_output"
            source = "whole_outer_tag_refine_accepted"
        item = {
            "source": source,
            "stage": "outer final accepted reprojection",
            "decision": decision,
            "decision_reason": row.get("decision_reason") or "",
            "observation_count": to_int(row.get("observation_count")),
            "residual_count": to_int(row.get("observation_count")),
            "median_error_px": finite_or_none(median),
            "p90_error_px": finite_or_none(p90),
            "max_error_px": finite_or_none(max_px),
            "under_100_fraction": finite_or_none(row.get("final_under_100_fraction") or row.get("after_under_100_fraction")),
            "under_300_fraction": finite_or_none(row.get("final_under_300_fraction") or row.get("after_under_300_fraction")),
        }
        item.update({key: value for key, value in intrinsics.get(label, {}).items() if value is not None and value != ""})
        quality[label] = item
    return quality


def intrinsic_image_size(item):
    size = item.get("image_size")
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        width = finite_or_none(size[0])
        height = finite_or_none(size[1])
        if width is not None and height is not None and width > 0 and height > 0:
            return width, height
    width = finite_or_none(item.get("image_width") or item.get("width"))
    height = finite_or_none(item.get("image_height") or item.get("height"))
    if width is not None and height is not None and width > 0 and height > 0:
        return width, height
    return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def intrinsic_sanity(item):
    width, height = intrinsic_image_size(item)
    max_dim = max(width, height)
    fx = finite_or_none(item.get("fx"))
    fy = finite_or_none(item.get("fy"))
    cx = finite_or_none(item.get("cx"))
    cy = finite_or_none(item.get("cy"))
    failures = []
    warnings = []

    if fx is None or fx <= 0:
        failures.append("missing_or_nonpositive_fx")
    if fy is None or fy <= 0:
        failures.append("missing_or_nonpositive_fy")

    if fx is not None and fy is not None and fx > 0 and fy > 0:
        ratio = fx / fy
        ratio_min, ratio_max = INTRINSICS_RATIO_HARD_RANGE
        if ratio < ratio_min or ratio > ratio_max:
            failures.append("fx_fy_ratio_outside_expected_range")
        scale_min, scale_max = INTRINSICS_FOCAL_SCALE_WARNING_RANGE
        fx_scale = fx / max_dim
        fy_scale = fy / max_dim
        if fx_scale < scale_min or fx_scale > scale_max or fy_scale < scale_min or fy_scale > scale_max:
            warnings.append("focal_scale_outside_expected_range")

    if cx is None or cy is None:
        failures.append("missing_principal_point")
    else:
        margin_x = width * INTRINSICS_CENTER_MARGIN_FRACTION
        margin_y = height * INTRINSICS_CENTER_MARGIN_FRACTION
        if cx < -margin_x or cx > width + margin_x or cy < -margin_y or cy > height + margin_y:
            failures.append("principal_point_outside_image_margin")

    flags = failures + [flag for flag in warnings if flag not in failures]
    if failures:
        status = "failed"
    elif warnings:
        status = "warning"
    else:
        status = "ok"
    return {
        "intrinsics_status": status,
        "intrinsics_flags": flags,
        "intrinsics_failure_flags": failures,
        "intrinsics_warning_flags": warnings,
        "intrinsics_image_size": [width, height],
    }


def summarize_intrinsic_sanity(cameras):
    summary = {
        "total_camera_count": len(cameras),
        "ok_camera_count": 0,
        "warning_camera_count": 0,
        "failed_camera_count": 0,
        "missing_camera_count": 0,
        "warning_cameras": [],
        "failed_cameras": [],
        "missing_cameras": [],
    }
    for camera in cameras:
        label = camera["label"]
        quality = camera.get("calibration_quality") or {}
        status = quality.get("intrinsics_status") or "missing"
        if status == "failed":
            summary["failed_camera_count"] += 1
            summary["failed_cameras"].append(label)
        elif status == "warning":
            summary["warning_camera_count"] += 1
            summary["warning_cameras"].append(label)
        elif status == "ok":
            summary["ok_camera_count"] += 1
        else:
            summary["missing_camera_count"] += 1
            summary["missing_cameras"].append(label)
    return summary


def attach_calibration_quality(cameras, args):
    inner = load_inner_quality_by_label(
        getattr(args, "inner_reprojection_metrics_tsv", None),
        getattr(args, "inner_intrinsics_dir", None),
        getattr(args, "inner_intrinsics_index_offset", None),
    )
    outer = load_outer_quality_by_label(
        getattr(args, "outer_reprojection_tsv", None),
        getattr(args, "outer_intrinsics_tsv", None),
        getattr(args, "outer_intrinsics_dir", None),
    )
    for camera in cameras:
        label = camera["label"]
        item = inner.get(label) if str(label).startswith("inner") else outer.get(label)
        if not item:
            item = {
                "source": "missing",
                "stage": "not available",
                "decision": "missing",
                "observation_count": None,
                "median_error_px": None,
                "p90_error_px": None,
                "max_error_px": None,
            }
        item.update(intrinsic_sanity(item))
        camera["calibration_quality"] = item
    return {
        "inner_source": str(Path(getattr(args, "inner_reprojection_metrics_tsv", "")).resolve()) if getattr(args, "inner_reprojection_metrics_tsv", None) else "",
        "outer_residual_source": str(Path(getattr(args, "outer_reprojection_tsv", "")).resolve()) if getattr(args, "outer_reprojection_tsv", None) else "",
        "outer_intrinsics_source": str(Path(getattr(args, "outer_intrinsics_tsv", "")).resolve()) if getattr(args, "outer_intrinsics_tsv", None) else "",
        "outer_intrinsics_dir": str(Path(getattr(args, "outer_intrinsics_dir", "")).resolve()) if getattr(args, "outer_intrinsics_dir", None) else "",
        "inner_camera_count": sum(1 for camera in cameras if str(camera["label"]).startswith("inner") and camera["calibration_quality"].get("source") != "missing"),
        "outer_camera_count": sum(1 for camera in cameras if not str(camera["label"]).startswith("inner") and camera["calibration_quality"].get("source") != "missing"),
        "intrinsic_sanity": summarize_intrinsic_sanity(cameras),
    }


def label_sort_key(label):
    pieces = re.split(r"(\d+)", str(label))
    key = []
    for piece in pieces:
        if piece.isdigit():
            key.append(int(piece))
        else:
            key.append(piece)
    return key


def umeyama_similarity(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] < 3 or source.shape[1] != 3:
        raise ValueError(f"Need at least three paired 3D points, got {source.shape} -> {target.shape}")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.eye(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        reflection[-1, -1] = -1.0
    rotation = vt.T @ reflection @ u.T
    source_variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    if source_variance <= 0:
        raise ValueError("COLMAP alignment anchors are degenerate")
    scale = float(np.sum(singular_values * np.diag(reflection)) / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    predicted = (scale * (rotation @ source.T)).T + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "singular_values": singular_values,
        "residuals": residuals,
        "source_mean": source_mean,
        "target_mean": target_mean,
    }


def transform_colmap_pose(world_tr_camera, sim3):
    rotation = sim3["rotation"] @ world_tr_camera[:3, :3]
    center = sim3["scale"] * sim3["rotation"] @ world_tr_camera[:3, 3] + sim3["translation"]
    return pose_matrix(rotation, center)


def to_three(point):
    point = np.asarray(point, dtype=np.float64)
    return [float(point[0]), float(-point[1]), float(-point[2])]


def vector_to_three(vector):
    vector = np.asarray(vector, dtype=np.float64)
    mapped = np.asarray([vector[0], -vector[1], -vector[2]], dtype=np.float64)
    norm = np.linalg.norm(mapped)
    if norm > 0:
        mapped /= norm
    return [float(mapped[0]), float(mapped[1]), float(mapped[2])]


def metric_vector_from_three(vector):
    vector = np.asarray(vector, dtype=np.float64)
    mapped = np.asarray([vector[0], -vector[1], -vector[2]], dtype=np.float64)
    norm = np.linalg.norm(mapped)
    if norm > 0:
        mapped /= norm
    return [float(mapped[0]), float(mapped[1]), float(mapped[2])]


def estimate_outer_column_gravity_alignment(cameras):
    centers_metric = {}
    centers_display = {}
    for camera in cameras:
        label = str(camera.get("label") or "")
        if not re.match(r"^[1-8]-[123]$", label):
            continue
        if label.startswith("4-"):
            continue
        if camera.get("center_metric") is not None:
            centers_metric[label] = np.asarray(camera.get("center_metric"), dtype=np.float64)
        elif label not in centers_display:
            centers_display[label] = np.asarray(camera.get("center"), dtype=np.float64)

    frame = estimate_studio_canonical_frame(centers_metric or centers_display)
    if frame is None:
        return None
    physical_up = -np.asarray(frame["axes_source"]["y"], dtype=np.float64)
    display_up = vector_to_three(physical_up) if centers_metric else [float(v) for v in physical_up]
    metric_up = metric_vector_from_three(display_up)
    result = {
        "method": frame["method"],
        "source": frame["source"],
        "display_up_vector": [float(v) for v in display_up],
        "metric_up_vector": metric_up,
        "column_count": int(frame["column_count"]),
        "used_columns": frame["used_columns"],
        "level_plane_count": int(frame["level_plane_count"]),
        "origin_source": frame["origin_source"],
        "axes_source": frame["axes_source"],
        "negative_z_gap_direction_source": frame["negative_z_gap_direction_source"],
        "negative_z_gap_labels": frame["negative_z_gap_labels"],
        "origin_level2_labels": frame["origin_level2_labels"],
        "level_plane_normals_source": frame["level_plane_normals_source"],
    }
    return result


def update_camera_display_geometry(camera):
    center_metric = np.asarray(camera["center_metric"], dtype=np.float64)
    basis_metric = camera["basis_metric"]
    camera["center_metric"] = [float(v) for v in center_metric]
    camera["basis_metric"] = {
        axis: [float(v) for v in np.asarray(vector, dtype=np.float64)]
        for axis, vector in basis_metric.items()
    }
    center = to_three(center_metric)
    x_axis = vector_to_three(camera["basis_metric"]["x"])
    y_axis = vector_to_three(camera["basis_metric"]["y"])
    z_axis = vector_to_three(camera["basis_metric"]["z"])
    camera["center"] = center
    camera["basis"] = {
        "x": x_axis,
        "y": y_axis,
        "z": z_axis,
    }
    camera["axes"] = {
        "x": axis_line(center, x_axis, 0.16),
        "y": axis_line(center, y_axis, 0.16),
        "z": axis_line(center, z_axis, 0.16),
    }


def apply_canonical_frame_to_cameras(cameras):
    label_to_center = {
        str(camera.get("label")): np.asarray(camera.get("center_metric"), dtype=np.float64)
        for camera in cameras
        if camera.get("center_metric") is not None
    }
    frame = estimate_studio_canonical_frame(label_to_center)
    if frame is None:
        return None
    for camera in cameras:
        if camera.get("center_metric") is None or camera.get("basis_metric") is None:
            continue
        camera["center_metric_source"] = [float(v) for v in camera["center_metric"]]
        camera["center_metric"] = [
            float(v) for v in transform_point_to_aligned(camera["center_metric"], frame)
        ]
        camera["basis_metric_source"] = {
            axis: [float(v) for v in vector]
            for axis, vector in camera["basis_metric"].items()
        }
        camera["basis_metric"] = {
            axis: [float(v) for v in transform_vector_to_aligned(vector, frame)]
            for axis, vector in camera["basis_metric"].items()
        }
        update_camera_display_geometry(camera)
    return frame


def estimate_tower_up_from_pose_yaml(path):
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    poses = load_pose_yaml(path)
    up_vectors = []
    for pose in poses:
        if pose is None:
            continue
        up = pose[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        norm = np.linalg.norm(up)
        if norm > 0 and np.all(np.isfinite(up)):
            up_vectors.append(up / norm)
    if not up_vectors:
        return None
    arr = np.asarray(up_vectors, dtype=np.float64)
    mean = np.mean(arr, axis=0)
    mean_norm = np.linalg.norm(mean)
    if mean_norm <= 0:
        return None
    mean /= mean_norm
    robust_keep = np.ones(len(arr), dtype=bool)
    robust_angle_threshold_deg = 30.0
    # Tower tag columns are physical gravity-aligned edges. Use the consensus
    # tower +z direction, not sparse outlier frame poses, for the viewer up.
    for _ in range(3):
        dots_iter = np.clip(arr @ mean, -1.0, 1.0)
        next_keep = dots_iter >= math.cos(math.radians(robust_angle_threshold_deg))
        if int(np.sum(next_keep)) < max(8, int(0.5 * len(arr))):
            break
        next_mean = np.mean(arr[next_keep], axis=0)
        next_norm = np.linalg.norm(next_mean)
        if next_norm <= 0:
            break
        robust_keep = next_keep
        mean = next_mean / next_norm
    robust_arr = arr[robust_keep]
    dots = np.clip(robust_arr @ mean, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    display = vector_to_three(mean)
    return {
        "source": str(path.resolve()),
        "sample_count": int(len(up_vectors)),
        "robust_sample_count": int(np.sum(robust_keep)),
        "rejected_sample_count": int(len(up_vectors) - np.sum(robust_keep)),
        "robust_angle_threshold_deg": robust_angle_threshold_deg,
        "metric_up_vector": [float(v) for v in mean],
        "display_up_vector": display,
        "min_dot": float(np.min(dots)),
        "max_angle_deg": float(np.max(angles)),
        "median_angle_deg": float(np.median(angles)),
        "p90_angle_deg": float(np.percentile(angles, 90)),
    }


def collect_pose_normals(path, local_normals, gravity=None, vertical_axis=None, max_vertical_angle_deg=None):
    if not path:
        return np.zeros((0, 3), dtype=np.float64)
    path = Path(path)
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float64)
    poses = load_pose_yaml(path)
    normals = []
    for pose in poses:
        if pose is None:
            continue
        rotation = pose[:3, :3]
        if gravity is not None and vertical_axis is not None and max_vertical_angle_deg is not None:
            vertical = rotation @ np.asarray(vertical_axis, dtype=np.float64)
            vertical_norm = np.linalg.norm(vertical)
            if vertical_norm <= 0:
                continue
            vertical = vertical / vertical_norm
            dot = float(np.clip(vertical @ gravity, -1.0, 1.0))
            if dot < math.cos(math.radians(max_vertical_angle_deg)):
                continue
        for local in local_normals:
            normal = rotation @ np.asarray(local, dtype=np.float64)
            norm = np.linalg.norm(normal)
            if norm > 0 and np.all(np.isfinite(normal)):
                normals.append(normal / norm)
    if not normals:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(normals, dtype=np.float64)


def local_tower_face_normals():
    normals = []
    for face_index in range(8):
        angle = 2.0 * math.pi * face_index / 8.0
        normals.append([math.cos(angle), math.sin(angle), 0.0])
    return normals


def board_horizontal_stats(normals, gravity):
    if normals.size == 0 or gravity is None:
        return {
            "sample_count": 0,
            "median_angle_from_horizontal_deg": None,
            "p90_angle_from_horizontal_deg": None,
            "max_angle_from_horizontal_deg": None,
            "median_abs_dot_gravity": None,
        }
    gravity = np.asarray(gravity, dtype=np.float64)
    gravity_norm = np.linalg.norm(gravity)
    if gravity_norm <= 0:
        return {
            "sample_count": 0,
            "median_angle_from_horizontal_deg": None,
            "p90_angle_from_horizontal_deg": None,
            "max_angle_from_horizontal_deg": None,
            "median_abs_dot_gravity": None,
        }
    gravity = gravity / gravity_norm
    dots = np.abs(np.clip(normals @ gravity, -1.0, 1.0))
    angles = np.degrees(np.arcsin(np.clip(dots, 0.0, 1.0)))
    return {
        "sample_count": int(normals.shape[0]),
        "median_angle_from_horizontal_deg": float(np.median(angles)),
        "p90_angle_from_horizontal_deg": float(np.percentile(angles, 90)),
        "max_angle_from_horizontal_deg": float(np.max(angles)),
        "median_abs_dot_gravity": float(np.median(dots)),
    }


def estimate_board_orientation_alignment(args, gravity_alignment):
    gravity = None
    if gravity_alignment:
        gravity = np.asarray(gravity_alignment["metric_up_vector"], dtype=np.float64)
    sources = [
        (
            "whole_tower_faces",
            getattr(args, "tower_pose_yaml", None),
            local_tower_face_normals(),
            "AprilTag tower face normals; they should be horizontal when the tower vertical axis is aligned with gravity.",
        ),
        (
            "large_marker_inner8_board",
            getattr(args, "large_marker_board_pose_yaml", None),
            [[0.0, 0.0, 1.0]],
            "Large marker board plane normal from fixed-intrinsic inner8 PnP poses.",
        ),
        (
            "large_marker_bridge_all32_board",
            getattr(args, "bridge_marker_board_pose_yaml", None),
            [[0.0, 0.0, 1.0]],
            "Large marker bridge board plane normal from all32 fixed-intrinsic PnP poses.",
        ),
        (
            "small_marker_inner8_board",
            getattr(args, "small_marker_board_pose_yaml", None),
            [[0.0, 0.0, 1.0]],
            "Small marker board plane normal from fixed-intrinsic inner8 probe poses.",
        ),
    ]
    source_stats = {}
    all_normals = []
    for name, path, local_normals, description in sources:
        if name == "whole_tower_faces" and gravity is not None and gravity_alignment:
            normals = collect_pose_normals(
                path,
                local_normals,
                gravity=gravity,
                vertical_axis=[0.0, 0.0, 1.0],
                max_vertical_angle_deg=gravity_alignment.get("robust_angle_threshold_deg", 30.0),
            )
        else:
            normals = collect_pose_normals(path, local_normals)
        if normals.size:
            all_normals.append(normals)
        stats = board_horizontal_stats(normals, gravity)
        stats.update({
            "source": str(Path(path).resolve()) if path else "",
            "description": description,
        })
        source_stats[name] = stats
    if all_normals and gravity is not None:
        aggregate_normals = np.concatenate(all_normals, axis=0)
    else:
        aggregate_normals = np.zeros((0, 3), dtype=np.float64)
    aggregate = board_horizontal_stats(aggregate_normals, gravity)
    return {
        "method": "viewer_gravity_alignment_with_board_normal_sanity",
        "gravity_source": gravity_alignment["source"] if gravity_alignment else "",
        "gravity_metric_up_vector": [float(v) for v in gravity] if gravity is not None else None,
        "gravity_display_up_vector": gravity_alignment["display_up_vector"] if gravity_alignment else None,
        "aggregate": aggregate,
        "sources": source_stats,
    }


def axis_line(center, basis_vector, length):
    center = np.asarray(center, dtype=np.float64)
    direction = np.asarray(basis_vector, dtype=np.float64)
    endpoint = center + length * direction
    return [
        [float(v) for v in center],
        [float(v) for v in endpoint],
    ]


def camera_from_rig_tr_camera(label, index, kind, source, rig_tr_camera, metrics):
    rotation = rig_tr_camera[:3, :3]
    camera = {
        "index": index,
        "label": label,
        "used": True,
        "kind": kind,
        "source": source,
        "center_metric": [float(v) for v in rig_tr_camera[:3, 3]],
        "basis_metric": {
            "x": rotation[:, 0],
            "y": rotation[:, 1],
            "z": rotation[:, 2],
        },
        "metrics": metrics,
        "image_url": "",
        "image_texture_url": "",
    }
    update_camera_display_geometry(camera)
    return camera


def bridge_center_from_camera_tr_rig(camera_tr_rig):
    return invert_pose(camera_tr_rig)[:3, 3]


def summarize_tracks(colmap_images):
    counts = [row["triangulated_point_count"] for row in colmap_images.values()]
    point2d_counts = [row["point2d_count"] for row in colmap_images.values()]
    if not counts:
        return {
            "triangulated_min": 0,
            "triangulated_median": 0,
            "triangulated_max": 0,
            "point2d_median": 0,
        }
    return {
        "triangulated_min": int(np.min(counts)),
        "triangulated_median": float(np.median(counts)),
        "triangulated_max": int(np.max(counts)),
        "point2d_median": float(np.median(point2d_counts)) if point2d_counts else 0,
    }


def read_image_dirs_file(path):
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    return [Path(item.strip()) for item in text.replace("\n", ",").split(",") if item.strip()]


def first_image_in_dir(image_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not image_dir or not Path(image_dir).is_dir():
        return None
    images = [path for path in sorted(Path(image_dir).iterdir()) if path.suffix.lower() in extensions]
    return images[0] if images else None


def safe_file_part(text):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def parse_image_dir_ordinal(path):
    match = re.match(r"cam(\d+)_", Path(path).name)
    return int(match.group(1)) if match else None


def parse_outer_label_from_image_dir(path):
    match = re.match(r"cam\d+_[^_]+_(\d+-\d+)$", Path(path).name)
    return match.group(1) if match else ""


def build_first_frame_map(args):
    image_map = {}
    for image_dir in read_image_dirs_file(args.combined_image_directories_file):
        image = first_image_in_dir(image_dir)
        if image is None:
            continue
        outer_label = parse_outer_label_from_image_dir(image_dir)
        if outer_label:
            image_map[outer_label] = image
            continue
        ordinal = parse_image_dir_ordinal(image_dir)
        if ordinal is not None and ordinal >= 24:
            image_map[f"inner{ordinal - 24}"] = image

    for index, image_dir in enumerate(read_image_dirs_file(args.inner_image_directories_file)):
        image = first_image_in_dir(image_dir)
        if image is not None:
            image_map.setdefault(f"inner{index}", image)

    for image_dir in read_image_dirs_file(args.outer_image_directories_file):
        image = first_image_in_dir(image_dir)
        outer_label = parse_outer_label_from_image_dir(image_dir)
        if image is not None and outer_label:
            image_map.setdefault(outer_label, image)
    return image_map


def clear_camera_frame_dir(frame_dir):
    if not frame_dir.exists():
        return
    for path in frame_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            path.unlink()


def write_viewer_texture_image(src, dst, max_width, quality):
    try:
        from PIL import Image, ImageOps
    except ImportError:
        shutil.copyfile(src, dst)
        return None

    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if max_width > 0:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image.thumbnail((max_width, max_width), resampling)
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, format="JPEG", quality=quality, optimize=True)
        return image.size


def attach_first_frame_images(cameras, args):
    output_dir = Path(args.output_html).resolve().parent
    frame_dir = output_dir / "camera_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    clear_camera_frame_dir(frame_dir)
    image_map = build_first_frame_map(args)
    copied = 0
    texture_pixels = 0
    max_width = int(getattr(args, "texture_max_width", 640))
    quality = int(getattr(args, "texture_jpeg_quality", 82))
    for camera in cameras:
        image = image_map.get(camera["label"])
        if image is None:
            continue
        dst = frame_dir / f"{camera['index']:02d}_{safe_file_part(camera['label'])}.jpg"
        texture_size = write_viewer_texture_image(image, dst, max_width, quality)
        dst.chmod(0o644)
        rel = dst.relative_to(output_dir).as_posix()
        camera["image_url"] = rel
        camera["image_texture_url"] = rel
        if texture_size:
            texture_pixels += texture_size[0] * texture_size[1]
        copied += 1
    return copied, {
        "first_frame_texture_max_width": max_width,
        "first_frame_texture_jpeg_quality": quality,
        "first_frame_texture_pixel_count": texture_pixels,
        "first_frame_texture_rgba_estimated_mb": round(texture_pixels * 4 / 1024 / 1024, 3),
    }


def copy_viewer_assets(output_dir, assets_dir):
    output_dir = Path(output_dir)
    assets_dir = Path(assets_dir)
    required = ["three.min.js", "OrbitControls.js", "TransformControls.js"]
    for name in required:
        src = assets_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, output_dir / name)


def build_metrics(
    bridge_summary,
    outer_summary,
    colmap_images,
    topdown_labels,
    sim3,
    outer_pose_source,
    outer_final_pose_yaml=None,
    outer_final_pose_count=0,
):
    bridge_votes = {}
    bridge_center_residuals = {}
    bridge_rot_residuals = {}
    for row in bridge_summary.get("outer_camera_summaries", []):
        label = row.get("label")
        if not label:
            continue
        bridge_votes[label] = row.get("vote_count")
        bridge_center_residuals[label] = row.get("center_residual_median_m")
        bridge_rot_residuals[label] = row.get("rotation_residual_median_deg")

    tracks = summarize_tracks(colmap_images)
    topdown_tracks = {
        label: {
            "point2d_count": colmap_images[label]["point2d_count"],
            "triangulated_point_count": colmap_images[label]["triangulated_point_count"],
        }
        for label in topdown_labels
        if label in colmap_images
    }
    if sim3 is None:
        sim3_metrics = {
            "sim3_scale": None,
            "sim3_residual_rms_m": None,
            "sim3_residual_median_m": None,
            "sim3_residual_max_m": None,
            "sim3_singular_values": [],
        }
    else:
        residuals = sim3["residuals"]
        sim3_metrics = {
            "sim3_scale": sim3["scale"],
            "sim3_residual_rms_m": float(np.sqrt(np.mean(residuals * residuals))),
            "sim3_residual_median_m": float(np.median(residuals)),
            "sim3_residual_max_m": float(np.max(residuals)),
            "sim3_singular_values": [float(x) for x in sim3["singular_values"]],
        }
    return {
        "display_camera_count": 0,
        "outer_pose_source": outer_pose_source,
        "outer_final_pose_yaml": str(Path(outer_final_pose_yaml).resolve()) if outer_final_pose_yaml else "",
        "outer_final_pose_count": outer_final_pose_count,
        "outer_colmap_registered_count": int(outer_summary.get("count") or len(colmap_images)),
        "outer_colmap_points3d_count": outer_summary.get("points3D_count"),
        "outer_colmap_mean_error_px": (outer_summary.get("point_error_mean_median_max") or [None])[0],
        "bridge_topdown_votes": bridge_votes,
        "bridge_topdown_center_residual_median_m": bridge_center_residuals,
        "bridge_topdown_rotation_residual_median_deg": bridge_rot_residuals,
        "topdown_colmap_tracks": topdown_tracks,
        "outer_colmap_track_summary": tracks,
        **sim3_metrics,
    }


def build_viewer_data(args):
    bridge_poses = load_pose_yaml(args.inner_bridge_pose_yaml)
    bridge_summary = load_json_if_present(args.bridge_summary_json)
    outer_summary = load_json_if_present(args.outer_colmap_summary_json)
    outer_final_pose_yaml = getattr(args, "outer_final_pose_yaml", None)
    outer_final_poses = None
    if outer_final_pose_yaml:
        outer_final_pose_yaml = Path(outer_final_pose_yaml)
        if not outer_final_pose_yaml.is_file():
            raise FileNotFoundError(f"--outer_final_pose_yaml does not exist: {outer_final_pose_yaml}")
        outer_final_poses = load_pose_yaml(outer_final_pose_yaml)
        if len(outer_final_poses) < len(OUTER_CAMERA_LABELS):
            raise ValueError(
                f"Outer final pose YAML has {len(outer_final_poses)} poses, "
                f"expected at least {len(OUTER_CAMERA_LABELS)}"
            )
        missing_outer = [
            label
            for index, label in enumerate(OUTER_CAMERA_LABELS)
            if outer_final_poses[index] is None
        ]
        if missing_outer:
            raise ValueError(f"Outer final pose YAML missing poses for labels: {missing_outer}")

    colmap_required = outer_final_poses is None
    if args.outer_colmap_images_txt and (colmap_required or Path(args.outer_colmap_images_txt).is_file()):
        colmap_images = load_colmap_images(args.outer_colmap_images_txt)
    else:
        colmap_images = {}
    topdown_labels = [label for label in args.topdown_labels.split(",") if label]
    topdown_indices = [int(index) for index in args.topdown_bridge_indices.split(",") if index]
    inner_indices = [int(index) for index in args.inner_bridge_indices.split(",") if index]
    if len(topdown_labels) != len(topdown_indices):
        raise ValueError("--topdown_labels and --topdown_bridge_indices must have the same length")

    missing_topdown = [label for label in topdown_labels if label not in colmap_images]
    if missing_topdown and outer_final_poses is None:
        raise ValueError(f"COLMAP images missing topdown alignment labels: {missing_topdown}")
    if max(topdown_indices + inner_indices, default=-1) >= len(bridge_poses):
        raise ValueError("Bridge pose YAML does not contain all requested bridge indices")

    sim3 = None
    if not missing_topdown:
        source_centers = []
        target_centers = []
        for label, index in zip(topdown_labels, topdown_indices):
            if bridge_poses[index] is None:
                raise ValueError(f"Bridge pose YAML missing topdown pose index {index} for {label}")
            source_centers.append(colmap_images[label]["center_world"])
            target_centers.append(bridge_center_from_camera_tr_rig(bridge_poses[index]))
        sim3 = umeyama_similarity(source_centers, target_centers)

    cameras = []
    next_index = 0
    if args.viewer_scope == "combined":
        for inner_ordinal, bridge_index in enumerate(inner_indices):
            if bridge_poses[bridge_index] is None:
                continue
            cameras.append(camera_from_rig_tr_camera(
                f"inner{inner_ordinal}",
                next_index,
                "inner",
                "bridge_metric_inner",
                invert_pose(bridge_poses[bridge_index]),
                {"bridge_index": bridge_index, "inner_ordinal": inner_ordinal},
            ))
            next_index += 1

    topdown_by_label = dict(zip(topdown_labels, topdown_indices))
    aligned_outer_bridge_pose_ready = False
    if outer_final_poses is not None:
        aligned_outer_bridge_pose_ready = (
            args.viewer_scope == "combined"
            and len(bridge_poses) >= len(OUTER_CAMERA_LABELS)
            and all(bridge_poses[index] is not None for index in range(len(OUTER_CAMERA_LABELS)))
        )
        for outer_index, label in enumerate(OUTER_CAMERA_LABELS):
            bridge_index = topdown_by_label.get(label)
            if aligned_outer_bridge_pose_ready:
                camera_tr_scene_rig = bridge_poses[outer_index]
                source = "outer_final_pose_yaml_bridge_aligned"
            else:
                camera_tr_scene_rig = outer_final_poses[outer_index]
                source = "outer_final_pose_yaml"
            metrics = {
                "outer_final_pose_index": outer_index,
                "bridge_aligned_outer_pose": bool(aligned_outer_bridge_pose_ready),
            }
            if label in colmap_images:
                metrics.update({
                    "colmap_image": colmap_images[label]["name"],
                    "point2d_count": colmap_images[label]["point2d_count"],
                    "triangulated_point_count": colmap_images[label]["triangulated_point_count"],
                })
            if bridge_index is not None and bridge_poses[bridge_index] is not None:
                metrics["bridge_index"] = bridge_index
            cameras.append(camera_from_rig_tr_camera(
                label,
                next_index,
                "outer_final",
                source,
                invert_pose(camera_tr_scene_rig),
                metrics,
            ))
            next_index += 1
    else:
        for label, bridge_index in zip(topdown_labels, topdown_indices):
            colmap = colmap_images[label]
            cameras.append(camera_from_rig_tr_camera(
                label,
                next_index,
                "outer_topdown",
                "bridge_metric_topdown",
                invert_pose(bridge_poses[bridge_index]),
                {
                    "bridge_index": bridge_index,
                    "colmap_image": colmap["name"],
                    "point2d_count": colmap["point2d_count"],
                    "triangulated_point_count": colmap["triangulated_point_count"],
                },
            ))
            next_index += 1

        topdown_set = set(topdown_labels)
        for label in sorted(colmap_images, key=label_sort_key):
            if label in topdown_set:
                continue
            colmap = colmap_images[label]
            cameras.append(camera_from_rig_tr_camera(
                label,
                next_index,
                "outer_colmap",
                "colmap_sim3_approx",
                transform_colmap_pose(colmap["world_tr_camera"], sim3),
                {
                    "colmap_image": colmap["name"],
                    "point2d_count": colmap["point2d_count"],
                    "triangulated_point_count": colmap["triangulated_point_count"],
                },
            ))
            next_index += 1

    canonical_frame = apply_canonical_frame_to_cameras(cameras)
    dataset_coverage = attach_dataset_coverage(cameras, args)
    calibration_quality = attach_calibration_quality(cameras, args)
    first_frame_count, texture_metrics = attach_first_frame_images(cameras, args)
    bounds = compute_bounds([camera["center"] for camera in cameras])
    if aligned_outer_bridge_pose_ready:
        outer_pose_source = "outer_final_pose_yaml_bridge_aligned"
    else:
        outer_pose_source = "outer_final_pose_yaml" if outer_final_poses is not None else "colmap_sim3_approx"
    metrics = build_metrics(
        bridge_summary,
        outer_summary,
        colmap_images,
        topdown_labels,
        sim3,
        outer_pose_source,
        outer_final_pose_yaml=outer_final_pose_yaml,
        outer_final_pose_count=len(OUTER_CAMERA_LABELS) if outer_final_poses is not None else 0,
    )
    metrics["display_camera_count"] = len(cameras)
    metrics["first_frame_image_count"] = first_frame_count
    metrics.update(texture_metrics)
    tower_up_alignment = estimate_tower_up_from_pose_yaml(getattr(args, "tower_pose_yaml", None))
    outer_column_up_alignment = estimate_outer_column_gravity_alignment(cameras)
    viewer_up_alignment = outer_column_up_alignment or tower_up_alignment
    board_orientation_alignment = estimate_board_orientation_alignment(args, viewer_up_alignment)
    metrics["calibration_quality"] = calibration_quality
    metrics["intrinsic_sanity"] = calibration_quality.get("intrinsic_sanity", {})
    metrics["viewer_default_up_alignment"] = viewer_up_alignment
    metrics["outer_column_gravity_alignment"] = outer_column_up_alignment
    metrics["tower_up_alignment"] = tower_up_alignment
    metrics["canonical_coordinate_frame"] = canonical_frame
    if board_orientation_alignment.get("aggregate"):
        metrics["board_normal_p90_angle_from_horizontal_deg"] = (
            board_orientation_alignment["aggregate"].get("p90_angle_from_horizontal_deg")
        )
    if outer_final_poses is None:
        coordinate_note = (
            "Scene coordinates map rig/OpenCV coordinates as x -> x, y -> -y, z -> -z. "
            "Non-topdown outer cameras are approximate COLMAP Sim(3)-aligned poses and are not final calibrated outer rig geometry."
        )
    elif aligned_outer_bridge_pose_ready:
        coordinate_note = (
            "Scene coordinates map rig/OpenCV coordinates as x -> x, y -> -y, z -> -z. "
            "Outer ring cameras come from the outer tower final camera_tr_rig YAML after a large-marker "
            "bridge SE3 alignment into the refined inner rig frame."
        )
    else:
        coordinate_note = (
            "Scene coordinates map rig/OpenCV coordinates as x -> x, y -> -y, z -> -z. "
            "Outer ring cameras come from the outer tower final camera_tr_rig YAML; "
            "top-down bridge anchors use bridge metric poses when present."
        )
    return {
        "title": args.title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "inner_bridge_pose_yaml": str(Path(args.inner_bridge_pose_yaml).resolve()),
            "bridge_summary_json": str(Path(args.bridge_summary_json).resolve()) if args.bridge_summary_json else "",
            "outer_colmap_images_txt": str(Path(args.outer_colmap_images_txt).resolve()),
            "outer_colmap_summary_json": str(Path(args.outer_colmap_summary_json).resolve()) if args.outer_colmap_summary_json else "",
            "outer_final_pose_yaml": str(Path(outer_final_pose_yaml).resolve()) if outer_final_pose_yaml else "",
            "tower_pose_yaml": str(Path(args.tower_pose_yaml).resolve()) if getattr(args, "tower_pose_yaml", None) else "",
            "combined_image_directories_file": str(Path(args.combined_image_directories_file).resolve()) if args.combined_image_directories_file else "",
        },
        "coordinate_note": coordinate_note,
        "frustum": {
            "default_near": args.default_near,
            "default_far": args.default_far,
            "half_width_over_depth": args.frustum_half_width_over_depth,
            "half_height_over_depth": args.frustum_half_height_over_depth,
            "fill_opacity": args.frustum_fill_opacity,
        },
        "topdown_labels": topdown_labels,
        "inner_bridge_indices": inner_indices,
        "metrics": metrics,
        "dataset_coverage": dataset_coverage,
        "viewer_options": {
            "enable_overlap": False,
            "single_canonical_viewer": True,
            "correspondence_data_url": getattr(args, "correspondence_data_url", ""),
            "canonical_coordinate_frame": canonical_frame,
            "default_reference_up_vector_three": (
                viewer_up_alignment["display_up_vector"] if viewer_up_alignment else None
            ),
            "up_alignment": viewer_up_alignment,
            "tower_up_alignment": tower_up_alignment,
            "outer_column_gravity_alignment": outer_column_up_alignment,
            "board_orientation_alignment": board_orientation_alignment,
            "default_visibility": {
                "inner": args.viewer_scope == "combined",
                "outer": True,
                "outer_topdown": True,
                "outer_colmap": args.viewer_scope == "outer",
                "outer_final": True,
            },
        },
        "sparse_point_cloud": {
            "source": "",
            "coordinate_frame": "combined_studio_rig",
            "point_count": 0,
            "positions": [],
            "colors": [],
        },
        "reprojection_reports": [],
        "cameras": cameras,
        "bounds": bounds,
    }


def compute_bounds(points):
    if not points:
        return {"center": [0.0, 0.0, 0.0], "radius": 2.0}
    arr = np.asarray(points, dtype=np.float64)
    center = arr.mean(axis=0)
    radius = float(np.max(np.linalg.norm(arr - center, axis=1)))
    return {
        "center": [float(v) for v in center],
        "radius": max(1.2, radius * 1.35),
    }


def json_dumps(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def replace_required(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"Could not patch combined viewer HTML section: {label}")
    return text.replace(old, new, 1)


def patch_combined_viewer_html(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    text = replace_required(
        text,
        "      <div class=\"metric\"><strong id=\"metric-board-normal\">-</strong><span>board normal p90 horizontal err</span></div>\n"
        "      <div class=\"metric\"><strong id=\"metric-overlap\">off</strong><span>strict all-camera overlap</span></div>",
        "      <div class=\"metric\"><strong id=\"metric-board-normal\">-</strong><span>board normal p90 horizontal err</span></div>\n"
        "      <div class=\"metric\"><strong id=\"metric-overlap\">off</strong><span>strict all-camera overlap</span></div>\n"
        "      <div class=\"metric\"><strong id=\"metric-intrinsics-sanity\">-</strong><span>intrinsic sanity failed / warning</span></div>",
        "intrinsic sanity summary metric",
    )
    text = replace_required(
        text,
        "          <tr><th>Cam</th><th>Set</th><th>Coverage</th><th>Raw obs</th><th>Solve obs</th><th>Median px</th><th>P90 px</th><th>Max px</th><th>fx</th><th>fy</th><th>Decision</th></tr>",
        "          <tr><th>Cam</th><th>Set</th><th>Coverage</th><th>Raw obs</th><th>Solve obs</th><th>Median px</th><th>P90 px</th><th>Max px</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>Intrinsics</th><th>Decision</th></tr>",
        "camera table intrinsic columns",
    )
    text = replace_required(
        text,
        "      min-width: 760px;",
        "      min-width: 1140px;",
        "camera table minimum width",
    )
    text = replace_required(
        text,
        "    tr.coverage-inactive {\n"
        "      color: var(--muted);\n"
        "      background: #f4f4f1;\n"
        "    }",
        "    tr.coverage-inactive {\n"
        "      color: var(--muted);\n"
        "      background: #f4f4f1;\n"
        "    }\n"
        "    tr.warning { background: #fff8df; }\n"
        "    tr.danger { background: #fff1f0; }\n"
        "    tr.warning:hover { background: #fff2bd; }\n"
        "    tr.danger:hover { background: #ffe2de; }\n"
        "    .intrinsics-status {\n"
        "      display: inline-block;\n"
        "      min-width: 62px;\n"
        "      padding: 2px 6px;\n"
        "      border: 1px solid var(--line);\n"
        "      border-radius: 999px;\n"
        "      text-align: center;\n"
        "      font-weight: 650;\n"
        "    }\n"
        "    .intrinsics-status.ok { background: #eef7ee; color: #1f6b33; border-color: #b6d7bd; }\n"
        "    .intrinsics-status.warning { background: #fff1bf; color: #704c00; border-color: #dfb948; }\n"
        "    .intrinsics-status.danger { background: #ffd8d2; color: #8c1d18; border-color: #e1968c; }\n"
        "    .intrinsics-status.missing { background: #eeeeee; color: var(--muted); }",
        "intrinsic status css",
    )
    text = replace_required(
        text,
        "    fx: null,\n"
        "    fy: null,\n"
        "  };",
        "    fx: null,\n"
        "    fy: null,\n"
        "    cx: null,\n"
        "    cy: null,\n"
        "    intrinsics_status: \"missing\",\n"
        "    intrinsics_flags: [],\n"
        "  };",
        "calibration quality fallback",
    )
    text = replace_required(
        text,
        """function buildTable() {
  const body = document.getElementById("camera-table");
  body.innerHTML = "";
  tableRows.clear();
  RIG_DATA.cameras.forEach((cam) => {
    const row = document.createElement("tr");
    row.dataset.index = cam.index;
    const c = coverageForCamera(cam);
    const q = calibrationQuality(cam);
    const residual = displayResidualQuality(cam);
    row.innerHTML = "<td>" + cameraLabel(cam) + "</td>"
      + "<td>" + cameraCategory(cam).replace("outer_final", "outer") + "</td>"
      + "<td>" + (c.active === false ? "inactive" : "active") + "</td>"
      + "<td>" + fmt(c.observation_count, 0) + "</td>"
      + "<td>" + fmt(residual.observation_count ?? residual.residual_count, 0) + "</td>"
      + "<td>" + fmt(residual.median_error_px, 3) + "</td>"
      + "<td>" + fmt(residual.p90_error_px, 3) + "</td>"
      + "<td>" + fmt(residual.max_error_px, 3) + "</td>"
      + "<td>" + fmt(q.fx, 1) + "</td>"
      + "<td>" + fmt(q.fy, 1) + "</td>"
      + "<td>" + (residual.decision || residual.source || q.decision || q.source || "-") + "</td>";
    row.addEventListener("click", () => selectCamera(cam.index, true));
    body.appendChild(row);
    tableRows.set(cam.index, row);
  });
}
""",
        """function intrinsicStatusClass(status) {
  if (status === "failed") return "danger";
  if (status === "warning") return "warning";
  if (status === "ok") return "ok";
  return "missing";
}

function buildTable() {
  const body = document.getElementById("camera-table");
  body.innerHTML = "";
  tableRows.clear();
  RIG_DATA.cameras.forEach((cam) => {
    const row = document.createElement("tr");
    row.dataset.index = cam.index;
    const c = coverageForCamera(cam);
    const q = calibrationQuality(cam);
    const residual = displayResidualQuality(cam);
    const status = q.intrinsics_status || "missing";
    const statusClass = intrinsicStatusClass(status);
    const flags = Array.isArray(q.intrinsics_flags) ? q.intrinsics_flags : [];
    const flagText = flags.length ? flags.join(", ") : "ok";
    row.classList.toggle("danger", status === "failed");
    row.classList.toggle("warning", status === "warning");
    row.innerHTML = "<td>" + cameraLabel(cam) + "</td>"
      + "<td>" + cameraCategory(cam).replace("outer_final", "outer") + "</td>"
      + "<td>" + (c.active === false ? "inactive" : "active") + "</td>"
      + "<td>" + fmt(c.observation_count, 0) + "</td>"
      + "<td>" + fmt(residual.observation_count ?? residual.residual_count, 0) + "</td>"
      + "<td>" + fmt(residual.median_error_px, 3) + "</td>"
      + "<td>" + fmt(residual.p90_error_px, 3) + "</td>"
      + "<td>" + fmt(residual.max_error_px, 3) + "</td>"
      + "<td>" + fmt(q.fx, 1) + "</td>"
      + "<td>" + fmt(q.fy, 1) + "</td>"
      + "<td>" + fmt(q.cx, 1) + "</td>"
      + "<td>" + fmt(q.cy, 1) + "</td>"
      + "<td><span class=\\"intrinsics-status " + statusClass + "\\" title=\\"" + flagText + "\\">" + status + "</span></td>"
      + "<td>" + (residual.decision || residual.source || q.decision || q.source || "-") + "</td>";
    row.addEventListener("click", () => selectCamera(cam.index, true));
    body.appendChild(row);
    tableRows.set(cam.index, row);
  });
}
""",
        "camera table renderer",
    )
    text = replace_required(
        text,
        "  document.getElementById(\"metric-board-normal\").textContent =\n"
        "    fmt(boardAggregate.p90_angle_from_horizontal_deg, 2) + \" deg\";\n"
        "}",
        "  document.getElementById(\"metric-board-normal\").textContent =\n"
        "    fmt(boardAggregate.p90_angle_from_horizontal_deg, 2) + \" deg\";\n"
        "  const sanity = ((RIG_DATA.metrics || {}).intrinsic_sanity || {});\n"
        "  const sanityEl = document.getElementById(\"metric-intrinsics-sanity\");\n"
        "  if (sanityEl) {\n"
        "    sanityEl.textContent = (sanity.failed_camera_count ?? 0) + \" failed / \"\n"
        "      + (sanity.warning_camera_count ?? 0) + \" warning\";\n"
        "  }\n"
        "}",
        "summary intrinsic sanity metric",
    )
    text = replace_required(
        text,
        "    + \"intrinsics: fx \" + fmt(q.fx, 1) + \", fy \" + fmt(q.fy, 1)\n"
        "    + \"; decision: \" + (residual.decision || residual.source || q.decision || q.source || \"-\") + \"<br>\"",
        "    + \"intrinsics: fx \" + fmt(q.fx, 1) + \", fy \" + fmt(q.fy, 1)\n"
        "    + \", cx \" + fmt(q.cx, 1) + \", cy \" + fmt(q.cy, 1)\n"
        "    + \"; sanity: \" + (q.intrinsics_status || \"missing\")\n"
        "    + \"; decision: \" + (residual.decision || residual.source || q.decision || q.source || \"-\") + \"<br>\"",
        "selected camera intrinsic detail",
    )
    path.write_text(text, encoding="utf-8")


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Combined Studio Rig Viewer</title>
  <style>
    :root {
      --bg: #101214;
      --panel: #f7f7f4;
      --panel-2: #ececea;
      --ink: #1f2428;
      --muted: #687076;
      --line: #d5d5cf;
      --inner: #39d0b2;
      --topdown: #ffb84d;
      --outer: #65a9ff;
      --danger: #c14632;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; overflow: hidden; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    #viewer {
      position: fixed;
      inset: 0;
    }
    .panel {
      position: fixed;
      z-index: 10;
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      border: 1px solid color-mix(in srgb, var(--line) 90%, transparent);
      border-radius: 8px;
      box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(8px);
    }
    .info {
      left: 18px;
      top: 18px;
      width: min(520px, calc(100vw - 36px));
      max-height: calc(100vh - 36px);
      overflow: auto;
      padding: 16px;
    }
    .controls {
      right: 18px;
      top: 18px;
      width: min(320px, calc(100vw - 36px));
      padding: 14px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .warning {
      margin: 10px 0 12px;
      padding: 9px 10px;
      border-left: 3px solid var(--danger);
      background: #fff4ef;
      color: #5d291e;
      font-size: 12px;
      line-height: 1.4;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .metric {
      min-width: 0;
      padding: 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
    }
    .metric strong {
      display: block;
      font-size: 17px;
      line-height: 1.1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin: 8px 0 10px;
      font-size: 12px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #343a40;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 2px;
      display: inline-block;
    }
    .table-wrap {
      max-height: 170px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      padding: 6px 8px;
      border-bottom: 1px solid #eeeeea;
      text-align: left;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: #f1f1ee;
      color: #42484d;
      font-weight: 650;
    }
    .control-section {
      padding: 10px 0;
      border-top: 1px solid var(--line);
    }
    .control-section:first-child {
      padding-top: 0;
      border-top: 0;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 28px;
      font-size: 13px;
      color: #2f353a;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      accent-color: #225ea8;
    }
    .range-row {
      display: grid;
      grid-template-columns: 44px 1fr 52px;
      gap: 8px;
      align-items: center;
      min-height: 32px;
      font-size: 13px;
    }
    input[type="range"] {
      width: 100%;
      accent-color: #225ea8;
    }
    button {
      width: 100%;
      min-height: 32px;
      border: 1px solid #b9c0c7;
      border-radius: 6px;
      background: #ffffff;
      color: #202428;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: #eef4ff; }
    .small {
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      word-break: break-word;
    }
    @media (max-width: 860px) {
      .info {
        width: calc(100vw - 24px);
        left: 12px;
        top: 12px;
        max-height: 46vh;
      }
      .controls {
        left: 12px;
        right: auto;
        top: auto;
        bottom: 12px;
        width: calc(100vw - 24px);
      }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <div id="viewer"></div>
  <section class="panel info">
    <h1 id="title"></h1>
    <p class="subtitle" id="subtitle"></p>
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background: var(--inner)"></span>inner bridge metric</span>
      <span class="legend-item"><span class="swatch" style="background: var(--topdown)"></span>outer top-down bridge metric</span>
      <span class="legend-item"><span class="swatch" style="background: var(--outer)"></span>outer COLMAP-aligned approximate</span>
    </div>
    <div class="warning">Only inner0..inner7 and 4-1/4-2/4-3 use bridge metric poses. The remaining outer ring cameras are approximate COLMAP poses transformed by a three-anchor Sim(3), suitable for layout smoke checks rather than final geometry.</div>
    <div class="metrics" id="metrics"></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Camera</th><th>Source</th><th>2D pts</th><th>Tracks</th></tr>
        </thead>
        <tbody id="camera-table"></tbody>
      </table>
    </div>
  </section>
  <section class="panel controls">
    <div class="control-section">
      <label class="check"><input id="show-inner" type="checkbox" checked> inner0..inner7</label>
      <label class="check"><input id="show-topdown" type="checkbox" checked> 4-1/4-2/4-3 bridge top-down</label>
      <label class="check"><input id="show-colmap" type="checkbox" checked> outer COLMAP-aligned</label>
      <label class="check"><input id="show-labels" type="checkbox" checked> camera labels</label>
    </div>
    <div class="control-section">
      <div class="range-row"><span>Near</span><input id="near-slider" type="range" min="0.05" max="2.0" step="0.05"><strong id="near-value"></strong></div>
      <div class="range-row"><span>Far</span><input id="far-slider" type="range" min="0.10" max="3.5" step="0.05"><strong id="far-value"></strong></div>
    </div>
    <div class="control-section">
      <button id="reset-camera">Reset View</button>
      <p class="small" id="input-note"></p>
    </div>
  </section>
  <script id="rig-data" type="application/json">__RIG_DATA__</script>
  <script type="module">
    import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js";
    import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/controls/OrbitControls.js";

    const RIG_DATA = JSON.parse(document.getElementById("rig-data").textContent);
    const COLORS = {
      inner: 0x39d0b2,
      outer_topdown: 0xffb84d,
      outer_colmap: 0x65a9ff,
    };
    const KIND_LABEL = {
      inner: "inner",
      outer_topdown: "bridge topdown",
      outer_colmap: "COLMAP approx",
    };

    const root = document.getElementById("viewer");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);

    const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.01, 1000);
    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    root.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const cameraGroup = new THREE.Group();
    const labelGroup = new THREE.Group();
    scene.add(cameraGroup, labelGroup);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x30343a, 1.4));
    const grid = new THREE.GridHelper(8, 16, 0x4a5058, 0x2a2e33);
    grid.position.y = 0;
    scene.add(grid);
    const axes = new THREE.AxesHelper(1.0);
    scene.add(axes);

    let nearDistance = RIG_DATA.frustum.default_near;
    let farDistance = RIG_DATA.frustum.default_far;
    const objects = [];

    function vec(v) {
      return new THREE.Vector3(v[0], v[1], v[2]);
    }

    function calcFrustum(cam) {
      const center = vec(cam.center);
      const x = vec(cam.basis.x);
      const y = vec(cam.basis.y);
      const z = vec(cam.basis.z);
      const sx = RIG_DATA.frustum.half_width_over_depth;
      const sy = RIG_DATA.frustum.half_height_over_depth;
      const plane = (distance) => [
        center.clone().addScaledVector(z, distance).addScaledVector(x, distance * sx).addScaledVector(y, distance * sy),
        center.clone().addScaledVector(z, distance).addScaledVector(x, -distance * sx).addScaledVector(y, distance * sy),
        center.clone().addScaledVector(z, distance).addScaledVector(x, -distance * sx).addScaledVector(y, -distance * sy),
        center.clone().addScaledVector(z, distance).addScaledVector(x, distance * sx).addScaledVector(y, -distance * sy),
      ];
      const near = plane(nearDistance);
      const far = plane(farDistance);
      const lines = [
        center, near[0], center, near[1], center, near[2], center, near[3],
        near[0], near[1], near[1], near[2], near[2], near[3], near[3], near[0],
        near[0], far[0], near[1], far[1], near[2], far[2], near[3], far[3],
        far[0], far[1], far[1], far[2], far[2], far[3], far[3], far[0],
      ];
      return {center, near, far, lines};
    }

    function lineGeometry(points) {
      return new THREE.BufferGeometry().setFromPoints(points);
    }

    function fillGeometry(frustum) {
      const vertices = frustum.near.concat(frustum.far);
      const positions = [];
      const faces = [
        [0, 1, 2], [0, 2, 3],
        [4, 7, 6], [4, 6, 5],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
      ];
      for (const face of faces) {
        for (const index of face) {
          const p = vertices[index];
          positions.push(p.x, p.y, p.z);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geometry.computeVertexNormals();
      return geometry;
    }

    function makeLabel(text, color) {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const pixelRatio = 2;
      canvas.width = 192 * pixelRatio;
      canvas.height = 54 * pixelRatio;
      ctx.scale(pixelRatio, pixelRatio);
      ctx.font = "600 22px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(16,18,20,0.78)";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      roundRect(ctx, 3, 7, 186, 40, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, 14, 28);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      const material = new THREE.SpriteMaterial({map: texture, transparent: true, depthTest: false});
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(0.34, 0.096, 1);
      return sprite;
    }

    function roundRect(ctx, x, y, width, height, radius) {
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + width - radius, y);
      ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
      ctx.lineTo(x + width, y + height - radius);
      ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
      ctx.lineTo(x + radius, y + height);
      ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
    }

    function makeCameraObject(cam) {
      const color = COLORS[cam.kind] || 0xffffff;
      const frustum = calcFrustum(cam);
      const group = new THREE.Group();
      group.userData.kind = cam.kind;
      group.userData.label = cam.label;

      const line = new THREE.LineSegments(
        lineGeometry(frustum.lines),
        new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.96})
      );
      const fill = new THREE.Mesh(
        fillGeometry(frustum),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: RIG_DATA.frustum.fill_opacity,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
      );
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.03, 12, 8),
        new THREE.MeshStandardMaterial({color, roughness: 0.55, metalness: 0.05})
      );
      sphere.position.copy(vec(cam.center));
      const zRay = new THREE.LineSegments(
        lineGeometry([vec(cam.center), vec(cam.center).addScaledVector(vec(cam.basis.z), 0.35)]),
        new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.38})
      );
      group.add(fill, line, sphere, zRay);
      cameraGroup.add(group);

      const labelColor = "#" + new THREE.Color(color).getHexString();
      const label = makeLabel(cam.label, labelColor);
      label.position.copy(vec(cam.center)).add(new THREE.Vector3(0, 0.12, 0));
      label.userData.kind = cam.kind;
      labelGroup.add(label);
      objects.push({cam, group, line, fill, label});
    }

    function refreshFrustums() {
      for (const item of objects) {
        const frustum = calcFrustum(item.cam);
        item.line.geometry.dispose();
        item.line.geometry = lineGeometry(frustum.lines);
        item.fill.geometry.dispose();
        item.fill.geometry = fillGeometry(frustum);
      }
    }

    function formatNumber(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    }

    function renderMetrics() {
      const m = RIG_DATA.metrics;
      const votes = Object.entries(m.bridge_topdown_votes || {})
        .map(([k, v]) => `${k}:${v}`)
        .join(" ");
      const tracks = m.outer_colmap_track_summary || {};
      const rows = [
        [m.display_camera_count, "displayed cameras"],
        [m.outer_colmap_registered_count, "outer COLMAP registered"],
        [formatNumber(m.sim3_scale, 4), "Sim(3) scale"],
        [formatNumber(m.sim3_residual_rms_m, 4) + " m", "Sim(3) anchor RMS"],
        [votes || "-", "bridge topdown votes"],
        [`${tracks.triangulated_min}/${formatNumber(tracks.triangulated_median, 1)}/${tracks.triangulated_max}`, "COLMAP tracks min/med/max"],
        [m.outer_colmap_points3d_count ?? "-", "COLMAP points3D"],
        [formatNumber(m.outer_colmap_mean_error_px, 3) + " px", "COLMAP mean reproj"],
      ];
      document.getElementById("metrics").innerHTML = rows.map(([value, label]) =>
        `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`
      ).join("");
    }

    function renderCameraTable() {
      document.getElementById("camera-table").innerHTML = RIG_DATA.cameras.map((cam) => {
        const point2d = cam.metrics.point2d_count ?? "";
        const tracks = cam.metrics.triangulated_point_count ?? "";
        return `<tr><td>${cam.label}</td><td>${KIND_LABEL[cam.kind]}</td><td>${point2d}</td><td>${tracks}</td></tr>`;
      }).join("");
    }

    function applyVisibility() {
      const visible = {
        inner: document.getElementById("show-inner").checked,
        outer_topdown: document.getElementById("show-topdown").checked,
        outer_colmap: document.getElementById("show-colmap").checked,
      };
      for (const item of objects) {
        item.group.visible = visible[item.cam.kind];
        item.label.visible = visible[item.cam.kind] && document.getElementById("show-labels").checked;
      }
    }

    function updateSliderLabels() {
      document.getElementById("near-value").textContent = nearDistance.toFixed(2) + " m";
      document.getElementById("far-value").textContent = farDistance.toFixed(2) + " m";
    }

    function resetView() {
      const center = vec(RIG_DATA.bounds.center);
      const radius = RIG_DATA.bounds.radius;
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(radius * 1.1, radius * 0.78, radius * 1.25));
      camera.near = Math.max(0.01, radius / 500);
      camera.far = Math.max(50, radius * 20);
      camera.updateProjectionMatrix();
      controls.update();
    }

    function initUi() {
      document.getElementById("title").textContent = RIG_DATA.title;
      document.getElementById("subtitle").textContent = `${RIG_DATA.generated_at} | ${RIG_DATA.coordinate_note}`;
      document.getElementById("input-note").textContent = RIG_DATA.inputs.outer_colmap_images_txt;
      const nearSlider = document.getElementById("near-slider");
      const farSlider = document.getElementById("far-slider");
      nearSlider.value = nearDistance;
      farSlider.value = farDistance;
      nearSlider.addEventListener("input", () => {
        nearDistance = Math.min(Number(nearSlider.value), farDistance - 0.05);
        nearSlider.value = nearDistance;
        updateSliderLabels();
        refreshFrustums();
      });
      farSlider.addEventListener("input", () => {
        farDistance = Math.max(Number(farSlider.value), nearDistance + 0.05);
        farSlider.value = farDistance;
        updateSliderLabels();
        refreshFrustums();
      });
      for (const id of ["show-inner", "show-topdown", "show-colmap", "show-labels"]) {
        document.getElementById(id).addEventListener("change", applyVisibility);
      }
      document.getElementById("reset-camera").addEventListener("click", resetView);
      updateSliderLabels();
      renderMetrics();
      renderCameraTable();
    }

    function onResize() {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }

    for (const cam of RIG_DATA.cameras) {
      makeCameraObject(cam);
    }
    initUi();
    resetView();
    applyVisibility();
    window.addEventListener("resize", onResize);

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    window.__combinedRigViewer = {
      data: RIG_DATA,
      camera,
      scene,
      controls,
      getCameraCount: () => RIG_DATA.cameras.length,
      getKindCounts: () => RIG_DATA.cameras.reduce((acc, cam) => {
        acc[cam.kind] = (acc[cam.kind] || 0) + 1;
        return acc;
      }, {}),
      getFrustumState: () => ({near: nearDistance, far: farDistance, objectCount: objects.length}),
    };
  </script>
</body>
</html>
"""


def write_html(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_rig_viewer_html(path, data)
    patch_combined_viewer_html(path)
    (path.parent / "rig_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner_bridge_pose_yaml", type=Path, default=DEFAULT_BRIDGE_POSE_YAML)
    parser.add_argument("--bridge_summary_json", type=Path, default=DEFAULT_BRIDGE_SUMMARY_JSON)
    parser.add_argument("--outer_colmap_images_txt", type=Path, default=DEFAULT_OUTER_IMAGES_TXT)
    parser.add_argument("--outer_colmap_summary_json", type=Path, default=DEFAULT_OUTER_SUMMARY_JSON)
    parser.add_argument("--outer_final_pose_yaml", type=Path, default=DEFAULT_OUTER_FINAL_POSE_YAML)
    parser.add_argument("--tower_pose_yaml", type=Path, default=DEFAULT_TOWER_POSE_YAML)
    parser.add_argument("--whole_coverage_tsv", type=Path, default=DEFAULT_WHOLE_COVERAGE_TSV)
    parser.add_argument("--large_marker_pnp_summary_tsv", type=Path, default=DEFAULT_LARGE_MARKER_PNP_SUMMARY_TSV)
    parser.add_argument("--small_marker_pnp_summary_tsv", type=Path, default=DEFAULT_SMALL_MARKER_PNP_SUMMARY_TSV)
    parser.add_argument("--inner_reprojection_metrics_tsv", type=Path, default=DEFAULT_INNER_REPROJECTION_METRICS_TSV)
    parser.add_argument("--inner_intrinsics_dir", type=Path, default=DEFAULT_INNER_INTRINSICS_DIR)
    parser.add_argument(
        "--inner_intrinsics_index_offset",
        type=int,
        default=-1,
        help=(
            "Index offset for compact inner0..7 intrinsics. Use -1 to auto-detect; "
            "all32 fixed-intrinsics directories use offset 24."
        ),
    )
    parser.add_argument("--outer_reprojection_tsv", type=Path, default=DEFAULT_OUTER_REPROJECTION_TSV)
    parser.add_argument("--outer_intrinsics_tsv", type=Path, default=None)
    parser.add_argument("--outer_intrinsics_dir", type=Path, default=DEFAULT_OUTER_INTRINSICS_DIR)
    parser.add_argument("--large_marker_board_pose_yaml", type=Path, default=DEFAULT_LARGE_MARKER_BOARD_POSE_YAML)
    parser.add_argument("--small_marker_board_pose_yaml", type=Path, default=DEFAULT_SMALL_MARKER_BOARD_POSE_YAML)
    parser.add_argument("--bridge_marker_board_pose_yaml", type=Path, default=DEFAULT_BRIDGE_MARKER_BOARD_POSE_YAML)
    parser.add_argument("--output_html", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--viewer_scope", choices=("combined", "outer"), default="combined")
    parser.add_argument("--viewer_assets_dir", type=Path, default=DEFAULT_VIEWER_ASSETS_DIR)
    parser.add_argument("--combined_image_directories_file", type=Path, default=DEFAULT_COMBINED_IMAGE_DIRS)
    parser.add_argument("--inner_image_directories_file", type=Path, default="")
    parser.add_argument("--outer_image_directories_file", type=Path, default="")
    parser.add_argument("--inner_bridge_indices", default=DEFAULT_INNER_BRIDGE_INDICES)
    parser.add_argument("--topdown_bridge_indices", default=DEFAULT_TOPDOWN_BRIDGE_INDICES)
    parser.add_argument("--topdown_labels", default="4-1,4-2,4-3")
    parser.add_argument("--default_near", type=float, default=0.3)
    parser.add_argument("--default_far", type=float, default=0.7)
    parser.add_argument("--frustum_half_width_over_depth", type=float, default=0.45)
    parser.add_argument("--frustum_half_height_over_depth", type=float, default=0.32)
    parser.add_argument("--frustum_fill_opacity", type=float, default=0.11)
    parser.add_argument("--texture_max_width", type=int, default=640)
    parser.add_argument("--texture_jpeg_quality", type=int, default=82)
    parser.add_argument("--correspondence_data_url", default="")
    parser.add_argument("--title", default="Combined Studio Rig Viewer: inner8 + outer24")
    args = parser.parse_args()

    data = build_viewer_data(args)
    copy_viewer_assets(Path(args.output_html).resolve().parent, args.viewer_assets_dir)
    write_html(args.output_html, data)
    kind_counts = {}
    for camera in data["cameras"]:
        kind_counts[camera["kind"]] = kind_counts.get(camera["kind"], 0) + 1
    print(f"Wrote {args.output_html}")
    print(json.dumps({
        "camera_count": len(data["cameras"]),
        "kind_counts": kind_counts,
        "outer_colmap_registered_count": data["metrics"]["outer_colmap_registered_count"],
        "sim3_scale": data["metrics"]["sim3_scale"],
        "sim3_residual_rms_m": data["metrics"]["sim3_residual_rms_m"],
        "sim3_residual_max_m": data["metrics"]["sim3_residual_max_m"],
        "output_html": str(args.output_html),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
