#!/usr/bin/env python3
"""Export the current 24+8 studio rig relative extrinsics as calibration files."""

import argparse
import ast
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from generate_combined_studio_rig_viewer import (
        DEFAULT_INNER_BRIDGE_INDICES,
        OUTER_CAMERA_LABELS,
        load_pose_yaml,
    )
    from studio_canonical_frame import (
        estimate_frame_from_camera_poses,
        transform_pose_to_aligned,
    )
except ModuleNotFoundError:
    from scripts.calib.generate_combined_studio_rig_viewer import (
        DEFAULT_INNER_BRIDGE_INDICES,
        OUTER_CAMERA_LABELS,
        load_pose_yaml,
    )
    from scripts.calib.studio_canonical_frame import (
        estimate_frame_from_camera_poses,
        transform_pose_to_aligned,
    )


DEFAULT_T0_ROOT = Path("/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3")
DEFAULT_FAST_ROOT = DEFAULT_T0_ROOT / "recalib_pipelines/fast_inner_bridge/latest"
DEFAULT_OUTER_ROOT = DEFAULT_T0_ROOT / "recalib_pipelines/outer_tower/latest"
DEFAULT_INNER_BRIDGE_POSE_YAML = (
    DEFAULT_FAST_ROOT
    / "bridge_colmap_inner_refined_v1/camera_tr_inner_refined_plus_outer_topdown.yaml"
)
DEFAULT_OUTER_FINAL_POSE_YAML = (
    DEFAULT_OUTER_ROOT
    / "tag_refine_robust/camera_tr_rig_delta_refined_accepted.yaml"
)
DEFAULT_OUTPUT_DIR = DEFAULT_FAST_ROOT / "calibration_artifacts/studio_32_extrinsics_current"
DEFAULT_INTRINSICS_DIR = DEFAULT_FAST_ROOT / "planned_inputs/bridge_all32_fixed_intrinsics"
DEFAULT_VIEWER_URL = (
    "http://192.168.2.0:9899/"
    "calib_2026_05_26_jpg_v3/recalib_pipelines/fast_inner_bridge/latest/"
    "combined_studio_rig_viewer_v1/index.html"
)


def matrix_to_quat_xyzw(rotation):
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q *= -1
    return q


def read_inner_camera_ids(manifest_path):
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.is_file():
        return {}
    rows = path.read_text(encoding="utf-8-sig").splitlines()
    if not rows:
        return {}
    header = rows[0].split("\t")
    try:
        camera_id_col = header.index("camera_id")
    except ValueError:
        return {}
    index_col = header.index("camera_index") if "camera_index" in header else None
    label_col = header.index("label") if "label" in header else None
    output_index_col = header.index("index") if "index" in header else None
    if index_col is None and label_col is None and output_index_col is None:
        return {}
    result = {}
    for row in rows[1:]:
        cols = row.split("\t")
        required_cols = [camera_id_col]
        if index_col is not None:
            required_cols.append(index_col)
        if label_col is not None:
            required_cols.append(label_col)
        if output_index_col is not None:
            required_cols.append(output_index_col)
        if len(cols) <= max(required_cols):
            continue
        inner_index = None
        try:
            if index_col is not None:
                inner_index = int(cols[index_col])
            elif label_col is not None and cols[label_col].startswith("inner"):
                inner_index = int(cols[label_col].removeprefix("inner"))
            elif output_index_col is not None:
                output_index = int(cols[output_index_col])
                if output_index >= 24:
                    inner_index = output_index - 24
        except ValueError:
            inner_index = None
        if inner_index is not None:
            result[inner_index] = cols[camera_id_col]
    return result


def validate_pose(index, pose):
    if pose is None:
        raise ValueError(f"Missing pose at output index {index}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"Non-finite pose at output index {index}")
    det = float(np.linalg.det(pose[:3, :3]))
    if abs(det - 1.0) > 1e-3:
        raise ValueError(f"Rotation determinant at output index {index} is {det}")


def format_float(value):
    return f"{float(value):.14g}"


def yaml_quote(value):
    text = str(value)
    if not text:
        return '""'
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def append_coordinate_transform_yaml(lines, coordinate_frame, indent=""):
    if not coordinate_frame:
        return
    lines.extend([
        f"{indent}# coordinate_transform is only for provenance/debugging.",
        f"{indent}# Final consumers should normally use camera_tr_studio_rig directly.",
        f"{indent}# p_aligned = R_aligned_from_source @ (p_source - origin_source).",
        f"{indent}# axes_source rows are the final +X/+Y/+Z axes expressed in source rig coordinates.",
        f"{indent}coordinate_transform:",
        f"{indent}  method: {yaml_quote(coordinate_frame['method'])}",
        f"{indent}  source_coordinate_frame: {yaml_quote(coordinate_frame['source_coordinate_frame'])}",
        f"{indent}  aligned_coordinate_frame: {yaml_quote(coordinate_frame['aligned_coordinate_frame'])}",
        f"{indent}  point_transform: {yaml_quote(coordinate_frame['point_transform'])}",
        f"{indent}  origin_source: [{', '.join(format_float(v) for v in coordinate_frame['origin_source'])}]",
        f"{indent}  aligned_from_source_rotation:",
    ])
    for row in coordinate_frame["aligned_from_source_rotation"]:
        lines.append(f"{indent}    - [{', '.join(format_float(v) for v in row)}]")
    lines.append(f"{indent}  source_from_aligned_rotation:")
    for row in coordinate_frame["source_from_aligned_rotation"]:
        lines.append(f"{indent}    - [{', '.join(format_float(v) for v in row)}]")
    lines.extend([
        f"{indent}  axes_source:",
        f"{indent}    x: [{', '.join(format_float(v) for v in coordinate_frame['axes_source']['x'])}]",
        f"{indent}    y: [{', '.join(format_float(v) for v in coordinate_frame['axes_source']['y'])}]",
        f"{indent}    z: [{', '.join(format_float(v) for v in coordinate_frame['axes_source']['z'])}]",
        f"{indent}  axis_meaning:",
        f"{indent}    x: {yaml_quote(coordinate_frame['axis_meaning']['x'])}",
        f"{indent}    y: {yaml_quote(coordinate_frame['axis_meaning']['y'])}",
        f"{indent}    z: {yaml_quote(coordinate_frame['axis_meaning']['z'])}",
        f"{indent}  positive_z_forward_direction_source: "
        f"[{', '.join(format_float(v) for v in coordinate_frame['positive_z_forward_direction_source'])}]",
        f"{indent}  negative_z_gap_direction_source: "
        f"[{', '.join(format_float(v) for v in coordinate_frame['negative_z_gap_direction_source'])}]",
        f"{indent}  negative_z_gap_labels: "
        f"[{', '.join(yaml_quote(v) for v in coordinate_frame['negative_z_gap_labels'])}]",
        f"{indent}  origin_level2_labels: "
        f"[{', '.join(yaml_quote(v) for v in coordinate_frame['origin_level2_labels'])}]",
        f"{indent}  used_columns: [{', '.join(yaml_quote(v) for v in coordinate_frame['used_columns'])}]",
        f"{indent}  level_plane_count: {coordinate_frame['level_plane_count']}",
    ])


STUDIO_RIG_Y_DOWN_Z_FORWARD_COMMENTS = [
    "# Coordinate convention, self-contained for downstream consumers:",
    "# - This YAML is the source of truth for the final 32-camera studio calibration.",
    "# - Do not assume cam0, COLMAP, or Three.js world axes; read the frame definition below.",
    "# - Extrinsics are T_camera_studio = camera_tr_studio_rig, a world-to-camera transform.",
    "# - For a studio point p_studio in meters: p_camera = R @ p_studio + t.",
    "# - p_camera is in the OpenCV camera frame: +x image right, +y image down, +z optical forward.",
    "# - The published studio_rig is a physical studio/world frame, not an OpenCV camera frame.",
    "# - studio_rig origin: mean center of non-4 outer *-2 cameras.",
    "# - studio_rig +Y: physical vertical down, oriented from the upper *-1 layer to the lower *-3 layer.",
    "# - studio_rig +Z: physical forward, opposite the missing 4-2 side gap.",
    "# - studio_rig -Z: backward, toward the missing 4-2 side gap.",
    "# - studio_rig +X: right-handed completion, so +X cross +Y = +Z.",
    "# - Outer labels are side-layer labels. 4-1/4-2/4-3 are top-down cameras and are excluded from frame fitting.",
    "# - Camera center in studio frame: C_studio = -R.T @ t.",
    "# - Camera-to-world pose is inverse(T_camera_studio).",
    "# - If a target world uses p_target = T_target_studio @ p_studio, export extrinsics as",
    "#   T_camera_target = T_camera_studio @ inverse(T_target_studio).",
    "# - For +Y-up viewers such as many Three.js scenes, apply an explicit world-frame conversion;",
    "#   never reinterpret this YAML as +Y-up in place.",
]


def write_pose_yaml(path, poses, coordinate_frame=None):
    lines = [
        "# Combined studio 24+8 relative extrinsics.",
        *STUDIO_RIG_Y_DOWN_Z_FORWARD_COMMENTS,
        f"pose_count: {len(poses)}",
    ]
    append_coordinate_transform_yaml(lines, coordinate_frame)
    lines.append("poses:")
    for index, pose in enumerate(poses):
        validate_pose(index, pose)
        qx, qy, qz, qw = matrix_to_quat_xyzw(pose[:3, :3])
        tx, ty, tz = pose[:3, 3]
        lines.extend([
            f"  - index: {index}",
            f"    tx: {tx:.14g}",
            f"    ty: {ty:.14g}",
            f"    tz: {tz:.14g}",
            f"    qx: {qx:.14g}",
            f"    qy: {qy:.14g}",
            f"    qz: {qz:.14g}",
            f"    qw: {qw:.14g}",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_intrinsics_yaml(path):
    fields = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()

    missing = [key for key in ["type", "width", "height", "parameters"] if key not in fields]
    if missing:
        raise ValueError(f"Missing intrinsics fields {missing} in {path}")

    return {
        "model": fields["type"],
        "width": int(fields["width"]),
        "height": int(fields["height"]),
        "parameters": [float(value) for value in ast.literal_eval(fields["parameters"])],
    }


def write_unified_camera_yaml(path, poses, camera_rows, intrinsics_dir, coordinate_frame=None):
    intrinsics_dir = Path(intrinsics_dir)
    frame_name = coordinate_frame["aligned_coordinate_frame"] if coordinate_frame else "studio_rig_current"
    lines = [
        "# Unified studio 24+8 camera calibration.",
        *STUDIO_RIG_Y_DOWN_Z_FORWARD_COMMENTS,
        "schema_version: 1",
        "artifact: studio_32_camera_calibration",
        f"coordinate_frame: {frame_name}",
        "pose_convention:",
        "  transform: camera_tr_studio_rig",
        "  meaning: rig point to camera coordinates",
        "  multiplication: right",
        "  translation_unit: meter",
        "  quaternion_order: qx qy qz qw",
        "  camera_frame: OpenCV +x right, +y down, +z forward",
        "index_convention:",
        "  outer: indices 0..23 follow labels 1-1,1-2,1-3,...,8-3",
        "  inner: indices 24..31 follow inner0..inner7",
        f"camera_count: {len(camera_rows)}",
    ]
    append_coordinate_transform_yaml(lines, coordinate_frame)
    lines.append("cameras:")
    for row in camera_rows:
        index = int(row["index"])
        pose = poses[index]
        validate_pose(index, pose)
        intrinsics_path = intrinsics_dir / f"intrinsics{index}.yaml"
        intrinsics = parse_intrinsics_yaml(intrinsics_path)
        qx, qy, qz, qw = matrix_to_quat_xyzw(pose[:3, :3])
        tx, ty, tz = pose[:3, 3]
        parameters = ", ".join(format_float(value) for value in intrinsics["parameters"])
        lines.extend([
            f"  - index: {index}",
            f"    label: {yaml_quote(row['label'])}",
            f"    group: {yaml_quote(row['group'])}",
            f"    camera_id: {yaml_quote(row.get('camera_id', ''))}",
            "    intrinsics:",
            f"      model: {intrinsics['model']}",
            f"      width: {intrinsics['width']}",
            f"      height: {intrinsics['height']}",
            f"      parameters: [{parameters}]",
            "    camera_tr_studio_rig:",
            f"      tx: {format_float(tx)}",
            f"      ty: {format_float(ty)}",
            f"      tz: {format_float(tz)}",
            f"      qx: {format_float(qx)}",
            f"      qy: {format_float(qy)}",
            f"      qz: {format_float(qz)}",
            f"      qw: {format_float(qw)}",
            "    sources:",
            f"      pose_yaml: {yaml_quote(row['source_yaml'])}",
            f"      pose_source_index: {row['source_index']}",
            f"      intrinsics_yaml: {yaml_quote(str(intrinsics_path.resolve()))}",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_label_tsv(path, camera_rows):
    lines = ["index\tlabel\tgroup\tcamera_id\tsource_yaml\tsource_index\n"]
    for row in camera_rows:
        lines.append(
            "\t".join([
                str(row["index"]),
                row["label"],
                row["group"],
                row.get("camera_id", ""),
                row["source_yaml"],
                str(row["source_index"]),
            ])
            + "\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def parse_indices(text):
    return [int(item) for item in str(text).split(",") if item.strip()]


def invert_pose(pose):
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def average_rotation(rotations):
    if not rotations:
        return np.eye(3, dtype=np.float64)
    matrix = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    u, _s, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def estimate_reference_from_bridge(bridge_poses, reference_poses):
    transforms = []
    for index in range(min(len(OUTER_CAMERA_LABELS), len(bridge_poses), len(reference_poses))):
        bridge_pose = bridge_poses[index]
        reference_pose = reference_poses[index]
        if bridge_pose is None or reference_pose is None:
            continue
        validate_pose(index, bridge_pose)
        validate_pose(index, reference_pose)
        transforms.append(invert_pose(reference_pose) @ bridge_pose)
    if not transforms:
        return np.eye(4, dtype=np.float64), {
            "outer_alignment_camera_count": 0,
            "outer_alignment_center_rms_m": None,
            "outer_alignment_rotation_rms_deg": None,
        }

    reference_from_bridge = np.eye(4, dtype=np.float64)
    reference_from_bridge[:3, :3] = average_rotation([transform[:3, :3] for transform in transforms])
    reference_from_bridge[:3, 3] = np.mean([transform[:3, 3] for transform in transforms], axis=0)

    center_residuals = []
    rotation_residuals = []
    for index in range(min(len(OUTER_CAMERA_LABELS), len(bridge_poses), len(reference_poses))):
        bridge_pose = bridge_poses[index]
        reference_pose = reference_poses[index]
        if bridge_pose is None or reference_pose is None:
            continue
        predicted_reference_pose = bridge_pose @ invert_pose(reference_from_bridge)
        predicted_center = invert_pose(predicted_reference_pose)[:3, 3]
        target_center = invert_pose(reference_pose)[:3, 3]
        center_residuals.append(float(np.linalg.norm(predicted_center - target_center)))
        delta_r = predicted_reference_pose[:3, :3] @ reference_pose[:3, :3].T
        cos_angle = max(-1.0, min(1.0, (float(np.trace(delta_r)) - 1.0) * 0.5))
        rotation_residuals.append(math.degrees(math.acos(cos_angle)))

    return reference_from_bridge, {
        "outer_alignment_camera_count": len(transforms),
        "outer_alignment_center_rms_m": float(np.sqrt(np.mean(np.square(center_residuals)))) if center_residuals else None,
        "outer_alignment_rotation_rms_deg": float(np.sqrt(np.mean(np.square(rotation_residuals)))) if rotation_residuals else None,
    }


def bridge_poses_in_outer_reference(bridge_poses, outer_reference_poses):
    if len(bridge_poses) < len(OUTER_CAMERA_LABELS):
        return bridge_poses, {
            "outer_alignment_camera_count": 0,
            "outer_alignment_center_rms_m": None,
            "outer_alignment_rotation_rms_deg": None,
            "note": "bridge pose YAML has no outer poses; assuming it is already in the outer reference frame",
        }
    reference_from_bridge, summary = estimate_reference_from_bridge(bridge_poses, outer_reference_poses)
    bridge_from_reference = invert_pose(reference_from_bridge)
    return [pose @ bridge_from_reference if pose is not None else None for pose in bridge_poses], summary


def build_combined_poses(args):
    inner_bridge_indices = parse_indices(args.inner_bridge_indices)
    if len(inner_bridge_indices) != 8:
        raise ValueError("--inner-bridge-indices must contain exactly 8 entries")

    outer_poses = load_pose_yaml(args.outer_final_pose_yaml)
    inner_bridge_poses = load_pose_yaml(args.inner_bridge_pose_yaml)
    inner_bridge_poses, bridge_alignment = bridge_poses_in_outer_reference(inner_bridge_poses, outer_poses)
    inner_camera_ids = read_inner_camera_ids(args.inner_manifest)

    poses = [None for _ in range(32)]
    rows = []
    for outer_index, label in enumerate(OUTER_CAMERA_LABELS):
        pose = outer_poses[outer_index]
        validate_pose(outer_index, pose)
        poses[outer_index] = pose
        rows.append({
            "index": outer_index,
            "label": label,
            "group": "outer",
            "camera_id": label,
            "source_yaml": str(Path(args.outer_final_pose_yaml).resolve()),
            "source_index": outer_index,
        })

    for inner_ordinal, source_index in enumerate(inner_bridge_indices):
        output_index = 24 + inner_ordinal
        pose = inner_bridge_poses[source_index]
        validate_pose(output_index, pose)
        label = f"inner{inner_ordinal}"
        poses[output_index] = pose
        rows.append({
            "index": output_index,
            "label": label,
            "group": "inner",
            "camera_id": inner_camera_ids.get(inner_ordinal, ""),
            "source_yaml": str(Path(args.inner_bridge_pose_yaml).resolve()),
            "source_index": source_index,
        })
    return poses, rows, bridge_alignment


def write_manifest(
        path,
        args,
        pose_yaml,
        label_tsv,
        unified_yaml,
        camera_rows,
        coordinate_frame=None,
        bridge_alignment=None):
    manifest = {
        "schema_version": 1,
        "artifact": "studio_32_relative_extrinsics",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "coordinate_frame": coordinate_frame["aligned_coordinate_frame"] if coordinate_frame else "studio_rig_current",
        "coordinate_transform": coordinate_frame or {},
        "bridge_outer_alignment": bridge_alignment or {},
        "pose_convention": {
            "transform": "camera_tr_studio_rig",
            "meaning": "rig point to camera coordinates",
            "multiplication": "right",
            "translation_unit": "meter",
            "quaternion_order": "qx qy qz qw",
            "camera_frame": "OpenCV +x right, +y down, +z forward",
        },
        "index_convention": {
            "outer": "indices 0..23 follow labels 1-1,1-2,1-3,...,8-3",
            "inner": "indices 24..31 follow inner0..inner7",
        },
        "outputs": {
            "pose_yaml": str(Path(pose_yaml).resolve()),
            "label_tsv": str(Path(label_tsv).resolve()),
            "unified_yaml": str(Path(unified_yaml).resolve()),
        },
        "inputs": {
            "outer_final_pose_yaml": str(Path(args.outer_final_pose_yaml).resolve()),
            "inner_bridge_pose_yaml": str(Path(args.inner_bridge_pose_yaml).resolve()),
            "inner_manifest": str(Path(args.inner_manifest).resolve()) if args.inner_manifest else "",
            "intrinsics_dir": str(Path(args.intrinsics_dir).resolve()),
            "viewer_url": args.viewer_url,
            "run_tag": args.run_tag,
        },
        "cameras": camera_rows,
        "notes": [
            "This file records the 24+8 relative extrinsics matching the current combined Three.js viewer.",
            "studio_32_cameras.yaml is the unified intrinsics + extrinsics file for algorithm consumers.",
            "Outer camera poses come from the latest accepted outer tower rig.",
            "Inner camera poses come from bridge indices 24..31 of the current all32 bridge solve.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-bridge-pose-yaml", type=Path, default=DEFAULT_INNER_BRIDGE_POSE_YAML)
    parser.add_argument("--outer-final-pose-yaml", type=Path, default=DEFAULT_OUTER_FINAL_POSE_YAML)
    parser.add_argument("--inner-manifest", type=Path, default=DEFAULT_T0_ROOT / "small_marker_inner8/manifest.tsv")
    parser.add_argument("--inner-bridge-indices", default=DEFAULT_INNER_BRIDGE_INDICES)
    parser.add_argument("--intrinsics-dir", type=Path, default=DEFAULT_INTRINSICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-tag", default="latest")
    parser.add_argument("--viewer-url", default=DEFAULT_VIEWER_URL)
    parser.add_argument("--no-canonical-studio-frame", action="store_true")
    args = parser.parse_args()

    poses, rows, bridge_alignment = build_combined_poses(args)
    coordinate_frame = None
    if not args.no_canonical_studio_frame:
        coordinate_frame = estimate_frame_from_camera_poses(poses, rows)
        if coordinate_frame is None:
            raise RuntimeError("Could not estimate canonical studio frame from non-4 outer side cameras")
        poses = [transform_pose_to_aligned(pose, coordinate_frame) for pose in poses]
    output_dir = Path(args.output_dir)
    pose_yaml = output_dir / "camera_tr_studio_rig.yaml"
    label_tsv = output_dir / "camera_labels.tsv"
    unified_yaml = output_dir / "studio_32_cameras.yaml"
    manifest_json = output_dir / "manifest.json"
    write_pose_yaml(pose_yaml, poses, coordinate_frame)
    write_label_tsv(label_tsv, rows)
    write_unified_camera_yaml(unified_yaml, poses, rows, args.intrinsics_dir, coordinate_frame)
    write_manifest(
        manifest_json,
        args,
        pose_yaml,
        label_tsv,
        unified_yaml,
        rows,
        coordinate_frame,
        bridge_alignment,
    )
    print(json.dumps({
        "bridge_outer_alignment": bridge_alignment,
        "coordinate_frame": coordinate_frame["aligned_coordinate_frame"] if coordinate_frame else "studio_rig_current",
        "pose_count": len(poses),
        "unified_yaml": str(unified_yaml),
        "pose_yaml": str(pose_yaml),
        "label_tsv": str(label_tsv),
        "manifest_json": str(manifest_json),
        "outer_count": sum(1 for row in rows if row["group"] == "outer"),
        "inner_count": sum(1 for row in rows if row["group"] == "inner"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
