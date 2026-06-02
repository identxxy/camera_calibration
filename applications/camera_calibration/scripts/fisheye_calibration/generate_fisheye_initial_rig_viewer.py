#!/usr/bin/env python3
"""Generate a Three.js viewer for the Seeker four-fisheye initial rig."""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
CALIB_SCRIPTS = REPO_ROOT / "scripts" / "calib"
if str(CALIB_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CALIB_SCRIPTS))

import generate_threejs_rig_viewer as rig_viewer  # noqa: E402


CAMERA_DEFAULTS = {
    "cam0": {
        "label": "left-up",
        "kind": "fisheye_upper",
        "slot": "left_up",
        "display_center": [-1.0, 0.0, 0.55],
    },
    "cam1": {
        "label": "left-down",
        "kind": "fisheye_lower",
        "slot": "left_down",
        "display_center": [-1.0, 0.0, -0.55],
    },
    "cam2": {
        "label": "right-down",
        "kind": "fisheye_lower",
        "slot": "right_down",
        "display_center": [1.0, 0.0, -0.55],
    },
    "cam3": {
        "label": "right-up",
        "kind": "fisheye_upper",
        "slot": "right_up",
        "display_center": [1.0, 0.0, 0.55],
    },
}


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def invert_transform(T):
    out = np.eye(4, dtype=float)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def transform_from_yaml(entry):
    if "T_cam_imu" in entry:
        return np.asarray(entry["T_cam_imu"], dtype=float)
    if "T_cam_rig" in entry:
        return np.asarray(entry["T_cam_rig"], dtype=float)
    if "T_cam_ref" in entry:
        return np.asarray(entry["T_cam_ref"], dtype=float)
    raise KeyError("camera entry must contain T_cam_imu, T_cam_rig, or T_cam_ref")


def camera_sort_key(name):
    if name.startswith("cam"):
        try:
            return (0, int(name[3:]))
        except ValueError:
            pass
    return (1, name)


def apply_display_center(camera_tr_reference, display_center):
    reference_tr_camera = invert_transform(camera_tr_reference)
    reference_tr_camera[:3, 3] = np.asarray(display_center, dtype=float)
    return invert_transform(reference_tr_camera)


def metric_center(camera_tr_reference):
    return invert_transform(camera_tr_reference)[:3, 3]


def norm3(values):
    values = np.asarray(values, dtype=float)
    return float(np.linalg.norm(values))


def normalize(values):
    values = np.asarray(values, dtype=float)
    norm = np.linalg.norm(values)
    if norm < 1e-12:
        return None
    return values / norm


def skew(values):
    x, y, z = [float(v) for v in values]
    return np.asarray([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=float)


def rotation_between_vectors(source, target):
    source = normalize(source)
    target = normalize(target)
    if source is None or target is None:
        return np.eye(3, dtype=float)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=float)
    if dot < -1.0 + 1e-10:
        axis = np.cross(source, np.asarray([1.0, 0.0, 0.0], dtype=float))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(source, np.asarray([0.0, 1.0, 0.0], dtype=float))
        axis = normalize(axis)
        return -np.eye(3, dtype=float) + 2.0 * np.outer(axis, axis)
    cross = np.cross(source, target)
    K = skew(cross)
    return np.eye(3, dtype=float) + K + K @ K * ((1.0 - dot) / float(cross @ cross))


def rotate_point(point, rotation):
    return [float(v) for v in (rotation @ np.asarray(point, dtype=float))]


def rotate_vector(vector, rotation):
    out = rotation @ np.asarray(vector, dtype=float)
    norm = np.linalg.norm(out)
    if norm > 0:
        out = out / norm
    return [float(v) for v in out]


def rotate_line(line, rotation):
    return [rotate_point(line[0], rotation), rotate_point(line[1], rotation)]


def rotate_camera_geometry(camera, rotation):
    camera["center"] = rotate_point(camera["center"], rotation)
    camera["basis"] = {
        axis: rotate_vector(vector, rotation)
        for axis, vector in camera["basis"].items()
    }
    camera["axes"] = {
        axis: rotate_line(line, rotation)
        for axis, line in camera["axes"].items()
    }
    camera["frustum_lines"] = [
        rotate_line(line, rotation)
        for line in camera["frustum_lines"]
    ]


def rotate_reference_frame(frame, rotation):
    frame["center"] = rotate_point(frame["center"], rotation)
    frame["axes"] = {
        axis: rotate_line(line, rotation)
        for axis, line in frame["axes"].items()
    }


def average_camera_vector(cameras, slot_names, field):
    values = []
    for camera in cameras:
        if camera["metrics"].get("slot") in slot_names:
            values.append(np.asarray(camera[field], dtype=float))
    if not values:
        return None
    return np.mean(values, axis=0)


def average_basis_vector(cameras, slot_names, axis):
    values = []
    for camera in cameras:
        if camera["metrics"].get("slot") in slot_names:
            values.append(np.asarray(camera["basis"][axis], dtype=float))
    if not values:
        return None
    return normalize(np.mean(values, axis=0))


def align_scene_up(cameras, reference_frames):
    upper_slots = {"left_up", "right_up"}
    lower_slots = {"left_down", "right_down"}
    upper_center = average_camera_vector(cameras, upper_slots, "center")
    lower_center = average_camera_vector(cameras, lower_slots, "center")
    source = None
    source_kind = "layer_offset"
    if upper_center is not None and lower_center is not None:
        source = normalize(upper_center - lower_center)
    if source is None:
        source = average_basis_vector(cameras, upper_slots, "z")
        source_kind = "upper_optical_axis"
    target = np.asarray([0.0, -1.0, 0.0], dtype=float)
    rotation = rotation_between_vectors(source, target)
    for camera in cameras:
        rotate_camera_geometry(camera, rotation)
    for frame in reference_frames:
        rotate_reference_frame(frame, rotation)
    return {
        "enabled": True,
        "source": source_kind,
        "source_vector_before_alignment": [float(v) for v in source.tolist()] if source is not None else None,
        "target_scene_up_vector": [0.0, -1.0, 0.0],
        "rotation_matrix_three": [[float(v) for v in row] for row in rotation.tolist()],
        "note": "A global viewer-basis rotation is applied after reading the YAML so the scene up axis follows the lower-to-upper camera slot offset. Camera frames remain CV: +X right, +Y down, +Z optical.",
    }


def collect_bound_points(cameras, reference_frames):
    points = []
    for camera in cameras:
        points.append(camera["center"])
        for line in camera["frustum_lines"]:
            points.extend(line)
    for frame in reference_frames:
        points.append(frame["center"])
        for line in frame["axes"].values():
            points.extend(line)
    return points


def parse_label_overrides(values):
    overrides = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Label override must be cam=label, got {raw!r}")
        cam, label = raw.split("=", 1)
        overrides[cam.strip()] = label.strip()
    return overrides


def find_assets_dir(explicit):
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    env = Path.cwd()
    candidates = []
    env_value = None
    try:
        import os

        env_value = os.environ.get("THREEJS_ASSETS_DIR")
    except Exception:
        env_value = None
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend([
        env / "viewer_assets",
        env / "reports" / "viewer_assets",
        REPO_ROOT / "viewer_assets",
        REPO_ROOT / "scripts" / "calib" / "viewer_assets",
    ])
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("three.min.js", "OrbitControls.js", "TransformControls.js")):
            return candidate
    return None


def copy_viewer_assets(output_dir, assets_dir):
    assets_dir = find_assets_dir(assets_dir)
    if assets_dir is None:
        raise FileNotFoundError(
            "Three.js assets not found. Pass --viewer-assets-dir or set THREEJS_ASSETS_DIR "
            "to a directory containing three.min.js, OrbitControls.js, and TransformControls.js."
        )
    for name in ("three.min.js", "OrbitControls.js", "TransformControls.js"):
        shutil.copy2(assets_dir / name, output_dir / name)
    return assets_dir


def build_fisheye_viewer_data(camchain, args):
    label_overrides = parse_label_overrides(args.label)
    cameras = []
    source_transforms = {}
    ordered_names = sorted(camchain.keys(), key=camera_sort_key)
    for index, name in enumerate(ordered_names):
        entry = camchain[name]
        default = CAMERA_DEFAULTS.get(name, {})
        label = label_overrides.get(name, default.get("label", name))
        kind = default.get("kind", "fisheye")
        camera_tr_reference = transform_from_yaml(entry)
        metric_camera_tr_reference = camera_tr_reference.copy()
        source_transforms[name] = {
            "T_cam_reference": metric_camera_tr_reference.tolist(),
            "metric_center_reference": metric_center(metric_camera_tr_reference).tolist(),
        }
        display_center = np.asarray(default.get("display_center", [float(index), 0.0, 0.0]), dtype=float)
        display_center = display_center * float(args.layout_scale)
        if args.layout == "metric":
            viewer_camera_tr_reference = metric_camera_tr_reference
        else:
            viewer_camera_tr_reference = apply_display_center(metric_camera_tr_reference, display_center)
        geometry = rig_viewer.build_camera_geometry(
            viewer_camera_tr_reference,
            args.frustum_depth,
            args.frustum_half_width,
            args.frustum_half_height,
            args.axis_length,
        )
        intrinsics = entry.get("intrinsics", [])
        distortion = entry.get("distortion_coeffs", [])
        resolution = entry.get("resolution", [])
        metrics = {
            "source_key": name,
            "slot": default.get("slot", ""),
            "camera_model": entry.get("camera_model", ""),
            "distortion_model": entry.get("distortion_model", ""),
            "fx": float(intrinsics[0]) if len(intrinsics) > 0 else None,
            "fy": float(intrinsics[1]) if len(intrinsics) > 1 else None,
            "cx": float(intrinsics[2]) if len(intrinsics) > 2 else None,
            "cy": float(intrinsics[3]) if len(intrinsics) > 3 else None,
            "distortion_norm": norm3(distortion) if distortion else None,
            "metric_center_norm_m": norm3(source_transforms[name]["metric_center_reference"]),
            "rostopic": entry.get("rostopic", ""),
        }
        camera = {
            "index": index,
            "label": label,
            "kind": kind,
            "used": True,
            "center": geometry["center"],
            "basis": geometry["basis"],
            "frustum_lines": geometry["frustum_lines"],
            "axes": geometry["axes"],
            "metrics": metrics,
            "calibration_quality": {
                "source": "fisheye_initial_kb8",
                "decision": "reference",
                "fx": metrics["fx"],
                "fy": metrics["fy"],
                "cx": metrics["cx"],
                "cy": metrics["cy"],
                "intrinsics_status": "reference",
                "intrinsics_flags": [],
            },
            "coverage": {
                "fisheye_initial": {
                    "active": True,
                    "status": "reference",
                    "quality": "reference",
                    "detail": "Initial four-fisheye rig viewer generated from Seeker KB8 camchain.",
                }
            },
        }
        cameras.append(camera)

    reference_frames = [{
        "label": args.imu_label,
        "center": [0.0, 0.0, 0.0],
        "axes": {
            "x": [[0.0, 0.0, 0.0], [float(args.imu_axis_length), 0.0, 0.0]],
            "y": [[0.0, 0.0, 0.0], [0.0, float(args.imu_axis_length), 0.0]],
            "z": [[0.0, 0.0, 0.0], [0.0, 0.0, float(args.imu_axis_length)]],
        },
        "color": "#fbbc04",
        "label_color": "#fbbc04",
        "marker_radius": 0.018,
        "note": "IMU / rig reference frame drawn after the same viewer-basis alignment as the cameras.",
    }]
    scene_alignment = align_scene_up(cameras, reference_frames)
    bound_points = collect_bound_points(cameras, reference_frames)

    return {
        "title": args.title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_camchain_yaml": str(Path(args.camchain_yaml).resolve()),
        "coordinate_note": (
            "Camera local frames use the OpenCV convention: +X image-right, "
            "+Y image-down, +Z optical-forward. The viewer converts CV to "
            "Three.js with [x, y, z] -> [x, -y, -z], then applies a global "
            "scene alignment so the lower-to-upper camera slot offset is "
            "Three.js -Y, which is the configured viewer world-up direction "
            "for this fisheye schematic."
        ),
        "scene_alignment": scene_alignment,
        "frustum": {
            "default_near": args.default_near,
            "default_far": args.default_far,
            "half_width_over_depth": args.frustum_half_width / args.frustum_depth,
            "half_height_over_depth": args.frustum_half_height / args.frustum_depth,
            "fill_opacity": args.frustum_fill_alpha,
        },
        "viewer_options": {
            "enable_overlap": False,
            "world_up_three": [0.0, -1.0, 0.0],
            "default_reference_up_vector_three": [0.0, -1.0, 0.0],
            "default_visibility": {
                "inner": True,
                "outer": True,
                "outer_topdown": True,
                "outer_colmap": True,
            },
            "layout": args.layout,
        },
        "dataset_coverage": {
            "default_mode": "fisheye_initial",
            "modes": ["fisheye_initial"],
        },
        "metrics": {
            "camera_count": len(cameras),
            "layout": args.layout,
            "model_counts": {},
        },
        "source_transforms": source_transforms,
        "reference_frames": reference_frames,
        "sparse_point_cloud": {
            "source": "",
            "coordinate_frame": "rig_reference",
            "point_count": 0,
            "positions": [],
            "colors": [],
        },
        "reprojection_reports": [],
        "cameras": cameras,
        "bounds": rig_viewer.compute_bounds(bound_points),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a four-fisheye initial rig Three.js viewer.")
    parser.add_argument("--camchain-yaml", required=True, help="Seeker KB8 camchain YAML.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--viewer-assets-dir", default="", help="Directory with three.min.js, OrbitControls.js, TransformControls.js.")
    parser.add_argument("--title", default="Fisheye Vision Initial Calibration Viewer")
    parser.add_argument("--layout", choices=["display-grid", "metric"], default="display-grid")
    parser.add_argument("--layout-scale", type=float, default=0.22)
    parser.add_argument("--frustum-depth", type=float, default=0.20)
    parser.add_argument("--frustum-half-width", type=float, default=0.13)
    parser.add_argument("--frustum-half-height", type=float, default=0.095)
    parser.add_argument("--default-near", type=float, default=0.18)
    parser.add_argument("--default-far", type=float, default=0.42)
    parser.add_argument("--frustum-fill-alpha", type=float, default=0.12)
    parser.add_argument("--axis-length", type=float, default=0.075)
    parser.add_argument("--imu-label", default="IMU / rig reference")
    parser.add_argument("--imu-axis-length", type=float, default=0.16)
    parser.add_argument("--label", action="append", default=[], help="Override labels as cam0=left-up.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = copy_viewer_assets(output_dir, args.viewer_assets_dir)
    camchain = load_yaml(args.camchain_yaml)
    rig_data = build_fisheye_viewer_data(camchain, args)
    rig_data["viewer_assets_source"] = str(Path(assets_dir).resolve())
    rig_viewer.write_html(output_dir / "index.html", rig_data)
    (output_dir / "rig_data.json").write_text(json.dumps(rig_data, indent=2), encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
