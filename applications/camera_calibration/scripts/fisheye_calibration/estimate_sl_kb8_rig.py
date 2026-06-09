#!/usr/bin/env python3
"""Estimate an SL four-fisheye KB8 rig from fixed intrinsics and full-board features."""

import argparse
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_visual_imu_rotation_from_full_board as fb  # noqa: E402
import optimize_sl_up_down_phi as phi_opt  # noqa: E402


CAM_ORDER = ["left_up", "left_down", "right_down", "right_up"]
CAM_TO_INDEX = {
    "left_up": "cam0",
    "left_down": "cam1",
    "right_down": "cam2",
    "right_up": "cam3",
}


def load_intrinsics(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intr = {}
    for cam, item in data["cameras"].items():
        fx, fy, cx, cy, k1, k2, k3, k4 = [float(v) for v in item["params"][:8]]
        intr[cam] = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k": [k1, k2, k3, k4],
            "width": int(item["width"]),
            "height": int(item["height"]),
        }
    return data, intr


def make_t(R, t):
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.asarray(R, dtype=float)
    out[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return out


def inv_t(T):
    out = np.eye(4, dtype=float)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def rot_to_quat_xyzw(R):
    return fb.rot_to_quat_xyzw(np.asarray(R, dtype=float))


def matrix_to_pose(T):
    q = rot_to_quat_xyzw(T[:3, :3])
    return {
        "tx": float(T[0, 3]),
        "ty": float(T[1, 3]),
        "tz": float(T[2, 3]),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "qz": float(q[2]),
        "qw": float(q[3]),
        "matrix": [[float(v) for v in row] for row in T],
    }


def stats(values):
    values = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def pair_estimate(dataset_path, left_name, right_name, intrinsics, args):
    dataset = fb.read_dataset(dataset_path)
    if dataset["num_cameras"] != 2:
        raise ValueError(f"{dataset_path} must have exactly two cameras")
    geometry = dataset["geometries"][0]
    transforms = []
    records = []
    rejected = {
        "too_few_left": 0,
        "too_few_right": 0,
        "pose_failed_left": 0,
        "pose_failed_right": 0,
    }

    for imageset in dataset["imagesets"]:
        poses = []
        for camera_index, camera_name in enumerate([left_name, right_name]):
            object_points = []
            image_points = []
            for feature_id, x, y in imageset["cameras"][camera_index]:
                point = fb.object_point_for_feature(feature_id, geometry)
                if point is None:
                    continue
                object_points.append(point)
                image_points.append((x, y))
            if len(object_points) < args.min_features:
                rejected[f"too_few_{'left' if camera_index == 0 else 'right'}"] += 1
                poses.append(None)
                continue
            object_xyz = np.asarray(object_points, dtype=float)
            image_uv = np.asarray(image_points, dtype=float)
            pose = fb.robust_pose_from_observations(
                object_xyz,
                image_uv,
                intrinsics[camera_name],
                args.min_features,
                args.max_pose_rmse_px,
                args.max_pose_inlier_px,
            )
            if pose is None:
                rejected[f"pose_failed_{'left' if camera_index == 0 else 'right'}"] += 1
                poses.append(None)
                continue
            poses.append(pose)

        if poses[0] is None or poses[1] is None:
            continue

        T_left_board = make_t(poses[0]["R_cam_board"], poses[0]["t_cam_board"])
        T_right_board = make_t(poses[1]["R_cam_board"], poses[1]["t_cam_board"])
        T_right_left = T_right_board @ inv_t(T_left_board)
        transforms.append(T_right_left)
        records.append({
            "imageset_index": int(imageset["index"]),
            "filename": imageset["filename"],
            "left_features": int(poses[0]["features"]),
            "right_features": int(poses[1]["features"]),
            "left_inliers": int(poses[0]["inliers"]),
            "right_inliers": int(poses[1]["inliers"]),
            "left_rmse_px": float(poses[0]["reprojection_rmse_px"]),
            "right_rmse_px": float(poses[1]["reprojection_rmse_px"]),
            "left_p95_px": float(poses[0]["reprojection_p95_px"]),
            "right_p95_px": float(poses[1]["reprojection_p95_px"]),
        })

    if len(transforms) < args.min_pair_views:
        raise RuntimeError(f"{left_name}/{right_name}: only {len(transforms)} valid pair views")

    R_avg, _ = fb.average_rotations([T[:3, :3] for T in transforms])
    t_med = np.median(np.asarray([T[:3, 3] for T in transforms], dtype=float), axis=0)
    T_initial = np.eye(4, dtype=float)
    T_initial[:3, :3] = R_avg
    T_initial[:3, 3] = t_med

    rot_err = np.asarray([fb.rotation_angle(T[:3, :3] @ T_initial[:3, :3].T) for T in transforms], dtype=float)
    trans_err = np.asarray([np.linalg.norm(T[:3, 3] - T_initial[:3, 3]) for T in transforms], dtype=float)
    rot_cut = max(args.min_rotation_inlier_rad, float(np.percentile(rot_err, args.inlier_percentile)))
    trans_cut = max(args.min_translation_inlier_m, float(np.percentile(trans_err, args.inlier_percentile)))
    keep = (rot_err <= rot_cut) & (trans_err <= trans_cut)
    if int(np.sum(keep)) >= args.min_pair_views:
        inlier_transforms = [T for T, flag in zip(transforms, keep) if flag]
    else:
        inlier_transforms = transforms
        keep = np.ones(len(transforms), dtype=bool)

    R_final, _ = fb.average_rotations([T[:3, :3] for T in inlier_transforms])
    t_final = np.median(np.asarray([T[:3, 3] for T in inlier_transforms], dtype=float), axis=0)
    T_final = np.eye(4, dtype=float)
    T_final[:3, :3] = R_final
    T_final[:3, 3] = t_final

    final_rot_err = [fb.rotation_angle(T[:3, :3] @ T_final[:3, :3].T) for T in inlier_transforms]
    final_trans_err = [np.linalg.norm(T[:3, 3] - T_final[:3, 3]) for T in inlier_transforms]
    for record, flag, re, te in zip(records, keep, rot_err, trans_err):
        record["rig_inlier"] = bool(flag)
        record["initial_rotation_error_deg"] = float(math.degrees(re))
        record["initial_translation_error_m"] = float(te)

    return {
        "left_camera": left_name,
        "right_camera": right_name,
        "dataset": str(Path(dataset_path).resolve()),
        "imagesets": int(len(dataset["imagesets"])),
        "valid_pair_views": int(len(transforms)),
        "rig_inlier_views": int(np.sum(keep)),
        "rejected": rejected,
        "T_right_left": T_final,
        "T_right_left_pose": matrix_to_pose(T_final),
        "left_rmse_px": stats([r["left_rmse_px"] for r in records]),
        "right_rmse_px": stats([r["right_rmse_px"] for r in records]),
        "rig_rotation_error_deg": stats([math.degrees(v) for v in final_rot_err]),
        "rig_translation_error_m": stats(final_trans_err),
        "records": records,
    }


def back_transform(axis):
    out = np.eye(4, dtype=float)
    if axis == "x":
        out[:3, :3] = np.diag([1.0, -1.0, -1.0])
    elif axis == "y":
        out[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    elif axis == "z":
        out[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    else:
        raise ValueError(axis)
    return out


def average_two_se3(a, b):
    R, _ = fb.average_rotations([a[:3, :3], b[:3, :3]])
    out = np.eye(4, dtype=float)
    out[:3, :3] = R
    out[:3, 3] = 0.5 * (a[:3, 3] + b[:3, 3])
    return out


def as_yaml_matrix(T):
    return [[float(v) for v in row] for row in T]


def build_camchain(kb8_json, poses):
    out = {}
    for cam_name in CAM_ORDER:
        entry = kb8_json["cameras"][cam_name]
        params = [float(v) for v in entry["params"][:8]]
        cam_key = CAM_TO_INDEX[cam_name]
        T_cam_ref = poses[cam_name]
        out[cam_key] = {
            "name": cam_name,
            "camera_model": "kb8",
            "distortion_model": "kb8",
            "intrinsics": params[:4],
            "distortion_coeffs": params[4:8],
            "resolution": [int(entry["width"]), int(entry["height"])],
            "T_cam_ref": as_yaml_matrix(T_cam_ref),
            "T_ref_cam": as_yaml_matrix(inv_t(T_cam_ref)),
            "reference_frame": "left_up_optical_frame",
            "rostopic": f"/camera/{cam_name}/h264",
        }
    return out


def write_camchain_yaml(kb8_json, poses, output_yaml):
    out = build_camchain(kb8_json, poses)
    with Path(output_yaml).open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)


def maybe_optimize_phi(camchain, result, args):
    if not args.optimize_up_down_phi:
        result["photometric_phi_optimization"] = {"enabled": False}
        return camchain
    if not args.phi_mcap or not args.phi_raw_root:
        raise ValueError("--optimize-up-down-phi requires --phi-mcap and --phi-raw-root")

    phi_args = SimpleNamespace(
        width=args.phi_width,
        height=args.phi_height,
        seam_width_px=args.phi_seam_width_px,
        opt_frames=args.phi_opt_frames,
        min_phi_deg=args.phi_min_deg,
        max_phi_deg=args.phi_max_deg,
        coarse_step_deg=args.phi_coarse_step_deg,
        fine_step_deg=args.phi_fine_step_deg,
    )
    frame_stamps = phi_opt.render.load_frame_timestamps(args.phi_mcap)
    left_rows = phi_opt.render.common_index_pairs(
        frame_stamps, "left_up", "left_down", args.phi_max_common_frames)
    right_rows = phi_opt.render.common_index_pairs(
        frame_stamps, "right_up", "right_down", args.phi_max_common_frames)
    if len(left_rows) < args.phi_opt_frames or len(right_rows) < args.phi_opt_frames:
        raise RuntimeError(
            f"Not enough synchronized frames for phi optimization: "
            f"left={len(left_rows)}, right={len(right_rows)}")

    left = phi_opt.optimize_side(
        camchain, args.phi_raw_root, left_rows, "left_up", "left_down", phi_args)
    right = phi_opt.optimize_side(
        camchain, args.phi_raw_root, right_rows, "right_up", "right_down", phi_args)
    optimized = phi_opt.apply_phi_to_camchain(
        camchain, left["best_phi_deg"], right["best_phi_deg"])
    summary = {
        "enabled": True,
        "model": "T_down_up = Rz(phi) * Ry(180deg), shared optical center",
        "left": left,
        "right": right,
    }
    result["photometric_phi_optimization"] = summary
    if args.phi_output_json:
        path = Path(args.phi_output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return optimized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb8-json", required=True)
    parser.add_argument("--top-dataset", required=True)
    parser.add_argument("--bottom-dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--min-features", type=int, default=120)
    parser.add_argument("--min-pair-views", type=int, default=8)
    parser.add_argument("--max-pose-rmse-px", type=float, default=8.0)
    parser.add_argument("--max-pose-inlier-px", type=float, default=12.0)
    parser.add_argument("--inlier-percentile", type=float, default=75.0)
    parser.add_argument("--min-rotation-inlier-rad", type=float, default=0.05)
    parser.add_argument("--min-translation-inlier-m", type=float, default=0.05)
    parser.add_argument("--back-axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--optimize-up-down-phi", action="store_true",
                        help="Enable photometric seam-loss optimization of up/down roll phi. Off by default.")
    parser.add_argument("--phi-mcap",
                        help="MCAP path used for photometric phi optimization when --optimize-up-down-phi is set.")
    parser.add_argument("--phi-raw-root",
                        help="Directory containing left_up.h264, left_down.h264, right_up.h264, right_down.h264.")
    parser.add_argument("--phi-output-json",
                        help="Optional JSON path for the phi loss curve and selected values.")
    parser.add_argument("--phi-width", type=int, default=960)
    parser.add_argument("--phi-height", type=int, default=480)
    parser.add_argument("--phi-seam-width-px", type=int, default=20)
    parser.add_argument("--phi-max-common-frames", type=int, default=600)
    parser.add_argument("--phi-opt-frames", type=int, default=12)
    parser.add_argument("--phi-min-deg", type=float, default=-5.0)
    parser.add_argument("--phi-max-deg", type=float, default=5.0)
    parser.add_argument("--phi-coarse-step-deg", type=float, default=1.0)
    parser.add_argument("--phi-fine-step-deg", type=float, default=0.1)
    args = parser.parse_args()

    kb8_json, intrinsics = load_intrinsics(args.kb8_json)
    top = pair_estimate(args.top_dataset, "left_up", "right_up", intrinsics, args)
    bottom = pair_estimate(args.bottom_dataset, "left_down", "right_down", intrinsics, args)

    T_back = back_transform(args.back_axis)
    T_right_left_top = top["T_right_left"]
    T_right_left_bottom_in_top = inv_t(T_back) @ bottom["T_right_left"] @ T_back
    T_right_left = average_two_se3(T_right_left_top, T_right_left_bottom_in_top)

    poses = {
        "left_up": np.eye(4, dtype=float),
        "left_down": T_back,
        "right_up": T_right_left,
        "right_down": T_back @ T_right_left,
    }

    result = {
        "format": "SL four-fisheye KB8 rig",
        "convention": "T_B_A maps points from camera A optical coordinates to camera B optical coordinates.",
        "reference_frame": "left_up_optical_frame",
        "camera_order": CAM_TO_INDEX,
        "constraints": {
            "left_up_left_down_shared_center": True,
            "right_up_right_down_shared_center": True,
            "up_down_back_to_back_rotation_degrees": 180,
            "back_rotation_axis": args.back_axis,
            "back_translation_m": [0.0, 0.0, 0.0],
        },
        "observed_pairs": {
            "T_right_up_left_up": top["T_right_left_pose"],
            "T_right_down_left_down": bottom["T_right_left_pose"],
            "T_right_down_left_down_converted_to_top_frame": matrix_to_pose(T_right_left_bottom_in_top),
        },
        "averaged": {
            "T_right_left": matrix_to_pose(T_right_left),
        },
        "poses_from_left_up_reference": {
            name: matrix_to_pose(T) for name, T in poses.items()
        },
        "quality": {
            "top_left_right": {k: v for k, v in top.items() if k not in {"T_right_left", "records"}},
            "bottom_left_right": {k: v for k, v in bottom.items() if k not in {"T_right_left", "records"}},
        },
    }

    records = {
        "top_left_right": top["records"],
        "bottom_left_right": bottom["records"],
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    camchain = build_camchain(kb8_json, poses)
    camchain = maybe_optimize_phi(camchain, result, args)
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.records_json).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with Path(args.output_yaml).open("w", encoding="utf-8") as f:
        yaml.safe_dump(camchain, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
