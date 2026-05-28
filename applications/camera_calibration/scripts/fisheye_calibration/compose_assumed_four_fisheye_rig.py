#!/usr/bin/env python3
"""Compose a four-fisheye rig from two observed horizontal pairs plus 180-degree assumptions."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def quat_to_rot(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n == 0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ])


def rot_to_quat(rot):
    m = np.asarray(rot, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


def pose_to_matrix(pose):
    q = [pose["qx"], pose["qy"], pose["qz"], pose["qw"]]
    t = [pose["tx"], pose["ty"], pose["tz"]]
    out = np.eye(4)
    out[:3, :3] = quat_to_rot(q)
    out[:3, 3] = t
    return out


def matrix_to_pose(matrix):
    q = rot_to_quat(matrix[:3, :3])
    t = matrix[:3, 3]
    return {
        "tx": float(t[0]),
        "ty": float(t[1]),
        "tz": float(t[2]),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "qz": float(q[2]),
        "qw": float(q[3]),
        "matrix": [[float(v) for v in row] for row in matrix],
    }


def load_camera_tr_rig(path, relative_index=1):
    data = yaml.safe_load(Path(path).read_text())
    poses = {int(p["index"]): p for p in data["poses"]}
    if relative_index not in poses:
        raise ValueError(f"Pose index {relative_index} not found in {path}")
    return pose_to_matrix(poses[relative_index])


def average_two_se3(a, b):
    qa = rot_to_quat(a[:3, :3])
    qb = rot_to_quat(b[:3, :3])
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = qa + qb
    q = q / np.linalg.norm(q)
    out = np.eye(4)
    out[:3, :3] = quat_to_rot(q)
    out[:3, 3] = 0.5 * (a[:3, 3] + b[:3, 3])
    return out


def relative_transforms(poses_in_lu):
    out = {}
    for a_name, t_a_lu in poses_in_lu.items():
        for b_name, t_b_lu in poses_in_lu.items():
            out[f"T_{b_name}_{a_name}"] = matrix_to_pose(t_b_lu @ np.linalg.inv(t_a_lu))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-camera-tr-rig", required=True)
    parser.add_argument("--bottom-camera-tr-rig", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-left", default="cam1")
    parser.add_argument("--top-right", default="cam2")
    parser.add_argument("--bottom-left", default="cam0")
    parser.add_argument("--bottom-right", default="cam3")
    parser.add_argument("--back-translation", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--back-rotation-axis", choices=["x", "y", "z"], default="y")
    args = parser.parse_args()

    t_top = load_camera_tr_rig(args.top_camera_tr_rig)
    t_bottom = load_camera_tr_rig(args.bottom_camera_tr_rig)

    back = np.eye(4)
    if args.back_rotation_axis == "x":
        back[:3, :3] = np.diag([1.0, -1.0, -1.0])
    elif args.back_rotation_axis == "y":
        back[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    else:
        back[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    back[:3, 3] = np.array(args.back_translation, dtype=float)

    t_top_from_bottom = np.linalg.inv(back) @ t_bottom @ back
    t_lr_avg = average_two_se3(t_top, t_top_from_bottom)
    t_bottom_consistent = back @ t_lr_avg @ np.linalg.inv(back)

    poses_in_lu = {
        args.top_left: np.eye(4),
        args.top_right: t_lr_avg,
        args.bottom_left: back,
        args.bottom_right: back @ t_lr_avg,
    }

    result = {
        "convention": "T_B_A maps a point from camera A coordinates into camera B coordinates: p_B = T_B_A * p_A",
        "source": "large_mcap_pairwise_plus_180deg_assumption",
        "camera_mapping": {
            "top_left": args.top_left,
            "top_right": args.top_right,
            "bottom_left": args.bottom_left,
            "bottom_right": args.bottom_right,
        },
        "assumptions": {
            "back_to_back_rotation_degrees": 180,
            "back_rotation_axis": args.back_rotation_axis,
            "back_translation": args.back_translation,
            "back_translation_note": "Zero unless CAD/measured baseline is supplied.",
        },
        "observed": {
            "T_top_right_top_left": matrix_to_pose(t_top),
            "T_bottom_right_bottom_left": matrix_to_pose(t_bottom),
            "T_top_from_bottom_after_back_to_back_conversion": matrix_to_pose(t_top_from_bottom),
        },
        "averaged": {
            "T_right_left": matrix_to_pose(t_lr_avg),
            "T_bottom_right_bottom_left_consistent": matrix_to_pose(t_bottom_consistent),
        },
        "poses_from_top_left_reference": {
            name: matrix_to_pose(matrix) for name, matrix in poses_in_lu.items()
        },
        "relative_transforms": relative_transforms(poses_in_lu),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
