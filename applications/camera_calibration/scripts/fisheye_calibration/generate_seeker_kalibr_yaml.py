#!/usr/bin/env python3
"""Generate a Seeker/Kalibr-style camchain YAML from KB8 intrinsics and assumed rig extrinsics."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


ROSTOPICS = {
    "cam0": "/fisheye/left/image_raw",
    "cam1": "/fisheye/right/image_raw",
    "cam2": "/fisheye/bright/image_raw",
    "cam3": "/fisheye/bleft/image_raw",
}


def kb8_theta_distorted(theta, k):
    t2 = theta * theta
    return theta * (1.0 + k[0] * t2 + k[1] * t2 * t2 + k[2] * t2 * t2 * t2 + k[3] * t2 * t2 * t2 * t2)


def fit_kb8_to_omni_radtan(params, width, height):
    fx, fy, cx, cy, k1, k2, k3, k4 = params
    max_radius = max(
        math.hypot(x - cx, y - cy)
        for x in (0.0, width - 1.0)
        for y in (0.0, height - 1.0)
    )
    # Keep the fit in the valid image circle, but avoid the singularity near pi.
    theta_max = min(2.35, max_radius / max(1.0, min(fx, fy)) * 1.08)
    theta = np.linspace(0.0, theta_max, 220)[1:]
    target = kb8_theta_distorted(theta, [k1, k2, k3, k4])

    best = None
    for xi in np.linspace(0.05, 2.2, 431):
        denom = np.cos(theta) + xi
        if np.any(np.abs(denom) < 1e-4):
            continue
        s = np.sin(theta) / denom
        if np.any(~np.isfinite(s)):
            continue
        design = np.stack([s, s**3, s**5, s**7], axis=1)
        coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
        pred = design @ coeff
        err = pred - target
        rms_norm = float(np.sqrt(np.mean(err * err)))
        if best is None or rms_norm < best["rms_norm"]:
            best = {
                "xi": float(xi),
                "coeff": coeff,
                "rms_norm": rms_norm,
                "max_norm": float(np.max(np.abs(err))),
                "theta_max": float(theta_max),
            }
    if best is None:
        raise RuntimeError("Could not fit omni+radtan approximation")

    a1, a3, a5, a7 = best["coeff"]
    if abs(a1) < 1e-12:
        raise RuntimeError("Degenerate omni+radtan fit")
    return {
        "xi": best["xi"],
        "fx": float(fx * a1),
        "fy": float(fy * a1),
        "cx": float(cx),
        "cy": float(cy),
        "distortion": [float(a3 / a1), float(a5 / a1), 0.0, 0.0, float(a7 / a1)],
        "fit": {
            "theta_max_rad": best["theta_max"],
            "rms_px_fx": float(best["rms_norm"] * fx),
            "rms_px_fy": float(best["rms_norm"] * fy),
            "max_px_fx": float(best["max_norm"] * fx),
            "max_px_fy": float(best["max_norm"] * fy),
        },
    }


def matrix_from_pose(pose):
    return np.array(pose["matrix"], dtype=float)


def inv_t(matrix):
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ matrix[:3, 3]
    return out


def as_list(matrix):
    return [[float(v) for v in row] for row in matrix]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb8-json", required=True)
    parser.add_argument("--extrinsics-json", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--metadata-json", required=True)
    args = parser.parse_args()

    kb8 = json.loads(Path(args.kb8_json).read_text())
    extr = json.loads(Path(args.extrinsics_json).read_text())

    # T_cam_imu is generated with imu == cam0 / left-up reference.
    poses_ref = {
        name: matrix_from_pose(pose)
        for name, pose in extr["poses_from_top_left_reference"].items()
    }

    output = {}
    metadata = {
        "source_kb8_json": args.kb8_json,
        "source_extrinsics_json": args.extrinsics_json,
        "intrinsics_conversion": "KB8 fitted to Seeker-supported omni+radtan approximation",
        "extrinsics_reference": "T_cam_imu uses imu == cam0/LU reference, not a measured IMU frame",
        "camera_order": {
            "cam0": "packed row 0 / left_up",
            "cam1": "packed row 1 / left_down",
            "cam2": "packed row 2 / right_down",
            "cam3": "packed row 3 / right_up",
        },
        "fit_quality": {},
    }

    for idx in range(4):
        cam = f"cam{idx}"
        cam_kb8 = kb8["cameras"][cam]
        width = int(cam_kb8["width"])
        height = int(cam_kb8["height"])
        fit = fit_kb8_to_omni_radtan(cam_kb8["params"], width, height)
        t_cam_imu = poses_ref[cam]
        entry = {
            "camera_model": "omni",
            "distortion_model": "radtan",
            "intrinsics": [fit["xi"], fit["fx"], fit["fy"], fit["cx"], fit["cy"]],
            "distortion_coeffs": fit["distortion"],
            "resolution": [width, height],
            "T_cam_imu": as_list(t_cam_imu),
            "T_imu_cam": as_list(inv_t(t_cam_imu)),
            "timeshift_cam_imu": 0.0,
            "rostopic": ROSTOPICS[cam],
        }
        if idx == 0:
            # Close the chain for tools that expect cam0 relative to cam3.
            entry["T_cn_cnm1"] = as_list(t_cam_imu @ inv_t(poses_ref["cam3"]))
        else:
            entry["T_cn_cnm1"] = as_list(t_cam_imu @ inv_t(poses_ref[f"cam{idx - 1}"]))
        output[cam] = entry
        metadata["fit_quality"][cam] = fit["fit"]

    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(output, f, sort_keys=False, default_flow_style=False)

    metadata_path = Path(args.metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
