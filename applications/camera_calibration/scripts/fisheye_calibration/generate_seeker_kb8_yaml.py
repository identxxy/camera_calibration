#!/usr/bin/env python3
"""Generate a Seeker/Kalibr-style camchain YAML with native KB8 intrinsics."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


ROSTOPICS = {
    "cam0": "/fisheye/left/image_raw",
    "cam1": "/fisheye/right/image_raw",
    "cam2": "/fisheye/bright/image_raw",
    "cam3": "/fisheye/bleft/image_raw",
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
        "intrinsics_conversion": "none; native KB8/equidistant parameters copied from KB8 JSON",
        "extrinsics_reference": "T_cam_imu uses imu == cam0/LU reference, not a measured IMU frame",
        "camera_order": {
            "cam0": "packed row 0 / left_up",
            "cam1": "packed row 1 / left_down",
            "cam2": "packed row 2 / right_down",
            "cam3": "packed row 3 / right_up",
        },
        "kb8_convention": {
            "camera_model": "kb8",
            "distortion_model": "kb8",
            "intrinsics": ["fx", "fy", "cx", "cy"],
            "distortion_coeffs": ["k1", "k2", "k3", "k4"],
            "source_ba_model": "CentralThinPrismFisheyeModel",
        },
    }

    for idx in range(4):
        cam = f"cam{idx}"
        cam_kb8 = kb8["cameras"][cam]
        params = [float(v) for v in cam_kb8["params"]]
        if len(params) < 8:
            raise ValueError(f"{cam} needs at least 8 KB8 params")
        width = int(cam_kb8["width"])
        height = int(cam_kb8["height"])
        t_cam_imu = poses_ref[cam]
        entry = {
            "camera_model": "kb8",
            "distortion_model": "kb8",
            "intrinsics": params[:4],
            "distortion_coeffs": params[4:8],
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

    output_path = Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(output, f, sort_keys=False, default_flow_style=False)

    metadata_path = Path(args.metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
