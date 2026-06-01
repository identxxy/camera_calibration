#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


CAMERA_RESULT_NAMES = {
    "cam0": "up_cam0",
    "cam1": "down_cam1",
    "cam2": "down_cam2",
    "cam3": "up_cam3",
}


def as_matrix(value):
    return np.asarray(value, dtype=float)


def invert_transform(t):
    t = as_matrix(t)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def invert_rt(R, t):
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rotation_angle(R):
    c = 0.5 * (float(np.trace(R)) - 1.0)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def rot_to_quat_xyzw(R):
    R = as_matrix(R)
    qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    qy = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    qz = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    q = np.asarray([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


def quat_to_rot_xyzw(q):
    x, y, z, w = [float(v) for v in q]
    return np.asarray([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=float)


def average_rotations(rotations):
    quats = []
    ref = None
    for R in rotations:
        q = rot_to_quat_xyzw(R)
        if ref is None:
            ref = q
        elif float(q @ ref) < 0:
            q = -q
        quats.append(q)
    A = np.zeros((4, 4), dtype=float)
    for q in quats:
        A += np.outer(q, q)
    vals, vecs = np.linalg.eigh(A)
    q = vecs[:, int(np.argmax(vals))]
    if ref is not None and float(q @ ref) < 0:
        q = -q
    q = q / np.linalg.norm(q)
    return quat_to_rot_xyzw(q), q


def load_results(results_dir):
    out = {}
    for cam, name in CAMERA_RESULT_NAMES.items():
        path = Path(results_dir) / name / "result.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        out[cam] = {
            "path": str(path),
            "data": data,
            "R_cam_imu": as_matrix(data["R_cam_imu"]),
        }
    return out


def matrix_to_list(m):
    return [[float(v) for v in row] for row in np.asarray(m)]


def parse_args():
    parser = argparse.ArgumentParser(description="Compose Seeker rotation-only VI calibration YAML.")
    parser.add_argument("--reference-yaml", required=True)
    parser.add_argument("--rotation-results-dir", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    reference = yaml.safe_load(Path(args.reference_yaml).read_text(encoding="utf-8"))
    results = load_results(args.rotation_results_dir)

    t_cam_cam0 = {cam: as_matrix(reference[cam]["T_cam_imu"]) for cam in reference}
    cam0_rotations = {}
    for cam, result in results.items():
        R_cam0_cam = t_cam_cam0[cam][:3, :3].T
        cam0_rotations[cam] = R_cam0_cam @ result["R_cam_imu"]

    R_cam0_imu_avg, q_avg = average_rotations(cam0_rotations.values())

    output = {}
    final_R = {}
    for cam, entry in reference.items():
        if not cam.startswith("cam"):
            continue
        new_entry = dict(entry)
        R_cam_cam0 = t_cam_cam0[cam][:3, :3]
        t_cam_imu = t_cam_cam0[cam][:3, 3]
        R_cam_imu = R_cam_cam0 @ R_cam0_imu_avg
        T = np.eye(4)
        T[:3, :3] = R_cam_imu
        T[:3, 3] = t_cam_imu
        new_entry["T_cam_imu"] = matrix_to_list(T)
        new_entry["T_imu_cam"] = matrix_to_list(invert_rt(R_cam_imu, t_cam_imu))
        new_entry["timeshift_cam_imu"] = 0.0
        output[cam] = new_entry
        final_R[cam] = R_cam_imu

    pair_consistency = []
    for a, b in (("cam1", "cam2"), ("cam0", "cam3")):
        R_b_a = t_cam_cam0[b][:3, :3] @ t_cam_cam0[a][:3, :3].T
        predicted = R_b_a @ results[a]["R_cam_imu"]
        measured = results[b]["R_cam_imu"]
        angle = rotation_angle(measured @ predicted.T)
        pair_consistency.append({
            "from": a,
            "to": b,
            "angle_error_rad": float(angle),
            "angle_error_deg": float(math.degrees(angle)),
        })

    summary = {
        "format": "seeker_vi_rotation_only_summary_v0",
        "reference_yaml": str(Path(args.reference_yaml).resolve()),
        "rotation_results_dir": str(Path(args.rotation_results_dir).resolve()),
        "output_yaml": str(Path(args.output_yaml).resolve()),
        "clock_assumption": "camera and IMU timestamps are in the same clock domain",
        "time_offset_s": 0.0,
        "averaged_R_cam0_imu": matrix_to_list(R_cam0_imu_avg),
        "averaged_q_cam0_imu_xyzw": [float(v) for v in q_avg.tolist()],
        "final_R_cam_imu": {cam: matrix_to_list(R) for cam, R in final_R.items()},
        "pair_consistency": pair_consistency,
        "rotation_results": {},
        "warnings": [
            "Rotation-only result: translation in the YAML is inherited from the previous camera-only rig, not a measured IMU lever arm.",
            "Single-AprilTag pose is noisier than full-board localization; treat this as a first-pass rotation estimate.",
            "Accelerometer/translation calibration was not solved in this pass.",
            "Kalibr could not extract this repository's custom board as an AprilGrid target.",
        ],
    }
    for cam, result in results.items():
        data = result["data"]
        residual = data.get("inlier_residual", {})
        summary["rotation_results"][cam] = {
            "source": result["path"],
            "valid_tag_poses": data.get("valid_tag_poses"),
            "angular_samples": data.get("angular_samples"),
            "angular_inliers": data.get("angular_inliers"),
            "median_residual_rad_s": residual.get("median_rad_s"),
            "p95_residual_rad_s": residual.get("p95_rad_s"),
            "R_cam_imu_raw": matrix_to_list(result["R_cam_imu"]),
            "R_cam0_imu_from_this_camera": matrix_to_list(cam0_rotations[cam]),
            "cam0_frame_residual_deg": float(math.degrees(rotation_angle(cam0_rotations[cam] @ R_cam0_imu_avg.T))),
        }

    Path(args.output_yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_yaml).write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_yaml": str(Path(args.output_yaml).resolve()),
        "summary_json": str(Path(args.summary_json).resolve()),
        "pair_consistency": pair_consistency,
        "warnings": summary["warnings"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
