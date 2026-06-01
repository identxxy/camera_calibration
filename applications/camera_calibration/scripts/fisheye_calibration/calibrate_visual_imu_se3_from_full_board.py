#!/usr/bin/env python3
"""Estimate full Seeker camera-IMU SE(3) from full-board visual poses and IMU."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

import calibrate_visual_imu_rotation_from_full_board as fb


NS_PER_S = 1_000_000_000
CAPTURE_BY_CAMERA = {
    "cam0": "up",
    "cam1": "down",
    "cam2": "down",
    "cam3": "up",
}


def load_imu(mcap_path, imu_topic):
    from mcap.reader import make_reader

    stamps = []
    gyros = []
    accels = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(topics=[imu_topic], log_time_order=True):
            if channel.topic != imu_topic or not schema or schema.name != "sensor_msgs/msg/Imu":
                continue
            imu = fb.parse_sensor_msgs_imu_cdr(message.data)
            stamps.append(int(imu["stamp_ns"]))
            gyros.append([float(v) for v in imu["angular_velocity"]])
            accels.append([float(v) for v in imu["linear_acceleration"]])
    return (
        np.asarray(stamps, dtype=np.int64),
        np.asarray(gyros, dtype=float),
        np.asarray(accels, dtype=float),
    )


def interpolate_vector(stamps_ns, values, query_ns):
    if query_ns < stamps_ns[0] or query_ns > stamps_ns[-1]:
        return None
    idx = int(np.searchsorted(stamps_ns, query_ns))
    if idx == 0:
        return values[0]
    if idx >= len(stamps_ns):
        return values[-1]
    t0 = stamps_ns[idx - 1]
    t1 = stamps_ns[idx]
    alpha = (query_ns - t0) / max(1, t1 - t0)
    return (1.0 - alpha) * values[idx - 1] + alpha * values[idx]


def load_camera_poses(cam, dataset_path, intr, args):
    dataset = fb.read_dataset(dataset_path)
    geometry = dataset["geometries"][0]
    poses = []
    rejected = {"missing_timestamp": 0, "too_few_features": 0, "pose_failed": 0, "non_monotonic": 0}
    last_t = None
    for imageset in dataset["imagesets"]:
        stamp = imageset["timestamp_ns"]
        if stamp is None:
            rejected["missing_timestamp"] += 1
            continue
        if last_t is not None and stamp <= last_t:
            rejected["non_monotonic"] += 1
            continue
        object_points = []
        image_points = []
        for feature_id, x, y in imageset["cameras"][0]:
            point = fb.object_point_for_feature(feature_id, geometry)
            if point is None:
                continue
            object_points.append(point)
            image_points.append((x, y))
        if len(object_points) < args.min_features:
            rejected["too_few_features"] += 1
            continue
        object_xyz = np.asarray(object_points, dtype=float)
        image_uv = np.asarray(image_points, dtype=float)
        try:
            pose = fb.robust_pose_from_observations(
                object_xyz,
                image_uv,
                intr,
                args.min_features,
                args.max_pose_rmse_px,
                args.max_pose_inlier_px)
        except Exception:
            pose = None
        if pose is None:
            rejected["pose_failed"] += 1
            continue
        R_cw = pose["R_cam_board"]
        t_cw = pose["t_cam_board"]
        R_wc = R_cw.T
        p_wc = -R_wc @ t_cw
        poses.append({
            "index": int(imageset["index"]),
            "src": imageset["src"],
            "timestamp_ns": int(stamp),
            "R_cw": R_cw,
            "R_wc": R_wc,
            "t_cw": t_cw,
            "p_wc": p_wc,
            "features": int(pose["features"]),
            "inliers": int(pose["inliers"]),
            "reprojection_rmse_px": float(pose["reprojection_rmse_px"]),
            "reprojection_p95_px": float(pose["reprojection_p95_px"]),
        })
        last_t = stamp
        if len(poses) % 100 == 0:
            print(f"{cam}: poses={len(poses)}", flush=True)
    return {
        "dataset_imagesets": len(dataset["imagesets"]),
        "poses": poses,
        "rejected": rejected,
        "geometry_cell_length_m": geometry["cell_length_in_meters"],
    }


def local_quadratic_accel(times_s, positions, center_index, half_window):
    lo = max(0, center_index - half_window)
    hi = min(len(times_s), center_index + half_window + 1)
    if hi - lo < 5:
        return None
    t0 = times_s[center_index]
    tau = times_s[lo:hi] - t0
    if np.max(tau) - np.min(tau) < 1e-3:
        return None
    X = np.column_stack([np.ones_like(tau), tau, 0.5 * tau * tau])
    coeff, *_ = np.linalg.lstsq(X, positions[lo:hi], rcond=None)
    return coeff[2]


def angular_velocity_between(R_cw_a, R_cw_b, dt_s):
    return fb.so3_log(R_cw_a @ R_cw_b.T) / dt_s


def compute_omega_series(times_s, rotations_cw, sign, half_window):
    omegas = [None] * len(times_s)
    for i in range(half_window, len(times_s) - half_window):
        dt = times_s[i + half_window] - times_s[i - half_window]
        if dt <= 1e-3:
            continue
        omegas[i] = sign * angular_velocity_between(
            rotations_cw[i - half_window],
            rotations_cw[i + half_window],
            dt)
    return omegas


def local_alpha(times_s, omegas, center_index, window_s):
    idxs = []
    t0 = times_s[center_index]
    for i, omega in enumerate(omegas):
        if omega is None:
            continue
        if abs(times_s[i] - t0) <= window_s:
            idxs.append(i)
    if len(idxs) < 5:
        return None
    tau = np.asarray([times_s[i] - t0 for i in idxs], dtype=float)
    W = np.asarray([omegas[i] for i in idxs], dtype=float)
    X = np.column_stack([np.ones_like(tau), tau])
    coeff, *_ = np.linalg.lstsq(X, W, rcond=None)
    return coeff[1]


def build_motion_samples(cam, pose_data, R_ci, sign, imu_data, args):
    poses = pose_data["poses"]
    times_ns = np.asarray([p["timestamp_ns"] for p in poses], dtype=np.int64)
    times_s = (times_ns - times_ns[0]).astype(float) / NS_PER_S
    rotations_cw = [p["R_cw"] for p in poses]
    positions = np.asarray([p["p_wc"] for p in poses], dtype=float)
    omega_series = compute_omega_series(times_s, rotations_cw, sign, args.rotation_half_window)
    stamps_ns, _gyros, accels = imu_data
    samples = []
    for i, pose in enumerate(poses):
        accel_world = local_quadratic_accel(times_s, positions, i, args.position_half_window)
        if accel_world is None:
            continue
        omega_c = omega_series[i]
        if omega_c is None:
            continue
        alpha_c = local_alpha(times_s, omega_series, i, args.alpha_window_s)
        if alpha_c is None:
            continue
        imu_accel = interpolate_vector(stamps_ns, accels, pose["timestamp_ns"])
        if imu_accel is None:
            continue
        if np.linalg.norm(omega_c) < args.min_omega_rad_s and np.linalg.norm(alpha_c) < args.min_alpha_rad_s2:
            continue
        R_cw = pose["R_cw"]
        R_iw = R_ci.T @ R_cw
        A_c = fb.skew(alpha_c) + fb.skew(omega_c) @ fb.skew(omega_c)
        samples.append({
            "timestamp_ns": int(pose["timestamp_ns"]),
            "R_iw": R_iw,
            "R_cw": R_cw,
            "A_c": A_c,
            "accel_world": accel_world,
            "imu_accel": imu_accel,
            "omega_norm": float(np.linalg.norm(omega_c)),
            "alpha_norm": float(np.linalg.norm(alpha_c)),
        })
    return samples


def solve_translation(samples_by_camera, prior, rotation_chain, args):
    captures = ["down", "up"]
    capture_index = {name: i for i, name in enumerate(captures)}
    # Unknowns: cam0_t_imu, g_down, g_up, accel_bias_down, accel_bias_up.
    n = 3 + 3 * len(captures) + 3 * len(captures)
    rows = []
    rhs = []
    sample_records = []
    for cam, samples in samples_by_camera.items():
        capture = CAPTURE_BY_CAMERA[cam]
        cap_idx = capture_index[capture]
        R_c0 = prior[cam]["T_cam_imu_prior"][:3, :3]
        t_c0 = prior[cam]["T_cam_imu_prior"][:3, 3]
        R_ci = rotation_chain[cam][:3, :3]
        for s in samples:
            A_c = s["A_c"]
            R_iw = s["R_iw"]
            y = s["imu_accel"] - R_iw @ s["accel_world"] - R_ci.T @ A_c @ t_c0
            M = np.zeros((3, n), dtype=float)
            M[:, 0:3] = R_ci.T @ A_c @ R_c0
            g0 = 3 + 3 * cap_idx
            b0 = 3 + 3 * len(captures) + 3 * cap_idx
            M[:, g0:g0 + 3] = -R_iw
            M[:, b0:b0 + 3] = np.eye(3)
            rows.append(M)
            rhs.append(y)
            sample_records.append({
                "camera": cam,
                "capture": capture,
                "timestamp_ns": s["timestamp_ns"],
                "omega_norm": s["omega_norm"],
                "alpha_norm": s["alpha_norm"],
            })
    if not rows:
        raise RuntimeError("No accelerometer samples available for SE3 solve")
    A = np.concatenate(rows, axis=0)
    b = np.concatenate(rhs, axis=0)

    def fit(mask):
        x, residuals, rank, singular = np.linalg.lstsq(A[mask], b[mask], rcond=None)
        r = (A @ x - b).reshape(-1, 3)
        norms = np.linalg.norm(r, axis=1)
        return x, residuals, rank, singular, r, norms

    all_mask = np.ones(A.shape[0], dtype=bool)
    x, residuals, rank, singular, residual_vecs, residual_norms = fit(all_mask)
    sample_norms = residual_norms
    sample_cutoff = float(np.percentile(sample_norms, args.accel_inlier_percentile))
    sample_inliers = sample_norms <= sample_cutoff
    row_mask = np.repeat(sample_inliers, 3)
    x, residuals, rank, singular, residual_vecs, residual_norms = fit(row_mask)
    final_sample_norms = residual_norms
    t_cam0_imu = x[0:3]
    g = {}
    bias = {}
    for cap, idx in capture_index.items():
        g0 = 3 + 3 * idx
        b0 = 3 + 3 * len(captures) + 3 * idx
        g[cap] = x[g0:g0 + 3]
        bias[cap] = x[b0:b0 + 3]
    condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None
    return {
        "t_cam0_imu": t_cam0_imu,
        "gravity": g,
        "accel_bias": bias,
        "rank": int(rank),
        "condition": condition,
        "singular_values": [float(v) for v in singular.tolist()],
        "samples": sample_records,
        "sample_inlier_count": int(np.sum(sample_inliers)),
        "sample_count": int(len(sample_inliers)),
        "initial_residual_norm_m_s2": fb.summarize(sample_norms.tolist()),
        "inlier_residual_norm_m_s2": fb.summarize(final_sample_norms[sample_inliers].tolist()),
        "all_residual_norm_m_s2": fb.summarize(final_sample_norms.tolist()),
    }


def matrix_to_list(m):
    return [[float(v) for v in row] for row in np.asarray(m)]


def invert_transform(T):
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def write_calibration(prior_yaml, prior, rotation_yaml, solve, output_yaml):
    rotation_data = yaml.safe_load(Path(rotation_yaml).read_text(encoding="utf-8"))
    T_0i = np.eye(4)
    T_0i[:3, :3] = np.asarray(rotation_data["cam0"]["T_cam_imu"], dtype=float)[:3, :3]
    T_0i[:3, 3] = solve["t_cam0_imu"]
    output = {}
    for cam in sorted(prior):
        entry = dict(prior_yaml[cam])
        T_c0 = prior[cam]["T_cam_imu_prior"].copy()
        T_ci = T_c0 @ T_0i
        entry["T_cam_imu"] = matrix_to_list(T_ci)
        entry["T_imu_cam"] = matrix_to_list(invert_transform(T_ci))
        output[cam] = entry
    cams = sorted(output)
    for i, cam in enumerate(cams):
        prev = cams[i - 1]
        T_cam = np.asarray(output[cam]["T_cam_imu"], dtype=float)
        T_prev = np.asarray(output[prev]["T_cam_imu"], dtype=float)
        output[cam]["T_cn_cnm1"] = matrix_to_list(T_cam @ np.linalg.inv(T_prev))
    Path(output_yaml).write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    return output, T_0i


def write_html(path, summary, output_yaml, summary_json):
    rows = []
    for cam, item in sorted(summary["cameras"].items()):
        rows.append(
            f"<tr><td>{cam}</td><td>{item['valid_poses']}</td><td>{item['motion_samples']}</td>"
            f"<td>{item['feature_median']:.1f}</td><td>{item['pose_rmse_median_px']:.3f}</td></tr>")
    g_rows = []
    for cap, g in sorted(summary["solve"]["gravity"].items()):
        b = summary["solve"]["accel_bias"][cap]
        g_rows.append(
            f"<tr><td>{cap}</td><td>{summary['solve']['gravity_norm'][cap]:.4f}</td>"
            f"<td>[{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]</td>"
            f"<td>[{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}]</td></tr>")
    t = summary["T_cam0_imu"]["translation"]
    residual = summary["solve"]["inlier_residual_norm_m_s2"]
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Full SE3 Visual-IMU Calibration</title>
<style>
body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:0; background:#f6f8fb; color:#1f2937; }}
main {{ max-width: 1180px; margin:0 auto; padding:32px 28px 48px; }}
h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
h2 {{ margin:28px 0 12px; font-size:20px; letter-spacing:0; }}
table {{ width:100%; border-collapse:collapse; background:white; border:1px solid #d9e0ea; }}
th,td {{ padding:10px 12px; border-bottom:1px solid #e5e9f0; text-align:left; vertical-align:top; }}
th {{ background:#eef3f8; }}
code {{ background:#edf2f7; border:1px solid #d8dee9; border-radius:4px; padding:1px 5px; }}
.panel {{ background:white; border:1px solid #d9e0ea; border-radius:8px; padding:14px 16px; margin:12px 0; }}
</style></head><body><main>
<h1>Full SE3 Visual-IMU Calibration</h1>
<p>Board 静止、rig 运动；使用 full-board visual pose、KB8 camera-camera rig prior、gyro rotation alignment 和 accelerometer lever-arm equation 估计物理 IMU frame 下的 <code>T_cam_imu</code>。</p>
<section class="panel"><p>Final YAML: <code>{output_yaml}</code></p><p>Summary JSON: <code>{summary_json}</code></p></section>
<h2>Estimated T_cam0_imu</h2>
<section class="panel"><p>translation: <code>[{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}] m</code></p>
<p>inlier accel residual median/p95: <code>{residual.get('median', float('nan')):.4f} / {residual.get('p95', float('nan')):.4f} m/s^2</code></p>
<p>linear system rank/condition: <code>{summary['solve']['rank']} / {summary['solve']['condition']:.3e}</code></p></section>
<h2>Per-Camera Visual Input</h2><table><thead><tr><th>Camera</th><th>Valid poses</th><th>Motion samples</th><th>Median features</th><th>Pose RMSE median px</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Gravity And Bias</h2><table><thead><tr><th>Capture</th><th>|g|</th><th>g_world</th><th>accel bias</th></tr></thead><tbody>{''.join(g_rows)}</tbody></table>
</main></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-yaml", required=True)
    parser.add_argument("--rotation-yaml", required=True)
    parser.add_argument("--rotation-summary", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--down-mcap", required=True)
    parser.add_argument("--up-mcap", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--report-html", required=True)
    parser.add_argument("--imu-topic", default="/seeker/imu")
    parser.add_argument("--min-features", type=int, default=80)
    parser.add_argument("--max-pose-rmse-px", type=float, default=3.0)
    parser.add_argument("--max-pose-inlier-px", type=float, default=5.0)
    parser.add_argument("--position-half-window", type=int, default=8)
    parser.add_argument("--rotation-half-window", type=int, default=4)
    parser.add_argument("--alpha-window-s", type=float, default=0.45)
    parser.add_argument("--min-omega-rad-s", type=float, default=0.06)
    parser.add_argument("--min-alpha-rad-s2", type=float, default=0.05)
    parser.add_argument("--accel-inlier-percentile", type=float, default=90.0)
    return parser.parse_args()


def main():
    args = parse_args()
    prior_yaml, prior = fb.load_kb8_prior(args.prior_yaml)
    rotation_data = yaml.safe_load(Path(args.rotation_yaml).read_text(encoding="utf-8"))
    rotation_summary = json.loads(Path(args.rotation_summary).read_text(encoding="utf-8"))
    rotation_chain = {
        cam: np.asarray(rotation_data[cam]["T_cam_imu"], dtype=float)
        for cam in sorted(prior)
    }
    imu = {
        "down": load_imu(args.down_mcap, args.imu_topic),
        "up": load_imu(args.up_mcap, args.imu_topic),
    }
    pose_by_camera = {}
    samples_by_camera = {}
    cameras_summary = {}
    for cam in sorted(prior):
        pose_data = load_camera_poses(
            cam,
            Path(args.feature_root) / f"{cam}_features.bin",
            prior[cam],
            args)
        pose_by_camera[cam] = pose_data
        sign = float(rotation_summary["cameras"][cam]["visual_omega_sign"])
        samples = build_motion_samples(
            cam,
            pose_data,
            rotation_chain[cam][:3, :3],
            sign,
            imu[CAPTURE_BY_CAMERA[cam]],
            args)
        samples_by_camera[cam] = samples
        poses = pose_data["poses"]
        cameras_summary[cam] = {
            "dataset_imagesets": pose_data["dataset_imagesets"],
            "valid_poses": len(poses),
            "motion_samples": len(samples),
            "rejected": pose_data["rejected"],
            "feature_median": float(np.median([p["features"] for p in poses])) if poses else None,
            "pose_rmse_median_px": float(np.median([p["reprojection_rmse_px"] for p in poses])) if poses else None,
            "pose_rmse_p95_px": float(np.percentile([p["reprojection_rmse_px"] for p in poses], 95)) if poses else None,
        }
        print(json.dumps({"camera": cam, **cameras_summary[cam]}, indent=2), flush=True)

    solve = solve_translation(samples_by_camera, prior, rotation_chain, args)
    output, T_0i = write_calibration(prior_yaml, prior, args.rotation_yaml, solve, args.output_yaml)
    solve_json = {
        "t_cam0_imu": [float(v) for v in solve["t_cam0_imu"].tolist()],
        "gravity": {k: [float(v) for v in val.tolist()] for k, val in solve["gravity"].items()},
        "gravity_norm": {k: float(np.linalg.norm(val)) for k, val in solve["gravity"].items()},
        "accel_bias": {k: [float(v) for v in val.tolist()] for k, val in solve["accel_bias"].items()},
        "rank": solve["rank"],
        "condition": solve["condition"],
        "singular_values": solve["singular_values"],
        "sample_count": solve["sample_count"],
        "sample_inlier_count": solve["sample_inlier_count"],
        "initial_residual_norm_m_s2": solve["initial_residual_norm_m_s2"],
        "inlier_residual_norm_m_s2": solve["inlier_residual_norm_m_s2"],
        "all_residual_norm_m_s2": solve["all_residual_norm_m_s2"],
    }
    summary = {
        "format": "seeker_full_board_visual_imu_se3_summary_v0",
        "method": "full-board visual pose + gyro rotation + accelerometer lever-arm linear solve",
        "T_cam0_imu": {
            "matrix": matrix_to_list(T_0i),
            "translation": [float(v) for v in T_0i[:3, 3].tolist()],
            "rotation": matrix_to_list(T_0i[:3, :3]),
        },
        "cameras": cameras_summary,
        "solve": solve_json,
        "output_yaml": str(Path(args.output_yaml).resolve()),
        "warnings": [
            "Camera-camera rig extrinsics come from the previous KB8 prior; physical IMU translation is estimated in cam0 frame.",
            "Translation quality depends on visual second derivatives and accelerometer bias observability; inspect gravity norm, residuals, and condition number before production use.",
            "Time offset is fixed at 0.0 s in this pass.",
        ],
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html(args.report_html, summary, Path(args.output_yaml).resolve(), Path(args.summary_json).resolve())
    print(json.dumps({
        "output_yaml": str(Path(args.output_yaml).resolve()),
        "summary_json": str(Path(args.summary_json).resolve()),
        "report_html": str(Path(args.report_html).resolve()),
        "t_cam0_imu_m": summary["T_cam0_imu"]["translation"],
        "gravity_norm": summary["solve"]["gravity_norm"],
        "accel_residual": summary["solve"]["inlier_residual_norm_m_s2"],
        "condition": summary["solve"]["condition"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
