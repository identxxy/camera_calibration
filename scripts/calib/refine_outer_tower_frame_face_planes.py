#!/usr/bin/env python3
"""Prototype BA for independent outer-tower planes per synchronized frame/face."""

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "refine_outer_tower_delta_prior.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("refine_outer_tower_delta_prior_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


INTRINSICS_REFINE_MODES = (
    "fixed",
    "per_camera_fxfycxcy",
    "per_camera_opencv5",
)


def default_tower_layout():
    return {
        "first_tag_id": 0,
        "face_id_stride": 32,
        "tag_columns": 2,
        "tag_rows": 16,
        "tag_size_m": 0.08,
        "tag_spacing_m": 0.02,
        "tag_rotation_degrees": 180,
    }


def tower_layout_from_args(args):
    return base.tower_layout_from_args(args)


def face_id_for_feature(feature_id, layout):
    face_id, local_tag_id = base.tower_face_and_local_tag_id(feature_id, layout)
    if face_id is None or local_tag_id is None:
        return None
    columns = int(layout["tag_columns"])
    rows = int(layout["tag_rows"])
    if local_tag_id < 0 or local_tag_id >= columns * rows:
        return None
    return int(face_id)


def face_local_point_for_feature(feature_id, layout):
    return base.tower_face_local_point_for_feature(feature_id, layout)


def make_observation(frame_idx, filename, cam_idx, feature_id, xy, local_point, face_id):
    return {
        "frame_index": int(frame_idx),
        "filename": str(filename),
        "camera_index": int(cam_idx),
        "feature_id": int(feature_id),
        "xy": np.asarray(xy, dtype=np.float64),
        "local_point": np.asarray(local_point, dtype=np.float64),
        "face_id": int(face_id),
        "key": (int(frame_idx), int(face_id)),
    }


def build_frame_face_observations(dataset, layout):
    observations = []
    by_frame_face = defaultdict(list)
    by_camera = [[] for _ in range(dataset["camera_count"])]
    for frame_idx, imageset in enumerate(dataset["imagesets"]):
        filename = imageset.get("filename", str(frame_idx))
        for cam_idx, features in enumerate(imageset["features"]):
            for x, y, feature_id in features:
                local_point = face_local_point_for_feature(feature_id, layout)
                face_id = face_id_for_feature(feature_id, layout)
                if local_point is None or face_id is None:
                    continue
                obs = make_observation(
                    frame_idx,
                    filename,
                    cam_idx,
                    feature_id,
                    [x, y],
                    local_point,
                    face_id)
                observations.append(obs)
                by_frame_face[obs["key"]].append(obs)
                by_camera[cam_idx].append(obs)
    return observations, dict(by_frame_face), by_camera


def transform_point(transform, point):
    return transform[:3, :3] @ point + transform[:3, 3]


def project_observation(observation, camera_poses, frame_face_poses, intrinsics):
    key = observation["key"]
    rig_tr_plane = frame_face_poses.get(key)
    if rig_tr_plane is None:
        return "missing_frame_face_pose", None, None
    cam_idx = observation["camera_index"]
    camera_tr_plane = camera_poses[cam_idx] @ rig_tr_plane
    point_camera = transform_point(camera_tr_plane, observation["local_point"])
    pixel = base.project_point(point_camera, intrinsics[cam_idx])
    if pixel is None or not np.all(np.isfinite(pixel)):
        return "invalid_projection", None, None
    residual = pixel - observation["xy"]
    return "ok", pixel, residual


def projection_residuals(observations, camera_poses, frame_face_poses, intrinsics):
    residuals = []
    for observation in observations:
        status, _pixel, residual = project_observation(
            observation,
            camera_poses,
            frame_face_poses,
            intrinsics)
        if status != "ok":
            residuals.extend([1000.0, 1000.0])
        else:
            residuals.extend(residual.tolist())
    return np.asarray(residuals, dtype=np.float64)


def residual_norms(residuals):
    if residuals.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.linalg.norm(residuals.reshape(-1, 2), axis=1)


def summarize_residuals(residuals):
    norms = residual_norms(residuals)
    if norms.size == 0:
        return {
            "count": 0,
            "mean_px": None,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
        }
    return {
        "count": int(norms.size),
        "mean_px": float(np.mean(norms)),
        "median_px": float(np.median(norms)),
        "p90_px": float(np.percentile(norms, 90)),
        "max_px": float(np.max(norms)),
    }


def clip_residual_norms(residuals, max_norm):
    if hasattr(base, "clip_residual_norms"):
        return base.clip_residual_norms(residuals, max_norm)
    if max_norm <= 0 or residuals.size < 2:
        return residuals
    shaped = residuals.reshape(-1, 2).copy()
    norms = np.linalg.norm(shaped, axis=1)
    mask = norms > max_norm
    if np.any(mask):
        shaped[mask] *= (max_norm / norms[mask])[:, None]
    return shaped.reshape(-1)


def load_cv2_or_exit():
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "OpenCV cv2 with calib3d support is required for per-frame-face PnP initialization."
        ) from exc
    return cv2


def cv_intrinsics(intrinsic):
    params = (list(intrinsic["params"]) + [0.0] * 12)[:12]
    camera_matrix = np.asarray([
        [params[0], 0.0, params[2]],
        [0.0, params[1], params[3]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.asarray(
        [params[4], params[5], params[10], params[11], params[6], params[7], params[8], params[9]],
        dtype=np.float64)
    return camera_matrix, dist_coeffs


def pose_from_rvec_tvec(cv2, rvec, tvec):
    rotation, _jacobian = cv2.Rodrigues(rvec)
    return base.pose_matrix(rotation, np.asarray(tvec, dtype=np.float64).reshape(3))


def solve_camera_frame_face_pnp(cv2, observations, intrinsic, args):
    if len(observations) < args.min_pnp_points:
        return None
    object_points = np.asarray([obs["local_point"] for obs in observations], dtype=np.float32)
    image_points = np.asarray([obs["xy"] for obs in observations], dtype=np.float32)
    camera_matrix, dist_coeffs = cv_intrinsics(intrinsic)
    solved = None
    inlier_indices = None

    if args.pnp_ransac and len(observations) >= max(args.min_pnp_points, 6):
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                None,
                None,
                False,
                args.pnp_ransac_iterations,
                args.pnp_ransac_threshold_px,
                0.99,
                flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok, rvec, tvec, inliers = False, None, None, None
        if ok and inliers is not None and len(inliers) >= args.min_pnp_points:
            inlier_indices = [int(index) for index in inliers.ravel()]
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points[inlier_indices],
                    image_points[inlier_indices],
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                ok = False
            if ok:
                solved = (rvec, tvec)

    if solved is None:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                None,
                None,
                False,
                flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok, rvec, tvec = False, None, None
        if not ok:
            return None
        solved = (rvec, tvec)
        inlier_indices = list(range(len(observations)))

    rvec, tvec = solved
    inlier_object = object_points[inlier_indices]
    inlier_image = image_points[inlier_indices]
    try:
        projected, _jacobian = cv2.projectPoints(
            inlier_object,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs)
    except cv2.error:
        return None
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - inlier_image, axis=1)
    median_error = float(np.median(errors)) if errors.size else float("inf")
    if args.max_pnp_median_error_px > 0 and median_error > args.max_pnp_median_error_px:
        return None
    return {
        "camera_tr_plane": pose_from_rvec_tvec(cv2, rvec, tvec),
        "median_error_px": median_error,
        "mean_error_px": float(np.mean(errors)) if errors.size else float("inf"),
        "inlier_count": int(len(inlier_indices)),
        "input_count": int(len(observations)),
    }


def observations_by_camera_for_key(observations):
    result = defaultdict(list)
    for observation in observations:
        result[observation["camera_index"]].append(observation)
    return result


def initialize_frame_face_poses(cv2, by_frame_face, camera_priors, intrinsics, args):
    frame_face_poses = {}
    quality = {}
    for key, observations in sorted(by_frame_face.items()):
        votes = []
        per_camera = observations_by_camera_for_key(observations)
        for cam_idx, camera_observations in sorted(per_camera.items()):
            solved = solve_camera_frame_face_pnp(
                cv2,
                camera_observations,
                intrinsics[cam_idx],
                args)
            if solved is None:
                continue
            rig_tr_plane = base.invert_pose(camera_priors[cam_idx]) @ solved["camera_tr_plane"]
            votes.append({
                "camera_index": cam_idx,
                "pose": rig_tr_plane,
                "median_error_px": solved["median_error_px"],
                "mean_error_px": solved["mean_error_px"],
                "inlier_count": solved["inlier_count"],
                "input_count": solved["input_count"],
            })
        errors = [vote["median_error_px"] for vote in votes]
        if len(votes) == 1:
            frame_face_poses[key] = votes[0]["pose"].copy()
            pose_average = "single_vote"
        elif len(votes) > 1:
            frame_face_poses[key] = base.robust_weighted_average_poses(
                [vote["pose"] for vote in votes],
                errors)
            pose_average = "robust_weighted"
        else:
            pose_average = "none"
        quality[key] = {
            "frame_index": int(key[0]),
            "face_id": int(key[1]),
            "observation_count": int(len(observations)),
            "camera_count": int(len(per_camera)),
            "pnp_vote_count": int(len(votes)),
            "pnp_median_error_px": float(np.median(errors)) if errors else None,
            "pnp_pose_average": pose_average,
            "pnp_votes": votes,
            "active": key in frame_face_poses,
        }
    return frame_face_poses, quality


def rebuild_observation_groups(observations, camera_count):
    by_frame_face = defaultdict(list)
    by_camera = [[] for _ in range(camera_count)]
    for observation in observations:
        by_frame_face[observation["key"]].append(observation)
        by_camera[observation["camera_index"]].append(observation)
    return dict(by_frame_face), by_camera


def filter_active_observations(observations, camera_poses, frame_face_poses, intrinsics, max_residual_px):
    kept = []
    removed = 0
    invalid = 0
    for observation in observations:
        if observation["key"] not in frame_face_poses:
            removed += 1
            invalid += 1
            continue
        if max_residual_px <= 0:
            kept.append(observation)
            continue
        status, _pixel, residual = project_observation(
            observation,
            camera_poses,
            frame_face_poses,
            intrinsics)
        if status != "ok":
            removed += 1
            invalid += 1
            continue
        if float(np.linalg.norm(residual)) <= max_residual_px:
            kept.append(observation)
        else:
            removed += 1
    return kept, {
        "enabled": bool(max_residual_px > 0),
        "max_residual_px": float(max_residual_px),
        "input_observations": int(len(observations)),
        "kept_observations": int(len(kept)),
        "removed_observations": int(removed),
        "missing_pose_or_invalid_projection": int(invalid),
    }


def active_camera_mask(by_camera, min_observations):
    return [len(observations) >= min_observations for observations in by_camera]


def current_camera_poses(camera_priors, deltas):
    return [deltas[i] @ camera_priors[i] for i in range(len(camera_priors))]


def intrinsics_prior_residual(delta, args):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size == 0:
        return np.asarray([], dtype=np.float64)
    residual = [
        delta[0] / args.intrinsics_focal_sigma_frac,
        delta[1] / args.intrinsics_focal_sigma_frac,
    ]
    if delta.size >= 4:
        residual.extend([
            delta[2] / args.intrinsics_principal_sigma_px,
            delta[3] / args.intrinsics_principal_sigma_px,
        ])
    if delta.size >= 9:
        residual.extend((delta[4:9] / args.intrinsics_distortion_sigma).tolist())
    return np.asarray(residual, dtype=np.float64)


def intrinsics_eps_and_step(dim, args):
    eps = [1e-5, 1e-5]
    max_step = [args.intrinsics_max_focal_step_frac, args.intrinsics_max_focal_step_frac]
    if dim >= 4:
        eps.extend([1e-3, 1e-3])
        max_step.extend([
            args.intrinsics_max_principal_step_px,
            args.intrinsics_max_principal_step_px,
        ])
    if dim >= 9:
        eps.extend([1e-5] * 5)
        max_step.extend([args.intrinsics_max_distortion_step] * 5)
    return np.asarray(eps[:dim], dtype=np.float64), np.asarray(max_step[:dim], dtype=np.float64)


def apply_intrinsics(base_intrinsics, mode, per_camera_deltas):
    return base.apply_intrinsics_refinement(base_intrinsics, mode, None, per_camera_deltas)


def optimize_bundle(
        observations_by_frame_face,
        observations_by_camera,
        camera_priors,
        frame_face_poses,
        base_intrinsics,
        args,
        initial_deltas=None,
        initial_per_camera_intrinsics_delta=None):
    if initial_deltas is None:
        deltas = [np.eye(4, dtype=np.float64) for _ in camera_priors]
    else:
        if len(initial_deltas) != len(camera_priors):
            raise ValueError("initial_deltas must match camera count")
        deltas = [np.asarray(delta, dtype=np.float64).copy() for delta in initial_deltas]
    intrinsics_dim = base.intrinsics_delta_dimension(args.intrinsics_refine_mode)
    per_camera_intrinsics_delta = None
    if args.intrinsics_refine_mode.startswith("per_camera"):
        if initial_per_camera_intrinsics_delta is None:
            per_camera_intrinsics_delta = [
                np.zeros(intrinsics_dim, dtype=np.float64) for _ in camera_priors
            ]
        else:
            if len(initial_per_camera_intrinsics_delta) != len(camera_priors):
                raise ValueError("initial_per_camera_intrinsics_delta must match camera count")
            per_camera_intrinsics_delta = [
                np.asarray(delta, dtype=np.float64).copy()
                for delta in initial_per_camera_intrinsics_delta
            ]
    active_cameras = active_camera_mask(observations_by_camera, args.min_camera_observations_for_delta)
    sigma_r = args.delta_rotation_sigma_deg * math.pi / 180.0
    sigma_t = args.delta_translation_sigma_m
    camera_delta_max_rotation_step = args.camera_delta_max_rotation_step_deg * math.pi / 180.0
    frame_face_max_rotation_step = args.frame_face_max_rotation_step_deg * math.pi / 180.0

    def refined_intrinsics():
        return apply_intrinsics(base_intrinsics, args.intrinsics_refine_mode, per_camera_intrinsics_delta)

    def optimizer_residuals(local_observations, local_camera_poses, local_frame_face_poses, local_intrinsics):
        return clip_residual_norms(
            projection_residuals(
                local_observations,
                local_camera_poses,
                local_frame_face_poses,
                local_intrinsics),
            args.optimizer_residual_clip_px)

    for _outer in range(max(0, int(args.outer_iterations))):
        camera_poses = current_camera_poses(camera_priors, deltas)
        intrinsics = refined_intrinsics()
        for key, observations in sorted(observations_by_frame_face.items()):
            if key not in frame_face_poses or len(observations) < args.min_frame_face_observations:
                continue

            def frame_face_residual(transform, key=key, obs=observations):
                local_frame_face_poses = dict(frame_face_poses)
                local_frame_face_poses[key] = transform
                return optimizer_residuals(
                    obs,
                    camera_poses,
                    local_frame_face_poses,
                    intrinsics)

            frame_face_poses[key] = base.optimize_block(
                frame_face_poses[key],
                frame_face_residual,
                args.block_iterations,
                damping=1e-3,
                max_rotation_step=frame_face_max_rotation_step,
                max_translation_step=args.frame_face_max_translation_step_m)

        for cam_idx, observations in enumerate(observations_by_camera):
            if not active_cameras[cam_idx]:
                continue

            def camera_residual(delta, cam_idx=cam_idx, obs=observations):
                local_camera_poses = current_camera_poses(camera_priors, deltas)
                local_camera_poses[cam_idx] = delta @ camera_priors[cam_idx]
                reproj = optimizer_residuals(
                    obs,
                    local_camera_poses,
                    frame_face_poses,
                    refined_intrinsics())
                xi = base.se3_log_approx(delta)
                prior = np.concatenate([xi[:3] / sigma_r, xi[3:] / sigma_t])
                return np.concatenate([reproj, prior])

            deltas[cam_idx] = base.optimize_block(
                deltas[cam_idx],
                camera_residual,
                args.block_iterations,
                damping=1e-3,
                max_rotation_step=camera_delta_max_rotation_step,
                max_translation_step=args.camera_delta_max_translation_step_m)

        if args.intrinsics_refine_mode.startswith("per_camera"):
            eps, max_step = intrinsics_eps_and_step(intrinsics_dim, args)
            for cam_idx, observations in enumerate(observations_by_camera):
                if not active_cameras[cam_idx]:
                    continue

                def camera_intrinsics_residual(delta, cam_idx=cam_idx, obs=observations):
                    local_deltas = list(per_camera_intrinsics_delta)
                    local_deltas[cam_idx] = delta
                    local_intrinsics = apply_intrinsics(
                        base_intrinsics,
                        args.intrinsics_refine_mode,
                        local_deltas)
                    reproj = optimizer_residuals(
                        obs,
                        current_camera_poses(camera_priors, deltas),
                        frame_face_poses,
                        local_intrinsics)
                    return np.concatenate([reproj, intrinsics_prior_residual(delta, args)])

                per_camera_intrinsics_delta[cam_idx] = base.optimize_vector(
                    per_camera_intrinsics_delta[cam_idx],
                    camera_intrinsics_residual,
                    args.intrinsics_block_iterations,
                    damping=1e-2,
                    eps=eps,
                    max_step=max_step)
                per_camera_intrinsics_delta[cam_idx] = base.clamp_intrinsics_delta_to_total_bounds(
                    per_camera_intrinsics_delta[cam_idx],
                    args.intrinsics_max_total_focal_delta_frac,
                    args.intrinsics_max_total_principal_delta_px,
                    args.intrinsics_max_total_distortion_delta)

    return {
        "deltas": deltas,
        "camera_poses": current_camera_poses(camera_priors, deltas),
        "frame_face_poses": frame_face_poses,
        "intrinsics": refined_intrinsics(),
        "per_camera_intrinsics_delta": per_camera_intrinsics_delta,
        "active_cameras": active_cameras,
    }


def format_float(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.8g}"


def observation_feature_fields(feature_id):
    return base.observation_feature_fields(feature_id)


def write_frame_face_pose_yaml(path, frame_face_poses, transform_name):
    lines = [
        f"# Each pose stores {transform_name}; plane coordinates are the face-local AprilTag board coordinates.",
        f"pose_count: {len(frame_face_poses)}",
        "poses:",
    ]
    for index, (key, pose) in enumerate(sorted(frame_face_poses.items())):
        qx, qy, qz, qw = base.matrix_to_quat_xyzw(pose[:3, :3])
        tx, ty, tz = pose[:3, 3]
        lines.extend([
            f"  - index: {index}",
            f"    frame_index: {int(key[0])}",
            f"    face_id: {int(key[1])}",
            f"    tx: {tx:.14g}",
            f"    ty: {ty:.14g}",
            f"    tz: {tz:.14g}",
            f"    qx: {qx:.14g}",
            f"    qy: {qy:.14g}",
            f"    qz: {qz:.14g}",
            f"    qw: {qw:.14g}",
        ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_frame_face_index_tsv(path, frame_face_poses):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["pose_index", "frame_index", "face_id"])
        writer.writeheader()
        for index, key in enumerate(sorted(frame_face_poses)):
            writer.writerow({
                "pose_index": index,
                "frame_index": key[0],
                "face_id": key[1],
            })


def camera_reprojection_rows(
        manifest,
        by_camera,
        before_camera_poses,
        after_camera_poses,
        before_frame_face_poses,
        after_frame_face_poses,
        before_intrinsics,
        after_intrinsics):
    rows = []
    for cam_idx, row in enumerate(manifest):
        observations = by_camera[cam_idx]
        before = residual_norms(projection_residuals(
            observations,
            before_camera_poses,
            before_frame_face_poses,
            before_intrinsics))
        after = residual_norms(projection_residuals(
            observations,
            after_camera_poses,
            after_frame_face_poses,
            after_intrinsics))

        def stat(norms, fn):
            if norms.size == 0:
                return ""
            return format_float(fn(norms))

        rows.append({
            "camera_index": cam_idx,
            "camera_id": row["camera_id"],
            "observation_count": int(after.size),
            "before_median_px": stat(before, np.median),
            "before_p90_px": stat(before, lambda values: np.percentile(values, 90)),
            "after_median_px": stat(after, np.median),
            "after_p90_px": stat(after, lambda values: np.percentile(values, 90)),
            "after_max_px": stat(after, np.max),
        })
    return rows


def write_camera_reprojection_tsv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "camera_index", "camera_id", "observation_count",
                "before_median_px", "before_p90_px",
                "after_median_px", "after_p90_px", "after_max_px",
            ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_observation_residuals_tsv(path, manifest, observations, camera_poses, frame_face_poses, intrinsics):
    fieldnames = [
        "frame_index", "filename", "camera_index", "camera_id",
        "feature_id", "tag_id", "corner_id", "face_id",
        "local_x", "local_y", "local_z",
        "observed_x", "observed_y",
        "projected_x", "projected_y",
        "residual_x_px", "residual_y_px", "residual_px",
        "projection_status",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for obs in observations:
            status, pixel, residual = project_observation(obs, camera_poses, frame_face_poses, intrinsics)
            residual_px = float(np.linalg.norm(residual)) if residual is not None else None
            tag_id, corner_id, face_id = observation_feature_fields(obs["feature_id"])
            writer.writerow({
                "frame_index": obs["frame_index"],
                "filename": obs["filename"],
                "camera_index": obs["camera_index"],
                "camera_id": manifest[obs["camera_index"]]["camera_id"],
                "feature_id": obs["feature_id"],
                "tag_id": tag_id,
                "corner_id": corner_id,
                "face_id": face_id,
                "local_x": format_float(obs["local_point"][0]),
                "local_y": format_float(obs["local_point"][1]),
                "local_z": format_float(obs["local_point"][2]),
                "observed_x": format_float(obs["xy"][0]),
                "observed_y": format_float(obs["xy"][1]),
                "projected_x": format_float(pixel[0] if pixel is not None else None),
                "projected_y": format_float(pixel[1] if pixel is not None else None),
                "residual_x_px": format_float(residual[0] if residual is not None else None),
                "residual_y_px": format_float(residual[1] if residual is not None else None),
                "residual_px": format_float(residual_px),
                "projection_status": status,
            })


def frame_face_quality_rows(
        by_frame_face,
        quality,
        initial_camera_poses,
        final_camera_poses,
        initial_frame_face_poses,
        final_frame_face_poses,
        initial_intrinsics,
        final_intrinsics):
    rows = []
    for key, observations in sorted(by_frame_face.items()):
        initial = summarize_residuals(projection_residuals(
            observations,
            initial_camera_poses,
            initial_frame_face_poses,
            initial_intrinsics))
        final = summarize_residuals(projection_residuals(
            observations,
            final_camera_poses,
            final_frame_face_poses,
            final_intrinsics))
        q = quality.get(key, {})
        rows.append({
            "frame_index": key[0],
            "filename": observations[0]["filename"] if observations else "",
            "face_id": key[1],
            "observation_count": len(observations),
            "camera_count": q.get("camera_count", ""),
            "pnp_vote_count": q.get("pnp_vote_count", 0),
            "pnp_median_error_px": format_float(q.get("pnp_median_error_px")),
            "pnp_pose_average": q.get("pnp_pose_average", "none"),
            "initial_median_px": format_float(initial["median_px"]),
            "initial_p90_px": format_float(initial["p90_px"]),
            "final_median_px": format_float(final["median_px"]),
            "final_p90_px": format_float(final["p90_px"]),
            "final_max_px": format_float(final["max_px"]),
            "active": "yes" if key in final_frame_face_poses else "no",
        })
    return rows


def write_frame_face_quality_tsv(path, rows):
    fieldnames = [
        "frame_index", "filename", "face_id", "observation_count", "camera_count",
        "pnp_vote_count", "pnp_median_error_px", "pnp_pose_average",
        "initial_median_px", "initial_p90_px",
        "final_median_px", "final_p90_px", "final_max_px", "active",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_camera_delta_tsv(path, manifest, deltas, by_camera, active_cameras):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "camera_index", "camera_id", "active",
                "observation_count", "delta_rotation_deg", "delta_translation_m",
            ])
        writer.writeheader()
        for idx, row in enumerate(manifest):
            xi = base.se3_log_approx(deltas[idx])
            writer.writerow({
                "camera_index": idx,
                "camera_id": row["camera_id"],
                "active": "yes" if active_cameras[idx] else "no",
                "observation_count": len(by_camera[idx]),
                "delta_rotation_deg": format_float(np.linalg.norm(xi[:3]) * 180.0 / math.pi),
                "delta_translation_m": format_float(np.linalg.norm(xi[3:])),
            })


def add_arguments(parser):
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--camera_prior_pose_yaml", required=True, type=Path)
    parser.add_argument("--intrinsics_dir", type=Path)
    parser.add_argument("--intrinsics_mode", choices=["colmap_fixed", "central_opencv"], default="colmap_fixed")
    parser.add_argument("--intrinsics_refine_mode", choices=INTRINSICS_REFINE_MODES, default="fixed")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--outer_iterations", type=int, default=5)
    parser.add_argument("--block_iterations", type=int, default=8)
    parser.add_argument("--min_pnp_points", type=int, default=8)
    parser.add_argument("--pnp_ransac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pnp_ransac_iterations", type=int, default=100)
    parser.add_argument("--pnp_ransac_threshold_px", type=float, default=8.0)
    parser.add_argument("--max_pnp_median_error_px", type=float, default=30.0)
    parser.add_argument("--min_frame_face_observations", type=int, default=8)
    parser.add_argument("--min_camera_observations_for_delta", type=int, default=32)
    parser.add_argument(
        "--initial_observation_residual_gate_px",
        type=float,
        default=None,
        help=(
            "Optional wide first-pass residual gate. When set, the script first "
            "optimizes with this gate, then re-gates with --observation_residual_gate_px "
            "and runs the final optimization from the first-pass deltas."
        ),
    )
    parser.add_argument("--observation_residual_gate_px", type=float, default=600.0)
    parser.add_argument("--optimizer_residual_clip_px", type=float, default=500.0)
    parser.add_argument("--delta_translation_sigma_m", type=float, default=0.20)
    parser.add_argument("--delta_rotation_sigma_deg", type=float, default=5.0)
    parser.add_argument("--camera_delta_max_rotation_step_deg", type=float, default=1.0)
    parser.add_argument("--camera_delta_max_translation_step_m", type=float, default=0.03)
    parser.add_argument("--frame_face_max_rotation_step_deg", type=float, default=5.0)
    parser.add_argument("--frame_face_max_translation_step_m", type=float, default=0.10)
    parser.add_argument("--intrinsics_focal_sigma_frac", type=float, default=0.01)
    parser.add_argument("--intrinsics_principal_sigma_px", type=float, default=8.0)
    parser.add_argument("--intrinsics_distortion_sigma", type=float, default=0.05)
    parser.add_argument("--intrinsics_max_focal_step_frac", type=float, default=0.002)
    parser.add_argument("--intrinsics_max_principal_step_px", type=float, default=1.0)
    parser.add_argument("--intrinsics_max_distortion_step", type=float, default=0.01)
    parser.add_argument("--intrinsics_max_total_focal_delta_frac", type=float, default=0.0)
    parser.add_argument("--intrinsics_max_total_principal_delta_px", type=float, default=0.0)
    parser.add_argument("--intrinsics_max_total_distortion_delta", type=float, default=0.0)
    parser.add_argument("--intrinsics_block_iterations", type=int, default=4)
    parser.add_argument("--tower_first_tag_id", type=int, default=0)
    parser.add_argument("--tower_face_id_stride", type=int, default=32)
    parser.add_argument("--tower_tag_columns", type=int, default=2)
    parser.add_argument("--tower_tag_rows", type=int, default=16)
    parser.add_argument("--tower_tag_size_m", type=float, default=0.08)
    parser.add_argument("--tower_tag_spacing_m", type=float, default=0.02)
    parser.add_argument("--tower_tag_rotation_degrees", type=int, default=180)


def validate_args(args):
    if args.outer_iterations < 0:
        raise ValueError("--outer_iterations must be non-negative")
    if args.block_iterations <= 0:
        raise ValueError("--block_iterations must be positive")
    if args.min_pnp_points < 4:
        raise ValueError("--min_pnp_points must be at least 4")
    if args.pnp_ransac_iterations <= 0:
        raise ValueError("--pnp_ransac_iterations must be positive")
    if args.pnp_ransac_threshold_px <= 0:
        raise ValueError("--pnp_ransac_threshold_px must be positive")
    if args.min_frame_face_observations < 4:
        raise ValueError("--min_frame_face_observations must be at least 4")
    if args.min_camera_observations_for_delta < 0:
        raise ValueError("--min_camera_observations_for_delta must be non-negative")
    if args.initial_observation_residual_gate_px is not None and args.initial_observation_residual_gate_px < 0:
        raise ValueError("--initial_observation_residual_gate_px must be non-negative")
    if args.observation_residual_gate_px < 0:
        raise ValueError("--observation_residual_gate_px must be non-negative")
    if args.delta_translation_sigma_m <= 0:
        raise ValueError("--delta_translation_sigma_m must be positive")
    if args.delta_rotation_sigma_deg <= 0:
        raise ValueError("--delta_rotation_sigma_deg must be positive")
    if args.camera_delta_max_rotation_step_deg <= 0:
        raise ValueError("--camera_delta_max_rotation_step_deg must be positive")
    if args.camera_delta_max_translation_step_m <= 0:
        raise ValueError("--camera_delta_max_translation_step_m must be positive")
    if args.frame_face_max_rotation_step_deg <= 0:
        raise ValueError("--frame_face_max_rotation_step_deg must be positive")
    if args.frame_face_max_translation_step_m <= 0:
        raise ValueError("--frame_face_max_translation_step_m must be positive")
    if args.tower_first_tag_id != 0:
        raise ValueError("This prototype expects face_id = tag_id // 32, so --tower_first_tag_id must be 0")
    if args.tower_face_id_stride != 32:
        raise ValueError("This prototype expects face_id = tag_id // 32, so --tower_face_id_stride must be 32")
    if args.tower_tag_columns != 2 or args.tower_tag_rows != 16:
        raise ValueError("This prototype expects a 2x16 tag layout per face")
    if args.tower_tag_size_m <= 0 or args.tower_tag_spacing_m < 0:
        raise ValueError("--tower_tag_size_m must be positive and --tower_tag_spacing_m must be non-negative")
    if args.tower_tag_rotation_degrees != 180:
        raise ValueError("This prototype expects --tower_tag_rotation_degrees 180")
    if args.intrinsics_refine_mode != "fixed":
        if args.intrinsics_focal_sigma_frac <= 0:
            raise ValueError("--intrinsics_focal_sigma_frac must be positive")
        if args.intrinsics_principal_sigma_px <= 0:
            raise ValueError("--intrinsics_principal_sigma_px must be positive")
    if args.intrinsics_refine_mode == "per_camera_opencv5" and args.intrinsics_distortion_sigma <= 0:
        raise ValueError("--intrinsics_distortion_sigma must be positive")


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    cv2 = load_cv2_or_exit()
    dataset = base.read_dataset(args.dataset)
    manifest = base.read_manifest(args.manifest, dataset["camera_count"])
    layout = tower_layout_from_args(args)
    intrinsics = base.collect_intrinsics(args.intrinsics_dir, manifest, dataset["image_sizes"], args.intrinsics_mode)
    camera_priors, prior_alignment = base.build_pose_yaml_prior(manifest, args.camera_prior_pose_yaml)

    observations_all, by_frame_face_all, _by_camera_all = build_frame_face_observations(dataset, layout)
    frame_face_poses_initial, quality = initialize_frame_face_poses(
        cv2,
        by_frame_face_all,
        camera_priors,
        intrinsics,
        args)
    if not frame_face_poses_initial:
        raise ValueError("No frame-face plane pose could be initialized from solvePnP.")

    observation_gate_stages = []
    initial_pass = None
    if args.initial_observation_residual_gate_px is not None:
        initial_observations, initial_gate = filter_active_observations(
            observations_all,
            camera_priors,
            frame_face_poses_initial,
            intrinsics,
            args.initial_observation_residual_gate_px)
        initial_by_frame_face, initial_by_camera = rebuild_observation_groups(
            initial_observations,
            dataset["camera_count"])
        if not initial_observations:
            raise ValueError("No observations remain after initial frame-face residual gating.")
        initial_optimized = optimize_bundle(
            initial_by_frame_face,
            initial_by_camera,
            camera_priors,
            dict(frame_face_poses_initial),
            intrinsics,
            args)
        initial_pass = {
            "observations": initial_observations,
            "by_frame_face": initial_by_frame_face,
            "by_camera": initial_by_camera,
            "optimized": initial_optimized,
        }
        initial_gate["stage"] = "initial"
        observation_gate_stages.append(initial_gate)
        gate_camera_poses = initial_optimized["camera_poses"]
        gate_frame_face_poses = initial_optimized["frame_face_poses"]
        gate_intrinsics = initial_optimized["intrinsics"]
    else:
        gate_camera_poses = camera_priors
        gate_frame_face_poses = frame_face_poses_initial
        gate_intrinsics = intrinsics

    observations, observation_gate = filter_active_observations(
        observations_all,
        gate_camera_poses,
        gate_frame_face_poses,
        gate_intrinsics,
        args.observation_residual_gate_px)
    observation_gate["stage"] = "final"
    observation_gate_stages.append(observation_gate)
    by_frame_face, by_camera = rebuild_observation_groups(observations, dataset["camera_count"])

    if not observations:
        raise ValueError("No observations remain after frame-face initialization/gating.")

    before = summarize_residuals(projection_residuals(
        observations,
        gate_camera_poses,
        gate_frame_face_poses,
        gate_intrinsics))

    optimized = optimize_bundle(
        by_frame_face,
        by_camera,
        camera_priors,
        dict(gate_frame_face_poses),
        intrinsics,
        args,
        initial_deltas=initial_pass["optimized"]["deltas"] if initial_pass else None,
        initial_per_camera_intrinsics_delta=(
            initial_pass["optimized"]["per_camera_intrinsics_delta"] if initial_pass else None
        ))
    refined_camera_poses = optimized["camera_poses"]
    refined_frame_face_poses = optimized["frame_face_poses"]
    refined_intrinsics = optimized["intrinsics"]
    deltas = optimized["deltas"]
    active_cameras = optimized["active_cameras"]

    after = summarize_residuals(projection_residuals(
        observations,
        refined_camera_poses,
        refined_frame_face_poses,
        refined_intrinsics))

    base.write_pose_yaml(args.output_dir / "camera_tr_rig_prior.yaml", camera_priors)
    base.write_pose_yaml(args.output_dir / "camera_tr_rig_delta_refined.yaml", refined_camera_poses)
    base.write_pose_yaml(args.output_dir / "camera_delta_from_prior.yaml", deltas)
    write_frame_face_pose_yaml(
        args.output_dir / "rig_tr_frame_face.yaml",
        refined_frame_face_poses,
        "rig_tr_frame_face")
    write_frame_face_pose_yaml(
        args.output_dir / "frame_face_tr_rig.yaml",
        {key: base.invert_pose(pose) for key, pose in refined_frame_face_poses.items()},
        "frame_face_tr_rig")
    write_frame_face_index_tsv(
        diagnostics_dir / "frame_face_pose_index.tsv",
        refined_frame_face_poses)
    base.write_intrinsics_dir(args.output_dir / "intrinsics_prior", manifest, intrinsics)
    base.write_intrinsics_dir(args.output_dir / "intrinsics_refined", manifest, refined_intrinsics)

    camera_rows = camera_reprojection_rows(
        manifest,
        by_camera,
        camera_priors,
        refined_camera_poses,
        frame_face_poses_initial,
        refined_frame_face_poses,
        intrinsics,
        refined_intrinsics)
    write_camera_reprojection_tsv(diagnostics_dir / "camera_reprojection.tsv", camera_rows)
    write_observation_residuals_tsv(
        diagnostics_dir / "observation_residuals.tsv",
        manifest,
        observations,
        refined_camera_poses,
        refined_frame_face_poses,
        refined_intrinsics)
    quality_rows = frame_face_quality_rows(
        by_frame_face,
        quality,
        camera_priors,
        refined_camera_poses,
        frame_face_poses_initial,
        refined_frame_face_poses,
        intrinsics,
        refined_intrinsics)
    write_frame_face_quality_tsv(diagnostics_dir / "frame_face_quality.tsv", quality_rows)
    write_camera_delta_tsv(diagnostics_dir / "camera_delta.tsv", manifest, deltas, by_camera, active_cameras)

    inactive_camera_ids = [
        manifest[i]["camera_id"]
        for i, active in enumerate(active_cameras)
        if not active
    ]
    summary = {
        "inputs": {
            "dataset": str(args.dataset),
            "manifest": str(args.manifest),
            "camera_prior_pose_yaml": str(args.camera_prior_pose_yaml),
            "intrinsics_dir": str(args.intrinsics_dir) if args.intrinsics_dir else "",
        },
        "settings": {
            "model": "independent_plane_pose_per_frame_face",
            "intrinsics_mode": args.intrinsics_mode,
            "intrinsics_refine_mode": args.intrinsics_refine_mode,
            "outer_iterations": args.outer_iterations,
            "block_iterations": args.block_iterations,
            "min_pnp_points": args.min_pnp_points,
            "pnp_ransac": bool(args.pnp_ransac),
            "pnp_ransac_threshold_px": args.pnp_ransac_threshold_px,
            "max_pnp_median_error_px": args.max_pnp_median_error_px,
            "min_frame_face_observations": args.min_frame_face_observations,
            "min_camera_observations_for_delta": args.min_camera_observations_for_delta,
            "initial_observation_residual_gate_px": args.initial_observation_residual_gate_px,
            "observation_residual_gate_px": args.observation_residual_gate_px,
            "optimizer_residual_clip_px": args.optimizer_residual_clip_px,
            "tower_first_tag_id": args.tower_first_tag_id,
            "tower_face_id_stride": args.tower_face_id_stride,
            "tower_tag_columns": args.tower_tag_columns,
            "tower_tag_rows": args.tower_tag_rows,
            "tower_tag_size_m": args.tower_tag_size_m,
            "tower_tag_spacing_m": args.tower_tag_spacing_m,
            "tower_tag_rotation_degrees": args.tower_tag_rotation_degrees,
        },
        "prior_alignment": prior_alignment,
        "observation_gate": observation_gate,
        "observation_gate_stages": observation_gate_stages,
        "observations": {
            "raw": len(observations_all),
            "used": len(observations),
        },
        "frame_faces": {
            "total_observed": len(by_frame_face_all),
            "initialized": len(frame_face_poses_initial),
            "used": len(by_frame_face),
        },
        "cameras": {
            "total": dataset["camera_count"],
            "active_delta": int(sum(active_cameras)),
            "inactive_delta": inactive_camera_ids,
        },
        "outputs": {
            "camera_tr_rig_delta_refined_yaml": str(args.output_dir / "camera_tr_rig_delta_refined.yaml"),
            "rig_tr_frame_face_yaml": str(args.output_dir / "rig_tr_frame_face.yaml"),
            "frame_face_tr_rig_yaml": str(args.output_dir / "frame_face_tr_rig.yaml"),
            "summary_json": str(args.output_dir / "summary.json"),
            "camera_reprojection_tsv": str(diagnostics_dir / "camera_reprojection.tsv"),
            "observation_residuals_tsv": str(diagnostics_dir / "observation_residuals.tsv"),
            "frame_face_quality_tsv": str(diagnostics_dir / "frame_face_quality.tsv"),
        },
        "residual_before": before,
        "residual_after": after,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
