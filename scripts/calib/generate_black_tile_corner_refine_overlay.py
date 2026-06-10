#!/usr/bin/env python3
"""Generate a diagnostic overlay for black-tile outer-corner refinement."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
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


def bilinear_sample(image, x, y):
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x >= width - 1 or y >= height - 1:
        return None
    x0 = int(math.floor(float(x)))
    y0 = int(math.floor(float(y)))
    dx = float(x) - x0
    dy = float(y) - y0
    v00 = float(image[y0, x0])
    v10 = float(image[y0, x0 + 1])
    v01 = float(image[y0 + 1, x0])
    v11 = float(image[y0 + 1, x0 + 1])
    return (
        (1.0 - dx) * (1.0 - dy) * v00
        + dx * (1.0 - dy) * v10
        + (1.0 - dx) * dy * v01
        + dx * dy * v11
    )


def fit_line(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    center = pts.mean(axis=0)
    centered = pts - center
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    direction /= norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    normal /= max(1e-12, float(np.linalg.norm(normal)))
    offset = -float(normal @ center)
    residuals = np.abs(pts @ normal + offset)
    return {
        "normal": normal,
        "offset": offset,
        "point_count": int(pts.shape[0]),
        "mean_residual": float(residuals.mean()) if residuals.size else 0.0,
        "median_residual": float(np.median(residuals)) if residuals.size else 0.0,
        "max_residual": float(residuals.max()) if residuals.size else 0.0,
    }


def robust_fit_line(points, max_residual_px):
    if len(points) < 2:
        return None
    line = fit_line(points)
    if line is None:
        return None
    pts = np.asarray(points, dtype=np.float64)
    residuals = np.abs(pts @ line["normal"] + line["offset"])
    median = float(np.median(residuals)) if residuals.size else 0.0
    threshold = max(float(max_residual_px), 2.5 * median)
    inliers = pts[residuals <= threshold]
    if inliers.shape[0] >= max(2, int(0.45 * pts.shape[0])):
        line = fit_line(inliers)
        if line is not None:
            line["raw_point_count"] = int(pts.shape[0])
            line["inlier_count"] = int(inliers.shape[0])
    return line


def intersect_lines(line_a, line_b):
    matrix = np.vstack([line_a["normal"], line_b["normal"]])
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-6:
        return None
    rhs = -np.asarray([line_a["offset"], line_b["offset"]], dtype=np.float64)
    point = np.linalg.solve(matrix, rhs)
    if not np.all(np.isfinite(point)):
        return None
    return point


def distance_to_line(point, line_a, line_b):
    point = np.asarray(point, dtype=np.float64)
    line_a = np.asarray(line_a, dtype=np.float64)
    line_b = np.asarray(line_b, dtype=np.float64)
    direction = line_b - line_a
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return float("inf")
    delta = point - line_a
    cross_value = float(direction[0] * delta[1] - direction[1] * delta[0])
    return float(abs(cross_value) / length)


def sample_edge_points(
        image,
        p0,
        p1,
        inside_normal,
        search_radius_px,
        sample_spacing_px,
        gradient_step_px,
        min_gradient,
        trim_fraction):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    edge = p1 - p0
    length = float(np.linalg.norm(edge))
    if length < 8.0:
        return [], []
    inside_normal = np.asarray(inside_normal, dtype=np.float64)
    inside_normal /= max(1e-12, float(np.linalg.norm(inside_normal)))
    sample_count = max(8, min(140, int(length / max(0.5, sample_spacing_px))))
    offset_step = max(0.25, min(1.0, 0.5 * float(gradient_step_px)))
    offsets = np.arange(-float(search_radius_px), float(search_radius_px) + 0.5 * offset_step, offset_step)
    points = []
    scores = []
    lo = float(trim_fraction)
    hi = 1.0 - float(trim_fraction)
    for sample_index in range(sample_count):
        alpha = (sample_index + 0.5) / sample_count
        if alpha < lo or alpha > hi:
            continue
        center = p0 * (1.0 - alpha) + p1 * alpha
        best_point = None
        best_score = -1e18
        for offset in offsets:
            candidate = center + inside_normal * float(offset)
            outside = candidate - inside_normal * float(gradient_step_px)
            inside = candidate + inside_normal * float(gradient_step_px)
            value_outside = bilinear_sample(image, outside[0], outside[1])
            value_inside = bilinear_sample(image, inside[0], inside[1])
            if value_outside is None or value_inside is None:
                continue
            # The black tile exterior is white background and the tile interior is black.
            score = (value_outside - value_inside) / max(1e-6, 2.0 * float(gradient_step_px))
            if score > best_score:
                best_score = score
                best_point = candidate
        if best_point is not None and best_score >= float(min_gradient):
            points.append(best_point)
            scores.append(float(best_score))
    return points, scores


def refine_black_tile_polygon(
        image,
        initial_corners,
        search_radius_px,
        sample_spacing_px,
        gradient_step_px,
        min_gradient,
        min_edge_points,
        max_line_residual_px,
        max_corner_shift_px,
        trim_fraction):
    initial = np.asarray(initial_corners, dtype=np.float64)
    if initial.shape != (4, 2):
        return None
    center = initial.mean(axis=0)
    lines = []
    edge_points = []
    edge_scores = []
    for edge_index in range(4):
        p0 = initial[edge_index]
        p1 = initial[(edge_index + 1) % 4]
        edge = p1 - p0
        midpoint = 0.5 * (p0 + p1)
        normal = np.asarray([-edge[1], edge[0]], dtype=np.float64)
        if float(normal @ (center - midpoint)) < 0:
            normal = -normal
        points, scores = sample_edge_points(
            image,
            p0,
            p1,
            normal,
            search_radius_px,
            sample_spacing_px,
            gradient_step_px,
            min_gradient,
            trim_fraction)
        edge_points.append(points)
        edge_scores.append(scores)
        if len(points) < int(min_edge_points):
            lines.append(None)
            continue
        lines.append(robust_fit_line(points, max_line_residual_px))

    refined = initial.copy()
    valid_corners = 0
    corner_shifts = []
    for corner_index in range(4):
        previous_line = lines[(corner_index - 1) % 4]
        next_line = lines[corner_index]
        if previous_line is None or next_line is None:
            corner_shifts.append(None)
            continue
        point = intersect_lines(previous_line, next_line)
        if point is None:
            corner_shifts.append(None)
            continue
        shift = float(np.linalg.norm(point - initial[corner_index]))
        if shift <= float(max_corner_shift_px):
            refined[corner_index] = point
            valid_corners += 1
            corner_shifts.append(shift)
        else:
            corner_shifts.append(None)

    all_scores = [score for scores in edge_scores for score in scores]
    valid_edges = sum(1 for line in lines if line is not None)
    return {
        "initial": initial,
        "refined": refined,
        "lines": lines,
        "edge_points": edge_points,
        "edge_scores": edge_scores,
        "valid_edges": int(valid_edges),
        "valid_corners": int(valid_corners),
        "mean_gradient": float(np.mean(all_scores)) if all_scores else None,
        "median_gradient": float(np.median(all_scores)) if all_scores else None,
        "corner_shifts": corner_shifts,
    }


def refine_black_tile_corners_with_subpix(
        image,
        initial_corners,
        search_radius_px,
        max_corner_shift_px,
        corner_quality_level,
        corner_min_distance_px,
        corner_block_size,
        corner_subpix_window_px,
        corner_geometric_weight):
    initial = np.asarray(initial_corners, dtype=np.float64)
    if initial.shape != (4, 2):
        return None
    refined = initial.copy()
    candidates_by_corner = []
    valid_corners = 0
    corner_shifts = []
    height, width = image.shape[:2]
    work = cv2.GaussianBlur(image, (3, 3), 0)
    for corner_index, corner in enumerate(initial):
        radius = int(math.ceil(float(search_radius_px)))
        x0 = max(0, int(math.floor(float(corner[0]))) - radius)
        y0 = max(0, int(math.floor(float(corner[1]))) - radius)
        x1 = min(width, int(math.floor(float(corner[0]))) + radius + 1)
        y1 = min(height, int(math.floor(float(corner[1]))) + radius + 1)
        roi = work[y0:y1, x0:x1]
        candidates = []
        if roi.shape[0] >= 8 and roi.shape[1] >= 8:
            points = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=32,
                qualityLevel=float(corner_quality_level),
                minDistance=float(corner_min_distance_px),
                blockSize=int(corner_block_size),
                useHarrisDetector=True,
                k=0.04)
            if points is not None:
                points = points.reshape(-1, 2).astype(np.float32)
                window = int(corner_subpix_window_px)
                cv2.cornerSubPix(
                    roi,
                    points.reshape(-1, 1, 2),
                    (window, window),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
                previous_corner = initial[(corner_index - 1) % 4]
                next_corner = initial[(corner_index + 1) % 4]
                for point in points:
                    image_point = point.astype(np.float64) + np.asarray([x0, y0], dtype=np.float64)
                    shift = float(np.linalg.norm(image_point - corner))
                    if shift > float(max_corner_shift_px):
                        continue
                    edge_error = (
                        distance_to_line(image_point, corner, previous_corner)
                        + distance_to_line(image_point, corner, next_corner))
                    score = edge_error + float(corner_geometric_weight) * shift
                    candidates.append({
                        "point": image_point,
                        "shift": shift,
                        "edge_error": float(edge_error),
                        "score": float(score),
                    })
        candidates.sort(key=lambda value: value["score"])
        candidates_by_corner.append(candidates)
        if candidates:
            best = candidates[0]
            refined[corner_index] = best["point"]
            valid_corners += 1
            corner_shifts.append(best["shift"])
        else:
            corner_shifts.append(None)
    if valid_corners >= 4:
        edge_lengths = [
            float(np.linalg.norm(refined[(index + 1) % 4] - refined[index]))
            for index in range(4)]
        initial_edge_lengths = [
            float(np.linalg.norm(initial[(index + 1) % 4] - initial[index]))
            for index in range(4)]
    else:
        edge_lengths = []
        initial_edge_lengths = []
    return {
        "initial": initial,
        "refined": refined,
        "candidates_by_corner": candidates_by_corner,
        "valid_edges": 0,
        "valid_corners": int(valid_corners),
        "mean_gradient": None,
        "median_gradient": None,
        "corner_shifts": corner_shifts,
        "edge_lengths": edge_lengths,
        "initial_edge_lengths": initial_edge_lengths,
    }


def refine_black_tile_corners_from_scaled_detector_box(
        image,
        detector_corners,
        red_box_scale_min,
        red_box_scale_max,
        red_box_target_scale,
        corner_roi_margin_px,
        max_lateral_error_px,
        corner_quality_level,
        corner_min_distance_px,
        corner_block_size,
        corner_subpix_window_px,
        scale_weight_px,
        use_edge_support=False,
        edge_canny_low=40.0,
        edge_canny_high=120.0,
        edge_arm_length_px=22.0,
        edge_arm_width_px=2.5,
        edge_scale_step=0.005,
        edge_lateral_step_px=1.5,
        edge_min_arm_support=0.16,
        edge_score_weight_px=18.0):
    detector = np.asarray(detector_corners, dtype=np.float64)
    if detector.shape != (4, 2):
        return None
    center = detector.mean(axis=0)
    refined = detector.copy()
    candidates_by_corner = []
    valid_corners = 0
    corner_shifts = []
    corner_scales = []
    corner_sources = []
    height, width = image.shape[:2]
    work = cv2.GaussianBlur(image, (3, 3), 0)
    edge_image = None
    if use_edge_support:
        edge_image = cv2.Canny(work, float(edge_canny_low), float(edge_canny_high), apertureSize=3, L2gradient=True)

    def local_edge_support(point, dir_a, dir_b):
        if edge_image is None:
            return 0.0, 0.0
        point = np.asarray(point, dtype=np.float64)
        supports = []
        for direction in (dir_a, dir_b):
            direction = np.asarray(direction, dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                supports.append(0.0)
                continue
            direction = direction / norm
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
            hits = 0
            total = 0
            arm_samples = max(6, int(float(edge_arm_length_px) / 2.0))
            side_samples = max(3, int(2.0 * float(edge_arm_width_px)) + 1)
            for arm_index in range(1, arm_samples + 1):
                distance = float(edge_arm_length_px) * arm_index / arm_samples
                center_point = point + direction * distance
                for side_index in range(side_samples):
                    offset = -float(edge_arm_width_px) + 2.0 * float(edge_arm_width_px) * side_index / max(1, side_samples - 1)
                    sample = center_point + normal * offset
                    x = int(round(float(sample[0])))
                    y = int(round(float(sample[1])))
                    if 0 <= x < width and 0 <= y < height:
                        total += 1
                        if edge_image[y, x] > 0:
                            hits += 1
            supports.append(float(hits) / max(1, total))
        return supports[0], supports[1]

    for corner_index, corner in enumerate(detector):
        ray = corner - center
        ray_length = float(np.linalg.norm(ray))
        if ray_length <= 1e-6:
            candidates_by_corner.append([])
            corner_shifts.append(None)
            corner_scales.append(None)
            corner_sources.append(None)
            continue
        band_points = [
            center + ray * float(red_box_scale_min),
            center + ray * float(red_box_scale_max),
        ]
        margin = float(corner_roi_margin_px)
        x0 = max(0, int(math.floor(min(point[0] for point in band_points) - margin)))
        y0 = max(0, int(math.floor(min(point[1] for point in band_points) - margin)))
        x1 = min(width, int(math.ceil(max(point[0] for point in band_points) + margin + 1.0)))
        y1 = min(height, int(math.ceil(max(point[1] for point in band_points) + margin + 1.0)))
        roi = work[y0:y1, x0:x1]
        candidates = []
        previous_corner = detector[(corner_index - 1) % 4]
        next_corner = detector[(corner_index + 1) % 4]
        direction_previous = previous_corner - corner
        direction_next = next_corner - corner
        if roi.shape[0] >= 8 and roi.shape[1] >= 8:
            points = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=48,
                qualityLevel=float(corner_quality_level),
                minDistance=float(corner_min_distance_px),
                blockSize=int(corner_block_size),
                useHarrisDetector=True,
                k=0.04)
            if points is not None:
                points = points.reshape(-1, 2).astype(np.float32)
                window = int(corner_subpix_window_px)
                cv2.cornerSubPix(
                    roi,
                    points.reshape(-1, 1, 2),
                    (window, window),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
                ray_norm_sq = float(ray @ ray)
                for point in points:
                    image_point = point.astype(np.float64) + np.asarray([x0, y0], dtype=np.float64)
                    delta = image_point - center
                    scale = float((delta @ ray) / ray_norm_sq)
                    if scale < float(red_box_scale_min) or scale > float(red_box_scale_max):
                        continue
                    projected = center + ray * scale
                    lateral_error = float(np.linalg.norm(image_point - projected))
                    if lateral_error > float(max_lateral_error_px):
                        continue
                    target_error = abs(scale - float(red_box_target_scale)) * ray_length
                    edge_a, edge_b = local_edge_support(image_point, direction_previous, direction_next)
                    edge_penalty = 0.0
                    if use_edge_support:
                        edge_penalty = float(edge_score_weight_px) * max(0.0, 2.0 - edge_a - edge_b)
                    score = lateral_error + float(scale_weight_px) * target_error + edge_penalty
                    candidates.append({
                        "point": image_point,
                        "scale": scale,
                        "shift": float(np.linalg.norm(image_point - corner)),
                        "lateral_error": lateral_error,
                        "edge_support_a": edge_a,
                        "edge_support_b": edge_b,
                        "source": "corner",
                        "score": float(score),
                    })
        if use_edge_support and edge_image is not None:
            ray_norm_sq = float(ray @ ray)
            ray_unit = ray / max(1e-9, ray_length)
            lateral_unit = np.asarray([-ray_unit[1], ray_unit[0]], dtype=np.float64)
            scale_values = np.arange(
                float(red_box_scale_min),
                float(red_box_scale_max) + 0.5 * float(edge_scale_step),
                float(edge_scale_step))
            lateral_values = np.arange(
                -float(max_lateral_error_px),
                float(max_lateral_error_px) + 0.5 * float(edge_lateral_step_px),
                float(edge_lateral_step_px))
            for scale in scale_values:
                scale = float(scale)
                base_point = center + ray * scale
                for lateral_offset in lateral_values:
                    image_point = base_point + lateral_unit * float(lateral_offset)
                    if image_point[0] < 0 or image_point[1] < 0 or image_point[0] >= width or image_point[1] >= height:
                        continue
                    edge_a, edge_b = local_edge_support(image_point, direction_previous, direction_next)
                    if edge_a < float(edge_min_arm_support) or edge_b < float(edge_min_arm_support):
                        continue
                    delta = image_point - center
                    actual_scale = float((delta @ ray) / ray_norm_sq)
                    if actual_scale < float(red_box_scale_min) or actual_scale > float(red_box_scale_max):
                        continue
                    projected = center + ray * actual_scale
                    lateral_error = float(np.linalg.norm(image_point - projected))
                    if lateral_error > float(max_lateral_error_px):
                        continue
                    target_error = abs(actual_scale - float(red_box_target_scale)) * ray_length
                    edge_penalty = float(edge_score_weight_px) * max(0.0, 2.0 - edge_a - edge_b)
                    score = lateral_error + float(scale_weight_px) * target_error + edge_penalty
                    candidates.append({
                        "point": image_point,
                        "scale": actual_scale,
                        "shift": float(np.linalg.norm(image_point - corner)),
                        "lateral_error": lateral_error,
                        "edge_support_a": edge_a,
                        "edge_support_b": edge_b,
                        "source": "edge",
                        "score": float(score),
                    })
        candidates.sort(key=lambda value: value["score"])
        candidates_by_corner.append(candidates)
        if candidates:
            best = candidates[0]
            refined[corner_index] = best["point"]
            valid_corners += 1
            corner_shifts.append(best["shift"])
            corner_scales.append(best["scale"])
            corner_sources.append(best.get("source"))
        else:
            corner_shifts.append(None)
            corner_scales.append(None)
            corner_sources.append(None)
    if valid_corners >= 4:
        edge_lengths = [
            float(np.linalg.norm(refined[(index + 1) % 4] - refined[index]))
            for index in range(4)]
        detector_edge_lengths = [
            float(np.linalg.norm(detector[(index + 1) % 4] - detector[index]))
            for index in range(4)]
    else:
        edge_lengths = []
        detector_edge_lengths = []
    return {
        "initial": detector,
        "refined": refined,
        "candidates_by_corner": candidates_by_corner,
        "valid_edges": 0,
        "valid_corners": int(valid_corners),
        "mean_gradient": None,
        "median_gradient": None,
        "corner_shifts": corner_shifts,
        "corner_scales": corner_scales,
        "corner_sources": corner_sources,
        "edge_lengths": edge_lengths,
        "initial_edge_lengths": detector_edge_lengths,
    }


def local_center(face, tag_id, pitch_m):
    local = int(tag_id) - int(face) * 32
    row = local // 2
    col = local % 2
    return np.asarray([(col - 0.5) * pitch_m, (row - 7.5) * pitch_m], dtype=np.float32)


def local_tile_corners(face, tag_id, pitch_m, tile_size_m):
    center = local_center(face, tag_id, pitch_m)
    half = 0.5 * float(tile_size_m)
    return np.asarray([
        [center[0] - half, center[1] - half],
        [center[0] + half, center[1] - half],
        [center[0] + half, center[1] + half],
        [center[0] - half, center[1] + half],
    ], dtype=np.float32)


def draw_poly(image, points, color, thickness, radius=0):
    pts = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.polylines(image, [pts.reshape(-1, 1, 2)], True, color, int(thickness), cv2.LINE_AA)
    if radius > 0:
        for point in pts:
            cv2.circle(image, tuple(point), int(radius), color, -1, cv2.LINE_AA)


def draw_legend(image, rows):
    y = 42
    for text, color in rows:
        cv2.putText(image, text, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, text, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.88, color, 2, cv2.LINE_AA)
        y += 38


def read_image_path(manifest_row, filename):
    for key in ("source_dir", "image_dir", "original_source_dir"):
        value = manifest_row.get(key)
        if value:
            candidate = Path(value) / filename
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not resolve image path for {manifest_row.get('camera_id')} {filename}")


def run(args):
    dataset = base.read_dataset(args.dataset)
    manifest = base.read_manifest(args.manifest, dataset["camera_count"])
    imageset = dataset["imagesets"][args.frame_index]
    camera_row = manifest[args.camera_index]
    image_path = read_image_path(camera_row, imageset["filename"])
    color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if color is None or gray is None:
        raise SystemExit(f"Could not read image: {image_path}")

    by_tag = defaultdict(dict)
    for x, y, feature_id in imageset["features"][args.camera_index]:
        tag_id, corner_id, face_id = base.observation_feature_fields(feature_id)
        if face_id is None:
            continue
        by_tag[(int(face_id), int(tag_id))][int(corner_id)] = np.asarray([float(x), float(y)], dtype=np.float32)
    complete = {key: value for key, value in by_tag.items() if len(value) == 4}
    by_face = defaultdict(list)
    for (face_id, tag_id), corners in complete.items():
        pts = np.asarray([corners[i] for i in range(4)], dtype=np.float32)
        by_face[face_id].append({
            "tag_id": int(tag_id),
            "detected_corners": pts,
            "detected_center": pts.mean(axis=0),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay = color.copy()
    all_points = []
    rows = []
    face_rows = []
    refined_count = 0

    for face_id, items in sorted(by_face.items()):
        homography = None
        inliers = None
        if len(items) >= args.min_face_tags:
            src = np.asarray(
                [local_center(face_id, item["tag_id"], args.tag_center_pitch_m) for item in items],
                dtype=np.float32)
            dst = np.asarray([item["detected_center"] for item in items], dtype=np.float32)
            homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, args.homography_ransac_px)
        if homography is None and args.refine_method not in ("red-scale", "red-scale-edge"):
            continue
        inlier_count = int(inliers.sum()) if inliers is not None else None
        face_rows.append({
            "face_id": face_id,
            "complete_tags": len(items),
            "homography_inliers": inlier_count,
            "homography_used": homography is not None,
        })
        for item in sorted(items, key=lambda value: value["tag_id"]):
            tag_id = item["tag_id"]
            detected = item["detected_corners"]
            if homography is not None:
                local = local_tile_corners(face_id, tag_id, args.tag_center_pitch_m, args.black_tile_size_m)
                initial = cv2.perspectiveTransform(local.reshape(-1, 1, 2), homography).reshape(-1, 2)
            else:
                detected_center = detected.mean(axis=0)
                initial = detected_center + (detected - detected_center) * float(args.red_box_target_scale)
            if args.refine_method == "edge":
                refined = refine_black_tile_polygon(
                    gray,
                    initial,
                    args.search_radius_px,
                    args.sample_spacing_px,
                    args.gradient_step_px,
                    args.min_gradient,
                    args.min_edge_points,
                    args.max_line_residual_px,
                    args.max_corner_shift_px,
                    args.trim_fraction)
            elif args.refine_method == "corner":
                refined = refine_black_tile_corners_with_subpix(
                    gray,
                    initial,
                    args.corner_search_radius_px,
                    args.max_corner_shift_px,
                    args.corner_quality_level,
                    args.corner_min_distance_px,
                    args.corner_block_size,
                    args.corner_subpix_window_px,
                    args.corner_geometric_weight)
            else:
                refined = refine_black_tile_corners_from_scaled_detector_box(
                    gray,
                    detected,
                    args.red_box_scale_min,
                    args.red_box_scale_max,
                    args.red_box_target_scale,
                    args.corner_roi_margin_px,
                    args.max_lateral_error_px,
                    args.corner_quality_level,
                    args.corner_min_distance_px,
                    args.corner_block_size,
                    args.corner_subpix_window_px,
                    args.scale_weight_px,
                    use_edge_support=(args.refine_method == "red-scale-edge"),
                    edge_canny_low=args.edge_canny_low,
                    edge_canny_high=args.edge_canny_high,
                    edge_arm_length_px=args.edge_arm_length_px,
                    edge_arm_width_px=args.edge_arm_width_px,
                    edge_scale_step=args.edge_scale_step,
                    edge_lateral_step_px=args.edge_lateral_step_px,
                    edge_min_arm_support=args.edge_min_arm_support,
                    edge_score_weight_px=args.edge_score_weight_px)
            draw_poly(overlay, detected, (0, 0, 255), 2, 0)
            draw_poly(overlay, initial, (255, 255, 0), 2, 3)
            all_points.extend(detected.tolist())
            all_points.extend(initial.tolist())
            if refined is not None:
                if refined["valid_corners"] >= args.min_valid_corners:
                    refined_count += 1
                    draw_poly(overlay, refined["refined"], (0, 255, 0), 3, 5)
                    all_points.extend(refined["refined"].tolist())
                elif args.refine_method in ("corner", "red-scale"):
                    for candidates in refined.get("candidates_by_corner", []):
                        for candidate in candidates[:3]:
                            cv2.circle(
                                overlay,
                                tuple(np.round(candidate["point"]).astype(np.int32)),
                                3,
                                (255, 0, 255),
                                -1,
                                cv2.LINE_AA)
                            all_points.append(candidate["point"].tolist())
                if args.refine_method == "edge":
                    for edge_points in refined["edge_points"]:
                        for point in edge_points[::max(1, len(edge_points) // 24)]:
                            cv2.circle(
                                overlay,
                                tuple(np.round(point).astype(np.int32)),
                                2,
                                (0, 180, 255),
                                -1,
                                cv2.LINE_AA)
                shifts = [value for value in refined["corner_shifts"] if value is not None]
                scales = [value for value in refined.get("corner_scales", []) if value is not None]
                rows.append({
                    "face_id": face_id,
                    "tag_id": tag_id,
                    "valid_edges": refined["valid_edges"],
                    "valid_corners": refined["valid_corners"],
                    "mean_gradient": refined["mean_gradient"],
                    "median_gradient": refined["median_gradient"],
                    "mean_corner_shift_px": float(np.mean(shifts)) if shifts else None,
                    "max_corner_shift_px": float(np.max(shifts)) if shifts else None,
                    "mean_corner_scale": float(np.mean(scales)) if scales else None,
                    "min_corner_scale": float(np.min(scales)) if scales else None,
                    "max_corner_scale": float(np.max(scales)) if scales else None,
                    "corner_sources": ",".join(value if value is not None else "missing" for value in refined.get("corner_sources", [])),
                })
            center = np.round(item["detected_center"]).astype(np.int32)
            cv2.putText(
                overlay,
                str(tag_id),
                tuple(center + np.asarray([7, -7])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA)

    draw_legend(overlay, [
        ("red: OpenCV AprilTag inner detector corners", (0, 0, 255)),
        ("cyan: predicted 8cm black-tile outer corner initial guess", (255, 255, 0)),
        ("green: subpixel corner picked in red-box scale band", (0, 255, 0)),
        ("orange dots: edge samples used for line fitting", (0, 180, 255)),
        ("magenta dots: partial corner candidates when a full tag is rejected", (255, 0, 255)),
    ])

    full_path = args.output_dir / "black_tile_refined_overlay_full.jpg"
    crop_path = args.output_dir / "black_tile_refined_overlay_crop.jpg"
    cv2.imwrite(str(full_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if all_points:
        pts = np.asarray(all_points, dtype=np.float64)
        x0 = max(0, int(math.floor(float(pts[:, 0].min()))) - args.crop_margin_px)
        y0 = max(0, int(math.floor(float(pts[:, 1].min()))) - args.crop_margin_px)
        x1 = min(overlay.shape[1] - 1, int(math.ceil(float(pts[:, 0].max()))) + args.crop_margin_px)
        y1 = min(overlay.shape[0] - 1, int(math.ceil(float(pts[:, 1].max()))) + args.crop_margin_px)
        crop = overlay[y0:y1 + 1, x0:x1 + 1]
    else:
        crop = overlay
    scale = min(1.0, args.max_crop_width_px / max(1, crop.shape[1]))
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    metrics_path = args.output_dir / "black_tile_refined_corners.tsv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "face_id",
            "tag_id",
            "valid_edges",
            "valid_corners",
            "mean_gradient",
            "median_gradient",
            "mean_corner_shift_px",
            "max_corner_shift_px",
            "mean_corner_scale",
            "min_corner_scale",
            "max_corner_scale",
            "corner_sources",
        ]
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "frame_index": args.frame_index,
        "camera_index": args.camera_index,
        "camera_id": camera_row.get("camera_id", ""),
        "image_path": str(image_path),
        "complete_tags": len(complete),
        "refined_tags_min_valid_corners": refined_count,
        "black_tile_size_m": args.black_tile_size_m,
        "tag_center_pitch_m": args.tag_center_pitch_m,
        "settings": {
            "refine_method": args.refine_method,
            "search_radius_px": args.search_radius_px,
            "corner_search_radius_px": args.corner_search_radius_px,
            "sample_spacing_px": args.sample_spacing_px,
            "gradient_step_px": args.gradient_step_px,
            "min_gradient": args.min_gradient,
            "min_edge_points": args.min_edge_points,
            "max_line_residual_px": args.max_line_residual_px,
            "max_corner_shift_px": args.max_corner_shift_px,
            "trim_fraction": args.trim_fraction,
            "homography_ransac_px": args.homography_ransac_px,
            "corner_quality_level": args.corner_quality_level,
            "corner_min_distance_px": args.corner_min_distance_px,
            "corner_block_size": args.corner_block_size,
            "corner_subpix_window_px": args.corner_subpix_window_px,
            "corner_geometric_weight": args.corner_geometric_weight,
            "red_box_scale_min": args.red_box_scale_min,
            "red_box_scale_max": args.red_box_scale_max,
            "red_box_target_scale": args.red_box_target_scale,
            "corner_roi_margin_px": args.corner_roi_margin_px,
            "max_lateral_error_px": args.max_lateral_error_px,
            "scale_weight_px": args.scale_weight_px,
            "edge_canny_low": args.edge_canny_low,
            "edge_canny_high": args.edge_canny_high,
            "edge_arm_length_px": args.edge_arm_length_px,
            "edge_arm_width_px": args.edge_arm_width_px,
            "edge_scale_step": args.edge_scale_step,
            "edge_lateral_step_px": args.edge_lateral_step_px,
            "edge_min_arm_support": args.edge_min_arm_support,
            "edge_score_weight_px": args.edge_score_weight_px,
        },
        "face_rows": face_rows,
        "outputs": {
            "full_overlay": str(full_path),
            "crop_overlay": str(crop_path),
            "metrics_tsv": str(metrics_path),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    face_table = "".join(
        "<tr>"
        f"<td>{row['face_id']}</td>"
        f"<td>{row['complete_tags']}</td>"
        f"<td>{row['homography_inliers'] if row['homography_inliers'] is not None else 'n/a'}</td>"
        "</tr>"
        for row in face_rows)
    html_path = args.output_dir / "index.html"
    html_path.write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>Black Tile Gradient Corner Refine</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #f7f7f4; color: #222; }}
img {{ max-width: 100%; border: 1px solid #ccc; background: #111; margin: 10px 0 24px; }}
code {{ background: #eee; padding: 1px 4px; border-radius: 4px; }}
table {{ border-collapse: collapse; background: #fff; margin: 12px 0; }}
td, th {{ border-bottom: 1px solid #ddd; padding: 6px 9px; text-align: left; }}
</style>
<h1>Black Tile Gradient Corner Refine</h1>
<p>Frame <code>{args.frame_index}</code>, camera <code>{args.camera_index} / {html.escape(camera_row.get('camera_id', ''))}</code>, image <code>{html.escape(str(image_path))}</code>.</p>
<p>Red = OpenCV inner detector corners. Cyan = predicted 8cm black-tile outer corner initial guess. Green = subpixel corner picked in the red-box scale band. Orange dots = edge samples used only by the explicit edge mode. Magenta dots = partial corner candidates when a full tag is rejected.</p>
<p>Complete tags: <code>{len(complete)}</code>; refined tags with at least {args.min_valid_corners} valid corners: <code>{refined_count}</code>.</p>
<table><thead><tr><th>face</th><th>complete tags</th><th>homography inliers</th></tr></thead><tbody>{face_table}</tbody></table>
<h2>Zoom Crop</h2>
<img src="black_tile_refined_overlay_crop.jpg">
<h2>Full Image</h2>
<img src="black_tile_refined_overlay_full.jpg">
<p>Metrics TSV: <code>black_tile_refined_corners.tsv</code>. Summary JSON: <code>summary.json</code>.</p>
""", encoding="utf-8")

    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--frame-index", required=True, type=int)
    parser.add_argument("--camera-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--black-tile-size-m", type=float, default=0.08)
    parser.add_argument("--tag-center-pitch-m", type=float, default=0.10)
    parser.add_argument("--min-face-tags", type=int, default=4)
    parser.add_argument("--homography-ransac-px", type=float, default=5.0)
    parser.add_argument("--refine-method", choices=("red-scale-edge", "red-scale", "corner", "edge"), default="red-scale-edge")
    parser.add_argument("--search-radius-px", type=float, default=18.0)
    parser.add_argument("--corner-search-radius-px", type=float, default=30.0)
    parser.add_argument("--sample-spacing-px", type=float, default=3.0)
    parser.add_argument("--gradient-step-px", type=float, default=1.5)
    parser.add_argument("--min-gradient", type=float, default=8.0)
    parser.add_argument("--min-edge-points", type=int, default=8)
    parser.add_argument("--max-line-residual-px", type=float, default=1.5)
    parser.add_argument("--max-corner-shift-px", type=float, default=28.0)
    parser.add_argument("--trim-fraction", type=float, default=0.16)
    parser.add_argument("--corner-quality-level", type=float, default=0.01)
    parser.add_argument("--corner-min-distance-px", type=float, default=3.0)
    parser.add_argument("--corner-block-size", type=int, default=5)
    parser.add_argument("--corner-subpix-window-px", type=int, default=5)
    parser.add_argument("--corner-geometric-weight", type=float, default=0.25)
    parser.add_argument("--red-box-scale-min", type=float, default=1.0)
    parser.add_argument("--red-box-scale-max", type=float, default=1.25)
    parser.add_argument("--red-box-target-scale", type=float, default=1.18)
    parser.add_argument("--corner-roi-margin-px", type=float, default=8.0)
    parser.add_argument("--max-lateral-error-px", type=float, default=12.0)
    parser.add_argument("--scale-weight-px", type=float, default=0.8)
    parser.add_argument("--edge-canny-low", type=float, default=40.0)
    parser.add_argument("--edge-canny-high", type=float, default=120.0)
    parser.add_argument("--edge-arm-length-px", type=float, default=22.0)
    parser.add_argument("--edge-arm-width-px", type=float, default=2.5)
    parser.add_argument("--edge-scale-step", type=float, default=0.005)
    parser.add_argument("--edge-lateral-step-px", type=float, default=1.5)
    parser.add_argument("--edge-min-arm-support", type=float, default=0.16)
    parser.add_argument("--edge-score-weight-px", type=float, default=18.0)
    parser.add_argument("--min-valid-corners", type=int, default=4)
    parser.add_argument("--crop-margin-px", type=int, default=260)
    parser.add_argument("--max-crop-width-px", type=float, default=1900.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
