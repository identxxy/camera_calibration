#!/usr/bin/env python3
"""Black-tile corner refinement for the studio AprilTag tower."""

from __future__ import annotations

import math

import numpy as np


DEFAULT_OPTIONS = {
    "method": "none",
    "red_box_scale_min": 1.0,
    "red_box_scale_max": 1.25,
    "red_box_target_scale": 1.18,
    "corner_roi_margin_px": 8.0,
    "max_lateral_error_px": 12.0,
    "corner_quality_level": 0.01,
    "corner_min_distance_px": 3.0,
    "corner_block_size": 5,
    "corner_subpix_window_px": 5,
    "scale_weight_px": 0.8,
    "edge_canny_low": 40.0,
    "edge_canny_high": 120.0,
    "edge_arm_length_px": 22.0,
    "edge_arm_width_px": 2.5,
    "edge_scale_step": 0.005,
    "edge_lateral_step_px": 1.5,
    "edge_min_arm_support": 0.16,
    "edge_score_weight_px": 18.0,
    "edge_fallback_only": True,
}


def with_defaults(options):
    merged = dict(DEFAULT_OPTIONS)
    merged.update(options or {})
    return merged


def local_edge_support(edge_image, point, dir_a, dir_b, arm_length_px, arm_width_px):
    height, width = edge_image.shape[:2]
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
        arm_samples = max(6, int(float(arm_length_px) / 2.0))
        side_samples = max(3, int(2.0 * float(arm_width_px)) + 1)
        for arm_index in range(1, arm_samples + 1):
            distance = float(arm_length_px) * arm_index / arm_samples
            center_point = point + direction * distance
            for side_index in range(side_samples):
                offset = (
                    -float(arm_width_px)
                    + 2.0 * float(arm_width_px) * side_index / max(1, side_samples - 1))
                sample = center_point + normal * offset
                x = int(round(float(sample[0])))
                y = int(round(float(sample[1])))
                if 0 <= x < width and 0 <= y < height:
                    total += 1
                    if edge_image[y, x] > 0:
                        hits += 1
        supports.append(float(hits) / max(1, total))
    return supports[0], supports[1]


def refine_scaled_detector_box(cv2, image, detector_corners, options):
    options = with_defaults(options)
    method = options["method"]
    detector = np.asarray(detector_corners, dtype=np.float64)
    if detector.shape != (4, 2):
        return None
    center = detector.mean(axis=0)
    refined = [None, None, None, None]
    corner_scales = [None, None, None, None]
    corner_sources = [None, None, None, None]
    height, width = image.shape[:2]
    work = cv2.GaussianBlur(image, (3, 3), 0)
    edge_image = None
    use_edge_support = method == "red-scale-edge"
    if use_edge_support:
        edge_image = cv2.Canny(
            work,
            float(options["edge_canny_low"]),
            float(options["edge_canny_high"]),
            apertureSize=3,
            L2gradient=True)

    scale_min = float(options["red_box_scale_min"])
    scale_max = float(options["red_box_scale_max"])
    target_scale = float(options["red_box_target_scale"])
    max_lateral = float(options["max_lateral_error_px"])

    for corner_index, corner in enumerate(detector):
        ray = corner - center
        ray_length = float(np.linalg.norm(ray))
        if ray_length <= 1e-6:
            continue
        band_points = [center + ray * scale_min, center + ray * scale_max]
        margin = float(options["corner_roi_margin_px"])
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
        ray_norm_sq = float(ray @ ray)
        if roi.shape[0] >= 8 and roi.shape[1] >= 8:
            points = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=48,
                qualityLevel=float(options["corner_quality_level"]),
                minDistance=float(options["corner_min_distance_px"]),
                blockSize=int(options["corner_block_size"]),
                useHarrisDetector=True,
                k=0.04)
            if points is not None:
                points = points.reshape(-1, 2).astype(np.float32)
                requested_window = int(options["corner_subpix_window_px"])
                max_window = min(
                    requested_window,
                    (int(roi.shape[1]) - 5) // 2,
                    (int(roi.shape[0]) - 5) // 2)
                if max_window >= 1:
                    cv2.cornerSubPix(
                        roi,
                        points.reshape(-1, 1, 2),
                        (max_window, max_window),
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
                for point in points:
                    image_point = point.astype(np.float64) + np.asarray([x0, y0], dtype=np.float64)
                    candidate = score_candidate(
                        image_point,
                        center,
                        ray,
                        ray_norm_sq,
                        ray_length,
                        scale_min,
                        scale_max,
                        target_scale,
                        max_lateral,
                        float(options["scale_weight_px"]),
                        edge_image,
                        direction_previous,
                        direction_next,
                        options)
                    if candidate:
                        candidate["source"] = "corner"
                        candidates.append(candidate)
        should_scan_edge = (
            use_edge_support
            and edge_image is not None
            and (not bool(options.get("edge_fallback_only", True)) or not candidates))
        if should_scan_edge:
            ray_unit = ray / max(1e-9, ray_length)
            lateral_unit = np.asarray([-ray_unit[1], ray_unit[0]], dtype=np.float64)
            scale_values = np.arange(
                scale_min,
                scale_max + 0.5 * float(options["edge_scale_step"]),
                float(options["edge_scale_step"]))
            lateral_values = np.arange(
                -max_lateral,
                max_lateral + 0.5 * float(options["edge_lateral_step_px"]),
                float(options["edge_lateral_step_px"]))
            for scale in scale_values:
                base_point = center + ray * float(scale)
                for lateral_offset in lateral_values:
                    image_point = base_point + lateral_unit * float(lateral_offset)
                    if image_point[0] < 0 or image_point[1] < 0 or image_point[0] >= width or image_point[1] >= height:
                        continue
                    candidate = score_candidate(
                        image_point,
                        center,
                        ray,
                        ray_norm_sq,
                        ray_length,
                        scale_min,
                        scale_max,
                        target_scale,
                        max_lateral,
                        float(options["scale_weight_px"]),
                        edge_image,
                        direction_previous,
                        direction_next,
                        options)
                    if not candidate:
                        continue
                    if (
                            candidate["edge_support_a"] < float(options["edge_min_arm_support"])
                            or candidate["edge_support_b"] < float(options["edge_min_arm_support"])):
                        continue
                    candidate["source"] = "edge"
                    candidates.append(candidate)
        candidates.sort(key=lambda value: value["score"])
        if candidates:
            best = candidates[0]
            refined[corner_index] = [float(best["point"][0]), float(best["point"][1])]
            corner_scales[corner_index] = best["scale"]
            corner_sources[corner_index] = best["source"]
    return {
        "corners": refined,
        "valid_corner_count": sum(1 for corner in refined if corner is not None),
        "corner_scales": corner_scales,
        "corner_sources": corner_sources,
    }


def score_candidate(
        image_point,
        center,
        ray,
        ray_norm_sq,
        ray_length,
        scale_min,
        scale_max,
        target_scale,
        max_lateral,
        scale_weight_px,
        edge_image,
        direction_previous,
        direction_next,
        options):
    delta = image_point - center
    scale = float((delta @ ray) / ray_norm_sq)
    if scale < scale_min or scale > scale_max:
        return None
    projected = center + ray * scale
    lateral_error = float(np.linalg.norm(image_point - projected))
    if lateral_error > max_lateral:
        return None
    target_error = abs(scale - target_scale) * ray_length
    edge_a = 0.0
    edge_b = 0.0
    edge_penalty = 0.0
    if edge_image is not None:
        edge_a, edge_b = local_edge_support(
            edge_image,
            image_point,
            direction_previous,
            direction_next,
            float(options["edge_arm_length_px"]),
            float(options["edge_arm_width_px"]))
        edge_penalty = float(options["edge_score_weight_px"]) * max(0.0, 2.0 - edge_a - edge_b)
    score = lateral_error + scale_weight_px * target_error + edge_penalty
    return {
        "point": np.asarray(image_point, dtype=np.float64),
        "scale": scale,
        "lateral_error": lateral_error,
        "edge_support_a": edge_a,
        "edge_support_b": edge_b,
        "score": float(score),
    }


def refine_detections(cv2, image, detections, options):
    options = with_defaults(options)
    if options["method"] == "none":
        return detections
    refined = []
    for detection in detections:
        corners = detection.get("corners", [])
        result = refine_scaled_detector_box(cv2, image, corners, options)
        updated = dict(detection)
        if result is None:
            updated["corners"] = [None, None, None, None]
            updated["black_tile_valid_corner_count"] = 0
            updated["black_tile_corner_sources"] = ["missing"] * 4
        else:
            updated["corners"] = result["corners"]
            updated["black_tile_valid_corner_count"] = result["valid_corner_count"]
            updated["black_tile_corner_scales"] = result["corner_scales"]
            updated["black_tile_corner_sources"] = [
                source if source is not None else "missing"
                for source in result["corner_sources"]
            ]
        refined.append(updated)
    return refined
