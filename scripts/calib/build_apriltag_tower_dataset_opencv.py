#!/usr/bin/env python3
"""Build a camera_calibration dataset from OpenCV AprilTag tower detections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
import struct
import sys
import time

import apriltag_tower_black_tile_refine as black_tile_refine
import distributed_apriltag_quality_filter as qc


_DETECT_WORKER = {}


def read_image_dirs(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Empty image-directories file: {path}")
    return [Path(item) for item in qc.split_values(text)]


def read_tsv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def list_frames(image_dirs):
    names = set()
    for image_dir in image_dirs:
        if not image_dir.is_dir():
            raise SystemExit(f"Image directory is missing: {image_dir}")
        for path in image_dir.iterdir():
            if path.is_file() or path.is_symlink():
                if path.suffix.lower() in qc.IMAGE_EXTENSIONS:
                    names.add(path.name)
    return sorted(names, key=qc.natural_key)


def first_image_size(cv2, image_dir):
    for path in sorted(image_dir.iterdir(), key=lambda item: qc.natural_key(item.name)):
        if path.suffix.lower() not in qc.IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            height, width = image.shape[:2]
            return width, height
    raise SystemExit(f"No decodable image found in: {image_dir}")


def frame_name_for_out_frame(out_frame, output_extension):
    return f"{int(out_frame):06d}{output_extension}"


def load_selected_image_index(path, output_extension):
    if path is None:
        return {}
    rows = read_tsv_rows(path)
    index = {}
    for row in rows:
        camera_id = str(row.get("camera_id", ""))
        time_id = str(row.get("time", ""))
        frame_id = qc.frame_text(row.get("frame_id", ""))
        if not camera_id or not time_id or frame_id == "":
            continue
        frame_name = Path(row.get("filtered_image", "")).name
        if not frame_name and row.get("out_frame", "") != "":
            frame_name = frame_name_for_out_frame(row["out_frame"], output_extension)
        index[(camera_id, time_id, frame_id)] = {
            "camera_index": int(row["camera_index"]),
            "frame_name": frame_name,
            "source": row.get("source", ""),
            "filtered_image": row.get("filtered_image", ""),
            "row": row,
        }
    return index


def detection_record_key(record):
    camera_id = str(record.get("camera_id", ""))
    time_id = str(record.get("time", ""))
    frame_id = qc.frame_text(record.get("frame_id", ""))
    if not camera_id or not time_id or frame_id == "":
        return None
    return camera_id, time_id, frame_id


def load_detection_records(paths):
    records = []
    for path in paths or []:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                record["_detections_jsonl"] = str(path)
                record["_line_number"] = line_number
                records.append(record)
    return records


def resolve_record_image_path(record, selected_item, image_dirs, camera_index, frame_name):
    candidates = []
    if record.get("image_path"):
        candidates.append(Path(record["image_path"]))
    if selected_item:
        if selected_item.get("source"):
            candidates.append(Path(selected_item["source"]))
        if selected_item.get("filtered_image"):
            candidates.append(Path(selected_item["filtered_image"]))
    if 0 <= camera_index < len(image_dirs):
        candidates.append(image_dirs[camera_index] / frame_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else None


def u32(value):
    return struct.pack(">I", int(value))


def i32(value):
    return struct.pack(">i", int(value))


def f32(value):
    return struct.pack("<f", float(value))


def read_tower_config(path):
    config = qc.parse_simple_yaml(path)
    config.setdefault("tag_family", "tag36h11")
    config.setdefault("faces", 8)
    config.setdefault("tag_columns", 2)
    config.setdefault("tag_rows", 16)
    config.setdefault("tag_size_m", 0.08)
    config.setdefault("tag_spacing_m", 0.02)
    config.setdefault("first_tag_id", 0)
    config.setdefault("face_id_stride", int(config["tag_columns"]) * int(config["tag_rows"]))
    config.setdefault(
        "face_width_m",
        int(config["tag_columns"]) * float(config["tag_size_m"])
        + (int(config["tag_columns"]) - 1) * float(config["tag_spacing_m"]))
    config.setdefault("face0_angle_degrees", 0.0)
    config.setdefault("tag_rotation_degrees", 0)
    return config


def physical_corner_for_opencv_corner(corner_id, tag_rotation_degrees):
    # OpenCV aruco returns marker corners in canonical image order
    # (top-left, top-right, bottom-right, bottom-left). The tower geometry
    # stores physical corners as lower-left, lower-right, upper-right,
    # upper-left as seen from outside the face.
    if tag_rotation_degrees == 0:
        return [3, 2, 1, 0][corner_id]
    if tag_rotation_degrees == 180:
        return [1, 0, 3, 2][corner_id]
    raise SystemExit(f"Unsupported tag_rotation_degrees: {tag_rotation_degrees}")


def build_tower_points(config):
    faces = int(config["faces"])
    columns = int(config["tag_columns"])
    rows = int(config["tag_rows"])
    tag_size = float(config["tag_size_m"])
    spacing = float(config["tag_spacing_m"])
    first_tag_id = int(config["first_tag_id"])
    face_id_stride = int(config["face_id_stride"])
    face_width = float(config["face_width_m"])
    face0_angle = math.radians(float(config["face0_angle_degrees"]))
    tag_rotation_degrees = int(config.get("tag_rotation_degrees", 0))

    pitch = tag_size + spacing
    half_tag = 0.5 * tag_size
    apothem = face_width / (2.0 * math.tan(math.pi / faces))
    points = {}
    for face in range(faces):
        theta = face0_angle + face * 2.0 * math.pi / faces
        normal = (math.cos(theta), math.sin(theta), 0.0)
        u_axis = (-math.sin(theta), math.cos(theta), 0.0)
        z_axis = (0.0, 0.0, 1.0)
        for row in range(rows):
            for col in range(columns):
                local_tag_id = row * columns + col
                tag_id = first_tag_id + face * face_id_stride + local_tag_id
                center_u = (col - 0.5 * (columns - 1)) * pitch
                center_z = (row - 0.5 * (rows - 1)) * pitch
                center = tuple(
                    normal[i] * apothem + u_axis[i] * center_u + z_axis[i] * center_z
                    for i in range(3))
                corners = [
                    tuple(center[i] - u_axis[i] * half_tag - z_axis[i] * half_tag for i in range(3)),
                    tuple(center[i] + u_axis[i] * half_tag - z_axis[i] * half_tag for i in range(3)),
                    tuple(center[i] + u_axis[i] * half_tag + z_axis[i] * half_tag for i in range(3)),
                    tuple(center[i] - u_axis[i] * half_tag + z_axis[i] * half_tag for i in range(3)),
                ]
                for corner_id in range(4):
                    physical_corner_id = physical_corner_for_opencv_corner(
                        corner_id,
                        tag_rotation_degrees)
                    points[tag_id * 4 + corner_id] = corners[physical_corner_id]
    return points


def write_dataset(path, image_sizes, imagesets, tower_points, cell_length):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"calib_data")
        stream.write(u32(1))
        stream.write(u32(len(image_sizes)))
        for width, height in image_sizes:
            stream.write(u32(width))
            stream.write(u32(height))

        stream.write(u32(len(imagesets)))
        for filename, camera_features in imagesets:
            encoded = filename.encode("utf-8")
            stream.write(u32(len(encoded)))
            stream.write(encoded)
            for features in camera_features:
                stream.write(u32(len(features)))
                for x, y, feature_id in features:
                    stream.write(f32(x))
                    stream.write(f32(y))
                    stream.write(i32(feature_id))

        stream.write(u32(1))
        stream.write(f32(cell_length))
        stream.write(u32(0))
        stream.write(u32(len(tower_points)))
        for feature_id in sorted(tower_points):
            x, y, z = tower_points[feature_id]
            stream.write(i32(feature_id))
            stream.write(f32(x))
            stream.write(f32(y))
            stream.write(f32(z))


def detections_to_features(detections):
    features = []
    for det in detections:
        tag_id = int(det["tag_id"])
        for corner_id, corner in enumerate(det["corners"]):
            if corner is None:
                continue
            x, y = corner
            if x is None or y is None:
                continue
            features.append((float(x), float(y), tag_id * 4 + corner_id))
    return features


def count_detection_corners(detections):
    return len(detections_to_features(detections))


def update_camera_detection_stats(per_camera, camera_index, detections, failed_decode=False):
    row = per_camera[camera_index]
    if failed_decode:
        row["failed_images"] += 1
        return
    row["decoded_images"] += 1
    if detections:
        row["images_with_tags"] += 1
    row["total_tags"] += len(detections)
    row["max_tags"] = max(row["max_tags"], len(detections))


def refine_detection_record(cv2, args, image, detections):
    if args.subpixel_refine_original:
        detections = qc.refine_detections_subpixel(
            cv2,
            image,
            detections,
            args.subpixel_window_size,
            args.subpixel_max_iterations,
            args.subpixel_epsilon)
    if args.edge_line_refine_original:
        detections = qc.refine_detections_edge_lines(
            image,
            detections,
            args.edge_line_search_radius_px,
            args.edge_line_sample_spacing_px,
            args.edge_line_gradient_step_px,
            args.edge_line_min_gradient,
            args.edge_line_min_edge_points,
            args.edge_line_max_shift_px,
            args.edge_line_polarity)
    if args.black_tile_corner_refine != "none":
        detections = black_tile_refine.refine_detections(
            cv2,
            image,
            detections,
            make_black_tile_refine_options(args))
    return detections


def refine_detection_record_with_options(cv2, options, image, detections):
    if options["subpixel_refine_original"]:
        detections = qc.refine_detections_subpixel(
            cv2,
            image,
            detections,
            options["subpixel_window_size"],
            options["subpixel_max_iterations"],
            options["subpixel_epsilon"])
    if options["edge_line_refine_original"]:
        detections = qc.refine_detections_edge_lines(
            image,
            detections,
            options["edge_line_search_radius_px"],
            options["edge_line_sample_spacing_px"],
            options["edge_line_gradient_step_px"],
            options["edge_line_min_gradient"],
            options["edge_line_min_edge_points"],
            options["edge_line_max_shift_px"],
            options["edge_line_polarity"])
    if options["black_tile_corner_refine"] != "none":
        detections = black_tile_refine.refine_detections(
            cv2,
            image,
            detections,
            make_black_tile_refine_options(options))
    return detections


def option_value(source, key):
    if isinstance(source, dict):
        return source[key]
    return getattr(source, key)


def make_black_tile_refine_options(source):
    return {
        "method": option_value(source, "black_tile_corner_refine"),
        "red_box_scale_min": float(option_value(source, "black_tile_red_box_scale_min")),
        "red_box_scale_max": float(option_value(source, "black_tile_red_box_scale_max")),
        "red_box_target_scale": float(option_value(source, "black_tile_red_box_target_scale")),
        "corner_roi_margin_px": float(option_value(source, "black_tile_corner_roi_margin_px")),
        "max_lateral_error_px": float(option_value(source, "black_tile_max_lateral_error_px")),
        "corner_quality_level": float(option_value(source, "black_tile_corner_quality_level")),
        "corner_min_distance_px": float(option_value(source, "black_tile_corner_min_distance_px")),
        "corner_block_size": int(option_value(source, "black_tile_corner_block_size")),
        "corner_subpix_window_px": int(option_value(source, "black_tile_corner_subpix_window_px")),
        "scale_weight_px": float(option_value(source, "black_tile_scale_weight_px")),
        "edge_canny_low": float(option_value(source, "black_tile_edge_canny_low")),
        "edge_canny_high": float(option_value(source, "black_tile_edge_canny_high")),
        "edge_arm_length_px": float(option_value(source, "black_tile_edge_arm_length_px")),
        "edge_arm_width_px": float(option_value(source, "black_tile_edge_arm_width_px")),
        "edge_scale_step": float(option_value(source, "black_tile_edge_scale_step")),
        "edge_lateral_step_px": float(option_value(source, "black_tile_edge_lateral_step_px")),
        "edge_min_arm_support": float(option_value(source, "black_tile_edge_min_arm_support")),
        "edge_score_weight_px": float(option_value(source, "black_tile_edge_score_weight_px")),
        "edge_fallback_only": bool(option_value(source, "black_tile_edge_fallback_only")),
    }


def make_detect_worker_options(args, image_dirs, valid_ids, dictionary_name):
    return {
        "image_dirs": [str(path) for path in image_dirs],
        "valid_ids": sorted(int(tag_id) for tag_id in valid_ids),
        "dictionary_name": dictionary_name,
        "detect_inverted": bool(args.detect_inverted),
        "error_correction_rate": float(args.error_correction_rate),
        "corner_refinement": args.corner_refinement,
        "corner_refinement_window_size": int(args.corner_refinement_window_size),
        "corner_refinement_max_iterations": int(args.corner_refinement_max_iterations),
        "corner_refinement_min_accuracy": float(args.corner_refinement_min_accuracy),
        "resize_factor": float(args.resize_factor),
        "subpixel_refine_original": bool(args.subpixel_refine_original),
        "subpixel_window_size": int(args.subpixel_window_size),
        "subpixel_max_iterations": int(args.subpixel_max_iterations),
        "subpixel_epsilon": float(args.subpixel_epsilon),
        "edge_line_refine_original": bool(args.edge_line_refine_original),
        "edge_line_search_radius_px": float(args.edge_line_search_radius_px),
        "edge_line_sample_spacing_px": float(args.edge_line_sample_spacing_px),
        "edge_line_gradient_step_px": float(args.edge_line_gradient_step_px),
        "edge_line_min_gradient": float(args.edge_line_min_gradient),
        "edge_line_min_edge_points": int(args.edge_line_min_edge_points),
        "edge_line_max_shift_px": float(args.edge_line_max_shift_px),
        "edge_line_polarity": args.edge_line_polarity,
        "black_tile_corner_refine": args.black_tile_corner_refine,
        "black_tile_red_box_scale_min": float(args.black_tile_red_box_scale_min),
        "black_tile_red_box_scale_max": float(args.black_tile_red_box_scale_max),
        "black_tile_red_box_target_scale": float(args.black_tile_red_box_target_scale),
        "black_tile_corner_roi_margin_px": float(args.black_tile_corner_roi_margin_px),
        "black_tile_max_lateral_error_px": float(args.black_tile_max_lateral_error_px),
        "black_tile_corner_quality_level": float(args.black_tile_corner_quality_level),
        "black_tile_corner_min_distance_px": float(args.black_tile_corner_min_distance_px),
        "black_tile_corner_block_size": int(args.black_tile_corner_block_size),
        "black_tile_corner_subpix_window_px": int(args.black_tile_corner_subpix_window_px),
        "black_tile_scale_weight_px": float(args.black_tile_scale_weight_px),
        "black_tile_edge_canny_low": float(args.black_tile_edge_canny_low),
        "black_tile_edge_canny_high": float(args.black_tile_edge_canny_high),
        "black_tile_edge_arm_length_px": float(args.black_tile_edge_arm_length_px),
        "black_tile_edge_arm_width_px": float(args.black_tile_edge_arm_width_px),
        "black_tile_edge_scale_step": float(args.black_tile_edge_scale_step),
        "black_tile_edge_lateral_step_px": float(args.black_tile_edge_lateral_step_px),
        "black_tile_edge_min_arm_support": float(args.black_tile_edge_min_arm_support),
        "black_tile_edge_score_weight_px": float(args.black_tile_edge_score_weight_px),
        "black_tile_edge_fallback_only": bool(args.black_tile_edge_fallback_only),
        "write_detection_rows": bool(args.detections_tsv),
        "write_empty_detections": bool(args.write_empty_detections),
    }


def init_detect_worker(options):
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV with aruco support is required.") from exc
    detector = qc.create_detector(
        cv2,
        options["dictionary_name"],
        options["detect_inverted"],
        options["error_correction_rate"],
        options["corner_refinement"],
        options["corner_refinement_window_size"],
        options["corner_refinement_max_iterations"],
        options["corner_refinement_min_accuracy"])
    _DETECT_WORKER.clear()
    _DETECT_WORKER.update({
        "cv2": cv2,
        "detector": detector,
        "options": options,
        "image_dirs": [Path(path) for path in options["image_dirs"]],
        "valid_ids": set(options["valid_ids"]),
    })


def detect_frame_worker(task):
    frame_index, frame_name = task
    cv2 = _DETECT_WORKER["cv2"]
    detector = _DETECT_WORKER["detector"]
    options = _DETECT_WORKER["options"]
    image_dirs = _DETECT_WORKER["image_dirs"]
    valid_ids = _DETECT_WORKER["valid_ids"]
    camera_features = []
    stats = []
    detection_rows = []
    for camera_index, image_dir in enumerate(image_dirs):
        image_path = image_dir / frame_name
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        row_stats = {
            "decoded_images": 0,
            "failed_images": 0,
            "images_with_tags": 0,
            "total_tags": 0,
            "max_tags": 0,
        }
        if image is None:
            row_stats["failed_images"] = 1
            camera_features.append([])
            stats.append(row_stats)
            continue
        row_stats["decoded_images"] = 1
        detect_image, scale = qc.resize_for_detection(cv2, image, options["resize_factor"])
        detections, rejected_count = qc.detect_markers(cv2, detector, detect_image)
        detections = qc.scale_detections(detections, scale)
        detections = [det for det in detections if det["tag_id"] in valid_ids]
        detections = refine_detection_record_with_options(cv2, options, image, detections)
        if detections:
            row_stats["images_with_tags"] = 1
        row_stats["total_tags"] = len(detections)
        row_stats["max_tags"] = len(detections)
        features = detections_to_features(detections)
        camera_features.append(features)
        stats.append(row_stats)
        if options["write_detection_rows"] and (detections or options["write_empty_detections"]):
            detection_rows.append({
                "frame_index": frame_index,
                "filename": frame_name,
                "camera_index": camera_index,
                "image_dir": str(image_dir),
                "tag_count": len(detections),
                "corner_count": len(features),
                "tag_ids": ",".join(str(det["tag_id"]) for det in sorted(detections, key=lambda item: item["tag_id"])),
                "rejected_count": rejected_count,
            })
    return frame_index, frame_name, camera_features, stats, detection_rows


def merge_per_camera_stats(per_camera, frame_stats):
    for camera_index, row_stats in enumerate(frame_stats):
        row = per_camera[camera_index]
        row["decoded_images"] += row_stats["decoded_images"]
        row["failed_images"] += row_stats["failed_images"]
        row["images_with_tags"] += row_stats["images_with_tags"]
        row["total_tags"] += row_stats["total_tags"]
        row["max_tags"] = max(row["max_tags"], row_stats["max_tags"])


def build_from_opencv_detection(cv2, args, image_dirs, frame_names, valid_ids, dictionary_name):
    per_camera = [
        {
            "camera_index": index,
            "image_dir": str(image_dir),
            "decoded_images": 0,
            "failed_images": 0,
            "images_with_tags": 0,
            "total_tags": 0,
            "max_tags": 0,
        }
        for index, image_dir in enumerate(image_dirs)
    ]
    imagesets = [None] * len(frame_names)
    detection_rows = []
    worker_options = make_detect_worker_options(args, image_dirs, valid_ids, dictionary_name)
    tasks = list(enumerate(frame_names))
    num_workers = max(1, int(args.num_workers))
    pool = None
    if num_workers == 1:
        init_detect_worker(worker_options)
        iterator = map(detect_frame_worker, tasks)
    else:
        context = mp.get_context(args.multiprocessing_start_method)
        pool = context.Pool(
            processes=num_workers,
            initializer=init_detect_worker,
            initargs=(worker_options,))
        iterator = pool.imap_unordered(detect_frame_worker, tasks, chunksize=max(1, int(args.worker_chunksize)))
    completed = 0
    try:
        for frame_index, frame_name, camera_features, frame_stats, frame_detection_rows in iterator:
            imagesets[frame_index] = (frame_name, camera_features)
            merge_per_camera_stats(per_camera, frame_stats)
            detection_rows.extend(frame_detection_rows)
            completed += 1
            if args.progress_interval > 0 and (
                    completed == len(tasks) or completed % args.progress_interval == 0):
                print(
                    f"[build_apriltag_tower_dataset] processed {completed}/{len(tasks)} frames",
                    file=sys.stderr,
                    flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if any(item is None for item in imagesets):
        missing = [index for index, item in enumerate(imagesets) if item is None]
        raise SystemExit(f"Missing worker results for frame indices: {missing[:20]}")
    return imagesets, per_camera, detection_rows


def build_from_detection_records(cv2, args, image_dirs, frame_names, valid_ids):
    selected_index = load_selected_image_index(args.selected_images_tsv, args.output_extension)
    if not selected_index:
        raise SystemExit("--input-detections-jsonl currently requires --selected-images-tsv.")
    input_records = load_detection_records(args.input_detections_jsonl)
    frame_index_by_name = {name: index for index, name in enumerate(frame_names)}
    imagesets = [(frame_name, [[] for _ in image_dirs]) for frame_name in frame_names]
    per_camera = [
        {
            "camera_index": index,
            "image_dir": str(image_dir),
            "decoded_images": 0,
            "failed_images": 0,
            "images_with_tags": 0,
            "total_tags": 0,
            "max_tags": 0,
        }
        for index, image_dir in enumerate(image_dirs)
    ]
    detection_rows = []
    skipped_records = {
        "missing_selected_image": 0,
        "missing_frame_name": 0,
        "invalid_camera_index": 0,
        "decode_failed": 0,
    }
    used_records = 0
    for record in input_records:
        key = detection_record_key(record)
        selected_item = selected_index.get(key) if key else None
        if selected_item is None:
            skipped_records["missing_selected_image"] += 1
            continue
        camera_index = int(selected_item["camera_index"])
        if camera_index < 0 or camera_index >= len(image_dirs):
            skipped_records["invalid_camera_index"] += 1
            continue
        frame_name = selected_item["frame_name"]
        frame_index = frame_index_by_name.get(frame_name)
        if frame_index is None:
            skipped_records["missing_frame_name"] += 1
            continue
        detections = [
            det for det in record.get("detections", [])
            if int(det.get("tag_id", -1)) in valid_ids
        ]
        if (
                args.subpixel_refine_original
                or args.edge_line_refine_original
                or args.black_tile_corner_refine != "none"):
            image_path = resolve_record_image_path(
                record,
                selected_item,
                image_dirs,
                camera_index,
                frame_name)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) if image_path else None
            if image is None:
                skipped_records["decode_failed"] += 1
                update_camera_detection_stats(per_camera, camera_index, [], failed_decode=True)
                continue
            detections = refine_detection_record(cv2, args, image, detections)
        features = detections_to_features(detections)
        imagesets[frame_index][1][camera_index] = features
        update_camera_detection_stats(per_camera, camera_index, detections)
        used_records += 1
        if args.detections_tsv and (detections or args.write_empty_detections):
            detection_rows.append({
                "frame_index": frame_index,
                "filename": frame_name,
                "camera_index": camera_index,
                "image_dir": str(image_dirs[camera_index]),
                "tag_count": len(detections),
                "corner_count": len(features),
                "tag_ids": ",".join(str(det["tag_id"]) for det in sorted(detections, key=lambda item: item["tag_id"])),
                "rejected_count": record.get("rejected_count", ""),
            })
    return imagesets, per_camera, detection_rows, {
        "input_detection_record_count": len(input_records),
        "used_detection_record_count": used_records,
        "skipped_detection_records": skipped_records,
        "selected_images_tsv": str(args.selected_images_tsv),
        "input_detections_jsonl": [str(path) for path in args.input_detections_jsonl],
    }


def run(args):
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV with aruco support is required. Install opencv-contrib-python-headless.") from exc
    if not hasattr(cv2, "aruco"):
        raise SystemExit("Installed OpenCV does not include cv2.aruco.")

    image_dirs = read_image_dirs(args.image_directories_file)
    frame_names = list_frames(image_dirs)
    if args.max_frames > 0:
        frame_names = frame_names[:args.max_frames]
    if not frame_names:
        raise SystemExit("No image frames found.")

    tower_config = read_tower_config(args.tower_config)
    valid_ids = qc.tower_valid_tag_ids(tower_config)
    if not valid_ids:
        raise SystemExit(f"Tower config did not expand any valid ids: {args.tower_config}")
    dictionary_name = qc.dictionary_from_config(tower_config, args.dictionary)

    image_sizes = [first_image_size(cv2, image_dir) for image_dir in image_dirs]
    tower_points = build_tower_points(tower_config)
    cell_length = float(tower_config["tag_size_m"]) + float(tower_config["tag_spacing_m"])

    start = time.time()
    source_mode = "opencv_detect_images"
    extra_summary = {}
    if args.input_detections_jsonl:
        source_mode = "detections_jsonl"
        imagesets, per_camera, detection_rows, extra_summary = build_from_detection_records(
            cv2,
            args,
            image_dirs,
            frame_names,
            valid_ids)
    else:
        imagesets, per_camera, detection_rows = build_from_opencv_detection(
            cv2,
            args,
            image_dirs,
            frame_names,
            valid_ids,
            dictionary_name)

    write_dataset(args.output_dataset, image_sizes, imagesets, tower_points, cell_length)

    summary = {
        "mode": "opencv_apriltag_tower_dataset",
        "source_mode": source_mode,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": time.time() - start,
        "image_directories_file": str(args.image_directories_file),
        "output_dataset": str(args.output_dataset),
        "tower_config": str(args.tower_config),
        "dictionary": dictionary_name,
        "resize_factor": args.resize_factor,
        "corner_refinement": args.corner_refinement,
        "subpixel_refine_original": bool(args.subpixel_refine_original),
        "subpixel_window_size": args.subpixel_window_size,
        "subpixel_max_iterations": args.subpixel_max_iterations,
        "subpixel_epsilon": args.subpixel_epsilon,
        "edge_line_refine_original": bool(args.edge_line_refine_original),
        "edge_line_search_radius_px": args.edge_line_search_radius_px,
        "edge_line_sample_spacing_px": args.edge_line_sample_spacing_px,
        "edge_line_gradient_step_px": args.edge_line_gradient_step_px,
        "edge_line_min_gradient": args.edge_line_min_gradient,
        "edge_line_min_edge_points": args.edge_line_min_edge_points,
        "edge_line_max_shift_px": args.edge_line_max_shift_px,
        "edge_line_polarity": args.edge_line_polarity,
        "black_tile_corner_refine": args.black_tile_corner_refine,
        "black_tile_refine_options": make_black_tile_refine_options(args),
        "num_workers": args.num_workers,
        "worker_chunksize": args.worker_chunksize,
        "multiprocessing_start_method": args.multiprocessing_start_method,
        "camera_count": len(image_dirs),
        "imageset_count": len(imagesets),
        "known_3d_point_count": len(tower_points),
        "total_tags": sum(row["total_tags"] for row in per_camera),
        "per_camera": per_camera,
    }
    summary.update(extra_summary)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.per_camera_tsv:
        Path(args.per_camera_tsv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.per_camera_tsv).open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "camera_index", "image_dir", "decoded_images", "failed_images",
                "images_with_tags", "total_tags", "max_tags",
            ]
            writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_camera)
    if args.detections_tsv:
        Path(args.detections_tsv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.detections_tsv).open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "frame_index", "filename", "camera_index", "image_dir",
                "tag_count", "corner_count", "tag_ids", "rejected_count",
            ]
            writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
            writer.writeheader()
            writer.writerows(detection_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-directories-file", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--tower-config", type=Path, default=qc.DEFAULT_TOWER_CONFIG)
    parser.add_argument("--dictionary", default="")
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument("--detect-inverted", action="store_true", default=True)
    parser.add_argument("--no-detect-inverted", dest="detect_inverted", action="store_false")
    parser.add_argument("--error-correction-rate", type=float, default=0.6)
    parser.add_argument(
        "--corner-refinement",
        choices=["none", "subpix", "contour", "apriltag"],
        default="subpix")
    parser.add_argument("--corner-refinement-window-size", type=int, default=5)
    parser.add_argument("--corner-refinement-max-iterations", type=int, default=30)
    parser.add_argument("--corner-refinement-min-accuracy", type=float, default=0.01)
    parser.add_argument("--subpixel-refine-original", action="store_true", default=True)
    parser.add_argument("--no-subpixel-refine-original", dest="subpixel_refine_original", action="store_false")
    parser.add_argument("--subpixel-window-size", type=int, default=5)
    parser.add_argument("--subpixel-max-iterations", type=int, default=30)
    parser.add_argument("--subpixel-epsilon", type=float, default=0.01)
    parser.add_argument("--edge-line-refine-original", action="store_true")
    parser.add_argument("--edge-line-search-radius-px", type=float, default=5.0)
    parser.add_argument("--edge-line-sample-spacing-px", type=float, default=2.0)
    parser.add_argument("--edge-line-gradient-step-px", type=float, default=1.0)
    parser.add_argument("--edge-line-min-gradient", type=float, default=2.0)
    parser.add_argument("--edge-line-min-edge-points", type=int, default=8)
    parser.add_argument("--edge-line-max-shift-px", type=float, default=4.0)
    parser.add_argument(
        "--edge-line-polarity",
        choices=["outside_white_inside_black", "outside_black_inside_white", "absolute"],
        default="outside_white_inside_black")
    parser.add_argument(
        "--black-tile-corner-refine",
        choices=["none", "red-scale", "red-scale-edge"],
        default="none",
        help=(
            "Replace OpenCV AprilTag inner-detector corners with physical 8 cm black-tile "
            "outer corners found in the red-box scale band. red-scale-edge also uses local "
            "short edge support. Missing outer corners are omitted at corner level."))
    parser.add_argument("--black-tile-red-box-scale-min", type=float, default=1.0)
    parser.add_argument("--black-tile-red-box-scale-max", type=float, default=1.25)
    parser.add_argument("--black-tile-red-box-target-scale", type=float, default=1.18)
    parser.add_argument("--black-tile-corner-roi-margin-px", type=float, default=8.0)
    parser.add_argument("--black-tile-max-lateral-error-px", type=float, default=12.0)
    parser.add_argument("--black-tile-corner-quality-level", type=float, default=0.01)
    parser.add_argument("--black-tile-corner-min-distance-px", type=float, default=3.0)
    parser.add_argument("--black-tile-corner-block-size", type=int, default=5)
    parser.add_argument("--black-tile-corner-subpix-window-px", type=int, default=5)
    parser.add_argument("--black-tile-scale-weight-px", type=float, default=0.8)
    parser.add_argument("--black-tile-edge-canny-low", type=float, default=40.0)
    parser.add_argument("--black-tile-edge-canny-high", type=float, default=120.0)
    parser.add_argument("--black-tile-edge-arm-length-px", type=float, default=22.0)
    parser.add_argument("--black-tile-edge-arm-width-px", type=float, default=2.5)
    parser.add_argument("--black-tile-edge-scale-step", type=float, default=0.005)
    parser.add_argument("--black-tile-edge-lateral-step-px", type=float, default=1.5)
    parser.add_argument("--black-tile-edge-min-arm-support", type=float, default=0.16)
    parser.add_argument("--black-tile-edge-score-weight-px", type=float, default=18.0)
    parser.add_argument("--black-tile-edge-fallback-only", action="store_true", default=True)
    parser.add_argument("--black-tile-edge-always-scan", dest="black_tile_edge_fallback_only", action="store_false")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of frame-level OpenCV detection workers. Default keeps the historical serial path.")
    parser.add_argument("--worker-chunksize", type=int, default=1)
    parser.add_argument(
        "--multiprocessing-start-method",
        choices=["fork", "spawn", "forkserver"],
        default="fork")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print frame progress to stderr every N completed frames; 0 disables progress logs.")
    parser.add_argument(
        "--input-detections-jsonl",
        action="append",
        type=Path,
        default=[],
        help=(
            "Use existing distributed worker detections instead of re-running OpenCV detection. "
            "Pair with --selected-images-tsv from aggregate mode; original-image subpixel "
            "or edge-line refinement can still be applied before writing the dataset."))
    parser.add_argument(
        "--selected-images-tsv",
        type=Path,
        default=None,
        help="selected_images.tsv from distributed aggregate mode, used to map worker detections to staged frame names.")
    parser.add_argument(
        "--output-extension",
        default=".jpg",
        help="Staged image extension used when selected_images.tsv has only out_frame.")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--per-camera-tsv", type=Path, default=None)
    parser.add_argument("--detections-tsv", type=Path, default=None)
    parser.add_argument("--write-empty-detections", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
