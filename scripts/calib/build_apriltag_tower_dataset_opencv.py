#!/usr/bin/env python3
"""Build a camera_calibration dataset from OpenCV AprilTag tower detections."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import time

import distributed_apriltag_quality_filter as qc


def read_image_dirs(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Empty image-directories file: {path}")
    return [Path(item) for item in qc.split_values(text)]


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

    detector = qc.create_detector(
        cv2,
        dictionary_name,
        args.detect_inverted,
        args.error_correction_rate,
        args.corner_refinement,
        args.corner_refinement_window_size,
        args.corner_refinement_max_iterations,
        args.corner_refinement_min_accuracy)
    image_sizes = [first_image_size(cv2, image_dir) for image_dir in image_dirs]
    tower_points = build_tower_points(tower_config)
    cell_length = float(tower_config["tag_size_m"]) + float(tower_config["tag_spacing_m"])

    start = time.time()
    imagesets = []
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
    for frame_index, frame_name in enumerate(frame_names):
        camera_features = []
        for camera_index, image_dir in enumerate(image_dirs):
            image_path = image_dir / frame_name
            features = []
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                per_camera[camera_index]["failed_images"] += 1
                camera_features.append(features)
                continue
            per_camera[camera_index]["decoded_images"] += 1
            detect_image, scale = qc.resize_for_detection(cv2, image, args.resize_factor)
            detections, rejected_count = qc.detect_markers(cv2, detector, detect_image)
            detections = qc.scale_detections(detections, scale)
            detections = [det for det in detections if det["tag_id"] in valid_ids]
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
            if detections:
                per_camera[camera_index]["images_with_tags"] += 1
            per_camera[camera_index]["total_tags"] += len(detections)
            per_camera[camera_index]["max_tags"] = max(per_camera[camera_index]["max_tags"], len(detections))
            for det in detections:
                tag_id = int(det["tag_id"])
                for corner_id, (x, y) in enumerate(det["corners"]):
                    features.append((float(x), float(y), tag_id * 4 + corner_id))
            camera_features.append(features)
            if args.detections_tsv and (detections or args.write_empty_detections):
                detection_rows.append({
                    "frame_index": frame_index,
                    "filename": frame_name,
                    "camera_index": camera_index,
                    "image_dir": str(image_dir),
                    "tag_count": len(detections),
                    "corner_count": 4 * len(detections),
                    "tag_ids": ",".join(str(det["tag_id"]) for det in sorted(detections, key=lambda item: item["tag_id"])),
                    "rejected_count": rejected_count,
                })
        imagesets.append((frame_name, camera_features))

    write_dataset(args.output_dataset, image_sizes, imagesets, tower_points, cell_length)

    summary = {
        "mode": "opencv_apriltag_tower_dataset",
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
        "camera_count": len(image_dirs),
        "imageset_count": len(imagesets),
        "known_3d_point_count": len(tower_points),
        "total_tags": sum(row["total_tags"] for row in per_camera),
        "per_camera": per_camera,
    }
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
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--per-camera-tsv", type=Path, default=None)
    parser.add_argument("--detections-tsv", type=Path, default=None)
    parser.add_argument("--write-empty-detections", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
