#!/usr/bin/env python3
"""Client-side AprilTag tower coverage detector using OpenCV ArUco.

This script is intended for loose distributed coverage checks on w1/w2/w3/w4.
It writes data-quality report artifacts, not a final calibration dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import time


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "OpenCV Python import failed. Install opencv-contrib-python on the client "
            f"or use --dry-run for path validation only. Error: {exc}")
    if not hasattr(cv2, "aruco"):
        raise SystemExit("cv2.aruco is unavailable. Install opencv-contrib-python.")
    return cv2


def parse_image_directories(args):
    if args.image_directories_file:
        text = Path(args.image_directories_file).read_text().strip()
    else:
        text = args.image_directories or ""
    dirs = [
        Path(item.strip()).expanduser()
        for item in re.split(r"[,\r\n]+", text)
        if item.strip()
    ]
    if not dirs:
        raise SystemExit("No image directories were provided.")
    for path in dirs:
        if not path.is_dir():
            raise SystemExit(f"Image directory does not exist: {path}")
    return dirs


def list_images(image_dir, max_frames, stride):
    files = [
        path for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if stride > 1:
        files = files[::stride]
    if max_frames and max_frames > 0:
        files = files[:max_frames]
    return files


def create_detector(cv2, args):
    dictionary_name = args.dictionary
    if not hasattr(cv2.aruco, dictionary_name):
        raise SystemExit(f"OpenCV has no dictionary: cv2.aruco.{dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(parameters, "detectInvertedMarker"):
        parameters.detectInvertedMarker = args.detect_inverted
    if hasattr(parameters, "errorCorrectionRate"):
        parameters.errorCorrectionRate = args.error_correction_rate
    if hasattr(parameters, "cornerRefinementMethod"):
        methods = {
            "none": "CORNER_REFINE_NONE",
            "subpix": "CORNER_REFINE_SUBPIX",
            "contour": "CORNER_REFINE_CONTOUR",
            "apriltag": "CORNER_REFINE_APRILTAG",
        }
        parameters.cornerRefinementMethod = getattr(
            cv2.aruco,
            methods[args.corner_refinement],
            getattr(cv2.aruco, "CORNER_REFINE_NONE", 0))
    if hasattr(parameters, "cornerRefinementWinSize"):
        parameters.cornerRefinementWinSize = args.corner_refinement_window_size
    if hasattr(parameters, "cornerRefinementMaxIterations"):
        parameters.cornerRefinementMaxIterations = args.corner_refinement_max_iterations
    if hasattr(parameters, "cornerRefinementMinAccuracy"):
        parameters.cornerRefinementMinAccuracy = args.corner_refinement_min_accuracy
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def detect(cv2, detector, image):
    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=parameters)
    detections = []
    if ids is None:
        return detections, len(rejected) if rejected is not None else 0
    for marker_corners, marker_id in zip(corners, ids.flatten().tolist()):
        pts = marker_corners.reshape(-1, 2).tolist()
        detections.append({
            "tag_id": int(marker_id),
            "corners": [[float(x), float(y)] for x, y in pts],
        })
    return detections, len(rejected) if rejected is not None else 0


def maybe_resize_for_detection(cv2, image, resize_factor):
    if resize_factor >= 0.999:
        return image, 1.0
    if resize_factor <= 0:
        raise SystemExit("--resize-factor must be positive.")
    resized = cv2.resize(
        image,
        None,
        fx=resize_factor,
        fy=resize_factor,
        interpolation=cv2.INTER_AREA)
    return resized, 1.0 / resize_factor


def scale_detections(detections, scale):
    if abs(scale - 1.0) < 1e-9:
        return detections
    scaled = []
    for detection in detections:
        scaled.append({
            "tag_id": detection["tag_id"],
            "corners": [
                [float(x) * scale, float(y) * scale]
                for x, y in detection["corners"]
            ],
        })
    return scaled


def refine_detections_subpixel(cv2, image, detections, args):
    if not detections or not args.subpixel_refine_original:
        return detections
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required for subpixel corner refinement.") from exc
    corners = []
    lengths = []
    for detection in detections:
        pts = detection.get("corners", [])
        lengths.append(len(pts))
        for x, y in pts:
            corners.append([float(x), float(y)])
    if not corners:
        return detections
    corner_array = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        args.subpixel_max_iterations,
        args.subpixel_epsilon,
    )
    refined = cv2.cornerSubPix(
        image,
        corner_array,
        (args.subpixel_window_size, args.subpixel_window_size),
        (-1, -1),
        criteria).reshape(-1, 2)
    output = []
    cursor = 0
    for detection, length in zip(detections, lengths):
        item = dict(detection)
        item["corners"] = [
            [float(x), float(y)]
            for x, y in refined[cursor:cursor + length]
        ]
        item["subpixel_refined"] = True
        output.append(item)
        cursor += length
    return output


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_time_from_image_dir(image_dir):
    parent = image_dir.parent.name
    if re.match(r"^\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}$", parent):
        return parent
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-directories")
    parser.add_argument("--image-directories-file", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--client-id", default="")
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    parser.add_argument("--detect-inverted", action="store_true", default=True)
    parser.add_argument("--no-detect-inverted", dest="detect_inverted",
                        action="store_false")
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
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resize-factor", type=float, default=1.0)
    parser.add_argument("--good-min-tags", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1.")

    image_dirs = parse_image_directories(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rows = []
        for camera_index, image_dir in enumerate(image_dirs):
            files = list_images(image_dir, args.max_frames, args.stride)
            rows.append({
                "camera_index": camera_index,
                "camera_name": image_dir.name,
                "image_dir": str(image_dir),
                "image_count": len(files),
            })
        write_tsv(
            args.output_dir / "dry_run_images.tsv",
            rows,
            ["camera_index", "camera_name", "image_dir", "image_count"])
        (args.output_dir / "client_summary.json").write_text(json.dumps({
            "client_id": args.client_id,
            "mode": "dry_run",
            "camera_count": len(image_dirs),
            "image_count": sum(row["image_count"] for row in rows),
            "stride": args.stride,
            "resize_factor": args.resize_factor,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Dry run OK for {len(image_dirs)} camera directories.")
        return 0

    cv2 = load_cv2()
    detector = create_detector(cv2, args)

    coverage_rows = []
    failed_rows = []
    good_rows = []
    detections_path = args.output_dir / "detections.jsonl"
    start_time = time.time()
    total_frames = 0
    total_detections = 0

    with detections_path.open("w", encoding="utf-8") as out:
        for camera_index, image_dir in enumerate(image_dirs):
            files = list_images(image_dir, args.max_frames, args.stride)
            positive_frames = 0
            camera_detections = 0
            max_tags = 0
            first_positive = ""
            last_positive = ""
            width = 0
            height = 0

            for frame_index, image_path in enumerate(files):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    failed_rows.append({
                        "camera_index": camera_index,
                        "camera_name": image_dir.name,
                        "filename": image_path.name,
                        "reason": "decode_failed",
                    })
                    continue
                height, width = image.shape[:2]
                detection_image, corner_scale = maybe_resize_for_detection(
                    cv2,
                    image,
                    args.resize_factor)
                detections, rejected_count = detect(cv2, detector, detection_image)
                detections = scale_detections(detections, corner_scale)
                detections = refine_detections_subpixel(cv2, image, detections, args)
                tag_count = len(detections)
                if detections:
                    positive_frames += 1
                    if first_positive == "":
                        first_positive = frame_index
                    last_positive = frame_index
                if tag_count >= args.good_min_tags:
                    good_rows.append({
                        "client_id": args.client_id,
                        "camera_index": camera_index,
                        "camera_name": image_dir.name,
                        "source_time": source_time_from_image_dir(image_dir),
                        "image_dir": str(image_dir),
                        "filename": image_path.name,
                        "image_path": str(image_path),
                        "frame_index": frame_index,
                        "tag_count": tag_count,
                        "corner_count": tag_count * 4,
                        "rejected_count": rejected_count,
                    })
                max_tags = max(max_tags, tag_count)
                camera_detections += tag_count
                total_detections += tag_count
                total_frames += 1
                out.write(json.dumps({
                    "client_id": args.client_id,
                    "camera_index": camera_index,
                    "camera_name": image_dir.name,
                    "source_time": source_time_from_image_dir(image_dir),
                    "image_dir": str(image_dir),
                    "filename": image_path.name,
                    "image_path": str(image_path),
                    "frame_index": frame_index,
                    "tag_count": tag_count,
                    "corner_count": tag_count * 4,
                    "width": width,
                    "height": height,
                    "detections": detections,
                    "rejected_count": rejected_count,
                }) + "\n")

            coverage_rows.append({
                "client_id": args.client_id,
                "camera_index": camera_index,
                "camera_name": image_dir.name,
                "image_dir": str(image_dir),
                "width": width,
                "height": height,
                "total_frames": len(files),
                "positive_frames": positive_frames,
                "positive_ratio": positive_frames / len(files) if files else 0.0,
                "total_tags": camera_detections,
                "max_tags": max_tags,
                "first_positive_frame": first_positive,
                "last_positive_frame": last_positive,
            })

    write_tsv(
        args.output_dir / "coverage.tsv",
        coverage_rows,
        [
            "client_id", "camera_index", "camera_name", "image_dir", "width",
            "height", "total_frames", "positive_frames", "positive_ratio",
            "total_tags", "max_tags", "first_positive_frame",
            "last_positive_frame",
        ])
    write_tsv(
        args.output_dir / "failed_images.tsv",
        failed_rows,
        ["camera_index", "camera_name", "filename", "reason"])
    write_tsv(
        args.output_dir / "good_images.tsv",
        good_rows,
        [
            "client_id", "camera_index", "camera_name", "source_time",
            "image_dir", "filename", "image_path", "frame_index",
            "tag_count", "corner_count", "rejected_count",
        ])

    summary = {
        "client_id": args.client_id,
        "mode": "distributed_quality_report",
        "quality_validation_only": True,
        "dictionary": args.dictionary,
        "camera_count": len(image_dirs),
        "total_frames": total_frames,
        "total_detections": total_detections,
        "stride": args.stride,
        "resize_factor": args.resize_factor,
        "detect_inverted": args.detect_inverted,
        "good_min_tags": args.good_min_tags,
        "good_image_count": len(good_rows),
        "elapsed_sec": time.time() - start_time,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "client_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
