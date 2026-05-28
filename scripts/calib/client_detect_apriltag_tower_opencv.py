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


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resize-factor", type=float, default=1.0)
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
                if detections:
                    positive_frames += 1
                    if first_positive == "":
                        first_positive = frame_index
                    last_positive = frame_index
                max_tags = max(max_tags, len(detections))
                camera_detections += len(detections)
                total_detections += len(detections)
                total_frames += 1
                out.write(json.dumps({
                    "client_id": args.client_id,
                    "camera_index": camera_index,
                    "camera_name": image_dir.name,
                    "filename": image_path.name,
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
