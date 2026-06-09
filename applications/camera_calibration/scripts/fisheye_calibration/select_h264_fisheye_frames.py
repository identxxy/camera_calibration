#!/usr/bin/env python3
"""Select sharp, board-visible stills from local H264 fisheye streams."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_fisheye_intrinsics_from_mcap.py"
SPEC = importlib.util.spec_from_file_location("prepare_fisheye_intrinsics_from_mcap", PREPARE_SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def next_token(stream):
    token = bytearray()
    in_comment = False
    while True:
        b = stream.read(1)
        if not b:
            return None
        c = b[0]
        if in_comment:
            if c in (10, 13):
                in_comment = False
            continue
        if c == ord("#"):
            in_comment = True
            continue
        if chr(c).isspace():
            if token:
                return token.decode("ascii")
            continue
        token.append(c)


def iter_ppm_stream(stream):
    while True:
        magic = next_token(stream)
        if magic is None:
            return
        if magic != "P6":
            raise RuntimeError(f"Expected P6 frame, got {magic!r}")
        width = int(next_token(stream))
        height = int(next_token(stream))
        max_value = int(next_token(stream))
        if max_value != 255:
            raise RuntimeError(f"Unsupported PPM max value: {max_value}")
        payload = stream.read(width * height * 3)
        if len(payload) != width * height * 3:
            raise RuntimeError(f"Truncated PPM frame: expected {width * height * 3}, got {len(payload)}")
        yield np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))


def select_stream(args, camera_name, h264_path, detector):
    out_dir = args.output_root / "images" / camera_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "metadata" / f"{camera_name}_frames.jsonl"
    ffmpeg_log_path = args.output_root / "metadata" / f"{camera_name}_ffmpeg.log"
    selector = prepare.CameraFrameSelector(
        min_sharpness=args.min_sharpness,
        min_tags=args.min_tags,
        min_board_motion_px=args.min_board_motion_px,
        max_selected=args.max_selected_per_camera,
    )
    stats = {
        "camera": camera_name,
        "source": str(h264_path),
        "processed": 0,
        "selected": 0,
        "decision_counts": {},
    }
    vf = f"select=not(mod(n\\,{args.stride}))"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(h264_path),
        "-vf",
        vf,
        "-vsync",
        "0",
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "ppm",
        "-",
    ]
    with ffmpeg_log_path.open("w", encoding="utf-8") as ffmpeg_log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=ffmpeg_log)
        with log_path.open("w", encoding="utf-8") as log_file:
            try:
                for decoded_index, rgb in enumerate(iter_ppm_stream(proc.stdout)):
                    original_index = decoded_index * args.stride
                    gray = prepare.rgb_to_gray(rgb)
                    sharp_gray = gray[::args.sharpness_downsample, ::args.sharpness_downsample]
                    sharpness = prepare.laplacian_variance(sharp_gray)
                    tag_gray = gray[::args.tag_detect_downsample, ::args.tag_detect_downsample]
                    detection = detector.detect(tag_gray)
                    scale = args.tag_detect_downsample
                    if detection["centroid_x"] is not None:
                        detection["centroid_x"] *= scale
                        detection["centroid_y"] *= scale
                        detection["area"] *= scale * scale
                    metrics = prepare.FrameMetrics(
                        sharpness=sharpness,
                        tag_count=int(detection["tag_count"]),
                        board_cx=detection["centroid_x"],
                        board_cy=detection["centroid_y"],
                        board_area=float(detection["area"]),
                        mean_tag_margin=float(detection["mean_margin"]),
                    )
                    decision = selector.should_select(metrics, rgb.shape[1], rgb.shape[0])
                    stats["processed"] += 1
                    stats["decision_counts"].setdefault(decision.reason, 0)
                    stats["decision_counts"][decision.reason] += 1
                    image_path = None
                    if decision.selected:
                        stats["selected"] += 1
                        image_path = out_dir / f"{camera_name}_{original_index:06d}.png"
                        prepare.write_image(image_path, rgb, "png", args.convert_bin)
                    record = {
                        "camera": camera_name,
                        "decoded_index": decoded_index,
                        "original_frame_index": original_index,
                        "sharpness": sharpness,
                        "tag_count": int(detection["tag_count"]),
                        "tag_ids": detection["ids"],
                        "board_cx": detection["centroid_x"],
                        "board_cy": detection["centroid_y"],
                        "board_area": detection["area"],
                        "mean_tag_margin": detection["mean_margin"],
                        "selected": bool(decision.selected),
                        "decision": decision.reason,
                        "board_motion_px": decision.board_motion_px,
                        "image_path": str(image_path) if image_path else None,
                    }
                    log_file.write(json.dumps(record, sort_keys=True) + "\n")
                    if stats["selected"] >= args.max_selected_per_camera:
                        proc.terminate()
                        break
            finally:
                if proc.stdout:
                    proc.stdout.close()
                proc.wait()
    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-selected-per-camera", type=int, default=140)
    parser.add_argument("--min-sharpness", type=float, default=110.0)
    parser.add_argument("--min-tags", type=int, default=1)
    parser.add_argument("--min-board-motion-px", type=float, default=90.0)
    parser.add_argument("--sharpness-downsample", type=int, default=2)
    parser.add_argument("--tag-detect-downsample", type=int, default=2)
    parser.add_argument("--tag-quad-decimate", type=float, default=1.0)
    parser.add_argument("--tag-bits-corrected", type=int, default=2)
    parser.add_argument("--convert-bin", default="convert")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata").mkdir(parents=True, exist_ok=True)
    detector_exe = prepare.build_apriltag_detector(REPO_ROOT, args.output_root)
    detector = prepare.ApriltagCliDetector(
        detector_exe,
        args.output_root / "tmp",
        quad_decimate=args.tag_quad_decimate,
        bits_corrected=args.tag_bits_corrected,
    )
    cameras = ["left_up", "left_down", "right_down", "right_up"]
    summary = []
    for camera_name in cameras:
        stats = select_stream(args, camera_name, args.raw_root / f"{camera_name}.h264", detector)
        summary.append(stats)
        print(json.dumps(stats, sort_keys=True))
    summary_path = args.output_root / "metadata" / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
