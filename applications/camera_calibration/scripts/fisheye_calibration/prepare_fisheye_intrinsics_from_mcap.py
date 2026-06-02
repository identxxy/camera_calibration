#!/usr/bin/env python3
"""Prepare still images for per-camera fisheye intrinsic calibration from MCAP."""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Layout:
    name: str
    tiles: list


@dataclass
class FrameMetrics:
    sharpness: float
    tag_count: int
    board_cx: float = None
    board_cy: float = None
    board_area: float = 0.0
    mean_tag_margin: float = 0.0


@dataclass
class SelectionDecision:
    selected: bool
    reason: str
    board_motion_px: float = None


class CameraFrameSelector:
    def __init__(self, min_sharpness, min_tags, min_board_motion_px, max_selected=0):
        self.min_sharpness = float(min_sharpness)
        self.min_tags = int(min_tags)
        self.min_board_motion_px = float(min_board_motion_px)
        self.max_selected = int(max_selected)
        self.last_selected = None
        self.selected_count = 0

    def should_select(self, metrics, width, height):
        if self.max_selected > 0 and self.selected_count >= self.max_selected:
            return SelectionDecision(False, "max_selected")
        if metrics.sharpness < self.min_sharpness:
            return SelectionDecision(False, "blur")
        if metrics.tag_count < self.min_tags:
            return SelectionDecision(False, "no_board")

        motion = None
        if self.last_selected is not None:
            motion = board_motion_pixels(metrics, self.last_selected, width, height)
            if motion < self.min_board_motion_px:
                return SelectionDecision(False, "near_duplicate", motion)

        self.last_selected = metrics
        self.selected_count += 1
        return SelectionDecision(True, "selected", motion)


class ApriltagCliDetector:
    def __init__(self, executable, temp_dir, quad_decimate=1.0, bits_corrected=2):
        self.executable = str(executable)
        self.temp_dir = Path(temp_dir)
        self.quad_decimate = float(quad_decimate)
        self.bits_corrected = int(bits_corrected)
        self.counter = 0

    def detect(self, gray):
        path = self.temp_dir / f"apriltag_input_{self.counter:08d}.pgm"
        self.counter += 1
        write_pgm(path, gray)
        try:
            result = subprocess.run(
                [self.executable, str(path), str(self.quad_decimate), str(self.bits_corrected)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line[:line.rfind("}") + 1])
        raise RuntimeError(f"AprilTag detector did not emit JSON. stdout={result.stdout!r} stderr={result.stderr!r}")


def repo_root_from_script():
    for parent in Path(__file__).resolve().parents:
        if (parent / "applications/camera_calibration").exists():
            return parent
    raise RuntimeError("Could not locate repository root from script path")


def resolve_layout(layout_name, width, height, camera_count=4):
    if layout_name == "auto":
        if camera_count == 4 and height >= width * 3:
            layout_name = "vertical4"
        elif camera_count == 4 and width >= height * 3:
            layout_name = "horizontal4"
        elif camera_count == 4:
            layout_name = "grid2x2"
        else:
            layout_name = "single"

    if layout_name == "single":
        return Layout("single", [(0, 0, width, height)])
    if layout_name == "vertical4":
        tile_h = height // 4
        return Layout("vertical4", [(0, i * tile_h, width, (i + 1) * tile_h) for i in range(4)])
    if layout_name == "horizontal4":
        tile_w = width // 4
        return Layout("horizontal4", [(i * tile_w, 0, (i + 1) * tile_w, height) for i in range(4)])
    if layout_name == "grid2x2":
        tile_w = width // 2
        tile_h = height // 2
        return Layout("grid2x2", [
            (0, 0, tile_w, tile_h),
            (tile_w, 0, width, tile_h),
            (0, tile_h, tile_w, height),
            (tile_w, tile_h, width, height),
        ])

    raise ValueError(f"Unsupported packed layout: {layout_name}")


def laplacian_variance(gray):
    if gray.size == 0 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    img = gray.astype(np.int16, copy=False)
    lap = (
        img[1:-1, :-2] + img[1:-1, 2:] +
        img[:-2, 1:-1] + img[2:, 1:-1] -
        4 * img[1:-1, 1:-1]
    )
    return float(lap.astype(np.float32).var())


def board_motion_pixels(metrics, previous, width, height):
    if metrics.board_cx is None or previous.board_cx is None:
        return math.inf

    center_motion = math.hypot(metrics.board_cx - previous.board_cx,
                               metrics.board_cy - previous.board_cy)
    scale_motion = 0.0
    if metrics.board_area > 0 and previous.board_area > 0:
        scale_motion = abs(math.log(metrics.board_area / previous.board_area)) * 0.25 * min(width, height)
    return math.hypot(center_motion, scale_motion)


def rgb_to_gray(rgb):
    rgb_u16 = rgb.astype(np.uint16, copy=False)
    gray = (77 * rgb_u16[:, :, 0] + 150 * rgb_u16[:, :, 1] + 29 * rgb_u16[:, :, 2]) >> 8
    return gray.astype(np.uint8)


def parse_binary_pnm(data):
    offset = 0

    def next_token():
        nonlocal offset
        while offset < len(data):
            value = data[offset]
            if value == ord("#"):
                while offset < len(data) and data[offset] not in b"\r\n":
                    offset += 1
            elif chr(value).isspace():
                offset += 1
            else:
                break
        start = offset
        while offset < len(data) and not chr(data[offset]).isspace():
            offset += 1
        return data[start:offset].decode("ascii")

    magic = next_token()
    if magic not in ("P5", "P6"):
        raise ValueError(f"Unsupported PNM magic: {magic}")
    width = int(next_token())
    height = int(next_token())
    max_value = int(next_token())
    if max_value != 255:
        raise ValueError(f"Unsupported PNM max value: {max_value}")
    if offset < len(data) and chr(data[offset]).isspace():
        offset += 1

    channels = 1 if magic == "P5" else 3
    expected = width * height * channels
    payload = data[offset:offset + expected]
    if len(payload) != expected:
        raise ValueError(f"Truncated PNM payload: expected {expected}, got {len(payload)}")
    array = np.frombuffer(payload, dtype=np.uint8)
    if channels == 1:
        return array.reshape((height, width))
    return array.reshape((height, width, channels))


def ppm_bytes(rgb):
    header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii")
    return header + np.ascontiguousarray(rgb).tobytes()


def write_pgm(path, gray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P5\n{gray.shape[1]} {gray.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(gray).tobytes())


def write_image(path, rgb, output_format, convert_bin):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "ppm":
        path.write_bytes(ppm_bytes(rgb))
        return
    if output_format == "png":
        subprocess.run(
            [convert_bin, "ppm:-", str(path)],
            input=ppm_bytes(rgb),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return
    raise ValueError(f"Unsupported output image format: {output_format}")


def decode_jpeg_to_rgb(jpeg_bytes, convert_bin):
    result = subprocess.run(
        [convert_bin, "jpeg:-", "ppm:-"],
        input=jpeg_bytes,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    rgb = parse_binary_pnm(result.stdout)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Decoded JPEG did not produce RGB PPM data")
    return rgb


def discover_compressed_topics(mcap_path):
    from mcap.reader import make_reader

    topics = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, _ in reader.iter_messages(log_time_order=False):
            if schema and schema.name == "sensor_msgs/msg/CompressedImage":
                item = topics.setdefault(channel.topic, {
                    "count": 0,
                    "message_encoding": channel.message_encoding,
                    "schema": schema.name,
                })
                item["count"] += 1
    return topics


def iter_compressed_images(mcap_path, topic):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(
                topics=[topic], log_time_order=True):
            stamp = decoded.header.stamp.sec * 1_000_000_000 + decoded.header.stamp.nanosec
            yield {
                "topic": channel.topic,
                "log_time": message.log_time,
                "publish_time": message.publish_time,
                "stamp": stamp,
                "format": decoded.format,
                "data": bytes(decoded.data),
            }


def build_apriltag_detector(repo_root, output_root):
    apriltag_root = repo_root / "applications/camera_calibration/third_party/apriltag"
    helper_source = Path(__file__).with_name("apriltag_detect_pnm.c")
    executable = output_root / "tools/apriltag_detect_pnm"
    executable.parent.mkdir(parents=True, exist_ok=True)

    common_sources = sorted(glob.glob(str(apriltag_root / "common/*.c")))
    sources = [
        str(helper_source),
        str(apriltag_root / "apriltag.c"),
        str(apriltag_root / "apriltag_quad_thresh.c"),
        str(apriltag_root / "tag36h11.c"),
    ] + common_sources
    cmd = [
        "gcc", "-O3", "-std=gnu99", "-w",
        "-I", str(apriltag_root),
    ] + sources + ["-lm", "-lpthread", "-o", str(executable)]
    subprocess.run(cmd, check=True)
    return executable


def default_build_dir(repo_root):
    env_build_dir = os.environ.get("CAMERA_CALIBRATION_BUILD_DIR")
    if env_build_dir:
        return Path(env_build_dir)

    for build_name in ("build_docker", "build_codex", "build_verify", "build"):
        build_dir = repo_root / build_name
        binary = build_dir / "applications/camera_calibration/camera_calibration"
        if binary.exists():
            return build_dir
    return repo_root / "build"


def make_calibration_scripts(output_root, pattern_files, image_format, model, cell_length, pyramid_levels):
    output_root = Path(output_root)
    scripts_dir = output_root / "calibration_commands"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    all_script = scripts_dir / "run_all_calibrations.sh"
    build_dir = default_build_dir(repo_root_from_script())
    binary_default = os.environ.get(
        "CALIBRATION_BIN",
        str(build_dir / "applications/camera_calibration/camera_calibration"))
    ld_default = os.environ.get(
        "CALIBRATION_LD_LIBRARY_PATH",
        f"{build_dir / 'applications/camera_calibration'}:{build_dir}")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"CALIBRATION_BIN=${{CALIBRATION_BIN:-{binary_default}}}",
        f"export LD_LIBRARY_PATH=${{LD_LIBRARY_PATH:-{ld_default}}}",
        "export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}",
        "",
    ]

    for cam_dir in sorted((output_root / "images").glob("cam*")):
        camera_name = cam_dir.name
        dataset = output_root / "datasets" / f"{camera_name}_features.bin"
        result_dir = output_root / "calibration" / f"{camera_name}_{model}"
        lines.extend([
            f"mkdir -p {json.dumps(str(dataset.parent))} {json.dumps(str(result_dir))}",
            "\"${CALIBRATION_BIN}\" \\",
            f"  --image_directories {json.dumps(str(cam_dir))} \\",
            f"  --pattern_files {json.dumps(str(pattern_files))} \\",
            f"  --dataset_output_path {json.dumps(str(dataset))} \\",
            f"  --output_directory {json.dumps(str(result_dir))} \\",
            f"  --model {model} \\",
            f"  --cell_length_in_pixels {cell_length} \\",
            f"  --num_pyramid_levels {pyramid_levels} \\",
            "  --no_cuda_feature_detection",
            "",
        ])

    all_script.write_text("\n".join(lines), encoding="utf-8")
    all_script.chmod(0o755)
    return all_script


def parse_args():
    repo_root = repo_root_from_script()
    default_patterns = ",".join(
        str(p) for p in sorted((repo_root / "applications/camera_calibration/patterns").glob(
            "pattern_resolution_*_apriltag_*.yaml")))

    parser = argparse.ArgumentParser(
        description="Extract useful per-camera fisheye calibration stills from a packed ROS2 MCAP.")
    parser.add_argument("--mcap", required=True, help="Input MCAP file.")
    parser.add_argument("--output-root", required=True, help="Output directory for selected images and reports.")
    parser.add_argument("--topic", default="", help="CompressedImage topic. Auto-detects when omitted.")
    parser.add_argument("--layout", default="auto",
                        choices=["auto", "single", "vertical4", "horizontal4", "grid2x2"],
                        help="How the four cameras are packed into each decoded image.")
    parser.add_argument("--camera-count", type=int, default=4)
    parser.add_argument("--camera-prefix", default="cam")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth MCAP image message.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed packed frames. 0 means all.")
    parser.add_argument("--max-selected-per-camera", type=int, default=120)
    parser.add_argument("--min-sharpness", type=float, default=60.0)
    parser.add_argument("--sharpness-downsample", type=int, default=2)
    parser.add_argument("--min-tags", type=int, default=1)
    parser.add_argument("--min-board-motion-px", type=float, default=80.0)
    parser.add_argument("--tag-detect-downsample", type=int, default=2)
    parser.add_argument("--tag-quad-decimate", type=float, default=1.0)
    parser.add_argument("--tag-bits-corrected", type=int, default=2)
    parser.add_argument("--image-format", default="png", choices=["png", "ppm"])
    parser.add_argument("--pattern-file", default=default_patterns,
                        help="Comma-separated calibration pattern YAML path(s) for generated commands.")
    parser.add_argument("--model", default="central_generic",
                        choices=["central_generic", "central_thin_prism_fisheye", "central_opencv", "central_radial"])
    parser.add_argument("--cell-length-in-pixels", type=int, default=40)
    parser.add_argument("--num-pyramid-levels", type=int, default=4)
    parser.add_argument("--convert-bin", default=shutil.which("convert") or "convert")
    parser.add_argument("--no-board-detector", action="store_true",
                        help="Skip AprilTag detection. Use only for decoder/debug checks.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model != "central_generic" and args.num_pyramid_levels != 1:
        print(f"Model {args.model} is not grid-based; forcing --num_pyramid_levels 1")
        args.num_pyramid_levels = 1

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_root / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    topics = discover_compressed_topics(args.mcap)
    if args.topic:
        topic = args.topic
    else:
        if len(topics) != 1:
            raise RuntimeError(f"Expected one CompressedImage topic, found {sorted(topics)}. Pass --topic explicitly.")
        topic = next(iter(topics))

    detector = None
    if not args.no_board_detector:
        executable = build_apriltag_detector(repo_root_from_script(), output_root)
        detector = ApriltagCliDetector(
            executable,
            temp_dir,
            quad_decimate=args.tag_quad_decimate,
            bits_corrected=args.tag_bits_corrected,
        )

    layout = None
    selectors = []
    stats = []
    jsonl_files = []
    processed = 0
    seen = 0

    try:
        for image_message in iter_compressed_images(args.mcap, topic):
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            seen += 1
            if args.stride > 1 and (seen - 1) % args.stride != 0:
                continue

            rgb = decode_jpeg_to_rgb(image_message["data"], args.convert_bin)
            height, width = rgb.shape[:2]
            if layout is None:
                layout = resolve_layout(args.layout, width=width, height=height, camera_count=args.camera_count)
                selectors = [
                    CameraFrameSelector(
                        args.min_sharpness,
                        args.min_tags,
                        args.min_board_motion_px,
                        max_selected=args.max_selected_per_camera,
                    )
                    for _ in layout.tiles
                ]
                stats = [
                    {"processed": 0, "selected": 0, "decision_counts": {}}
                    for _ in layout.tiles
                ]
                jsonl_files = [
                    (metadata_dir / f"{args.camera_prefix}{i}_frames.jsonl").open("w", encoding="utf-8")
                    for i in range(len(layout.tiles))
                ]
                print(f"Resolved layout: {layout.name}, full image {width}x{height}, cameras={len(layout.tiles)}")

            for camera_index, tile in enumerate(layout.tiles):
                x0, y0, x1, y1 = tile
                crop = rgb[y0:y1, x0:x1]
                crop_h, crop_w = crop.shape[:2]
                gray = rgb_to_gray(crop)
                sharp_gray = gray[::max(1, args.sharpness_downsample), ::max(1, args.sharpness_downsample)]
                sharpness = laplacian_variance(sharp_gray)

                detection = {"tag_count": 0, "centroid_x": None, "centroid_y": None,
                             "area": 0, "mean_margin": 0, "ids": []}
                if detector is not None:
                    tag_gray = gray[::max(1, args.tag_detect_downsample), ::max(1, args.tag_detect_downsample)]
                    detection = detector.detect(tag_gray)
                    scale = max(1, args.tag_detect_downsample)
                    if detection["centroid_x"] is not None:
                        detection["centroid_x"] *= scale
                        detection["centroid_y"] *= scale
                        detection["area"] *= scale * scale

                metrics = FrameMetrics(
                    sharpness=sharpness,
                    tag_count=int(detection["tag_count"]),
                    board_cx=detection["centroid_x"],
                    board_cy=detection["centroid_y"],
                    board_area=float(detection["area"]),
                    mean_tag_margin=float(detection["mean_margin"]),
                )
                decision = selectors[camera_index].should_select(metrics, crop_w, crop_h)

                stats[camera_index]["processed"] += 1
                stats[camera_index]["decision_counts"].setdefault(decision.reason, 0)
                stats[camera_index]["decision_counts"][decision.reason] += 1
                record = {
                    "packed_frame_index": processed,
                    "mcap_message_index": seen - 1,
                    "stamp_ns": image_message["stamp"],
                    "log_time_ns": image_message["log_time"],
                    "camera_index": camera_index,
                    "tile": tile,
                    "sharpness": sharpness,
                    "tag_count": metrics.tag_count,
                    "board_cx": metrics.board_cx,
                    "board_cy": metrics.board_cy,
                    "board_area": metrics.board_area,
                    "mean_tag_margin": metrics.mean_tag_margin,
                    "tag_ids": detection.get("ids", []),
                    "selected": decision.selected,
                    "reason": decision.reason,
                    "board_motion_px": decision.board_motion_px,
                }

                if decision.selected:
                    stats[camera_index]["selected"] += 1
                    camera_name = f"{args.camera_prefix}{camera_index}"
                    suffix = "ppm" if args.image_format == "ppm" else "png"
                    image_name = f"{selectors[camera_index].selected_count - 1:06d}_src{processed:06d}_{image_message['stamp']}.{suffix}"
                    image_path = output_root / "images" / camera_name / image_name
                    write_image(image_path, crop, args.image_format, args.convert_bin)
                    record["image_path"] = str(image_path)

                jsonl_files[camera_index].write(json.dumps(record, sort_keys=True) + "\n")

            processed += 1
            if processed % 25 == 0:
                selected_counts = [s["selected"] for s in stats]
                print(f"processed packed frames={processed}, selected per camera={selected_counts}", flush=True)
    finally:
        for f in jsonl_files:
            f.close()

    summary = {
        "mcap": str(args.mcap),
        "topic": topic,
        "compressed_topics": topics,
        "layout": layout.name if layout else None,
        "tiles": layout.tiles if layout else [],
        "processed_packed_frames": processed,
        "seen_messages": seen,
        "settings": {
            "pattern_files": args.pattern_file,
            "min_sharpness": args.min_sharpness,
            "min_tags": args.min_tags,
            "min_board_motion_px": args.min_board_motion_px,
            "tag_detect_downsample": args.tag_detect_downsample,
            "max_selected_per_camera": args.max_selected_per_camera,
        },
        "cameras": stats,
    }
    summary_path = metadata_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    calibration_script = make_calibration_scripts(
        output_root,
        args.pattern_file,
        args.image_format,
        args.model,
        args.cell_length_in_pixels,
        args.num_pyramid_levels,
    )
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote calibration command script: {calibration_script}")
    print(json.dumps(summary["cameras"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
