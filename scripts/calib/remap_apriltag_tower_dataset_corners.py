#!/usr/bin/env python3
"""Remap AprilTag corner ids inside a camera_calibration dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import time


def read_exact(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise EOFError("Unexpected end of dataset")
    return data


def read_u32(stream):
    return struct.unpack(">I", read_exact(stream, 4))[0]


def read_i32(stream):
    return struct.unpack(">i", read_exact(stream, 4))[0]


def read_f32(stream):
    return struct.unpack("<f", read_exact(stream, 4))[0]


def u32(value):
    return struct.pack(">I", int(value))


def i32(value):
    return struct.pack(">i", int(value))


def f32(value):
    return struct.pack("<f", float(value))


def remap_feature_id(feature_id, offset):
    tag_id = int(feature_id) // 4
    corner_id = int(feature_id) % 4
    return tag_id * 4 + ((corner_id + int(offset)) % 4)


def remap_dataset(input_path, output_path, offset):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    feature_count_total = 0
    imageset_count = 0
    camera_count = 0
    with input_path.open("rb") as src, output_path.open("wb") as dst:
        header = read_exact(src, 10)
        if header != b"calib_data":
            raise ValueError(f"Invalid dataset header: {input_path}")
        version = read_u32(src)
        if version not in (0, 1):
            raise ValueError(f"Unsupported dataset version {version}: {input_path}")
        dst.write(header)
        dst.write(u32(version))

        camera_count = read_u32(src)
        dst.write(u32(camera_count))
        for _ in range(camera_count):
            width = read_u32(src)
            height = read_u32(src)
            dst.write(u32(width))
            dst.write(u32(height))

        imageset_count = read_u32(src)
        dst.write(u32(imageset_count))
        for _ in range(imageset_count):
            name_len = read_u32(src)
            name = read_exact(src, name_len)
            dst.write(u32(name_len))
            dst.write(name)
            for _camera in range(camera_count):
                feature_count = read_u32(src)
                dst.write(u32(feature_count))
                feature_count_total += feature_count
                for _feature in range(feature_count):
                    x = read_f32(src)
                    y = read_f32(src)
                    feature_id = read_i32(src)
                    dst.write(f32(x))
                    dst.write(f32(y))
                    dst.write(i32(remap_feature_id(feature_id, offset)))

        # Known geometry is already in the target physical corner convention.
        remainder = src.read()
        dst.write(remainder)

    return {
        "mode": "remap_apriltag_tower_dataset_corners",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_dataset": str(input_path),
        "output_dataset": str(output_path),
        "corner_id_offset": int(offset),
        "camera_count": int(camera_count),
        "imageset_count": int(imageset_count),
        "feature_observation_count": int(feature_count_total),
        "elapsed_sec": time.time() - start,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--corner-id-offset", required=True, type=int, choices=[0, 1, 2, 3])
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = remap_dataset(args.input_dataset, args.output_dataset, args.corner_id_offset)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
