#!/usr/bin/env python3
"""Rewrite AprilTag tower dataset corner ids and known 3D geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import time

import build_apriltag_tower_dataset_opencv as tower_builder


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


def parse_corner_permutation(text):
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if sorted(values) != [0, 1, 2, 3]:
        raise ValueError(f"Corner permutation must contain 0,1,2,3 exactly once: {text}")
    return values


def remap_feature_id(feature_id, permutation):
    tag_id = int(feature_id) // 4
    corner_id = int(feature_id) % 4
    return tag_id * 4 + permutation[corner_id]


def build_config(args):
    config = tower_builder.read_tower_config(args.tower_config)
    if args.face_width_m is not None:
        config["face_width_m"] = float(args.face_width_m)
    if args.tag_rotation_degrees is not None:
        config["tag_rotation_degrees"] = int(args.tag_rotation_degrees)
    return config


def rewrite_dataset(args):
    permutation = parse_corner_permutation(args.corner_id_permutation)
    config = build_config(args)
    tower_points = tower_builder.build_tower_points(config)
    cell_length = float(config["tag_size_m"]) + float(config["tag_spacing_m"])

    start = time.time()
    feature_count_total = 0
    imageset_count = 0
    camera_count = 0
    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    with args.input_dataset.open("rb") as src, args.output_dataset.open("wb") as dst:
        header = read_exact(src, 10)
        if header != b"calib_data":
            raise ValueError(f"Invalid dataset header: {args.input_dataset}")
        version = read_u32(src)
        if version not in (0, 1):
            raise ValueError(f"Unsupported dataset version {version}: {args.input_dataset}")
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
                    dst.write(i32(remap_feature_id(feature_id, permutation)))

        if version >= 1:
            # Discard existing known geometry and write the requested tower model.
            geometry_count = read_u32(src)
            for _geometry in range(geometry_count):
                _cell_length = read_f32(src)
                fixed_coord_count = read_u32(src)
                for _fixed in range(fixed_coord_count):
                    _ = read_i32(src)
                    _ = read_i32(src)
                    _ = read_i32(src)
                point_count = read_u32(src)
                for _point in range(point_count):
                    _ = read_i32(src)
                    _ = read_f32(src)
                    _ = read_f32(src)
                    _ = read_f32(src)
            trailing = src.read()
            if trailing:
                raise ValueError("Unexpected trailing bytes after known geometry.")
            dst.write(u32(1))
            dst.write(f32(cell_length))
            dst.write(u32(0))
            dst.write(u32(len(tower_points)))
            for feature_id in sorted(tower_points):
                x, y, z = tower_points[feature_id]
                dst.write(i32(feature_id))
                dst.write(f32(x))
                dst.write(f32(y))
                dst.write(f32(z))

    return {
        "mode": "rewrite_apriltag_tower_dataset_geometry",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_dataset": str(args.input_dataset),
        "output_dataset": str(args.output_dataset),
        "tower_config": str(args.tower_config),
        "corner_id_permutation": permutation,
        "face_width_m": float(config["face_width_m"]),
        "tag_rotation_degrees": int(config["tag_rotation_degrees"]),
        "camera_count": int(camera_count),
        "imageset_count": int(imageset_count),
        "feature_observation_count": int(feature_count_total),
        "known_3d_point_count": int(len(tower_points)),
        "elapsed_sec": time.time() - start,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--tower-config", required=True, type=Path)
    parser.add_argument("--corner-id-permutation", default="0,1,2,3")
    parser.add_argument("--face-width-m", type=float)
    parser.add_argument("--tag-rotation-degrees", type=int)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = rewrite_dataset(args)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
