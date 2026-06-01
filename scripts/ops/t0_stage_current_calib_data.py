#!/usr/bin/env python3
"""Stage studio calibration captures on t0.

The script creates normalized symlink datasets from the mounted Windows Dshare
folders. It intentionally does not overwrite existing staging roots.
"""

from pathlib import Path
import argparse
import json
import os
import re


OUTER_CAMERAS = [
    ("w4_D", "1-1"), ("w4_D", "1-2"), ("w4_D", "1-3"),
    ("w4_D", "2-1"), ("w4_D", "2-2"), ("w4_D", "2-3"),
    ("w4_D", "3-1"), ("w4_D", "3-2"), ("w4_D", "3-3"),
    ("w4_D", "4-1"), ("w4_D", "4-2"), ("w4_D", "4-3"),
    ("w3_D", "5-1"), ("w3_D", "5-2"), ("w3_D", "5-3"),
    ("w3_D", "6-1"), ("w3_D", "6-2"), ("w3_D", "6-3"),
    ("w3_D", "7-1"), ("w3_D", "7-2"), ("w3_D", "7-3"),
    ("w3_D", "8-1"), ("w3_D", "8-2"), ("w3_D", "8-3"),
]

INNER_CAMERAS = [
    ("w1_D", "22463688"), ("w1_D", "22463690"),
    ("w1_D", "22587611"), ("w1_D", "22587616"),
    ("w2_D", "22463689"), ("w2_D", "22463691"),
    ("w2_D", "22463702"), ("w2_D", "22587614"),
]

SESSIONS = {
    "whole_outer_tower": {
        "source_marker": "whole",
        "time": "2026_05_26-14_08_40",
        "pattern": "apriltag_tower_8faces_2x16_8cm.yaml",
        "cameras": OUTER_CAMERAS,
    },
    "large_marker_bridge_all32": {
        "source_marker": "large_marker",
        "time": "2026_05_26-14_13_47",
        "pattern": "pattern_resolution_17x24_segments_16_apriltag_0.yaml",
        "cameras": OUTER_CAMERAS + INNER_CAMERAS,
    },
    "large_marker_inner8": {
        "source_marker": "large_marker",
        "time": "2026_05_26-14_13_47",
        "pattern": "pattern_resolution_17x24_segments_16_apriltag_0.yaml",
        "cameras": INNER_CAMERAS,
    },
    "small_marker_inner8": {
        "source_marker": "small_marker",
        "time": "2026_05_26-14_16_38",
        "pattern": "pattern_resolution_50x72_segments_16_apriltag_3.yaml",
        "cameras": INNER_CAMERAS,
    },
}

IMAGE_ID_RE = re.compile(r"_(\d+)\.jpe?g$", re.IGNORECASE)


def scan_camera(mount, marker, time, machine, camera_id):
    src = mount / machine / "output" / "calib" / marker / time / camera_id
    if not src.is_dir():
        raise FileNotFoundError(src)

    frames = {}
    for path in sorted(src.iterdir()):
        if not path.is_file():
            continue
        match = IMAGE_ID_RE.search(path.name)
        if not match:
            continue
        frames[int(match.group(1))] = path

    image_ids = sorted(frames)
    contiguous = (
        bool(image_ids)
        and image_ids[0] == 0
        and image_ids == list(range(image_ids[-1] + 1))
    )
    return src, frames, image_ids, contiguous


def stage_session(out_root, mount, name, spec, max_tail_trim):
    scanned = []
    counts = []

    for index, (machine, camera_id) in enumerate(spec["cameras"]):
        src, frames, image_ids, contiguous = scan_camera(
            mount, spec["source_marker"], spec["time"], machine, camera_id)
        scanned.append({
            "index": index,
            "machine": machine,
            "camera_id": camera_id,
            "src": src,
            "frames": frames,
            "ids": image_ids,
            "contiguous": contiguous,
        })
        counts.append(len(image_ids))

    max_count = max(counts)
    valid = []
    excluded = []

    for item in scanned:
        count = len(item["ids"])
        tail_short = max_count - count
        if item["contiguous"] and tail_short <= max_tail_trim:
            valid.append(item)
        else:
            excluded.append({
                "index": item["index"],
                "machine": item["machine"],
                "camera_id": item["camera_id"],
                "count": count,
                "max_count": max_count,
                "contiguous_from_zero": item["contiguous"],
                "reason": "not a valid tail-trim case",
            })

    if not valid:
        raise RuntimeError(f"No valid cameras for {name}")

    common_ids = sorted(set.intersection(*(set(item["ids"]) for item in valid)))
    seq_dir = out_root / name
    images_dir = seq_dir / "images"
    images_dir.mkdir(parents=True)

    manifest_lines = [
        "camera_index\tstage_name\tmachine\tcamera_id\tsource_dir\tframe_count\n"
    ]
    image_dirs = []

    for new_index, item in enumerate(valid):
        machine_label = item["machine"].replace("_D", "")
        stage_name = f"cam{new_index:02d}_{machine_label}_{item['camera_id']}"
        dst_dir = images_dir / stage_name
        dst_dir.mkdir()

        for out_frame, frame_id in enumerate(common_ids):
            src = item["frames"][frame_id]
            dst = dst_dir / f"{out_frame:06d}.jpg"
            os.symlink(src, dst)

        image_dirs.append(str(dst_dir))
        manifest_lines.append(
            f"{new_index}\t{stage_name}\t{item['machine']}\t"
            f"{item['camera_id']}\t{item['src']}\t{len(common_ids)}\n"
        )

    (seq_dir / "manifest.tsv").write_text("".join(manifest_lines))
    (seq_dir / "image_directories.txt").write_text(",".join(image_dirs) + "\n")

    summary = {
        "source_marker": spec["source_marker"],
        "time": spec["time"],
        "pattern": spec["pattern"],
        "max_source_count": max_count,
        "common_frame_count": len(common_ids),
        "valid_camera_count": len(valid),
        "excluded_camera_count": len(excluded),
        "excluded": excluded,
    }
    (seq_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def session_specs(args):
    specs = json.loads(json.dumps(SESSIONS))
    if args.whole_time:
        specs["whole_outer_tower"]["time"] = args.whole_time
    if args.large_marker_time:
        specs["large_marker_bridge_all32"]["time"] = args.large_marker_time
        specs["large_marker_inner8"]["time"] = args.large_marker_time
    if args.small_marker_time:
        specs["small_marker_inner8"]["time"] = args.small_marker_time
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mount-root",
        default="/home/ubuntu/cameras_mount",
        help="Root containing w1_D, w2_D, w3_D, and w4_D CIFS mounts.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3",
        help="New staging root to create. Existing paths are never overwritten.",
    )
    parser.add_argument(
        "--max-tail-trim",
        type=int,
        default=2,
        help="Maximum allowed terminal frame-count difference for a valid camera.",
    )
    parser.add_argument("--whole-time", default="", help="Override whole capture timestamp.")
    parser.add_argument("--large-marker-time", default="", help="Override large_marker capture timestamp.")
    parser.add_argument("--small-marker-time", default="", help="Override small_marker capture timestamp.")
    args = parser.parse_args()

    mount = Path(args.mount_root)
    out_root = Path(args.output_root)
    if out_root.exists():
        raise SystemExit(f"Refusing to overwrite existing staging root: {out_root}")

    out_root.mkdir(parents=True)
    summary = {"staging_root": str(out_root), "sessions": {}}
    for name, spec in session_specs(args).items():
        summary["sessions"][name] = stage_session(
            out_root, mount, name, spec, args.max_tail_trim)

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
