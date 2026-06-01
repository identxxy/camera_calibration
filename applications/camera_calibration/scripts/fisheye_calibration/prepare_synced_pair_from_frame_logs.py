#!/usr/bin/env python3
"""Create synchronized pair image directories from MCAP frame JSONL logs."""

import argparse
import json
import os
from pathlib import Path


def read_selected_by_src(path, min_tags, min_sharpness):
    selected = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("selected"):
                continue
            if int(record.get("tag_count") or 0) < min_tags:
                continue
            if float(record.get("sharpness") or 0.0) < min_sharpness:
                continue
            image_path = record.get("image_path")
            if not image_path:
                continue
            selected[int(record["packed_frame_index"])] = record
    return selected


def relink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.unlink()
    except FileNotFoundError:
        pass
    os.symlink(os.path.relpath(src, dst.parent), dst)


def main():
    parser = argparse.ArgumentParser(
        description="Create same-filename two-camera dirs from prepare_fisheye_intrinsics_from_mcap JSONL logs.")
    parser.add_argument("--log-a", required=True)
    parser.add_argument("--log-b", required=True)
    parser.add_argument("--cam-a", required=True)
    parser.add_argument("--cam-b", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pair-name", required=True)
    parser.add_argument("--min-tags", type=int, default=1)
    parser.add_argument("--min-sharpness", type=float, default=0.0)
    parser.add_argument("--trim-before-ns", type=int, default=0,
                        help="Drop images before this timestamp.")
    parser.add_argument("--trim-after-ns", type=int, default=0,
                        help="Drop images after this timestamp. 0 disables the upper bound.")
    args = parser.parse_args()

    a = read_selected_by_src(args.log_a, args.min_tags, args.min_sharpness)
    b = read_selected_by_src(args.log_b, args.min_tags, args.min_sharpness)
    output_root = Path(args.output_root).resolve() / args.pair_name
    dir_a = output_root / args.cam_a
    dir_b = output_root / args.cam_b

    common = sorted(set(a) & set(b))
    kept = []
    for src in common:
        rec_a = a[src]
        rec_b = b[src]
        stamp_ns = int(rec_a["stamp_ns"])
        if args.trim_before_ns and stamp_ns < args.trim_before_ns:
            continue
        if args.trim_after_ns and stamp_ns > args.trim_after_ns:
            continue
        out_name = f"src{src:06d}.png"
        relink(Path(rec_a["image_path"]), dir_a / out_name)
        relink(Path(rec_b["image_path"]), dir_b / out_name)
        kept.append({
            "index": len(kept),
            "src": src,
            "timestamp_ns": stamp_ns,
            "cam_a_stamp_ns": int(rec_a["stamp_ns"]),
            "cam_b_stamp_ns": int(rec_b["stamp_ns"]),
            "cam_a_image": str(Path(rec_a["image_path"])),
            "cam_b_image": str(Path(rec_b["image_path"])),
            "cam_a_tag_count": int(rec_a.get("tag_count") or 0),
            "cam_b_tag_count": int(rec_b.get("tag_count") or 0),
            "cam_a_sharpness": float(rec_a.get("sharpness") or 0.0),
            "cam_b_sharpness": float(rec_b.get("sharpness") or 0.0),
        })

    manifest_tsv = output_root / "manifest.tsv"
    manifest_json = output_root / "manifest.json"
    manifest_tsv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_tsv.open("w", encoding="utf-8") as f:
        f.write("index\tsrc\ttimestamp_ns\tcam_a_stamp_ns\tcam_b_stamp_ns\tcam_a_tag_count\tcam_b_tag_count\tcam_a_sharpness\tcam_b_sharpness\n")
        for item in kept:
            f.write(
                f"{item['index']}\t{item['src']}\t{item['timestamp_ns']}\t"
                f"{item['cam_a_stamp_ns']}\t{item['cam_b_stamp_ns']}\t"
                f"{item['cam_a_tag_count']}\t{item['cam_b_tag_count']}\t"
                f"{item['cam_a_sharpness']:.6f}\t{item['cam_b_sharpness']:.6f}\n")
    manifest_json.write_text(json.dumps({
        "pair_name": args.pair_name,
        "cam_a": args.cam_a,
        "cam_b": args.cam_b,
        "common": len(common),
        "kept": len(kept),
        "trim_before_ns": args.trim_before_ns,
        "trim_after_ns": args.trim_after_ns,
        "output_root": str(output_root),
        "frames": kept,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"pair={args.pair_name}")
    print(f"cam_a={args.cam_a} cam_b={args.cam_b}")
    print(f"common={len(common)} kept={len(kept)}")
    print(f"output={output_root}")


if __name__ == "__main__":
    main()
