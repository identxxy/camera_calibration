#!/usr/bin/env python3
"""Create synchronized two-camera image dirs from per-camera feature logs."""

import argparse
import os
import re
from pathlib import Path


FEATURE_RE = re.compile(r"/([^/\s]+_src(\d+)_\d+\.png):\s+(\d+) features")


def parse_log(path):
    by_src = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        match = FEATURE_RE.search(line)
        if not match:
            continue
        filename = match.group(1)
        src = int(match.group(2))
        features = int(match.group(3))
        by_src[src] = (filename, features)
    return by_src


def relink(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.unlink()
    except FileNotFoundError:
        pass
    os.symlink(os.path.relpath(src, dst.parent), dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--log-a", required=True)
    parser.add_argument("--log-b", required=True)
    parser.add_argument("--cam-a", type=int, required=True)
    parser.add_argument("--cam-b", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pair-name", required=True)
    parser.add_argument("--min-features", type=int, default=1)
    args = parser.parse_args()

    image_root = Path(args.image_root)
    output_root = Path(args.output_root) / args.pair_name
    a_dir = output_root / f"cam{args.cam_a}"
    b_dir = output_root / f"cam{args.cam_b}"

    a = parse_log(args.log_a)
    b = parse_log(args.log_b)
    common = sorted(set(a) & set(b))
    kept = []
    for src in common:
        a_filename, a_features = a[src]
        b_filename, b_features = b[src]
        if a_features < args.min_features or b_features < args.min_features:
            continue
        out_name = f"src{src:06d}.png"
        relink(image_root / f"cam{args.cam_a}" / a_filename, a_dir / out_name)
        relink(image_root / f"cam{args.cam_b}" / b_filename, b_dir / out_name)
        kept.append({
            "src": src,
            f"cam{args.cam_a}_features": a_features,
            f"cam{args.cam_b}_features": b_features,
        })

    manifest = output_root / "manifest.tsv"
    with manifest.open("w") as f:
        f.write("src\tcam_a_features\tcam_b_features\n")
        for item in kept:
            f.write(
                f"{item['src']}\t"
                f"{item[f'cam{args.cam_a}_features']}\t"
                f"{item[f'cam{args.cam_b}_features']}\n"
            )

    print(f"pair={args.pair_name}")
    print(f"cam_a=cam{args.cam_a} cam_b=cam{args.cam_b}")
    print(f"common={len(common)} kept={len(kept)}")
    print(f"output={output_root}")


if __name__ == "__main__":
    main()
