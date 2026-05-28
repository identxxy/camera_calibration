#!/usr/bin/env python3
"""Parallel feature extraction orchestration for camera_calibration.

This script runs the existing C++ feature extractor once per camera directory,
then merges the single-camera dataset shards into the normal synchronized
multi-camera dataset using --merge_single_camera_datasets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def parse_image_directories(args):
    if args.image_directories_file:
        text = Path(args.image_directories_file).read_text().strip()
    else:
        text = args.image_directories or ""
    dirs = [Path(item).expanduser() for item in text.split(",") if item.strip()]
    if not dirs:
        raise SystemExit("No image directories were provided.")
    for path in dirs:
        if not path.is_dir():
            raise SystemExit(f"Image directory does not exist: {path}")
    return dirs


def build_extract_command(args, image_dir, shard_path):
    command = [
        str(args.binary),
        "--image_directories",
        str(image_dir),
        "--dataset_output_path",
        str(shard_path),
        "--refinement_window_half_extent",
        str(args.refinement_window_half_extent),
    ]
    if args.apriltag_tower_config:
        command += ["--apriltag_tower_config", str(args.apriltag_tower_config)]
    if args.pattern_files:
        command += ["--pattern_files", str(args.pattern_files)]
    if args.feature_refinement_type:
        command += ["--feature_refinement_type", args.feature_refinement_type]
    if args.no_cuda_feature_detection:
        command += ["--no_cuda_feature_detection"]
    return command


def run_one(args, camera_index, image_dir, shard_path, log_path):
    if args.resume and shard_path.exists() and log_path.exists():
        return camera_index, 0, "resume"

    command = build_extract_command(args, image_dir, shard_path)
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    start = time.time()
    with log_path.open("w") as log:
        log.write("command: " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=args.repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    return camera_index, completed.returncode, f"{elapsed:.1f}s"


def merge_shards(args, shard_paths):
    command = [
        str(args.binary),
        "--merge_single_camera_datasets",
        "--dataset_files",
        ",".join(str(path) for path in shard_paths),
        "--dataset_output_path",
        str(args.output_dataset),
    ]
    print("Merging shards:")
    print("  " + " ".join(command))
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(command, cwd=args.repo_root, env=env).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--image-directories")
    parser.add_argument("--image-directories-file", type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--apriltag-tower-config", type=Path)
    parser.add_argument("--pattern-files")
    parser.add_argument("--refinement-window-half-extent", type=int, default=15)
    parser.add_argument("--feature-refinement-type", default="")
    parser.add_argument("--no-cuda-feature-detection", action="store_true")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.repo_root = args.repo_root.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    args.output_dataset = args.output_dataset.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()

    if bool(args.apriltag_tower_config) == bool(args.pattern_files):
        raise SystemExit("Provide exactly one of --apriltag-tower-config or --pattern-files.")
    if not args.binary.is_file():
        raise SystemExit(f"Binary does not exist: {args.binary}")
    if args.output_dataset.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite output dataset: {args.output_dataset}")
    if args.work_dir.exists() and args.overwrite and not args.resume:
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    image_dirs = parse_image_directories(args)
    shard_paths = [
        args.work_dir / f"camera_{camera_index:02d}.bin"
        for camera_index in range(len(image_dirs))
    ]
    log_paths = [
        args.work_dir / f"camera_{camera_index:02d}.log"
        for camera_index in range(len(image_dirs))
    ]

    print(f"Extracting {len(image_dirs)} camera shards with {args.jobs} jobs.")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = []
        for camera_index, image_dir in enumerate(image_dirs):
            futures.append(executor.submit(
                run_one,
                args,
                camera_index,
                image_dir,
                shard_paths[camera_index],
                log_paths[camera_index]))
        for future in concurrent.futures.as_completed(futures):
            camera_index, returncode, detail = future.result()
            status = "ok" if returncode == 0 else "failed"
            print(f"camera {camera_index:02d}: {status} ({detail})")
            if returncode != 0:
                failures.append(camera_index)

    if failures:
        print(f"Failed camera shards: {failures}", file=sys.stderr)
        return 1

    for path in shard_paths:
        if not path.exists():
            print(f"Missing shard: {path}", file=sys.stderr)
            return 1

    return merge_shards(args, shard_paths)


if __name__ == "__main__":
    raise SystemExit(main())
