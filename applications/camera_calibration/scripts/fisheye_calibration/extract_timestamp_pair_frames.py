#!/usr/bin/env python3
"""Extract timestamp-matched H264 frames for pairwise SL fisheye rig calibration."""

import argparse
import importlib.util
import json
import os
import struct
import subprocess
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
from mcap.reader import make_reader


SCRIPT_DIR = Path(__file__).resolve().parent
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_fisheye_intrinsics_from_mcap.py"
SPEC = importlib.util.spec_from_file_location("prepare_fisheye_intrinsics_from_mcap", PREPARE_SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)

TOPICS = {
    "left_up": "/camera/left_up/h264",
    "left_down": "/camera/left_down/h264",
    "right_down": "/camera/right_down/h264",
    "right_up": "/camera/right_up/h264",
}


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
            raise RuntimeError(f"Truncated PPM frame")
        yield np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))


def load_frame_timestamps(mcap_path):
    out = {name: [] for name in TOPICS}
    topic_to_name = {topic: name for name, topic in TOPICS.items()}
    with Path(mcap_path).open("rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(log_time_order=True):
            name = topic_to_name.get(channel.topic)
            if name is None:
                continue
            sec, nsec = struct.unpack_from("<II", bytes(message.data), 0)
            out[name].append(int(sec) * 1_000_000_000 + int(nsec))
    return out


def load_selected(metadata_root, camera):
    rows = []
    with (Path(metadata_root) / f"{camera}_frames.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("selected"):
                rows.append(int(row["original_frame_index"]))
    return rows


def nearest_index(stamps, target_ns):
    pos = bisect_left(stamps, target_ns)
    candidates = []
    for idx in (pos - 1, pos, pos + 1):
        if 0 <= idx < len(stamps):
            candidates.append((abs(stamps[idx] - target_ns), idx))
    if not candidates:
        return None, None
    return min(candidates)


def build_pair_records(pair_name, left, right, selected_root, frame_stamps, threshold_ns):
    selected_root = Path(selected_root)
    left_selected = load_selected(selected_root / "metadata", left)
    right_selected = load_selected(selected_root / "metadata", right)
    records = {}

    for left_idx in left_selected:
        delta, right_idx = nearest_index(frame_stamps[right], frame_stamps[left][left_idx])
        if right_idx is not None and delta <= threshold_ns:
            records[(left_idx, right_idx)] = {
                "pair": pair_name,
                "left_camera": left,
                "right_camera": right,
                "left_frame": int(left_idx),
                "right_frame": int(right_idx),
                "left_timestamp_ns": int(frame_stamps[left][left_idx]),
                "right_timestamp_ns": int(frame_stamps[right][right_idx]),
                "delta_ns": int(delta),
                "anchor": left,
            }

    for right_idx in right_selected:
        delta, left_idx = nearest_index(frame_stamps[left], frame_stamps[right][right_idx])
        if left_idx is not None and delta <= threshold_ns:
            records.setdefault((left_idx, right_idx), {
                "pair": pair_name,
                "left_camera": left,
                "right_camera": right,
                "left_frame": int(left_idx),
                "right_frame": int(right_idx),
                "left_timestamp_ns": int(frame_stamps[left][left_idx]),
                "right_timestamp_ns": int(frame_stamps[right][right_idx]),
                "delta_ns": int(delta),
                "anchor": right,
            })

    return list(records.values())


def extract_camera_frames(raw_root, output_root, camera, indices):
    indices = sorted(set(int(i) for i in indices))
    if not indices:
        return {}
    out_dir = Path(output_root) / "frames" / camera
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(indices)
    existing = {}
    for idx in indices:
        path = out_dir / f"{camera}_{idx:06d}.png"
        if path.exists():
            existing[idx] = str(path.resolve())
    if set(existing) == wanted:
        return existing

    max_index = max(indices)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(Path(raw_root) / f"{camera}.h264"),
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
    log_path = Path(output_root) / f"{camera}_extract_ffmpeg.log"
    written = {}
    written.update(existing)
    wanted_to_decode = wanted - set(existing)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)
        try:
            for idx, rgb in enumerate(iter_ppm_stream(proc.stdout)):
                if idx in wanted_to_decode:
                    path = out_dir / f"{camera}_{idx:06d}.png"
                    prepare.write_image(path, rgb, "png", "convert")
                    written[idx] = str(path.resolve())
                    if len(written) == len(wanted):
                        proc.terminate()
                        break
                if idx >= max_index and len(written) == len(wanted):
                    proc.terminate()
                    break
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
    missing = sorted(wanted - set(written))
    if missing:
        raise RuntimeError(f"{camera}: missing extracted frames {missing[:10]} ({len(missing)} total)")
    return written


def symlink_force(src, dst):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(src, dst.parent), dst)


def materialize_pair_dirs(records_by_pair, extracted, output_root):
    summaries = []
    for pair_name, records in records_by_pair.items():
        if not records:
            continue
        left = records[0]["left_camera"]
        right = records[0]["right_camera"]
        pair_dir = Path(output_root) / "pairs" / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)
        manifest = pair_dir / "manifest.tsv"
        with manifest.open("w", encoding="utf-8") as f:
            f.write("src\tleft_frame\tright_frame\tleft_timestamp_ns\tright_timestamp_ns\tdelta_ns\tanchor\n")
            for seq, record in enumerate(sorted(records, key=lambda r: (r["left_timestamp_ns"], r["right_timestamp_ns"]))):
                filename = f"src{seq:06d}.png"
                symlink_force(extracted[left][record["left_frame"]], pair_dir / left / filename)
                symlink_force(extracted[right][record["right_frame"]], pair_dir / right / filename)
                f.write(
                    f"{seq}\t{record['left_frame']}\t{record['right_frame']}\t"
                    f"{record['left_timestamp_ns']}\t{record['right_timestamp_ns']}\t"
                    f"{record['delta_ns']}\t{record['anchor']}\n"
                )
        summaries.append({
            "pair": pair_name,
            "left_camera": left,
            "right_camera": right,
            "records": len(records),
            "max_delta_ns": max(r["delta_ns"] for r in records),
            "manifest": str(manifest.resolve()),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--selected-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threshold-ms", type=float, default=2.0)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame_stamps = load_frame_timestamps(args.mcap)
    threshold_ns = int(args.threshold_ms * 1_000_000)
    records_by_pair = {
        "top_left_right": build_pair_records("top_left_right", "left_up", "right_up", args.selected_root, frame_stamps, threshold_ns),
        "bottom_left_right": build_pair_records("bottom_left_right", "left_down", "right_down", args.selected_root, frame_stamps, threshold_ns),
    }
    needed = {camera: set() for camera in TOPICS}
    for records in records_by_pair.values():
        for record in records:
            needed[record["left_camera"]].add(record["left_frame"])
            needed[record["right_camera"]].add(record["right_frame"])

    extracted = {}
    for camera, indices in needed.items():
        extracted[camera] = extract_camera_frames(args.raw_root, output_root, camera, indices)

    summary = materialize_pair_dirs(records_by_pair, extracted, output_root)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
