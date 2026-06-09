#!/usr/bin/env python3
"""Optimize up/down shared-center roll phi from panorama seam photometric loss."""

import argparse
import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
RENDER_SCRIPT = THIS_DIR / "render_sl_kb8_panoramas.py"
SPEC = importlib.util.spec_from_file_location("render_sl_kb8_panoramas", RENDER_SCRIPT)
render = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = render
SPEC.loader.exec_module(render)


def rz(degrees):
    rad = math.radians(float(degrees))
    c = math.cos(rad)
    s = math.sin(rad)
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    return out


def back_y():
    out = np.eye(4, dtype=float)
    out[:3, :3] = np.diag([-1.0, 1.0, -1.0])
    return out


def invert_t(T):
    return render.invert_t(T)


def as_matrix_list(T):
    return [[float(v) for v in row] for row in np.asarray(T, dtype=float)]


def apply_phi_to_camchain(camchain, left_phi_deg, right_phi_deg):
    data = copy.deepcopy(camchain)
    by_name = {entry["name"]: entry for entry in data.values()}
    R_back = back_y()
    for up_name, down_name, phi in (
        ("left_up", "left_down", left_phi_deg),
        ("right_up", "right_down", right_phi_deg),
    ):
        T_up_ref = np.asarray(by_name[up_name]["T_cam_ref"], dtype=float)
        T_down_ref = rz(phi) @ R_back @ T_up_ref
        by_name[down_name]["T_cam_ref"] = as_matrix_list(T_down_ref)
        by_name[down_name]["T_ref_cam"] = as_matrix_list(invert_t(T_down_ref))
        by_name[down_name]["up_down_phi_z_deg"] = float(phi)
        by_name[down_name]["up_down_phi_model"] = "T_down_up = Rz(phi) * Ry(180deg), shared optical center"
    return data


def camchain_to_cams(camchain):
    cams = {}
    for entry in camchain.values():
        name = entry["name"]
        fx, fy, cx, cy = [float(v) for v in entry["intrinsics"]]
        k = [float(v) for v in entry["distortion_coeffs"]]
        cams[name] = {
            "width": int(entry["resolution"][0]),
            "height": int(entry["resolution"][1]),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k": k,
            "T_cam_ref": np.asarray(entry["T_cam_ref"], dtype=float),
        }
    return cams


def select_evenly(rows, count):
    if len(rows) <= count:
        return list(rows)
    indices = sorted(set(round(i * (len(rows) - 1) / (count - 1)) for i in range(count)))
    while len(indices) < count:
        for idx in range(len(rows)):
            if idx not in indices:
                indices.append(idx)
                break
        indices = sorted(indices)
    return [rows[i] for i in indices[:count]]


def rgb_to_gray(rgb):
    return (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )


def load_gray_frames(raw_root, camera, indices):
    indices = sorted(set(int(v) for v in indices))
    if not indices:
        return {}
    proc, gen = render.start_decoder(raw_root, camera)
    frames = {}
    current = -1
    max_index = max(indices)
    wanted = set(indices)
    try:
        while current < max_index:
            rgb = next(gen)
            current += 1
            if current in wanted:
                frames[current] = rgb_to_gray(rgb)
                if len(frames) == len(wanted):
                    break
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.terminate()
        proc.wait()
    missing = sorted(wanted - set(frames))
    if missing:
        raise RuntimeError(f"{camera}: missing decoded frames {missing[:8]}")
    return frames


def load_pair_gray_frames(raw_root, pair_rows, up_name, down_name):
    up_indices = [row[up_name] for row in pair_rows]
    down_indices = [row[down_name] for row in pair_rows]
    return {
        up_name: load_gray_frames(raw_root, up_name, up_indices),
        down_name: load_gray_frames(raw_root, down_name, down_indices),
    }


def bilinear_sample_gray(img, mx, my, valid):
    h, w = img.shape[:2]
    x0 = np.floor(mx).astype(np.int32)
    y0 = np.floor(my).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = valid & (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    wx = (mx - x0).astype(np.float32)
    wy = (my - y0).astype(np.float32)
    top = img[y0c, x0c] * (1.0 - wx) + img[y0c, x1c] * wx
    bottom = img[y1c, x0c] * (1.0 - wx) + img[y1c, x1c] * wx
    out = top * (1.0 - wy) + bottom * wy
    out[~valid] = 0.0
    return out, valid


def dilate_seam(mask, radius):
    out = mask.copy()
    for _ in range(max(0, int(radius))):
        expanded = out.copy()
        expanded[:, 1:] |= out[:, :-1]
        expanded[:, :-1] |= out[:, 1:]
        expanded[1:, :] |= out[:-1, :]
        expanded[:-1, :] |= out[1:, :]
        out = expanded
    return out


def seam_band(maps, up_name, down_name, seam_width_px):
    up = maps[up_name]
    down = maps[down_name]
    overlap = up["valid"] & down["valid"]
    winner = down["weight"] > up["weight"]
    seam = np.zeros_like(overlap, dtype=bool)
    seam[:, 1:] |= overlap[:, 1:] & overlap[:, :-1] & (winner[:, 1:] != winner[:, :-1])
    seam[:, :-1] |= overlap[:, 1:] & overlap[:, :-1] & (winner[:, 1:] != winner[:, :-1])
    seam[1:, :] |= overlap[1:, :] & overlap[:-1, :] & (winner[1:, :] != winner[:-1, :])
    seam[:-1, :] |= overlap[1:, :] & overlap[:-1, :] & (winner[1:, :] != winner[:-1, :])
    radius = max(1, int(round(seam_width_px / 2.0)))
    band = dilate_seam(seam, radius) & overlap
    return band


def robust_overlap_loss(up_img, down_img, mask):
    values = (down_img - up_img)[mask]
    if values.size < 100:
        return None
    offset = float(np.median(values))
    residual = np.abs(values - offset)
    return float(np.median(residual))


def candidate_loss(base_camchain, raw_root, pair_rows, gray_frames, up_name, down_name, phi_deg, width, height, seam_width_px):
    if up_name.startswith("left"):
        camchain = apply_phi_to_camchain(base_camchain, phi_deg, 0.0)
    else:
        camchain = apply_phi_to_camchain(base_camchain, 0.0, phi_deg)
    cams = camchain_to_cams(camchain)
    maps = render.build_maps(cams, [up_name, down_name], width, height)
    band = seam_band(maps, up_name, down_name, seam_width_px)
    if int(np.sum(band)) < 100:
        return {
            "phi_deg": float(phi_deg),
            "loss": float("inf"),
            "frames": 0,
            "seam_pixels": int(np.sum(band)),
        }
    losses = []
    for row in pair_rows:
        up_gray = gray_frames[up_name][row[up_name]]
        down_gray = gray_frames[down_name][row[down_name]]
        up_sample, up_valid = bilinear_sample_gray(up_gray, maps[up_name]["x"], maps[up_name]["y"], maps[up_name]["valid"])
        down_sample, down_valid = bilinear_sample_gray(down_gray, maps[down_name]["x"], maps[down_name]["y"], maps[down_name]["valid"])
        mask = band & up_valid & down_valid
        loss = robust_overlap_loss(up_sample, down_sample, mask)
        if loss is not None and math.isfinite(loss):
            losses.append(loss)
    if not losses:
        value = float("inf")
    else:
        value = float(np.median(np.asarray(losses, dtype=float)))
    return {
        "phi_deg": float(phi_deg),
        "loss": value,
        "frames": len(losses),
        "seam_pixels": int(np.sum(band)),
    }


def optimize_side(base_camchain, raw_root, pair_rows, up_name, down_name, args):
    pair_rows = select_evenly(pair_rows, args.opt_frames)
    gray_frames = load_pair_gray_frames(raw_root, pair_rows, up_name, down_name)

    evaluated = []
    coarse = np.arange(args.min_phi_deg, args.max_phi_deg + 0.5 * args.coarse_step_deg, args.coarse_step_deg)
    for phi in coarse:
        evaluated.append(candidate_loss(
            base_camchain, raw_root, pair_rows, gray_frames, up_name, down_name,
            float(phi), args.width, args.height, args.seam_width_px))
    best = min(evaluated, key=lambda row: row["loss"])

    lo = best["phi_deg"] - args.coarse_step_deg
    hi = best["phi_deg"] + args.coarse_step_deg
    fine = np.arange(lo, hi + 0.5 * args.fine_step_deg, args.fine_step_deg)
    for phi in fine:
        if args.min_phi_deg <= phi <= args.max_phi_deg:
            evaluated.append(candidate_loss(
                base_camchain, raw_root, pair_rows, gray_frames, up_name, down_name,
                float(phi), args.width, args.height, args.seam_width_px))
    best = min(evaluated, key=lambda row: row["loss"])
    baseline = min(
        evaluated,
        key=lambda row: abs(row["phi_deg"]),
    )
    return {
        "up_camera": up_name,
        "down_camera": down_name,
        "best_phi_deg": float(best["phi_deg"]),
        "best_loss": float(best["loss"]),
        "baseline_phi_deg": float(baseline["phi_deg"]),
        "baseline_loss": float(baseline["loss"]),
        "loss_reduction": float(baseline["loss"] - best["loss"]),
        "opt_frames": len(pair_rows),
        "panorama_resolution": [int(args.width), int(args.height)],
        "seam_width_px": int(args.seam_width_px),
        "evaluations": sorted(evaluated, key=lambda row: row["phi_deg"]),
    }


def write_camchain(path, camchain):
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(camchain, f, sort_keys=False, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--base-camchain-yaml", required=True)
    parser.add_argument("--output-yaml", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seam-width-px", type=int, default=20)
    parser.add_argument("--max-common-frames", type=int, default=600)
    parser.add_argument("--opt-frames", type=int, default=12)
    parser.add_argument("--min-phi-deg", type=float, default=-180.0)
    parser.add_argument("--max-phi-deg", type=float, default=180.0)
    parser.add_argument("--coarse-step-deg", type=float, default=5.0)
    parser.add_argument("--fine-step-deg", type=float, default=0.5)
    args = parser.parse_args()

    base_camchain = yaml.safe_load(Path(args.base_camchain_yaml).read_text(encoding="utf-8"))
    frame_stamps = render.load_frame_timestamps(args.mcap)
    left_rows = render.common_index_pairs(frame_stamps, "left_up", "left_down", args.max_common_frames)
    right_rows = render.common_index_pairs(frame_stamps, "right_up", "right_down", args.max_common_frames)
    if len(left_rows) < args.opt_frames or len(right_rows) < args.opt_frames:
        raise RuntimeError(f"Not enough synchronized frames: left={len(left_rows)}, right={len(right_rows)}")

    left = optimize_side(base_camchain, args.raw_root, left_rows, "left_up", "left_down", args)
    right = optimize_side(base_camchain, args.raw_root, right_rows, "right_up", "right_down", args)
    optimized = apply_phi_to_camchain(base_camchain, left["best_phi_deg"], right["best_phi_deg"])
    write_camchain(args.output_yaml, optimized)

    summary = {
        "base_camchain_yaml": str(Path(args.base_camchain_yaml).resolve()),
        "optimized_camchain_yaml": str(Path(args.output_yaml).resolve()),
        "model": "T_down_up = Rz(phi) * Ry(180deg), shared optical center",
        "left": left,
        "right": right,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
