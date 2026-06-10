#!/usr/bin/env python3
"""Render left/right equirectangular panorama preview videos from an SL KB8 rig."""

import argparse
import json
import math
import struct
import subprocess
from pathlib import Path

import numpy as np
import yaml


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
            raise RuntimeError("Truncated PPM frame")
        yield np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))


def load_frame_timestamps(mcap_path):
    from mcap.reader import make_reader

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


def invert_t(T):
    out = np.eye(4, dtype=float)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def load_camchain(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cams = {}
    for _key, item in data.items():
        name = item["name"]
        fx, fy, cx, cy = [float(v) for v in item["intrinsics"]]
        k = [float(v) for v in item["distortion_coeffs"]]
        T = np.asarray(item["T_cam_ref"], dtype=float)
        cams[name] = {
            "width": int(item["resolution"][0]),
            "height": int(item["resolution"][1]),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k": k,
            "T_cam_ref": T,
        }
    return cams


def kb8_project_dirs(dirs_cam, intr):
    x = dirs_cam[..., 0]
    y = dirs_cam[..., 1]
    z = dirs_cam[..., 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    theta2 = theta * theta
    k1, k2, k3, k4 = intr["k"]
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4)
    scale = np.zeros_like(theta_d)
    ok = rho > 1e-12
    scale[ok] = theta_d[ok] / rho[ok]
    mx = scale * x
    my = scale * y
    u = intr["fx"] * mx + intr["cx"]
    v = intr["fy"] * my + intr["cy"]
    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (v >= 0)
        & (u < intr["width"] - 1)
        & (v < intr["height"] - 1)
    )
    # Fisheye lenses can validly image rays beyond the pinhole front hemisphere.
    # Use a smooth angular confidence that reaches zero at 180 degrees.
    weight = np.clip(np.cos(0.5 * theta), 0.0, 1.0)
    return u.astype(np.float32), v.astype(np.float32), valid, weight.astype(np.float32)


def build_maps(cams, pair, width, height):
    ref = pair[0]
    T_ref_cam = invert_t(cams[ref]["T_cam_ref"])
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    lon = (u[None, :] - 0.5) * (2.0 * math.pi)
    lat = (v[:, None] - 0.5) * math.pi
    cos_lat = np.cos(lat)
    dirs_ref = np.empty((height, width, 3), dtype=np.float32)
    dirs_ref[..., 0] = np.sin(lon) * cos_lat
    dirs_ref[..., 1] = np.sin(lat)
    dirs_ref[..., 2] = np.cos(lon) * cos_lat

    maps = {}
    for cam in pair:
        T_cam_pair_ref = cams[cam]["T_cam_ref"] @ T_ref_cam
        R = T_cam_pair_ref[:3, :3].astype(np.float32)
        dirs_cam = dirs_ref @ R.T
        mx, my, valid, weight = kb8_project_dirs(dirs_cam, cams[cam])
        maps[cam] = {"x": mx, "y": my, "valid": valid, "weight": weight * weight}
    return maps


def bilinear_sample(img, mx, my, valid):
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
    top = img[y0c, x0c].astype(np.float32) * (1.0 - wx[..., None]) + img[y0c, x1c].astype(np.float32) * wx[..., None]
    bottom = img[y1c, x0c].astype(np.float32) * (1.0 - wx[..., None]) + img[y1c, x1c].astype(np.float32) * wx[..., None]
    out = top * (1.0 - wy[..., None]) + bottom * wy[..., None]
    out[~valid] = 0.0
    return out, valid


def render_frame(images, maps, pair):
    accum = None
    weights = None
    for cam in pair:
        sampled, valid = bilinear_sample(images[cam], maps[cam]["x"], maps[cam]["y"], maps[cam]["valid"])
        weight = np.where(valid, maps[cam]["weight"], 0.0)
        if accum is None:
            accum = sampled * weight[..., None]
            weights = weight
        else:
            accum += sampled * weight[..., None]
            weights += weight
    denom = np.maximum(weights[..., None], 1e-6)
    return np.clip(accum / denom, 0, 255).astype(np.uint8)


def start_decoder(raw_root, camera):
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc, iter_ppm_stream(proc.stdout)


def start_encoder(path, width, height, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_video(raw_root, output_path, cams, pair, index_pairs, width, height, fps):
    maps = build_maps(cams, pair, width, height)
    decoders = {}
    for cam in pair:
        decoders[cam] = start_decoder(raw_root, cam)
    encoder = start_encoder(output_path, width, height, fps)
    current = {cam: -1 for cam in pair}
    frames = {}
    try:
        for out_idx, pair_indices in enumerate(index_pairs):
            images = {}
            for cam in pair:
                target = pair_indices[cam]
                proc, gen = decoders[cam]
                while current[cam] < target:
                    frames[cam] = next(gen)
                    current[cam] += 1
                images[cam] = frames[cam]
            pano = render_frame(images, maps, pair)
            encoder.stdin.write(pano.tobytes())
            if (out_idx + 1) % 100 == 0:
                print(f"{Path(output_path).name}: {out_idx + 1} frames", flush=True)
    finally:
        if encoder.stdin:
            encoder.stdin.close()
        encoder.wait()
        for proc, _gen in decoders.values():
            if proc.stdout:
                proc.stdout.close()
            proc.terminate()
            proc.wait()


def exact_index_by_timestamp(stamps):
    return {stamp: idx for idx, stamp in enumerate(stamps)}


def common_index_pairs(frame_stamps, a, b, max_frames):
    b_by_stamp = exact_index_by_timestamp(frame_stamps[b])
    rows = []
    for idx_a, stamp in enumerate(frame_stamps[a]):
        idx_b = b_by_stamp.get(stamp)
        if idx_b is not None:
            rows.append({a: idx_a, b: idx_b, "timestamp_ns": stamp})
            if len(rows) >= max_frames:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--camchain-yaml", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    cams = load_camchain(args.camchain_yaml)
    frame_stamps = load_frame_timestamps(args.mcap)
    output_root = Path(args.output_root)
    left_pairs = common_index_pairs(frame_stamps, "left_up", "left_down", args.max_frames)
    right_pairs = common_index_pairs(frame_stamps, "right_up", "right_down", args.max_frames)
    left_path = output_root / "left_panorama_kb8_preview.mp4"
    right_path = output_root / "right_panorama_kb8_preview.mp4"
    render_video(args.raw_root, left_path, cams, ["left_up", "left_down"], left_pairs, args.width, args.height, args.fps)
    render_video(args.raw_root, right_path, cams, ["right_up", "right_down"], right_pairs, args.width, args.height, args.fps)
    summary = {
        "left_video": str(left_path.resolve()),
        "right_video": str(right_path.resolve()),
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "frames": {
            "left": len(left_pairs),
            "right": len(right_pairs),
        },
        "camchain_yaml": str(Path(args.camchain_yaml).resolve()),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
