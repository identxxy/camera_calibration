#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


CAMERA_ROWS = {
    "cam0": "left_up",
    "cam1": "left_down",
    "cam2": "right_down",
    "cam3": "right_up",
}


def load_intrinsics(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cameras"]


def matrix_mul(a, b):
    out = [[0.0 for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for c in range(4):
            out[r][c] = sum(a[r][k] * b[k][c] for k in range(4))
    return out


def matrix_transpose3(m):
    out = [[0.0 for _ in range(4)] for _ in range(4)]
    for r in range(3):
        for c in range(3):
            out[r][c] = m[c][r]
    out[3][3] = 1.0
    return out


def invert_transform(t):
    inv = matrix_transpose3(t)
    for r in range(3):
        inv[r][3] = -sum(inv[r][c] * t[c][3] for c in range(3))
    return inv


def load_reference_transforms(path):
    # These transforms are placeholders from the previous camera-only rig file,
    # but they provide useful camera-camera chain initialization.
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {name: entry["T_cam_imu"] for name, entry in data.items() if name.startswith("cam")}


def format_float(value):
    text = f"{float(value):.16g}"
    return "0.0" if text == "-0" else text


def yaml_scalar(value):
    if isinstance(value, str):
        return value
    return format_float(value)


def write_list(lines, indent, values):
    lines.append(" " * indent + "- [" + ", ".join(format_float(v) for v in values) + "]")


def write_matrix(lines, key, matrix):
    lines.append(f"  {key}:")
    for row in matrix:
        write_list(lines, 4, row)


def write_camera_entry(lines, key, source_cam, topic, intrinsics, t_cn_cnm1=None):
    entry = intrinsics[source_cam]
    params = entry["params"]
    lines.append(f"{key}:")
    lines.append("  camera_model: pinhole")
    lines.append("  distortion_model: equidistant")
    lines.append("  intrinsics: [" + ", ".join(format_float(v) for v in params[:4]) + "]")
    lines.append("  distortion_coeffs: [" + ", ".join(format_float(v) for v in params[4:8]) + "]")
    lines.append(f"  resolution: [{int(entry['width'])}, {int(entry['height'])}]")
    lines.append(f"  rostopic: {topic}")
    if t_cn_cnm1 is not None:
        write_matrix(lines, "T_cn_cnm1", t_cn_cnm1)
    lines.append("")


def write_camchain(path, pair_name, source_cams, intrinsics, transforms, topic_prefix):
    first, second = source_cams
    first_t_ref = transforms[first]
    second_t_ref = transforms[second]
    t_second_first = matrix_mul(second_t_ref, invert_transform(first_t_ref))

    lines = [
        f"# Generated for {pair_name}.",
        f"# cam0 maps to original {first} / {CAMERA_ROWS.get(first, 'unknown')}.",
        f"# cam1 maps to original {second} / {CAMERA_ROWS.get(second, 'unknown')}.",
    ]
    write_camera_entry(lines, "cam0", first, f"{topic_prefix}{first[-1]}/image_raw", intrinsics)
    write_camera_entry(lines, "cam1", second, f"{topic_prefix}{second[-1]}/image_raw", intrinsics,
                       t_cn_cnm1=t_second_first)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return t_second_first


def write_aprilgrid(path, tag_size, tag_spacing, tag_rows, tag_cols):
    text = "\n".join([
        "target_type: 'aprilgrid'",
        f"tagCols: {int(tag_cols)}",
        f"tagRows: {int(tag_rows)}",
        f"tagSize: {format_float(tag_size)}",
        f"tagSpacing: {format_float(tag_spacing)}",
        "",
    ])
    Path(path).write_text(text, encoding="utf-8")


def write_imu(path, topic, update_rate):
    text = "\n".join([
        f"rostopic: {topic}",
        f"update_rate: {format_float(update_rate)}",
        "accelerometer_noise_density: 0.02",
        "accelerometer_random_walk: 0.002",
        "gyroscope_noise_density: 0.001",
        "gyroscope_random_walk: 0.0001",
        "",
    ])
    Path(path).write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kalibr YAML inputs for Seeker VI pair bags.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--intrinsics-json", required=True)
    parser.add_argument("--reference-camchain", required=True)
    parser.add_argument("--pair-name", required=True, choices=["up_cam0_cam3", "down_cam1_cam2"])
    parser.add_argument("--tag-size", type=float, default=0.011882352941176469 * 4.0)
    parser.add_argument("--tag-spacing", type=float, default=0.3)
    parser.add_argument("--tag-rows", type=int, default=3)
    parser.add_argument("--tag-cols", type=int, default=3)
    parser.add_argument("--imu-topic", default="/imu0")
    parser.add_argument("--topic-prefix", default="/cam")
    parser.add_argument("--imu-rate", type=float, default=200.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = load_intrinsics(args.intrinsics_json)
    transforms = load_reference_transforms(args.reference_camchain)
    pairs = {
        "up_cam0_cam3": ("cam0", "cam3"),
        "down_cam1_cam2": ("cam1", "cam2"),
    }
    source_cams = pairs[args.pair_name]

    camchain = output_dir / "camchain.yaml"
    target = output_dir / "aprilgrid_1x1_tag0.yaml"
    imu = output_dir / "imu.yaml"
    t_second_first = write_camchain(
        camchain, args.pair_name, source_cams, intrinsics, transforms, args.topic_prefix)
    write_aprilgrid(target, args.tag_size, args.tag_spacing, args.tag_rows, args.tag_cols)
    write_imu(imu, args.imu_topic, args.imu_rate)

    manifest = {
        "pair_name": args.pair_name,
        "source_cams": source_cams,
        "camchain": str(camchain),
        "target": str(target),
        "imu": str(imu),
        "target_note": (
            "Kalibr requires at least 3x3 AprilGrid config. The current desktop board "
            "visibly provides tag0 observations; this is a compatibility target for "
            "partial AprilTag extraction, not a true printed 3x3 grid."
        ),
        "camera_model": "Kalibr pinhole+equidistant initialized from exported KB8 intrinsics.",
        "T_second_first_initial": t_second_first,
    }
    (output_dir / "kalibr_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
