#!/usr/bin/env python3
"""Select high-quality synchronized frames from outer tower BA diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import time


RESIDUAL_KEYS = (
    "residual_px",
    "after_residual_px",
    "residual_norm_px",
    "reprojection_error_px",
    "reprojection_residual_px",
    "error_px",
    "norm_px",
    "after_error_px",
)


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of calib_data file")
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


def read_dataset(path):
    with Path(path).open("rb") as stream:
        header = read_exact(stream, 10)
        if header != b"calib_data":
            raise ValueError(f"Invalid dataset header: {path}")
        version = read_u32(stream)
        if version not in (0, 1):
            raise ValueError(f"Unsupported calib_data version {version}: {path}")
        camera_count = read_u32(stream)
        image_sizes = [(read_u32(stream), read_u32(stream)) for _ in range(camera_count)]

        imagesets = []
        for _ in range(read_u32(stream)):
            name_len = read_u32(stream)
            filename = read_exact(stream, name_len).decode("utf-8")
            camera_features = []
            for _camera_index in range(camera_count):
                features = []
                for _feature_index in range(read_u32(stream)):
                    features.append((read_f32(stream), read_f32(stream), read_i32(stream)))
                camera_features.append(features)
            imagesets.append({"filename": filename, "features": camera_features})

        geometry_blocks = []
        for _block_index in range(read_u32(stream)):
            block = {
                "cell_length": read_f32(stream),
                "topology_items": [],
                "known_points": [],
            }
            for _item_index in range(read_u32(stream)):
                block["topology_items"].append((read_i32(stream), read_i32(stream), read_i32(stream)))
            if version >= 1:
                for _point_index in range(read_u32(stream)):
                    block["known_points"].append(
                        (read_i32(stream), read_f32(stream), read_f32(stream), read_f32(stream)))
            geometry_blocks.append(block)

        extra = stream.read(1)
        if extra:
            raise ValueError(f"Unexpected trailing bytes in calib_data file: {path}")

    return {
        "version": version,
        "camera_count": camera_count,
        "image_sizes": image_sizes,
        "imagesets": imagesets,
        "geometry_blocks": geometry_blocks,
    }


def write_dataset(path, dataset, imagesets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"calib_data")
        stream.write(u32(dataset["version"]))
        stream.write(u32(dataset["camera_count"]))
        for width, height in dataset["image_sizes"]:
            stream.write(u32(width))
            stream.write(u32(height))

        stream.write(u32(len(imagesets)))
        for imageset in imagesets:
            encoded = imageset["filename"].encode("utf-8")
            stream.write(u32(len(encoded)))
            stream.write(encoded)
            if len(imageset["features"]) != dataset["camera_count"]:
                raise ValueError(
                    f"Imageset {imageset['filename']} has {len(imageset['features'])} cameras, "
                    f"expected {dataset['camera_count']}")
            for features in imageset["features"]:
                stream.write(u32(len(features)))
                for x, y, feature_id in features:
                    stream.write(f32(x))
                    stream.write(f32(y))
                    stream.write(i32(feature_id))

        stream.write(u32(len(dataset["geometry_blocks"])))
        for block in dataset["geometry_blocks"]:
            stream.write(f32(block["cell_length"]))
            stream.write(u32(len(block["topology_items"])))
            for a, b, c in block["topology_items"]:
                stream.write(i32(a))
                stream.write(i32(b))
                stream.write(i32(c))
            if dataset["version"] >= 1:
                stream.write(u32(len(block["known_points"])))
                for feature_id, x, y, z in block["known_points"]:
                    stream.write(i32(feature_id))
                    stream.write(f32(x))
                    stream.write(f32(y))
                    stream.write(f32(z))


def as_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def as_int(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(values, q):
    clean = sorted(value for value in values if as_float(value) is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])
    position = (len(clean) - 1) * q / 100.0
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(clean[lo])
    weight = position - lo
    return float(clean[lo]) * (1.0 - weight) + float(clean[hi]) * weight


def fmt_optional_float(value):
    if value is None:
        return ""
    return f"{float(value):.6f}"


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def residual_key_for_row(row):
    for key in RESIDUAL_KEYS:
        if key in row:
            return key
    return None


def load_frame_quality(path, dataset):
    rows = {}
    for row in read_tsv(path):
        frame_index = as_int(row.get("frame_index", row.get("imageset_index")))
        if frame_index is None:
            continue
        if frame_index < 0 or frame_index >= len(dataset["imagesets"]):
            raise ValueError(f"frame_quality frame_index {frame_index} is outside dataset range")
        filename = row.get("filename", "")
        if filename and filename != dataset["imagesets"][frame_index]["filename"]:
            raise ValueError(
                "frame_quality filename does not match dataset imageset: "
                f"index {frame_index}, diagnostics={filename}, dataset={dataset['imagesets'][frame_index]['filename']}")
        rows[frame_index] = row
    return rows


def residual_row_in_scope(row, residual_scope):
    if residual_scope == "all":
        return True
    if residual_scope == "ok":
        status = str(row.get("projection_status", "")).strip().lower()
        return status in ("", "ok", "valid")
    if residual_scope == "used_after_gate":
        used = str(row.get("used_after_gate", "")).strip().lower()
        if used in ("1", "true", "yes", "y"):
            return True
        if used in ("0", "false", "no", "n"):
            return False
        status = str(row.get("projection_status", "")).strip().lower()
        return status in ("", "ok", "valid")
    raise ValueError(f"Unsupported residual scope: {residual_scope}")


def load_residual_stats(path, dataset, residual_scope):
    values_by_frame = {index: [] for index in range(len(dataset["imagesets"]))}
    camera_counts_by_frame = {index: {} for index in range(len(dataset["imagesets"]))}
    skipped_rows = 0
    for row in read_tsv(path):
        if not residual_row_in_scope(row, residual_scope):
            skipped_rows += 1
            continue
        frame_index = as_int(row.get("frame_index", row.get("imageset_index")))
        if frame_index is None or frame_index < 0 or frame_index >= len(dataset["imagesets"]):
            skipped_rows += 1
            continue
        residual_key = residual_key_for_row(row)
        residual_px = as_float(row.get(residual_key)) if residual_key else None
        if residual_px is None:
            skipped_rows += 1
            continue
        values_by_frame[frame_index].append(residual_px)
        camera_index = as_int(row.get("camera_index", row.get("camera_idx")))
        if camera_index is not None:
            camera_counts = camera_counts_by_frame[frame_index]
            camera_counts[camera_index] = camera_counts.get(camera_index, 0) + 1

    stats = {}
    for frame_index, values in values_by_frame.items():
        stats[frame_index] = {
            "observation_count": len(values),
            "median_px": percentile(values, 50),
            "p90_px": percentile(values, 90),
            "camera_observation_counts": camera_counts_by_frame[frame_index],
        }
    return stats, skipped_rows


def active_from_quality(row):
    if row is None:
        return False
    if "active" not in row:
        return True
    return str(row.get("active", "")).strip().lower() in ("1", "true", "yes", "y")


def frame_candidate_rows(dataset, frame_quality, residual_stats, args):
    rows = []
    rejection_counts = {}
    for frame_index, imageset in enumerate(dataset["imagesets"]):
        quality = frame_quality.get(frame_index)
        stats = residual_stats.get(frame_index, {})
        camera_counts = stats.get("camera_observation_counts", {})
        qualifying_camera_count = sum(
            1 for count in camera_counts.values()
            if count >= args.camera_min_observations)
        observation_count = int(stats.get("observation_count", 0))
        median_px = stats.get("median_px")
        p90_px = stats.get("p90_px")
        active = active_from_quality(quality)
        reasons = []
        if quality is None:
            reasons.append("missing_frame_quality")
        if not active:
            reasons.append("inactive")
        if observation_count < args.min_frame_observations:
            reasons.append("too_few_observations")
        if qualifying_camera_count < args.min_frame_cameras:
            reasons.append("too_few_cameras")
        if median_px is None:
            reasons.append("missing_median_residual")
        elif median_px > args.max_frame_median_px:
            reasons.append("median_px_too_high")
        if p90_px is None:
            reasons.append("missing_p90_residual")
        elif p90_px > args.max_frame_p90_px:
            reasons.append("p90_px_too_high")

        for reason in reasons:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        rows.append({
            "old_imageset_index": frame_index,
            "filename": imageset["filename"],
            "active": "yes" if active else "no",
            "feature_count": "" if quality is None else quality.get("feature_count", ""),
            "pnp_vote_count": "" if quality is None else quality.get("pnp_vote_count", ""),
            "pnp_median_error_px": "" if quality is None else quality.get("pnp_median_error_px", ""),
            "observation_count": observation_count,
            "camera_count": qualifying_camera_count,
            "median_px": median_px,
            "p90_px": p90_px,
            "rejection_reasons": ",".join(reasons),
            "selected": not reasons,
        })
    return rows, rejection_counts


def select_frame_rows(candidate_rows, limit_frames):
    selected = [row for row in candidate_rows if row["selected"]]
    if limit_frames and limit_frames > 0:
        ranked = sorted(
            selected,
            key=lambda row: (
                float("inf") if row["median_px"] is None else row["median_px"],
                float("inf") if row["p90_px"] is None else row["p90_px"],
                -row["observation_count"],
                row["old_imageset_index"],
            ))
        selected_set = {row["old_imageset_index"] for row in ranked[:limit_frames]}
        selected = [row for row in selected if row["old_imageset_index"] in selected_set]
    selected = sorted(selected, key=lambda row: row["old_imageset_index"])
    for new_index, row in enumerate(selected):
        row["new_imageset_index"] = new_index
    return selected


def selected_frame_output_rows(selected_rows):
    rows = []
    for row in selected_rows:
        rows.append({
            "new_imageset_index": row["new_imageset_index"],
            "old_imageset_index": row["old_imageset_index"],
            "filename": row["filename"],
            "active": row["active"],
            "feature_count": row["feature_count"],
            "pnp_vote_count": row["pnp_vote_count"],
            "pnp_median_error_px": row["pnp_median_error_px"],
            "observation_count": row["observation_count"],
            "camera_count": row["camera_count"],
            "median_px": fmt_optional_float(row["median_px"]),
            "p90_px": fmt_optional_float(row["p90_px"]),
        })
    return rows


def filter_pnp_views(pnp_views, output_path, old_to_new, dataset):
    input_rows = read_tsv(pnp_views)
    if not input_rows:
        fieldnames = ["imageset_index"]
    else:
        fieldnames = list(input_rows[0].keys())
    if "imageset_index" not in fieldnames:
        raise ValueError(f"PnP views TSV must contain imageset_index: {pnp_views}")

    output_rows = []
    invalid_rows = 0
    for row in input_rows:
        old_index = as_int(row.get("imageset_index"))
        if old_index is None or old_index < 0 or old_index >= len(dataset["imagesets"]):
            invalid_rows += 1
            continue
        filename = row.get("filename", "")
        if filename and filename != dataset["imagesets"][old_index]["filename"]:
            raise ValueError(
                "PnP view filename does not match dataset imageset: "
                f"index {old_index}, pnp={filename}, dataset={dataset['imagesets'][old_index]['filename']}")
        if old_index not in old_to_new:
            continue
        output = dict(row)
        output["imageset_index"] = str(old_to_new[old_index])
        if "filename" in fieldnames:
            output["filename"] = dataset["imagesets"][old_index]["filename"]
        output_rows.append(output)

    write_tsv(output_path, output_rows, fieldnames)
    return {
        "input": str(pnp_views),
        "output": str(output_path),
        "input_rows": len(input_rows),
        "kept_rows": len(output_rows),
        "dropped_rows": len(input_rows) - len(output_rows),
        "invalid_rows": invalid_rows,
    }


def run(args):
    start = time.time()
    dataset = read_dataset(args.dataset)
    diagnostics_dir = args.refine_dir / "diagnostics"
    frame_quality_path = diagnostics_dir / "frame_quality.tsv"
    residuals_path = diagnostics_dir / "observation_residuals.tsv"
    if not frame_quality_path.exists():
        raise FileNotFoundError(f"Missing frame quality diagnostics: {frame_quality_path}")
    if not residuals_path.exists():
        raise FileNotFoundError(f"Missing observation residual diagnostics: {residuals_path}")

    frame_quality = load_frame_quality(frame_quality_path, dataset)
    residual_stats, skipped_residual_rows = load_residual_stats(
        residuals_path,
        dataset,
        args.residual_scope)
    candidate_rows, rejection_counts = frame_candidate_rows(dataset, frame_quality, residual_stats, args)
    selected_rows = select_frame_rows(candidate_rows, args.limit_frames)
    if not selected_rows and not args.allow_empty:
        raise SystemExit(
            "No frames passed selection. Re-run with looser thresholds or --allow_empty to write an empty subset.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = [row["old_imageset_index"] for row in selected_rows]
    selected_imagesets = [dataset["imagesets"][index] for index in selected_indices]
    output_dataset = args.output_dir / "dataset_subset.bin"
    write_dataset(output_dataset, dataset, selected_imagesets)

    selected_tsv = args.output_dir / "selected_frames.tsv"
    write_tsv(
        selected_tsv,
        selected_frame_output_rows(selected_rows),
        [
            "new_imageset_index",
            "old_imageset_index",
            "filename",
            "active",
            "feature_count",
            "pnp_vote_count",
            "pnp_median_error_px",
            "observation_count",
            "camera_count",
            "median_px",
            "p90_px",
        ],
    )

    old_to_new = {row["old_imageset_index"]: row["new_imageset_index"] for row in selected_rows}
    pnp_summary = None
    if args.pnp_views:
        pnp_summary = filter_pnp_views(
            args.pnp_views,
            args.output_dir / "pnp_views_subset.tsv",
            old_to_new,
            dataset)

    selected_camera_counts = {}
    for old_index in selected_indices:
        for camera_index, count in residual_stats[old_index]["camera_observation_counts"].items():
            selected_camera_counts[camera_index] = selected_camera_counts.get(camera_index, 0) + count

    summary = {
        "mode": "outer_tower_high_quality_subset",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": time.time() - start,
        "inputs": {
            "dataset": str(args.dataset),
            "refine_dir": str(args.refine_dir),
            "frame_quality": str(frame_quality_path),
            "observation_residuals": str(residuals_path),
            "pnp_views": str(args.pnp_views) if args.pnp_views else "",
        },
        "outputs": {
            "dataset": str(output_dataset),
            "selected_frames": str(selected_tsv),
            "pnp_views_subset": str(args.output_dir / "pnp_views_subset.tsv") if args.pnp_views else "",
            "summary": str(args.output_dir / "summary.json"),
        },
        "settings": {
            "max_frame_median_px": args.max_frame_median_px,
            "max_frame_p90_px": args.max_frame_p90_px,
            "min_frame_observations": args.min_frame_observations,
            "min_frame_cameras": args.min_frame_cameras,
            "camera_min_observations": args.camera_min_observations,
            "limit_frames": args.limit_frames,
            "allow_empty": bool(args.allow_empty),
            "residual_scope": args.residual_scope,
        },
        "dataset_version": dataset["version"],
        "camera_count": dataset["camera_count"],
        "input_frame_count": len(dataset["imagesets"]),
        "selected_frame_count": len(selected_rows),
        "selected_old_imageset_indices": selected_indices,
        "residual_rows_skipped": skipped_residual_rows,
        "rejection_counts": rejection_counts,
        "selected_observation_count": int(sum(row["observation_count"] for row in selected_rows)),
        "selected_camera_observations": {
            str(camera_index): int(count)
            for camera_index, count in sorted(selected_camera_counts.items())
        },
        "geometry_block_count": len(dataset["geometry_blocks"]),
        "pnp_views": pnp_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select high-quality synchronized imagesets from an outer AprilTag tower "
            "refine output and write a remapped calib_data subset."
        ))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--refine_dir", "--refine-dir", required=True, type=Path)
    parser.add_argument("--pnp_views", "--pnp-views", type=Path)
    parser.add_argument("--output_dir", "--output-dir", required=True, type=Path)
    parser.add_argument("--max_frame_median_px", "--max-frame-median-px", type=float, default=2.0)
    parser.add_argument("--max_frame_p90_px", "--max-frame-p90-px", type=float, default=5.0)
    parser.add_argument("--min_frame_observations", "--min-frame-observations", type=int, default=64)
    parser.add_argument("--min_frame_cameras", "--min-frame-cameras", type=int, default=4)
    parser.add_argument("--limit_frames", "--limit-frames", type=int, default=0)
    parser.add_argument("--camera_min_observations", "--camera-min-observations", type=int, default=1)
    parser.add_argument(
        "--residual_scope",
        "--residual-scope",
        choices=["used_after_gate", "ok", "all"],
        default="used_after_gate",
        help=(
            "Which rows from observation_residuals.tsv feed frame scoring. "
            "used_after_gate ignores final rejected outliers when the column exists."
        ))
    parser.add_argument("--allow_empty", "--allow-empty", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
