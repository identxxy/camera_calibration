#!/usr/bin/env python3
"""Build a camera_calibration tower dataset from distributed detections.jsonl."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import build_apriltag_tower_dataset_opencv as opencv_builder
import distributed_apriltag_quality_filter as qc


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_extension(value):
    value = str(value or ".jpg").strip()
    if not value:
        return ".jpg"
    if not value.startswith("."):
        return "." + value
    return value


def filename_from_path(value):
    if not value:
        return ""
    return Path(str(value)).name


def frame_text(value):
    return qc.frame_text(value)


def read_manifest(path):
    rows = read_tsv(path)
    if not rows:
        raise SystemExit(f"Manifest contains no cameras: {path}")
    required = ["camera_index", "stage_name", "camera_id", "frame_count"]
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise SystemExit(f"Manifest is missing required columns {missing}: {path}")
    rows = sorted(rows, key=lambda row: parse_int(row.get("camera_index"), len(rows)))
    seen_indices = set()
    seen_camera_ids = set()
    for expected, row in enumerate(rows):
        camera_index = parse_int(row.get("camera_index"), -1)
        camera_id = row.get("camera_id", "")
        if camera_index in seen_indices:
            raise SystemExit(f"Duplicate camera_index {camera_index} in manifest: {path}")
        if camera_id in seen_camera_ids:
            raise SystemExit(f"Duplicate camera_id {camera_id} in manifest: {path}")
        seen_indices.add(camera_index)
        seen_camera_ids.add(camera_id)
        if camera_index != expected:
            raise SystemExit(
                f"Manifest camera_index values must be contiguous after staging: "
                f"expected {expected}, got {camera_index}")
    return rows


def read_image_dirs(path):
    image_dirs = opencv_builder.read_image_dirs(path)
    if not image_dirs:
        raise SystemExit(f"Empty image-directories file: {path}")
    return image_dirs


def unique_append(items, item):
    if item and item not in items:
        items.append(item)


def discover_detection_jsonls(worker_outputs, explicit_jsonls):
    paths = []
    for item in explicit_jsonls:
        for value in qc.split_values(item):
            paths.append(Path(value))
    for item in worker_outputs:
        for value in qc.split_values(item):
            path = Path(value)
            if path.is_file() and path.name == "detections.jsonl":
                paths.append(path)
                continue
            direct = path / "detections.jsonl"
            if direct.is_file():
                paths.append(direct)
                continue
            if path.is_dir():
                paths.extend(sorted(path.glob("**/detections.jsonl")))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if not unique:
        raise SystemExit("No detections.jsonl found. Use --worker-output or --detections-jsonl.")
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise SystemExit("Detection JSONL paths do not exist: " + ", ".join(missing))
    return unique


def infer_selection_paths(manifest_path, explicit_paths, filename):
    if explicit_paths:
        paths = []
        for item in explicit_paths:
            for value in qc.split_values(item):
                paths.append(Path(value))
        return paths
    candidate = Path(manifest_path).parent / filename
    if candidate.is_file():
        return [candidate]
    return []


def staged_filename_from_row(row, default_extension):
    for field in ("filtered_image", "selected_filename", "staged_filename", "filename"):
        name = filename_from_path(row.get(field, ""))
        if name:
            return name
    out_frame = parse_int(row.get("out_frame"), 0)
    return f"{out_frame:06d}{default_extension}"


def infer_default_extension(selection_paths, default_extension):
    for path in selection_paths:
        if not Path(path).is_file():
            continue
        for row in read_tsv(path):
            for field in ("filtered_image", "selected_filename", "staged_filename", "filename"):
                suffix = Path(str(row.get(field, ""))).suffix
                if suffix:
                    return suffix
    return default_extension


def add_mapping(mapping, key, value, ambiguous):
    if not all(key):
        return
    existing = mapping.get(key)
    if existing is None:
        mapping[key] = value
    elif existing != value:
        ambiguous.add(key)


def load_selected_images(paths, default_extension):
    rows = []
    for path in paths:
        rows.extend(read_tsv(path))
    selection = {
        "rows": rows,
        "by_camera_time_frame": {},
        "by_camera_time_filename": {},
        "by_camera_filename": {},
        "by_camera_image_path": {},
        "ambiguous_camera_filename": set(),
    }
    for row in rows:
        camera_id = row.get("camera_id", "")
        item = {
            "camera_id": camera_id,
            "camera_index": parse_int(row.get("camera_index"), -1),
            "time": row.get("time", ""),
            "frame_id": frame_text(row.get("frame_id", "")),
            "out_frame": parse_int(row.get("out_frame"), 0),
            "staged_filename": staged_filename_from_row(row, default_extension),
            "source": row.get("source", ""),
            "filtered_image": row.get("filtered_image", ""),
        }
        source_filename = filename_from_path(row.get("source", ""))
        add_mapping(
            selection["by_camera_time_frame"],
            (camera_id, item["time"], item["frame_id"]),
            item,
            set())
        add_mapping(
            selection["by_camera_time_filename"],
            (camera_id, item["time"], source_filename),
            item,
            set())
        add_mapping(
            selection["by_camera_image_path"],
            (camera_id, row.get("source", "")),
            item,
            set())
        add_mapping(
            selection["by_camera_filename"],
            (camera_id, source_filename),
            item,
            selection["ambiguous_camera_filename"])
    return selection


def load_selected_frames(paths, default_extension):
    rows = []
    for path in paths:
        rows.extend(read_tsv(path))
    selection = {"rows": rows, "by_time_frame": {}, "frame_names": []}
    for row in rows:
        item = {
            "time": row.get("time", ""),
            "frame_id": frame_text(row.get("frame_id", "")),
            "out_frame": parse_int(row.get("out_frame"), len(selection["frame_names"])),
            "staged_filename": staged_filename_from_row(row, default_extension),
        }
        selection["by_time_frame"][(item["time"], item["frame_id"])] = item
        unique_append(selection["frame_names"], item["staged_filename"])
    selection["frame_names"].sort(key=qc.natural_key)
    return selection


def expected_frame_names(manifest_rows, selected_images, selected_frames, default_extension):
    names = []
    if selected_frames["frame_names"]:
        for name in selected_frames["frame_names"]:
            unique_append(names, name)
    for row in selected_images["rows"]:
        unique_append(names, staged_filename_from_row(row, default_extension))
    max_frame_count = max(parse_int(row.get("frame_count"), 0) for row in manifest_rows)
    for index in range(max_frame_count):
        unique_append(names, f"{index:06d}{default_extension}")
    names.sort(key=qc.natural_key)
    return names


def match_selected_image(record, selected_images):
    camera_id = str(record.get("camera_id", ""))
    time_id = str(record.get("time", ""))
    frame_id = frame_text(record.get("frame_id", ""))
    filename = filename_from_path(record.get("filename", ""))
    image_path = str(record.get("image_path", ""))
    for key, mapping in [
        ((camera_id, time_id, frame_id), selected_images["by_camera_time_frame"]),
        ((camera_id, time_id, filename), selected_images["by_camera_time_filename"]),
        ((camera_id, image_path), selected_images["by_camera_image_path"]),
    ]:
        item = mapping.get(key)
        if item is not None:
            return item
    filename_key = (camera_id, filename)
    if filename_key not in selected_images["ambiguous_camera_filename"]:
        return selected_images["by_camera_filename"].get(filename_key)
    return None


def match_selected_frame(record, selected_frames, default_extension):
    time_id = str(record.get("time", ""))
    frame_id = frame_text(record.get("frame_id", ""))
    item = selected_frames["by_time_frame"].get((time_id, frame_id))
    if item is not None:
        return item["staged_filename"]
    filename = filename_from_path(record.get("filename", ""))
    if filename and filename in selected_frames["frame_names"]:
        return filename
    return ""


def match_direct_record(record, frame_names, default_extension):
    filename = filename_from_path(record.get("filename", ""))
    if filename in frame_names:
        return filename
    frame_id = frame_text(record.get("frame_id", ""))
    try:
        index = int(frame_id)
    except (TypeError, ValueError):
        return ""
    candidate = f"{index:06d}{default_extension}"
    if candidate in frame_names:
        return candidate
    return ""


def match_exact_staged_filename(record, frame_names):
    filename = filename_from_path(record.get("filename", ""))
    if filename in frame_names:
        return filename
    return ""


def record_to_features(record, valid_ids):
    features = []
    valid_tag_count = 0
    invalid_tag_count = 0
    malformed_tag_count = 0
    tag_ids = []
    for detection in record.get("detections", []) or []:
        try:
            tag_id = int(detection.get("tag_id"))
        except (TypeError, ValueError):
            malformed_tag_count += 1
            continue
        corners = detection.get("corners", [])
        if tag_id not in valid_ids:
            invalid_tag_count += 1
            continue
        if len(corners) != 4:
            malformed_tag_count += 1
            continue
        tag_features = []
        malformed = False
        for corner_id, corner in enumerate(corners):
            if len(corner) != 2:
                malformed = True
                break
            try:
                x, y = corner
                tag_features.append((float(x), float(y), tag_id * 4 + corner_id))
            except (TypeError, ValueError):
                malformed = True
                break
        if malformed:
            malformed_tag_count += 1
            continue
        valid_tag_count += 1
        tag_ids.append(tag_id)
        features.extend(tag_features)
    return {
        "features": features,
        "valid_tag_count": valid_tag_count,
        "invalid_tag_count": invalid_tag_count,
        "malformed_tag_count": malformed_tag_count,
        "tag_ids": tag_ids,
    }


def update_image_size(image_sizes_by_camera_id, camera_id, record):
    width = parse_int(record.get("width"), 0)
    height = parse_int(record.get("height"), 0)
    if width <= 0 or height <= 0:
        return
    size = (width, height)
    previous = image_sizes_by_camera_id.get(camera_id)
    if previous is None:
        image_sizes_by_camera_id[camera_id] = size
    elif previous != size:
        raise SystemExit(
            f"Inconsistent image size for camera {camera_id}: {previous} vs {size}")


def load_jsonl_records(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                record["_detections_jsonl"] = str(path)
                record["_line_number"] = line_number
                yield record


def choose_features(existing, candidate):
    if existing is None:
        return candidate
    if candidate["valid_tag_count"] > existing["valid_tag_count"]:
        return candidate
    if candidate["valid_tag_count"] == existing["valid_tag_count"]:
        if len(candidate["features"]) > len(existing["features"]):
            return candidate
    return existing


def build_dataset_from_detections(args):
    manifest_rows = read_manifest(args.manifest)
    image_dirs = read_image_dirs(args.image_directories_file)
    if len(image_dirs) != len(manifest_rows):
        raise SystemExit(
            f"image_directories has {len(image_dirs)} entries, but manifest has {len(manifest_rows)} cameras")
    jsonl_paths = discover_detection_jsonls(args.worker_output, args.detections_jsonl)
    selected_image_paths = infer_selection_paths(args.manifest, args.selected_images, "selected_images.tsv")
    selected_frame_paths = infer_selection_paths(args.manifest, args.selected_frames, "selected_frames.tsv")
    default_extension = clean_extension(args.default_extension)
    default_extension = infer_default_extension(selected_frame_paths + selected_image_paths, default_extension)

    selected_images = load_selected_images(selected_image_paths, default_extension)
    selected_frames = load_selected_frames(selected_frame_paths, default_extension)
    frame_names = expected_frame_names(
        manifest_rows,
        selected_images,
        selected_frames,
        default_extension)
    if not frame_names:
        raise SystemExit("No staged frame names could be inferred from manifest/selection files.")

    camera_index_by_id = {
        row.get("camera_id", ""): parse_int(row.get("camera_index"), index)
        for index, row in enumerate(manifest_rows)
    }
    frame_index_by_name = {name: index for index, name in enumerate(frame_names)}
    tower_config = opencv_builder.read_tower_config(args.tower_config)
    valid_ids = qc.tower_valid_tag_ids(tower_config)
    if not valid_ids:
        raise SystemExit(f"Tower config did not expand any valid ids: {args.tower_config}")
    tower_points = opencv_builder.build_tower_points(tower_config)
    cell_length = float(tower_config["tag_size_m"]) + float(tower_config["tag_spacing_m"])

    per_camera = [
        {
            "camera_index": index,
            "stage_name": row.get("stage_name", ""),
            "camera_id": row.get("camera_id", ""),
            "image_dir": str(image_dirs[index]),
            "frame_count": parse_int(row.get("frame_count"), 0),
            "jsonl_records": 0,
            "matched_records": 0,
            "images_with_tags": 0,
            "total_tags": 0,
            "total_corners": 0,
            "max_tags": 0,
            "width": 0,
            "height": 0,
        }
        for index, row in enumerate(manifest_rows)
    ]
    features_by_camera_frame = {}
    image_sizes_by_camera_id = {}
    detection_rows = []
    has_selection_mapping = bool(selected_images["rows"] or selected_frames["rows"])
    total_records = 0
    ignored_unknown_camera = 0
    ignored_unstaged = 0
    duplicate_matched_records = 0
    filtered_invalid_tags = 0
    malformed_tags = 0

    for record in load_jsonl_records(jsonl_paths):
        total_records += 1
        camera_id = str(record.get("camera_id", ""))
        if camera_id not in camera_index_by_id:
            ignored_unknown_camera += 1
            continue
        camera_index = camera_index_by_id[camera_id]
        per_camera[camera_index]["jsonl_records"] += 1
        update_image_size(image_sizes_by_camera_id, camera_id, record)

        staged_filename = ""
        selection_item = match_selected_image(record, selected_images)
        if selection_item is not None:
            staged_filename = selection_item["staged_filename"]
        if not staged_filename:
            staged_filename = match_selected_frame(record, selected_frames, default_extension)
        if not staged_filename:
            if has_selection_mapping:
                staged_filename = match_exact_staged_filename(record, frame_names)
            else:
                staged_filename = match_direct_record(record, frame_names, default_extension)
        if staged_filename not in frame_index_by_name:
            ignored_unstaged += 1
            continue

        converted = record_to_features(record, valid_ids)
        filtered_invalid_tags += converted["invalid_tag_count"]
        malformed_tags += converted["malformed_tag_count"]
        key = (camera_index, staged_filename)
        candidate = {
            **converted,
            "camera_id": camera_id,
            "camera_index": camera_index,
            "staged_filename": staged_filename,
            "source_time": record.get("time", ""),
            "source_frame_id": frame_text(record.get("frame_id", "")),
            "source_filename": filename_from_path(record.get("filename", "")),
            "image_path": record.get("image_path", ""),
            "detections_jsonl": record.get("_detections_jsonl", ""),
            "line_number": record.get("_line_number", ""),
        }
        if key in features_by_camera_frame:
            duplicate_matched_records += 1
        features_by_camera_frame[key] = choose_features(features_by_camera_frame.get(key), candidate)

    for (camera_index, staged_filename), item in sorted(
            features_by_camera_frame.items(),
            key=lambda pair: (pair[0][0], qc.natural_key(pair[0][1]))):
        tag_count = item["valid_tag_count"]
        corner_count = len(item["features"])
        per_camera[camera_index]["matched_records"] += 1
        if tag_count:
            per_camera[camera_index]["images_with_tags"] += 1
        per_camera[camera_index]["total_tags"] += tag_count
        per_camera[camera_index]["total_corners"] += corner_count
        per_camera[camera_index]["max_tags"] = max(per_camera[camera_index]["max_tags"], tag_count)
        detection_rows.append({
            "frame_index": frame_index_by_name[staged_filename],
            "filename": staged_filename,
            "camera_index": camera_index,
            "stage_name": per_camera[camera_index]["stage_name"],
            "camera_id": item["camera_id"],
            "source_time": item["source_time"],
            "source_frame_id": item["source_frame_id"],
            "source_filename": item["source_filename"],
            "image_path": item["image_path"],
            "tag_count": tag_count,
            "corner_count": corner_count,
            "tag_ids": ",".join(str(tag_id) for tag_id in sorted(item["tag_ids"])),
            "filtered_invalid_tags": item["invalid_tag_count"],
            "malformed_tags": item["malformed_tag_count"],
            "detections_jsonl": item["detections_jsonl"],
            "line_number": item["line_number"],
        })

    image_sizes = []
    missing_sizes = []
    for camera_index, row in enumerate(manifest_rows):
        camera_id = row.get("camera_id", "")
        size = image_sizes_by_camera_id.get(camera_id)
        if size is None:
            missing_sizes.append(camera_id)
            image_sizes.append((0, 0))
            continue
        image_sizes.append(size)
        per_camera[camera_index]["width"] = size[0]
        per_camera[camera_index]["height"] = size[1]
    if missing_sizes:
        raise SystemExit(
            "Missing width/height in detections.jsonl for staged cameras: "
            + ", ".join(missing_sizes))

    imagesets = []
    for frame_name in frame_names:
        camera_features = []
        for camera_index in range(len(manifest_rows)):
            item = features_by_camera_frame.get((camera_index, frame_name))
            camera_features.append(item["features"] if item is not None else [])
        imagesets.append((frame_name, camera_features))

    opencv_builder.write_dataset(
        args.output_dataset,
        image_sizes,
        imagesets,
        tower_points,
        cell_length)

    summary = {
        "mode": "detections_apriltag_tower_dataset",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "image_directories_file": str(args.image_directories_file),
        "detection_jsonl_files": [str(path) for path in jsonl_paths],
        "selected_images": [str(path) for path in selected_image_paths],
        "selected_frames": [str(path) for path in selected_frame_paths],
        "output_dataset": str(args.output_dataset),
        "tower_config": str(args.tower_config),
        "camera_count": len(manifest_rows),
        "imageset_count": len(imagesets),
        "known_3d_point_count": len(tower_points),
        "total_jsonl_records": total_records,
        "matched_records": sum(row["matched_records"] for row in per_camera),
        "duplicate_matched_records": duplicate_matched_records,
        "ignored_unknown_camera_records": ignored_unknown_camera,
        "ignored_unstaged_records": ignored_unstaged,
        "filtered_invalid_tags": filtered_invalid_tags,
        "malformed_tags": malformed_tags,
        "total_tags": sum(row["total_tags"] for row in per_camera),
        "total_corners": sum(row["total_corners"] for row in per_camera),
        "default_extension": default_extension,
        "per_camera": per_camera,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    if args.per_camera_tsv:
        write_tsv(
            args.per_camera_tsv,
            per_camera,
            [
                "camera_index", "stage_name", "camera_id", "image_dir", "frame_count",
                "jsonl_records", "matched_records", "images_with_tags",
                "total_tags", "total_corners", "max_tags", "width", "height",
            ])
    if args.detections_tsv:
        write_tsv(
            args.detections_tsv,
            detection_rows,
            [
                "frame_index", "filename", "camera_index", "stage_name", "camera_id",
                "source_time", "source_frame_id", "source_filename", "image_path",
                "tag_count", "corner_count", "tag_ids", "filtered_invalid_tags",
                "malformed_tags", "detections_jsonl", "line_number",
            ])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a camera_calibration calib_data binary from distributed "
            "AprilTag tower detections.jsonl without re-reading 4K source images."))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-directories-file", required=True, type=Path)
    parser.add_argument("--worker-output", action="append", default=[])
    parser.add_argument("--detections-jsonl", action="append", default=[])
    parser.add_argument("--selected-images", action="append", default=[])
    parser.add_argument("--selected-frames", action="append", default=[])
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--tower-config", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--per-camera-tsv", type=Path, default=None)
    parser.add_argument("--detections-tsv", type=Path, default=None)
    parser.add_argument("--default-extension", default=".jpg")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(build_dataset_from_detections(parse_args()))
