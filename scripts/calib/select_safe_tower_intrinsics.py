#!/usr/bin/env python3
"""Select plausible AprilTag tower intrinsics into a safe fallback directory."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def to_float(row, key):
    value = row.get(key, "")
    if value == "":
        return None
    try:
        value = float(value)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def to_int(row, key):
    try:
        return int(row.get(key, "") or 0)
    except ValueError:
        return 0


def add_range_reason(reasons, name, value, lo, hi):
    if value is None:
        reasons.append(f"missing_{name}")
    elif value < lo or value > hi:
        reasons.append(f"{name}_out_of_range:{value:.8g}")


def evaluate_row(row, args):
    reasons = []
    if row.get("status") != "solved":
        reasons.append(f"status_{row.get('status', 'unknown')}")
        return reasons

    width = to_float(row, "width")
    height = to_float(row, "height")
    rms = to_float(row, "rms")
    usable_views = to_int(row, "usable_views")
    usable_points = to_int(row, "usable_points")
    bbox_area_ratio = to_float(row, "bbox_area_ratio")
    fx = to_float(row, "fx")
    fy = to_float(row, "fy")
    cx = to_float(row, "cx")
    cy = to_float(row, "cy")
    k1 = to_float(row, "k1")
    k2 = to_float(row, "k2")
    k3 = to_float(row, "k3")
    p1 = to_float(row, "p1")
    p2 = to_float(row, "p2")

    if rms is None or rms > args.max_rms_px:
        reasons.append("rms_too_high" if rms is not None else "missing_rms")
    if usable_views < args.min_usable_views:
        reasons.append(f"usable_views_below_{args.min_usable_views}")
    if usable_points < args.min_usable_points:
        reasons.append(f"usable_points_below_{args.min_usable_points}")
    if bbox_area_ratio is None or bbox_area_ratio < args.min_bbox_area_ratio:
        reasons.append(f"bbox_area_ratio_below_{args.min_bbox_area_ratio}")

    if width is None or height is None:
        reasons.append("missing_image_size")
    else:
        add_range_reason(
            reasons,
            "fx",
            fx,
            args.min_focal_ratio * width,
            args.max_focal_ratio * width)
        add_range_reason(
            reasons,
            "fy",
            fy,
            args.min_focal_ratio * width,
            args.max_focal_ratio * width)
        if fx is not None and fy is not None and min(abs(fx), abs(fy)) > 1e-12:
            aspect = max(abs(fx), abs(fy)) / min(abs(fx), abs(fy))
            if aspect > args.max_focal_aspect_ratio:
                reasons.append(f"focal_aspect_too_high:{aspect:.8g}")
        add_range_reason(
            reasons,
            "cx",
            cx,
            -args.principal_margin_px,
            width + args.principal_margin_px)
        add_range_reason(
            reasons,
            "cy",
            cy,
            -args.principal_margin_px,
            height + args.principal_margin_px)

    add_range_reason(reasons, "k1", k1, -args.max_abs_k1, args.max_abs_k1)
    add_range_reason(reasons, "k2", k2, -args.max_abs_k2, args.max_abs_k2)
    add_range_reason(reasons, "k3", k3, -args.max_abs_k3, args.max_abs_k3)
    add_range_reason(reasons, "p1", p1, -args.max_abs_p, args.max_abs_p)
    add_range_reason(reasons, "p2", p2, -args.max_abs_p, args.max_abs_p)
    return reasons


def yaml_candidates(intrinsics_dir, row, include_opencv):
    camera_index = row.get("camera_index")
    user_id = row.get("user_id")
    candidates = [(intrinsics_dir / f"intrinsics{camera_index}_{user_id}.yaml", True)]
    if include_opencv:
        candidates.append((intrinsics_dir / f"opencv_intrinsics{camera_index}_{user_id}.yaml", False))
    return candidates


def clean_output_dir(output_dir):
    if not output_dir.exists():
        return
    for path in output_dir.glob("*.yaml"):
        path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Copy RMS/parameter-gated tower intrinsics to a safe fallback directory.")
    parser.add_argument("--summary-tsv", required=True, type=Path)
    parser.add_argument("--intrinsics-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-rms-px", type=float, default=5.0)
    parser.add_argument("--min-usable-views", type=int, default=10)
    parser.add_argument("--min-usable-points", type=int, default=80)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.05)
    parser.add_argument("--min-focal-ratio", type=float, default=0.5)
    parser.add_argument("--max-focal-ratio", type=float, default=1.6)
    parser.add_argument("--max-focal-aspect-ratio", type=float, default=1.25)
    parser.add_argument("--principal-margin-px", type=float, default=0.0)
    parser.add_argument("--max-abs-k1", type=float, default=1.0)
    parser.add_argument("--max-abs-k2", type=float, default=8.0)
    parser.add_argument("--max-abs-k3", type=float, default=20.0)
    parser.add_argument("--max-abs-p", type=float, default=0.1)
    parser.add_argument("--no-opencv-yaml", action="store_true")
    parser.add_argument("--overwrite-clean", action="store_true")
    args = parser.parse_args()

    intrinsics_dir = args.intrinsics_dir or args.summary_tsv.parent
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.glob("*.yaml")) and not args.overwrite_clean:
        raise FileExistsError(
            f"{output_dir} already contains YAML files; use --overwrite-clean to remove stale selections")
    if args.overwrite_clean:
        clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(args.summary_tsv)
    selection_rows = []
    selected = []
    for row in rows:
        reasons = evaluate_row(row, args)
        candidates = yaml_candidates(intrinsics_dir, row, not args.no_opencv_yaml)
        copied = []
        missing_required = [path.name for path, required in candidates if required and not path.exists()]
        if not reasons and missing_required:
            reasons.extend(f"missing_yaml:{name}" for name in missing_required)
        if not reasons:
            for src, _required in candidates:
                if not src.exists():
                    continue
                dst = output_dir / src.name
                shutil.copy2(src, dst)
                copied.append(dst.name)
            selected.append(row.get("user_id", ""))
        selection_rows.append({
            "camera_index": row.get("camera_index", ""),
            "stage_name": row.get("stage_name", ""),
            "machine": row.get("machine", ""),
            "user_id": row.get("user_id", ""),
            "selected": "yes" if not reasons else "no",
            "reject_reasons": ",".join(reasons),
            "copied_files": ",".join(copied),
            "rms": row.get("rms", ""),
            "usable_views": row.get("usable_views", ""),
            "usable_points": row.get("usable_points", ""),
            "bbox_area_ratio": row.get("bbox_area_ratio", ""),
            "fx": row.get("fx", ""),
            "fy": row.get("fy", ""),
            "cx": row.get("cx", ""),
            "cy": row.get("cy", ""),
            "k1": row.get("k1", ""),
            "k2": row.get("k2", ""),
            "p1": row.get("p1", ""),
            "p2": row.get("p2", ""),
            "k3": row.get("k3", ""),
        })

    fieldnames = [
        "camera_index", "stage_name", "machine", "user_id", "selected",
        "reject_reasons", "copied_files", "rms", "usable_views", "usable_points",
        "bbox_area_ratio", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
    ]
    tsv_path = output_dir / "selection_summary.tsv"
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selection_rows)

    result = {
        "summary_tsv": str(args.summary_tsv.resolve()),
        "intrinsics_dir": str(intrinsics_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "settings": {
            "max_rms_px": args.max_rms_px,
            "min_usable_views": args.min_usable_views,
            "min_usable_points": args.min_usable_points,
            "min_bbox_area_ratio": args.min_bbox_area_ratio,
            "min_focal_ratio": args.min_focal_ratio,
            "max_focal_ratio": args.max_focal_ratio,
            "max_focal_aspect_ratio": args.max_focal_aspect_ratio,
            "principal_margin_px": args.principal_margin_px,
            "max_abs_k1": args.max_abs_k1,
            "max_abs_k2": args.max_abs_k2,
            "max_abs_k3": args.max_abs_k3,
            "max_abs_p": args.max_abs_p,
            "include_opencv_yaml": not args.no_opencv_yaml,
        },
        "camera_count": len(rows),
        "selected_count": len(selected),
        "selected_user_ids": selected,
        "selection_summary_tsv": str(tsv_path.resolve()),
    }
    json_path = output_dir / "selection_summary.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
