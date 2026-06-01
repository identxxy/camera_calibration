#!/usr/bin/env python3
"""Analyze residual tails from outer AprilTag tower refinement diagnostics."""

import argparse
import csv
import html
import json
import math
from pathlib import Path


KNOWN_CAMERA_TABLES = {
    "camera_reprojection.tsv",
    "camera_acceptance.tsv",
    "camera_delta.tsv",
    "camera_intrinsics.tsv",
    "frame_quality.tsv",
}

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

CAMERA_KEYS = ("camera_id", "camera", "camera_label", "label")
CAMERA_INDEX_KEYS = ("camera_index", "cam_idx", "camera_idx")
FRAME_KEYS = ("frame_index", "imageset_index", "frame_id", "image_id", "frame", "filename", "image_name")
TAG_KEYS = ("tag_id", "tag", "apriltag_id", "marker_id", "feature_id")
FACE_KEYS = ("face_id", "face", "board_face", "tower_face")
CORNER_KEYS = ("corner_id", "corner", "tag_corner")


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def fmt_float(value, digits=3):
    value = as_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def first_value(row, keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return ""


def percentile(values, q):
    values = sorted(v for v in values if as_float(v) is not None)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * q / 100.0
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(values[lo])
    weight = position - lo
    return float(values[lo]) * (1.0 - weight) + float(values[hi]) * weight


def summarize_values(values):
    clean = [as_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return {
            "count": 0,
            "mean_px": None,
            "median_px": None,
            "p90_px": None,
            "max_px": None,
            "under_300_fraction": None,
        }
    return {
        "count": len(clean),
        "mean_px": float(sum(clean) / len(clean)),
        "median_px": percentile(clean, 50),
        "p90_px": percentile(clean, 90),
        "max_px": float(max(clean)),
        "under_300_fraction": float(sum(1 for value in clean if value < 300.0) / len(clean)),
    }


def numeric_camera_row(row, acceptance):
    camera_id = row.get("camera_id", "")
    merged = dict(row)
    if acceptance:
        for key in ("decision", "output_pose", "reason", "active_delta", "used_observation"):
            merged[key] = acceptance.get(key, merged.get(key, ""))
    for key in (
            "observation_count",
            "before_median_px",
            "before_p90_px",
            "after_median_px",
            "after_p90_px",
            "after_max_px",
            "after_under_100_fraction",
            "after_under_300_fraction"):
        value = as_float(merged.get(key))
        if value is not None:
            merged[key] = value
    merged["camera_id"] = camera_id
    return merged


def load_camera_rows(diagnostics_dir):
    reprojection_path = diagnostics_dir / "camera_reprojection.tsv"
    if not reprojection_path.exists():
        raise FileNotFoundError(f"Missing required diagnostics table: {reprojection_path}")
    acceptance_path = diagnostics_dir / "camera_acceptance.tsv"
    acceptance_rows = read_tsv(acceptance_path) if acceptance_path.exists() else []
    acceptance_by_camera = {row.get("camera_id", ""): row for row in acceptance_rows}
    rows = [
        numeric_camera_row(row, acceptance_by_camera.get(row.get("camera_id", "")))
        for row in read_tsv(reprojection_path)
    ]
    return rows, {
        "camera_reprojection": str(reprojection_path.resolve()),
        "camera_acceptance": str(acceptance_path.resolve()) if acceptance_path.exists() else None,
    }


def camera_is_prior_only(row):
    output_pose = row.get("output_pose", "")
    decision = row.get("decision", "")
    if output_pose == "prior":
        return True
    return decision in {
        "rejected_to_prior",
        "excluded_prior_only",
        "inactive_prior_only",
        "ungated_prior_only",
    }


def camera_is_accepted(row):
    output_pose = row.get("output_pose", "")
    decision = row.get("decision", "")
    return output_pose == "refined" or decision in {
        "accepted_refined",
        "refined_no_acceptance_gate",
    }


def summarize_camera_group(rows):
    return {
        "count": len(rows),
        "cameras": [row.get("camera_id", "") for row in rows],
        "after_p90_px": summarize_values(row.get("after_p90_px") for row in rows),
        "after_max_px": summarize_values(row.get("after_max_px") for row in rows),
        "after_under_300_fraction": summarize_values(row.get("after_under_300_fraction") for row in rows),
    }


def summarize_camera_groups(rows):
    accepted = [row for row in rows if camera_is_accepted(row)]
    prior_only = [row for row in rows if camera_is_prior_only(row)]
    other = [row for row in rows if row not in accepted and row not in prior_only]
    return {
        "accepted_refined": summarize_camera_group(accepted),
        "prior_only": summarize_camera_group(prior_only),
        "other": summarize_camera_group(other),
    }


def sort_numeric(rows, key, reverse=True):
    return sorted(
        [row for row in rows if as_float(row.get(key)) is not None],
        key=lambda row: as_float(row.get(key)),
        reverse=reverse)


def simplify_camera_rows(rows):
    output = []
    for row in rows:
        output.append({
            "camera_index": row.get("camera_index", ""),
            "camera_id": row.get("camera_id", ""),
            "decision": row.get("decision", ""),
            "output_pose": row.get("output_pose", ""),
            "observation_count": row.get("observation_count", ""),
            "after_median_px": as_float(row.get("after_median_px")),
            "after_p90_px": as_float(row.get("after_p90_px")),
            "after_max_px": as_float(row.get("after_max_px")),
            "after_under_300_fraction": as_float(row.get("after_under_300_fraction")),
        })
    return output


def residual_key_for_row(row):
    for key in RESIDUAL_KEYS:
        if as_float(row.get(key)) is not None:
            return key
    for key in row:
        lower = key.lower()
        if (
                lower.endswith("_residual_px")
                or lower.endswith("_error_px")
                or lower.endswith("_norm_px")):
            if as_float(row.get(key)) is not None:
                return key
    return None


def iter_json_records(obj):
    if isinstance(obj, list):
        for item in obj:
            yield from iter_json_records(item)
    elif isinstance(obj, dict):
        if residual_key_for_row(obj):
            yield obj
        else:
            for value in obj.values():
                if isinstance(value, (list, dict)):
                    yield from iter_json_records(value)


def load_json_rows(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict(row) for row in iter_json_records(obj) if isinstance(row, dict)]


def candidate_residual_files(diagnostics_dir):
    if not diagnostics_dir.exists():
        return []
    candidates = []
    for path in sorted(diagnostics_dir.iterdir()):
        if path.name in KNOWN_CAMERA_TABLES:
            continue
        if path.suffix.lower() not in (".tsv", ".json"):
            continue
        name = path.name.lower()
        has_residual_term = "residual" in name or "reprojection" in name or "error" in name
        has_scope_term = any(term in name for term in ("observation", "frame", "tag", "corner", "face"))
        if has_residual_term and has_scope_term:
            candidates.append(path)
    return candidates


def normalize_residual_record(row, source_file):
    residual_key = residual_key_for_row(row)
    residual_px = as_float(row.get(residual_key)) if residual_key else None
    if residual_px is None:
        return None
    record = {
        "source_file": str(Path(source_file).resolve()),
        "residual_px": residual_px,
        "camera_id": first_value(row, CAMERA_KEYS),
        "camera_index": first_value(row, CAMERA_INDEX_KEYS),
        "frame_index": first_value(row, FRAME_KEYS),
        "tag_id": first_value(row, TAG_KEYS),
        "corner_id": first_value(row, CORNER_KEYS),
        "face_id": first_value(row, FACE_KEYS),
        "used_after_gate": str(row.get("used_after_gate", "")),
        "projection_status": str(row.get("projection_status", "")),
    }
    return record


def load_residual_records(diagnostics_dir):
    records = []
    source_files = []
    for path in candidate_residual_files(diagnostics_dir):
        try:
            rows = read_tsv(path) if path.suffix.lower() == ".tsv" else load_json_rows(path)
        except (OSError, csv.Error, json.JSONDecodeError, TypeError):
            continue
        normalized = [
            normalize_residual_record(row, path)
            for row in rows
        ]
        normalized = [row for row in normalized if row is not None]
        if not normalized:
            continue
        records.extend(normalized)
        source_files.append(str(path.resolve()))
    return records, source_files


def group_records(records, field):
    buckets = {}
    for record in records:
        key = record.get(field, "")
        if key == "":
            continue
        buckets.setdefault(key, []).append(record["residual_px"])
    rows = []
    for key, values in buckets.items():
        summary = summarize_values(values)
        row = {field: key}
        row.update(summary)
        rows.append(row)
    return sorted(rows, key=lambda row: (row.get("max_px") or -1.0, row.get("p90_px") or -1.0), reverse=True)


def top_rows(rows, limit):
    return rows[:limit] if limit > 0 else rows


def html_cell(value):
    if value is None:
        value = ""
    return html.escape(str(value))


def render_table(rows, columns):
    if not rows:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html_cell(label)}</th>" for _key, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt_float(value)
            cells.append(f"<td>{html_cell(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><tr>" + header + "</tr>\n" + "\n".join(body) + "\n</table>"


def render_html(summary):
    camera_columns = [
        ("camera_id", "camera"),
        ("decision", "decision"),
        ("output_pose", "output"),
        ("observation_count", "obs"),
        ("after_median_px", "median px"),
        ("after_p90_px", "p90 px"),
        ("after_max_px", "max px"),
        ("after_under_300_fraction", "<300 frac"),
    ]
    group_rows = []
    for name, data in summary["camera_groups"].items():
        group_rows.append({
            "group": name,
            "count": data["count"],
            "cameras": ", ".join(data["cameras"]),
        })
    recommendations = "\n".join(
        f"<li>{html_cell(item)}</li>"
        for item in summary["recommendations"])
    observation = summary["observation_diagnostics"]
    if observation["available"]:
        observation_html = f"""
<p>Loaded {observation["row_count"]} residual records from {len(observation["source_files"])} file(s).</p>
<h2>Worst Observations</h2>
{render_table(summary["worst_observations"], [
    ("residual_px", "residual px"),
    ("camera_id", "camera"),
    ("camera_index", "camera idx"),
    ("frame_index", "frame"),
    ("tag_id", "tag"),
    ("corner_id", "corner"),
    ("face_id", "face"),
    ("used_after_gate", "used"),
    ("projection_status", "status"),
    ("source_file", "source"),
])}
<h2>Worst By Camera</h2>
{render_table(summary["worst_by_camera"], [("camera_id", "camera"), ("count", "count"), ("p90_px", "p90 px"), ("max_px", "max px"), ("under_300_fraction", "<300 frac")])}
<h2>Worst By Frame</h2>
{render_table(summary["worst_by_frame"], [("frame_index", "frame"), ("count", "count"), ("p90_px", "p90 px"), ("max_px", "max px"), ("under_300_fraction", "<300 frac")])}
<h2>Worst By Tag</h2>
{render_table(summary["worst_by_tag"], [("tag_id", "tag"), ("count", "count"), ("p90_px", "p90 px"), ("max_px", "max px"), ("under_300_fraction", "<300 frac")])}
<h2>Worst By Face</h2>
{render_table(summary["worst_by_face"], [("face_id", "face"), ("count", "count"), ("p90_px", "p90 px"), ("max_px", "max px"), ("under_300_fraction", "<300 frac")])}
"""
    else:
        observation_html = """
<p><strong>Current refine output is missing per-observation residual diagnostics.</strong>
Only per-camera residual tails can be analyzed. Add an observation-level residual export
to inspect specific camera/frame/tag/face outliers.</p>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Outer Tag Residual Tail Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; line-height: 1.45; color: #1f2933; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 24px; font-size: 13px; }}
th, td {{ border: 1px solid #d8dee4; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fa; }}
td:nth-child(n+4), th:nth-child(n+4) {{ text-align: right; }}
code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
.panel {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 12px; }}
</style>
</head>
<body>
<h1>Outer Tag Residual Tail Report</h1>
<p>refine output: <code>{html_cell(summary["refine_output_dir"])}</code></p>
<div class="grid">
<div class="panel">
<h2>Camera Groups</h2>
{render_table(group_rows, [("group", "group"), ("count", "count"), ("cameras", "cameras")])}
</div>
<div class="panel">
<h2>Suggested Next Diagnostics</h2>
<ul>{recommendations}</ul>
</div>
</div>
<h2>Worst Cameras By p90</h2>
{render_table(summary["worst_cameras_by_p90"], camera_columns)}
<h2>Worst Cameras By max</h2>
{render_table(summary["worst_cameras_by_max"], camera_columns)}
<h2>Worst Cameras By under_300_fraction</h2>
{render_table(summary["worst_cameras_by_under_300_fraction"], camera_columns)}
<h2>Observation-Level Tail</h2>
{observation_html}
</body>
</html>
"""


def recommendations(observation_available):
    items = [
        "Run post-optimization trimming on the worst camera residual tails, then re-evaluate accepted-output p90/max.",
        "Inspect per-frame/tag outlier patterns before changing solver weights; repeated frame/tag failures usually indicate bad tower detections or a pose-initialization issue.",
        "Compare accepted_refined and prior_only groups so rejected cameras are not hidden by aggregate residuals.",
    ]
    if not observation_available:
        items.append(
            "Export per-observation residual diagnostics from the refine stage before doing per-frame/tag outlier inspection.")
    return items


def analyze_refine_output(refine_output_dir, output_dir, limit=20):
    refine_output_dir = Path(refine_output_dir)
    output_dir = Path(output_dir)
    diagnostics_dir = refine_output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_rows, camera_sources = load_camera_rows(diagnostics_dir)
    residual_records, source_files = load_residual_records(diagnostics_dir)
    worst_observations = sorted(
        residual_records,
        key=lambda row: row["residual_px"],
        reverse=True)

    observation_available = bool(residual_records)
    summary = {
        "refine_output_dir": str(refine_output_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "camera_diagnostics": camera_sources,
        "camera_count": len(camera_rows),
        "camera_groups": summarize_camera_groups(camera_rows),
        "worst_cameras_by_p90": simplify_camera_rows(top_rows(sort_numeric(camera_rows, "after_p90_px"), limit)),
        "worst_cameras_by_max": simplify_camera_rows(top_rows(sort_numeric(camera_rows, "after_max_px"), limit)),
        "worst_cameras_by_under_300_fraction": simplify_camera_rows(
            top_rows(sort_numeric(camera_rows, "after_under_300_fraction", reverse=False), limit)),
        "observation_diagnostics": {
            "available": observation_available,
            "source_files": source_files,
            "row_count": len(residual_records),
            "message": (
                "per-observation residual diagnostics loaded"
                if observation_available
                else "missing per-observation residual diagnostics"),
        },
        "worst_observations": top_rows(worst_observations, limit),
        "worst_by_camera": top_rows(group_records(residual_records, "camera_id"), limit),
        "worst_by_frame": top_rows(group_records(residual_records, "frame_index"), limit),
        "worst_by_tag": top_rows(group_records(residual_records, "tag_id"), limit),
        "worst_by_face": top_rows(group_records(residual_records, "face_id"), limit),
        "recommendations": recommendations(observation_available),
    }
    summary["outputs"] = {
        "summary_json": str((output_dir / "residual_tail_summary.json").resolve()),
        "report_html": str((output_dir / "residual_tail_report.html").resolve()),
    }

    (output_dir / "residual_tail_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8")
    (output_dir / "residual_tail_report.html").write_text(
        render_html(summary),
        encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze residual tails from a tag_refine_robust output directory.")
    parser.add_argument("refine_output_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for residual_tail_summary.json and residual_tail_report.html.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or (args.refine_output_dir / "diagnostics" / "residual_tail")
    summary = analyze_refine_output(args.refine_output_dir, output_dir, limit=args.limit)
    print(json.dumps({
        "summary_json": summary["outputs"]["summary_json"],
        "report_html": summary["outputs"]["report_html"],
        "observation_diagnostics": summary["observation_diagnostics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
