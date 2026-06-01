#!/usr/bin/env python3
"""Summarize AprilTag tower delta-prior refinement diagnostics."""

import argparse
import csv
import html
import json
import math
from pathlib import Path


def read_tsv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_summary(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def fmt(value, digits=3):
    value = as_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def ratio_fmt(value):
    value = as_float(value)
    if value is None:
        return ""
    return f"{value:.3f}"


def classify_camera(camera_id, delta_row, summary, acceptance_row):
    if acceptance_row:
        return acceptance_row.get("decision", ""), acceptance_row.get("reason", "")

    cameras = summary.get("cameras", {})
    settings = summary.get("settings", {})
    accepted = set(cameras.get("accepted_refined", []))
    prior_only = set(cameras.get("prior_only", []))
    excluded = set(cameras.get("excluded_from_observation_residual", []))
    acceptance_enabled = as_float(settings.get("accept_camera_median_px")) is not None
    acceptance_enabled = acceptance_enabled and float(settings.get("accept_camera_median_px", 0.0)) > 0.0

    active = (delta_row or {}).get("active")
    used = (delta_row or {}).get("used")
    if camera_id in accepted:
        return "accepted_refined", "passes_acceptance_gate"
    if camera_id in excluded or used == "no":
        return "excluded_prior_only", "below_min_camera_observations_for_use"
    if camera_id in prior_only or active == "no":
        return "inactive_prior_only", "below_min_camera_observations_for_delta"
    if acceptance_enabled:
        return "rejected_to_prior", "failed_acceptance_gate"
    return "refined_no_acceptance_gate", "acceptance_gate_disabled"


def merge_camera_rows(reprojection_rows, delta_rows, acceptance_rows, summary):
    delta_by_id = {row.get("camera_id"): row for row in delta_rows}
    acceptance_by_id = {row.get("camera_id"): row for row in acceptance_rows}
    rows = []
    for reproj in reprojection_rows:
        camera_id = reproj.get("camera_id", "")
        delta = delta_by_id.get(camera_id, {})
        decision, decision_reason = classify_camera(
            camera_id,
            delta,
            summary,
            acceptance_by_id.get(camera_id))
        rows.append({
            "camera_index": reproj.get("camera_index", ""),
            "camera_id": camera_id,
            "decision": decision,
            "decision_reason": decision_reason,
            "active_delta": delta.get("active", ""),
            "used_observation": delta.get("used", ""),
            "observation_count": reproj.get("observation_count", ""),
            "final_median_px": reproj.get("after_median_px", ""),
            "final_p90_px": reproj.get("after_p90_px", ""),
            "final_max_px": reproj.get("after_max_px", ""),
            "final_under_100_fraction": reproj.get("after_under_100_fraction", ""),
            "final_under_300_fraction": reproj.get("after_under_300_fraction", ""),
        })
    return rows


def decision_counts(rows):
    counts = {}
    for row in rows:
        key = row.get("decision", "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def sort_key(row):
    index = as_int(row.get("camera_index"))
    return index


def make_status_rows(rows):
    status_rows = []
    for row in sorted(rows, key=sort_key):
        output_pose = "refined" if row["decision"] in (
            "accepted_refined",
            "refined_no_acceptance_gate",
        ) else "prior"
        status_rows.append({
            "camera_index": row["camera_index"],
            "camera_id": row["camera_id"],
            "decision": row["decision"],
            "output_pose": output_pose,
            "reason": row["decision_reason"],
            "observation_count": row["observation_count"],
            "final_median_px": row["final_median_px"],
            "final_p90_px": row["final_p90_px"],
            "final_under_300_fraction": row["final_under_300_fraction"],
        })
    return status_rows


def render_markdown(refine_dir, rows, status_rows, summary, output_files):
    counts = decision_counts(rows)
    final = summary.get("residual_after_output_accepted") or summary.get("residual_after", {})
    lines = [
        "# Outer Tower Tag Refine Summary",
        "",
        f"refine_output_dir: `{refine_dir}`",
        "",
        "## Final Accepted Reprojection Error",
        "",
        "Residuals are final accepted-output reprojection errors in pixels.",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| count | {final.get('count', '')} |",
        f"| median_px | {fmt(final.get('median_px'))} |",
        f"| mean_px | {fmt(final.get('mean_px'))} |",
        f"| p90_px | {fmt(final.get('p90_px'))} |",
        f"| max_px | {fmt(final.get('max_px'))} |",
        "",
        "## Camera Decisions",
        "",
        "| decision | count |",
        "| --- | ---: |",
    ]
    for decision in sorted(counts):
        lines.append(f"| {decision} | {counts[decision]} |")
    lines.extend([
        "",
        "## Per-Camera Residuals",
        "",
        "| idx | camera | decision | obs | final med | final p90 | final max | <100 | <300 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sorted(rows, key=sort_key):
        lines.append(
            "| {camera_index} | {camera_id} | {decision} | {observation_count} | "
            "{final_median_px} | {final_p90_px} | {final_max_px} | "
            "{final_under_100_fraction} | {final_under_300_fraction} |".format(**row))
    lines.extend([
        "",
        "## Output Tables",
        "",
    ])
    for label, path in output_files:
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def html_cell(value):
    return html.escape(str(value))


def render_html(refine_dir, rows, status_rows, summary, output_files):
    counts = decision_counts(rows)
    final = summary.get("residual_after_output_accepted") or summary.get("residual_after", {})
    count_rows = "\n".join(
        f"<tr><td>{html_cell(decision)}</td><td>{count}</td></tr>"
        for decision, count in sorted(counts.items()))
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html_cell(row['camera_index'])}</td>"
        f"<td>{html_cell(row['camera_id'])}</td>"
        f"<td>{html_cell(row['decision'])}</td>"
        f"<td>{html_cell(row['observation_count'])}</td>"
        f"<td>{html_cell(row['final_median_px'])}</td>"
        f"<td>{html_cell(row['final_p90_px'])}</td>"
        f"<td>{html_cell(row['final_max_px'])}</td>"
        f"<td>{html_cell(row['final_under_100_fraction'])}</td>"
        f"<td>{html_cell(row['final_under_300_fraction'])}</td>"
        "</tr>"
        for row in sorted(rows, key=sort_key))
    file_items = "\n".join(
        f"<li><code>{html_cell(path)}</code></li>"
        for _label, path in output_files)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Outer Tower Tag Refine Summary</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; line-height: 1.4; color: #1f2933; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d8dee4; padding: 6px 8px; text-align: left; }}
th {{ background: #f6f8fa; }}
td:nth-child(n+4), th:nth-child(n+4) {{ text-align: right; }}
code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
.panel {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 12px; }}
</style>
</head>
<body>
<h1>Outer Tower Tag Refine Summary</h1>
<p><code>{html_cell(refine_dir)}</code></p>
<div class="grid">
<div class="panel">
<h2>Final Accepted Reprojection Error</h2>
<p>Residuals are final accepted-output reprojection errors in pixels.</p>
<table>
<tr><th>metric</th><th>value</th></tr>
<tr><td>count</td><td>{html_cell(final.get('count', ''))}</td></tr>
<tr><td>median_px</td><td>{fmt(final.get('median_px'))}</td></tr>
<tr><td>mean_px</td><td>{fmt(final.get('mean_px'))}</td></tr>
<tr><td>p90_px</td><td>{fmt(final.get('p90_px'))}</td></tr>
<tr><td>max_px</td><td>{fmt(final.get('max_px'))}</td></tr>
</table>
</div>
<div class="panel">
<h2>Camera Decisions</h2>
<table>
<tr><th>decision</th><th>count</th></tr>
{count_rows}
</table>
</div>
</div>
<h2>Per-Camera Residuals</h2>
<table>
<tr><th>idx</th><th>camera</th><th>decision</th><th>obs</th><th>final med</th><th>final p90</th><th>final max</th><th>&lt;100</th><th>&lt;300</th></tr>
{table_rows}
</table>
<h2>Output Tables</h2>
<ul>
{file_items}
</ul>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Summarize diagnostics from refine_outer_tower_delta_prior.py output_dir.")
    parser.add_argument("refine_output_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-format", choices=["html", "markdown", "both"], default="both")
    args = parser.parse_args()

    refine_dir = args.refine_output_dir
    diagnostics_dir = refine_dir / "diagnostics"
    reprojection_path = diagnostics_dir / "camera_reprojection_accepted.tsv"
    if not reprojection_path.exists():
        reprojection_path = diagnostics_dir / "camera_reprojection.tsv"
    if not reprojection_path.exists():
        raise FileNotFoundError(f"Missing required diagnostics table: {reprojection_path}")

    output_dir = args.output_dir or (diagnostics_dir / "refine_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_summary(refine_dir / "summary.json")
    reprojection_rows = read_tsv(reprojection_path)
    delta_rows = read_tsv(diagnostics_dir / "camera_delta.tsv")
    acceptance_rows = read_tsv(diagnostics_dir / "camera_acceptance.tsv")

    merged_rows = merge_camera_rows(reprojection_rows, delta_rows, acceptance_rows, summary)
    status_rows = make_status_rows(merged_rows)

    residual_tsv = output_dir / "per_camera_residuals.tsv"
    status_tsv = output_dir / "camera_status.tsv"
    residual_fields = [
        "camera_index", "camera_id", "decision", "decision_reason",
        "observation_count", "final_median_px", "final_p90_px", "final_max_px",
        "final_under_100_fraction", "final_under_300_fraction",
    ]
    status_fields = [
        "camera_index", "camera_id", "decision", "output_pose", "reason",
        "observation_count", "final_median_px", "final_p90_px",
        "final_under_300_fraction",
    ]
    write_tsv(residual_tsv, sorted(merged_rows, key=sort_key), residual_fields)
    write_tsv(status_tsv, status_rows, status_fields)

    output_files = [
        ("per-camera final reprojection residuals", residual_tsv.resolve()),
        ("camera status", status_tsv.resolve()),
    ]
    reports = []
    if args.report_format in ("markdown", "both"):
        md_path = output_dir / "outer_tag_refine_summary.md"
        md_path.write_text(
            render_markdown(refine_dir.resolve(), merged_rows, status_rows, summary, output_files),
            encoding="utf-8")
        reports.append(md_path)
    if args.report_format in ("html", "both"):
        html_path = output_dir / "outer_tag_refine_summary.html"
        html_path.write_text(
            render_html(refine_dir.resolve(), merged_rows, status_rows, summary, output_files),
            encoding="utf-8")
        reports.append(html_path)

    result = {
        "refine_output_dir": str(refine_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "camera_count": len(merged_rows),
        "decision_counts": decision_counts(merged_rows),
        "reports": [str(path.resolve()) for path in reports],
        "tables": [str(residual_tsv.resolve()), str(status_tsv.resolve())],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
