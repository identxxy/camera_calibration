#!/usr/bin/env python3
"""Generate a compact HTML report for frame-face outer tower BA."""

import argparse
import csv
import html
import json
from pathlib import Path


def esc(value):
    return html.escape(str(value if value is not None else ""))


def metric_row(name, value):
    return f"<tr><th>{esc(name)}</th><td>{esc(value)}</td></tr>"


def read_camera_rows(output_dir):
    path = output_dir / "diagnostics" / "camera_reprojection.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def run(args):
    output_dir = Path(args.output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    camera_rows = read_camera_rows(output_dir)
    residual = summary["residual_after"]
    observations = summary["observations"]
    frame_faces = summary["frame_faces"]
    cameras = summary["cameras"]

    summary_rows = [
        metric_row("Model", summary["settings"]["model"]),
        metric_row(
            "Residual median / p90 / max",
            f"{residual['median_px']:.3f} / {residual['p90_px']:.3f} / {residual['max_px']:.3f} px"),
        metric_row("Used tag corners", observations["used"]),
        metric_row("Raw tag corners", observations["raw"]),
        metric_row(
            "Frame-face groups used / initialized / observed",
            f"{frame_faces['used']} / {frame_faces['initialized']} / {frame_faces['total_observed']}"),
        metric_row("Active delta cameras", f"{cameras['active_delta']} / {cameras['total']}"),
        metric_row("Inactive/under-constrained cameras", ", ".join(cameras["inactive_delta"])),
        metric_row("Pose YAML", "camera_tr_rig_delta_refined.yaml"),
        metric_row("Intrinsics", "intrinsics_refined/"),
    ]

    camera_header = [
        "camera_id",
        "observation_count",
        "after_median_px",
        "after_p90_px",
        "after_max_px",
    ]
    camera_table = "".join(
        "<tr>" + "".join(f"<td>{esc(row.get(key, ''))}</td>" for key in camera_header) + "</tr>"
        for row in camera_rows)
    header_cells = "".join(f"<th>{esc(key)}</th>" for key in camera_header)

    text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Outer Tower Frame-Face BA Report</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; color: #202124; }}
h1 {{ margin-bottom: 8px; }}
.note {{ color: #5f6368; max-width: 980px; line-height: 1.45; }}
table {{ border-collapse: collapse; margin: 20px 0; font-size: 14px; }}
th, td {{ border: 1px solid #dadce0; padding: 7px 10px; text-align: left; }}
th {{ background: #f1f3f4; }}
.good {{ color: #137333; font-weight: 600; }}
code {{ background: #f1f3f4; padding: 2px 4px; border-radius: 4px; }}
</style></head><body>
<h1>Outer Tower Frame-Face BA Report</h1>
<p class="note">This run treats every synchronized frame and every visible tower face as an independent planar AprilTag target. It does not use face width, octagonal-prism geometry, or fixed face-to-face rotations.</p>
<h2>Summary</h2><table>{''.join(summary_rows)}</table>
<p class="note"><span class="good">Result:</span> global used-corner residual is median {residual['median_px']:.3f}px and p90 {residual['p90_px']:.3f}px. Cameras listed as inactive are not solved by this dataset and need targeted capture or bridge data.</p>
<h2>Per-Camera Residual</h2>
<table><tr>{header_cells}</tr>{camera_table}</table>
<h2>Files</h2>
<table>
<tr><th>camera poses</th><td><code>camera_tr_rig_delta_refined.yaml</code></td></tr>
<tr><th>camera priors</th><td><code>camera_tr_rig_prior.yaml</code></td></tr>
<tr><th>camera deltas</th><td><code>camera_delta_from_prior.yaml</code></td></tr>
<tr><th>frame-face poses</th><td><code>rig_tr_frame_face.yaml</code></td></tr>
<tr><th>diagnostics</th><td><code>diagnostics/</code></td></tr>
</table>
</body></html>
"""
    report_path = output_dir / "index.html"
    report_path.write_text(text, encoding="utf-8")
    print(report_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
