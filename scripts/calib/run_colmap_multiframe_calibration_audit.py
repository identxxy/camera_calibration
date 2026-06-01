#!/usr/bin/env python3
"""Run COLMAP current-calibration BA audits over multiple synchronized frames."""

import argparse
import csv
import datetime as _datetime
import html
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def parse_frames(text):
    frames = []
    for chunk in text.replace(",", " ").split():
        if not chunk:
            continue
        frames.append(int(chunk))
    if not frames:
        raise ValueError("No frame ids were provided")
    return frames


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return None
    return sum(values) / len(values)


def std(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if len(values) < 2:
        return 0.0 if values else None
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def percentile(values, q):
    values = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def weighted_mean(rows, section, key):
    total = 0.0
    weight = 0.0
    for row in rows:
        stat = row["summary"].get(section, {})
        value = stat.get(key)
        count = stat.get("count") or stat.get("observation_count") or 0
        if value is None or not count:
            continue
        total += float(value) * float(count)
        weight += float(count)
    if weight == 0:
        return None
    return total / weight


def weighted_rmse(rows, section):
    total = 0.0
    weight = 0.0
    for row in rows:
        stat = row["summary"].get(section, {})
        value = stat.get("rmse_px")
        count = stat.get("count") or stat.get("observation_count") or 0
        if value is None or not count:
            continue
        total += float(value) ** 2 * float(count)
        weight += float(count)
    if weight == 0:
        return None
    return math.sqrt(total / weight)


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def summarize_delta_series(values):
    return {
        "mean": mean(values),
        "std": std(values),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max_abs": max((abs(float(x)) for x in values if x is not None), default=None),
    }


def collect_camera_deltas(frame_outputs):
    by_camera = {}
    for item in frame_outputs:
        table = item["output_root"] / "camera_deltas.tsv"
        if not table.exists():
            continue
        for row in read_tsv(table):
            label = row["label"]
            entry = by_camera.setdefault(label, {
                "label": label,
                "camera_index": int(row["camera_index"]),
                "frames": 0,
                "fx_delta": [],
                "fy_delta": [],
                "cx_delta": [],
                "cy_delta": [],
                "translation_delta_m": [],
                "rotation_delta_deg": [],
            })
            entry["frames"] += 1
            for key in ["fx_delta", "fy_delta", "cx_delta", "cy_delta", "translation_delta_m", "rotation_delta_deg"]:
                raw = row.get(key)
                if raw not in (None, ""):
                    entry[key].append(float(raw))

    rows = []
    for label, entry in by_camera.items():
        rows.append({
            "label": label,
            "camera_index": entry["camera_index"],
            "frames": entry["frames"],
            "fx_mean": mean(entry["fx_delta"]),
            "fy_mean": mean(entry["fy_delta"]),
            "cx_mean": mean(entry["cx_delta"]),
            "cy_mean": mean(entry["cy_delta"]),
            "translation_mean_m": mean(entry["translation_delta_m"]),
            "translation_p90_m": percentile(entry["translation_delta_m"], 0.9),
            "rotation_mean_deg": mean(entry["rotation_delta_deg"]),
            "rotation_p90_deg": percentile(entry["rotation_delta_deg"], 0.9),
        })
    rows.sort(key=lambda row: row["camera_index"])
    return rows


def write_camera_delta_tsv(path, rows):
    fields = [
        "camera_index", "label", "frames",
        "fx_mean", "fy_mean", "cx_mean", "cy_mean",
        "translation_mean_m", "translation_p90_m",
        "rotation_mean_deg", "rotation_p90_deg",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(output_root, summary, frame_rows, camera_rows, url_prefix):
    def link(path, label):
        path = Path(path)
        href = html.escape(str(path))
        if url_prefix:
            rel = path.relative_to(output_root)
            href = html.escape(url_prefix.rstrip("/") + "/" + str(rel))
        return f'<a href="{href}">{html.escape(label)}</a>'

    css = """
    body { font-family: system-ui, sans-serif; margin: 28px; line-height: 1.45; color: #1f2933; }
    table { border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d5dce5; padding: 6px 8px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    th { background: #f3f6fa; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 24px; }
    .card { border: 1px solid #d5dce5; border-radius: 8px; padding: 12px; background: #fbfcfe; }
    .label { color: #607086; font-size: 12px; }
    .value { font-size: 22px; font-weight: 650; margin-top: 2px; }
    code { background: #f3f6fa; padding: 2px 4px; border-radius: 4px; }
    .note { color: #4b5b6c; max-width: 980px; }
    """
    cards = [
        ("Frames", summary["ok_frame_count"]),
        ("Weighted RMSE before", fmt(summary["aggregate"]["weighted_rmse_before_px"])),
        ("Weighted RMSE after", fmt(summary["aggregate"]["weighted_rmse_after_px"])),
        ("RMSE drop", fmt(summary["aggregate"]["weighted_rmse_drop_pct"], 2) + "%"),
    ]
    rows_html = []
    for row in frame_rows:
        s = row["summary"]
        before = s["reprojection_before"]
        after = s["reprojection_after"]
        rows_html.append(
            "<tr>"
            f"<td>{row['frame_id']}</td>"
            f"<td>{s['frame']['out_frame']}</td>"
            f"<td>{before.get('registered_image_count', '-')}</td>"
            f"<td>{fmt(before.get('points3D_count'), 0)}</td>"
            f"<td>{fmt(before.get('count'), 0)}</td>"
            f"<td>{fmt(before.get('rmse_px'))}</td>"
            f"<td>{fmt(after.get('rmse_px'))}</td>"
            f"<td>{fmt(after.get('median_px'))}</td>"
            f"<td>{fmt(after.get('p90_px'))}</td>"
            f"<td>{fmt(s['camera_delta_summary']['translation_delta_m']['mean_abs'])}</td>"
            f"<td>{fmt(s['camera_delta_summary']['rotation_delta_deg']['mean_abs'])}</td>"
            f"<td>{link(row['output_root'] / 'report.html', 'single-frame report')}</td>"
            "</tr>"
        )
    cam_html = []
    for row in camera_rows:
        cam_html.append(
            "<tr>"
            f"<td>{row['camera_index']}</td>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{row['frames']}</td>"
            f"<td>{fmt(row['fx_mean'])}</td>"
            f"<td>{fmt(row['fy_mean'])}</td>"
            f"<td>{fmt(row['cx_mean'])}</td>"
            f"<td>{fmt(row['cy_mean'])}</td>"
            f"<td>{fmt(row['translation_mean_m'])}</td>"
            f"<td>{fmt(row['translation_p90_m'])}</td>"
            f"<td>{fmt(row['rotation_mean_deg'])}</td>"
            f"<td>{fmt(row['rotation_p90_deg'])}</td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<meta charset="utf-8">
<title>COLMAP multi-frame calibration audit</title>
<style>{css}</style>
<h1>COLMAP Multi-frame Calibration Audit</h1>
<p class="note">This is a diagnostic audit initialized from the current studio calibration.
COLMAP BA is allowed to refine FULL_OPENCV intrinsics and camera poses per frame, so large
intrinsic/distortion changes should be interpreted as a warning instead of a production calibration.</p>
<div class="cards">
{''.join(f'<div class="card"><div class="label">{html.escape(k)}</div><div class="value">{html.escape(str(v))}</div></div>' for k, v in cards)}
</div>
<p><b>Current YAML:</b> <code>{html.escape(summary['current_yaml'])}</code></p>
<p><b>Staged root:</b> <code>{html.escape(summary['staged_root'])}</code></p>
<p><b>Frames:</b> <code>{html.escape(' '.join(str(x) for x in summary['frames']))}</code></p>
<h2>Frame Results</h2>
<table>
<thead><tr><th>Frame</th><th>Out frame</th><th>Registered</th><th>Points</th><th>Obs</th><th>RMSE before</th><th>RMSE after</th><th>Median after</th><th>P90 after</th><th>Mean dT m</th><th>Mean dR deg</th><th>Report</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
<h2>Per-camera Mean Delta Across Frames</h2>
<table>
<thead><tr><th>Idx</th><th>Camera</th><th>Frames</th><th>dFx mean</th><th>dFy mean</th><th>dCx mean</th><th>dCy mean</th><th>dT mean m</th><th>dT p90 m</th><th>dR mean deg</th><th>dR p90 deg</th></tr></thead>
<tbody>{''.join(cam_html)}</tbody>
</table>
"""
    (output_root / "report.html").write_text(html_text, encoding="utf-8")


def run_one_frame(args, frame_id, output_root):
    frame_root = output_root / f"frame_{frame_id:06d}"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("colmap_frame_calibration_audit.py")),
        "--current-yaml", str(args.current_yaml),
        "--staged-root", str(args.staged_root),
        "--output-root", str(frame_root),
        "--frame-id", str(frame_id),
        "--colmap-bin", str(args.colmap_bin),
        "--max-image-size", str(args.max_image_size),
        "--max-num-features", str(args.max_num_features),
        "--num-threads", str(args.num_threads),
        "--ba-iterations", str(args.ba_iterations),
    ]
    if args.fix_intrinsics:
        cmd.append("--fix-intrinsics")
    if args.overwrite:
        cmd.append("--overwrite")
    log_path = output_root / "logs" / f"frame_{frame_id:06d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(cmd) + "\n\n")
        stream.flush()
        proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, frame_root, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-yaml", required=True, type=Path)
    parser.add_argument("--staged-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frames", required=True, help="Frame ids separated by comma or spaces.")
    parser.add_argument("--colmap-bin", default="/home/ubuntu/miniconda3/envs/colmap4/bin/colmap")
    parser.add_argument("--max-image-size", type=int, default=3200)
    parser.add_argument("--max-num-features", type=int, default=12000)
    parser.add_argument("--num-threads", type=int, default=12)
    parser.add_argument("--ba-iterations", type=int, default=80)
    parser.add_argument("--url-prefix", default="")
    parser.add_argument("--fix-intrinsics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    if output_root.exists() and args.overwrite:
        import shutil
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frames = parse_frames(args.frames)
    start = time.time()
    frame_outputs = []
    failed = []
    for frame_id in frames:
        print(f"[multi-frame audit] running frame {frame_id}", flush=True)
        returncode, frame_root, log_path = run_one_frame(args, frame_id, output_root)
        if returncode != 0:
            failed.append({"frame_id": frame_id, "returncode": returncode, "log": str(log_path)})
            print(f"[multi-frame audit] frame {frame_id} failed; see {log_path}", flush=True)
            continue
        summary_path = frame_root / "summary.json"
        if not summary_path.exists():
            failed.append({"frame_id": frame_id, "returncode": returncode, "log": str(log_path), "error": "missing summary.json"})
            continue
        frame_outputs.append({
            "frame_id": frame_id,
            "output_root": frame_root,
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        })

    camera_rows = collect_camera_deltas(frame_outputs)
    write_camera_delta_tsv(output_root / "camera_delta_means.tsv", camera_rows)

    aggregate = {
        "weighted_rmse_before_px": weighted_rmse(frame_outputs, "reprojection_before"),
        "weighted_rmse_after_px": weighted_rmse(frame_outputs, "reprojection_after"),
        "weighted_mean_before_px": weighted_mean(frame_outputs, "reprojection_before", "mean_px"),
        "weighted_mean_after_px": weighted_mean(frame_outputs, "reprojection_after", "mean_px"),
        "median_after_mean_px": mean([row["summary"]["reprojection_after"].get("median_px") for row in frame_outputs]),
        "p90_after_mean_px": mean([row["summary"]["reprojection_after"].get("p90_px") for row in frame_outputs]),
        "registered_image_count_mean": mean([row["summary"]["reprojection_after"].get("registered_image_count") for row in frame_outputs]),
        "points3D_count_mean": mean([row["summary"]["reprojection_after"].get("points3D_count") for row in frame_outputs]),
        "observation_count_sum": sum(int(row["summary"]["reprojection_after"].get("count") or 0) for row in frame_outputs),
        "translation_delta_mean_m": mean([row["summary"]["camera_delta_summary"]["translation_delta_m"]["mean_abs"] for row in frame_outputs]),
        "rotation_delta_mean_deg": mean([row["summary"]["camera_delta_summary"]["rotation_delta_deg"]["mean_abs"] for row in frame_outputs]),
    }
    if aggregate["weighted_rmse_before_px"] and aggregate["weighted_rmse_after_px"] is not None:
        aggregate["weighted_rmse_drop_px"] = aggregate["weighted_rmse_before_px"] - aggregate["weighted_rmse_after_px"]
        aggregate["weighted_rmse_drop_pct"] = 100.0 * aggregate["weighted_rmse_drop_px"] / aggregate["weighted_rmse_before_px"]
    else:
        aggregate["weighted_rmse_drop_px"] = None
        aggregate["weighted_rmse_drop_pct"] = None

    summary = {
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.time() - start,
        "command": " ".join(sys.argv),
        "current_yaml": str(args.current_yaml),
        "staged_root": str(args.staged_root),
        "output_root": str(output_root),
        "ba_intrinsics_mode": "fixed" if args.fix_intrinsics else "refined",
        "frames": frames,
        "ok_frame_count": len(frame_outputs),
        "failed": failed,
        "aggregate": aggregate,
        "frame_summaries": [
            {
                "frame_id": row["frame_id"],
                "output_root": str(row["output_root"]),
                "reprojection_before": row["summary"]["reprojection_before"],
                "reprojection_after": row["summary"]["reprojection_after"],
                "camera_delta_summary": row["summary"]["camera_delta_summary"],
            }
            for row in frame_outputs
        ],
    }
    write_json(output_root / "summary.json", summary)
    make_report(output_root, summary, frame_outputs, camera_rows, args.url_prefix)
    print(json.dumps({
        "output_root": str(output_root),
        "report": str(output_root / "report.html"),
        "ok_frame_count": len(frame_outputs),
        "failed": failed,
        "aggregate": aggregate,
    }, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
