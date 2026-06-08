#!/usr/bin/env python3
"""Generate per-camera image-plane feature coverage and reprojection reports."""

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm

try:
    import generate_inner_calibration_report as inner_report
except ModuleNotFoundError:
    from scripts.calib import generate_inner_calibration_report as inner_report


REPROJECTION_COLORMAP_VMIN_PX = 1e-1
REPROJECTION_COLORMAP_VMAX_PX = 1e1


def finite_float(value, default=float("nan")):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_intrinsics_yaml(path):
    node = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {
        "type": node.get("type", node.get("model", "")),
        "width": int(node["width"]),
        "height": int(node["height"]),
        "parameters": np.asarray(node.get("parameters", []), dtype=np.float64),
        "path": str(Path(path).resolve(strict=False)),
    }


def load_intrinsics_for_camera(intrinsics_dir, camera_index):
    intrinsics_dir = Path(intrinsics_dir)
    candidates = [
        intrinsics_dir / f"intrinsics{camera_index}.yaml",
        *sorted(intrinsics_dir.glob(f"intrinsics{camera_index}_*.yaml")),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return load_intrinsics_yaml(candidate)
    raise FileNotFoundError(f"missing intrinsics for camera {camera_index} in {intrinsics_dir}")


def discover_intrinsics(intrinsics_dir):
    intrinsics_dir = Path(intrinsics_dir)
    pattern = re.compile(r"^intrinsics(\d+)(?:_(.+))?\.ya?ml$")
    discovered = {}
    for path in sorted(intrinsics_dir.glob("intrinsics*.y*ml")):
        match = pattern.match(path.name)
        if not match:
            continue
        camera_index = int(match.group(1))
        label = match.group(2) or f"camera{camera_index}"
        discovered.setdefault(camera_index, {"path": path, "label": label})
    return discovered


def read_tsv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def residuals_from_tsv(path, intrinsics_dir, include_statuses=None):
    include_statuses = set(include_statuses or ["ok", ""])
    grouped = {}
    labels = {}
    for row in read_tsv(path):
        status = row.get("projection_status", "")
        if status not in include_statuses:
            continue
        try:
            camera_index = int(row["camera_index"])
        except (KeyError, ValueError):
            continue

        observed_x = finite_float(row.get("observed_x"))
        observed_y = finite_float(row.get("observed_y"))
        if not (math.isfinite(observed_x) and math.isfinite(observed_y)):
            continue

        residual_x = finite_float(row.get("residual_x_px"))
        residual_y = finite_float(row.get("residual_y_px"))
        if not (math.isfinite(residual_x) and math.isfinite(residual_y)):
            projected_x = finite_float(row.get("projected_x"))
            projected_y = finite_float(row.get("projected_y"))
            if not (math.isfinite(projected_x) and math.isfinite(projected_y)):
                continue
            residual_x = projected_x - observed_x
            residual_y = projected_y - observed_y

        item = grouped.setdefault(camera_index, {"observed": [], "errors": [], "image_indices": []})
        item["observed"].append((observed_x, observed_y))
        item["errors"].append((residual_x, residual_y))
        item["image_indices"].append(int(finite_float(row.get("frame_index"), -1)))
        labels.setdefault(camera_index, row.get("camera_id", "") or f"camera{camera_index}")

    intrinsics_files = discover_intrinsics(intrinsics_dir)
    camera_indices = sorted(set(grouped) | set(intrinsics_files))
    cameras = []
    for camera_index in camera_indices:
        entry = grouped.get(camera_index, {"observed": [], "errors": [], "image_indices": []})
        residuals = {
            "observed": np.asarray(entry["observed"], dtype=np.float64).reshape((-1, 2)),
            "errors": np.asarray(entry["errors"], dtype=np.float64).reshape((-1, 2)),
            "image_indices": np.asarray(entry["image_indices"], dtype=np.int32),
            "skipped_missing_point": 0,
            "skipped_projection": 0,
        }
        label = labels.get(camera_index)
        if label is None and camera_index in intrinsics_files:
            label = intrinsics_files[camera_index]["label"]
        cameras.append({
            "camera_index": camera_index,
            "camera_label": label or f"camera{camera_index}",
            "intrinsics": load_intrinsics_for_camera(intrinsics_dir, camera_index),
            "residuals": residuals,
        })
    return cameras


def label_from_manifest(manifest, camera_index):
    row = manifest.get(camera_index, {})
    label_parts = []
    for key in ("camera_name", "staged_camera", "camera_id", "user_id", "sn"):
        if row.get(key):
            label_parts.append(row[key])
    return " / ".join(label_parts) if label_parts else f"camera{camera_index}"


def residuals_from_dataset_state(dataset_path, state_dir, manifest_path=""):
    dataset = inner_report.read_dataset(dataset_path)
    state = inner_report.load_state(state_dir, dataset["camera_count"])
    manifest = inner_report.load_manifest(manifest_path)
    cameras = []
    for camera_index in range(dataset["camera_count"]):
        cameras.append({
            "camera_index": camera_index,
            "camera_label": label_from_manifest(manifest, camera_index),
            "intrinsics": state["intrinsics"][camera_index],
            "residuals": inner_report.compute_camera_residuals(dataset, state, camera_index),
        })
    return cameras, {
        "camera_count": dataset["camera_count"],
        "imageset_count": len(dataset["imagesets"]),
    }


def sample_indices(count, max_count):
    if count <= max_count:
        return np.arange(count)
    rng = np.random.default_rng(20260603)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def parse_camera_indices(spec):
    if not spec:
        return None
    result = set()
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            start = int(start)
            end = int(end)
            if end < start:
                raise ValueError(f"invalid descending camera index range: {item}")
            result.update(range(start, end + 1))
        else:
            result.add(int(item))
    return result


def filter_cameras(cameras, index_spec):
    indices = parse_camera_indices(index_spec)
    if indices is None:
        return cameras
    return [camera for camera in cameras if int(camera["camera_index"]) in indices]


def plot_feature_coverage(camera_index, camera_label, intrinsics, residuals, output_path, max_arrows):
    observed = residuals["observed"]
    errors = residuals["errors"]
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    magnitudes = np.linalg.norm(errors, axis=1) if len(errors) else np.zeros(0, dtype=np.float64)

    fig_width = 5.4
    fig_height = fig_width * height / max(width, 1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.set_title(f"Camera {camera_index}: {camera_label}", fontsize=8)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [px]", fontsize=7)
    ax.set_ylabel("y [px]", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(color="#d0d7de", linewidth=0.35, alpha=0.55)

    if len(observed):
        heat = ax.hexbin(
            observed[:, 0],
            observed[:, 1],
            gridsize=80,
            extent=(0, width, 0, height),
            mincnt=1,
            bins="log",
            cmap="Blues",
            alpha=0.48,
            linewidths=0,
        )
        heat.set_zorder(0)

        indices = sample_indices(len(observed), max_arrows)
        obs = observed[indices]
        err = errors[indices]
        mag = np.clip(
            magnitudes[indices],
            REPROJECTION_COLORMAP_VMIN_PX,
            REPROJECTION_COLORMAP_VMAX_PX,
        )
        norm = LogNorm(vmin=REPROJECTION_COLORMAP_VMIN_PX, vmax=REPROJECTION_COLORMAP_VMAX_PX)
        lengths = np.linalg.norm(err, axis=1)
        scale_factor = np.ones_like(lengths)
        nonzero = lengths > 1e-9
        scaled_lengths = np.clip(lengths[nonzero] * 45.0, 2.0, 45.0)
        scale_factor[nonzero] = scaled_lengths / lengths[nonzero]
        err_display = err * scale_factor[:, None]
        quiver = ax.quiver(
            obs[:, 0],
            obs[:, 1],
            err_display[:, 0],
            err_display[:, 1],
            mag,
            cmap="turbo",
            norm=norm,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0013,
            headwidth=3.4,
            headlength=4.4,
            headaxislength=3.8,
            alpha=0.92,
        )
        cbar = fig.colorbar(quiver, ax=ax, shrink=0.82)
        cbar.set_label("reprojection error [px], log scale, 1e-1..1e1")
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.yaxis.label.set_size(7)
        ax.text(
            0.01,
            0.99,
            f"{len(observed):,} accumulated observed feature locations\n"
            f"{len(indices):,} reprojection arrows shown",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="#1f2328",
            fontsize=6.5,
            bbox={"facecolor": "#ffffff", "alpha": 0.76, "edgecolor": "none", "pad": 3},
        )
    else:
        ax.text(0.5, 0.5, "no valid residuals", transform=ax.transAxes, ha="center", va="center")

    fig.savefig(output_path, dpi=135)
    plt.close(fig)


def camera_metrics(camera, plot_path):
    residuals = camera["residuals"]
    observed = residuals["observed"]
    errors = residuals["errors"]
    magnitudes = np.linalg.norm(errors, axis=1) if len(errors) else np.zeros(0, dtype=np.float64)
    if len(magnitudes):
        median = float(np.median(magnitudes))
        mean = float(np.mean(magnitudes))
        p90 = float(np.percentile(magnitudes, 90))
        max_error = float(np.max(magnitudes))
        min_x, min_y = np.min(observed, axis=0)
        max_x, max_y = np.max(observed, axis=0)
        bbox_area = max(0.0, float(max_x - min_x)) * max(0.0, float(max_y - min_y))
        image_area = max(1.0, float(camera["intrinsics"]["width"]) * float(camera["intrinsics"]["height"]))
        bbox_fraction = bbox_area / image_area
        frame_count = int(len(set(int(v) for v in residuals.get("image_indices", []))))
    else:
        median = mean = p90 = max_error = float("nan")
        bbox_fraction = 0.0
        frame_count = 0
    return {
        "camera_index": int(camera["camera_index"]),
        "camera_label": str(camera["camera_label"]),
        "residual_count": int(len(magnitudes)),
        "frame_count": frame_count,
        "median_error_px": median,
        "mean_error_px": mean,
        "p90_error_px": p90,
        "max_error_px": max_error,
        "observed_bbox_area_fraction": float(bbox_fraction),
        "skipped_missing_point": int(residuals.get("skipped_missing_point", 0)),
        "skipped_projection": int(residuals.get("skipped_projection", 0)),
        "plot_path": str(plot_path),
    }


def fmt(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def write_metrics_tsv(path, rows):
    fields = [
        "camera_index",
        "camera_label",
        "residual_count",
        "frame_count",
        "median_error_px",
        "mean_error_px",
        "p90_error_px",
        "max_error_px",
        "observed_bbox_area_fraction",
        "skipped_missing_point",
        "skipped_projection",
        "plot_path",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_html(path, title, summary, camera_rows):
    table_rows = []
    for row in camera_rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['camera_index']}</td>"
            f"<td>{html.escape(row['camera_label'])}</td>"
            f"<td>{row['residual_count']:,}</td>"
            f"<td>{row['frame_count']:,}</td>"
            f"<td>{fmt(row['median_error_px'])}</td>"
            f"<td>{fmt(row['p90_error_px'])}</td>"
            f"<td>{fmt(row['observed_bbox_area_fraction'])}</td>"
            "</tr>"
        )

    figures = []
    for row in camera_rows:
        rel = Path(row["plot_path"]).name
        figures.append(
            "<section class='camera'>"
            f"<h2>Camera {row['camera_index']} <span>{html.escape(row['camera_label'])}</span></h2>"
            f"<img src='{html.escape(rel)}' alt='Camera {row['camera_index']} feature coverage and reprojection'>"
            "</section>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f7; color: #1f2328; }}
    header {{ padding: 28px 36px 18px; background: #20242a; color: white; }}
    header h1 {{ margin: 0 0 10px; font-size: 26px; }}
    header p {{ margin: 4px 0; color: #c9d1d9; }}
    main {{ padding: 20px 24px 32px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin-bottom: 28px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 9px 10px; text-align: right; font-size: 13px; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #eef1f4; font-weight: 650; }}
    .camera-grid {{ display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; align-items: start; }}
    .camera {{ margin: 0; background: white; padding: 7px; border: 1px solid #d8dee4; }}
    .camera h2 {{ margin: 0 0 6px; font-size: 12px; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .camera h2 span {{ color: #57606a; font-weight: 500; }}
    .camera img {{ display: block; width: 100%; object-fit: contain; border: 1px solid #d8dee4; background: #f8fafc; }}
    code {{ background: #eef1f4; color: #24292f; padding: 2px 5px; border-radius: 4px; }}
    @media (max-width: 1600px) {{ .camera-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }} }}
    @media (max-width: 900px) {{ .camera-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Source type: <code>{html.escape(summary['source_type'])}</code></p>
    <p>Source: <code>{html.escape(summary['source'])}</code></p>
    <p>Each plot shows accumulated observed feature locations as a log-density background; arrows show projected minus observed residual. The residual colormap is fixed to log scale from <code>10^-1</code> to <code>10^1</code> px for cross-camera comparison.</p>
  </header>
  <main>
    <table>
      <thead><tr><th>Camera</th><th>Label</th><th>Residuals</th><th>Frames</th><th>Median px</th><th>P90 px</th><th>Observed bbox frac</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
    <div class="camera-grid">{''.join(figures)}</div>
  </main>
</body>
</html>
"""
    Path(path).write_text(html_text, encoding="utf-8")


def generate_report(cameras, summary, output_dir, title, max_arrows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_rows = []
    for camera in cameras:
        camera_index = int(camera["camera_index"])
        plot_path = output_dir / f"camera{camera_index:02d}_feature_coverage_reprojection.png"
        plot_feature_coverage(
            camera_index,
            str(camera["camera_label"]),
            camera["intrinsics"],
            camera["residuals"],
            plot_path,
            max_arrows,
        )
        camera_rows.append(camera_metrics(camera, plot_path))

    summary = {
        **summary,
        "output_dir": str(output_dir.resolve(strict=False)),
        "camera_count": len(cameras),
    }
    (output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "cameras": camera_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_metrics_tsv(output_dir / "camera_metrics.tsv", camera_rows)
    write_html(output_dir / "index.html", title, summary, camera_rows)
    return output_dir / "index.html"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate image-plane intrinsic feature coverage and reprojection plots.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--residuals-tsv", type=Path)
    input_group.add_argument("--dataset", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--intrinsics-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--camera-indices", default="", help="Optional comma/range filter, e.g. 0-23 or 0,3,8.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="Intrinsic Feature Coverage Report")
    parser.add_argument("--max-arrows-per-camera", type=int, default=60000)
    parser.add_argument("--include-projection-status", default="ok")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.residuals_tsv:
        if not args.intrinsics_dir:
            parser.error("--intrinsics-dir is required with --residuals-tsv")
        statuses = [item.strip() for item in args.include_projection_status.split(",")]
        cameras = residuals_from_tsv(args.residuals_tsv, args.intrinsics_dir, statuses)
        summary = {
            "source_type": "residuals_tsv",
            "source": str(args.residuals_tsv.resolve(strict=False)),
            "intrinsics_dir": str(args.intrinsics_dir.resolve(strict=False)),
        }
    else:
        if not args.state_dir:
            parser.error("--state-dir is required with --dataset")
        cameras, dataset_summary = residuals_from_dataset_state(
            args.dataset,
            args.state_dir,
            args.manifest or "",
        )
        summary = {
            "source_type": "dataset_state",
            "source": str(args.dataset.resolve(strict=False)),
            "state_dir": str(args.state_dir.resolve(strict=False)),
            **dataset_summary,
        }

    cameras = filter_cameras(cameras, args.camera_indices)
    if args.camera_indices:
        summary["camera_indices"] = args.camera_indices

    index = generate_report(cameras, summary, args.output_dir, args.title, args.max_arrows_per_camera)
    print(index)


if __name__ == "__main__":
    main()
