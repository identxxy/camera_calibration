#!/usr/bin/env python3
"""Generate an HTML report from per-camera OpenCV intrinsic residual TSVs."""

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


REPROJECTION_COLORMAP_VMIN_PX = 1e-1
REPROJECTION_COLORMAP_VMAX_PX = 1e1


def read_summary(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def read_residuals(path):
    observed = []
    errors = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            observed.append((float(row["observed_x"]), float(row["observed_y"])))
            errors.append((float(row["error_x"]), float(row["error_y"])))
    if not observed:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.asarray(observed, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def sample_indices(count, max_count):
    if count <= max_count:
        return np.arange(count)
    rng = np.random.default_rng(20260527)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def plot_camera(row, observed, errors, output_path, max_arrows):
    width = int(row["width"])
    height = int(row["height"])
    magnitudes = np.linalg.norm(errors, axis=1)
    title = f"Camera {row['camera_index']}: {row['user_id']}"

    fig_width = 5.4
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    ax.set_title(title, fontsize=8)
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

        display_scale = 600.0
        lengths = np.linalg.norm(err, axis=1)
        scale_factor = np.ones_like(lengths)
        nonzero = lengths > 1e-12
        scaled_lengths = np.clip(lengths[nonzero] * display_scale, 2.0, 55.0)
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
            f"{len(observed):,} residuals, {len(indices):,} arrows shown",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="#1f2328",
            fontsize=6.5,
            bbox={"facecolor": "#ffffff", "alpha": 0.76, "edgecolor": "none", "pad": 3},
        )
    else:
        ax.text(0.5, 0.5, "no valid residuals", transform=ax.transAxes, ha="center", va="center")

    fig.savefig(output_path, dpi=110)
    plt.close(fig)


def float_or_nan(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def write_html(output_path, source_dir, camera_rows):
    table_rows = []
    figures = []
    for row in camera_rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['camera_index']}</td>"
            f"<td>{html.escape(row['stage_name'])}</td>"
            f"<td>{html.escape(row['user_id'])}</td>"
            f"<td>{row['usable_views']}</td>"
            f"<td>{row['usable_points']}</td>"
            f"<td>{row['rms']:.4f}</td>"
            f"<td>{row['median_error_px']:.4f}</td>"
            f"<td>{row['p90_error_px']:.4f}</td>"
            f"<td>{row['fx']:.2f}</td>"
            f"<td>{row['fy']:.2f}</td>"
            "</tr>"
        )
        figures.append(
            "<section class='camera'>"
            f"<h2>Camera {row['camera_index']} <span>{html.escape(row['user_id'])}</span></h2>"
            f"<img src='{html.escape(Path(row['plot_path']).name)}' loading='lazy' "
            f"alt='Camera {row['camera_index']} reprojection arrows'>"
            "</section>"
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
    <title>OpenCV Intrinsics Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f7; color: #1f2328; }}
    header {{ padding: 28px 36px 18px; background: #20242a; color: white; }}
    header h1 {{ margin: 0 0 10px; font-size: 26px; }}
    header p {{ margin: 4px 0; color: #c9d1d9; }}
    main {{ padding: 24px 36px 40px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin-bottom: 28px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: 9px 10px; text-align: right; font-size: 13px; }}
    th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #eef1f4; font-weight: 650; }}
    .camera-grid {{ display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; align-items: start; }}
    .camera {{ margin: 0; background: white; padding: 7px; border: 1px solid #d8dee4; }}
    .camera h2 {{ margin: 0 0 6px; font-size: 12px; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .camera h2 span {{ color: #57606a; font-weight: 500; }}
    .camera a {{ display: block; }}
    .camera img {{ display: block; width: 100%; object-fit: contain; border: 1px solid #d8dee4; background: #f8fafc; }}
    code {{ background: #eef1f4; padding: 2px 5px; border-radius: 4px; }}
    @media (max-width: 1600px) {{ .camera-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }} }}
    @media (max-width: 900px) {{ .camera-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  </style>
</head>
<body>
  <header>
    <h1>OpenCV Intrinsics Report</h1>
    <p>Source: <code>{html.escape(str(source_dir))}</code></p>
    <p>Arrows show projected minus observed residual after per-camera OpenCV calibration. The residual colormap is fixed to log scale from <code>10^-1</code> to <code>10^1</code> px for cross-camera comparison.</p>
  </header>
  <main>
    <table>
      <thead><tr><th>Camera</th><th>Stage</th><th>User ID</th><th>Views</th><th>Points</th><th>RMS px</th><th>Median px</th><th>P90 px</th><th>fx</th><th>fy</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
    <div class="camera-grid">{''.join(figures)}</div>
  </main>
</body>
</html>
"""
    Path(output_path).write_text(html_text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-arrows-per-camera", type=int, default=60000)
    args = parser.parse_args(argv)

    intrinsics_dir = Path(args.intrinsics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_summary(intrinsics_dir / "intrinsics_summary.tsv")
    camera_rows = []
    for row in summary_rows:
        if row.get("status") != "solved":
            continue
        camera_index = int(row["camera_index"])
        user_id = row["user_id"]
        residual_path = intrinsics_dir / f"residuals_camera{camera_index}_{user_id}.tsv"
        observed, errors = read_residuals(residual_path)
        magnitudes = np.linalg.norm(errors, axis=1)
        plot_path = output_dir / f"camera{camera_index:02d}_reprojection_arrows_log.png"
        plot_camera(row, observed, errors, plot_path, args.max_arrows_per_camera)

        camera_rows.append({
            "camera_index": camera_index,
            "stage_name": row["stage_name"],
            "machine": row["machine"],
            "user_id": user_id,
            "usable_views": int(row["usable_views"]),
            "usable_points": int(row["usable_points"]),
            "rms": float_or_nan(row["rms"]),
            "fx": float_or_nan(row["fx"]),
            "fy": float_or_nan(row["fy"]),
            "cx": float_or_nan(row["cx"]),
            "cy": float_or_nan(row["cy"]),
            "median_error_px": float(np.median(magnitudes)) if len(magnitudes) else float("nan"),
            "mean_error_px": float(np.mean(magnitudes)) if len(magnitudes) else float("nan"),
            "p90_error_px": float(np.percentile(magnitudes, 90)) if len(magnitudes) else float("nan"),
            "max_error_px": float(np.max(magnitudes)) if len(magnitudes) else float("nan"),
            "residual_count": int(len(magnitudes)),
            "plot_path": str(plot_path),
        })

    with (output_dir / "camera_metrics.tsv").open("w", encoding="utf-8") as f:
        fields = [
            "camera_index",
            "stage_name",
            "machine",
            "user_id",
            "usable_views",
            "usable_points",
            "rms",
            "median_error_px",
            "mean_error_px",
            "p90_error_px",
            "max_error_px",
            "fx",
            "fy",
            "cx",
            "cy",
            "residual_count",
            "plot_path",
        ]
        f.write("\t".join(fields) + "\n")
        for row in camera_rows:
            f.write("\t".join(str(row[field]) for field in fields) + "\n")

    (output_dir / "summary.json").write_text(json.dumps({
        "source_dir": str(intrinsics_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "cameras": camera_rows,
    }, indent=2), encoding="utf-8")
    write_html(output_dir / "index.html", intrinsics_dir.resolve(), camera_rows)
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
