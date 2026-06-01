#!/usr/bin/env python3
"""Audit calibration report artifacts on t0 and generate a cleanup review page."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import html
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import quote


DEFAULT_ROOT = "/home/ubuntu/calib_data"
DEFAULT_BASE_URL = "http://192.168.2.0:9899"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/calib_data/report_audit_20260601_current"

STAGE_ROOT = "calib_2026_05_26_jpg_v3"
PIPELINE_ROOT = f"{STAGE_ROOT}/recalib_pipelines"
CURRENT_RUN_ROOT = "studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50"
FAST_INNER_BRIDGE_LATEST = f"{CURRENT_RUN_ROOT}/inner_bridge"
OUTER_TOWER_LATEST = f"{CURRENT_RUN_ROOT}/outer_tower/frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1"
LEGACY_OUTER_TOWER_LATEST = f"{PIPELINE_ROOT}/outer_tower/latest"
CURRENT_WHOLE_ROOT = "calib_2026_05_31_v3"
CURRENT_WHOLE_OUTER24 = f"{CURRENT_WHOLE_ROOT}/whole_outer24_filtered_min4_hybrid_min4cam"
CURRENT_WHOLE_ALL32 = f"{CURRENT_WHOLE_ROOT}/whole_all32_filtered_min4_hybrid_min4cam"
CURRENT_OUTER = OUTER_TOWER_LATEST
STABLE_INNER_VIEWER = (
    f"{STAGE_ROOT}/final_inner8_calibration_v1/reports/interactive_rig_viewer_v1/index.html"
)
CURRENT_STUDIO32_YAML = (
    f"{CURRENT_RUN_ROOT}/calibration_artifacts/"
    "studio_32_cameras_current/studio_32_cameras.yaml"
)

KNOWN_KEEP_CURRENT = {
    f"{FAST_INNER_BRIDGE_LATEST}/combined_studio_rig_viewer_v1/index.html":
        "current overall 24+8 viewer entry",
    CURRENT_STUDIO32_YAML:
        "current machine-readable 32-camera calibration artifact",
    f"{CURRENT_OUTER}/index.html":
        "current outer solve report from 2026-05-31 capture",
    f"{CURRENT_WHOLE_OUTER24}/index.html":
        "2026-05-31 outer24 filtered capture quality index",
    f"{CURRENT_WHOLE_ALL32}/index.html":
        "2026-05-31 all32 filtered capture quality index",
    f"{CURRENT_WHOLE_OUTER24}/opencv_tower_dataset_fullres_coverage/coverage_report.html":
        "outer24 full-resolution AprilTag coverage report",
    f"{CURRENT_WHOLE_OUTER24}/opencv_tower_dataset_fullres_corner_offset2_coverage/coverage_report.html":
        "outer24 corner-offset2 AprilTag coverage report",
    f"{FAST_INNER_BRIDGE_LATEST}/quality_report/index.html":
        "inner/bridge data collection quality report",
    f"{FAST_INNER_BRIDGE_LATEST}/final_report/index.html":
        "fast inner/bridge final report",
    f"{FAST_INNER_BRIDGE_LATEST}/summary.json":
        "fast inner/bridge machine-readable summary",
    f"{CURRENT_OUTER}/summary.json":
        "current outer24 machine-readable summary",
}

KNOWN_KEEP_HISTORICAL = {
    STABLE_INNER_VIEWER:
        "standalone inner8 viewer kept as historical diagnostic; canonical UI uses the unified viewer",
    f"{LEGACY_OUTER_TOWER_LATEST}/viewer/index.html":
        "stable 2026-05-26 outer tower viewer kept as historical comparison",
    f"{LEGACY_OUTER_TOWER_LATEST}/final_report/index.html":
        "stable 2026-05-26 outer tower final report kept as historical comparison",
    f"{LEGACY_OUTER_TOWER_LATEST}/quality_report/index.html":
        "stable 2026-05-26 outer tower quality report kept as historical comparison",
    f"{LEGACY_OUTER_TOWER_LATEST}/summary.json":
        "stable 2026-05-26 outer tower machine-readable summary",
}

KNOWN_DELETE_DIRS = {
    f"{FAST_INNER_BRIDGE_LATEST}/reports/interactive_inner_viewer":
        "verified blank inner viewer artifact: rig_data has cameras but no first-frame textures",
}

KNOWN_DIAGNOSTIC = {
    f"{CURRENT_WHOLE_ROOT}/distributed_qc/index.html":
        "run/collection log; current snapshot records w1-w4 hostname failures, not final data quality",
    f"{CURRENT_WHOLE_ROOT}/distributed_qc/distributed_summary.json":
        "machine-readable run log for distributed QC hostname failure diagnostics",
}

REPORT_SUFFIXES = {
    ".html",
    ".json",
}
HTML_NAME_RE = re.compile(r"(index|report|viewer|coverage|summary|diagnostic|qc)", re.I)
JSON_NAME_RE = re.compile(r"(summary|report|metrics|rig_data|scene_data|diagnostic|qc)", re.I)
PRUNE_DIR_RE = re.compile(
    r"^(cam\d+|camera_\d+|camera_frames|frames|images|raw|rgb|jpg|jpeg|"
    r"thumbnails|sparse|dense|database|colmap_workspace|conda_.*|env|envs|"
    r"bin|lib|include|share|pkgs|__pycache__|node_modules|site-packages)$",
    re.I,
)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Entry:
    rel_path: str
    url: str
    kind: str
    category: str
    action: str
    reason: str
    size_bytes: int
    mtime_iso: str
    title: str
    signals: list[str]
    delete_group: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def rel_url(base_url: str, rel_path: str) -> str:
    quoted = "/".join(quote(part) for part in rel_path.split("/"))
    return f"{base_url.rstrip('/')}/{quoted}"


def should_skip(path: Path, root: Path, output_dir: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    parts = set(rel.parts)
    if {"site-packages", "node_modules", "__pycache__", ".git"} & parts:
        return True
    try:
        path.relative_to(output_dir)
        return True
    except ValueError:
        return False


def is_report_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name
    if suffix == ".html" and HTML_NAME_RE.search(name):
        return True
    if suffix == ".json" and JSON_NAME_RE.search(name):
        return True
    return False


def strip_markup(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return " ".join(text.split())


def read_text_sample(path: Path, limit: int = 262144) -> str:
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit)
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return ""


def extract_title(path: Path) -> str:
    if path.suffix.lower() != ".html":
        return ""
    sample = read_text_sample(path)
    for regex in (TITLE_RE, H1_RE):
        match = regex.search(sample)
        if match:
            return strip_markup(match.group(1))[:160]
    return ""


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def detect_json_signals(path: Path, rel_path: str, payload) -> list[str]:
    signals = []
    if not isinstance(payload, dict):
        return signals
    if "cameras" in payload and isinstance(payload["cameras"], list):
        signals.append(f"cameras={len(payload['cameras'])}")
        texture_count = 0
        for cam in payload["cameras"]:
            if isinstance(cam, dict) and (cam.get("image_url") or cam.get("thumbnail_url")):
                texture_count += 1
        signals.append(f"camera_textures={texture_count}")
    if "frames" in payload and isinstance(payload["frames"], list):
        signals.append(f"frames={len(payload['frames'])}")
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        metrics = payload["metrics"]
        for key in ("outer_pose_source", "mean_reprojection_error_px", "accepted", "status"):
            if key in metrics:
                signals.append(f"{key}={metrics[key]}")
    for key in ("status", "success", "ok", "dry_run", "errors", "failed"):
        if key in payload:
            signals.append(f"{key}={payload[key]}")
    if "distributed_summary" in rel_path or "distributed_qc" in rel_path:
        text = json.dumps(payload, ensure_ascii=False)[:20000]
        if "Name or service not known" in text or "Could not resolve hostname" in text:
            signals.append("hostname_resolution_failure")
    return signals


def detect_html_signals(path: Path) -> list[str]:
    if path.suffix.lower() != ".html":
        return []
    sample = read_text_sample(path)
    signals = []
    if "Outer Tower Viewer Placeholder" in sample or "data-viewer-placeholder" in sample:
        signals.append("placeholder_viewer")
    if "Name or service not known" in sample or "Could not resolve hostname" in sample:
        signals.append("hostname_resolution_failure")
    if "dry-run" in sample.lower() or "dry_run" in sample.lower():
        signals.append("dry_run")
    return signals


def classify(root: Path, path: Path, rel_path: str, signals: list[str]) -> tuple[str, str, str, str]:
    rel_dir = str(Path(rel_path).parent)
    for delete_dir, reason in KNOWN_DELETE_DIRS.items():
        if rel_path == delete_dir or rel_path.startswith(f"{delete_dir}/"):
            return "delete_known_bad", "delete", reason, delete_dir
    if rel_path in KNOWN_KEEP_CURRENT:
        return "keep_current", "keep", KNOWN_KEEP_CURRENT[rel_path], ""
    if rel_path in KNOWN_KEEP_HISTORICAL:
        return "keep_historical", "keep", KNOWN_KEEP_HISTORICAL[rel_path], ""
    if rel_path in KNOWN_DIAGNOSTIC:
        return "diagnostic_not_primary", "ask", KNOWN_DIAGNOSTIC[rel_path], rel_dir
    if "placeholder_viewer" in signals:
        return "delete_candidate", "ask", "viewer is an explicit placeholder, not a real visualization", rel_dir
    if "hostname_resolution_failure" in signals:
        return "diagnostic_not_primary", "ask", "diagnostic/run log contains hostname resolution failures", rel_dir
    lower_rel = rel_path.lower()
    if "panel_runs/" in lower_rel:
        return "scratch_or_dry_run", "ask", "panel dry-run/scratch output; usually safe to delete after review", rel_dir
    if "dry_run" in lower_rel or "dry-run" in lower_rel or "dry_run" in signals:
        return "scratch_or_dry_run", "ask", "dry-run artifact; usually safe to delete after review", rel_dir
    if any(token in lower_rel for token in ("debug", "diagnostic", "tmp", "scratch")):
        return "diagnostic_not_primary", "ask", "debug/diagnostic artifact; not a primary report entry", rel_dir
    if path.suffix.lower() == ".json" and path.name in {"rig_data.json", "scene_data.json"}:
        if (root / Path(rel_path).with_name("index.html")).exists():
            return "viewer_data", "keep_with_viewer", "viewer payload file; keep if the sibling viewer is kept", ""
        return "unclear", "ask", "viewer payload without an obvious sibling index.html", rel_dir
    if rel_path.endswith("index.html") or rel_path.endswith("coverage_report.html"):
        return "unclear", "ask", "report/viewer not in curated keep list; needs owner decision", rel_dir
    return "machine_summary", "ask", "machine-readable report/summary not in curated keep list", rel_dir


def make_entry(root: Path, base_url: str, path: Path) -> Entry:
    rel_path = path.relative_to(root).as_posix()
    stat = path.stat()
    signals = detect_html_signals(path)
    payload = None
    if path.suffix.lower() == ".json":
        payload = read_json(path)
        signals.extend(detect_json_signals(path, rel_path, payload))
    title = extract_title(path)
    kind = "html report"
    if "viewer" in rel_path.lower():
        kind = "3D viewer" if path.suffix.lower() == ".html" else "viewer data"
    elif "coverage" in rel_path.lower():
        kind = "coverage report"
    elif path.suffix.lower() == ".json":
        kind = "machine-readable report"
    category, action, reason, delete_group = classify(root, path, rel_path, signals)
    return Entry(
        rel_path=rel_path,
        url=rel_url(base_url, rel_path),
        kind=kind,
        category=category,
        action=action,
        reason=reason,
        size_bytes=stat.st_size,
        mtime_iso=time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(stat.st_mtime)),
        title=title,
        signals=signals,
        delete_group=delete_group,
    )


def collect_entries(root: Path, output_dir: Path, base_url: str) -> list[Entry]:
    entries = []
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if should_skip(current, root, output_dir):
            dirnames[:] = []
            continue
        dirnames[:] = [
            dirname for dirname in dirnames
            if not PRUNE_DIR_RE.match(dirname)
            and not should_skip(current / dirname, root, output_dir)
        ]
        for filename in filenames:
            path = current / filename
            if not is_report_file(path):
                continue
            try:
                key = path.resolve()
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            entries.append(make_entry(root, base_url, path))
    entries.sort(key=lambda item: (item.category, item.rel_path))
    return entries


def summarize_delete_groups(entries: list[Entry]) -> list[dict]:
    grouped = defaultdict(list)
    reasons = {}
    for entry in entries:
        if entry.delete_group and entry.action in {"delete", "ask"}:
            grouped[entry.delete_group].append(entry)
            reasons.setdefault(entry.delete_group, entry.reason)
    rows = []
    for group, group_entries in sorted(grouped.items()):
        actions = Counter(entry.action for entry in group_entries)
        rows.append({
            "path": group,
            "url": "",
            "file_count": len(group_entries),
            "total_size_bytes": sum(entry.size_bytes for entry in group_entries),
            "strongest_action": "delete" if actions.get("delete") else "ask",
            "reason": reasons[group],
            "sample_files": [entry.rel_path for entry in group_entries[:6]],
        })
    return rows


def fmt_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def render_table(entries: list[Entry]) -> str:
    rows = []
    for entry in entries:
        signals = ", ".join(entry.signals) if entry.signals else "-"
        title = entry.title or "-"
        rows.append(
            "<tr>"
            f"<td><span class='badge {html.escape(entry.category)}'>{html.escape(entry.category)}</span></td>"
            f"<td>{html.escape(entry.action)}</td>"
            f"<td><a href='{html.escape(entry.url)}'>{html.escape(entry.rel_path)}</a>"
            f"<div class='muted'>{html.escape(title)}</div></td>"
            f"<td>{html.escape(entry.kind)}</td>"
            f"<td>{html.escape(fmt_size(entry.size_bytes))}</td>"
            f"<td>{html.escape(entry.mtime_iso)}</td>"
            f"<td>{html.escape(entry.reason)}</td>"
            f"<td>{html.escape(signals)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_delete_groups(groups: list[dict]) -> str:
    rows = []
    for group in groups:
        samples = "<br>".join(html.escape(path) for path in group["sample_files"])
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(group['strongest_action'])}</strong></td>"
            f"<td>{html.escape(group['path'])}</td>"
            f"<td>{group['file_count']}</td>"
            f"<td>{html.escape(fmt_size(group['total_size_bytes']))}</td>"
            f"<td>{html.escape(group['reason'])}</td>"
            f"<td class='muted'>{samples}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_html(root: Path, base_url: str, output_dir: Path, entries: list[Entry], groups: list[dict]) -> str:
    counts = Counter(entry.category for entry in entries)
    action_counts = Counter(entry.action for entry in entries)
    generated = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    category_cards = "\n".join(
        f"<div class='card'><div class='num'>{count}</div><div>{html.escape(category)}</div></div>"
        for category, count in sorted(counts.items())
    )
    action_cards = "\n".join(
        f"<div class='card'><div class='num'>{count}</div><div>{html.escape(action)}</div></div>"
        for action, count in sorted(action_counts.items())
    )
    json_url = rel_url(base_url, f"{output_dir.relative_to(root).as_posix()}/report_inventory.json")
    css = """
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f7f7f4; color: #202124; }
    header { padding: 28px 34px 18px; background: #ffffff; border-bottom: 1px solid #deded8; }
    h1 { margin: 0 0 8px; font-size: 28px; font-weight: 680; letter-spacing: 0; }
    h2 { margin: 34px 0 12px; font-size: 20px; }
    main { padding: 24px 34px 42px; }
    .muted { color: #676b70; font-size: 12px; margin-top: 4px; }
    .summary { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 8px; }
    .card { background: #fff; border: 1px solid #deded8; border-radius: 8px; padding: 10px 14px; min-width: 120px; }
    .num { font-size: 24px; font-weight: 700; line-height: 1.1; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #deded8; }
    th, td { border-bottom: 1px solid #ecece7; padding: 9px 10px; text-align: left; vertical-align: top; font-size: 13px; }
    th { background: #eeeeea; font-weight: 650; position: sticky; top: 0; z-index: 1; }
    a { color: #175ea8; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .badge { display: inline-block; border-radius: 5px; padding: 2px 7px; font-size: 12px; white-space: nowrap; background: #eceff3; }
    .keep_current { background: #dcebdd; }
    .keep_historical, .viewer_data { background: #e7e8db; }
    .delete_known_bad, .delete_candidate { background: #f4d7d3; }
    .diagnostic_not_primary, .scratch_or_dry_run { background: #f0dfbf; }
    .unclear, .machine_summary { background: #e2e0f4; }
    code { background: #eeeeea; padding: 1px 4px; border-radius: 4px; }
    .note { max-width: 1100px; line-height: 1.5; }
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>t0 Calibration Report Audit</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>t0 Calibration Report Audit</h1>
    <div class="muted">Generated: {html.escape(generated)} · Root: <code>{html.escape(str(root))}</code></div>
    <p class="note">这是只读扫描结果，用来决定哪些 report/viewer 保留、删除或需要确认。物理删除前请先看下面的 delete groups 和 unclear/ask 项。</p>
    <p><a href="{html.escape(json_url)}">report_inventory.json</a></p>
  </header>
  <main>
    <h2>Category Summary</h2>
    <div class="summary">{category_cards}</div>
    <h2>Action Summary</h2>
    <div class="summary">{action_cards}</div>
    <h2>Delete / Ask Groups</h2>
    <table>
      <thead><tr><th>Action</th><th>Directory</th><th>Files</th><th>Size</th><th>Reason</th><th>Sample files</th></tr></thead>
      <tbody>{render_delete_groups(groups)}</tbody>
    </table>
    <h2>All Report / Viewer Artifacts</h2>
    <table>
      <thead><tr><th>Category</th><th>Action</th><th>Path</th><th>Kind</th><th>Size</th><th>Mtime</th><th>Reason</th><th>Signals</th></tr></thead>
      <tbody>{render_table(entries)}</tbody>
    </table>
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not root.exists():
        raise SystemExit(f"root does not exist: {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = collect_entries(root, output_dir, args.base_url)
    groups = summarize_delete_groups(entries)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "root": str(root),
        "base_url": args.base_url,
        "summary": {
            "categories": dict(Counter(entry.category for entry in entries)),
            "actions": dict(Counter(entry.action for entry in entries)),
            "entry_count": len(entries),
        },
        "delete_groups": groups,
        "entries": [asdict(entry) for entry in entries],
    }
    (output_dir / "report_inventory.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        render_html(root, args.base_url, output_dir, entries, groups),
        encoding="utf-8",
    )
    print(output_dir / "index.html")
    print(output_dir / "report_inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
