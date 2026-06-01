#!/usr/bin/env python3
"""Serve t0 calibration reports from calib_data over HTTP."""

from __future__ import annotations

import argparse
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import quote, urlencode


DEFAULT_ROOT = "/home/ubuntu/calib_data"
DEFAULT_REPORT_BASE_URL = "http://192.168.2.0:9899"
DEFAULT_PANEL_URL = "http://192.168.2.0:9898/"
CURRENT_ENTRY_REL = "current_calibration/index.html"
STAGE_ROOT = "calib_2026_05_26_jpg_v3"
PIPELINE_ROOT = f"{STAGE_ROOT}/recalib_pipelines"
CURRENT_RUN_ROOT = "studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50"
FAST_INNER_BRIDGE_LATEST = f"{CURRENT_RUN_ROOT}/inner_bridge"
OUTER_TOWER_LATEST = f"{CURRENT_RUN_ROOT}/outer_tower/frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1"
CURRENT_WHOLE_ROOT = "calib_2026_05_31_v3"
CURRENT_WHOLE_OUTER24 = f"{CURRENT_WHOLE_ROOT}/whole_outer24_filtered_min4_hybrid_min4cam"
CURRENT_WHOLE_ALL32 = f"{CURRENT_WHOLE_ROOT}/whole_all32_filtered_min4_hybrid_min4cam"
CURRENT_OUTER_CANDIDATE = OUTER_TOWER_LATEST
UNIFIED_VIEWER = f"{FAST_INNER_BRIDGE_LATEST}/combined_studio_rig_viewer_v1/index.html"
STUDIO32_YAML = (
    f"{CURRENT_RUN_ROOT}/calibration_artifacts/"
    "studio_32_cameras_current/studio_32_cameras.yaml"
)
STABLE_INNER_VIEWER = (
    f"{STAGE_ROOT}/final_inner8_calibration_v1/reports/interactive_rig_viewer_v1/index.html"
)


EXCLUDED_REPORT_PATHS = {
    # This fast-pipeline artifact has camera poses but no first-frame textures or
    # sparse context, so it renders as an effectively blank viewer.
    f"{FAST_INNER_BRIDGE_LATEST}/reports/interactive_inner_viewer/index.html",
}


TOOL_LINKS = [
    {
        "label": "一键标定 Panel",
        "description": "采集后从 9898 panel 以 dry-run 优先启动 pipeline wrapper。",
        "kind": "operation",
        "url": "panel",
    },
]


REPORT_GROUPS = [
    {
        "title": "1. Inner Capture QC",
        "subtitle": "small_marker / large_marker calib board data quality",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "operate_small_marker_inner",
        "description": (
            "内圈 calib board 采集质量入口。这里只看同步、尾帧裁剪、掉帧排除、"
            "角点覆盖和可用相机集合。"
        ),
        "items": [
            {
                "label": "Small marker data QC",
                "path": f"{STAGE_ROOT}/small_marker_inner8/coverage_gate_pattern3_v1/coverage_report.html",
                "kind": "data collection quality",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
            {
                "label": "Large marker bridge input QC",
                "path": f"{FAST_INNER_BRIDGE_LATEST}/quality_report/index.html",
                "kind": "data collection quality",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [
            "这是采集质量，不是最终 solve result。",
            "operation 入口只负责触发受控 panel mode，不把临时 report 提升为首页入口。",
        ],
    },
    {
        "title": "2. Inner Solve Result",
        "subtitle": "inner 8-camera intrinsics/extrinsics quality",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "operate_small_marker_inner",
        "description": (
            "内圈 8 相机解算结果入口。最终 3D 姿态只在统一 viewer 中查看，"
            "这里保留 reprojection / intrinsics 报告。"
        ),
        "items": [
            {
                "label": "Small marker refined reprojection report",
                "path": (
                    f"{STAGE_ROOT}/final_inner8_calibration_v1/reports/"
                    "report_small_grid4_refined_reprojection_v1/index.html"
                ),
                "kind": "final report",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
            {
                "label": "Inner/bridge wrapper final report",
                "path": f"{FAST_INNER_BRIDGE_LATEST}/final_report/index.html",
                "kind": "final report",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [
            "standalone inner viewer 是历史诊断产物；当前首页只提升一个 unified 3D viewer。",
        ],
    },
    {
        "title": "3. Outer Capture QC",
        "subtitle": "whole AprilTag tower data quality",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "operate_whole_outer_cage",
        "description": (
            "whole / tower 采集质量入口。这里回答每台 outer camera 是否有足够 "
            "AprilTag 观测进入后续 refine。"
        ),
        "items": [
            {
                "label": "Whole outer24 data QC",
                "path": f"{CURRENT_WHOLE_OUTER24}/index.html",
                "kind": "data collection quality",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [
            "distributed logs、coverage drill-down 和 COLMAP audits 保留在 run directory，不出现在首页。",
        ],
    },
    {
        "title": "4. Outer Solve Result",
        "subtitle": "outer 24-camera final solve diagnostics",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "operate_whole_outer_cage",
        "description": (
            "当前外圈解算结果入口。production 报告基于 05-31 run root；"
            "底层 frame-face refine 不把八棱柱 face_width 几何当作生产入口。"
        ),
        "items": [
            {
                "label": "Outer solve final report",
                "path": f"{CURRENT_OUTER_CANDIDATE}/index.html",
                "kind": "final report",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
            {
                "label": "Outer solve summary.json",
                "path": f"{CURRENT_OUTER_CANDIDATE}/summary.json",
                "kind": "machine-readable summary",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [
            "旧 COLMAP vote、side-prior、face_width 和 report inventory 都是历史诊断，不在首页展开。",
        ],
    },
    {
        "title": "5. Combined Bridge / 32-Camera Result",
        "subtitle": "one canonical 3D viewer and one machine-readable YAML",
        "status": "pipeline",
        "status_label": "canonical",
        "panel_mode": "operate_large_marker_bridge",
        "description": (
            "最终统一入口。viewer 内部提供 inner only / outer only / combined 和 "
            "whole / large marker / small marker coverage 模式。"
        ),
        "items": [
            {
                "label": "Unified 3D viewer",
                "path": UNIFIED_VIEWER,
                "kind": "3D viewer",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
            {
                "label": "studio_32_cameras.yaml",
                "path": STUDIO32_YAML,
                "kind": "machine-readable calibration",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
            {
                "label": "Bridge summary.json",
                "path": f"{FAST_INNER_BRIDGE_LATEST}/bridge_colmap_inner_refined_v1/bridge_summary.json",
                "kind": "bridge summary",
                "status_if_exists": "ready",
                "status_if_missing": "not produced yet",
            },
        ],
        "notes": [
            "这是下游算法和人工检查共同使用的 canonical 32-camera result。",
        ],
    },
]


class ReportHandler(SimpleHTTPRequestHandler):
    server_version = "CameraCalibReportHTTP/1.0"

    def __init__(self, *args, directory=None, **kwargs):
        self.root = Path(directory or DEFAULT_ROOT).resolve()
        super().__init__(*args, directory=str(self.root), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        request_path = self.path.split("?", 1)[0]
        if request_path in ("/", "/index.html"):
            if self._serve_current_entry():
                return
            self._serve_index()
            return
        if self.path == "/healthz":
            self._serve_health()
            return
        super().do_GET()

    def _current_entry_path(self):
        path = self.root / CURRENT_ENTRY_REL
        if path.is_file():
            return path
        return None

    def _serve_current_entry(self):
        path = self._current_entry_path()
        if path is None:
            return False
        try:
            data = path.read_bytes()
        except OSError:
            return False
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return True

    def _serve_health(self):
        payload = {
            "ok": True,
            "root": str(self.root),
            "service": "camera-calibration-report-http",
        }
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _candidate_reports(self):
        names = set()
        patterns = ("**/index.html", "**/*report*.html")
        for pattern in patterns:
            for path in self.root.glob(pattern):
                if not path.is_file():
                    continue
                try:
                    rel = path.relative_to(self.root)
                except ValueError:
                    continue
                names.add(rel.as_posix())
        return sorted(
            names,
            key=lambda name: (self.root / name).stat().st_mtime,
            reverse=True,
        )[:200]

    def _same_host_panel_url(self):
        host = self.headers.get("Host", "127.0.0.1").split(",", 1)[0].strip()
        if host.startswith("["):
            hostname = host.split("]", 1)[0] + "]"
        else:
            hostname = host.rsplit(":", 1)[0]
        return f"http://{hostname}:9898/"

    def _curated_paths(self):
        paths = set()
        for group in REPORT_GROUPS:
            for item in group.get("items", []):
                if item.get("path"):
                    paths.add(item["path"])
        return paths

    def _path_href(self, rel):
        base_url = getattr(self.server, "report_base_url", DEFAULT_REPORT_BASE_URL)
        return base_url.rstrip("/") + "/" + quote(rel)

    def _pipeline_summary(self, rel):
        for prefix in (FAST_INNER_BRIDGE_LATEST, OUTER_TOWER_LATEST):
            if rel == prefix or rel.startswith(prefix + "/"):
                summary_path = self.root / prefix / "summary.json"
                if not summary_path.is_file():
                    return {}
                try:
                    return json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return {}
        return {}

    def _panel_mode_href(self, panel_url, mode):
        if not mode:
            return panel_url
        separator = "&" if "?" in panel_url else "?"
        return panel_url.rstrip("/") + "/" + separator + urlencode({"mode": mode})

    def _render_tool_link(self, item, panel_url):
        href = panel_url if item.get("url") == "panel" else item.get("url", "#")
        return (
            f"<a class=\"tool-link\" href=\"{html.escape(href)}\">"
            f"<strong>{html.escape(item['label'])}</strong>"
            f"<span>{html.escape(item['kind'])}</span>"
            f"<small>{html.escape(item['description'])}</small>"
            "</a>"
        )

    def _is_outer_placeholder_viewer(self, path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return True
        return (
            "Outer Tower Viewer Placeholder" in text
            or "data-viewer-placeholder" in text
        )

    def _viewer_ready(self, rel, path):
        if rel.endswith("outer_tower/latest/viewer/index.html"):
            return (
                (path.parent / "scene_data.json").is_file()
                and not self._is_outer_placeholder_viewer(path)
            )
        if rel.endswith("combined_studio_rig_viewer_v1/index.html"):
            rig_data_path = path.parent / "rig_data.json"
            if not rig_data_path.is_file():
                return False
            try:
                rig_data = json.loads(rig_data_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            current_outer = (
                self.root
                / OUTER_TOWER_LATEST
                / "camera_tr_rig_delta_refined.yaml"
            )
            legacy_outer = (
                self.root
                / OUTER_TOWER_LATEST
                / "tag_refine_robust/camera_tr_rig_delta_refined_accepted.yaml"
            )
            outer_final = ((rig_data.get("inputs") or {}).get("outer_final_pose_yaml") or "")
            outer_source = ((rig_data.get("metrics") or {}).get("outer_pose_source") or "")
            return (
                outer_source in {"outer_final_pose_yaml", "outer_final_pose_yaml_bridge_aligned"}
                and outer_final in {str(current_outer), str(legacy_outer)}
            )
        if rel.endswith("reports/interactive_inner_viewer/index.html") or rel.endswith("interactive_rig_viewer_v1/index.html"):
            rig_data_path = path.parent / "rig_data.json"
            if not rig_data_path.is_file():
                return False
            try:
                rig_data = json.loads(rig_data_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            return any(camera.get("image_url") for camera in rig_data.get("cameras", []))
        return True

    def _report_item_ready(self, item):
        rel = item.get("path", "")
        if not rel:
            return False
        path = self.root / rel
        if not path.is_file():
            return False
        if "3D viewer" in item.get("kind", ""):
            return self._viewer_ready(rel, path)
        return True

    def _render_report_item(self, item):
        rel = item.get("path", "")
        ready = self._report_item_ready(item)
        status = item["status_if_exists"] if ready else item["status_if_missing"]
        state_class = " ready" if ready else " missing"
        summary = self._pipeline_summary(rel)
        if ready and (
            summary.get("dry_run")
            or summary.get("mode") == "dry_run"
            or summary.get("args", {}).get("dry_run")
        ):
            status = "preview / dry-run latest"
            state_class = " ready diagnostic"
        if item.get("diagnostic"):
            state_class += " diagnostic"
        label = html.escape(item["label"])
        kind = html.escape(item.get("kind", "report"))
        href = self._path_href(rel)
        path_text = html.escape(href)
        title = f"<a href=\"{html.escape(href)}\">{label}</a>"
        return (
            f"<li class=\"report-item{state_class}\">"
            "<div>"
            f"{title}"
            f"<small>{kind} · {html.escape(status)}</small>"
            "</div>"
            f"<code>{path_text}</code>"
            "</li>"
        )

    def _render_report_group(self, group, panel_url):
        notes = "".join(
            f"<li>{html.escape(note)}</li>" for note in group.get("notes", [])
        )
        items = "".join(self._render_report_item(item) for item in group.get("items", []))
        panel_href = self._panel_mode_href(panel_url, group.get("panel_mode", ""))
        action = (
            "<a class=\"pipeline-action\" "
            f"href=\"{html.escape(panel_href)}\">Run from panel</a>"
        )
        return (
            f"<section class=\"report-group {html.escape(group['status'])}\">"
            "<div class=\"group-head\">"
            "<div>"
            f"<h2>{html.escape(group['title'])}</h2>"
            f"<p class=\"subtitle\">{html.escape(group['subtitle'])}</p>"
            "</div>"
            f"<span class=\"status-pill\">{html.escape(group['status_label'])}</span>"
            "</div>"
            f"<p>{html.escape(group['description'])}</p>"
            f"{action}"
            f"<ul class=\"report-list\">{items}</ul>"
            f"<ul class=\"notes\">{notes}</ul>"
            "</section>"
        )

    def _serve_index(self):
        panel_url = getattr(self.server, "panel_url", DEFAULT_PANEL_URL) or self._same_host_panel_url()
        tool_links = "".join(self._render_tool_link(item, panel_url) for item in TOOL_LINKS)
        report_groups = "".join(self._render_report_group(group, panel_url) for group in REPORT_GROUPS)
        body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Calibration Reports</title>
  <style>
    :root {{
      --ink: #1f2328;
      --muted: #57606a;
      --line: #d0d7de;
      --soft: #f6f8fa;
      --blue: #0969da;
      --green: #1a7f37;
      --amber: #9a6700;
      --violet: #8250df;
      --red: #cf222e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{ border-bottom: 1px solid var(--line); padding: 28px 32px 22px; }}
    main {{ padding: 24px 32px 40px; }}
    h1 {{ font-size: 25px; margin: 0 0 8px; }}
    h2 {{ font-size: 17px; margin: 0; }}
    p {{ color: var(--muted); line-height: 1.45; }}
    .subtitle {{ margin: 5px 0 0; font-size: 13px; color: var(--muted); }}
    .tools {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }}
    .tool-link {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 4px 14px;
      min-width: 280px;
      max-width: 460px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      color: var(--ink);
      text-decoration: none;
    }}
    .tool-link:hover {{ border-color: var(--blue); text-decoration: none; }}
    .tool-link span {{ color: var(--blue); font-size: 12px; text-transform: uppercase; }}
    .tool-link small {{ grid-column: 1 / -1; color: var(--muted); line-height: 1.35; }}
    .report-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      align-items: stretch;
    }}
    .report-group {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-height: 300px;
    }}
    .report-group.available {{ border-top: 4px solid var(--green); }}
    .report-group.pipeline {{ border-top: 4px solid var(--violet); }}
    .report-group.partial {{ border-top: 4px solid var(--amber); }}
    .report-group.missing {{ border-top: 4px solid var(--red); }}
    .report-group.diagnostic {{ border-top: 4px solid var(--amber); }}
    .group-head {{ display: flex; gap: 12px; justify-content: space-between; align-items: flex-start; }}
    .status-pill {{
      flex: 0 0 auto;
      border-radius: 999px;
      padding: 3px 9px;
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .available .status-pill {{ color: var(--green); }}
    .pipeline .status-pill {{ color: var(--violet); }}
    .partial .status-pill, .diagnostic .status-pill {{ color: var(--amber); }}
    .missing .status-pill {{ color: var(--red); }}
    .pipeline-action {{
      display: inline-flex;
      margin-top: 4px;
      border: 1px solid var(--blue);
      border-radius: 6px;
      padding: 7px 10px;
      color: var(--blue);
      font-size: 13px;
      font-weight: 650;
    }}
    .pipeline-action:hover {{ background: #ddf4ff; text-decoration: none; }}
    .report-list, .notes {{ margin: 14px 0 0; padding: 0; list-style: none; }}
    .report-item {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 6px;
      border-top: 1px solid var(--line);
      padding: 10px 0;
    }}
    .report-item:first-child {{ border-top: 0; padding-top: 0; }}
    .report-item small {{ display: block; margin-top: 3px; color: var(--muted); }}
    .report-item.ready small {{ color: var(--green); }}
    .report-item.diagnostic small {{ color: var(--amber); }}
    .report-item.missing strong {{ color: var(--red); }}
    .report-item code {{
      display: block;
      overflow-wrap: anywhere;
      background: var(--soft);
      padding: 5px 6px;
      border-radius: 5px;
      color: var(--muted);
      font-size: 12px;
    }}
    .notes li {{ margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.35; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; }}
    th {{ background: var(--soft); }}
    details {{ margin-top: 26px; }}
    summary {{ cursor: pointer; color: var(--muted); }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    header code, main > p code {{ background: var(--soft); padding: 2px 4px; border-radius: 4px; }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .report-grid {{ grid-template-columns: 1fr; }}
      .tool-link {{ min-width: 0; width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Camera Calibration Reports</h1>
    <p>最终报告入口只展示 canonical reports。所有 report href 使用完整 9899 URL。服务根目录: <code>{html.escape(str(self.root))}</code>。</p>
    <div class="tools">{tool_links}</div>
  </header>
  <main>
    <div class="report-grid">{report_groups}</div>
  </main>
</body>
</html>
"""
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9899)
    parser.add_argument("--public-url", default=DEFAULT_REPORT_BASE_URL)
    parser.add_argument("--panel-url", default=DEFAULT_PANEL_URL)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    def handler(*handler_args, **handler_kwargs):
        return ReportHandler(*handler_args, directory=str(root), **handler_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.report_base_url = args.public_url
    server.panel_url = args.panel_url
    print(f"Serving {root} on http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
