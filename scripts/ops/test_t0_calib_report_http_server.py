#!/usr/bin/env python3
"""Focused tests for the t0 calibration report HTTP server."""

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import t0_calib_report_http_server as report_server  # noqa: E402


def make_handler(root):
    handler = report_server.ReportHandler.__new__(report_server.ReportHandler)
    handler.root = Path(root)
    handler.server = SimpleNamespace(
        report_base_url="http://reports.example",
        runs_root="/tmp/panel_runs",
    )
    return handler


def write_text(path, text="ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_server.json.dumps(payload), encoding="utf-8")


def primary_group_items():
    return [group["items"][0] for group in report_server.REPORT_GROUPS]


def find_report_item(label=None, path_suffix=None):
    for group in report_server.REPORT_GROUPS:
        for item in group["items"]:
            if label is not None and item.get("label") == label:
                return item
            if path_suffix is not None and item.get("path", "").endswith(path_suffix):
                return item
    raise AssertionError(f"Could not find report item label={label!r} path_suffix={path_suffix!r}")


class T0CalibReportHttpServerTest(unittest.TestCase):
    def test_current_entry_path_remains_addressable_but_not_homepage_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler = make_handler(root)

            self.assertIsNone(handler._current_entry_path())

            write_text(root / report_server.CURRENT_ENTRY_REL, "<html>current</html>")

            self.assertEqual(root / report_server.CURRENT_ENTRY_REL, handler._current_entry_path())

    def test_console_contract_has_three_quick_actions_and_dependency_steps(self):
        self.assertEqual(
            [action["slug"] for action in report_server.QUICK_ACTIONS],
            ["full-dry-run", "full-run", "fast-bridge"],
        )
        self.assertEqual(
            [dataset["slug"] for dataset in report_server.CAPTURE_DATASETS],
            ["large", "tower", "bridge", "small"],
        )
        self.assertEqual(
            {
                dataset["slug"]: dataset["qc_step_slug"]
                for dataset in report_server.CAPTURE_DATASETS
            },
            {
                "large": "outer-large-marker",
                "tower": "whole-outer-cage",
                "bridge": "large-marker-bridge",
                "small": "small-marker-inner",
            },
        )
        self.assertEqual(
            [step["slug"] for step in report_server.WORKFLOW_STEPS],
            [
                "outer-large-marker",
                "whole-outer-cage",
                "small-marker-inner",
                "inner-rig-extrinsics",
                "large-marker-bridge",
                "publish-current",
            ],
        )
        self.assertEqual(
            {
                step["slug"]: step["required_capture_slugs"]
                for step in report_server.WORKFLOW_STEPS
            },
            {
                "outer-large-marker": ["large"],
                "whole-outer-cage": ["tower", "bridge"],
                "small-marker-inner": ["small"],
                "inner-rig-extrinsics": ["bridge", "small"],
                "large-marker-bridge": ["bridge"],
                "publish-current": ["large", "tower", "bridge", "small"],
            },
        )
        self.assertEqual(
            report_server.WORKFLOW_GRAPH_EDGES,
            [
                ("outer-large-marker", "whole-outer-cage", "outer K"),
                ("small-marker-inner", "inner-rig-extrinsics", "inner K"),
                ("whole-outer-cage", "large-marker-bridge", "outer pose"),
                ("inner-rig-extrinsics", "large-marker-bridge", "inner pose"),
                ("large-marker-bridge", "publish-current", "final all32"),
            ],
        )
        for step in report_server.WORKFLOW_STEPS:
            self.assertIn("capture_date", step)
            self.assertIn("result_reports", step)
            if step["slug"] != "publish-current":
                self.assertIn("capture_reports", step)
                self.assertTrue(step["capture_reports"], step["slug"])
            for section in ("capture_reports", "result_reports"):
                for item in step.get(section, []):
                    self.assertNotRegex(item["label"], r"^\\d+[_\\.]")

    def test_workflow_graph_shows_capture_and_process_dates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler = make_handler(root)
            write_text(root / report_server.OUTER_INTRINSIC_REPORT, "<html>outer k</html>")

            graph = handler._render_workflow_graph()
            expected_process_date = report_server.datetime.fromtimestamp(
                (root / report_server.OUTER_INTRINSIC_REPORT).stat().st_mtime
            ).strftime("%Y-%m-%d")

            self.assertIn("capture: 2026-06-04", graph)
            self.assertIn(f"process: {expected_process_date}", graph)
            self.assertIn("capture: mixed inputs", graph)

    def test_console_language_defaults_to_english_with_toggle(self):
        handler = make_handler("/tmp")
        script = handler._page_script()
        html_page_source = inspect.getsource(report_server.ReportHandler._html_page)

        self.assertEqual(report_server.DEFAULT_LANGUAGE, "en")
        self.assertIn('let lang = "en";', script)
        self.assertIn('LANGUAGE_STORAGE_KEY = "calibConsoleLanguageV2"', script)
        self.assertIn('localStorage.getItem(LANGUAGE_STORAGE_KEY) || "en"', script)
        self.assertIn('<html lang="en">', html_page_source)
        self.assertIn('<body class="lang-en">', html_page_source)
        self.assertIn('data-lang="zh"', handler._language_toggle())
        self.assertIn('data-lang="en"', handler._language_toggle())

    def test_9899_console_reuses_panel_whitelist_modes(self):
        modes = report_server.MODE_DEFINITIONS

        self.assertIn("run_studio_calibration_pipeline", modes)
        self.assertIn("operate_whole_outer_cage", modes)
        self.assertIn("operate_large_marker_bridge", modes)
        self.assertIn("operate_small_marker_inner", modes)

    def test_full_run_button_publishes_current_and_disables_pipeline_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = make_handler(tmpdir)
            action = report_server.QUICK_ACTION_BY_SLUG["full-run"]

            payload = handler._action_payload(action)

            self.assertEqual(payload["mode"], "run_studio_calibration_pipeline")
            self.assertFalse(payload["dry_run"])
            self.assertTrue(payload["params"]["publish_current"])
            self.assertFalse(payload["params"]["pipeline_dry_run"])

            dry_run_card = handler._render_quick_action(
                report_server.QUICK_ACTION_BY_SLUG["full-dry-run"]
            )
            self.assertIn("quick-action debug-control", dry_run_card)

    def test_data_collect_overview_and_detail_explain_four_capture_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = make_handler(tmpdir)

            overview = handler._render_data_collect_overview()
            self.assertIn("Data Collect: 4 Required Captures", overview)
            for slug in ("large", "tower", "bridge", "small"):
                self.assertIn(f"/data-collect/{slug}", overview)

            bridge_body = handler._data_collect_detail_body(
                report_server.CAPTURE_DATASET_BY_SLUG["bridge"]
            )
            self.assertIn("Run Data QC / Aggregate", bridge_body)
            self.assertIn("Data Capture / QC Report", bridge_body)
            self.assertIn("Acceptance Criteria", bridge_body)
            self.assertIn("must cover both inner and outer cameras", bridge_body)
            self.assertIn("large_marker_bridge_all32", bridge_body)
            self.assertIn("/operation/inner-rig-extrinsics", bridge_body)
            self.assertIn("/operation/large-marker-bridge", bridge_body)

    def test_readiness_overview_reports_capture_and_yaml_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler = make_handler(root)
            write_text(root / report_server.INNER_CAPTURE_REPORT, "<html>inner qc</html>")
            write_text(root / report_server.FINAL_STUDIO32_YAML, "cameras: []\n")

            overview = handler._render_readiness_overview()

            self.assertIn("Readiness: Current Data and Result Status", overview)
            self.assertIn("/data-collect/small", overview)
            self.assertIn("Final YAML", overview)
            self.assertIn("Ready", overview)
            self.assertIn("Missing", overview)
            self.assertIn("for reconstruction / SLAM / 3DGS", overview)

    def test_operation_detail_sections_include_capture_dependencies_then_run_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = make_handler(tmpdir)
            step = report_server.WORKFLOW_BY_SLUG["whole-outer-cage"]
            body = handler._operation_detail_body(step)

            order = [
                body.index("Required Data Collect"),
                body.index("/data-collect/tower"),
                body.index("/data-collect/bridge"),
                body.index("Data Paths"),
                body.index("Output Paths"),
                body.index("Run This Step"),
                body.index("Calibration Result Report"),
                body.index("Advanced Run Parameters"),
            ]

            self.assertEqual(order, sorted(order))
            self.assertNotIn("Run Capture QC / Aggregate", body)
            self.assertNotIn("Capture QC / Aggregate Report", body)
            self.assertIn("Capture QC now lives on the Data Collect detail pages", body)
            self.assertIn('data-confirm="true"', body)
            self.assertIn("Dry-run This Step", body)
            self.assertIn("debug-control", body)
            self.assertIn("高级运行参数", body)
            self.assertNotIn("shell command", body)
            self.assertNotIn("白名单", body)

    def test_primary_curated_entries_render_ready_with_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler = make_handler(root)
            for item in primary_group_items():
                write_text(root / item["path"], "<html>real report</html>")

            primary_paths = {item["path"] for item in primary_group_items()}
            self.assertTrue(primary_paths.issubset(handler._curated_paths()))

            write_text(
                root / report_server.UNIFIED_VIEWER.replace("index.html", "rig_data.json"),
                report_server.json.dumps({
                    "cameras": [{"label": "1-1"}],
                    "inputs": {
                        "outer_final_pose_yaml": str(
                            root
                            / report_server.OUTER_TOWER_LATEST
                            / "camera_tr_rig_delta_refined.yaml"
                        ),
                    },
                    "metrics": {"outer_pose_source": "outer_final_pose_yaml_bridge_aligned"},
                }),
            )

            for item in primary_group_items():
                rendered = handler._render_report_item(item)
                self.assertIn("report-item ready", rendered)
                self.assertIn("ready", rendered)

    def test_combined_viewer_missing_rig_data_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler = make_handler(root)
            item = find_report_item(path_suffix="01_3d_viewer/index.html")
            write_text(root / item["path"], "<html>combined viewer</html>")

            rendered = handler._render_report_item(item)

            self.assertIn("report-item missing", rendered)
            self.assertIn("not produced yet", rendered)

    def test_combined_viewer_colmap_fallback_rig_data_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler = make_handler(root)
            item = find_report_item(path_suffix="01_3d_viewer/index.html")
            write_text(root / item["path"], "<html>combined viewer</html>")
            write_json(
                root / item["path"].replace("index.html", "rig_data.json"),
                {
                    "cameras": [{"label": "1-1"}],
                    "inputs": {"outer_final_pose_yaml": ""},
                    "metrics": {"outer_pose_source": "colmap_sim3_approx"},
                },
            )

            rendered = handler._render_report_item(item)

            self.assertIn("report-item missing", rendered)
            self.assertIn("not produced yet", rendered)

    def test_combined_viewer_outer_final_rig_data_is_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler = make_handler(root)
            item = find_report_item(path_suffix="01_3d_viewer/index.html")
            write_text(root / item["path"], "<html>combined viewer</html>")
            write_json(
                root / item["path"].replace("index.html", "rig_data.json"),
                {
                    "cameras": [{"label": "1-1"}],
                    "inputs": {
                        "outer_final_pose_yaml": str(
                            root
                            / report_server.OUTER_TOWER_LATEST
                            / "camera_tr_rig_delta_refined.yaml"
                        ),
                    },
                    "metrics": {"outer_pose_source": "outer_final_pose_yaml_bridge_aligned"},
                },
            )

            rendered = handler._render_report_item(item)

            self.assertIn('report-item ready', rendered)
            self.assertIn("ready", rendered)

    def test_fallback_groups_do_not_surface_legacy_inventory(self):
        labels = [
            item["label"]
            for group in report_server.REPORT_GROUPS
            for item in group["items"]
        ]

        self.assertNotIn("Report inventory / cleanup audit", labels)
        self.assertNotIn("Stable 2026-05-26 outer tower viewer", labels)
        self.assertNotIn("Inner solve 3D viewer", labels)
        self.assertNotIn("Bridge summary.json", labels)
        self.assertNotIn("Outer solve summary.json", labels)

    def test_fallback_groups_are_final_yaml_one_viewer_and_seven_reports(self):
        titles = [group["title"] for group in report_server.REPORT_GROUPS]

        self.assertEqual(len(titles), 9)
        self.assertEqual(titles[0], "Final Calibration Artifact")
        self.assertEqual(titles[1], "Overall Viewer")
        self.assertEqual(titles[2:], [
            "Inner Capture / QC",
            "Inner Intrinsic Result",
            "Inner Extrinsic Result",
            "Outer Capture / QC",
            "Outer Intrinsic Result",
            "Outer Extrinsic Result",
            "Bridge Result",
        ])


if __name__ == "__main__":
    unittest.main()
