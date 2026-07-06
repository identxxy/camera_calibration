#!/usr/bin/env python3
"""Focused tests for the t0 calibration report HTTP server."""

from pathlib import Path
from types import SimpleNamespace
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
    handler.server = SimpleNamespace(report_base_url="http://reports.example")
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
    def test_root_entry_prefers_current_calibration_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handler = make_handler(root)

            self.assertIsNone(handler._current_entry_path())

            write_text(root / report_server.CURRENT_ENTRY_REL, "<html>current</html>")

            self.assertEqual(root / report_server.CURRENT_ENTRY_REL, handler._current_entry_path())

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
            "1. Inner Capture Report",
            "2. Inner Intrinsic Report",
            "3. Inner Extrinsic Report",
            "4. Outer Capture Report",
            "5. Outer Intrinsic Report",
            "6. Outer Extrinsic Report",
            "7. Bridge Result Report",
        ])


if __name__ == "__main__":
    unittest.main()
