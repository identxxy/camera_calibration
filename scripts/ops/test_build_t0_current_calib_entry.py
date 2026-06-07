#!/usr/bin/env python3
"""Tests for current calibration entry generation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/ops/build_t0_current_calib_entry.py"


class BuildT0CurrentCalibEntryTest(unittest.TestCase):
    def test_dynamic_current_run_paths_are_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "current_calibration"
            bridge_rel = "runs/studio_latest/inner_bridge"
            outer_rel = "runs/studio_latest/outer_tower/frame_face"
            outer_report_rel = "runs/studio_latest/outer_tower_wrapper"
            studio32_yaml_rel = "runs/studio_latest/calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml"
            whole_data_rel = "runs/studio_latest/whole_filtered/index.html"
            whole_qc_rel = "runs/studio_latest/distributed_qc/index.html"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--root", str(root),
                    "--base-url", "http://t0.example",
                    "--output-dir", str(output_dir),
                    "--current-bridge-run-rel", bridge_rel,
                    "--current-outer-run-rel", outer_rel,
                    "--current-outer-report-rel", outer_report_rel,
                    "--whole-data-report-rel", whole_data_rel,
                    "--whole-distributed-qc-rel", whole_qc_rel,
                    "--studio32-yaml-rel", studio32_yaml_rel,
                    "--write-root-index",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            registry = json.loads((output_dir / "report_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(
                registry["final_viewer"]["canonical_current_url"],
                f"http://t0.example/{bridge_rel}/combined_studio_rig_viewer_v1/index.html",
            )
            self.assertEqual(
                registry["final_viewer"]["canonical_studio32_yaml_url"],
                f"http://t0.example/{studio32_yaml_rel}",
            )
            categories = {group["id"]: group for group in registry["canonical_report_categories"]}
            self.assertEqual(
                list(categories),
                [
                    "inner_capture_qc",
                    "inner_solve_result",
                    "outer_capture_qc",
                    "outer_solve_diagnostics_result",
                    "combined_bridge_32_camera_result",
                ],
            )
            self.assertEqual(
                categories["outer_solve_diagnostics_result"]["items"][0]["url"],
                f"http://t0.example/{outer_report_rel}/index.html",
            )
            self.assertEqual(
                categories["outer_solve_diagnostics_result"]["items"][1]["url"],
                f"http://t0.example/{outer_rel}/summary.json",
            )
            self.assertEqual(
                categories["outer_capture_qc"]["items"][0]["url"],
                f"http://t0.example/{whole_data_rel}",
            )
            self.assertEqual(len(categories["outer_capture_qc"]["items"]), 1)
            self.assertNotIn(whole_qc_rel, json.dumps(categories, ensure_ascii=False))
            self.assertEqual(
                categories["combined_bridge_32_camera_result"]["items"][1]["url"],
                f"http://t0.example/{studio32_yaml_rel}",
            )
            index_html = (output_dir / "index.html").read_text(encoding="utf-8")
            expected_entries = [
                "Overall 3D Viewer",
                "Inner Capture Report",
                "Inner Intrinsic Report",
                "Inner Extrinsic Report",
                "Outer Capture Report",
                "Outer Intrinsic Report",
                "Outer Extrinsic Report",
                "Bridge Result Report",
            ]
            for entry in expected_entries:
                self.assertIn(entry, index_html)
            self.assertIn("Final 32-camera YAML", index_html)
            self.assertIn(studio32_yaml_rel, index_html)
            self.assertIn("one viewer + seven reports", index_html)
            self.assertIn("reports/01_3d_viewer/index.html", index_html)
            self.assertIn("reports/09_bridge_result_large_marker/index.html", index_html)
            self.assertNotIn("report_registry.json", index_html)
            self.assertNotIn("采集后处理入口", index_html)
            self.assertNotIn("mode=run_studio_calibration_pipeline", index_html)
            self.assertNotIn("operations/whole.html", index_html)
            self.assertNotIn("reports/08_bridge_capture_large_marker/index.html", index_html)
            root_index_html = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn("one viewer + seven reports", root_index_html)
            self.assertIn("Final 32-camera YAML", root_index_html)
            self.assertNotIn("report_registry.json", root_index_html)
            operations = {entry["id"]: entry for entry in registry["operation_entries"]}
            self.assertIn("mode=operate_whole_outer_cage", operations["whole"]["operation"]["panel_url"])
            self.assertIn("mode=operate_large_marker_bridge", operations["large_marker"]["operation"]["panel_url"])
            self.assertIn("mode=operate_small_marker_inner", operations["small_marker"]["operation"]["panel_url"])


if __name__ == "__main__":
    unittest.main()
