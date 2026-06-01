#!/usr/bin/env python3
"""Tests for the one-command studio calibration pipeline wrapper."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/run_studio_calibration_pipeline.py"


class StudioCalibrationPipelineTest(unittest.TestCase):
    def test_dry_run_plans_outer_bridge_and_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whole_root = root / "whole_root"
            inner_root = root / "inner_root"
            output_root = root / "studio_run"
            inner_prior = root / "priors" / "inner_state"
            outer_prior = root / "priors" / "outer_images.txt"
            outer_delta_prior = root / "priors" / "outer_delta.yaml"
            outer_intrinsics_prior = root / "priors" / "outer_intrinsics"
            whole_data_report = whole_root / "whole_outer24_filtered_min4_hybrid_min4cam" / "index.html"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--whole-data-root", str(whole_root),
                    "--inner-data-root", str(inner_root),
                    "--output-root", str(output_root),
                    "--run-tag", "test_run",
                    "--inner-prior", str(inner_prior),
                    "--outer-prior", str(outer_prior),
                    "--outer-frame-face-prior-pose-yaml", str(outer_delta_prior),
                    "--outer-frame-face-intrinsics-dir", str(outer_intrinsics_prior),
                    "--whole-data-report", str(whole_data_report),
                    "--dry-run",
                    "--publish-current",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}

            self.assertEqual(summary["mode"], "dry_run")
            self.assertEqual(summary["run_timing"]["stage_count"], 4)
            self.assertIn("outer_tower", summary["run_timing"]["stage_durations_s"])
            self.assertTrue(stages["outer_tower"]["requested"])
            self.assertTrue(stages["inner_bridge"]["requested"])
            self.assertTrue(stages["export_unified_cameras"]["requested"])
            self.assertTrue(stages["publish_current"]["requested"])

            outer_pose = (
                output_root
                / "outer_tower"
                / "frame_face_refine_wide50_then_gate6"
                / "camera_tr_rig_delta_refined.yaml"
            )
            outer_intrinsics = (
                output_root
                / "outer_tower"
                / "frame_face_refine_wide50_then_gate6"
                / "intrinsics_refined"
            )
            outer_command = stages["outer_tower"]["commands"][0]
            self.assertIn("--frame-face-refine-preset wide50_then_gate6", outer_command)
            self.assertIn(f"--frame-face-prior-pose-yaml {outer_delta_prior}", outer_command)
            self.assertIn(f"--frame-face-intrinsics-dir {outer_intrinsics_prior}", outer_command)

            bridge_command = stages["inner_bridge"]["commands"][0]
            self.assertIn(f"--outer-final-pose-yaml {outer_pose}", bridge_command)
            self.assertIn(f"--outer-intrinsics {outer_intrinsics}", bridge_command)
            self.assertIn(f"--inner-prior {inner_prior}", bridge_command)
            self.assertIn(f"--outer-prior {outer_prior}", bridge_command)
            self.assertIn("--run-large-bridge", bridge_command)
            self.assertIn("--run-reports", bridge_command)

            export_command = stages["export_unified_cameras"]["commands"][0]
            unified_yaml = output_root / "calibration_artifacts" / "studio_32_cameras_current" / "studio_32_cameras.yaml"
            self.assertIn("export_combined_studio_extrinsics.py", export_command)
            self.assertIn(f"--outer-final-pose-yaml {outer_pose}", export_command)
            self.assertIn("--intrinsics-dir", export_command)
            self.assertEqual(summary["outputs"]["unified_camera_yaml"], str(unified_yaml))
            self.assertEqual(summary["report_urls"]["unified_camera_yaml"], unified_yaml.as_uri())

            publish_command = stages["publish_current"]["commands"][0]
            self.assertIn("--current-bridge-run-rel", publish_command)
            self.assertIn("--current-outer-run-rel", publish_command)
            self.assertIn("frame_face_refine_wide50_then_gate6", publish_command)
            self.assertIn("--whole-data-report-rel", publish_command)
            self.assertEqual(summary["run_tag"], "test_run")
            self.assertIn(str(output_root / "index.html"), completed.stdout)

    def test_outer_only_dry_run_skips_bridge_and_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "studio_run"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--whole-data-root", str(root / "whole_root"),
                    "--inner-data-root", str(root / "inner_root"),
                    "--output-root", str(output_root),
                    "--outer-only",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            self.assertTrue(stages["outer_tower"]["requested"])
            self.assertFalse(stages["inner_bridge"]["requested"])
            self.assertFalse(stages["export_unified_cameras"]["requested"])
            self.assertFalse(stages["publish_current"]["requested"])

    def test_default_current_priors_are_planned(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "studio_run"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output-root", str(output_root),
                    "--run-tag", "default_priors",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            outer_command = stages["outer_tower"]["commands"][0]
            bridge_command = stages["inner_bridge"]["commands"][0]

            self.assertIn("calib_2026_05_31_v3", outer_command)
            self.assertIn("frame_face_refine_wide50_then_gate6", outer_command)
            self.assertIn("recalib_20260531_193215_v2_outer_wide50", outer_command)
            self.assertIn("final_inner8_calibration_v1", bridge_command)
            self.assertIn("colmap_outer24_firstframe_colmap404_v3", bridge_command)


if __name__ == "__main__":
    unittest.main()
