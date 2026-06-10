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
            outer_large_opencv_intrinsics = root / "priors" / "outer_large_opencv_intrinsics"
            outer_large_qc_root = root / "outer_large_qc"
            whole_qc_root = whole_root / "whole_outer24_filtered_min4_fullres_min4cam"
            outer_intrinsic_metrics = root / "priors" / "outer_intrinsic_metrics.tsv"
            whole_data_report = whole_root / "whole_outer24_filtered_min4_hybrid_min4cam" / "index.html"
            outer_intrinsic_metrics.parent.mkdir(parents=True)
            outer_intrinsic_metrics.write_text("camera_index\tuser_id\tresidual_count\n", encoding="utf-8")
            (whole_qc_root).mkdir(parents=True)
            (whole_qc_root / "per_camera_stats.tsv").write_text("camera_id\tpassing_images\n", encoding="utf-8")
            (whole_qc_root / "opencv_tower_dataset_black_tile_red_scale_edge.bin").write_bytes(b"calib_data")
            (whole_qc_root / "manifest.tsv").write_text("camera_index\tcamera_id\n0\t1-1\n", encoding="utf-8")

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
                    "--outer-large-opencv-intrinsics-dir", str(outer_large_opencv_intrinsics),
                    "--outer-large-qc-root", str(outer_large_qc_root),
                    "--whole-qc-root", str(whole_qc_root),
                    "--outer-intrinsic-metrics-tsv", str(outer_intrinsic_metrics),
                    "--whole-data-report", str(whole_data_report),
                    "--run-small-quality",
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
            self.assertEqual(summary["run_timing"]["stage_count"], 8)
            self.assertIn("outer_tower", summary["run_timing"]["stage_durations_s"])
            self.assertTrue(stages["outer_tower"]["requested"])
            self.assertTrue(stages["generate_outer_intrinsic_report"]["requested"])
            self.assertTrue(stages["inner_bridge"]["requested"])
            self.assertTrue(stages["export_unified_cameras"]["requested"])
            self.assertTrue(stages["export_large_marker_correspondences"]["requested"])
            self.assertTrue(stages["export_small_marker_correspondences"]["requested"])
            self.assertTrue(stages["generate_advanced_correspondence_viewer"]["requested"])
            self.assertTrue(stages["publish_current"]["requested"])

            outer_pose = (
                output_root
                / "outer_tower"
                / "frame_face_refine_wide200_then_gate6"
                / "camera_tr_rig_delta_refined.yaml"
            )
            outer_intrinsics = (
                output_root
                / "outer_tower"
                / "frame_face_refine_wide200_then_gate6"
                / "intrinsics_refined"
            )
            outer_command = stages["outer_tower"]["commands"][0]
            self.assertIn("--frame-face-refine-preset wide200_then_gate6", outer_command)
            self.assertIn(f"--frame-face-prior-pose-yaml {outer_delta_prior}", outer_command)
            self.assertIn(f"--frame-face-intrinsics-dir {outer_intrinsics_prior}", outer_command)

            outer_intrinsic_report = output_root / "reports" / "outer_intrinsics_outer_large_marker"
            outer_intrinsic_command = stages["generate_outer_intrinsic_report"]["commands"][0]
            self.assertIn("generate_opencv_intrinsics_report.py", outer_intrinsic_command)
            self.assertIn(f"--intrinsics-dir {outer_large_opencv_intrinsics}", outer_intrinsic_command)
            self.assertIn(f"--output-dir {outer_intrinsic_report}", outer_intrinsic_command)

            bridge_command = stages["inner_bridge"]["commands"][0]
            self.assertIn(f"--outer-final-pose-yaml {outer_pose}", bridge_command)
            self.assertIn(f"--outer-intrinsics {outer_intrinsics}", bridge_command)
            self.assertIn(f"--outer-intrinsic-metrics-tsv {outer_intrinsic_metrics}", bridge_command)
            self.assertIn(f"--whole-coverage-tsv {whole_qc_root / 'per_camera_stats.tsv'}", bridge_command)
            self.assertIn(f"--inner-prior {inner_prior}", bridge_command)
            self.assertIn(f"--outer-prior {outer_prior}", bridge_command)
            self.assertIn("--run-large-bridge", bridge_command)
            self.assertIn("--run-small-fixed-rig-quality", bridge_command)
            self.assertIn("--run-reports", bridge_command)

            export_command = stages["export_unified_cameras"]["commands"][0]
            unified_yaml = output_root / "calibration_artifacts" / "studio_32_cameras_current" / "studio_32_cameras.yaml"
            large_ba_state = (
                output_root
                / "inner_bridge"
                / "large_marker_bridge_all32"
                / "fixed_points_joint_ba_stride1_dense_v1"
            )
            self.assertIn("export_combined_studio_extrinsics.py", export_command)
            self.assertIn(f"--outer-final-pose-yaml {outer_pose}", export_command)
            self.assertIn(f"--inner-bridge-pose-yaml {large_ba_state / 'camera_tr_rig.yaml'}", export_command)
            self.assertIn("--intrinsics-dir", export_command)
            self.assertEqual(summary["outputs"]["unified_camera_yaml"], str(unified_yaml))
            self.assertEqual(summary["outputs"]["large_marker_state_dir"], str(large_ba_state))
            self.assertEqual(summary["report_urls"]["unified_camera_yaml"], unified_yaml.as_uri())

            large_corr_command = stages["export_large_marker_correspondences"]["commands"][0]
            self.assertIn("export_calibration_correspondence_residuals.py", large_corr_command)
            self.assertIn("--dataset-name large", large_corr_command)
            self.assertIn(f"--state-dir {large_ba_state}", large_corr_command)
            self.assertIn("--camera-index-offset 0", large_corr_command)
            self.assertIn(f"--reference-studio32-yaml {unified_yaml}", large_corr_command)

            small_corr_command = stages["export_small_marker_correspondences"]["commands"][0]
            self.assertIn("export_calibration_correspondence_residuals.py", small_corr_command)
            self.assertIn("--dataset-name small", small_corr_command)
            self.assertIn("--camera-index-offset 24", small_corr_command)

            advanced_command = stages["generate_advanced_correspondence_viewer"]["commands"][0]
            self.assertIn("generate_studio_correspondence_viewer.py", advanced_command)
            self.assertIn(f"--studio32-yaml {unified_yaml}", advanced_command)
            self.assertIn("--outer-observation-residuals-tsv", advanced_command)
            self.assertIn(
                f"--outer-raw-dataset {whole_qc_root / 'opencv_tower_dataset_black_tile_red_scale_edge.bin'}",
                advanced_command,
            )
            self.assertIn(f"--outer-raw-manifest {whole_qc_root / 'manifest.tsv'}", advanced_command)
            self.assertIn("--large-correspondence-tsv", advanced_command)
            self.assertIn("--small-correspondence-tsv", advanced_command)
            self.assertIn("--large-pnp-dir", advanced_command)
            self.assertEqual(
                summary["outputs"]["advanced_correspondence_viewer"],
                str(output_root / "advanced_correspondence_viewer_v1" / "index.html"),
            )

            publish_command = stages["publish_current"]["commands"][0]
            self.assertIn("publish_t0_clean_calib_reports.py", publish_command)
            self.assertIn(f"--run-tag test_run", publish_command)
            self.assertIn(f"--outer-large-intrinsic-report {outer_intrinsic_report}", publish_command)
            self.assertIn(f"--outer-large-qc-root {outer_large_qc_root}", publish_command)
            self.assertIn(f"--whole-qc-root {whole_qc_root}", publish_command)
            self.assertIn(f"--outer-frame-face-report-root {outer_pose.parent}", publish_command)
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
            self.assertTrue(stages["generate_outer_intrinsic_report"]["requested"])
            self.assertFalse(stages["inner_bridge"]["requested"])
            self.assertFalse(stages["export_unified_cameras"]["requested"])
            self.assertFalse(stages["export_large_marker_correspondences"]["requested"])
            self.assertFalse(stages["export_small_marker_correspondences"]["requested"])
            self.assertFalse(stages["generate_advanced_correspondence_viewer"]["requested"])
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
            outer_intrinsic_command = stages["generate_outer_intrinsic_report"]["commands"][0]
            bridge_command = stages["inner_bridge"]["commands"][0]

            self.assertIn("calib_2026_05_31_fullres_probe_v1", outer_command)
            self.assertIn("frame_face_refine_wide200_then_gate6", outer_command)
            self.assertIn("recalib_20260608_rigid_yaw45_v2", outer_command)
            self.assertIn("calib_2026_06_04_outer_large_marker_v2", outer_intrinsic_command)
            self.assertIn("final_inner8_calibration_v1", bridge_command)
            self.assertIn("colmap_outer24_firstframe_colmap404_v3", bridge_command)
            self.assertNotIn("--run-small-fixed-rig-quality", bridge_command)

    def test_nondefault_large_frame_stride_updates_large_pnp_diagnostic_dir(self):
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
                    "--large-frame-stride", "3",
                    "--dry-run",
                    "--bridge-only",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            advanced_command = stages["generate_advanced_correspondence_viewer"]["commands"][0]
            expected_pnp_dir = (
                output_root
                / "inner_bridge"
                / "large_marker_bridge_all32"
                / "fixed_intrinsic_bridge_pnp_stride3_v1"
            )
            expected_ba_dir = (
                output_root
                / "inner_bridge"
                / "large_marker_bridge_all32"
                / "fixed_points_joint_ba_stride3_dense_v1"
            )

            self.assertEqual(summary["outputs"]["large_pnp_dir"], str(expected_pnp_dir))
            self.assertEqual(summary["outputs"]["large_marker_state_dir"], str(expected_ba_dir))
            self.assertIn(f"--large-pnp-dir {expected_pnp_dir}", advanced_command)
            self.assertNotIn("fixed_intrinsic_bridge_pnp_stride1_v1", advanced_command)


if __name__ == "__main__":
    unittest.main()
