#!/usr/bin/env python3
"""Focused tests for the local calibration panel server."""

from pathlib import Path
import json
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibration_panel_server as panel  # noqa: E402


class CalibrationPanelServerTest(unittest.TestCase):
    def test_whole_operation_form_defaults_match_production_builder(self):
        params = {
            param["name"]: param.get("default")
            for param in panel.MODE_DEFINITIONS["operate_whole_outer_cage"]["params"]
        }

        self.assertFalse(params["run_colmap_vote"])
        self.assertFalse(params["run_side_prior"])
        self.assertFalse(params["run_tag_refine"])
        self.assertTrue(params["run_frame_face_refine"])
        self.assertFalse(params["run_viewer"])
        self.assertTrue(params["run_reports"])

    def test_rejects_unknown_run_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = panel.JobManager(
                repo_root=root,
                runs_root=root / "runs",
                python_bin=sys.executable,
            )
            with self.assertRaises(ValueError):
                manager.start_job("not_a_mode", {}, dry_run=True)

    def test_stage_data_dry_run_writes_job_metadata_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts" / "ops").mkdir(parents=True)
            manager = panel.JobManager(
                repo_root=root,
                runs_root=root / "runs",
                python_bin=sys.executable,
            )
            job = manager.start_job(
                "stage_data",
                {
                    "mount_root": "/mnt/cameras",
                    "output_root": "/tmp/staged_calib",
                    "max_tail_trim": 2,
                },
                dry_run=True,
            )
            manager.wait_for_job(job["id"], timeout=5)

            job_dir = Path(job["run_dir"])
            saved = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            log_text = (job_dir / "run.log").read_text(encoding="utf-8")

            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["mode"], "stage_data")
            self.assertTrue(saved["dry_run"])
            self.assertIn("DRY RUN", log_text)
            self.assertIn("t0_stage_current_calib_data.py", log_text)
            self.assertIn("--max-tail-trim 2", log_text)

    def test_inner_bridge_default_uses_fixed_rig_quality_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts" / "calib").mkdir(parents=True)
            manager = panel.JobManager(
                repo_root=root,
                runs_root=root / "runs",
                python_bin=sys.executable,
            )
            job = manager.start_job(
                "run_inner_bridge_recalib_pipeline",
                {},
                dry_run=True,
            )
            manager.wait_for_job(job["id"], timeout=5)

            job_dir = Path(job["run_dir"])
            saved = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            command = saved["steps"][0]["command"]

            self.assertIn("--inner-refine-mode fixed_rig", command)
            self.assertIn("--inner-joint-max-ba-iterations 3", command)
            self.assertIn("--large-inner-frame-stride 1", command)
            self.assertIn("--run-small-fixed-rig-quality", command)
            self.assertIn("--run-large-bridge", command)
            self.assertNotIn("--run-small-refine", command)

    def test_outer_tower_exposes_intrinsic_refine_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts" / "calib").mkdir(parents=True)
            manager = panel.JobManager(
                repo_root=root,
                runs_root=root / "runs",
                python_bin=sys.executable,
            )
            job = manager.start_job(
                "run_outer_tower_recalib_pipeline",
                {
                    "run_tag_refine": True,
                    "tag_intrinsics_refine_mode": "per_camera_fxfy",
                    "tag_intrinsics_focal_sigma_frac": 0.005,
                    "tag_intrinsics_max_focal_step_frac": 0.001,
                },
                dry_run=True,
            )
            manager.wait_for_job(job["id"], timeout=5)

            job_dir = Path(job["run_dir"])
            saved = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            command = saved["steps"][0]["command"]

            self.assertIn("--run-tag-refine", command)
            self.assertIn("--run-colmap-vote", command)
            self.assertIn("--run-side-prior", command)
            self.assertIn("--run-frame-face-refine", command)
            self.assertIn("--frame-face-refine-preset wide50_then_gate6", command)
            self.assertIn("--colmap-jobs 4", command)
            self.assertIn("--tag-min-camera-observations-for-use 16", command)
            self.assertIn("--tag-min-camera-observations-for-delta 10", command)
            self.assertIn("--tag-post-refine-observation-residual-gate-px 190.0", command)
            self.assertIn("--tag-post-refine-outer-iterations 2", command)
            self.assertIn("--tag-intrinsics-refine-mode per_camera_fxfy", command)
            self.assertIn("--tag-intrinsics-focal-sigma-frac 0.005", command)
            self.assertIn("--tag-intrinsics-max-focal-step-frac 0.001", command)

    def test_studio_pipeline_mode_uses_current_production_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts" / "calib").mkdir(parents=True)
            manager = panel.JobManager(
                repo_root=root,
                runs_root=root / "runs",
                python_bin=sys.executable,
            )
            job = manager.start_job(
                "run_studio_calibration_pipeline",
                {
                    "publish_current": True,
                    "run_tag": "panel_smoke",
                },
                dry_run=True,
            )
            manager.wait_for_job(job["id"], timeout=5)

            command = json.loads(
                (Path(job["run_dir"]) / "job.json").read_text(encoding="utf-8")
            )["steps"][0]["command"]

            self.assertIn("run_studio_calibration_pipeline.py", command)
            self.assertIn("--whole-data-root /home/ubuntu/calib_data/calib_2026_05_31_v3", command)
            self.assertIn("--inner-data-root /home/ubuntu/calib_data/calib_2026_05_31_v3", command)
            self.assertIn("--outer-preset wide50_then_gate6", command)
            self.assertIn("--outer-frame-face-prior-pose-yaml", command)
            self.assertIn("frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1", command)
            self.assertIn("--run-small-quality", command)
            self.assertIn("--publish-current", command)
            self.assertIn("--dry-run", command)

    def test_operation_aliases_build_clean_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts" / "calib").mkdir(parents=True)
            manager = panel.JobManager(
                repo_root=root,
                runs_root=root / "runs",
                python_bin=sys.executable,
            )

            whole = manager.start_job("operate_whole_outer_cage", {}, dry_run=True)
            large = manager.start_job("operate_large_marker_bridge", {}, dry_run=True)
            small = manager.start_job("operate_small_marker_inner", {}, dry_run=True)
            for job in (whole, large, small):
                manager.wait_for_job(job["id"], timeout=5)

            whole_command = json.loads(
                (Path(whole["run_dir"]) / "job.json").read_text(encoding="utf-8")
            )["steps"][0]["command"]
            large_command = json.loads(
                (Path(large["run_dir"]) / "job.json").read_text(encoding="utf-8")
            )["steps"][0]["command"]
            small_command = json.loads(
                (Path(small["run_dir"]) / "job.json").read_text(encoding="utf-8")
            )["steps"][0]["command"]

            self.assertIn("run_outer_tower_recalib_pipeline.py", whole_command)
            self.assertIn("calib_2026_05_31_v3", whole_command)
            self.assertIn("--whole-dir /home/ubuntu/calib_data/calib_2026_05_31_v3/whole_outer24_filtered_min4_hybrid_min4cam", whole_command)
            self.assertIn("--run-frame-face-refine", whole_command)
            self.assertIn("--frame-face-refine-preset wide50_then_gate6", whole_command)
            self.assertIn("frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1", whole_command)
            self.assertNotIn("--run-colmap-vote", whole_command)
            self.assertNotIn("--run-side-prior", whole_command)
            self.assertNotIn("--run-tag-refine", whole_command)

            self.assertIn("run_inner_bridge_recalib_pipeline.py", large_command)
            self.assertIn("--run-large-inner-init", large_command)
            self.assertIn("--run-large-bridge", large_command)
            self.assertNotIn("--run-small-refine", large_command)

            self.assertIn("run_inner_bridge_recalib_pipeline.py", small_command)
            self.assertIn("--inner-refine-mode fixed_then_joint", small_command)
            self.assertIn("--run-small-fixed-rig-quality", small_command)
            self.assertIn("--run-small-refine", small_command)
            self.assertNotIn("--run-large-bridge", small_command)


if __name__ == "__main__":
    unittest.main()
