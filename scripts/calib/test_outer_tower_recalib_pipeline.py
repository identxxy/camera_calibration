#!/usr/bin/env python3
"""Focused tests for the outer tower recalibration wrapper."""

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/run_outer_tower_recalib_pipeline.py"
SCRIPT_DIR = SCRIPT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_outer_tower_recalib_pipeline as outer_pipeline  # noqa: E402


class OuterTowerRecalibPipelineTest(unittest.TestCase):
    def test_frame_face_default_uses_fullres_raw_wide_then_strict_preset_and_safe_geometry_prior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_2026_05_31_v3"
            output_root = root / "out"
            whole_dir = data_root / "whole_outer_tower"
            frame_root = (
                data_root
                / "whole_outer24_filtered_min4_hybrid_min4cam"
                / "pnp_inlier_filter_facewidth025_optwidth_v1"
            )
            selected = frame_root / "selected_outer_frame_face_current"
            safe_geometry = frame_root / "tag_refine_safe5coeff_percam_fxfycxcy_optwidth_v1"
            weakk = frame_root / "frame_face_planes_all616_weakK_then_fixed_gate20_v1"
            dataset = whole_dir / "opencv_tower_dataset_fullres.bin"
            manifest = whole_dir / "manifest.tsv"
            selected_prior = selected / "camera_tr_rig_delta_refined.yaml"
            prior = safe_geometry / "camera_tr_rig_prior.yaml"
            intrinsics = weakk / "intrinsics_refined"
            dataset.parent.mkdir(parents=True)
            intrinsics.mkdir(parents=True)
            (selected / "intrinsics_refined").mkdir(parents=True)
            weakk.mkdir(parents=True, exist_ok=True)
            dataset.write_bytes(b"")
            manifest.write_text("camera_index\tcamera_id\n", encoding="utf-8")
            safe_geometry.mkdir(parents=True, exist_ok=True)
            selected_prior.write_text("poses: []\n", encoding="utf-8")
            prior.write_text("poses: []\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-frame-face-refine",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            command = stages["frame_face_refine"]["commands"][0]

            self.assertIn(f"--camera_prior_pose_yaml {prior}", command)
            self.assertIn(f"--dataset {dataset}", command)
            self.assertIn(f"--manifest {manifest}", command)
            self.assertIn(f"--intrinsics_dir {intrinsics}", command)
            self.assertIn("--initial_observation_residual_gate_px 50.0", command)
            self.assertIn("--observation_residual_gate_px 6.0", command)
            self.assertIn("--pnp_ransac_iterations 1000", command)
            self.assertIn("--max_pnp_median_error_px 5.0", command)
            self.assertIn("--min_frame_face_observations 8", command)
            self.assertIn("--min_camera_observations_for_delta 8", command)
            self.assertIn("--outer_iterations 12", command)
            self.assertIn("frame_face_refine_wide50_then_gate6", command)
            self.assertEqual(summary["inputs"]["frame_face_prior_pose_yaml"]["path"], str(prior.resolve()))
            self.assertEqual(summary["inputs"]["frame_face_intrinsics_dir"]["path"], str(intrinsics.resolve()))

    def test_frame_face_default_discovers_fullres_filtered_stage_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_2026_05_31_fullres_probe_v1"
            output_root = root / "out"
            stage_root = data_root / "whole_outer24_filtered_min4_fullres_min4cam"
            dataset = stage_root / "opencv_tower_dataset_fullres.bin"
            manifest = stage_root / "manifest.tsv"
            prior = root / "prior.yaml"
            intrinsics = root / "intrinsics"
            stage_root.mkdir(parents=True)
            intrinsics.mkdir()
            dataset.write_bytes(b"")
            manifest.write_text("camera_index\tcamera_id\n", encoding="utf-8")
            prior.write_text("poses: []\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-frame-face-refine",
                    "--frame-face-prior-pose-yaml", str(prior),
                    "--frame-face-intrinsics-dir", str(intrinsics),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            command = stages["frame_face_refine"]["commands"][0]

            self.assertIn(f"--dataset {dataset}", command)
            self.assertIn(f"--manifest {manifest}", command)
            self.assertEqual(summary["inputs"]["frame_face_dataset"]["path"], str(dataset.resolve()))
            self.assertEqual(summary["inputs"]["frame_face_manifest"]["path"], str(manifest.resolve()))

    def test_frame_face_wide50_gate4_refine_stage_is_planned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            data_root.mkdir()
            dataset = root / "tower_fullres.bin"
            manifest = root / "manifest.tsv"
            prior = root / "camera_tr_rig.yaml"
            intrinsics = root / "intrinsics"
            dataset.write_bytes(b"")
            manifest.write_text("camera_index\tcamera_id\n", encoding="utf-8")
            prior.write_text("poses: []\n", encoding="utf-8")
            intrinsics.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-frame-face-refine",
                    "--frame-face-refine-preset", "wide50_then_gate4",
                    "--frame-face-dataset", str(dataset),
                    "--frame-face-manifest", str(manifest),
                    "--frame-face-prior-pose-yaml", str(prior),
                    "--frame-face-intrinsics-dir", str(intrinsics),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            command = stages["frame_face_refine"]["commands"][0]

            self.assertTrue(stages["frame_face_refine"]["requested"])
            self.assertIn("refine_outer_tower_frame_face_planes.py", command)
            self.assertIn("--initial_observation_residual_gate_px 50.0", command)
            self.assertIn("--observation_residual_gate_px 4.0", command)
            self.assertIn("--optimizer_residual_clip_px 20.0", command)
            self.assertIn("--pnp_ransac_iterations 1000", command)
            self.assertIn("--min_camera_observations_for_delta 8", command)
            self.assertIn("frame_face_refine_wide50_then_gate4", command)

    def test_frame_face_gate10_refine_stage_is_planned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            data_root.mkdir()
            dataset = root / "tower_pnp_inliers.bin"
            manifest = root / "manifest.tsv"
            prior = root / "camera_tr_rig.yaml"
            intrinsics = root / "intrinsics"
            dataset.write_bytes(b"")
            manifest.write_text("camera_index\tcamera_id\n", encoding="utf-8")
            prior.write_text("poses: []\n", encoding="utf-8")
            intrinsics.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-frame-face-refine",
                    "--frame-face-refine-preset", "gate10",
                    "--frame-face-dataset", str(dataset),
                    "--frame-face-manifest", str(manifest),
                    "--frame-face-prior-pose-yaml", str(prior),
                    "--frame-face-intrinsics-dir", str(intrinsics),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            command = stages["frame_face_refine"]["commands"][0]

            self.assertTrue(stages["frame_face_refine"]["requested"])
            self.assertIn("refine_outer_tower_frame_face_planes.py", command)
            self.assertIn("--observation_residual_gate_px 10.0", command)
            self.assertIn("--pnp_ransac_threshold_px 4.0", command)
            self.assertIn("--max_pnp_median_error_px 4.0", command)
            self.assertIn("--min_frame_face_observations 12", command)
            self.assertEqual(summary["final"]["source"], "frame_face_refine_expected")
            self.assertEqual(summary["frame_face_refine"]["status"], "missing")

            coverage_stage = stages["intrinsic_feature_coverage_report"]
            coverage_command = coverage_stage["commands"][0]
            self.assertTrue(coverage_stage["requested"])
            self.assertIn("generate_intrinsic_feature_coverage_report.py", coverage_command)
            self.assertIn(
                f"--residuals-tsv {output_root / 'frame_face_refine_gate10/diagnostics/observation_residuals.tsv'}",
                coverage_command,
            )
            self.assertIn(
                f"--intrinsics-dir {output_root / 'frame_face_refine_gate10/intrinsics_refined'}",
                coverage_command,
            )
            self.assertEqual(summary["intrinsic_feature_coverage"]["status"], "missing")
            self.assertIn(
                "intrinsic_feature_coverage_report/index.html",
                summary["final"]["intrinsic_feature_coverage_index"],
            )
            final_html = (output_root / "final_report/index.html").read_text(encoding="utf-8")
            self.assertIn("Intrinsic feature coverage report", final_html)

    def test_tag_intrinsics_refine_mode_is_planned_for_tag_refine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            data_root.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-tag-refine",
                    "--tag-intrinsics-refine-mode", "shared_fxfy",
                    "--tag-intrinsics-focal-sigma-frac", "0.005",
                    "--tag-accept-max-intrinsic-focal-delta-frac", "0.01",
                    "--tag-optimize-tower-face-width",
                    "--tag-tower-face-width-initial-m", "0.25",
                    "--tag-tower-face-width-sigma-m", "0.02",
                    "--force",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            command = stages["tag_refine_robust"]["commands"][0]
            consensus_command = stages["pnp_pose_consensus"]["commands"][0]

            self.assertIn("provenance", summary)
            self.assertIn("--run-tag-refine", summary["provenance"]["argv"])
            self.assertIn("git", summary["provenance"])
            self.assertTrue(stages["pnp_pose_consensus"]["requested"])
            self.assertIn("filter_pnp_views_by_pose_consensus.py", consensus_command)
            self.assertIn("--center-threshold-m 0.35", consensus_command)
            self.assertIn("--rotation-threshold-deg 15.0", consensus_command)
            self.assertIn("pnp_pose_consensus/pnp_views_consensus.tsv", command)
            self.assertIn("--intrinsics_refine_mode shared_fxfy", command)
            self.assertIn("--intrinsics_focal_sigma_frac 0.005", command)
            self.assertIn("--intrinsics_distortion_sigma 0.05", command)
            self.assertIn("--intrinsics_max_total_focal_delta_frac 0.02", command)
            self.assertIn("--intrinsics_max_total_principal_delta_px 16.0", command)
            self.assertIn("--intrinsics_max_total_distortion_delta 0.0", command)
            self.assertIn("--accept_camera_max_intrinsic_focal_delta_frac 0.01", command)
            self.assertIn("--accept_camera_max_intrinsic_distortion_delta 0.15", command)
            self.assertIn("--min_camera_observations_for_use 16", command)
            self.assertIn("--min_camera_observations_for_delta 10", command)
            self.assertIn("--post_refine_observation_residual_gate_px 190.0", command)
            self.assertIn("--post_refine_outer_iterations 2", command)
            self.assertIn("--tower_face_width_initial_m 0.25", command)
            self.assertIn("--tower_face_width_sigma_m 0.02", command)
            self.assertIn("--optimize_tower_face_width", command)

    def test_tag_opencv5_distortion_refine_args_are_planned_for_tag_refine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            data_root.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-tag-refine",
                    "--tag-intrinsics-refine-mode", "per_camera_opencv5",
                    "--tag-intrinsics-distortion-sigma", "0.04",
                    "--tag-intrinsics-max-distortion-step", "0.006",
                    "--tag-intrinsics-max-total-distortion-delta", "0.12",
                    "--tag-accept-max-intrinsic-distortion-delta", "0.11",
                    "--force",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            command = stages["tag_refine_robust"]["commands"][0]

            self.assertIn("--intrinsics_refine_mode per_camera_opencv5", command)
            self.assertIn("--intrinsics_distortion_sigma 0.04", command)
            self.assertIn("--intrinsics_max_distortion_step 0.006", command)
            self.assertIn("--intrinsics_max_total_distortion_delta 0.12", command)
            self.assertIn("--accept_camera_max_intrinsic_distortion_delta 0.11", command)

    def test_dry_run_writes_run_manifest_and_summary_timing_without_t0_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            data_root.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-reports",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            manifest_path = output_root / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            final_report = (output_root / "final_report" / "index.html").read_text(encoding="utf-8")

            self.assertEqual(summary["run_manifest"], str(manifest_path))
            self.assertIn("run_manifest_url", summary)
            self.assertEqual(summary["run_timing"]["stage_count"], len(summary["stages"]))
            self.assertGreaterEqual(summary["run_timing"]["total_duration_s"], 0.0)
            self.assertEqual(manifest["data_root"], str(data_root.resolve()))
            self.assertEqual(manifest["whole_dir"], str((data_root / "whole_outer_tower").resolve()))
            self.assertEqual(manifest["output_root"], str(output_root.resolve()))
            self.assertEqual(manifest["whole_sequence"], "whole_outer_tower")
            self.assertEqual(manifest["summary_json"], str(output_root / "summary.json"))
            self.assertEqual(manifest["final_pose_yaml"], summary["final"]["pose_yaml"])
            self.assertEqual(manifest["final_source"], summary["final"]["source"])
            self.assertEqual(len(manifest["stages"]), len(summary["stages"]))
            self.assertIn("duration_s", manifest["stages"][0])
            self.assertIn("missing_inputs", manifest["stages"][0])
            self.assertIn("Run Timing / Recalib Inputs", final_report)
            self.assertIn("run_manifest.json", final_report)

    def test_execute_stages_records_real_stage_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            logs_dir = output_root / "logs"
            stage = {
                "name": "timed_stage",
                "requested": True,
                "commands": [[sys.executable, "-c", "import time; time.sleep(0.02)"]],
                "inputs": [],
                "outputs": [],
            }

            before = time.time()
            results = outer_pipeline.execute_stages(
                [stage],
                {"output_root": output_root, "logs_dir": logs_dir},
                dry_run=False,
                force=False,
            )
            after = time.time()

            result = results[0]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["return_codes"], [0])
            self.assertTrue(Path(result["log"]).is_file())
            self.assertNotEqual(result["started_at"], "")
            self.assertNotEqual(result["finished_at"], "")
            self.assertGreater(result["duration_s"], 0.0)
            started = outer_pipeline.parse_iso_timestamp(result["started_at"])
            finished = outer_pipeline.parse_iso_timestamp(result["finished_at"])
            self.assertGreaterEqual(started.timestamp(), before - 1.0)
            self.assertLessEqual(finished.timestamp(), after + 1.0)

    def test_legacy_tag_summary_reports_fixed_intrinsics_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tag_dir = Path(tmp)
            (tag_dir / "summary.json").write_text(
                json.dumps({
                    "cameras": {
                        "accepted_refined": ["1-1"],
                        "prior_only": ["1-2"],
                    },
                    "settings": {"observation_residual_gate_px": 600.0},
                    "post_refine_observation_gate": {
                        "enabled": True,
                        "threshold_px": 190.0,
                    },
                }),
                encoding="utf-8",
            )
            (tag_dir / "camera_tr_rig_delta_refined_accepted.yaml").write_text(
                "poses: []\n", encoding="utf-8")

            summary = outer_pipeline.summarize_tag_refine(tag_dir, "fixed")

            self.assertEqual(summary["intrinsics"]["refine_mode"], "fixed")
            self.assertEqual(summary["intrinsics"]["source"], "wrapper_default_or_legacy_summary")
            self.assertEqual(summary["settings"]["observation_residual_gate_px"], 600.0)
            self.assertEqual(summary["post_refine_observation_gate"]["threshold_px"], 190.0)

    def test_tag_summary_joins_per_camera_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tag_dir = Path(tmp)
            diagnostics = tag_dir / "diagnostics"
            diagnostics.mkdir(parents=True)
            (tag_dir / "summary.json").write_text(
                json.dumps({
                    "cameras": {
                        "accepted_refined": ["1-1"],
                        "prior_only": ["1-2"],
                    },
                    "intrinsics": {"refine_mode": "fixed"},
                }),
                encoding="utf-8",
            )
            (tag_dir / "camera_tr_rig_delta_refined_accepted.yaml").write_text(
                "poses: []\n", encoding="utf-8")
            (diagnostics / "camera_acceptance.tsv").write_text(
                "camera_index\tcamera_id\tdecision\toutput_pose\treason\tactive_delta\tused_observation\tobservation_count\tafter_median_px\tafter_under_300_fraction\n"
                "0\t1-1\taccepted_refined\trefined\tpasses_acceptance_gate\tyes\tyes\t12\t3.0\t1\n",
                encoding="utf-8",
            )
            (diagnostics / "camera_reprojection.tsv").write_text(
                "camera_index\tcamera_id\tobservation_count\tbefore_median_px\tbefore_p90_px\tafter_median_px\tafter_p90_px\tafter_max_px\tafter_under_100_fraction\tafter_under_300_fraction\n"
                "0\t1-1\t12\t10\t20\t3.0\t9.0\t25.0\t1\t1\n",
                encoding="utf-8",
            )
            (diagnostics / "camera_delta.tsv").write_text(
                "camera_index\tcamera_id\tactive\tused\tobservation_count\tused_observation_count\tdelta_rotation_deg\tdelta_translation_m\tcolmap_tracks\n"
                "0\t1-1\tyes\tyes\t20\t12\t0.5\t0.02\t\n",
                encoding="utf-8",
            )
            (diagnostics / "camera_intrinsics.tsv").write_text(
                "camera_index\tcamera_id\tdecision\toutput_intrinsics\treason\tbase_fx\tbase_fy\tbase_cx\tbase_cy\trefined_fx\trefined_fy\trefined_cx\trefined_cy\toutput_fx\toutput_fy\toutput_cx\toutput_cy\tfx_delta_frac\tfy_delta_frac\tmax_abs_focal_delta_frac\tcx_delta_px\tcy_delta_px\tprincipal_delta_px\n"
                "0\t1-1\taccepted_refined\trefined\tpasses_acceptance_gate\t5000\t5000\t2000\t1500\t5010\t4990\t2002\t1499\t5010\t4990\t2002\t1499\t0.002\t-0.002\t0.002\t2\t-1\t2.236\n",
                encoding="utf-8",
            )

            summary = outer_pipeline.summarize_tag_refine(tag_dir, "fixed")

            self.assertEqual(len(summary["camera_report_rows"]), 1)
            row = summary["camera_report_rows"][0]
            self.assertEqual(row["camera_id"], "1-1")
            self.assertEqual(row["after_p90_px"], "9.0")
            self.assertEqual(row["delta_rotation_deg"], "0.5")
            self.assertEqual(row["used_observation_count"], "12")
            self.assertEqual(row["intrinsic_decision"], "accepted_refined")
            self.assertEqual(row["output_intrinsics"], "refined")
            self.assertEqual(row["max_abs_focal_delta_frac"], "0.002")
            self.assertEqual(row["principal_delta_px"], "2.236")

    def test_residual_tail_stage_is_planned_with_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            data_root.mkdir()

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-reports",
                    "--force",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            self.assertIn("residual_tail_report", stages)
            self.assertTrue(stages["residual_tail_report"]["requested"])
            self.assertIn("analyze_outer_tag_residual_tail.py", stages["residual_tail_report"]["commands"][0])
            self.assertIn("residual_tail", summary)

    def test_residual_tail_summary_reports_present_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            (report_dir / "residual_tail_summary.json").write_text(
                json.dumps({
                    "camera_count": 24,
                    "observation_diagnostics": {
                        "available": False,
                        "message": "missing per-observation residual diagnostics",
                    },
                }),
                encoding="utf-8",
            )
            (report_dir / "residual_tail_report.html").write_text("<html></html>", encoding="utf-8")

            summary = outer_pipeline.summarize_residual_tail(report_dir)

            self.assertEqual(summary["status"], "present")
            self.assertEqual(summary["camera_count"], 24)
            self.assertFalse(summary["observation_diagnostics_available"])
            self.assertIn("missing per-observation", summary["observation_diagnostics_message"])

    def test_diagnostic_tag_refine_does_not_become_final_without_promote_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tag_dir = root / "tag_refine_robust"
            side_dir = root / "side_prior"
            ransac_dir = root / "ransac"
            tag_dir.mkdir()
            side_dir.mkdir()
            ransac_dir.mkdir()
            (tag_dir / "camera_tr_rig_delta_refined_accepted.yaml").write_text(
                "diagnostic: true\n", encoding="utf-8")
            (tag_dir / "summary.json").write_text(
                json.dumps({"settings": {"intrinsics_refine_mode": "per_camera_fxfy"}}),
                encoding="utf-8",
            )
            side_pose = side_dir / "camera_tr_rig_side_prior.yaml"
            side_pose.write_text("side: true\n", encoding="utf-8")
            args = type("Args", (), {
                "run_colmap_vote": False,
                "run_all": False,
                "run_side_prior": False,
                "run_tag_refine": True,
                "tag_intrinsics_refine_mode": "per_camera_fxfy",
                "promote_diagnostic_tag_refine": False,
            })()
            paths = {
                "tag_refine_dir": tag_dir,
                "existing_tag_dir": tag_dir,
                "side_prior_dir": side_dir,
                "existing_side_dir": side_dir,
                "colmap_ransac_dir": ransac_dir,
                "existing_ransac_dir": ransac_dir,
                "previous_outer_rig": root / "missing_previous.yaml",
            }

            pose, source = outer_pipeline.final_pose_candidate(paths, args, [])

            self.assertEqual(pose, side_pose)
            self.assertEqual(source, "existing_side_prior")

    def test_force_requested_stage_removes_stale_output_before_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            logs_dir = output_root / "logs"
            stale = output_root / "stage" / "old.txt"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale\n", encoding="utf-8")
            stage = {
                "name": "failing_stage",
                "requested": True,
                "commands": [[sys.executable, "-c", "import sys; sys.exit(5)"]],
                "inputs": [],
                "outputs": [stale],
            }

            results = outer_pipeline.execute_stages(
                [stage],
                {"output_root": output_root, "logs_dir": logs_dir},
                dry_run=False,
                force=True,
            )

            self.assertEqual(results[0]["status"], "failed")
            self.assertFalse(stale.exists())

    def test_bridge_prior_override_decision_requires_metric_gate_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "bridge_summary.json"
            args = type("Args", (), {
                "bridge_prior_override_policy": "gate",
                "bridge_prior_override_labels": "4-1,4-2,4-3",
            })()
            paths = {"bridge_summary_json": summary_path}

            summary_path.write_text(
                json.dumps({"quality_gates": {"metric_bridge": {"status": "fail", "passed": False}}}),
                encoding="utf-8")
            blocked = outer_pipeline.bridge_prior_override_decision(args, paths)
            self.assertEqual(blocked["effective_labels"], "")

            summary_path.write_text(
                json.dumps({"quality_gates": {"metric_bridge": {"status": "pass", "passed": True}}}),
                encoding="utf-8")
            enabled = outer_pipeline.bridge_prior_override_decision(args, paths)
            self.assertEqual(enabled["effective_labels"], "4-1,4-2,4-3")
            self.assertEqual(enabled["reason"], "metric_bridge_gate_passed")


if __name__ == "__main__":
    unittest.main()
