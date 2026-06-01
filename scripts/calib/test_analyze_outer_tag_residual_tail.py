#!/usr/bin/env python3
"""Focused tests for outer tag residual-tail diagnostics."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_outer_tag_residual_tail as analyze_tail  # noqa: E402


CAMERA_REPROJECTION_HEADER = (
    "camera_index\tcamera_id\tobservation_count\tbefore_median_px\tbefore_p90_px\t"
    "after_median_px\tafter_p90_px\tafter_max_px\tafter_under_100_fraction\t"
    "after_under_300_fraction\n"
)

CAMERA_ACCEPTANCE_HEADER = (
    "camera_index\tcamera_id\tdecision\toutput_pose\treason\tactive_delta\t"
    "used_observation\tobservation_count\tafter_median_px\tafter_under_300_fraction\n"
)


class AnalyzeOuterTagResidualTailTest(unittest.TestCase):
    def write_camera_tables(self, refine_dir):
        diagnostics = refine_dir / "diagnostics"
        diagnostics.mkdir(parents=True)
        (diagnostics / "camera_reprojection.tsv").write_text(
            CAMERA_REPROJECTION_HEADER
            + "0\t1-1\t40\t20\t60\t12\t80\t120\t0.95\t1.0\n"
            + "1\t1-2\t50\t30\t90\t40\t320\t900\t0.50\t0.70\n"
            + "2\t4-1\t25\t25\t85\t35\t180\t450\t0.80\t0.88\n",
            encoding="utf-8",
        )
        (diagnostics / "camera_acceptance.tsv").write_text(
            CAMERA_ACCEPTANCE_HEADER
            + "0\t1-1\taccepted_refined\trefined\tpasses_acceptance_gate\tyes\tyes\t40\t12\t1.0\n"
            + "1\t1-2\trejected_to_prior\tprior\tfailed_acceptance_gate\tyes\tyes\t50\t40\t0.70\n"
            + "2\t4-1\tinactive_prior_only\tprior\tbelow_min_camera_observations_for_delta\tno\tyes\t25\t35\t0.88\n",
            encoding="utf-8",
        )

    def test_per_camera_only_degrades_to_camera_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refine_dir = root / "tag_refine_robust"
            output_dir = root / "tail"
            self.write_camera_tables(refine_dir)

            summary = analyze_tail.analyze_refine_output(refine_dir, output_dir)

            self.assertFalse(summary["observation_diagnostics"]["available"])
            self.assertEqual(summary["worst_cameras_by_p90"][0]["camera_id"], "1-2")
            self.assertEqual(summary["worst_cameras_by_max"][0]["camera_id"], "1-2")
            self.assertEqual(summary["worst_cameras_by_under_300_fraction"][0]["camera_id"], "1-2")
            self.assertEqual(summary["camera_groups"]["accepted_refined"]["count"], 1)
            self.assertEqual(summary["camera_groups"]["prior_only"]["count"], 2)
            self.assertTrue((output_dir / "residual_tail_summary.json").exists())
            html = (output_dir / "residual_tail_report.html").read_text(encoding="utf-8")
            self.assertIn("missing per-observation residual diagnostics", html)
            self.assertIn("post-optimization trimming", html)

    def test_per_observation_tsv_summarizes_worst_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refine_dir = root / "tag_refine_robust"
            output_dir = root / "tail"
            diagnostics = refine_dir / "diagnostics"
            self.write_camera_tables(refine_dir)
            (diagnostics / "observation_residuals.tsv").write_text(
                "camera_id\tframe_index\ttag_id\tcorner_id\tface_id\tused_after_gate\tprojection_status\tresidual_px\n"
                "1-1\t3\t10\t0\tfront\tyes\tok\t40\n"
                "1-2\t7\t42\t1\tright\tno\tok\t650\n"
                "1-2\t7\t43\t2\tright\tyes\tok\t300\n"
                "4-1\t9\t42\t3\ttop\tno\tok\t220\n",
                encoding="utf-8",
            )

            summary = analyze_tail.analyze_refine_output(refine_dir, output_dir)

            self.assertTrue(summary["observation_diagnostics"]["available"])
            self.assertEqual(summary["observation_diagnostics"]["source_files"], [
                str((diagnostics / "observation_residuals.tsv").resolve())
            ])
            self.assertEqual(summary["worst_observations"][0]["camera_id"], "1-2")
            self.assertEqual(summary["worst_observations"][0]["frame_index"], "7")
            self.assertEqual(summary["worst_observations"][0]["tag_id"], "42")
            self.assertEqual(summary["worst_observations"][0]["corner_id"], "1")
            self.assertAlmostEqual(summary["worst_observations"][0]["residual_px"], 650.0)
            self.assertEqual(summary["worst_observations"][0]["used_after_gate"], "no")
            self.assertEqual(summary["worst_observations"][0]["projection_status"], "ok")
            self.assertEqual(summary["worst_by_camera"][0]["camera_id"], "1-2")
            self.assertEqual(summary["worst_by_frame"][0]["frame_index"], "7")
            self.assertEqual(summary["worst_by_tag"][0]["tag_id"], "42")
            self.assertEqual(summary["worst_by_face"][0]["face_id"], "right")
            html = (output_dir / "residual_tail_report.html").read_text(encoding="utf-8")
            self.assertIn("Worst Observations", html)
            self.assertIn("corner", html)
            self.assertIn("right", html)


if __name__ == "__main__":
    unittest.main()
