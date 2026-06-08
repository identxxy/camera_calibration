#!/usr/bin/env python3
"""Tests for OpenCV intrinsic residual report generation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("generate_opencv_intrinsics_report.py")
SPEC = importlib.util.spec_from_file_location("generate_opencv_intrinsics_report", SCRIPT_PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class GenerateOpenCVIntrinsicsReportTest(unittest.TestCase):
    def test_report_uses_fixed_log_colormap_and_compact_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intrinsics_dir = root / "opencv_intrinsics"
            output_dir = root / "report"
            intrinsics_dir.mkdir()
            (intrinsics_dir / "intrinsics_summary.tsv").write_text(
                "\t".join([
                    "camera_index",
                    "stage_name",
                    "machine",
                    "user_id",
                    "status",
                    "usable_views",
                    "usable_points",
                    "rms",
                    "fx",
                    "fy",
                    "cx",
                    "cy",
                    "width",
                    "height",
                ])
                + "\n"
                + "0\tcam00_w4_1-1\tw4_D\t1-1\tsolved\t1\t2\t0.2\t500\t501\t320\t240\t640\t480\n",
                encoding="utf-8",
            )
            (intrinsics_dir / "residuals_camera0_1-1.tsv").write_text(
                "observed_x\tobserved_y\terror_x\terror_y\n"
                "10\t20\t0.1\t0.2\n"
                "30\t40\t2.0\t1.0\n",
                encoding="utf-8",
            )

            report.main([
                "--intrinsics-dir", str(intrinsics_dir),
                "--output-dir", str(output_dir),
                "--max-arrows-per-camera", "10",
            ])

            html = (output_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("10^-1", html)
            self.assertIn("10^1", html)
            self.assertIn("grid-template-columns: repeat(8", html)
            self.assertNotIn("background: #111", html)
            self.assertTrue((output_dir / "camera00_reprojection_arrows_log.png").is_file())
            self.assertEqual(report.REPROJECTION_COLORMAP_VMIN_PX, 1e-1)
            self.assertEqual(report.REPROJECTION_COLORMAP_VMAX_PX, 1e1)


if __name__ == "__main__":
    unittest.main()
