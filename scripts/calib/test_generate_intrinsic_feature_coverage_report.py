#!/usr/bin/env python3
"""Tests for intrinsic image-plane feature coverage reports."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_intrinsic_feature_coverage_report as report  # noqa: E402


def write_intrinsics(path, width=640, height=480):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "type: CentralOpenCVModel",
            f"width: {width}",
            f"height: {height}",
            "parameters: [500, 501, 320, 240, 0, 0, 0, 0, 0, 0, 0, 0]",
            "",
        ]),
        encoding="utf-8",
    )


class IntrinsicFeatureCoverageReportTest(unittest.TestCase):
    def test_residual_tsv_report_writes_per_camera_plots_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            residuals = root / "observation_residuals.tsv"
            residuals.write_text(
                "\t".join([
                    "frame_index", "filename", "camera_index", "camera_id",
                    "feature_id", "observed_x", "observed_y",
                    "projected_x", "projected_y",
                    "residual_x_px", "residual_y_px", "residual_px",
                    "projection_status",
                ])
                + "\n"
                + "0\t000000.jpg\t0\t1-1\t10\t10\t20\t11\t22\t1\t2\t2.236\tok\n"
                + "0\t000000.jpg\t0\t1-1\t11\t30\t40\t31\t41\t1\t1\t1.414\tok\n"
                + "1\t000001.jpg\t1\t1-2\t12\t100\t120\t98\t125\t-2\t5\t5.385\tok\n"
                + "1\t000001.jpg\t1\t1-2\t13\t130\t140\t0\t0\t0\t0\t0\tbehind_camera\n",
                encoding="utf-8",
            )
            intrinsics = root / "intrinsics"
            write_intrinsics(intrinsics / "intrinsics0_1-1.yaml")
            write_intrinsics(intrinsics / "intrinsics1.yaml")
            write_intrinsics(intrinsics / "intrinsics2_4-1.yaml")
            output = root / "report"

            report.main([
                "--residuals-tsv", str(residuals),
                "--intrinsics-dir", str(intrinsics),
                "--output-dir", str(output),
                "--title", "Outer Intrinsic Coverage",
                "--max-arrows-per-camera", "100",
            ])

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["summary"]["source_type"], "residuals_tsv")
            self.assertEqual(summary["summary"]["camera_count"], 3)
            counts = {row["camera_index"]: row["residual_count"] for row in summary["cameras"]}
            self.assertEqual(counts, {0: 2, 1: 1, 2: 0})
            self.assertTrue((output / "camera00_feature_coverage_reprojection.png").is_file())
            self.assertTrue((output / "camera01_feature_coverage_reprojection.png").is_file())
            self.assertTrue((output / "camera02_feature_coverage_reprojection.png").is_file())
            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("Outer Intrinsic Coverage", html)
            self.assertIn("camera-grid", html)
            self.assertIn("grid-template-columns: repeat(8", html)
            self.assertNotIn("background: #111", html)
            self.assertIn("camera00_feature_coverage_reprojection.png", html)
            self.assertIn("accumulated observed feature locations", html)
            self.assertIn("10^-1", html)
            self.assertIn("10^1", html)
            self.assertEqual(report.REPROJECTION_COLORMAP_VMIN_PX, 1e-1)
            self.assertEqual(report.REPROJECTION_COLORMAP_VMAX_PX, 1e1)

    def test_intrinsics_lookup_accepts_camera_id_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intrinsics = root / "intrinsics"
            write_intrinsics(intrinsics / "intrinsics7_3-2.yaml", width=4096, height=3000)

            loaded = report.load_intrinsics_for_camera(intrinsics, 7)

            self.assertEqual(loaded["width"], 4096)
            self.assertEqual(loaded["height"], 3000)

    def test_camera_indices_filter_keeps_only_requested_cameras(self):
        cameras = [
            {"camera_index": 0},
            {"camera_index": 1},
            {"camera_index": 2},
            {"camera_index": 5},
        ]

        filtered = report.filter_cameras(cameras, "1-2,5")

        self.assertEqual([camera["camera_index"] for camera in filtered], [1, 2, 5])


if __name__ == "__main__":
    unittest.main()
