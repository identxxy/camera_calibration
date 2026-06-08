#!/usr/bin/env python3
"""Tests for the canonical t0 clean calibration publisher."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("publish_t0_clean_calib_reports.py")
SPEC = importlib.util.spec_from_file_location("publish_t0_clean_calib_reports", SCRIPT_PATH)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class PublishT0CleanCalibReportsTest(unittest.TestCase):
    def test_canonical_report_contract_is_explicit_and_unique(self):
        reports = publisher.CANONICAL_REPORTS
        self.assertEqual(len(reports), 7)
        self.assertEqual([report["number"] for report in reports], [str(i) for i in range(1, 8)])
        self.assertEqual(
            len({report["relative_index"] for report in reports}),
            len(reports),
        )
        self.assertTrue(all(report["relative_index"].endswith("/index.html") for report in reports))
        inner_extrinsic = next(report for report in reports if report["number"] == "3")
        self.assertIn("pixel reprojection residual", inner_extrinsic["description"])
        self.assertNotIn("layout report", inner_extrinsic["description"])

    def test_intrinsic_wrappers_document_fixed_log_colormap_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inner_dir = root / "inner"
            outer_dir = root / "outer"
            inner_dir.mkdir()
            outer_dir.mkdir()
            metrics = (
                "camera_index\tcamera_label\tframe_count\tresidual_count\tmedian_error_px\tp90_error_px\tmax_error_px\n"
                "0\tcam0\t3\t12\t0.12\t0.45\t0.80\n"
            )
            (inner_dir / "camera_metrics.tsv").write_text(metrics, encoding="utf-8")
            (inner_dir / "summary.json").write_text('{"camera_count": 1}\n', encoding="utf-8")
            (outer_dir / "camera_metrics.tsv").write_text(
                "camera_index\tuser_id\tusable_views\tusable_points\tresidual_count\tmedian_error_px\tp90_error_px\tmax_error_px\n"
                "0\t1-1\t3\t12\t12\t0.12\t0.45\t0.80\n",
                encoding="utf-8",
            )
            (outer_dir / "summary.json").write_text('{"camera_count": 1}\n', encoding="utf-8")

            publisher.publish_inner_intrinsic_wrapper(inner_dir)
            publisher.publish_outer_intrinsic_wrapper(outer_dir)

            inner_html = (inner_dir / "index.html").read_text(encoding="utf-8")
            outer_html = (outer_dir / "index.html").read_text(encoding="utf-8")
            for html in (inner_html, outer_html):
                self.assertIn("Color Scale", html)
                self.assertIn("10^-1", html)
                self.assertIn("10^1", html)
                self.assertIn("fixed log reprojection-error colormap", html)

    def test_inner_extrinsic_wrapper_reports_pixel_residuals_without_static_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "inner_extrinsic"
            report_dir.mkdir()
            (report_dir / "summary.json").write_text('{"camera_count": 1}\n', encoding="utf-8")
            (report_dir / "rig_layout_3d.png").write_bytes(b"not-used")
            metrics = root / "camera_metrics.tsv"
            metrics.write_text(
                "camera_index\tcamera_label\tresidual_count\tframe_count\tmedian_error_px\tmean_error_px\tp90_error_px\tmax_error_px\n"
                "0\tinner0\t12\t3\t0.12\t0.20\t0.45\t0.80\n",
                encoding="utf-8",
            )
            pnp = root / "camera_pnp_summary.tsv"
            pnp.write_text(
                "camera_index\tconnected\tpositive_views\tsolved_views\n"
                "0\tyes\t3\t3\n",
                encoding="utf-8",
            )

            publisher.publish_inner_extrinsic_wrapper(report_dir, metrics, pnp)

            html = (report_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Per-Camera Pixel Reprojection Error", html)
            self.assertIn("small-marker fixed-rig reprojection residuals in pixels", html)
            self.assertIn("0.12", html)
            self.assertNotIn("rig_layout_3d.png", html)
            self.assertNotIn("<img", html)


if __name__ == "__main__":
    unittest.main()
