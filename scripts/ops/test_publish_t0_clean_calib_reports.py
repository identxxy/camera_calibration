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
    def test_outer_frame_face_report_root_defaults_under_configured_run(self):
        old_values = {
            name: getattr(publisher, name)
            for name in [
                "ROOT",
                "BASE_URL",
                "RUN_TAG",
                "RUN",
                "CURRENT",
                "REPORTS",
                "FINAL_YAML",
                "CURRENT_FINAL_YAML",
                "CORRESPONDENCE_JSON",
                "CURRENT_CORRESPONDENCE_JSON",
                "OUTER_LARGE_INTRINSIC_REPORT",
                "OUTER_LARGE_QC_ROOT",
                "WHOLE_QC_ROOT",
                "OUTER_FRAME_FACE_REPORT_ROOT",
                "BRIDGE_CAMERA_ORIGIN_PROJECTION_REPORT",
            ]
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = type("Args", (), {
                    "root": str(root),
                    "base_url": "http://example.test",
                    "run_tag": "new_run",
                    "current_dir": "",
                    "outer_large_intrinsic_report": str(root / "outer_intrinsic_report"),
                    "outer_large_qc_root": str(root / "outer_large_qc"),
                    "whole_qc_root": str(root / "whole_qc"),
                    "outer_frame_face_report_root": "",
                })()

                publisher.configure(args)

                self.assertEqual(
                    publisher.OUTER_FRAME_FACE_REPORT_ROOT,
                    root / "studio_calibration_runs/new_run/outer_tower/frame_face_refine_wide200_then_gate6",
                )
                self.assertEqual(
                    publisher.BRIDGE_CAMERA_ORIGIN_PROJECTION_REPORT,
                    root / "studio_calibration_runs/new_run/inner_bridge/reports/bridge_all32_camera_origin_projection",
                )
        finally:
            for name, value in old_values.items():
                setattr(publisher, name, value)

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

    def test_readme_report_contract_matches_publisher_paths(self):
        readme = SCRIPT_PATH.with_name("README_t0_report_contract.md").read_text(encoding="utf-8")
        for report in publisher.CANONICAL_REPORTS:
            self.assertIn(
                f"/current_calibration/reports/{report['relative_index']}",
                readme,
            )
        stale_paths = [
            "03_inner_intrinsic_small_marker",
            "04_inner_extrinsic_small_marker",
            "05_outer_capture_whole_and_outer_large_marker",
            "06_outer_intrinsic_outer_large_marker",
            "07_outer_extrinsic_whole",
        ]
        for path in stale_paths:
            self.assertNotIn(path, readme)

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

    def test_correspondence_residuals_can_be_aggregated_for_inner_extrinsic_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_tsv = root / "correspondence_residuals.tsv"
            output_tsv = root / "camera_metrics.tsv"
            input_tsv.write_text(
                "dataset\tframe_index\tcamera_index\tcamera_label\tresidual_px\tprojection_status\n"
                "small\t0\t0\tinner0\t0.10\tok\n"
                "small\t0\t0\tinner0\t0.30\tok\n"
                "small\t1\t0\tinner0\t2.00\tmissing_point\n"
                "small\t2\t1\tinner1\t0.40\tok\n",
                encoding="utf-8",
            )

            result = publisher.write_camera_metrics_from_correspondence(input_tsv, output_tsv)

            self.assertEqual(result, output_tsv)
            rows = output_tsv.read_text(encoding="utf-8")
            self.assertIn("inner0", rows)
            self.assertIn("inner1", rows)
            self.assertIn("0.200000", rows)
            self.assertNotIn("2.00", rows)


if __name__ == "__main__":
    unittest.main()
