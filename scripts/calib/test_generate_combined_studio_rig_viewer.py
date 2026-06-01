#!/usr/bin/env python3
"""Focused tests for the combined studio rig viewer report."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_combined_studio_rig_viewer as combined_viewer  # noqa: E402


def write_intrinsics_yaml(path, fx, fy, cx, cy, width=4096, height=3000):
    path.write_text(
        "type : CentralOpenCVModel\n"
        f"width : {width}\n"
        f"height : {height}\n"
        f"parameters : [{fx}, {fy}, {cx}, {cy}, 0, 0, 0, 0]\n",
        encoding="utf-8",
    )


class CombinedStudioRigViewerTest(unittest.TestCase):
    def test_estimate_outer_column_gravity_alignment_excludes_topdown_side(self):
        cameras = []
        for side in range(1, 9):
            for level in (1, 2, 3):
                center = [float(side), float(level), 0.0]
                if side == 4:
                    center = [float(side), -100.0 * float(level), 0.0]
                cameras.append({
                    "label": f"{side}-{level}",
                    "kind": "outer_final",
                    "center": center,
                })

        alignment = combined_viewer.estimate_outer_column_gravity_alignment(cameras)

        self.assertIsNotNone(alignment)
        self.assertEqual(alignment["method"], "outer_column_mean_displacement_excluding_4_topdown")
        self.assertEqual(alignment["column_count"], 7)
        self.assertNotIn("4", alignment["used_columns"])
        self.assertEqual(alignment["segment_count"], 14)
        self.assertAlmostEqual(alignment["display_up_vector"][0], 0.0, places=6)
        self.assertAlmostEqual(alignment["display_up_vector"][1], 1.0, places=6)
        self.assertAlmostEqual(alignment["display_up_vector"][2], 0.0, places=6)

    def test_attach_calibration_quality_adds_intrinsic_sanity_status_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inner_metrics = root / "inner_metrics.tsv"
            inner_metrics.write_text(
                "camera_index\tresidual_count\tmedian_error_px\tp90_error_px\tmax_error_px\n"
                "0\t120\t0.12\t0.31\t1.50\n",
                encoding="utf-8",
            )
            inner_intrinsics = root / "inner_intrinsics"
            inner_intrinsics.mkdir()
            write_intrinsics_yaml(inner_intrinsics / "intrinsics0.yaml", 4700, 4701, 2048, 1500)

            outer_residuals = root / "outer_residuals.tsv"
            outer_residuals.write_text(
                "camera_id\tdecision\tobservation_count\tfinal_median_px\tfinal_p90_px\tfinal_max_px\n"
                "1-1\taccepted_refined\t330\t1.10\t2.20\t3.30\n"
                "1-2\taccepted_refined\t331\t1.00\t2.00\t3.00\n"
                "1-3\taccepted_refined\t332\t0.90\t1.90\t2.90\n",
                encoding="utf-8",
            )
            outer_intrinsics = root / "outer_intrinsics"
            outer_intrinsics.mkdir()
            write_intrinsics_yaml(outer_intrinsics / "intrinsics0.yaml", 4915.2, 4916.3, 2048, 1500)
            write_intrinsics_yaml(outer_intrinsics / "intrinsics1.yaml", 1000.0, 1000.0, 1200.0, 250.0, width=1000, height=500)
            write_intrinsics_yaml(outer_intrinsics / "intrinsics2.yaml", 1000.0, 1000.0, 2048.0, 1500.0)

            args = type("Args", (), {
                "inner_reprojection_metrics_tsv": inner_metrics,
                "inner_intrinsics_dir": inner_intrinsics,
                "outer_reprojection_tsv": outer_residuals,
                "outer_intrinsics_tsv": None,
                "outer_intrinsics_dir": outer_intrinsics,
            })()
            cameras = [
                {"label": "inner0"},
                {"label": "1-1"},
                {"label": "1-2"},
                {"label": "1-3"},
            ]

            summary = combined_viewer.attach_calibration_quality(cameras, args)
            by_label = {camera["label"]: camera["calibration_quality"] for camera in cameras}

            self.assertEqual(by_label["inner0"]["intrinsics_status"], "ok")
            self.assertEqual(by_label["1-1"]["intrinsics_status"], "ok")
            self.assertEqual(by_label["1-2"]["intrinsics_status"], "failed")
            self.assertIn("principal_point_outside_image_margin", by_label["1-2"]["intrinsics_flags"])
            self.assertEqual(by_label["1-3"]["intrinsics_status"], "warning")
            self.assertIn("focal_scale_outside_expected_range", by_label["1-3"]["intrinsics_flags"])
            self.assertEqual(summary["intrinsic_sanity"]["failed_camera_count"], 1)
            self.assertEqual(summary["intrinsic_sanity"]["warning_camera_count"], 1)
            self.assertEqual(summary["intrinsic_sanity"]["ok_camera_count"], 2)

    def test_write_html_adds_intrinsic_sanity_table_columns_and_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_html = Path(tmp) / "viewer/index.html"
            data = {
                "title": "test viewer",
                "generated_at": "2026-05-30T00:00:00",
                "frustum": {
                    "default_near": 0.3,
                    "default_far": 0.7,
                    "fill_opacity": 0.1,
                },
                "metrics": {
                    "intrinsic_sanity": {
                        "failed_camera_count": 1,
                        "warning_camera_count": 1,
                    },
                },
                "viewer_options": {},
                "dataset_coverage": {},
                "cameras": [{
                    "index": 0,
                    "label": "1-2",
                    "kind": "outer_final",
                    "center": [0, 0, 0],
                    "basis": {
                        "x": [1, 0, 0],
                        "y": [0, 1, 0],
                        "z": [0, 0, 1],
                    },
                    "axes": {},
                    "metrics": {},
                    "calibration_quality": {
                        "median_error_px": 1.0,
                        "p90_error_px": 2.0,
                        "fx": 1000.0,
                        "fy": 1000.0,
                        "cx": 1200.0,
                        "cy": 250.0,
                        "intrinsics_status": "failed",
                        "intrinsics_flags": ["principal_point_outside_image_margin"],
                    },
                }],
                "bounds": {"center": [0, 0, 0], "radius": 1.2},
                "sparse_point_cloud": {},
                "reprojection_reports": [],
            }

            combined_viewer.write_html(output_html, data)

            html = output_html.read_text(encoding="utf-8")
            self.assertIn("<th>cx</th><th>cy</th><th>Intrinsics</th>", html)
            self.assertIn("metric-intrinsics-sanity", html)
            self.assertIn("intrinsics-status", html)
            self.assertIn("function gravityAlignedAxis(localAxis)", html)
            self.assertIn("function horizontalRigForward()", html)
            self.assertIn("const gravityUp = WORLD_UP.clone();", html)
            self.assertIn("const controlsUp = gravityUp.clone().negate();", html)
            self.assertIn("multiplyScalar(-radius * 2.85 * zoom)", html)
            self.assertIn("offset = rigForward.clone().multiplyScalar(radius * 2.65 * zoom);", html)
            self.assertIn("rebuildOrbitControlsForCurrentUp(true, controlsUp);", html)
            self.assertEqual(
                json.loads((output_html.parent / "rig_data.json").read_text(encoding="utf-8")),
                data,
            )


if __name__ == "__main__":
    unittest.main()
