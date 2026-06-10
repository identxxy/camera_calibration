#!/usr/bin/env python3
"""Focused tests for the combined studio rig viewer report."""

import json
import math
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


def write_colmap_images_txt(path, labels):
    lines = ["# minimal COLMAP images.txt fixture\n"]
    for image_id, label in enumerate(labels, 1):
        tx = -float(image_id)
        ty = -0.2 * float(image_id % 3)
        tz = -0.1 * float(image_id % 2)
        lines.append(
            f"{image_id} 1 0 0 0 {tx:.6f} {ty:.6f} {tz:.6f} "
            f"{image_id} cam{image_id:02d}_{label}_f000000.jpg\n"
        )
        lines.append("0 0 -1 10 10 1\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_pose_yaml(path, count):
    lines = [f"pose_count: {count}", "poses:"]
    for index in range(count):
        lines.extend([
            f"  - index: {index}",
            "    qw: 1.0",
            "    qx: 0.0",
            "    qy: 0.0",
            "    qz: 0.0",
            f"    tx: {0.1 * index:.6f}",
            f"    ty: {0.01 * index:.6f}",
            f"    tz: {0.001 * index:.6f}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CombinedStudioRigViewerTest(unittest.TestCase):
    def test_estimate_outer_column_gravity_alignment_excludes_topdown_side(self):
        cameras = []
        for side in range(1, 9):
            theta = 2.0 * math.pi * (side - 1) / 8.0
            for level in (1, 2, 3):
                center = [math.cos(theta), float(level), math.sin(theta)]
                if side == 4:
                    center = [float(side), -100.0 * float(level), 0.0]
                cameras.append({
                    "label": f"{side}-{level}",
                    "kind": "outer_final",
                    "center_metric": center,
                })

        alignment = combined_viewer.estimate_outer_column_gravity_alignment(cameras)

        self.assertIsNotNone(alignment)
        self.assertEqual(alignment["method"], "outer_level_plane_mean_normal_origin_level2_gap4")
        self.assertEqual(alignment["column_count"], 7)
        self.assertNotIn("4", alignment["used_columns"])
        self.assertEqual(alignment["level_plane_count"], 3)
        self.assertAlmostEqual(alignment["display_up_vector"][0], 0.0, places=6)
        self.assertAlmostEqual(alignment["display_up_vector"][1], 1.0, places=6)
        self.assertAlmostEqual(alignment["display_up_vector"][2], 0.0, places=6)
        self.assertEqual(alignment["negative_z_gap_labels"], ["3-2", "5-2"])
        self.assertNotIn("4-2", alignment["origin_level2_labels"])

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

    def test_frame_face_pose_yaml_provides_whole_tower_orientation_sanity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_face_pose = root / "rig_tr_frame_face.yaml"
            frame_face_pose.write_text(
                "pose_count: 1\n"
                "poses:\n"
                "  - index: 0\n"
                "    frame_index: 244\n"
                "    face_id: 0\n"
                "    tx: 0.0\n"
                "    ty: 0.0\n"
                "    tz: 0.0\n"
                "    qx: -0.7071067811865476\n"
                "    qy: 0.0\n"
                "    qz: 0.0\n"
                "    qw: 0.7071067811865476\n",
                encoding="utf-8",
            )
            args = type("Args", (), {
                "tower_pose_yaml": frame_face_pose,
                "large_marker_board_pose_yaml": None,
                "bridge_marker_board_pose_yaml": None,
                "small_marker_board_pose_yaml": None,
            })()
            gravity_alignment = {
                "source": "test_gravity",
                "metric_up_vector": [0.0, 1.0, 0.0],
                "display_up_vector": [0.0, -1.0, 0.0],
                "robust_angle_threshold_deg": 30.0,
            }

            alignment = combined_viewer.estimate_board_orientation_alignment(args, gravity_alignment)
            whole = alignment["sources"]["whole_tower_faces"]

            self.assertEqual(whole["sample_count"], 1)
            self.assertEqual(whole["source"], str(frame_face_pose.resolve()))
            self.assertIn("independent frame-face plane normals", whole["description"])
            self.assertAlmostEqual(whole["median_angle_from_horizontal_deg"], 0.0, places=6)

    def test_attach_intrinsic_residuals_keeps_intrinsic_metrics_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inner_metrics = root / "inner_metrics.tsv"
            inner_metrics.write_text(
                "camera_index\tresidual_count\tframe_count\tmedian_error_px\tp90_error_px\tmax_error_px\n"
                "0\t240\t20\t0.05\t0.14\t0.40\n",
                encoding="utf-8",
            )
            outer_metrics = root / "outer_metrics.tsv"
            outer_metrics.write_text(
                "user_id\tresidual_count\tusable_views\tusable_points\tmedian_error_px\tp90_error_px\tmax_error_px\n"
                "1-1\t1200\t80\t1200\t0.07\t0.19\t0.60\n",
                encoding="utf-8",
            )
            args = type("Args", (), {
                "inner_intrinsic_metrics_tsv": inner_metrics,
                "outer_intrinsic_metrics_tsv": outer_metrics,
            })()
            cameras = [{"label": "inner0"}, {"label": "1-1"}, {"label": "2-1"}]

            summary = combined_viewer.attach_intrinsic_residuals(cameras, args)

            self.assertEqual(summary["inner_camera_count"], 1)
            self.assertEqual(summary["outer_camera_count"], 1)
            self.assertEqual(cameras[0]["intrinsic_residual"]["stage"], "inner intrinsic calibration")
            self.assertAlmostEqual(cameras[0]["intrinsic_residual"]["median_error_px"], 0.05)
            self.assertEqual(cameras[1]["intrinsic_residual"]["stage"], "outer intrinsic calibration")
            self.assertAlmostEqual(cameras[1]["intrinsic_residual"]["p90_error_px"], 0.19)
            self.assertEqual(cameras[2]["intrinsic_residual"]["source"], "missing")

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
                "viewer_options": {
                    "correspondence_data_url": "../../advanced_correspondence_viewer_v1/correspondence_data.json",
                },
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
            self.assertIn("Camera Intrinsic Residuals / Dataset Residuals", html)
            self.assertIn("<th>Intrinsic obs</th><th>Intrinsic med px</th><th>Intrinsic p90 px</th>", html)
            self.assertIn("<th>Dataset obs</th><th>Dataset med px</th><th>Dataset p90 px</th>", html)
            self.assertIn("<th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>Intrinsics</th>", html)
            self.assertIn("metric-intrinsics-sanity", html)
            self.assertIn("intrinsics-status", html)
            self.assertIn("function intrinsicResidualQuality(cam)", html)
            self.assertIn('const isBridgeBa = c.status === "all32_bridge_ba_residuals";', html)
            self.assertIn('source: isBridgeBa ? coverageMode + "_ba" : coverageMode + "_pnp"', html)
            self.assertIn("median_error_px: c.median_error_px ?? c.median_view_error_px ?? null", html)
            self.assertIn("p90_error_px: c.p90_error_px ?? null", html)
            self.assertIn("intrinsic residual:", html)
            self.assertIn("dataset/extrinsic residual:", html)
            self.assertIn("function gravityAlignedAxis(localAxis)", html)
            self.assertIn("function horizontalRigForward()", html)
            self.assertIn("const gravityUp = WORLD_UP.clone();", html)
            self.assertIn("const controlsUp = gravityUp.clone();", html)
            self.assertIn("multiplyScalar(radius * 2.85 * zoom)", html)
            self.assertIn("offset = rigForward.clone().multiplyScalar(radius * 2.65 * zoom);", html)
            self.assertIn("rebuildOrbitControlsForCurrentUp(true, controlsUp);", html)
            self.assertIn("Load Corr", html)
            self.assertIn("id=\"correspondence-frame-slider\"", html)
            self.assertIn("id=\"correspondence-all-frames\"", html)
            self.assertIn("id=\"correspondence-shared-only\"", html)
            self.assertIn("Shared tracks only", html)
            self.assertIn("id=\"correspondence-group-mode\"", html)
            self.assertIn("Timeline", html)
            self.assertIn("Face ID", html)
            self.assertNotIn("By face in frame", html)
            self.assertNotIn("By timestamp, all faces", html)
            self.assertIn("Timeline trail", html)
            self.assertIn("Face ID mode arranges the whole capture", html)
            self.assertIn("id=\"correspondence-point-group\"", html)
            self.assertIn("Max shown", html)
            self.assertIn("Residual <=", html)
            self.assertIn("function correspondenceGroupModeName()", html)
            self.assertIn("function correspondenceGroupKey(obs, includeFrame)", html)
            self.assertIn("function correspondenceTrackKey(obs, includeFrame)", html)
            self.assertIn("function correspondenceFrameSharedTrackStats(dataset)", html)
            self.assertIn("function filterSharedCorrespondenceTracks(candidates)", html)
            self.assertIn("function balancedCorrespondenceSelection(candidates, maxShown)", html)
            self.assertIn("TOWER_FACE_PLANE_WIDTH_M = 0.25", html)
            self.assertIn("function appendFrameFaceOutlines(dataset, frame", html)
            self.assertIn("function appendFrameFaceTimelineTrail(dataset, currentFrame", html)
            self.assertIn("Timeline mode shows the selected synchronized frame", html)
            self.assertIn("final independent frame-face plane poses from BA", html)
            self.assertIn("no ideal octagon and no face-width constraint", html)
            self.assertNotIn("anchor face", html)
            self.assertNotIn("Anchor face", html)
            self.assertNotIn("nominal rigid tower", html)
            self.assertNotIn("function buildRigidTowerContext", html)
            self.assertIn("function populateCorrespondenceFrameControl()", html)
            self.assertIn("function populateCorrespondencePointGroupControl()", html)
            self.assertIn("All frames", html)
            self.assertIn("All faces", html)
            self.assertIn("All points", html)
            self.assertIn("shared tracks=", html)
            self.assertIn("No shared tracks survived", html)
            self.assertIn("function updateCorrespondenceOverlay()", html)
            self.assertIn("correspondenceObjectCount", html)
            self.assertEqual(
                json.loads((output_html.parent / "rig_data.json").read_text(encoding="utf-8")),
                data,
            )

    def test_outer_final_pose_yaml_default_none_allows_colmap_fallback(self):
        parser = combined_viewer.build_arg_parser()
        parsed = parser.parse_args([])
        self.assertIsNone(parsed.outer_final_pose_yaml)
        self.assertIsNone(parsed.tower_pose_yaml)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge_yaml = root / "bridge.yaml"
            colmap_images = root / "images.txt"
            output_html = root / "viewer/index.html"

            write_pose_yaml(bridge_yaml, 32)
            write_colmap_images_txt(colmap_images, ["4-1", "4-2", "4-3", "1-1"])

            args = type("Args", (), {
                "inner_bridge_pose_yaml": bridge_yaml,
                "bridge_summary_json": None,
                "outer_colmap_images_txt": colmap_images,
                "outer_colmap_summary_json": None,
                "outer_final_pose_yaml": None,
                "tower_pose_yaml": None,
                "whole_coverage_tsv": None,
                "large_marker_pnp_summary_tsv": None,
                "large_marker_correspondence_tsv": None,
                "small_marker_pnp_summary_tsv": None,
                "inner_reprojection_metrics_tsv": None,
                "inner_intrinsic_metrics_tsv": None,
                "inner_intrinsics_dir": None,
                "inner_intrinsics_index_offset": -1,
                "outer_reprojection_tsv": None,
                "outer_intrinsics_tsv": None,
                "outer_intrinsics_dir": None,
                "outer_intrinsic_metrics_tsv": None,
                "large_marker_board_pose_yaml": None,
                "small_marker_board_pose_yaml": None,
                "bridge_marker_board_pose_yaml": None,
                "output_html": output_html,
                "viewer_scope": "combined",
                "combined_image_directories_file": None,
                "inner_image_directories_file": None,
                "outer_image_directories_file": None,
                "texture_max_width": 768,
                "texture_jpeg_quality": 82,
                "inner_bridge_indices": "24,25,26,27,28,29,30,31",
                "topdown_bridge_indices": "9,10,11",
                "topdown_labels": "4-1,4-2,4-3",
                "default_near": 0.3,
                "default_far": 0.7,
                "frustum_half_width_over_depth": 0.45,
                "frustum_half_height_over_depth": 0.32,
                "frustum_fill_opacity": 0.11,
                "title": "fallback viewer",
                "correspondence_data_url": "",
            })()

            data = combined_viewer.build_viewer_data(args)

            self.assertEqual(data["metrics"]["outer_pose_source"], "colmap_sim3_approx")
            self.assertEqual(data["inputs"]["outer_final_pose_yaml"], "")
            self.assertFalse(data["metrics"]["bridge_outer_alignment"]["available"])
            outer_sources = [
                camera["source"]
                for camera in data["cameras"]
                if not str(camera["label"]).startswith("inner")
            ]
            self.assertIn("colmap_sim3_approx", outer_sources)
            self.assertNotIn("outer_final_pose_yaml", outer_sources)


if __name__ == "__main__":
    unittest.main()
