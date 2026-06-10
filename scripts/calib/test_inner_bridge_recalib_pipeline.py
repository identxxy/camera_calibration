#!/usr/bin/env python3
"""Focused tests for the fast inner/bridge recalibration wrapper."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/run_inner_bridge_recalib_pipeline.py"
SCRIPT_DIR = SCRIPT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_inner_bridge_recalib_pipeline as inner_pipeline  # noqa: E402
import generate_combined_studio_rig_viewer as combined_viewer  # noqa: E402

OUTER_CAMERAS = [
    ("w4_D", "1-1"), ("w4_D", "1-2"), ("w4_D", "1-3"),
    ("w4_D", "2-1"), ("w4_D", "2-2"), ("w4_D", "2-3"),
    ("w4_D", "3-1"), ("w4_D", "3-2"), ("w4_D", "3-3"),
    ("w4_D", "4-1"), ("w4_D", "4-2"), ("w4_D", "4-3"),
    ("w3_D", "5-1"), ("w3_D", "5-2"), ("w3_D", "5-3"),
    ("w3_D", "6-1"), ("w3_D", "6-2"), ("w3_D", "6-3"),
    ("w3_D", "7-1"), ("w3_D", "7-2"), ("w3_D", "7-3"),
    ("w3_D", "8-1"), ("w3_D", "8-2"), ("w3_D", "8-3"),
]
INNER_CAMERAS = [
    ("w1_D", "22463688"), ("w1_D", "22463690"),
    ("w1_D", "22587611"), ("w1_D", "22587616"),
    ("w2_D", "22463689"), ("w2_D", "22463691"),
    ("w2_D", "22463702"), ("w2_D", "22587614"),
]


def write_session(root, name, cameras):
    session = root / name
    session.mkdir(parents=True)
    image_dirs = []
    manifest_rows = ["camera_index\tstage_name\tmachine\tcamera_id\tframe_count\n"]
    for index, (machine, camera_id) in enumerate(cameras):
        image_dir = session / f"cam{index:02d}_{machine}_{camera_id}"
        image_dir.mkdir()
        for frame in range(2):
            (image_dir / f"frame_{frame:06d}.jpg").write_bytes(b"")
        image_dirs.append(str(image_dir))
        manifest_rows.append(
            f"{index}\tcam{index:02d}_{machine}_{camera_id}\t{machine}\t{camera_id}\t2\n"
        )
    (session / "image_directories.txt").write_text(",".join(image_dirs) + "\n", encoding="utf-8")
    (session / "manifest.tsv").write_text("".join(manifest_rows), encoding="utf-8")


def write_intrinsics(root):
    outer_dir = root / "whole_outer_tower/fixed_intrinsic_pnp_colmap_fallback_v1"
    inner_dir = root / "final_inner8_calibration_v1/intrinsics/small_marker_opencv_grid4_pattern3_v2"
    outer_dir.mkdir(parents=True)
    inner_dir.mkdir(parents=True)
    for index in range(len(OUTER_CAMERAS)):
        (outer_dir / f"intrinsics{index}.yaml").write_text(f"outer: {index}\n", encoding="utf-8")
    for index, (_machine, camera_id) in enumerate(INNER_CAMERAS):
        (inner_dir / f"intrinsics{index}_{camera_id}.yaml").write_text(f"inner: {index}\n", encoding="utf-8")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class InnerBridgeRecalibPipelineTest(unittest.TestCase):
    def test_binary_fallback_prefers_current_integration_build_before_releases(self):
        fallback_paths = [str(path) for path in inner_pipeline.DEFAULT_T0_BINARY_FALLBACKS]

        integration_index = next(
            index for index, path in enumerate(fallback_paths)
            if "camera_calibration_integration_build" in path
        )
        release_index = next(
            index for index, path in enumerate(fallback_paths)
            if "camera_calibration_release_1771ad3" in path
        )
        self.assertLess(integration_index, release_index)

    def test_force_clear_stage_outputs_removes_declared_outputs_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "out"
            input_dataset = root / "stage/input.bin"
            dataset = root / "stage/output.bin"
            work_dir = root / "stage/work"
            dataset.parent.mkdir(parents=True)
            input_dataset.write_text("input\n", encoding="utf-8")
            dataset.write_text("stale\n", encoding="utf-8")
            work_dir.mkdir(parents=True)
            (work_dir / "stale.txt").write_text("stale\n", encoding="utf-8")
            stage = {
                "name": "extract_features",
                "inputs": {
                    "dataset": str(input_dataset),
                },
                "outputs": {
                    "dataset": str(dataset),
                    "work_dir": str(work_dir),
                    "input_dataset_copy_for_report": str(input_dataset),
                },
            }
            fingerprint = inner_pipeline.stage_fingerprint_path(output_root, stage)
            fingerprint.parent.mkdir(parents=True)
            fingerprint.write_text("stale\n", encoding="utf-8")

            inner_pipeline.clear_stage_outputs(stage, output_root)

            self.assertFalse(dataset.exists())
            self.assertFalse(work_dir.exists())
            self.assertTrue(input_dataset.exists())
            self.assertFalse(fingerprint.exists())

    def test_prepare_bridge_intrinsics_normalizes_outer_central_opencv_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outer_dir = root / "outer"
            inner_dir = root / "inner"
            outer_dir.mkdir()
            inner_dir.mkdir()
            for index, (_machine, camera_id) in enumerate(OUTER_CAMERAS):
                (outer_dir / f"intrinsics{index}_{camera_id}.yaml").write_text(
                    "type: CentralOpenCV\n"
                    "width: 4096\n"
                    "height: 3000\n"
                    "parameters: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n",
                    encoding="utf-8",
                )
            for index, (_machine, camera_id) in enumerate(INNER_CAMERAS):
                (inner_dir / f"intrinsics{index}_{camera_id}.yaml").write_text(
                    "type : CentralOpenCVModel\n"
                    "width : 4096\n"
                    "height : 3000\n"
                    "parameters : [1, 1, 0, 0, 0, 0, 0, 0]\n",
                    encoding="utf-8",
                )
            layout = {
                "expected_camera_count": 32,
                "inner_indices": list(range(24, 32)),
            }

            result = inner_pipeline.prepare_bridge_intrinsics(
                root / "out",
                outer_dir,
                inner_dir,
                layout,
            )

            self.assertTrue(result["ready"])
            copied = (root / "out/planned_inputs/bridge_all32_fixed_intrinsics/intrinsics0.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("type: CentralOpenCVModel", copied)

    def test_infer_outer_intrinsic_metrics_finds_current_outer_large_marker_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_2026_05_31_v3"
            outer_intrinsics = (
                root / "studio_calibration_runs/run/outer_tower/"
                "frame_face_refine_gate6/intrinsics_refined"
            )
            metrics = (
                root / "current_calibration/reports/"
                "06_outer_intrinsics_outer_large_marker/camera_metrics.tsv"
            )
            outer_intrinsics.mkdir(parents=True)
            metrics.parent.mkdir(parents=True)
            metrics.write_text("camera_index\tuser_id\tresidual_count\n", encoding="utf-8")

            inferred = inner_pipeline.infer_outer_intrinsic_metrics_tsv(
                outer_intrinsics,
                data_root,
            )

            self.assertEqual(inferred, metrics.resolve(strict=False))

    def test_infer_whole_coverage_prefers_fullres_before_hybrid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fullres = root / "whole_outer24_filtered_min4_fullres_min4cam" / "per_camera_stats.tsv"
            hybrid = root / "whole_outer24_filtered_min4_hybrid_min4cam" / "per_camera_stats.tsv"
            fullres.parent.mkdir(parents=True)
            hybrid.parent.mkdir(parents=True)
            fullres.write_text("camera_id\tpassing_images\n", encoding="utf-8")
            hybrid.write_text("camera_id\tpassing_images\n", encoding="utf-8")

            inferred = inner_pipeline.infer_whole_coverage_tsv(root)

            self.assertEqual(inferred, fullres.resolve(strict=False))

    def test_combined_viewer_uses_outer_final_pose_yaml_for_outer_ring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge_yaml = root / "bridge.yaml"
            outer_final_yaml = root / "outer_final.yaml"
            write_pose_yaml(bridge_yaml, 32)
            write_pose_yaml(outer_final_yaml, 24)
            output_html = root / "viewer/index.html"
            whole_tsv = root / "whole_stats.tsv"
            whole_tsv.write_text(
                "camera_id\tselected_passing_frames\tpassing_images\ttotal_tags\tmax_tags\n"
                "1-1\t0\t20\t80\t6\n"
                "22463688\t4\t7\t28\t5\n",
                encoding="utf-8",
            )
            large_pnp = root / "large_pnp.tsv"
            large_pnp.write_text(
                "camera_index\tuser_id\tconnected\tpositive_views\tsolved_views\ttotal_inliers\tmedian_view_error_px\n"
                "0\t1-1\tyes\t5\t4\t179\t0.42\n"
                "24\t22463688\tyes\t75\t39\t9338\t0.05\n",
                encoding="utf-8",
            )
            small_pnp = root / "small_pnp.tsv"
            small_pnp.write_text(
                "camera_index\tuser_id\tconnected\tpositive_views\tsolved_views\ttotal_inliers\tmedian_view_error_px\n"
                "0\t22463688\tyes\t23\t9\t1311\t0.06\n",
                encoding="utf-8",
            )
            inner_metrics = root / "inner_camera_metrics.tsv"
            inner_metrics.write_text(
                "camera_index\tcamera_label\tresidual_count\tmedian_error_px\tmean_error_px\tp90_error_px\tmax_error_px\n"
                "0\t22463688\t120\t0.12\t0.20\t0.31\t1.50\n",
                encoding="utf-8",
            )
            inner_intrinsics = root / "inner_intrinsics"
            inner_intrinsics.mkdir()
            (inner_intrinsics / "intrinsics0.yaml").write_text(
                "type : CentralOpenCVModel\n"
                "width : 4096\n"
                "height : 3000\n"
                "parameters : [4700, 4701, 2048, 1500, 0, 0, 0, 0]\n",
                encoding="utf-8",
            )
            outer_residuals = root / "outer_residuals.tsv"
            outer_residuals.write_text(
                "camera_index\tcamera_id\tdecision\tdecision_reason\tobservation_count\tfinal_median_px\tfinal_p90_px\tfinal_max_px\n"
                "0\t1-1\taccepted_refined\tpasses\t330\t1.10\t2.20\t3.30\n",
                encoding="utf-8",
            )
            outer_intrinsics = root / "outer_intrinsics.tsv"
            outer_intrinsics.write_text(
                "camera_index\tcamera_id\toutput_fx\toutput_fy\toutput_cx\toutput_cy\toutput_intrinsics\n"
                "0\t1-1\t4915.2\t4916.3\t2048\t1500\trefined\n",
                encoding="utf-8",
            )
            tower_yaml = root / "tower_up.yaml"
            write_pose_yaml(tower_yaml, 3)

            args = type("Args", (), {
                "inner_bridge_pose_yaml": bridge_yaml,
                "bridge_summary_json": None,
                "outer_colmap_images_txt": root / "missing_images.txt",
                "outer_colmap_summary_json": None,
                "outer_final_pose_yaml": outer_final_yaml,
                "tower_pose_yaml": tower_yaml,
                "whole_coverage_tsv": whole_tsv,
                "large_marker_pnp_summary_tsv": large_pnp,
                "small_marker_pnp_summary_tsv": small_pnp,
                "inner_reprojection_metrics_tsv": inner_metrics,
                "inner_intrinsics_dir": inner_intrinsics,
                "outer_reprojection_tsv": outer_residuals,
                "outer_intrinsics_tsv": outer_intrinsics,
                "large_marker_board_pose_yaml": None,
                "small_marker_board_pose_yaml": None,
                "bridge_marker_board_pose_yaml": None,
                "output_html": output_html,
                "viewer_scope": "combined",
                "combined_image_directories_file": None,
                "inner_image_directories_file": None,
                "outer_image_directories_file": None,
                "inner_bridge_indices": "24,25,26,27,28,29,30,31",
                "topdown_bridge_indices": "9,10,11",
                "topdown_labels": "4-1,4-2,4-3",
                "default_near": 0.3,
                "default_far": 0.7,
                "frustum_half_width_over_depth": 0.45,
                "frustum_half_height_over_depth": 0.32,
                "frustum_fill_opacity": 0.11,
                "title": "test viewer",
            })()

            data = combined_viewer.build_viewer_data(args)

            self.assertEqual(len(data["cameras"]), 32)
            outer_sources = [
                camera["source"]
                for camera in data["cameras"]
                if not str(camera["label"]).startswith("inner")
            ]
            self.assertNotIn("colmap_sim3_approx", outer_sources)
            self.assertEqual(outer_sources.count("outer_final_pose_yaml"), 24)
            self.assertNotIn("bridge_metric_topdown", outer_sources)
            self.assertTrue(data["metrics"]["bridge_outer_alignment"]["available"])
            self.assertEqual(
                data["inputs"]["outer_final_pose_yaml"],
                str(outer_final_yaml.resolve()),
            )
            self.assertEqual(
                data["metrics"]["outer_final_pose_yaml"],
                str(outer_final_yaml.resolve()),
            )
            cameras_by_label = {camera["label"]: camera for camera in data["cameras"]}
            self.assertTrue(cameras_by_label["1-1"]["coverage"]["whole"]["active"])
            self.assertEqual(cameras_by_label["1-1"]["coverage"]["whole"]["observation_count"], 330)
            self.assertEqual(cameras_by_label["1-1"]["coverage"]["whole"]["raw_selected_passing_frames"], 0)
            self.assertTrue(cameras_by_label["inner0"]["coverage"]["whole"]["active"])
            self.assertTrue(cameras_by_label["inner0"]["coverage"]["large_marker"]["active"])
            self.assertTrue(cameras_by_label["inner0"]["coverage"]["small_marker"]["active"])
            self.assertFalse(cameras_by_label["inner1"]["coverage"]["small_marker"]["active"])
            self.assertEqual(cameras_by_label["inner0"]["calibration_quality"]["residual_count"], 120)
            self.assertEqual(cameras_by_label["inner0"]["calibration_quality"]["fx"], 4700.0)
            self.assertEqual(cameras_by_label["1-1"]["calibration_quality"]["decision"], "accepted_refined")
            self.assertEqual(cameras_by_label["1-1"]["calibration_quality"]["fx"], 4915.2)
            board_alignment = data["viewer_options"]["board_orientation_alignment"]
            self.assertIn("whole_tower_faces", board_alignment["sources"])
            self.assertIn("aggregate", board_alignment)

    def test_combined_viewer_downsamples_first_frame_textures(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_html = root / "viewer/index.html"
            image_dir = root / "cam24_inner0"
            image_dir.mkdir()
            source_image = image_dir / "000000.jpg"
            Image.new("RGB", (1024, 768), color=(180, 120, 70)).save(source_image)
            image_dirs_file = root / "image_dirs.txt"
            image_dirs_file.write_text(str(image_dir), encoding="utf-8")

            stale_dir = output_html.parent / "camera_frames"
            stale_dir.mkdir(parents=True)
            stale_file = stale_dir / "stale.jpg"
            stale_file.write_bytes(b"stale")

            cameras = [{
                "index": 0,
                "label": "inner0",
                "image_url": "",
                "image_texture_url": "",
            }]
            args = type("Args", (), {
                "output_html": output_html,
                "combined_image_directories_file": image_dirs_file,
                "inner_image_directories_file": None,
                "outer_image_directories_file": None,
                "texture_max_width": 256,
                "texture_jpeg_quality": 80,
            })()

            count, metrics = combined_viewer.attach_first_frame_images(cameras, args)

            self.assertEqual(count, 1)
            self.assertFalse(stale_file.exists())
            self.assertEqual(cameras[0]["image_url"], "camera_frames/00_inner0.jpg")
            self.assertEqual(cameras[0]["image_texture_url"], cameras[0]["image_url"])
            texture_path = output_html.parent / cameras[0]["image_url"]
            with Image.open(texture_path) as texture:
                self.assertLessEqual(max(texture.size), 256)
                self.assertEqual(metrics["first_frame_texture_pixel_count"], texture.width * texture.height)
            self.assertLess(metrics["first_frame_texture_rgba_estimated_mb"], 0.3)

    def test_pipeline_passes_outer_final_pose_yaml_to_combined_viewer_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            outer_final_yaml = root / "outer_tower/latest/tag_refine_robust/camera_tr_rig_delta_refined_accepted.yaml"
            write_session(data_root, "small_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_bridge_all32", OUTER_CAMERAS + INNER_CAMERAS)
            write_intrinsics(data_root)
            write_pose_yaml(outer_final_yaml, 24)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--outer-final-pose-yaml", str(outer_final_yaml),
                    "--dry-run",
                    "--run-large-bridge",
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
            viewer_stage = stages["generate_combined_bridge_viewer"]
            bridge_stage = stages["evaluate_inner_outer_bridge_alignment"]
            viewer_command = viewer_stage["planned_command"]
            bridge_command = bridge_stage["planned_command"]

            self.assertEqual(summary["priors"]["outer_final_pose_yaml"], str(outer_final_yaml.resolve()))
            self.assertTrue(summary["final_yaml_candidates"]["outer_final_pose_ready"])
            self.assertEqual(
                summary["final_yaml_candidates"]["combined_bridge_outer_pose_source"],
                "outer_final_pose_yaml",
            )
            self.assertEqual(viewer_stage["inputs"]["outer_pose_source"], "outer_final_pose_yaml")
            self.assertIn("--outer_final_pose_yaml", viewer_command)
            self.assertIn(str(outer_final_yaml), viewer_command)
            self.assertIn("--inner_reprojection_metrics_tsv", viewer_command)
            self.assertIn("--inner_intrinsics_dir", viewer_command)
            self.assertIn("--outer_camera_tr_rig", bridge_command)
            self.assertIn(str(outer_final_yaml), bridge_command)

    def test_all32_bridge_dry_run_generates_combined_intrinsics_and_all32_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            write_session(data_root, "small_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_bridge_all32", OUTER_CAMERAS + INNER_CAMERAS)
            write_intrinsics(data_root)
            outer_prior = data_root / "colmap_outer24_firstframe_colmap404_v3/fixed_intrinsics/sparse_txt_final24_fixedK_ba/images.txt"
            outer_prior.parent.mkdir(parents=True)
            outer_prior.write_text("# empty test COLMAP file\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-large-bridge",
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
            bridge_intrinsics = Path(summary["priors"]["bridge_intrinsics"])

            self.assertIn("provenance", summary)
            self.assertIn("--run-large-bridge", summary["provenance"]["argv"])
            self.assertIn("git", summary["provenance"])
            self.assertEqual(summary["bridge_layout"]["inner_indices"], list(range(24, 32)))
            self.assertEqual(summary["bridge_layout"]["outer_indices"], list(range(24)))
            self.assertEqual(summary["bridge_layout"]["outer_labels"], [camera_id for _machine, camera_id in OUTER_CAMERAS])
            self.assertEqual(summary["bridge_intrinsics"]["ready_count"], 32)
            self.assertTrue((bridge_intrinsics / "intrinsics0.yaml").is_file())
            self.assertTrue((bridge_intrinsics / "intrinsics23.yaml").is_file())
            self.assertEqual((bridge_intrinsics / "intrinsics24.yaml").read_text(encoding="utf-8"), "inner: 0\n")
            self.assertEqual((bridge_intrinsics / "intrinsics31.yaml").read_text(encoding="utf-8"), "inner: 7\n")

            pnp_command = stages["estimate_large_marker_bridge_pnp"]["planned_command"]
            self.assertIn(f"--fixed_intrinsics_directory {bridge_intrinsics}", pnp_command)
            self.assertNotIn("small_marker_opencv_grid4_pattern3_v2", pnp_command)

            ba_command = stages["refine_large_marker_bridge_joint_ba"]["planned_command"]
            self.assertIn("--debug_fix_points", ba_command)
            self.assertIn("--debug_fix_intrinsics", ba_command)
            self.assertIn("--model central_opencv", ba_command)
            self.assertNotIn("--localize_only", ba_command)
            self.assertIn("--max_ba_iterations 80", ba_command)

            eval_command = stages["evaluate_inner_outer_bridge_alignment"]["planned_command"]
            self.assertIn("--inner_indices 24,25,26,27,28,29,30,31", eval_command)
            self.assertIn("--outer_indices " + ",".join(str(index) for index in range(24)), eval_command)
            self.assertIn("--outer_labels " + ",".join(camera_id for _machine, camera_id in OUTER_CAMERAS), eval_command)

            viewer_command = stages["generate_combined_bridge_viewer"]["planned_command"]
            self.assertIn("--inner_bridge_indices 24,25,26,27,28,29,30,31", viewer_command)
            self.assertIn("--topdown_bridge_indices " + ",".join(str(index) for index in range(24)), viewer_command)
            self.assertIn("--correspondence_data_url ../../advanced_correspondence_viewer_v1/correspondence_data.json", viewer_command)
            self.assertNotIn("/home/vox/calib_data", pnp_command + eval_command + viewer_command)

    def test_legacy_bridge_outputs_do_not_make_final_viewer_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            outer_prior = data_root / "colmap_outer24_firstframe_colmap404_v3/fixed_intrinsics/sparse_txt_final24_fixedK_ba/images.txt"
            write_session(data_root, "small_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_bridge_all32", OUTER_CAMERAS + INNER_CAMERAS)
            write_intrinsics(data_root)
            outer_prior.parent.mkdir(parents=True)
            outer_prior.write_text("# empty test COLMAP file\n", encoding="utf-8")
            legacy_dir = output_root / "bridge_colmap_inner_refined_v1"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "bridge_summary.json").write_text('{"legacy": true}\n', encoding="utf-8")
            (legacy_dir / "camera_tr_inner_refined_plus_outer_topdown.yaml").write_text(
                "pose_count: 0\nposes:\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--outer-prior", str(outer_prior),
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
            stages = {stage["name"]: stage for stage in summary["stages"]}

            self.assertEqual(
                stages["generate_combined_bridge_viewer"]["status"],
                "blocked_missing_inputs",
            )
            self.assertTrue(
                summary["final_yaml_candidates"]["legacy_bridge_pose_yaml"].endswith(
                    "bridge_colmap_inner_refined_v1/camera_tr_inner_refined_plus_outer_topdown.yaml"
                )
            )
            self.assertTrue(
                summary["final_yaml_candidates"]["bridge_pose_yaml"].endswith(
                    "large_marker_bridge_all32/fixed_points_joint_ba_stride1_dense_v1/camera_tr_rig.yaml"
                )
            )

    def test_dry_run_plans_correspondence_residual_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            write_session(data_root, "small_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_bridge_all32", OUTER_CAMERAS + INNER_CAMERAS)
            write_intrinsics(data_root)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-large-inner-init",
                    "--run-small-fixed-rig-quality",
                    "--run-large-bridge",
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
            candidates = summary["final_yaml_candidates"]

            expected = {
                "export_large_inner_marker_correspondence_residuals":
                    candidates["large_inner_marker_correspondence_residuals_tsv"],
                "export_small_marker_correspondence_residuals":
                    candidates["small_marker_correspondence_residuals_tsv"],
                "export_inner_reprojection_correspondence_residuals":
                    candidates["inner_reprojection_correspondence_residuals_tsv"],
                "export_large_marker_bridge_correspondence_residuals":
                    candidates["large_marker_bridge_correspondence_residuals_tsv"],
            }
            for stage_name, output_tsv in expected.items():
                self.assertIn(stage_name, stages)
                command = stages[stage_name]["planned_command"]
                self.assertIn("export_calibration_correspondence_residuals.py", command)
                self.assertIn("--output-tsv", command)
                self.assertIn(output_tsv, command)
                self.assertEqual(stages[stage_name]["outputs"]["correspondence_residuals_tsv"], output_tsv)

            self.assertTrue(
                candidates["large_inner_marker_correspondence_residuals_tsv"].endswith(
                    "large_marker_inner8/fixed_intrinsic_large_marker_inner8_init_v1/correspondence_residuals.tsv"
                )
            )
            self.assertTrue(
                candidates["small_marker_correspondence_residuals_tsv"].endswith(
                    "small_marker_inner8/fixed_intrinsic_small_grid4_quality_probe_v1/correspondence_residuals.tsv"
                )
            )
            self.assertTrue(
                candidates["inner_reprojection_correspondence_residuals_tsv"].endswith(
                    "reports/inner_reprojection/correspondence_residuals.tsv"
                )
            )
            self.assertTrue(
                candidates["large_marker_bridge_correspondence_residuals_tsv"].endswith(
                    "large_marker_bridge_all32/fixed_points_joint_ba_stride1_dense_v1/correspondence_residuals.tsv"
                )
            )
            self.assertTrue(
                candidates["large_marker_bridge_initializer_correspondence_residuals_tsv"].endswith(
                    "large_marker_bridge_all32/fixed_intrinsic_bridge_pnp_stride1_v1/correspondence_residuals.tsv"
                )
            )

    def test_dry_run_writes_manifest_and_final_report_timing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            write_session(data_root, "small_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_bridge_all32", OUTER_CAMERAS + INNER_CAMERAS)
            write_intrinsics(data_root)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-large-bridge",
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
            manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
            final_report = (output_root / "final_report/index.html").read_text(encoding="utf-8")

            self.assertEqual(summary["run_manifest"], str(output_root / "run_manifest.json"))
            self.assertEqual(manifest["inputs"]["data_root"], str(data_root.resolve()))
            self.assertEqual(manifest["inputs"]["small_marker_sequence"], "small_marker_inner8")
            self.assertEqual(manifest["inputs"]["large_inner_marker_sequence"], "large_marker_inner8")
            self.assertEqual(manifest["inputs"]["large_marker_sequence"], "large_marker_bridge_all32")
            self.assertEqual(manifest["inputs"]["outer_source"], summary["priors"]["outer_final_pose_yaml"])
            self.assertEqual(
                manifest["inputs"]["outer_source_kind"],
                summary["final_yaml_candidates"]["combined_bridge_outer_pose_source"],
            )
            self.assertIn("outer_source_path", manifest["inputs"])
            self.assertGreaterEqual(manifest["total_duration_s"], 0.0)
            self.assertEqual(summary["run_timing"]["total_duration_s"], manifest["total_duration_s"])
            self.assertEqual(len(manifest["stages"]), len(summary["stages"]))
            for stage in manifest["stages"]:
                self.assertIn("started_at", stage)
                self.assertIn("finished_at", stage)
                self.assertIn("duration_s", stage)
                self.assertIn("command", stage)
                self.assertIn("status", stage)
            self.assertIn("Run Timing / Recalib Inputs", final_report)
            self.assertIn(str(data_root.resolve()), final_report)
            self.assertIn("large_marker_bridge_all32", final_report)
            self.assertIn("run_manifest.json", final_report)

    def test_partial_inner_capture_blocks_fixed_intrinsic_reindexing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "calib_data"
            output_root = root / "out"
            write_session(data_root, "small_marker_inner8", INNER_CAMERAS)
            write_session(data_root, "large_marker_inner8", INNER_CAMERAS[:2] + INNER_CAMERAS[3:])
            write_session(data_root, "large_marker_bridge_all32", OUTER_CAMERAS + INNER_CAMERAS)
            write_intrinsics(data_root)

            stale_state = (
                output_root
                / "large_marker_inner8"
                / "fixed_intrinsic_large_marker_inner8_init_v1"
            )
            stale_state.mkdir(parents=True)
            (stale_state / "camera_tr_rig.yaml").write_text("stale: true\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                    "--run-large-inner-init",
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
            contracts = summary["final_yaml_candidates"]["input_contracts"]

            self.assertFalse(contracts["large_marker_inner8"]["ready"])
            self.assertIn(
                "large_marker_inner8_requires_8_canonical_cameras_got_7",
                contracts["large_marker_inner8"]["notes"],
            )
            self.assertEqual(stages["estimate_large_inner_fixed_intrinsic_rig"]["status"], "blocked_missing_inputs")
            self.assertEqual(summary["final_yaml_candidates"]["inner_prior_source"], "configured_inner_prior")
            self.assertNotEqual(
                summary["final_yaml_candidates"]["effective_inner_prior_state_dir"],
                str(stale_state),
            )

    def test_non_dry_run_blocked_requested_stage_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            output_root = root / "out"
            write_session(data_root, "large_marker_inner8", INNER_CAMERAS[:-1])
            write_intrinsics(data_root)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--run-large-inner-init",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertNotEqual(completed.returncode, 0)
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            self.assertEqual(stages["estimate_large_inner_fixed_intrinsic_rig"]["status"], "blocked_missing_inputs")
            self.assertIn("estimate_large_inner_fixed_intrinsic_rig", completed.stdout)

    def test_failed_allow_failure_stage_does_not_write_success_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            output_root.mkdir()
            old_output = output_root / "old.tsv"
            old_output.write_text("stale\n", encoding="utf-8")
            args = type("Args", (), {"dry_run": False})()
            stage = inner_pipeline.make_stage(
                "allow_failure_probe",
                "ready_to_run",
                {"input": "synthetic"},
                {"summary": str(old_output)},
                planned_command=(
                    f"{sys.executable} -c \"import sys; sys.exit(7)\""
                ),
                allow_failure=True,
            )

            stages = inner_pipeline.execute_requested_stages(
                args, [stage], output_root, REPO_ROOT)

            self.assertEqual(stages[0]["status"], "failed_allowed")
            self.assertFalse(
                inner_pipeline.stage_fingerprint_path(output_root, stage).exists())


if __name__ == "__main__":
    unittest.main()
