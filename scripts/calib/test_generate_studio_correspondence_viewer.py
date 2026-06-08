#!/usr/bin/env python3
"""Focused tests for the advanced studio correspondence viewer."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/generate_studio_correspondence_viewer.py"
PIPELINE_SCRIPT = REPO_ROOT / "scripts/calib/run_studio_calibration_pipeline.py"
SPEC = importlib.util.spec_from_file_location("generate_studio_correspondence_viewer", SCRIPT)
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


def write_pose_yaml(path, poses, pose_type="rig_tr_frame_face"):
    lines = [f"type: {pose_type}", f"pose_count: {len(poses)}", "poses:"]
    for index, pose in enumerate(poses):
        lines.extend([
            f"- index: {index}",
            f"  tx: {pose.get('tx', 0.0)}",
            f"  ty: {pose.get('ty', 0.0)}",
            f"  tz: {pose.get('tz', 0.0)}",
            f"  qx: {pose.get('qx', 0.0)}",
            f"  qy: {pose.get('qy', 0.0)}",
            f"  qz: {pose.get('qz', 0.0)}",
            f"  qw: {pose.get('qw', 1.0)}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_frame_face_pose_yaml(path):
    path.write_text(
        "type: rig_tr_frame_face\n"
        "poses:\n"
        "- frame_index: 0\n"
        "  face_id: 0\n"
        "  tx: 1.0\n"
        "  ty: 2.0\n"
        "  tz: 3.0\n"
        "  qx: 0.0\n"
        "  qy: 0.0\n"
        "  qz: 0.0\n"
        "  qw: 1.0\n",
        encoding="utf-8",
    )


def write_intrinsics_yaml(path):
    path.write_text(
        "type : CentralOpenCVModel\n"
        "width : 4096\n"
        "height : 3000\n"
        "parameters : [3000, 3000, 2048, 1500, 0, 0, 0, 0, 0, 0, 0, 0]\n",
        encoding="utf-8",
    )


class StudioCorrespondenceViewerTest(unittest.TestCase):
    def test_frame_face_inverse_pose_file_is_auto_inverted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rig_tr_face = root / "rig_tr_frame_face.yaml"
            rig_tr_face.write_text(
                "type: rig_tr_frame_face\n"
                "poses:\n"
                "- frame_index: 0\n"
                "  face_id: 0\n"
                "  tx: 1.0\n"
                "  ty: 2.0\n"
                "  tz: 3.0\n"
                "  qx: 0.0\n"
                "  qy: 0.0\n"
                "  qz: 0.0\n"
                "  qw: 1.0\n",
                encoding="utf-8",
            )
            face_tr_rig = root / "frame_face_tr_rig.yaml"
            face_tr_rig.write_text(
                "type: frame_face_tr_rig\n"
                "poses:\n"
                "- frame_index: 0\n"
                "  face_id: 0\n"
                "  tx: -1.0\n"
                "  ty: -2.0\n"
                "  tz: -3.0\n"
                "  qx: 0.0\n"
                "  qy: 0.0\n"
                "  qz: 0.0\n"
                "  qw: 1.0\n",
                encoding="utf-8",
            )

            point = [0.5, 0.25, -0.5]
            direct = viewer.transform_point(viewer.load_frame_face_poses(rig_tr_face)[(0, 0)], point)
            inverted = viewer.transform_point(viewer.load_frame_face_poses(face_tr_rig)[(0, 0)], point)

            self.assertEqual([round(float(value), 6) for value in direct], [1.5, 2.25, 2.5])
            self.assertEqual([round(float(value), 6) for value in inverted], [1.5, 2.25, 2.5])

    def test_generator_writes_outer_correspondence_and_inner_pose_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "viewer"
            assets_dir = root / "assets"
            assets_dir.mkdir()
            for name in ("three.min.js", "OrbitControls.js"):
                (assets_dir / name).write_text(f"// {name}\n", encoding="utf-8")

            studio32 = root / "studio_32_cameras.yaml"
            studio32.write_text(
                "cameras:\n"
                "- index: 0\n"
                "  label: 1-1\n"
                "  camera_id: 1-1\n"
                "  tx: 0.0\n"
                "  ty: 0.0\n"
                "  tz: 0.0\n"
                "  qx: 0.0\n"
                "  qy: 0.0\n"
                "  qz: 0.0\n"
                "  qw: 1.0\n"
                "- index: 24\n"
                "  label: inner0\n"
                "  camera_id: inner0\n"
                "  tx: 2.0\n"
                "  ty: 0.0\n"
                "  tz: 0.0\n"
                "  qx: 0.0\n"
                "  qy: 0.0\n"
                "  qz: 0.0\n"
                "  qw: 1.0\n"
                "coordinate_transform:\n"
                "  method: test_transform\n"
                "  source_coordinate_frame: studio_rig_current\n"
                "  aligned_coordinate_frame: studio_rig_y_down_z_forward\n"
                "  point_transform: p_aligned = R_aligned_from_source @ (p_source - origin_source)\n"
                "  origin_source: [1.0, 2.0, 3.0]\n"
                "  aligned_from_source_rotation:\n"
                "    - [1.0, 0.0, 0.0]\n"
                "    - [0.0, 1.0, 0.0]\n"
                "    - [0.0, 0.0, 1.0]\n",
                encoding="utf-8",
            )

            residuals = root / "observation_residuals.tsv"
            with residuals.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    delimiter="\t",
                    fieldnames=[
                        "frame_index", "filename", "camera_index", "camera_id",
                        "feature_id", "tag_id", "corner_id", "face_id",
                        "local_x", "local_y", "local_z",
                        "observed_x", "observed_y", "projected_x", "projected_y",
                        "residual_x_px", "residual_y_px", "residual_px",
                        "projection_status",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "frame_index": "0",
                    "filename": "frame_000000.jpg",
                    "camera_index": "0",
                    "camera_id": "1-1",
                    "feature_id": "100",
                    "tag_id": "25",
                    "corner_id": "0",
                    "face_id": "0",
                    "local_x": "0.5",
                    "local_y": "0.25",
                    "local_z": "0.0",
                    "observed_x": "100.0",
                    "observed_y": "110.0",
                    "projected_x": "102.0",
                    "projected_y": "111.0",
                    "residual_x_px": "2.0",
                    "residual_y_px": "1.0",
                    "residual_px": "2.236",
                    "projection_status": "ok",
                })

            frame_face_poses = root / "rig_tr_frame_face.yaml"
            write_frame_face_pose_yaml(frame_face_poses)

            large_dir = root / "large_pnp"
            large_dir.mkdir()
            (large_dir / "points.yaml").write_text(
                "points: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]\n"
                "feature_id_to_point_index:\n"
                "- feature_id: 10\n"
                "  point_index: 0\n"
                "- feature_id: 11\n"
                "  point_index: 1\n",
                encoding="utf-8",
            )
            write_pose_yaml(large_dir / "rig_tr_global.yaml", [{"tx": 0.0, "ty": 0.0, "tz": 1.0}])
            write_pose_yaml(large_dir / "camera_tr_rig.yaml", [{"tx": 0.0, "ty": 0.0, "tz": 0.0}])
            write_intrinsics_yaml(large_dir / "intrinsics0.yaml")
            (large_dir / "pnp_views.tsv").write_text(
                "imageset_index\tcamera_index\tfilename\tstatus\tinlier_count\treprojection_rmse_px\n"
                "0\t0\tlarge_000.jpg\tsolved\t2\t0.25\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--output-dir", str(output_dir),
                    "--studio32-yaml", str(studio32),
                    "--outer-observation-residuals-tsv", str(residuals),
                    "--outer-frame-face-pose-yaml", str(frame_face_poses),
                    "--large-pnp-dir", str(large_dir),
                    "--viewer-assets-dir", str(assets_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertTrue((output_dir / "index.html").is_file())
            data = json.loads((output_dir / "correspondence_data.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["cameras"]), 2)
            self.assertEqual(data["summary"]["outer"]["observation_count"], 1)
            self.assertEqual(data["summary"]["outer"]["frame_face_pose_count"], 1)
            self.assertEqual(data["outer"]["observations"][0]["world"], [0.5, 0.25, 0.0])
            self.assertEqual(len(data["outer"]["frame_face_poses"]), 1)
            frame_face_pose = data["outer"]["frame_face_poses"][0]
            self.assertEqual(frame_face_pose["frame_index"], 0)
            self.assertEqual(frame_face_pose["face_id"], 0)
            self.assertEqual(frame_face_pose["origin_three"], [0.0, -0.0, -0.0])
            self.assertEqual(frame_face_pose["axis_x_three"], [1.0, -0.0, -0.0])
            self.assertEqual(frame_face_pose["axis_y_three"], [0.0, -1.0, -0.0])
            self.assertEqual(frame_face_pose["axis_z_three"], [0.0, -0.0, -1.0])
            self.assertEqual(
                data["viewer_options"]["coordinate_transform"]["aligned_coordinate_frame"],
                "studio_rig_y_down_z_forward",
            )
            self.assertEqual(data["datasets"]["large"]["point_count"], 2)
            self.assertEqual(data["datasets"]["large"]["views"][0]["kind"], "per_view_pose_summary")
            html = (output_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("correspondence_data.json", html)
            self.assertIn("feature-level observed/projected/world correspondences", html)

    def test_marker_correspondence_counts_loaded_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            correspondence = root / "large_correspondences.tsv"
            fieldnames = [
                "dataset", "imageset_index", "camera_index", "camera_label",
                "filename", "feature_id", "point_index",
                "world_x", "world_y", "world_z",
                "camera_center_x", "camera_center_y", "camera_center_z",
                "observed_x", "observed_y", "projected_x", "projected_y",
                "residual_x_px", "residual_y_px", "residual_px",
                "projection_status",
            ]
            with correspondence.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
                writer.writeheader()
                for feature_id, residual in [(10, 0.25), (11, 0.5)]:
                    writer.writerow({
                        "dataset": "large",
                        "imageset_index": "3",
                        "camera_index": "0",
                        "camera_label": "1-1",
                        "filename": "000003.jpg",
                        "feature_id": str(feature_id),
                        "point_index": str(feature_id - 10),
                        "world_x": str(0.1 * feature_id),
                        "world_y": "0.0",
                        "world_z": "1.0",
                        "camera_center_x": "0.0",
                        "camera_center_y": "0.0",
                        "camera_center_z": "0.0",
                        "observed_x": "100.0",
                        "observed_y": "101.0",
                        "projected_x": "100.1",
                        "projected_y": "101.1",
                        "residual_x_px": "0.1",
                        "residual_y_px": "0.2",
                        "residual_px": str(residual),
                        "projection_status": "ok",
                    })

            data = viewer.load_marker_correspondences(
                "large",
                correspondence,
                max_rows=10,
                cameras=[{
                    "index": 0,
                    "label": "1-1",
                    "camera_id": "1-1",
                    "center": [0.0, 0.0, 0.0],
                }],
            )

            self.assertEqual(data["point_count"], 2)
            self.assertEqual(data["view_count"], 1)
            self.assertEqual(len(data["sample_points_three"]), 2)
            self.assertEqual(data["summary"]["observation_count"], 2)
            self.assertEqual(data["summary"]["loaded_observation_count"], 2)

    def test_pipeline_dry_run_includes_advanced_correspondence_viewer_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "studio_run"

            subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE_SCRIPT),
                    "--whole-data-root", str(root / "whole_root"),
                    "--inner-data-root", str(root / "inner_root"),
                    "--output-root", str(output_root),
                    "--run-tag", "test_run",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            stages = {stage["name"]: stage for stage in summary["stages"]}
            self.assertIn("generate_advanced_correspondence_viewer", stages)
            self.assertTrue(stages["generate_advanced_correspondence_viewer"]["requested"])
            command = stages["generate_advanced_correspondence_viewer"]["commands"][0]
            self.assertIn("generate_studio_correspondence_viewer.py", command)
            self.assertIn("--outer-observation-residuals-tsv", command)
            self.assertIn("--large-correspondence-tsv", command)
            self.assertIn("--small-correspondence-tsv", command)
            self.assertIn("--large-pnp-dir", command)
            self.assertEqual(
                summary["outputs"]["advanced_correspondence_viewer"],
                str(output_root / "advanced_correspondence_viewer_v1" / "index.html"),
            )
            self.assertEqual(summary["run_timing"]["stage_count"], 8)


if __name__ == "__main__":
    unittest.main()
