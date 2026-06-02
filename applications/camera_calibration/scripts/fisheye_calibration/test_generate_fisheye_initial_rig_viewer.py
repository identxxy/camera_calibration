#!/usr/bin/env python3
"""Focused tests for the four-fisheye initial rig viewer."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parent / "generate_fisheye_initial_rig_viewer.py"


def identity():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotate_x_pi():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


class GenerateFisheyeInitialRigViewerTest(unittest.TestCase):
    def test_generates_display_grid_viewer_from_kb8_camchain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camchain = {}
            topics = {
                "cam0": "/fisheye/left/image_raw",
                "cam1": "/fisheye/right/image_raw",
                "cam2": "/fisheye/bright/image_raw",
                "cam3": "/fisheye/bleft/image_raw",
            }
            transforms = {
                "cam0": identity(),
                "cam1": rotate_x_pi(),
                "cam2": rotate_x_pi(),
                "cam3": identity(),
            }
            for cam, topic in topics.items():
                camchain[cam] = {
                    "camera_model": "kb8",
                    "distortion_model": "kb8",
                    "intrinsics": [380.0, 381.0, 544.0, 640.0],
                    "distortion_coeffs": [-0.03, -0.02, 0.01, 0.0],
                    "resolution": [1088, 1280],
                    "T_cam_imu": transforms[cam],
                    "T_imu_cam": transforms[cam],
                    "rostopic": topic,
                    "timeshift_cam_imu": 0.0,
                }
            camchain_path = root / "camchain.yaml"
            camchain_path.write_text(yaml.safe_dump(camchain), encoding="utf-8")

            assets = root / "assets"
            assets.mkdir()
            for name in ("three.min.js", "OrbitControls.js", "TransformControls.js"):
                (assets / name).write_text(f"// {name}\n", encoding="utf-8")

            output = root / "viewer"
            subprocess.run([
                sys.executable,
                str(SCRIPT),
                "--camchain-yaml", str(camchain_path),
                "--output-dir", str(output),
                "--viewer-assets-dir", str(assets),
            ], check=True)

            html = (output / "index.html").read_text(encoding="utf-8")
            data = json.loads((output / "rig_data.json").read_text(encoding="utf-8"))
            self.assertIn("Fisheye Vision Initial Calibration Viewer", html)
            for name in ("three.min.js", "OrbitControls.js", "TransformControls.js"):
                self.assertTrue((output / name).is_file())
            by_label = {cam["label"]: cam for cam in data["cameras"]}
            self.assertEqual(set(by_label), {"left-up", "right-up", "right-down", "left-down"})
            self.assertLess(by_label["left-up"]["center"][1], 0.0)
            self.assertLess(by_label["right-up"]["center"][1], 0.0)
            self.assertGreater(by_label["left-down"]["center"][1], 0.0)
            self.assertGreater(by_label["right-down"]["center"][1], 0.0)
            self.assertLess(by_label["left-up"]["center"][0], 0.0)
            self.assertGreater(by_label["right-up"]["center"][0], 0.0)
            self.assertEqual(by_label["left-up"]["metrics"]["source_key"], "cam0")
            self.assertEqual(by_label["left-down"]["metrics"]["source_key"], "cam1")
            self.assertEqual(by_label["right-down"]["metrics"]["source_key"], "cam2")
            self.assertEqual(by_label["right-up"]["metrics"]["source_key"], "cam3")
            self.assertEqual(by_label["left-down"]["metrics"]["slot"], "left_down")
            self.assertEqual(by_label["right-up"]["metrics"]["slot"], "right_up")
            self.assertLess(by_label["left-up"]["basis"]["z"][1], -0.99)
            self.assertLess(by_label["right-up"]["basis"]["z"][1], -0.99)
            self.assertGreater(by_label["left-down"]["basis"]["z"][1], 0.99)
            self.assertGreater(by_label["right-down"]["basis"]["z"][1], 0.99)
            self.assertEqual(data["scene_alignment"]["source"], "layer_offset")
            self.assertEqual(data["scene_alignment"]["target_scene_up_vector"], [0.0, -1.0, 0.0])
            self.assertEqual(data["viewer_options"]["world_up_three"], [0.0, -1.0, 0.0])
            self.assertEqual(data["viewer_options"]["default_reference_up_vector_three"], [0.0, -1.0, 0.0])
            self.assertEqual(data["viewer_options"]["layout"], "display-grid")
            self.assertEqual(data["dataset_coverage"]["default_mode"], "fisheye_initial")
            self.assertEqual(data["source_transforms"]["cam0"]["metric_center_reference"], [0.0, 0.0, 0.0])
            self.assertEqual(len(data["reference_frames"]), 1)
            self.assertEqual(data["reference_frames"][0]["label"], "IMU / rig reference")
            self.assertIn("referenceFrameCount", html)


if __name__ == "__main__":
    unittest.main()
