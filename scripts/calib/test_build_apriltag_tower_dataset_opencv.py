#!/usr/bin/env python3
"""Tests for OpenCV AprilTag tower dataset geometry."""

import math
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts/calib"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_apriltag_tower_dataset_opencv as builder  # noqa: E402


class BuildAprilTagTowerDatasetOpenCVTest(unittest.TestCase):
    def base_config(self):
        return {
            "faces": 8,
            "tag_columns": 2,
            "tag_rows": 16,
            "tag_size_m": 0.08,
            "tag_spacing_m": 0.02,
            "first_tag_id": 0,
            "face_id_stride": 32,
            "face_width_m": 0.24,
            "face0_angle_degrees": 0,
        }

    def assert_point_close(self, actual, expected):
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=7)

    def test_rotated_tags_map_opencv_corners_to_measured_physical_corners(self):
        rotated_config = self.base_config()
        rotated_config["tag_rotation_degrees"] = 180

        rotated_points = builder.build_tower_points(rotated_config)
        expected_apothem = 0.24 / (2.0 * math.tan(math.pi / 8.0))

        # The verified preview images are generated with each tag bitmap
        # rotated 180 degrees in place. For tag 0 in the bottom-left cell,
        # OpenCV reports corners as physical lower-right, lower-left,
        # upper-left, upper-right.
        self.assert_point_close(rotated_points[0 * 4 + 0], (expected_apothem, -0.01, -0.79))
        self.assert_point_close(rotated_points[0 * 4 + 1], (expected_apothem, -0.09, -0.79))
        self.assert_point_close(rotated_points[0 * 4 + 2], (expected_apothem, -0.09, -0.71))
        self.assert_point_close(rotated_points[0 * 4 + 3], (expected_apothem, -0.01, -0.71))

    def test_default_tower_yaml_uses_measured_face_width_for_apothem(self):
        config = builder.read_tower_config(
            REPO_ROOT / "applications/camera_calibration/patterns/apriltag_tower_8faces_2x16_8cm.yaml")
        points = builder.build_tower_points(config)

        expected_apothem = 0.24 / (2.0 * math.tan(math.pi / 8.0))
        self.assertEqual(float(config["face_width_m"]), 0.24)
        self.assertEqual(int(config["tag_rotation_degrees"]), 180)
        self.assertAlmostEqual(points[0 * 4 + 0][0], expected_apothem, places=7)

    def test_detections_to_features_keeps_partial_corner_observations(self):
        features = builder.detections_to_features([
            {
                "tag_id": 17,
                "corners": [[1.0, 2.0], None, [5.0, 6.0], [None, None]],
            },
        ])

        self.assertEqual(features, [(1.0, 2.0, 68), (5.0, 6.0, 70)])


if __name__ == "__main__":
    unittest.main()
