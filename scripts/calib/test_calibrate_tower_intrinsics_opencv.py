#!/usr/bin/env python3
"""Focused tests for fast OpenCV tower intrinsic calibration helpers."""

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_tower_intrinsics_opencv as calib  # noqa: E402


class CalibrateTowerIntrinsicsOpenCVTest(unittest.TestCase):
    def test_opencv5_coeffs_map_to_repo_central_opencv_order(self):
        camera_matrix = np.asarray([
            [1000.0, 0.0, 320.0],
            [0.0, 1100.0, 240.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.asarray([0.1, -0.2, 0.01, -0.02, 0.3], dtype=np.float64)

        intrinsic = calib.intrinsic_from_opencv5(640, 480, camera_matrix, dist_coeffs)

        self.assertEqual(intrinsic["width"], 640)
        self.assertEqual(intrinsic["height"], 480)
        self.assertEqual(
            intrinsic["params"],
            [1000.0, 1100.0, 320.0, 240.0, 0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.01, -0.02])

    def test_repo_params_map_to_opencv5_coeff_order(self):
        intrinsic = {
            "width": 640,
            "height": 480,
            "params": [1000.0, 1100.0, 320.0, 240.0, 0.1, -0.2, 0.3, 4.0, 5.0, 6.0, 0.01, -0.02],
        }

        camera_matrix, dist_coeffs = calib.opencv5_from_intrinsic(intrinsic)

        self.assertTrue(np.allclose(camera_matrix, [
            [1000.0, 0.0, 320.0],
            [0.0, 1100.0, 240.0],
            [0.0, 0.0, 1.0],
        ]))
        self.assertTrue(np.allclose(dist_coeffs.ravel(), [0.1, -0.2, 0.01, -0.02, 0.3]))

    def test_uniform_sample_views_respects_limit_and_keeps_range(self):
        views = [{"frame_index": index} for index in range(10)]

        sampled = calib.uniform_sample_views(views, 4)

        self.assertLessEqual(len(sampled), 4)
        self.assertEqual(sampled[0]["frame_index"], 0)
        self.assertEqual(sampled[-1]["frame_index"], 9)
        self.assertEqual([view["frame_index"] for view in sampled], sorted(view["frame_index"] for view in sampled))

    def test_calibrate_camera_job_reports_insufficient_views(self):
        job = {
            "camera_index": 0,
            "camera_id": "1-1",
            "user_id": "1-1",
            "stage_name": "cam00_w4_1-1",
            "machine": "w4_D",
            "image_size": (640, 480),
            "views": [
                {
                    "frame_index": 0,
                    "filename": "frame_000000",
                    "object_points": np.zeros((8, 3), dtype=np.float32),
                    "image_points": np.zeros((8, 2), dtype=np.float32),
                },
            ],
            "candidate_views": 1,
            "candidate_points": 8,
            "input_observations": 8,
            "initial_intrinsic": {
                "width": 640,
                "height": 480,
                "params": [640.0, 640.0, 320.0, 240.0] + [0.0] * 8,
            },
            "prior_source": "initial_guess",
            "min_views": 2,
            "calibration_mode": "fxfycxcy_k1k2p1p2k3",
        }

        result = calib.calibrate_camera_job(job)

        self.assertEqual(result["status"], "insufficient_views")
        self.assertEqual(result["usable_views"], 1)
        self.assertEqual(result["usable_points"], 8)
        self.assertEqual(result["output_source"], "initial_guess")


if __name__ == "__main__":
    unittest.main()
