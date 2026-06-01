#!/usr/bin/env python3

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import calibrate_visual_imu_se3_from_full_board as se3


class VisualImuSe3ExportTest(unittest.TestCase):
    def test_export_writes_inverse_transforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rotation_yaml = tmp / "rotation.yaml"
            output_yaml = tmp / "output.yaml"

            prior_yaml = {
                "cam0": {
                    "camera_model": "kb8",
                    "T_cam_imu": np.eye(4).tolist(),
                },
                "cam1": {
                    "camera_model": "kb8",
                    "T_cam_imu": [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            }
            prior = {
                cam: {"T_cam_imu_prior": np.asarray(entry["T_cam_imu"], dtype=float)}
                for cam, entry in prior_yaml.items()
            }
            with rotation_yaml.open("w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "cam0": {"T_cam_imu": np.eye(4).tolist()},
                    "cam1": {"T_cam_imu": np.eye(4).tolist()},
                }, f)

            solve = {"t_cam0_imu": np.asarray([0.1, -0.2, 0.3], dtype=float)}
            output, T_0i = se3.write_calibration(prior_yaml, prior, rotation_yaml, solve, output_yaml)

            self.assertTrue(output_yaml.exists())
            self.assertTrue(np.allclose(T_0i[:3, 3], solve["t_cam0_imu"]))
            for cam, entry in output.items():
                T_cam_imu = np.asarray(entry["T_cam_imu"], dtype=float)
                T_imu_cam = np.asarray(entry["T_imu_cam"], dtype=float)
                self.assertTrue(
                    np.allclose(T_cam_imu @ T_imu_cam, np.eye(4), atol=1e-9),
                    cam,
                )
            self.assertTrue(np.allclose(np.asarray(output["cam0"]["T_cam_imu"])[:3, 3], [0.1, -0.2, 0.3]))
            self.assertTrue(np.allclose(np.asarray(output["cam1"]["T_cam_imu"])[:3, 3], [1.1, -0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
