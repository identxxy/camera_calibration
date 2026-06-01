#!/usr/bin/env python3
"""Regression tests for inner calibration report dataset parsing."""

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("generate_inner_calibration_report.py")
SPEC = importlib.util.spec_from_file_location("generate_inner_calibration_report", SCRIPT_PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def u32(value):
    return struct.pack(">I", value)


def f32(value):
    return struct.pack("<f", value)


class GenerateInnerCalibrationReportDatasetTest(unittest.TestCase):
    def test_version0_known_geometry_does_not_read_count3d(self):
        payload = bytearray()
        payload += b"calib_data"
        payload += u32(0)  # dataset version
        payload += u32(1)  # camera count
        payload += u32(640)
        payload += u32(480)
        payload += u32(0)  # imageset count
        payload += u32(1)  # known geometry count
        payload += f32(0.01)
        payload += u32(0)  # count_2d
        # Version 0 ends the known-geometry record here. A regression would
        # incorrectly try to read a missing count_3d field.

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "features_v0.bin"
            dataset_path.write_bytes(payload)
            parsed = report.read_dataset(dataset_path)

        self.assertEqual(parsed["camera_count"], 1)
        self.assertEqual(parsed["image_sizes"], [(640, 480)])
        self.assertEqual(len(parsed["known_geometries"]), 1)
        self.assertEqual(parsed["known_geometries"][0]["feature_id_to_position"], {})
        self.assertEqual(parsed["known_geometries"][0]["feature_id_to_position3d"], {})


if __name__ == "__main__":
    unittest.main()
