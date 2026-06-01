#!/usr/bin/env python3
"""Regression tests for Seeker calibration report dataset parsing."""

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("build_seeker_calibration_report.py")
SPEC = importlib.util.spec_from_file_location("build_seeker_calibration_report", SCRIPT_PATH)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def u32(value):
    return struct.pack(">I", value)


class BuildSeekerCalibrationReportDatasetTest(unittest.TestCase):
    def test_version1_minimal_dataset_is_accepted(self):
        payload = bytearray()
        payload += b"calib_data"
        payload += u32(1)  # dataset version
        payload += u32(1)  # camera count
        payload += u32(640)
        payload += u32(480)
        payload += u32(0)  # imageset count
        payload += u32(0)  # known geometry count

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "features_v1.bin"
            dataset_path.write_bytes(payload)
            parsed = report.read_dataset_feature_points(dataset_path)

        self.assertEqual(parsed["points"], [])
        self.assertEqual(parsed["observations"], [])
        self.assertEqual(parsed["imagesets"], 0)
        self.assertEqual(parsed["features"], 0)
        self.assertEqual(parsed["image_size"], (640, 480))

    def test_unsupported_dataset_version_is_rejected(self):
        payload = b"calib_data" + u32(2)

        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "features_v2.bin"
            dataset_path.write_bytes(payload)
            with self.assertRaises(ValueError):
                report.read_dataset_feature_points(dataset_path)


if __name__ == "__main__":
    unittest.main()
