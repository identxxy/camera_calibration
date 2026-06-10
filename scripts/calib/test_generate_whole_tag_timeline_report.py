#!/usr/bin/env python3
"""Focused tests for whole AprilTag timeline report generation."""

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/generate_whole_tag_timeline_report.py"
SPEC = importlib.util.spec_from_file_location("generate_whole_tag_timeline_report", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def u32(value):
    return struct.pack(">I", int(value))


def i32(value):
    return struct.pack(">i", int(value))


def f32(value):
    return struct.pack("<f", float(value))


def write_dataset(path):
    imagesets = [
        ("000000.jpg", [
            [(10.0, 20.0, 0), (30.0, 20.0, 1), (30.0, 40.0, 2), (10.0, 40.0, 3)],
            [],
        ]),
        ("000002.jpg", [
            [(100.0, 120.0, 4), (130.0, 120.0, 5), (130.0, 160.0, 6), (100.0, 160.0, 7)],
            [(200.0, 220.0, 4), (230.0, 220.0, 5), (230.0, 260.0, 6), (200.0, 260.0, 7)],
        ]),
    ]
    with Path(path).open("wb") as stream:
        stream.write(b"calib_data")
        stream.write(u32(1))
        stream.write(u32(2))
        stream.write(u32(640))
        stream.write(u32(480))
        stream.write(u32(640))
        stream.write(u32(480))
        stream.write(u32(len(imagesets)))
        for filename, camera_features in imagesets:
            encoded = filename.encode("utf-8")
            stream.write(u32(len(encoded)))
            stream.write(encoded)
            for features in camera_features:
                stream.write(u32(len(features)))
                for x, y, feature_id in features:
                    stream.write(f32(x))
                    stream.write(f32(y))
                    stream.write(i32(feature_id))
        stream.write(u32(0))


class WholeTagTimelineReportTest(unittest.TestCase):
    def test_report_data_preserves_original_frame_gaps_and_shared_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.bin"
            write_dataset(dataset_path)
            dataset = report.read_dataset(dataset_path)
            data = report.summarize_dataset(dataset, {}, {})

            self.assertEqual(data["frame_range"], [0, 2])
            self.assertEqual(data["frame_count"], 3)
            self.assertEqual(data["frames_with_any_tag"], 2)
            self.assertEqual(data["max_cameras_per_frame"], 2)
            self.assertEqual(data["max_shared_tags_per_frame"], 1)
            self.assertEqual(data["cameras"][0]["frames_with_tags"], 2)
            self.assertEqual(data["cameras"][1]["frames_with_tags"], 1)
            self.assertEqual(data["covis_rows"][1]["camera_count"], 0)

    def test_report_writes_html_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.bin"
            output_dir = root / "report"
            write_dataset(dataset_path)
            dataset = report.read_dataset(dataset_path)
            data = report.summarize_dataset(dataset, {}, {})
            index = report.render_report(data, output_dir, "Test Whole Timeline", dataset_path, "", "")

            self.assertTrue(index.is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            html = index.read_text(encoding="utf-8")
            self.assertIn("Global Co-Visibility Timeline", html)
            self.assertIn("Per-Camera Tag Timeline", html)


if __name__ == "__main__":
    unittest.main()
