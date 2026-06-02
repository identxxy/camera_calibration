#!/usr/bin/env python3
"""Focused tests for calibration correspondence residual TSV export."""

import csv
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/export_calibration_correspondence_residuals.py"


def write_dataset(path):
    payload = bytearray()
    payload += b"calib_data"
    payload += struct.pack(">I", 1)
    payload += struct.pack(">I", 1)
    payload += struct.pack(">II", 640, 480)
    payload += struct.pack(">I", 1)
    filename = b"frame_000123.jpg"
    payload += struct.pack(">I", len(filename))
    payload += filename
    payload += struct.pack(">I", 1)
    payload += struct.pack("<ff", 102.0, 203.0)
    payload += struct.pack(">i", 7)
    payload += struct.pack(">I", 0)
    path.write_bytes(payload)


def write_pose_yaml(path, pose_count=1):
    path.write_text(
        f"pose_count: {pose_count}\n"
        "poses:\n"
        "  - index: 0\n"
        "    tx: 0.0\n"
        "    ty: 0.0\n"
        "    tz: 0.0\n"
        "    qx: 0.0\n"
        "    qy: 0.0\n"
        "    qz: 0.0\n"
        "    qw: 1.0\n",
        encoding="utf-8",
    )


class ExportCalibrationCorrespondenceResidualsTest(unittest.TestCase):
    def test_exports_feature_level_reprojection_residuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.bin"
            state_dir = root / "state"
            output_tsv = root / "correspondence_residuals.tsv"
            state_dir.mkdir()
            write_dataset(dataset)
            write_pose_yaml(state_dir / "rig_tr_global.yaml")
            write_pose_yaml(state_dir / "camera_tr_rig.yaml")
            (state_dir / "points.yaml").write_text(
                "points: [0.1, 0.2, 2.0]\n"
                "feature_id_to_point_index:\n"
                "  - feature_id: 7\n"
                "    point_index: 0\n",
                encoding="utf-8",
            )
            (state_dir / "intrinsics0.yaml").write_text(
                "type: CentralOpenCVModel\n"
                "width: 640\n"
                "height: 480\n"
                "parameters: [10, 20, 100, 200, 0, 0, 0, 0, 0, 0, 0, 0]\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.tsv"
            manifest.write_text(
                "camera_index\tstage_name\tmachine\tcamera_id\tframe_count\n"
                "0\tcam0\tmachine0\tcamA\t1\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset", str(dataset),
                    "--state-dir", str(state_dir),
                    "--manifest", str(manifest),
                    "--dataset-name", "unit_dataset",
                    "--output-tsv", str(output_tsv),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            with output_tsv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["dataset"], "unit_dataset")
            self.assertEqual(row["imageset_index"], "0")
            self.assertEqual(row["frame_index"], "123")
            self.assertEqual(row["camera_index"], "0")
            self.assertEqual(row["camera_label"], "camA")
            self.assertEqual(row["filename"], "frame_000123.jpg")
            self.assertEqual(row["feature_id"], "7")
            self.assertEqual(row["point_index"], "0")
            self.assertEqual(row["world_x"], "0.1")
            self.assertEqual(row["world_y"], "0.2")
            self.assertEqual(row["world_z"], "2")
            self.assertEqual(row["observed_x"], "102")
            self.assertEqual(row["observed_y"], "203")
            self.assertEqual(row["projected_x"], "100.5")
            self.assertEqual(row["projected_y"], "202")
            self.assertEqual(row["residual_x_px"], "-1.5")
            self.assertEqual(row["residual_y_px"], "-1")
            self.assertTrue(math.isclose(float(row["residual_px"]), math.sqrt(3.25)))
            self.assertEqual(row["projection_status"], "ok")


if __name__ == "__main__":
    unittest.main()
