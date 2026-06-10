#!/usr/bin/env python3
"""Regression tests for camera-origin projection reports."""

import csv
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/generate_camera_origin_projection_report.py"
HAS_CV2 = importlib.util.find_spec("cv2") is not None


def write_image(path):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 64, 3), 80, dtype=np.uint8)
    cv2.imwrite(str(path), image)


class CameraOriginProjectionReportTest(unittest.TestCase):
    @unittest.skipUnless(HAS_CV2, "OpenCV Python bindings are required to generate projection overlays")
    def test_studio_yaml_matches_inner_camera_by_serial_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outer_dir = root / "images" / "outer"
            inner_dir = root / "images" / "inner"
            write_image(outer_dir / "1-1_0000.jpg")
            write_image(inner_dir / "22463688_0000.jpg")

            manifest = root / "manifest.tsv"
            manifest.write_text(
                "\n".join([
                    "camera_index\tcamera_id\tstage_name\tmachine\tframe_count\tstatus\tsource_dir",
                    f"0\t1-1\touter_1_1\tw3\t1\tusable\t{outer_dir}",
                    f"99\t22463688\tcam24_w1_22463688\tw1\t1\tusable\t{inner_dir}",
                    "",
                ]),
                encoding="utf-8",
            )

            studio_yaml = root / "studio_32_cameras.yaml"
            studio_yaml.write_text(
                """
cameras:
  - index: 0
    label: "1-1"
    group: "outer"
    camera_id: "1-1"
    intrinsics:
      width: 64
      height: 48
      parameters: [50, 50, 32, 24, 0, 0, 0, 0, 0, 0, 0, 0]
    camera_tr_studio_rig: {tx: 0, ty: 0, tz: 0, qx: 0, qy: 0, qz: 0, qw: 1}
  - index: 24
    label: "inner0"
    group: "inner"
    camera_id: "22463688"
    intrinsics:
      width: 64
      height: 48
      parameters: [50, 50, 32, 24, 0, 0, 0, 0, 0, 0, 0, 0]
    camera_tr_studio_rig: {tx: 0, ty: 0, tz: -2, qx: 0, qy: 0, qz: 0, qw: 1}
""",
                encoding="utf-8",
            )

            output_dir = root / "report"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--studio-yaml", str(studio_yaml),
                    "--camera-group", "all",
                    "--manifest", str(manifest),
                    "--output-dir", str(output_dir),
                    "--frame-index", "0",
                    "--max-image-width", "64",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            with (output_dir / "camera_origin_projections.tsv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            view_labels = {row["view_label"] for row in rows}
            target_labels = {row["target_label"] for row in rows}
            self.assertEqual(view_labels, {"1-1", "inner0"})
            self.assertEqual(target_labels, {"1-1", "inner0"})
            self.assertEqual(len(rows), 2)
            self.assertTrue((output_dir / "images/camera00_1-1_origin_projection.jpg").is_file())
            self.assertTrue((output_dir / "images/camera00_1-1_origin_projection_outer_targets.jpg").is_file())
            self.assertTrue((output_dir / "images/camera00_1-1_origin_projection_inner_targets.jpg").is_file())
            self.assertTrue((output_dir / "images/camera24_22463688_origin_projection.jpg").is_file())
            html = (output_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn('data-mode="all" class="active"', html)
            self.assertIn("Default image mode is <code>all 32 targets</code>", html)


if __name__ == "__main__":
    unittest.main()
