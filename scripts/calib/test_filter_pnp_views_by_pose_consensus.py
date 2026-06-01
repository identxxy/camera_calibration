#!/usr/bin/env python3
"""Tests for per-frame PnP pose consensus filtering."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import filter_pnp_views_by_pose_consensus as consensus  # noqa: E402


class FilterPnpViewsByPoseConsensusTest(unittest.TestCase):
    def test_keeps_dominant_same_frame_pose_cluster_and_rejects_outlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pose_yaml = root / "camera_tr_rig.yaml"
            pose_yaml.write_text(
                "\n".join([
                    "pose_count: 3",
                    "poses:",
                    "  - index: 0",
                    "    tx: 0",
                    "    ty: 0",
                    "    tz: 0",
                    "    qx: 0",
                    "    qy: 0",
                    "    qz: 0",
                    "    qw: 1",
                    "  - index: 1",
                    "    tx: 0",
                    "    ty: 0",
                    "    tz: 0",
                    "    qx: 0",
                    "    qy: 0",
                    "    qz: 0",
                    "    qw: 1",
                    "  - index: 2",
                    "    tx: 0",
                    "    ty: 0",
                    "    tz: 0",
                    "    qx: 0",
                    "    qy: 0",
                    "    qz: 0",
                    "    qw: 1",
                    "",
                ]),
                encoding="utf-8")
            pnp_views = root / "pnp_views.tsv"
            fieldnames = [
                "imageset_index", "filename", "camera_index", "status",
                "median_error_px", "points", "inliers",
                "tx", "ty", "tz", "qx", "qy", "qz", "qw",
            ]
            with pnp_views.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "imageset_index": 0, "filename": "000000.jpg", "camera_index": 0,
                    "status": "solved", "median_error_px": 1.0, "points": 24, "inliers": 24,
                    "tx": 0.0, "ty": 0.0, "tz": 1.0,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                })
                writer.writerow({
                    "imageset_index": 0, "filename": "000000.jpg", "camera_index": 1,
                    "status": "solved", "median_error_px": 1.5, "points": 24, "inliers": 24,
                    "tx": 0.05, "ty": 0.0, "tz": 1.0,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                })
                writer.writerow({
                    "imageset_index": 0, "filename": "000000.jpg", "camera_index": 2,
                    "status": "solved", "median_error_px": 0.5, "points": 24, "inliers": 24,
                    "tx": 3.0, "ty": 0.0, "tz": 1.0,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                })

            output = root / "pnp_views_consensus.tsv"
            args = type("Args", (), {
                "pnp_views": pnp_views,
                "camera_prior_pose_yaml": pose_yaml,
                "output_pnp_views": output,
                "summary_json": root / "summary.json",
                "per_frame_tsv": root / "per_frame.tsv",
                "center_threshold_m": 0.2,
                "rotation_threshold_deg": 5.0,
                "max_median_error_px": 8.0,
                "min_points": 0,
                "min_inliers": 0,
                "min_consensus_votes": 1,
            })()

            consensus.run(args)

            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))

            self.assertEqual(rows[0]["status"], "solved")
            self.assertEqual(rows[1]["status"], "solved")
            self.assertEqual(rows[2]["status"], "rejected_pose_consensus")

    def test_rejects_weak_low_inlier_pnp_view_before_consensus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pose_yaml = root / "camera_tr_rig.yaml"
            pose_yaml.write_text(
                "\n".join([
                    "pose_count: 2",
                    "poses:",
                    "  - index: 0",
                    "    tx: 0",
                    "    ty: 0",
                    "    tz: 0",
                    "    qx: 0",
                    "    qy: 0",
                    "    qz: 0",
                    "    qw: 1",
                    "  - index: 1",
                    "    tx: 0",
                    "    ty: 0",
                    "    tz: 0",
                    "    qx: 0",
                    "    qy: 0",
                    "    qz: 0",
                    "    qw: 1",
                    "",
                ]),
                encoding="utf-8")
            pnp_views = root / "pnp_views.tsv"
            fieldnames = [
                "imageset_index", "filename", "camera_index", "status",
                "median_error_px", "points", "inliers",
                "tx", "ty", "tz", "qx", "qy", "qz", "qw",
            ]
            with pnp_views.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "imageset_index": 0, "filename": "000000.jpg", "camera_index": 0,
                    "status": "solved", "median_error_px": 1.0, "points": 24, "inliers": 20,
                    "tx": 0.0, "ty": 0.0, "tz": 1.0,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                })
                writer.writerow({
                    "imageset_index": 0, "filename": "000000.jpg", "camera_index": 1,
                    "status": "solved", "median_error_px": 1.0, "points": 4, "inliers": 4,
                    "tx": 0.02, "ty": 0.0, "tz": 1.0,
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                })

            output = root / "pnp_views_consensus.tsv"
            args = type("Args", (), {
                "pnp_views": pnp_views,
                "camera_prior_pose_yaml": pose_yaml,
                "output_pnp_views": output,
                "summary_json": root / "summary.json",
                "per_frame_tsv": None,
                "center_threshold_m": 0.2,
                "rotation_threshold_deg": 5.0,
                "max_median_error_px": 8.0,
                "min_points": 16,
                "min_inliers": 16,
                "min_consensus_votes": 1,
            })()

            consensus.run(args)

            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))

            self.assertEqual(rows[0]["status"], "solved")
            self.assertEqual(rows[1]["status"], "rejected_pose_consensus")


if __name__ == "__main__":
    unittest.main()
