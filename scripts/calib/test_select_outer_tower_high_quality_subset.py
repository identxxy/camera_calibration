#!/usr/bin/env python3
"""Tests for outer tower high-quality subset selection."""

import csv
import contextlib
import io
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/select_outer_tower_high_quality_subset.py"
SPEC = importlib.util.spec_from_file_location("select_outer_tower_high_quality_subset", SCRIPT)
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def u32(value):
    return struct.pack(">I", int(value))


def i32(value):
    return struct.pack(">i", int(value))


def f32(value):
    return struct.pack("<f", float(value))


def write_minimal_dataset(path):
    imagesets = [
        ("frame_000.jpg", [[(10.0, 20.0, 0), (11.0, 21.0, 1)], [(30.0, 40.0, 0)]]),
        ("frame_001.jpg", [[(12.0, 22.0, 0)], [(32.0, 42.0, 0)]]),
        ("frame_002.jpg", [[(13.0, 23.0, 2)], [(33.0, 43.0, 2), (34.0, 44.0, 1)]]),
    ]
    with Path(path).open("wb") as stream:
        stream.write(b"calib_data")
        stream.write(u32(1))
        stream.write(u32(2))
        stream.write(u32(640))
        stream.write(u32(480))
        stream.write(u32(800))
        stream.write(u32(600))
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
        stream.write(u32(1))
        stream.write(f32(0.08))
        stream.write(u32(1))
        stream.write(i32(7))
        stream.write(i32(8))
        stream.write(i32(9))
        stream.write(u32(3))
        for feature_id, xyz in {
                0: (0.0, 0.0, 0.0),
                1: (1.0, 0.0, 0.0),
                2: (0.0, 1.0, 0.0),
        }.items():
            stream.write(i32(feature_id))
            for value in xyz:
                stream.write(f32(value))


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class SelectOuterTowerHighQualitySubsetTest(unittest.TestCase):
    def test_filters_dataset_frames_and_remaps_pnp_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.bin"
            refine_dir = root / "refine"
            diagnostics = refine_dir / "diagnostics"
            pnp_views = root / "pnp_views.tsv"
            out = root / "subset"
            diagnostics.mkdir(parents=True)

            write_minimal_dataset(dataset)
            write_tsv(
                diagnostics / "frame_quality.tsv",
                [
                    {"frame_index": 0, "filename": "frame_000.jpg", "feature_count": 3, "active": "yes"},
                    {"frame_index": 1, "filename": "frame_001.jpg", "feature_count": 2, "active": "yes"},
                    {"frame_index": 2, "filename": "frame_002.jpg", "feature_count": 3, "active": "yes"},
                ],
                ["frame_index", "filename", "feature_count", "active"],
            )
            residual_rows = []
            for frame_index, residuals in {
                    0: [(0, 0.2), (0, 0.4), (1, 0.3)],
                    1: [(0, 8.0), (1, 9.0), (1, 10.0)],
                    2: [(0, 0.1), (1, 0.2), (1, 0.5)],
            }.items():
                for camera_index, residual_px in residuals:
                    residual_rows.append({
                        "frame_index": frame_index,
                        "camera_index": camera_index,
                        "residual_px": residual_px,
                        "used_after_gate": "yes",
                        "projection_status": "ok",
                    })
            write_tsv(
                diagnostics / "observation_residuals.tsv",
                residual_rows,
                ["frame_index", "camera_index", "residual_px", "used_after_gate", "projection_status"],
            )
            pnp_fields = [
                "imageset_index", "filename", "camera_index", "status", "median_error_px",
                "points", "inliers", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
            ]
            write_tsv(
                pnp_views,
                [
                    {
                        "imageset_index": 0, "filename": "frame_000.jpg", "camera_index": 0,
                        "status": "solved", "median_error_px": 0.5, "points": 24, "inliers": 24,
                        "tx": 0, "ty": 0, "tz": 1, "qx": 0, "qy": 0, "qz": 0, "qw": 1,
                    },
                    {
                        "imageset_index": 1, "filename": "frame_001.jpg", "camera_index": 0,
                        "status": "solved", "median_error_px": 0.5, "points": 24, "inliers": 24,
                        "tx": 0, "ty": 0, "tz": 1, "qx": 0, "qy": 0, "qz": 0, "qw": 1,
                    },
                    {
                        "imageset_index": 2, "filename": "frame_002.jpg", "camera_index": 1,
                        "status": "solved", "median_error_px": 0.5, "points": 24, "inliers": 24,
                        "tx": 0, "ty": 0, "tz": 1, "qx": 0, "qy": 0, "qz": 0, "qw": 1,
                    },
                ],
                pnp_fields,
            )

            args = type("Args", (), {
                "dataset": dataset,
                "refine_dir": refine_dir,
                "pnp_views": pnp_views,
                "output_dir": out,
                "max_frame_median_px": 1.0,
                "max_frame_p90_px": 1.0,
                "min_frame_observations": 3,
                "min_frame_cameras": 2,
                "limit_frames": 0,
                        "camera_min_observations": 1,
                        "residual_scope": "used_after_gate",
                        "allow_empty": False,
                    })()
            with contextlib.redirect_stdout(io.StringIO()):
                SELECTOR.run(args)

            selected = read_tsv(out / "selected_frames.tsv")
            self.assertEqual([row["old_imageset_index"] for row in selected], ["0", "2"])
            self.assertEqual([row["new_imageset_index"] for row in selected], ["0", "1"])

            subset = SELECTOR.read_dataset(out / "dataset_subset.bin")
            self.assertEqual(subset["version"], 1)
            self.assertEqual(subset["image_sizes"], [(640, 480), (800, 600)])
            self.assertEqual([item["filename"] for item in subset["imagesets"]], ["frame_000.jpg", "frame_002.jpg"])
            self.assertEqual(subset["geometry_blocks"][0]["topology_items"], [(7, 8, 9)])
            self.assertEqual([item[0] for item in subset["geometry_blocks"][0]["known_points"]], [0, 1, 2])

            pnp_subset = read_tsv(out / "pnp_views_subset.tsv")
            self.assertEqual([row["imageset_index"] for row in pnp_subset], ["0", "1"])
            self.assertEqual([row["filename"] for row in pnp_subset], ["frame_000.jpg", "frame_002.jpg"])
            self.assertEqual([row["camera_index"] for row in pnp_subset], ["0", "1"])

            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_frame_count"], 2)
            self.assertEqual(summary["pnp_views"]["kept_rows"], 2)
            self.assertEqual(summary["pnp_views"]["dropped_rows"], 1)


if __name__ == "__main__":
    unittest.main()
