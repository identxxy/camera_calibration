#!/usr/bin/env python3
"""Focused tests for distributed AprilTag quality filtering."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/distributed_apriltag_quality_filter.py"
SPEC = importlib.util.spec_from_file_location("distributed_apriltag_quality_filter", SCRIPT)
FILTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILTER)


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class DistributedAprilTagQualityFilterTest(unittest.TestCase):
    def test_edge_line_refinement_moves_corners_toward_binary_square_edges(self):
        import numpy as np

        image = np.full((80, 80), 255, dtype=np.uint8)
        image[20:60, 20:60] = 0
        detections = [{
            "tag_id": 9,
            "corners": [
                [17.8, 22.1],
                [57.5, 17.6],
                [61.9, 57.7],
                [22.2, 62.3],
            ],
        }]
        expected = np.asarray([
            [19.5, 19.5],
            [59.5, 19.5],
            [59.5, 59.5],
            [19.5, 59.5],
        ], dtype=np.float32)
        before = np.asarray(detections[0]["corners"], dtype=np.float32)

        refined = FILTER.refine_detections_edge_lines(
            image,
            detections,
            search_radius_px=5.0,
            sample_spacing_px=2.0,
            gradient_step_px=1.0,
            min_gradient=20.0,
            min_edge_points=8,
            max_shift_px=6.0)
        after = np.asarray(refined[0]["corners"], dtype=np.float32)

        self.assertTrue(refined[0]["edge_line_refined"])
        self.assertLess(
            float(np.linalg.norm(after - expected, axis=1).mean()),
            float(np.linalg.norm(before - expected, axis=1).mean()))

    def test_subpixel_refinement_updates_detection_corners(self):
        class FakeAruco:
            CORNER_REFINE_NONE = 0
            CORNER_REFINE_SUBPIX = 1

        class FakeCv2:
            TERM_CRITERIA_EPS = 1
            TERM_CRITERIA_MAX_ITER = 2
            aruco = FakeAruco()

            @staticmethod
            def cornerSubPix(_image, corners, _window, _zero_zone, _criteria):
                refined = corners.copy()
                refined[:, 0, 0] += 0.25
                refined[:, 0, 1] -= 0.5
                return refined

        refined = FILTER.refine_detections_subpixel(
            FakeCv2,
            [[0] * 16 for _ in range(16)],
            [{
                "tag_id": 7,
                "corners": [[1.0, 2.0], [3.0, 4.0]],
            }],
            window_size=5,
            max_iterations=30,
            epsilon=0.01)

        self.assertEqual(refined[0]["tag_id"], 7)
        self.assertTrue(refined[0]["subpixel_refined"])
        self.assertEqual(refined[0]["corners"], [[1.25, 1.5], [3.25, 3.5]])

    def test_tower_config_expands_valid_ids(self):
        config = FILTER.parse_simple_yaml(
            REPO_ROOT / "applications/camera_calibration/patterns/apriltag_tower_8faces_2x16_8cm.yaml")
        valid_ids = FILTER.tower_valid_tag_ids(config)
        self.assertEqual(len(valid_ids), 256)
        self.assertIn(0, valid_ids)
        self.assertIn(255, valid_ids)
        self.assertEqual(FILTER.machine_for_metric({"worker_id": "w4_fullres"}, "4-3"), "w4_D")

    def test_aggregate_writes_synchronized_filtered_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_root = root / "source"
            worker = root / "worker_w4"
            out = root / "filtered_whole"
            time_id = "2026_05_26-14_08_40"
            cameras = ["1-1", "1-2"]

            manifest_rows = []
            metric_rows = []
            for index, camera_id in enumerate(cameras):
                src_dir = src_root / "w4_D/output/calib/whole" / time_id / camera_id
                src_dir.mkdir(parents=True)
                for frame_id in range(3):
                    (src_dir / f"{camera_id}_{frame_id:04d}.jpg").write_text(
                        f"{camera_id} {frame_id}\n", encoding="utf-8")
                manifest_rows.append({
                    "camera_index": index,
                    "stage_name": f"cam{index:02d}_w4_{camera_id}",
                    "machine": "w4_D",
                    "camera_id": camera_id,
                    "source_dir": str(src_dir),
                    "frame_count": 3,
                })

            tag_counts = {
                ("1-1", 0): 4,
                ("1-1", 1): 2,
                ("1-1", 2): 5,
                ("1-2", 0): 4,
                ("1-2", 1): 5,
                ("1-2", 2): 0,
            }
            for camera_id in cameras:
                for frame_id in range(3):
                    filename = f"{camera_id}_{frame_id:04d}.jpg"
                    metric_rows.append({
                        "worker_id": "w4",
                        "time": time_id,
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "filename": filename,
                        "image_path": str(
                            src_root
                            / "w4_D/output/calib/whole"
                            / time_id
                            / camera_id
                            / filename),
                        "decode_ok": "1",
                        "tag_count": tag_counts[(camera_id, frame_id)],
                        "corner_count": tag_counts[(camera_id, frame_id)] * 4,
                    })

            worker.mkdir()
            base_manifest = root / "manifest.tsv"
            write_tsv(
                base_manifest,
                manifest_rows,
                ["camera_index", "stage_name", "machine", "camera_id", "source_dir", "frame_count"],
            )
            write_tsv(
                worker / "per_image_metrics.tsv",
                metric_rows,
                [
                    "worker_id", "time", "camera_id", "frame_id", "filename",
                    "image_path", "decode_ok", "tag_count", "corner_count",
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "aggregate",
                    "--worker-output", str(worker),
                    "--base-manifest", str(base_manifest),
                    "--output-dir", str(out),
                    "--time", time_id,
                    "--min-tags", "4",
                    "--min-cameras-per-frame", "2",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            filtered_manifest = read_tsv(out / "manifest.tsv")
            self.assertEqual([row["camera_id"] for row in filtered_manifest], cameras)
            self.assertEqual([row["frame_count"] for row in filtered_manifest], ["1", "1"])
            self.assertIn("original_source_dir", filtered_manifest[0])
            image_dirs = (out / "image_directories.txt").read_text(encoding="utf-8").strip().split(",")
            self.assertEqual(len(image_dirs), 2)
            for image_dir in image_dirs:
                files = sorted(Path(image_dir).iterdir())
                self.assertEqual([path.name for path in files], ["000000.jpg"])
                self.assertTrue(files[0].is_symlink())

            frames = read_tsv(out / "selected_frames.tsv")
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["frame_id"], "0")
            self.assertEqual(frames[0]["passing_camera_count"], "2")

            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_frame_count"], 1)
            self.assertEqual(summary["camera_count"], 2)
            self.assertEqual(summary["selection"]["min_tags"], 4)
            self.assertEqual(summary["selected_passing_camera_count_histogram"], {"2": 1})

    def test_dry_run_detect_lists_camera_time_without_cv2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "whole"
            image_dir = input_root / "T0" / "1-1"
            image_dir.mkdir(parents=True)
            (image_dir / "1-1_0000.jpg").write_text("not a real jpeg\n", encoding="utf-8")
            out = root / "worker_out"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "detect",
                    "--input-root", str(input_root),
                    "--output-dir", str(out),
                    "--worker-id", "w4",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            rows = read_tsv(out / "dry_run_images.tsv")
            self.assertEqual(rows[0]["time"], "T0")
            self.assertEqual(rows[0]["camera_id"], "1-1")
            self.assertEqual(rows[0]["image_count"], "1")

    def test_aggregate_multiple_times_without_base_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "worker_w4"
            out = root / "filtered_whole"
            camera_id = "1-1"
            metric_rows = []
            for time_id in ["T0", "T1"]:
                src_dir = root / "source" / time_id / camera_id
                src_dir.mkdir(parents=True)
                filename = f"{camera_id}_0000.jpg"
                image_path = src_dir / filename
                image_path.write_text(f"{time_id}\n", encoding="utf-8")
                metric_rows.append({
                    "worker_id": "w4",
                    "time": time_id,
                    "camera_id": camera_id,
                    "frame_id": "0",
                    "filename": filename,
                    "image_path": str(image_path),
                    "decode_ok": "1",
                    "tag_count": "4",
                    "corner_count": "16",
                })

            worker.mkdir()
            write_tsv(
                worker / "per_image_metrics.tsv",
                metric_rows,
                [
                    "worker_id", "time", "camera_id", "frame_id", "filename",
                    "image_path", "decode_ok", "tag_count", "corner_count",
                ],
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "aggregate",
                    "--worker-output", str(worker),
                    "--output-dir", str(out),
                    "--time", "T0,T1",
                    "--min-tags", "4",
                    "--min-cameras-per-frame", "1",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            frames = read_tsv(out / "selected_frames.tsv")
            self.assertEqual([(row["time"], row["frame_id"]) for row in frames], [("T0", "0"), ("T1", "0")])
            self.assertEqual([row["out_frame"] for row in frames], ["0", "1"])
            image_dir = Path((out / "image_directories.txt").read_text(encoding="utf-8").strip())
            self.assertEqual([path.name for path in sorted(image_dir.iterdir())], ["000000.jpg", "000001.jpg"])
            linked_text = [
                path.resolve().read_text(encoding="utf-8")
                for path in sorted(image_dir.iterdir())
            ]
            self.assertEqual(linked_text, ["T0\n", "T1\n"])
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["times"], ["T0", "T1"])
            self.assertEqual(summary["selected_frame_count"], 2)
            self.assertEqual(summary["selected_passing_camera_count_histogram"], {"1": 2})

    def test_aggregate_uses_best_metric_without_duplicate_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "T0" / "4-3"
            source.mkdir(parents=True)
            image_path = source / "4-3_0000.jpg"
            image_path.write_text("frame\n", encoding="utf-8")
            base_manifest = root / "manifest.tsv"
            write_tsv(
                base_manifest,
                [{
                    "camera_index": 0,
                    "stage_name": "cam0",
                    "machine": "w4_D",
                    "camera_id": "4-3",
                    "source_dir": str(source),
                    "frame_count": 0,
                }],
                ["camera_index", "stage_name", "machine", "camera_id", "source_dir", "frame_count"],
            )
            for worker_name, tag_count in [("halfres", 1), ("fullres", 6)]:
                worker = root / worker_name
                worker.mkdir()
                write_tsv(
                    worker / "per_image_metrics.tsv",
                    [{
                        "worker_id": worker_name,
                        "time": "T0",
                        "camera_id": "4-3",
                        "frame_id": "0",
                        "filename": image_path.name,
                        "image_path": str(image_path),
                        "decode_ok": "1",
                        "tag_count": str(tag_count),
                        "corner_count": str(tag_count * 4),
                    }],
                    [
                        "worker_id", "time", "camera_id", "frame_id", "filename",
                        "image_path", "decode_ok", "tag_count", "corner_count",
                    ],
                )

            out = root / "filtered"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "aggregate",
                    "--worker-output", str(root / "halfres"),
                    "--worker-output", str(root / "fullres"),
                    "--base-manifest", str(base_manifest),
                    "--output-dir", str(out),
                    "--time", "T0",
                    "--min-tags", "4",
                    "--min-cameras-per-frame", "1",
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            passing = read_tsv(out / "images_min4.tsv")
            self.assertEqual(len(passing), 1)
            self.assertEqual(passing[0]["worker_id"], "fullres")
            self.assertEqual(passing[0]["tag_count"], "6")
            camera_stats = read_tsv(out / "per_camera_stats.tsv")
            self.assertEqual(camera_stats[0]["total_images"], "1")
            self.assertEqual(camera_stats[0]["passing_images"], "1")
            self.assertEqual(camera_stats[0]["total_tags"], "6")
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metric_row_count"], 2)
            self.assertEqual(summary["chosen_metric_row_count"], 1)

    def test_aggregate_can_require_all_base_manifest_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "worker"
            out = root / "filtered_whole"
            worker.mkdir()
            base_manifest = root / "manifest.tsv"
            write_tsv(
                base_manifest,
                [
                    {"camera_index": 0, "stage_name": "cam0", "machine": "w4_D", "camera_id": "1-1", "source_dir": "", "frame_count": 0},
                    {"camera_index": 1, "stage_name": "cam1", "machine": "w4_D", "camera_id": "1-2", "source_dir": "", "frame_count": 0},
                ],
                ["camera_index", "stage_name", "machine", "camera_id", "source_dir", "frame_count"],
            )
            write_tsv(
                worker / "per_image_metrics.tsv",
                [{
                    "worker_id": "w4",
                    "time": "T0",
                    "camera_id": "1-1",
                    "frame_id": "0",
                    "filename": "1-1_0000.jpg",
                    "image_path": str(root / "missing.jpg"),
                    "decode_ok": "1",
                    "tag_count": "4",
                    "corner_count": "16",
                }],
                [
                    "worker_id", "time", "camera_id", "frame_id", "filename",
                    "image_path", "decode_ok", "tag_count", "corner_count",
                ],
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "aggregate",
                    "--worker-output", str(worker),
                    "--base-manifest", str(base_manifest),
                    "--require-all-base-manifest-cameras",
                    "--output-dir", str(out),
                    "--allow-missing-source",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("1-2", completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
