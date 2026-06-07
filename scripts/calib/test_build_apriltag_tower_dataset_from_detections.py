#!/usr/bin/env python3
"""Tests for building tower calib_data from distributed detection JSONL."""

from pathlib import Path
import csv
import json
import struct
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/calib/build_apriltag_tower_dataset_from_detections.py"


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of calib_data")
    return data


def read_u32(stream):
    return struct.unpack(">I", read_exact(stream, 4))[0]


def read_i32(stream):
    return struct.unpack(">i", read_exact(stream, 4))[0]


def read_f32(stream):
    return struct.unpack("<f", read_exact(stream, 4))[0]


def read_dataset(path):
    with Path(path).open("rb") as stream:
        if read_exact(stream, 10) != b"calib_data":
            raise ValueError("Invalid dataset header")
        version = read_u32(stream)
        camera_count = read_u32(stream)
        image_sizes = [(read_u32(stream), read_u32(stream)) for _ in range(camera_count)]
        imagesets = []
        for _ in range(read_u32(stream)):
            name = read_exact(stream, read_u32(stream)).decode("utf-8")
            features_by_camera = []
            for _camera_index in range(camera_count):
                features = []
                for _feature_index in range(read_u32(stream)):
                    features.append((read_f32(stream), read_f32(stream), read_i32(stream)))
                features_by_camera.append(features)
            imagesets.append({"filename": name, "features": features_by_camera})
        geometry_count = read_u32(stream)
        geometries = []
        for _ in range(geometry_count):
            cell_length = read_f32(stream)
            count_2d = read_u32(stream)
            for _item in range(count_2d):
                read_i32(stream)
                read_i32(stream)
                read_i32(stream)
            count_3d = read_u32(stream)
            for _item in range(count_3d):
                read_i32(stream)
                read_f32(stream)
                read_f32(stream)
                read_f32(stream)
            geometries.append({"cell_length": cell_length, "count_3d": count_3d})
    return {
        "version": version,
        "camera_count": camera_count,
        "image_sizes": image_sizes,
        "imagesets": imagesets,
        "geometries": geometries,
    }


class BuildAprilTagTowerDatasetFromDetectionsTest(unittest.TestCase):
    def test_builds_staged_dataset_without_reading_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_root = root / "stage"
            cam0 = stage_root / "images/cam00_1-1"
            cam1 = stage_root / "images/cam01_1-2"
            cam0.mkdir(parents=True)
            cam1.mkdir(parents=True)

            manifest = stage_root / "manifest.tsv"
            image_dirs = stage_root / "image_directories.txt"
            selected_images = stage_root / "selected_images.tsv"
            tower_config = root / "tower.yaml"
            detections = root / "worker/detections.jsonl"
            output_dataset = root / "tower.bin"
            summary_json = root / "summary.json"
            per_camera_tsv = root / "per_camera.tsv"
            detections_tsv = root / "detections.tsv"

            write_tsv(
                manifest,
                [
                    {
                        "camera_index": 0,
                        "stage_name": "cam00_1-1",
                        "machine": "w4_D",
                        "camera_id": "1-1",
                        "source_dir": str(cam0),
                        "frame_count": 1,
                    },
                    {
                        "camera_index": 1,
                        "stage_name": "cam01_1-2",
                        "machine": "w4_D",
                        "camera_id": "1-2",
                        "source_dir": str(cam1),
                        "frame_count": 1,
                    },
                ],
                ["camera_index", "stage_name", "machine", "camera_id", "source_dir", "frame_count"],
            )
            image_dirs.write_text(f"{cam0},{cam1}\n", encoding="utf-8")
            write_tsv(
                selected_images,
                [
                    {
                        "out_frame": 0,
                        "time": "T0",
                        "frame_id": 42,
                        "frame_key": "T0::42",
                        "camera_index": 0,
                        "camera_id": "1-1",
                        "tag_count": 1,
                        "corner_count": 4,
                        "source": "/orig/T0/1-1/raw_0042.jpg",
                        "filtered_image": str(cam0 / "000000.jpg"),
                    },
                    {
                        "out_frame": 0,
                        "time": "T0",
                        "frame_id": 42,
                        "frame_key": "T0::42",
                        "camera_index": 1,
                        "camera_id": "1-2",
                        "tag_count": 1,
                        "corner_count": 4,
                        "source": "/orig/T0/1-2/raw_0042.jpg",
                        "filtered_image": str(cam1 / "000000.jpg"),
                    },
                ],
                [
                    "out_frame", "time", "frame_id", "frame_key", "camera_index", "camera_id",
                    "tag_count", "corner_count", "source", "filtered_image",
                ],
            )
            tower_config.write_text(
                "\n".join([
                    "tag_family: tag36h11",
                    "faces: 4",
                    "tag_columns: 1",
                    "tag_rows: 1",
                    "tag_size_m: 0.08",
                    "tag_spacing_m: 0.02",
                    "first_tag_id: 10",
                    "face_id_stride: 1",
                    "face_width_m: 0.12",
                    "tag_rotation_degrees: 180",
                    "",
                ]),
                encoding="utf-8",
            )
            detections.parent.mkdir(parents=True)
            records = [
                {
                    "worker_id": "w4",
                    "time": "T0",
                    "camera_id": "1-2",
                    "filename": "raw_0042.jpg",
                    "frame_id": 42,
                    "image_path": "/orig/T0/1-2/raw_0042.jpg",
                    "width": 640,
                    "height": 480,
                    "detections": [
                        {"tag_id": 10, "corners": [[21, 22], [23, 24], [25, 26], [27, 28]]},
                        {"tag_id": 999, "corners": [[1, 1], [2, 2], [3, 3], [4, 4]]},
                    ],
                },
                {
                    "worker_id": "w4",
                    "time": "T0",
                    "camera_id": "1-1",
                    "filename": "raw_0042.jpg",
                    "frame_id": 42,
                    "image_path": "/orig/T0/1-1/raw_0042.jpg",
                    "width": 800,
                    "height": 600,
                    "detections": [
                        {"tag_id": 10, "corners": [[11, 12], [13, 14], [15, 16], [17, 18]]},
                    ],
                },
                {
                    "worker_id": "w4",
                    "time": "T0",
                    "camera_id": "1-1",
                    "filename": "raw_0000.jpg",
                    "frame_id": 0,
                    "image_path": "/orig/T0/1-1/raw_0000.jpg",
                    "width": 800,
                    "height": 600,
                    "detections": [
                        {"tag_id": 10, "corners": [[201, 202], [203, 204], [205, 206], [207, 208]]},
                    ],
                },
                {
                    "worker_id": "w4",
                    "time": "T0",
                    "camera_id": "1-1",
                    "filename": "raw_0007.jpg",
                    "frame_id": 7,
                    "image_path": "/orig/T0/1-1/raw_0007.jpg",
                    "width": 800,
                    "height": 600,
                    "detections": [
                        {"tag_id": 10, "corners": [[101, 102], [103, 104], [105, 106], [107, 108]]},
                    ],
                },
            ]
            detections.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest", str(manifest),
                    "--image-directories-file", str(image_dirs),
                    "--worker-output", str(detections.parent),
                    "--detections-jsonl", str(detections),
                    "--selected-images", str(selected_images),
                    "--output-dataset", str(output_dataset),
                    "--tower-config", str(tower_config),
                    "--summary-json", str(summary_json),
                    "--per-camera-tsv", str(per_camera_tsv),
                    "--detections-tsv", str(detections_tsv),
                ],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            dataset = read_dataset(output_dataset)
            self.assertEqual(dataset["version"], 1)
            self.assertEqual(dataset["camera_count"], 2)
            self.assertEqual(dataset["image_sizes"], [(800, 600), (640, 480)])
            self.assertEqual([item["filename"] for item in dataset["imagesets"]], ["000000.jpg"])
            self.assertEqual(
                dataset["imagesets"][0]["features"][0],
                [(11.0, 12.0, 40), (13.0, 14.0, 41), (15.0, 16.0, 42), (17.0, 18.0, 43)],
            )
            self.assertEqual(
                dataset["imagesets"][0]["features"][1],
                [(21.0, 22.0, 40), (23.0, 24.0, 41), (25.0, 26.0, 42), (27.0, 28.0, 43)],
            )
            self.assertEqual(dataset["geometries"][0]["count_3d"], 16)

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "detections_apriltag_tower_dataset")
            self.assertEqual(summary["filtered_invalid_tags"], 1)
            self.assertEqual(summary["ignored_unstaged_records"], 2)
            self.assertEqual(summary["duplicate_matched_records"], 0)
            self.assertEqual(summary["total_tags"], 2)


if __name__ == "__main__":
    unittest.main()
