#!/usr/bin/env python3
"""Focused tests for the distributed SSH runner report aggregation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import server_run_distributed_clients as runner


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class ServerRunDistributedClientsTest(unittest.TestCase):
    def test_aggregate_reads_worker_summary_and_images_min_tsv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / "worker"
            worker.mkdir()
            (worker / "worker_summary.json").write_text(
                json.dumps({
                    "camera_time_count": 2,
                    "total_images": 12,
                    "passing_images": 3,
                    "elapsed_sec": 4.5,
                }),
                encoding="utf-8")
            write_tsv(
                worker / "per_camera_stats.tsv",
                [{"camera_id": "1-1", "passing_images": "3"}],
                ["camera_id", "passing_images"])
            write_tsv(
                worker / "images_min4.tsv",
                [{"camera_id": "1-1", "frame_id": "0", "tag_count": "4"}],
                ["camera_id", "frame_id", "tag_count"])

            output = root / "out"
            runner.aggregate(
                {"clients": [{
                    "name": "w4",
                    "host": "w4",
                    "local_output_dir": str(worker),
                }]},
                output,
                [{"name": "w4", "host": "w4", "returncode": 0}],
                [])

            status = read_tsv(output / "client_status.tsv")
            self.assertEqual(status[0]["camera_count"], "")
            self.assertEqual(status[0]["total_frames"], "")
            self.assertEqual(status[0]["good_image_count"], "3")
            good = read_tsv(output / "merged_good_images.tsv")
            self.assertEqual(good[0]["client_name"], "w4")
            self.assertEqual(good[0]["tag_count"], "4")


if __name__ == "__main__":
    unittest.main()
