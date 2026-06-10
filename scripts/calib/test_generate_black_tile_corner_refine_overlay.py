#!/usr/bin/env python3
"""Tests for black tile corner refinement diagnostics."""

from pathlib import Path
import importlib.util
import sys
import unittest


try:
    import cv2  # noqa: F401
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False
    np = None


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = SCRIPT_DIR / "apriltag_tower_black_tile_refine.py"


@unittest.skipIf(not HAS_CV2, "OpenCV is required for corner refinement tests")
class BlackTileCornerRefineOverlayTest(unittest.TestCase):
    def load_module(self):
        spec = importlib.util.spec_from_file_location("black_tile_refine", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_edge_support_finds_red_box_scaled_tile_corners(self):
        module = self.load_module()
        image = np.full((240, 240), 255, dtype=np.uint8)
        cv2.rectangle(image, (73, 73), (167, 167), 0, thickness=-1)
        detector_corners = np.asarray([
            [80.0, 160.0],
            [80.0, 80.0],
            [160.0, 80.0],
            [160.0, 160.0],
        ], dtype=np.float64)

        refined = module.refine_scaled_detector_box(
            cv2,
            image,
            detector_corners,
            {"method": "red-scale-edge"})

        self.assertEqual(refined["valid_corner_count"], 4)
        for scale in refined["corner_scales"]:
            self.assertAlmostEqual(scale, 1.175, delta=0.05)

    def test_refined_detections_keep_partial_corners_without_fabricating_points(self):
        module = self.load_module()
        image = np.full((240, 240), 255, dtype=np.uint8)
        cv2.rectangle(image, (73, 73), (167, 167), 0, thickness=-1)
        detections = [{
            "tag_id": 5,
            "corners": [[80.0, 160.0], [80.0, 80.0], [160.0, 80.0], [160.0, 160.0]],
        }]

        refined = module.refine_detections(cv2, image, detections, {"method": "red-scale-edge"})

        self.assertEqual(refined[0]["black_tile_valid_corner_count"], 4)
        self.assertEqual(len(refined[0]["corners"]), 4)
        self.assertTrue(all(corner is not None for corner in refined[0]["corners"]))

    def test_tiny_roi_near_image_boundary_does_not_raise(self):
        module = self.load_module()
        image = np.full((40, 40), 255, dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (12, 12), 0, thickness=-1)
        detector_corners = np.asarray([
            [2.0, 12.0],
            [2.0, 2.0],
            [12.0, 2.0],
            [12.0, 12.0],
        ], dtype=np.float64)

        refined = module.refine_scaled_detector_box(
            cv2,
            image,
            detector_corners,
            {"method": "red-scale-edge"})

        self.assertIn("valid_corner_count", refined)


if __name__ == "__main__":
    unittest.main()
