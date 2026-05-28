import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("prepare_fisheye_intrinsics_from_mcap.py")
SPEC = importlib.util.spec_from_file_location("prepare_fisheye_intrinsics_from_mcap", SCRIPT_PATH)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class FramePreparationTest(unittest.TestCase):
    def test_auto_layout_prefers_vertical_four_for_seeker_packed_size(self):
        layout = prep.resolve_layout("auto", width=1088, height=5120, camera_count=4)
        self.assertEqual(layout.name, "vertical4")
        self.assertEqual(layout.tiles, [(0, 0, 1088, 1280),
                                        (0, 1280, 1088, 2560),
                                        (0, 2560, 1088, 3840),
                                        (0, 3840, 1088, 5120)])

    def test_laplacian_sharpness_distinguishes_edge_image_from_flat_image(self):
        flat = np.full((32, 32), 128, dtype=np.uint8)
        edge = flat.copy()
        edge[:, 16:] = 255
        self.assertEqual(prep.laplacian_variance(flat), 0.0)
        self.assertGreater(prep.laplacian_variance(edge), 1000.0)

    def test_motion_gate_uses_board_centroid_and_scale(self):
        selector = prep.CameraFrameSelector(min_sharpness=10.0, min_tags=1, min_board_motion_px=30.0)
        first = prep.FrameMetrics(sharpness=20.0, tag_count=2, board_cx=100.0,
                                  board_cy=100.0, board_area=400.0)
        near = prep.FrameMetrics(sharpness=20.0, tag_count=2, board_cx=110.0,
                                 board_cy=105.0, board_area=420.0)
        far = prep.FrameMetrics(sharpness=20.0, tag_count=2, board_cx=160.0,
                                board_cy=100.0, board_area=400.0)

        self.assertTrue(selector.should_select(first, width=640, height=480).selected)
        self.assertFalse(selector.should_select(near, width=640, height=480).selected)
        self.assertTrue(selector.should_select(far, width=640, height=480).selected)


if __name__ == "__main__":
    unittest.main()
