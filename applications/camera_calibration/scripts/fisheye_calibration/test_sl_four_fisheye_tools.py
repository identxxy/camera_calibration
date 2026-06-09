#!/usr/bin/env python3
"""Focused tests for SL four-fisheye utility helpers."""

import importlib.util
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_module(name):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract = load_module("extract_foxglove_h264_from_mcap")
estimate = load_module("estimate_sl_kb8_rig")
optimize_phi = load_module("optimize_sl_up_down_phi")
render = load_module("render_sl_kb8_panoramas")


def ros1_string(value):
    data = value.encode("utf-8")
    return struct.pack("<I", len(data)) + data


def minimal_camchain():
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    out = {}
    for key, name in (
        ("cam0", "left_up"),
        ("cam1", "left_down"),
        ("cam2", "right_down"),
        ("cam3", "right_up"),
    ):
        out[key] = {
            "name": name,
            "camera_model": "kb8",
            "distortion_model": "kb8",
            "intrinsics": [480.0, 480.0, 896.0, 896.0],
            "distortion_coeffs": [0.08, -0.01, 0.0, 0.0],
            "resolution": [1792, 1792],
            "T_cam_ref": identity,
            "T_ref_cam": identity,
        }
    return out


class SLFourFisheyeToolsTest(unittest.TestCase):
    def test_parse_foxglove_compressed_video_ros1_payload(self):
        payload = b"\x00\x00\x00\x01h264-nal"
        message = (
            struct.pack("<II", 12, 34)
            + ros1_string("left_up")
            + struct.pack("<I", len(payload))
            + payload
            + ros1_string("h264")
        )

        decoded = extract.parse_compressed_video_ros1(message)

        self.assertEqual(decoded["timestamp_ns"], 12_000_000_034)
        self.assertEqual(decoded["frame_id"], "left_up")
        self.assertEqual(decoded["format"], "h264")
        self.assertEqual(decoded["data"], payload)

    def test_kb8_projection_keeps_fisheye_back_hemisphere_if_pixel_is_in_image(self):
        intr = {
            "width": 1792,
            "height": 1792,
            "fx": 480.0,
            "fy": 480.0,
            "cx": 896.0,
            "cy": 896.0,
            "k": [0.0, 0.0, 0.0, 0.0],
        }
        dirs = np.asarray([[[0.0, 0.0, -1.0]]], dtype=np.float32)

        _mx, _my, valid, weight = render.kb8_project_dirs(dirs, intr)

        self.assertTrue(bool(valid[0, 0]))
        self.assertAlmostEqual(float(weight[0, 0]), 0.0, places=6)

    def test_phi_metadata_is_disabled_by_default_and_explicit_when_applied(self):
        camchain = minimal_camchain()
        result = {}
        args = SimpleNamespace(optimize_up_down_phi=False)

        unchanged = estimate.maybe_optimize_phi(camchain, result, args)

        self.assertIs(unchanged, camchain)
        self.assertEqual(result["photometric_phi_optimization"], {"enabled": False})
        self.assertNotIn("up_down_phi_z_deg", unchanged["cam1"])

        optimized = optimize_phi.apply_phi_to_camchain(camchain, -0.6, -1.2)
        self.assertAlmostEqual(optimized["cam1"]["up_down_phi_z_deg"], -0.6)
        self.assertAlmostEqual(optimized["cam2"]["up_down_phi_z_deg"], -1.2)


if __name__ == "__main__":
    unittest.main()
