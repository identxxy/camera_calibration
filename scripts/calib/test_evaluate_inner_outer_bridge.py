#!/usr/bin/env python3
"""Focused tests for inner/outer bridge quality gates."""

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_inner_outer_bridge as bridge  # noqa: E402


class EvaluateInnerOuterBridgeTest(unittest.TestCase):
    def test_outer_final_alignment_transforms_whole_outer_rig_into_inner_frame(self):
        rotation = np.eye(3)
        outer_rig_tr_inner_rig = bridge.pose_matrix(rotation, np.asarray([0.5, -0.25, 0.1]))
        outer_camera_tr_rig = [
            bridge.pose_matrix(rotation, np.asarray([0.0, 0.0, 0.0])),
            bridge.pose_matrix(rotation, np.asarray([1.0, 0.0, 0.0])),
            bridge.pose_matrix(rotation, np.asarray([0.0, 1.0, 0.0])),
        ]
        summaries = []
        for index, label in enumerate(["4-1", "4-2", "4-3"]):
            camera_tr_inner_rig = outer_camera_tr_rig[index] @ outer_rig_tr_inner_rig
            summaries.append({
                "label": label,
                "camera_tr_inner_rig": camera_tr_inner_rig.tolist(),
            })

        alignment = bridge.build_outer_final_alignment(
            summaries,
            outer_camera_tr_rig,
            [0, 1, 2],
        )

        self.assertEqual(alignment["status"], "ready")
        self.assertLess(alignment["center_residual_max_m"], 1e-9)
        self.assertLess(alignment["rotation_residual_max_deg"], 1e-9)
        np.testing.assert_allclose(
            np.asarray(alignment["outer_rig_tr_inner_rig"]),
            outer_rig_tr_inner_rig,
            atol=1e-9,
        )

    def test_metric_bridge_gate_passes_while_colmap_diagnostic_can_be_weak(self):
        summary = {
            "inner_board_pose_summary": {
                "frame_count": 233,
                "inner_support_median": 3.0,
            },
            "outer_camera_summaries": [
                {"vote_count": 122, "center_residual_p90_m": 0.156, "rotation_residual_p90_deg": 2.84},
                {"vote_count": 158, "center_residual_p90_m": 0.135, "rotation_residual_p90_deg": 2.96},
                {"vote_count": 75, "center_residual_p90_m": 0.127, "rotation_residual_p90_deg": 1.59},
            ],
            "colmap_alignment": {
                "inner_triangle_area_m2": 0.248,
                "rotation_residual_median_deg": 89.0,
                "pairwise_distances": [
                    {"distance_ratio_colmap_per_meter": 0.72},
                    {"distance_ratio_colmap_per_meter": 2.03},
                    {"distance_ratio_colmap_per_meter": 3.20},
                ],
                "per_camera": {
                    "4-1": {"triangulated_point_count": 40},
                    "4-2": {"triangulated_point_count": 192},
                    "4-3": {"triangulated_point_count": 5},
                },
            },
        }
        args = type("Args", (), {
            "bridge_gate_min_inner_frames": 50,
            "bridge_gate_min_inner_support_median": 3.0,
            "min_outer_votes": 10,
            "bridge_gate_max_center_p90_m": 0.25,
            "bridge_gate_max_rotation_p90_deg": 5.0,
            "bridge_gate_min_triangle_area_m2": 0.02,
            "colmap_gate_min_triangulated_tracks": 30,
            "colmap_gate_max_pairwise_ratio_spread": 2.0,
            "colmap_gate_max_rotation_residual_median_deg": 15.0,
        })()

        gates = bridge.build_quality_gates(summary, args)

        self.assertEqual(gates["metric_bridge"]["status"], "pass")
        self.assertEqual(gates["colmap_prior_diagnostic"]["status"], "weak_or_inconsistent")
        self.assertIn(
            "min_colmap_triangulated_tracks",
            gates["colmap_prior_diagnostic"]["failed_checks"],
        )
        self.assertIn(
            "colmap_rotation_residual_median",
            gates["colmap_prior_diagnostic"]["failed_checks"],
        )


if __name__ == "__main__":
    unittest.main()
