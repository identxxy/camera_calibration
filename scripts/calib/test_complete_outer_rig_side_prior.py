#!/usr/bin/env python3
"""Focused tests for outer side-prior completion helpers."""

from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import complete_outer_rig_side_prior as side_prior  # noqa: E402


class CompleteOuterRigSidePriorTest(unittest.TestCase):
    def test_bridge_pose_override_replaces_completed_pose_by_label(self):
        manifest = [
            {"camera_id": "4-1", "camera_index": 0},
            {"camera_id": "4-2", "camera_index": 1},
        ]
        completed = [
            side_prior.base.pose_matrix(side_prior.np.eye(3), [1.0, 0.0, 0.0]),
            side_prior.base.pose_matrix(side_prior.np.eye(3), [2.0, 0.0, 0.0]),
        ]
        theta = side_prior.math.radians(15.0)
        rotation_z = side_prior.np.asarray([
            [side_prior.math.cos(theta), -side_prior.math.sin(theta), 0.0],
            [side_prior.math.sin(theta), side_prior.math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        bridge_pose = side_prior.base.pose_matrix(
            rotation_z,
            [3.0, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            bridge_yaml = Path(tmpdir) / "bridge.yaml"
            side_prior.base.write_pose_yaml(bridge_yaml, [None, bridge_pose])
            rows, updated = side_prior.apply_bridge_pose_overrides(
                completed,
                manifest,
                bridge_yaml,
                {"4-2": 1},
                "4-2")

        self.assertEqual([row["camera_id"] for row in rows], ["4-2"])
        self.assertTrue(side_prior.np.allclose(updated[0], completed[0]))
        self.assertTrue(side_prior.np.allclose(updated[1], bridge_pose))
        self.assertGreater(rows[0]["center_delta_m"], 0.0)
        self.assertGreater(rows[0]["rotation_delta_deg"], 14.0)


if __name__ == "__main__":
    unittest.main()
