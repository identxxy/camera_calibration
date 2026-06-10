#!/usr/bin/env python3
"""Focused tests for per-frame-face outer tower plane refinement helpers."""

from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import refine_outer_tower_frame_face_planes as refine_ff  # noqa: E402


class RefineOuterTowerFrameFacePlanesTest(unittest.TestCase):
    def test_feature_id_to_face_local_point_uses_exact_2x16_rotated_layout(self):
        layout = refine_ff.default_tower_layout()

        point = refine_ff.face_local_point_for_feature((1 * 32 + 0) * 4 + 0, layout)

        self.assertEqual(refine_ff.face_id_for_feature((1 * 32 + 0) * 4 + 0, layout), 1)
        self.assertTrue(refine_ff.base.np.allclose(point, [0.0, -0.01, -0.79]))

    def test_frame_face_grouping_keeps_faces_independent_within_each_frame(self):
        dataset = {
            "camera_count": 2,
            "image_sizes": [(640, 480), (640, 480)],
            "imagesets": [
                {
                    "filename": "frame000",
                    "features": [
                        [(10.0, 20.0, 0), (11.0, 21.0, 1)],
                        [(12.0, 22.0, 32 * 4)],
                    ],
                },
                {
                    "filename": "frame001",
                    "features": [
                        [(13.0, 23.0, 0)],
                        [(14.0, 24.0, 32 * 4)],
                    ],
                },
            ],
        }

        observations, by_frame_face, by_camera = refine_ff.build_frame_face_observations(
            dataset,
            refine_ff.default_tower_layout())

        self.assertEqual(len(observations), 5)
        self.assertEqual(sorted(by_frame_face), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(len(by_frame_face[(0, 0)]), 2)
        self.assertEqual(len(by_frame_face[(0, 1)]), 1)
        self.assertEqual(len(by_camera[0]), 3)
        self.assertEqual(len(by_camera[1]), 2)

    def test_rigid_yaw45_tower_model_builds_shared_frame_faces(self):
        args = type("Args", (), {
            "tower_face_count": 8,
            "tower_face0_angle_degrees": 0.0,
            "tower_face_width_initial_m": 0.25,
        })()
        poses = refine_ff.frame_face_poses_from_tower_poses(
            {311: refine_ff.base.np.eye(4, dtype=refine_ff.base.np.float64)},
            {(311, 0), (311, 1)},
            args,
            0.0,
        )

        n0 = poses[(311, 0)][:3, 0]
        n1 = poses[(311, 1)][:3, 0]
        z0 = poses[(311, 0)][:3, 2]
        z1 = poses[(311, 1)][:3, 2]
        angle_deg = refine_ff.math.degrees(refine_ff.math.acos(float(refine_ff.base.np.dot(n0, n1))))

        self.assertAlmostEqual(angle_deg, 45.0, places=6)
        self.assertTrue(refine_ff.base.np.allclose(z0, z1))
        self.assertTrue(refine_ff.base.np.allclose(z0, [0.0, 0.0, 1.0]))

    def test_flex_tower_model_applies_bounded_face_yaw_and_offsets(self):
        args = type("Args", (), {
            "tower_model": "flex_yaw_offset_tower",
            "tower_face_count": 8,
            "tower_face0_angle_degrees": 0.0,
            "tower_face_width_initial_m": 0.25,
            "flex_face_yaw_max_deg": 8.0,
            "flex_face_radial_offset_max_m": 0.05,
            "flex_face_tangent_offset_max_m": 0.02,
        })()
        delta = refine_ff.zero_face_geometry_delta(args)
        face_count = args.tower_face_count
        delta[1] = refine_ff.math.radians(5.0)
        delta[face_count + 1] = 0.02
        delta[2 * face_count + 1] = 0.01

        poses = refine_ff.frame_face_poses_from_tower_poses(
            {0: refine_ff.base.np.eye(4, dtype=refine_ff.base.np.float64)},
            {(0, 0), (0, 1)},
            args,
            0.0,
            delta,
        )

        n0 = poses[(0, 0)][:3, 0]
        n1 = poses[(0, 1)][:3, 0]
        angle_deg = refine_ff.math.degrees(refine_ff.math.acos(float(refine_ff.base.np.dot(n0, n1))))
        self.assertAlmostEqual(angle_deg, 50.0, places=6)

        yaw_only = refine_ff.zero_face_geometry_delta(args)
        yaw_only[1] = delta[1]
        yaw_only_face1 = refine_ff.tower_face_transform(
            1,
            0.25,
            args.tower_face_count,
            args.tower_face0_angle_degrees,
            yaw_only,
            args,
        )
        shifted = poses[(0, 1)][:3, 3] - yaw_only_face1[:3, 3]
        shifted_radial = float(refine_ff.base.np.dot(shifted, n1))
        shifted_tangent = float(refine_ff.base.np.dot(shifted, poses[(0, 1)][:3, 1]))
        self.assertAlmostEqual(shifted_radial, 0.02, places=6)
        self.assertAlmostEqual(shifted_tangent, 0.01, places=6)

        over = refine_ff.zero_face_geometry_delta(args)
        over[0] = refine_ff.math.radians(20.0)
        over[face_count] = 0.20
        over[2 * face_count] = -0.20
        clamped = refine_ff.clamp_face_geometry_delta(args, over)
        self.assertAlmostEqual(clamped[0], refine_ff.math.radians(8.0), places=12)
        self.assertAlmostEqual(clamped[face_count], 0.05, places=12)
        self.assertAlmostEqual(clamped[2 * face_count], -0.02, places=12)

    def test_flex_face_geometry_prior_penalizes_invalid_adjacent_separation(self):
        args = type("Args", (), {
            "tower_model": "flex_yaw_offset_tower",
            "tower_face_count": 8,
            "tower_face0_angle_degrees": 0.0,
            "tower_face_width_initial_m": 0.25,
            "flex_face_yaw_max_deg": 8.0,
            "flex_face_radial_offset_max_m": 0.05,
            "flex_face_tangent_offset_max_m": 0.02,
            "flex_face_yaw_sigma_deg": 3.0,
            "flex_face_radial_offset_sigma_m": 0.015,
            "flex_face_tangent_offset_sigma_m": 0.010,
            "flex_face_adjacent_angle_sigma_deg": 1.0,
            "flex_face_adjacent_angle_min_deg": 40.0,
            "flex_face_adjacent_angle_max_deg": 50.0,
        })()
        valid = refine_ff.zero_face_geometry_delta(args)
        invalid = refine_ff.zero_face_geometry_delta(args)
        invalid[0] = refine_ff.math.radians(8.0)
        invalid[1] = refine_ff.math.radians(-8.0)

        valid_prior = refine_ff.face_geometry_prior_residual(args, valid)
        invalid_prior = refine_ff.face_geometry_prior_residual(args, invalid)

        self.assertAlmostEqual(float(refine_ff.base.np.linalg.norm(valid_prior)), 0.0)
        self.assertGreater(float(refine_ff.base.np.linalg.norm(invalid_prior)), 5.0)

    def test_residuals_are_zero_at_ground_truth_frame_face_and_camera_poses(self):
        layout = refine_ff.default_tower_layout()
        intrinsic = {
            "width": 640,
            "height": 480,
            "params": [500.0, 520.0, 320.0, 240.0] + [0.0] * 8,
        }
        camera_poses = [
            refine_ff.base.np.eye(4, dtype=refine_ff.base.np.float64),
            refine_ff.base.pose_matrix(refine_ff.base.np.eye(3), [0.05, 0.0, 0.0]),
        ]
        rig_tr_plane = refine_ff.base.pose_matrix(refine_ff.base.np.eye(3), [0.0, 0.0, 2.0])
        features_by_camera = []
        for camera_pose in camera_poses:
            camera_features = []
            for feature_id in [0, 1, 2, 3, 4, 5, 6, 7]:
                local = refine_ff.face_local_point_for_feature(feature_id, layout)
                point_rig = rig_tr_plane[:3, :3] @ local + rig_tr_plane[:3, 3]
                point_camera = camera_pose[:3, :3] @ point_rig + camera_pose[:3, 3]
                xy = refine_ff.base.project_point(point_camera, intrinsic)
                camera_features.append((float(xy[0]), float(xy[1]), feature_id))
            features_by_camera.append(camera_features)
        dataset = {
            "camera_count": 2,
            "image_sizes": [(640, 480), (640, 480)],
            "imagesets": [{"filename": "frame000", "features": features_by_camera}],
        }
        observations, _by_frame_face, _by_camera = refine_ff.build_frame_face_observations(
            dataset,
            layout)

        residuals = refine_ff.projection_residuals(
            observations,
            camera_poses,
            {(0, 0): rig_tr_plane},
            [intrinsic, intrinsic])

        self.assertLess(float(refine_ff.base.np.linalg.norm(residuals)), 1e-9)

    def test_optimize_bundle_uses_initial_camera_deltas(self):
        args = type("Args", (), {
            "intrinsics_refine_mode": "fixed",
            "min_camera_observations_for_delta": 1,
            "delta_rotation_sigma_deg": 3.0,
            "delta_translation_sigma_m": 0.12,
            "camera_delta_max_rotation_step_deg": 1.0,
            "camera_delta_max_translation_step_m": 0.03,
            "frame_face_max_rotation_step_deg": 5.0,
            "frame_face_max_translation_step_m": 0.10,
            "outer_iterations": 0,
            "block_iterations": 1,
            "optimizer_residual_clip_px": 30.0,
            "intrinsics_focal_sigma_frac": 0.01,
            "intrinsics_principal_sigma_px": 8.0,
            "intrinsics_distortion_sigma": 0.05,
            "intrinsics_max_focal_step_frac": 0.002,
            "intrinsics_max_principal_step_px": 1.0,
            "intrinsics_max_distortion_step": 0.01,
            "intrinsics_block_iterations": 1,
            "intrinsics_max_total_focal_delta_frac": 0.0,
            "intrinsics_max_total_principal_delta_px": 0.0,
            "intrinsics_max_total_distortion_delta": 0.0,
        })()
        prior = refine_ff.base.np.eye(4, dtype=refine_ff.base.np.float64)
        initial_delta = refine_ff.base.pose_matrix(refine_ff.base.np.eye(3), [0.12, -0.03, 0.04])

        optimized = refine_ff.optimize_bundle(
            {},
            [[]],
            [prior],
            {},
            [{"width": 640, "height": 480, "params": [500.0, 500.0, 320.0, 240.0] + [0.0] * 8}],
            args,
            initial_deltas=[initial_delta],
        )

        self.assertTrue(refine_ff.base.np.allclose(optimized["deltas"][0], initial_delta))
        self.assertTrue(refine_ff.base.np.allclose(optimized["camera_poses"][0], initial_delta @ prior))

    def test_frame_face_pose_yaml_writes_explicit_transform_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rig_tr_frame_face.yaml"

            refine_ff.write_frame_face_pose_yaml(
                path,
                {(244, 0): refine_ff.base.np.eye(4, dtype=refine_ff.base.np.float64)},
                "rig_tr_frame_face",
            )

            text = path.read_text(encoding="utf-8")
            self.assertIn("type: rig_tr_frame_face", text)
            self.assertIn("frame_index: 244", text)
            self.assertIn("face_id: 0", text)


if __name__ == "__main__":
    unittest.main()
