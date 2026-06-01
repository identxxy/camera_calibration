#!/usr/bin/env python3
"""Focused tests for outer tower intrinsic refinement helpers."""

from pathlib import Path
import csv
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import refine_outer_tower_delta_prior as refine  # noqa: E402


class RefineOuterTowerDeltaPriorTest(unittest.TestCase):
    def test_tower_face_width_delta_moves_points_along_face_normal(self):
        point = refine.np.asarray([0.0, 0.0, 0.0], dtype=refine.np.float64)

        adjusted = refine.adjusted_tower_point(
            point,
            feature_id=0,
            tower_face_width_delta_m=0.02)

        expected_apothem_delta = 0.02 / (2.0 * refine.math.tan(refine.math.pi / 8.0))
        self.assertAlmostEqual(adjusted[0], expected_apothem_delta)
        self.assertAlmostEqual(adjusted[1], 0.0)
        self.assertAlmostEqual(adjusted[2], 0.0)

    def test_tower_face_width_delta_uses_feature_face_id(self):
        point = refine.np.asarray([0.0, 0.0, 0.0], dtype=refine.np.float64)

        adjusted = refine.adjusted_tower_point(
            point,
            feature_id=64 * 4,
            tower_face_width_delta_m=0.02)

        expected_apothem_delta = 0.02 / (2.0 * refine.math.tan(refine.math.pi / 8.0))
        self.assertAlmostEqual(adjusted[0], 0.0, places=12)
        self.assertAlmostEqual(adjusted[1], expected_apothem_delta)
        self.assertAlmostEqual(adjusted[2], 0.0)

    def test_projection_residuals_uses_tower_face_width_delta(self):
        observation = refine.make_observation(
            0,
            0,
            refine.np.asarray([100.0, 0.0], dtype=refine.np.float64),
            refine.np.asarray([1.0, 0.0, 1.0], dtype=refine.np.float64),
            0)
        intrinsics = [{
            "width": 640,
            "height": 480,
            "params": [100.0, 100.0, 0.0, 0.0] + [0.0] * 8,
        }]

        residual = refine.projection_residuals(
            [observation],
            [refine.np.eye(4)],
            [refine.np.eye(4)],
            intrinsics,
            tower_face_width_delta_m=0.02)

        expected_apothem_delta = 0.02 / (2.0 * refine.math.tan(refine.math.pi / 8.0))
        self.assertAlmostEqual(residual[0], 100.0 * expected_apothem_delta)
        self.assertAlmostEqual(residual[1], 0.0)

    def test_projection_residuals_uses_tower_face_pose_delta(self):
        observation = refine.make_observation(
            0,
            0,
            refine.np.asarray([100.0, 0.0], dtype=refine.np.float64),
            refine.np.asarray([1.0, 0.0, 1.0], dtype=refine.np.float64),
            0)
        intrinsics = [{
            "width": 640,
            "height": 480,
            "params": [100.0, 100.0, 0.0, 0.0] + [0.0] * 8,
        }]
        face_deltas = [refine.np.eye(4) for _ in range(8)]
        face_deltas[0] = refine.pose_matrix(refine.np.eye(3), [0.02, 0.0, 0.0])

        residual = refine.projection_residuals(
            [observation],
            [refine.np.eye(4)],
            [refine.np.eye(4)],
            intrinsics,
            tower_face_width_delta_m=0.0,
            tower_face_pose_deltas=face_deltas)

        self.assertAlmostEqual(residual[0], 2.0)
        self.assertAlmostEqual(residual[1], 0.0)

    def test_independent_face_plane_model_ignores_dataset_3d_point_and_face_width(self):
        observation = refine.make_observation(
            0,
            0,
            refine.np.asarray([100.0, 0.0], dtype=refine.np.float64),
            refine.np.asarray([999.0, 999.0, 999.0], dtype=refine.np.float64),
            0)
        intrinsics = [{
            "width": 640,
            "height": 480,
            "params": [100.0, 100.0, 0.0, 0.0] + [0.0] * 8,
        }]
        layout = {
            "first_tag_id": 0,
            "face_id_stride": 1,
            "tag_columns": 1,
            "tag_rows": 1,
            "tag_size_m": 0.08,
            "tag_spacing_m": 0.02,
            "tag_rotation_degrees": 180,
        }
        face_base_poses = [refine.pose_matrix(refine.np.eye(3), [1.0, 0.0, 1.04])]
        face_deltas = [refine.np.eye(4)]

        residual = refine.projection_residuals(
            [observation],
            [refine.np.eye(4)],
            [refine.np.eye(4)],
            intrinsics,
            tower_face_width_delta_m=100.0,
            tower_face_pose_deltas=face_deltas,
            tower_point_model="independent_face_planes",
            tower_face_base_poses=face_base_poses,
            tower_layout=layout)

        self.assertAlmostEqual(residual[0], 0.0)
        self.assertAlmostEqual(residual[1], 4.0)

    def test_intrinsics_refinement_updates_fxfycxcy_and_preserves_distortion(self):
        base = [{
            "width": 4000,
            "height": 3000,
            "params": [5000.0, 5100.0, 2000.0, 1500.0, 0.1, -0.02, 0.003, 0.04, 0.005, -0.001, 0.0002, -0.0003],
        }]
        per_camera = [refine.intrinsics_delta_from_values([0.01, -0.02, 3.0, -4.0])]

        refined = refine.apply_intrinsics_refinement(
            base,
            "per_camera_fxfycxcy",
            None,
            per_camera)

        self.assertAlmostEqual(refined[0]["params"][0], 5000.0 * refine.math.exp(0.01))
        self.assertAlmostEqual(refined[0]["params"][1], 5100.0 * refine.math.exp(-0.02))
        self.assertAlmostEqual(refined[0]["params"][2], 2003.0)
        self.assertAlmostEqual(refined[0]["params"][3], 1496.0)
        self.assertEqual(refined[0]["params"][4:], base[0]["params"][4:])

    def test_intrinsics_refinement_updates_opencv5_distortion_order(self):
        base = [{
            "width": 4000,
            "height": 3000,
            "params": [
                5000.0, 5100.0, 2000.0, 1500.0,
                0.10, -0.02, 0.003, 0.04, 0.005, -0.001, 0.0002, -0.0003,
            ],
        }]
        per_camera = [refine.intrinsics_delta_from_values([
            0.01, -0.02, 3.0, -4.0,
            0.001, -0.002, 0.0003, -0.0004, 0.005,
        ])]

        refined = refine.apply_intrinsics_refinement(
            base,
            "per_camera_opencv5",
            None,
            per_camera)

        params = refined[0]["params"]
        self.assertAlmostEqual(params[0], 5000.0 * refine.math.exp(0.01))
        self.assertAlmostEqual(params[1], 5100.0 * refine.math.exp(-0.02))
        self.assertAlmostEqual(params[2], 2003.0)
        self.assertAlmostEqual(params[3], 1496.0)
        self.assertAlmostEqual(params[4], 0.101)
        self.assertAlmostEqual(params[5], -0.022)
        self.assertAlmostEqual(params[6], 0.008)
        self.assertAlmostEqual(params[10], 0.0005)
        self.assertAlmostEqual(params[11], -0.0007)
        self.assertAlmostEqual(params[7], base[0]["params"][7])
        self.assertAlmostEqual(params[8], base[0]["params"][8])
        self.assertAlmostEqual(params[9], base[0]["params"][9])

    def test_intrinsics_total_focal_trust_region_clamps_log_delta_to_prior_relative_bound(self):
        delta = refine.intrinsics_delta_from_values([0.05, -0.05, 25.0, -30.0])

        clamped = refine.clamp_intrinsics_delta_to_total_bounds(delta, 0.02, 16.0)

        self.assertAlmostEqual(clamped[0], refine.math.log(1.02))
        self.assertAlmostEqual(clamped[1], refine.math.log(0.98))
        self.assertAlmostEqual(refine.math.exp(clamped[0]) - 1.0, 0.02)
        self.assertAlmostEqual(1.0 - refine.math.exp(clamped[1]), 0.02)
        self.assertAlmostEqual(clamped[2], 16.0)
        self.assertAlmostEqual(clamped[3], -16.0)
        self.assertTrue(refine.np.allclose(delta, [0.05, -0.05, 25.0, -30.0]))

    def test_intrinsics_total_distortion_trust_region_clamps_opencv5_delta(self):
        delta = refine.intrinsics_delta_from_values([
            0.05, -0.05, 25.0, -30.0,
            0.20, -0.30, 0.02, -0.04, 0.10,
        ])

        clamped = refine.clamp_intrinsics_delta_to_total_bounds(delta, 0.02, 16.0, 0.03)

        self.assertAlmostEqual(clamped[4], 0.03)
        self.assertAlmostEqual(clamped[5], -0.03)
        self.assertAlmostEqual(clamped[6], 0.02)
        self.assertAlmostEqual(clamped[7], -0.03)
        self.assertAlmostEqual(clamped[8], 0.03)

    def test_intrinsics_acceptance_falls_back_when_delta_exceeds_gate(self):
        manifest = [{"camera_id": "1-1"}, {"camera_id": "1-2"}]
        base = [
            {"width": 4000, "height": 3000, "params": [5000.0, 5000.0, 2000.0, 1500.0] + [0.0] * 8},
            {"width": 4000, "height": 3000, "params": [5000.0, 5000.0, 2000.0, 1500.0] + [0.0] * 8},
        ]
        refined = [
            {"width": 4000, "height": 3000, "params": [5050.0, 5040.0, 2004.0, 1501.0] + [0.0] * 8},
            {"width": 4000, "height": 3000, "params": [5300.0, 5000.0, 2000.0, 1500.0] + [0.0] * 8},
        ]
        args = type("Args", (), {
            "accept_camera_max_intrinsic_focal_delta_frac": 0.02,
            "accept_camera_max_intrinsic_principal_delta_px": 8.0,
            "accept_camera_max_intrinsic_distortion_delta": 0.15,
        })()

        accepted, accepted_ids, rows = refine.accepted_refined_intrinsics(
            manifest,
            base,
            refined,
            ["1-1", "1-2"],
            args,
            "per_camera_fxfycxcy")

        self.assertEqual(accepted_ids, ["1-1"])
        self.assertEqual(accepted[0]["params"], refined[0]["params"])
        self.assertEqual(accepted[1]["params"], base[1]["params"])
        self.assertEqual(rows[0]["decision"], "accepted_refined")
        self.assertEqual(rows[1]["decision"], "rejected_to_prior")
        self.assertEqual(rows[1]["reason"], "intrinsic_delta_exceeds_acceptance_gate")

    def test_intrinsics_acceptance_falls_back_when_opencv5_distortion_delta_exceeds_gate(self):
        manifest = [{"camera_id": "1-1"}, {"camera_id": "1-2"}]
        base = [
            {"width": 4000, "height": 3000, "params": [5000.0, 5000.0, 2000.0, 1500.0] + [0.0] * 8},
            {"width": 4000, "height": 3000, "params": [5000.0, 5000.0, 2000.0, 1500.0] + [0.0] * 8},
        ]
        refined = [
            {"width": 4000, "height": 3000, "params": [5020.0, 5010.0, 2001.0, 1499.0, 0.01, 0.0, 0.02, 0, 0, 0, 0.001, -0.001]},
            {"width": 4000, "height": 3000, "params": [5020.0, 5010.0, 2001.0, 1499.0, 0.20, 0.0, 0.0, 0, 0, 0, 0.001, -0.001]},
        ]
        args = type("Args", (), {
            "accept_camera_max_intrinsic_focal_delta_frac": 0.02,
            "accept_camera_max_intrinsic_principal_delta_px": 8.0,
            "accept_camera_max_intrinsic_distortion_delta": 0.15,
        })()

        accepted, accepted_ids, rows = refine.accepted_refined_intrinsics(
            manifest,
            base,
            refined,
            ["1-1", "1-2"],
            args,
            "per_camera_opencv5")

        self.assertEqual(accepted_ids, ["1-1"])
        self.assertEqual(accepted[0]["params"], refined[0]["params"])
        self.assertEqual(accepted[1]["params"], base[1]["params"])
        self.assertEqual(rows[0]["decision"], "accepted_refined")
        self.assertEqual(rows[1]["decision"], "rejected_to_prior")
        self.assertEqual(rows[1]["max_abs_distortion_delta"], "0.2")

    def test_camera_reprojection_accepted_rows_can_be_written_separately(self):
        manifest = [{"camera_id": "1-1"}]
        observation = refine.make_observation(
            0,
            0,
            refine.np.asarray([0.0, 0.0], dtype=refine.np.float64),
            refine.np.asarray([0.0, 0.0, 1.0], dtype=refine.np.float64),
            129)
        observations_by_camera = [[observation]]
        tower_poses = [refine.np.eye(4)]
        prior_pose = [refine.np.eye(4)]
        candidate_pose = [refine.pose_matrix(refine.np.eye(3), [1.0, 0.0, 0.0])]
        intrinsics = [{
            "width": 640,
            "height": 480,
            "params": [100.0, 100.0, 0.0, 0.0] + [0.0] * 8,
        }]
        candidate_rows = refine.summarize_camera_reprojection(
            manifest,
            observations_by_camera,
            prior_pose,
            candidate_pose,
            tower_poses,
            intrinsics,
            intrinsics)
        accepted_rows = refine.summarize_camera_reprojection(
            manifest,
            observations_by_camera,
            prior_pose,
            prior_pose,
            tower_poses,
            intrinsics,
            intrinsics)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "camera_reprojection_accepted.tsv"
            refine.write_camera_reprojection_tsv(path, accepted_rows)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

        self.assertAlmostEqual(float(candidate_rows[0]["after_median_px"]), 100.0)
        self.assertAlmostEqual(float(rows[0]["after_median_px"]), 0.0)

    def test_collect_intrinsics_accepts_index_only_colmap_fallback_files(self):
        manifest = [{"camera_id": "1-1"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "intrinsics0.yaml"
            path.write_text(
                "\n".join([
                    "type : CentralOpenCVModel",
                    "width : 4096",
                    "height : 3000",
                    "parameters : [3851.9, 4993.1, 2050.3, 1437.3, 0, 0, 0, 0, 0, 0, 0, 0]",
                    "",
                ]),
                encoding="utf-8")

            intrinsics = refine.collect_intrinsics(
                tmpdir,
                manifest,
                [(4096, 3000)],
                "central_opencv")

        self.assertAlmostEqual(intrinsics[0]["params"][0], 3851.9)
        self.assertAlmostEqual(intrinsics[0]["params"][1], 4993.1)
        self.assertAlmostEqual(intrinsics[0]["params"][2], 2050.3)

    def test_camera_acceptance_output_prior_pose_includes_rejected_and_inactive(self):
        manifest = [{"camera_id": "1-1"}, {"camera_id": "1-2"}, {"camera_id": "1-3"}]
        camera_rows = [
            {"camera_id": "1-1", "observation_count": 50, "after_median_px": 10.0, "after_under_300_fraction": 1.0},
            {"camera_id": "1-2", "observation_count": 50, "after_median_px": 900.0, "after_under_300_fraction": 0.0},
            {"camera_id": "1-3", "observation_count": 3, "after_median_px": None, "after_under_300_fraction": 0.0},
        ]
        args = type("Args", (), {
            "accept_camera_median_px": 350.0,
            "allow_ungated_accepted_output": False,
        })()

        rows = refine.summarize_camera_acceptance(
            manifest,
            camera_rows,
            active_camera=[True, True, False],
            used_camera=[True, True, False],
            accepted_camera_ids=["1-1"],
            args=args,
        )
        output_prior_pose = [
            row["camera_id"] for row in rows if row["output_pose"] == "prior"
        ]

        self.assertEqual(output_prior_pose, ["1-2", "1-3"])
        self.assertEqual(rows[1]["decision"], "rejected_to_prior")
        self.assertEqual(rows[2]["decision"], "excluded_prior_only")

    def test_pnp_tower_initialization_weights_lower_median_error_vote_more(self):
        camera_priors = [refine.np.eye(4), refine.np.eye(4)]
        low_error_pose = refine.pose_matrix(refine.np.eye(3), [0.0, 0.0, 0.0])
        high_error_pose = refine.pose_matrix(refine.np.eye(3), [10.0, 0.0, 0.0])
        pnp_views = [
            {
                "camera_index": 0,
                "imageset_index": 0,
                "median_error_px": 1.0,
                "camera_tr_tower": low_error_pose,
            },
            {
                "camera_index": 1,
                "imageset_index": 0,
                "median_error_px": 10.0,
                "camera_tr_tower": high_error_pose,
            },
        ]

        tower_poses, frame_pnp_quality = refine.initialize_tower_poses_from_pnp(
            pnp_views, camera_priors, frame_count=1, min_votes=1)

        self.assertLess(tower_poses[0][0, 3], 1.0)
        self.assertEqual(frame_pnp_quality[0]["pnp_vote_count"], 2)
        self.assertAlmostEqual(frame_pnp_quality[0]["pnp_median_error_px"], 5.5)
        self.assertEqual(frame_pnp_quality[0]["pnp_pose_average"], "robust_weighted")

    def test_pnp_tower_initialization_keeps_single_vote_pose(self):
        camera_priors = [refine.np.eye(4)]
        pose = refine.pose_matrix(
            refine.so3_exp([0.0, 0.0, refine.math.radians(12.0)]),
            [1.0, 2.0, 3.0])
        pnp_views = [{
            "camera_index": 0,
            "imageset_index": 0,
            "median_error_px": 3.0,
            "camera_tr_tower": pose,
        }]

        tower_poses, frame_pnp_quality = refine.initialize_tower_poses_from_pnp(
            pnp_views, camera_priors, frame_count=1, min_votes=1)

        self.assertTrue(refine.np.allclose(tower_poses[0], pose))
        self.assertEqual(frame_pnp_quality[0]["pnp_vote_count"], 1)
        self.assertAlmostEqual(frame_pnp_quality[0]["pnp_median_error_px"], 3.0)
        self.assertEqual(frame_pnp_quality[0]["pnp_pose_average"], "single_vote")

    def test_bridge_prior_override_replaces_full_pose_by_label_mapping(self):
        manifest = [{"camera_id": "4-1"}, {"camera_id": "4-2"}]
        priors = [
            refine.pose_matrix(refine.np.eye(3), [1.0, 0.0, 0.0]),
            refine.pose_matrix(refine.np.eye(3), [2.0, 0.0, 0.0]),
        ]
        bridge_pose = refine.pose_matrix(
            refine.so3_exp([0.0, 0.0, refine.math.radians(20.0)]),
            [3.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            bridge_yaml = Path(tmpdir) / "bridge.yaml"
            refine.write_pose_yaml(bridge_yaml, [None, bridge_pose])

            rows, updated = refine.apply_bridge_prior_overrides(
                priors,
                manifest,
                bridge_yaml,
                "4-2",
                {"4-2": 1})

        self.assertEqual([row["camera_id"] for row in rows], ["4-2"])
        self.assertIs(updated[0], priors[0])
        self.assertTrue(refine.np.allclose(updated[1], bridge_pose))
        self.assertGreater(rows[0]["center_delta_m"], 0.0)
        self.assertGreater(rows[0]["rotation_delta_deg"], 19.0)

    def test_observation_residual_diagnostics_preserve_tag_corner_face_and_gate_state(self):
        manifest = [{"camera_id": "1-1"}]
        camera_poses = [refine.np.eye(4)]
        tower_poses = [refine.np.eye(4), None]
        intrinsics = [{
            "width": 640,
            "height": 480,
            "params": [100.0, 100.0, 0.0, 0.0] + [0.0] * 8,
        }]
        kept = refine.make_observation(
            0,
            0,
            refine.np.asarray([0.0, 0.0], dtype=refine.np.float64),
            refine.np.asarray([0.0, 0.0, 1.0], dtype=refine.np.float64),
            129)
        removed = refine.make_observation(
            0,
            0,
            refine.np.asarray([30.0, 40.0], dtype=refine.np.float64),
            refine.np.asarray([0.0, 0.0, 1.0], dtype=refine.np.float64),
            130)
        missing_pose = refine.make_observation(
            1,
            0,
            refine.np.asarray([1.0, 2.0], dtype=refine.np.float64),
            refine.np.asarray([0.0, 0.0, 1.0], dtype=refine.np.float64),
            131)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "observation_residuals.tsv"
            refine.write_observation_residuals(
                path,
                manifest,
                [[kept, removed], [missing_pose]],
                [[kept], []],
                camera_poses,
                tower_poses,
                intrinsics)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

        self.assertEqual(len(rows), 3)
        for field in (
                "frame_index",
                "camera_index",
                "camera_id",
                "feature_id",
                "tag_id",
                "corner_id",
                "face_id",
                "observed_x",
                "observed_y",
                "projected_x",
                "projected_y",
                "residual_x_px",
                "residual_y_px",
                "residual_px",
                "used_after_gate",
                "projection_status"):
            self.assertIn(field, rows[0])
        self.assertEqual(rows[0]["feature_id"], "129")
        self.assertEqual(rows[0]["tag_id"], "32")
        self.assertEqual(rows[0]["corner_id"], "1")
        self.assertEqual(rows[0]["face_id"], "1")
        self.assertEqual(rows[0]["used_after_gate"], "yes")
        self.assertEqual(rows[0]["projection_status"], "ok")
        self.assertEqual(rows[1]["feature_id"], "130")
        self.assertEqual(rows[1]["tag_id"], "32")
        self.assertEqual(rows[1]["corner_id"], "2")
        self.assertEqual(rows[1]["face_id"], "1")
        self.assertEqual(rows[1]["used_after_gate"], "no")
        self.assertEqual(rows[1]["projection_status"], "ok")
        self.assertAlmostEqual(float(rows[1]["residual_x_px"]), -30.0)
        self.assertAlmostEqual(float(rows[1]["residual_y_px"]), -40.0)
        self.assertAlmostEqual(float(rows[1]["residual_px"]), 50.0)
        self.assertEqual(rows[2]["feature_id"], "131")
        self.assertEqual(rows[2]["tag_id"], "32")
        self.assertEqual(rows[2]["corner_id"], "3")
        self.assertEqual(rows[2]["face_id"], "1")
        self.assertEqual(rows[2]["used_after_gate"], "no")
        self.assertEqual(rows[2]["projection_status"], "missing_tower_pose")
        self.assertEqual(rows[2]["projected_x"], "")
        self.assertEqual(rows[2]["residual_px"], "")

    def test_post_refine_observation_gate_disabled_summary_field(self):
        obs = refine.make_observation(
            0,
            0,
            refine.np.asarray([0.0, 0.0], dtype=refine.np.float64),
            refine.np.asarray([0.0, 0.0, 1.0], dtype=refine.np.float64),
            129)

        summary = refine.make_post_refine_observation_gate_summary(
            enabled=False,
            threshold_px=0.0,
            outer_iterations=2,
            input_observations=[[obs]],
            kept_observations=[[obs]],
            removed_observations=0,
            missing_pose_or_invalid_projection=0)

        self.assertEqual(summary["enabled"], False)
        self.assertEqual(summary["threshold_px"], 0.0)
        self.assertEqual(summary["outer_iterations"], 2)
        self.assertEqual(summary["input_observations"], 1)
        self.assertEqual(summary["kept_observations"], 1)
        self.assertEqual(summary["removed_observations"], 0)
        self.assertEqual(summary["missing_pose_or_invalid_projection"], 0)

    def test_post_refine_observation_gate_trims_high_refined_residual(self):
        camera_poses = [refine.np.eye(4)]
        tower_poses = [refine.np.eye(4)]
        intrinsics = [{
            "width": 640,
            "height": 480,
            "params": [100.0, 100.0, 0.0, 0.0] + [0.0] * 8,
        }]
        low_residual = refine.make_observation(
            0,
            0,
            refine.np.asarray([1.0, 0.0], dtype=refine.np.float64),
            refine.np.asarray([0.0, 0.0, 1.0], dtype=refine.np.float64),
            129)
        high_residual = refine.make_observation(
            0,
            0,
            refine.np.asarray([30.0, 40.0], dtype=refine.np.float64),
            refine.np.asarray([0.0, 0.0, 1.0], dtype=refine.np.float64),
            130)

        filtered_by_frame, filtered_by_camera, summary = (
            refine.filter_post_refine_observations_by_projection_gate(
                [[low_residual, high_residual]],
                camera_count=1,
                camera_poses=camera_poses,
                tower_poses=tower_poses,
                intrinsics=intrinsics,
                max_residual_px=10.0,
                outer_iterations=2))

        self.assertEqual(
            [[refine.observation_gate_key(obs) for obs in frame] for frame in filtered_by_frame],
            [[refine.observation_gate_key(low_residual)]])
        self.assertEqual(
            [[refine.observation_gate_key(obs) for obs in camera] for camera in filtered_by_camera],
            [[refine.observation_gate_key(low_residual)]])
        self.assertEqual(summary["enabled"], True)
        self.assertEqual(summary["threshold_px"], 10.0)
        self.assertEqual(summary["input_observations"], 2)
        self.assertEqual(summary["kept_observations"], 1)
        self.assertEqual(summary["removed_observations"], 1)
        self.assertEqual(summary["missing_pose_or_invalid_projection"], 0)


if __name__ == "__main__":
    unittest.main()
