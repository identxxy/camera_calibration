# Studio Calibration Script Inventory

Last updated: 2026-07-05

This document classifies `scripts/calib` and `scripts/ops` entrypoints so future
operators do not mistake legacy/bootstrap helpers for the production path.

## Human-Facing Production Entrypoints

Use these first for routine studio work:

- `scripts/calib/run_studio_calibration_pipeline.py`
  - Main all32 orchestration wrapper.
  - Supports full all32 regression, `--outer-only`, `--bridge-only`, dry-runs,
    unified YAML export, correspondence JSON export, and current publishing.
- `scripts/calib/run_inner_bridge_recalib_pipeline.py`
  - Lower-level large-marker bridge and small-marker inner quality wrapper.
  - Used directly for small-marker quality-only runs, and internally by the
    all32 wrapper.
- `scripts/calib/run_outer_tower_recalib_pipeline.py`
  - Lower-level whole/tower wrapper.
  - Production outer refine uses the `wide200_then_gate6` preset with
    black-tile physical-corner observations.
- `scripts/ops/publish_t0_clean_calib_reports.py`
  - The only current homepage publisher for `http://192.168.2.0:9899/`.
  - Do not replace it with old report registries or globbed report indexes.
- `scripts/calib/calibration_panel_server.py`
  - Backend for the 9898 operator panel.

## Production Helpers

These are called by production wrappers or are safe to run when their inputs are
explicit:

- `scripts/calib/distributed_apriltag_quality_filter.py`
- `scripts/calib/server_run_distributed_clients.py`
- `scripts/calib/client_detect_apriltag_tower_opencv.py`
- `scripts/calib/parallel_extract_features.py`
- `scripts/calib/calibrate_tower_intrinsics_opencv.py`
- `scripts/calib/refine_outer_tower_frame_face_planes.py`
- `scripts/calib/refine_outer_tower_delta_prior.py`
- `scripts/calib/export_combined_studio_extrinsics.py`
- `scripts/calib/export_calibration_correspondence_residuals.py`
- `scripts/calib/generate_studio_correspondence_viewer.py`
- `scripts/calib/generate_combined_studio_rig_viewer.py`
- `scripts/calib/generate_threejs_rig_viewer.py`
- `scripts/calib/generate_camera_origin_projection_report.py`
- `scripts/calib/generate_intrinsic_feature_coverage_report.py`
- `scripts/calib/generate_opencv_intrinsics_report.py`
- `scripts/calib/generate_outer_frame_face_report.py`
- `scripts/calib/generate_rig_extrinsics_report.py`
- `scripts/calib/generate_inner_calibration_report.py`
- `scripts/calib/studio_canonical_frame.py`

## Diagnostic And Bootstrap Helpers

These are useful for debugging, historical bootstrap, or one-off analysis. They
should not be presented as routine human operations unless a runbook explicitly
calls for them:

- `scripts/calib/run_outer_colmap_frame_vote.py`
- `scripts/calib/vote_outer_colmap_runs.py`
- `scripts/calib/complete_outer_rig_side_prior.py`
- `scripts/calib/evaluate_inner_outer_bridge.py`
- `scripts/calib/filter_pnp_views_by_pose_consensus.py`
- `scripts/calib/select_outer_tower_high_quality_subset.py`
- `scripts/calib/analyze_outer_tag_residual_tail.py`
- `scripts/calib/generate_outer_colmap_scene_viewer.py`
- `scripts/calib/generate_whole_tag_timeline_report.py`
- `scripts/calib/generate_black_tile_corner_refine_overlay.py`
- `scripts/calib/apriltag_tower_black_tile_refine.py`
- `scripts/calib/remap_apriltag_tower_dataset_corners.py`
- `scripts/calib/build_apriltag_tower_dataset_from_detections.py`
- `scripts/calib/build_apriltag_tower_dataset_opencv.py`

## Operator Documentation

Start here:

- `scripts/calib/studio_calibration_human_runbook.html`
- `scripts/calib/README_studio_operation_commands.md`
- `scripts/calib/README_studio_calibration_pipeline.md`
- `scripts/calib/README_studio_32_camera_system.md`
- `scripts/calib/README_studio_coordinate_frames.md`
- `scripts/ops/README_t0_report_contract.md`
- `AGENTS.md`

The HTML runbook is intentionally self-contained and command-oriented. The
Markdown documents carry deeper rationale and agent-facing context.

## Hygiene Rules

- Generated calibration reports belong under `/home/ubuntu/calib_data/...` on
  t0 or under ignored local research archives, not under source directories.
- Do not promote standalone correspondence viewers, dated scratch reports, or
  `report_registry.json` to the 9899 homepage.
- Avoid `git clean -fdX` in this repo because it would delete ignored
  `studio/` notes. Clean explicit artifacts instead.
- If `build_docker/` is root-owned, remove it manually with
  `sudo rm -rf /home/vox/camera_calibration/build_docker`.
