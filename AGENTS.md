# Project Instructions

This file adds project-local instructions for `/home/vox/camera_calibration`.
User-level instructions in `~/.codex/AGENTS.md` still apply.

## Parallel Codex Worktrees

Use Git worktrees for concurrent Codex sessions. Do not switch branches inside a
shared working directory while another session may be using it.

Current branch/worktree split:

- Head-mounted fisheye calibration:
  - branch: `calib/headset-fisheye`
  - cwd: `/home/vox/camera_calibration/.worktrees/calib-headset-fisheye`
- Studio pinhole camera matrix calibration:
  - branch: `calib/studio-pinhole`
  - cwd: `/home/vox/camera_calibration/.worktrees/calib-studio-pinhole`
- Main checkout:
  - branch: `master`
  - cwd: `/home/vox/camera_calibration`
  - role: coordination, shared docs, and existing uncommitted work unless moved intentionally.
- Historical distributed AprilTag QC checkout:
  - branch: `calib/distributed-apriltag-qc`
  - cwd: `/home/vox/camera_calibration/.worktrees/calib-distributed-apriltag-qc`
  - role: historical reference only; do not treat it as the current production
    studio pipeline unless explicitly asked.

Before starting a session, run:

```bash
git status --short
git worktree list
```

Do not delete worktrees or branches without explicit user confirmation.

## Calibration Direction

Keep the headset fisheye and studio pinhole calibration work separate until the
data flow and output camera models are stable. They can be merged later if the
shared code path becomes obvious.

For headset fisheye calibration, prefer evaluating `central_generic` first, then
`noncentral_generic` if residuals or hardware geometry justify it. Use
`central_thin_prism_fisheye` when a compact parametric model is required.

For studio pinhole camera matrix calibration, keep the target product explicit:
intrinsic matrix `K`, distortion convention, image resolution, and whether the
result must be consumable by OpenCV, COLMAP, SLAM, or this repository's generic
model loaders.

For the fixed outer studio ring, cameras `4-1`, `4-2`, and `4-3` are top-down
view cameras. The other outer-ring cameras are mounted roughly horizontally and
look inward toward the studio center. Treat this as a hardware/layout prior when
checking COLMAP initialization, rig visualization, AprilTag tower geometry, and
inner/outer bridge consistency.

## Studio Capture Data Quality Rules

For synchronized studio rig captures, two frame-count anomalies are expected and
must be handled differently:

- A machine or four-camera group may stop 1-2 frames earlier at the tail of a
  sequence while the start remains aligned. Treat this as a normal tail-trim
  case: stage only the common prefix / common image-id intersection.
- If an individual camera drops frames because of an unstable connection, the
  whole sequence for that camera is invalid. Exclude that camera from that
  sequence instead of trying to keep partial observations.

These rules apply before feature extraction and bundle adjustment. The staging
step should decide the active camera set per sequence, then normalize filenames
only over the valid common frame set.

## Studio 32-Camera Calibration Quick Start

The current studio system has 32 synchronized cameras:

- outer ring: 24 fixed cameras, indexed `1-1..8-3`;
- inner ring: 8 movable cameras, indexed as `inner0..inner7` in inner-only
  products and remapped to all32 indices `24..31` in bridge/unified products.

Outer cameras `4-1`, `4-2`, and `4-3` are top-down view cameras. In the all32
bridge convention they are outer indices `9`, `10`, and `11`. The other outer
cameras are roughly horizontal inward-looking cameras. Use this as a hardware
prior when judging visualizations, COLMAP bootstrap results, bridge consistency,
and gravity/world-frame alignment.

There are four capture/data modes:

- `outer_large_marker`: low-density A4 board, pattern `_0`, captured by W3/W4
  outer cameras. This is the production way to initialize/refresh outer24
  intrinsics. Run Windows distributed QC first, aggregate on t0 with
  `distributed_apriltag_quality_filter.py aggregate --stage-mode passing-images`,
  then run the repo C++ board detector and OpenCV per-camera intrinsic solve.
  These intrinsics are relatively fixed and should normally be reused.
- `whole`: move the AprilTag tower through the studio. This is for outer-ring
  extrinsic refinement and broad inner/outer co-visibility. It should start
  from trusted outer intrinsics and a trusted coarse outer rig. The production
  tower path uses one shared tower pose per synchronized frame, fixed
  `360/8 = 45 deg` adjacent face yaw, and an optimized global face-width weak
  parameter. Flexible/independent face models are diagnostic fallback paths, not
  the default production output.
  The physical printed tower black tiles are 8 cm with 2 cm white tile gaps.
  OpenCV AprilTag detector corners land on the inner detector square, so they
  should only identify tag IDs and provide a red-box scale prior. Production
  whole BA must use red-box scale + local edge-supported black-tile outer
  corners, with physical geometry `tower_tag_size_m = 0.08` and
  `tower_tag_spacing_m = 0.02`. The production outer-tower refine preset is
  `wide200_then_gate6`: keep same-ID support with a loose 200 px initialization
  gate, optimize the shared tower/frame model, then write the accepted result
  with a strict 6 px final gate. The older
  detector-corner geometry
  `0.06710408594834662 / 0.03289591405165339` is legacy diagnostic-only for
  datasets that still store raw OpenCV inner detector corners.
- `large_marker`: low-density A4 board, pattern `_0`. This is the production
  inner/outer bridge board. It binds the refined inner rig to the outer studio
  frame through all32 observations.
- `small_marker`: high-density A4 board, pattern `_3`. This is the inner-ring
  precision/quality board. Use it for inner8 intrinsics/extrinsics quality or
  diagnostic refine, not as the outer bridge input.

Canonical operator entrypoints on t0:

```text
Panel root:        http://192.168.2.0:9898/
Report root:       http://192.168.2.0:9899/
Production all32:  http://192.168.2.0:9898/?mode=run_studio_calibration_pipeline
Whole only:        http://192.168.2.0:9898/?mode=operate_whole_outer_cage
Large bridge:      http://192.168.2.0:9898/?mode=operate_large_marker_bridge
Small inner:       http://192.168.2.0:9898/?mode=operate_small_marker_inner
Full bootstrap:    http://192.168.2.0:9898/?mode=run_outer_tower_recalib_pipeline
```

Use `run_studio_calibration_pipeline` for normal reproducible all32 runs after
QC/staging. Use `operate_whole_outer_cage` for production whole-only outer delta
refine. `run_outer_tower_recalib_pipeline` is the diagnostic/full bootstrap
entrypoint: it can re-enable COLMAP frame voting, RANSAC rig voting, and
side-prior completion when the existing outer prior is missing or visibly wrong.

Important calibration boundary:

- The first usable outer coarse prior came from multi-frame COLMAP scene voting,
  RANSAC rig voting, and side-prior completion.
- Current production recalibration does not treat COLMAP/RANSAC as the final
  calibration. It starts from a trusted prior and runs AprilTag frame-face /
  tag-plane delta refinement plus large-marker bridge validation.
- If the outer cage only moved slightly and camera order/topology is unchanged,
  do not rerun COLMAP; refine from the trusted prior.
- If the outer cage was physically reconfigured, labels changed, or the prior is
  visually invalid, rerun the full bootstrap path to build a new coarse prior,
  then still promote only a result that passes frame-face/bridge/residual gates.

The current portable final product is a unified 32-camera YAML plus report/viewer:

```text
studio_32_cameras.yaml
combined_studio_rig_viewer_v1/index.html
advanced_correspondence_viewer_v1/correspondence_data.json
```

`advanced_correspondence_viewer_v1/correspondence_data.json` is a data asset
loaded by the unified 3D viewer. Do not promote the standalone advanced
correspondence HTML as a final human report.

Final YAML extrinsics are `camera_tr_studio_rig`: rig/world points transform into
OpenCV camera coordinates. OpenCV `+x right, +y down, +z forward` applies only to
the camera frame. The published `studio_rig` is a physical studio/world frame,
not cam0 and not an OpenCV camera frame. Its origin is the mean center of non-4
`*-2` outer cameras, `+Y` is vertical down from `*-1` to `*-3`, `+Z` is forward
opposite the missing `4-2` side gap, `-Z` points toward that backward gap, and
`+X` completes a right-handed frame. The YAML also stores the
`coordinate_transform` block that maps from the pre-alignment source rig.
For non-4 outer side columns, the physical layer order is `*-1` top, `*-2`
middle, `*-3` bottom.

When importing the YAML elsewhere, keep the transform direction explicit:
`p_camera = T_camera_studio * p_studio`. For camera centers, invert it:
`C_studio = -R_camera_studio.T @ t_camera_studio`. For a target world frame where
`p_target = T_target_studio * p_studio`, use
`T_camera_target = T_camera_studio * inverse(T_target_studio)`. The previous
2026-06-03 `+Y up` studio frame converts to the current `+Y down, +Z forward`
frame with `diag(-1, -1, 1)`, not a single-axis reflection.

For detailed runbooks, start with:

```text
scripts/calib/studio_calibration_human_runbook.html
scripts/calib/README_studio_operation_commands.md
scripts/calib/README_script_inventory.md
scripts/calib/README_studio_coordinate_frames.md
scripts/calib/README_studio_32_camera_system.md
scripts/calib/README_studio_calibration_pipeline.md
scripts/calib/README_calibration_panel.md
```

Repository hygiene note: do not use a blind `git clean -fdX` in this checkout.
It would remove ignored `studio/` knowledge/discussion notes. For local artifact
cleanup, delete explicit build/cache/report/raw-data paths instead. If a stale
Docker build directory is root-owned, remove it manually with
`sudo rm -rf /home/vox/camera_calibration/build_docker`.

The ignored `studio/` tree is an internal agent coordination/archive layer, not
the canonical human runbook. Prefer the tracked `scripts/calib/README_*` and
`scripts/calib/studio_calibration_human_runbook.html` documents for reproducible
operator instructions.
