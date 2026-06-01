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
