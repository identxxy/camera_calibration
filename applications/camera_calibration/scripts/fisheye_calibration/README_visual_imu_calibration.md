# Visual-Inertial Calibration Assessment

This note scopes the next step from four-fisheye intrinsic calibration to
headset visual-inertial calibration from MCAP captures.

## Current State

The current fisheye tooling extracts four image streams from MCAP, screens
sharp board observations, prepares per-camera intrinsic calibration, exports
KB8-compatible Seeker YAML, and generates QA reports.

The current Seeker YAML generators write `T_cam_imu`, `T_imu_cam`, and
`timeshift_cam_imu`, but those values are placeholders based on an `imu == cam0`
reference. They are not measured IMU extrinsics.

## Recommendation

This is suitable to continue in the headset fisheye calibration WIP, but the
first milestone should be a Kalibr-compatible visual-inertial bridge rather than
a native KB8 visual-inertial bundle adjuster.

Recommended first milestone:

1. Parse image and IMU topics from MCAP.
2. Export a Kalibr-compatible multi-camera + IMU dataset.
3. Run multi-camera visual-inertial calibration with fixed or strongly seeded
   camera intrinsics.
4. Convert measured `T_cam_imu`, `T_imu_cam`, and `timeshift_cam_imu` back into
   the Seeker KB8 YAML format.
5. Add a report section for IMU/image timing, target coverage, camera-IMU
   residuals, and transform consistency.

This keeps the implementation moderate and uses a proven continuous-time
visual-inertial optimizer. A native optimizer that uses this repository's KB8
projection directly is possible, but it is a larger project and should be a
second milestone only if the bridge is not accurate enough.

## Capture Requirement

The visual-inertial dataset must move the camera-IMU rig. Moving only the
calibration board in front of a stationary headset is not sufficient: the camera
sees target motion, but the IMU does not observe the same motion, so camera-IMU
extrinsics and time offset are not observable.

Use a static calibration target and move the headset rig around it.

Required capture properties:

- Board or Aprilgrid is static in the world during the sequence.
- Headset rig moves as one rigid body with the IMU.
- Start and end with several seconds of stillness for gravity and bias checks.
- Include roll, pitch, yaw, translation, and non-constant acceleration.
- Avoid a pure yaw-only or pure rotation-only sequence.
- Keep motion smooth enough to avoid severe blur, but not so slow that IMU
  excitation is weak.
- Ensure all four cameras observe the board over the sequence. Simultaneous
  visibility is not required for every frame, but the observation graph must
  connect all cameras through the rigid rig.
- Preserve raw image timestamps and raw IMU timestamps from the same clock
  domain when possible.

Practical target sequence:

- 60-120 seconds per capture.
- 10 seconds still at the beginning.
- Slow figure-eight and arc motions in front of the board.
- Deliberate roll and pitch changes.
- Several left/right/up/down translations.
- 10 seconds still at the end.

## Expected Outputs

The calibrated product should include:

- Four camera intrinsics, preferably keeping the current KB8 product for the
  driver.
- Four measured camera-IMU transforms.
- Camera-IMU time offset.
- IMU noise and bias random-walk estimates if available from the solver.
- A driver-readable Seeker calibration YAML where `T_cam_imu` is no longer a
  placeholder.
- A QA report covering reprojection error, camera coverage, IMU residuals,
  estimated time offset, and transform stability across repeated runs.

## Difficulty

Kalibr bridge path: medium difficulty. Most work is reliable MCAP export,
topic/timestamp validation, model conversion, and report generation.

Native KB8 visual-inertial optimization path: high difficulty. It requires
continuous-time trajectory modeling, IMU preintegration or spline residuals,
bias/noise handling, time-offset optimization, and robust initialization.

The bridge path is appropriate for the current WIP. The native path should be
separate work after the bridge reveals whether model mismatch is actually a
limiting factor.

## Kalibr vs This Repository

Kalibr is better suited for the first measured visual-inertial calibration:

- It has a mature visual-inertial formulation for camera-IMU extrinsics,
  temporal offset, IMU noise, and bias random walk.
- It models the moving camera-IMU rig with a continuous-time trajectory, which
  is the right abstraction for asynchronous image and high-rate IMU samples.
- Its output format is close to the camchain YAML style already used by the
  Seeker driver integration.
- It is a good independent reference implementation for validating whether the
  capture itself contains enough motion excitation.

Kalibr is weaker for this project in several ways:

- Its camera model set is more constrained than this repository's generic and
  non-central model family.
- A compact driver-side KB8 product may need model conversion or a fixed
  intrinsic workflow instead of direct native optimization.
- It adds an external toolchain and dataset conversion step.
- Debugging failures can be harder because the optimizer, target extraction, and
  model assumptions live outside the current codebase.

This repository is better for the existing camera-only workflow:

- It already owns the MCAP-to-board-observation preparation, quality screening,
  KB8 export, Seeker YAML generation, and static QA report.
- It supports flexible camera models including `central_generic`,
  `noncentral_generic`, and `central_thin_prism_fisheye`.
- It is easier to customize for Seeker-specific reporting, filtering, and driver
  file formats.
- It avoids a lossy handoff when staying within camera-only calibration.

This repository is weaker for visual-inertial calibration today:

- There is no established native IMU residual path in the current workflow.
- `T_cam_imu`, `T_imu_cam`, and `timeshift_cam_imu` are currently generated as
  placeholders, not measured quantities.
- Implementing the missing pieces natively would require trajectory
  parameterization, IMU residuals, bias/noise modeling, time-offset
  optimization, and careful initialization.

Practical decision:

- Use this repository for image extraction, intrinsic calibration, QA, and final
  Seeker YAML/report generation.
- Use Kalibr as the first camera-IMU solver.
- Only build a native KB8 visual-inertial optimizer if Kalibr's camera model
  conversion or residual quality becomes the bottleneck.
