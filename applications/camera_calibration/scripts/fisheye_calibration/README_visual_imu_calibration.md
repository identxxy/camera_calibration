# Visual-IMU Calibration From Full-Board Features

This note records the current headset fisheye visual-inertial calibration path.
The implementation is intentionally split into a native visual frontend and a
Python inertial backend:

```text
native C++ full-board detector
  -> cam*_features.bin
  -> Python rotation / SE(3) visual-IMU solvers
  -> Seeker KB8 YAML + JSON + HTML report
```

The C++ detector remains part of the reproducible pipeline. It extracts the
repository's full A4 tagged-pattern observations; it should not be replaced by
Kalibr AprilGrid detection for this board.

## Position In The Full Calibration Flow

Visual-IMU calibration is the second solve stage of the headset calibration
flow. Run the vision-only four-fisheye solve first to obtain:

```text
KB8 intrinsics for cam0..cam3
camera-camera rig extrinsics
driver-readable camera rig YAML
```

The VI stage then keeps those camera parameters fixed and estimates the physical
IMU frame relative to the fixed four-camera rig. The vision-only solve and the
VI solve may use the same MCAP if that capture has enough visual coverage and
enough IMU motion excitation. In practice the solver estimates `T_cam0_imu`,
then derives the other cameras' `T_cam_i_imu` using the fixed camera-camera rig.

Do not treat VI as four independent camera-IMU calibrations. The four fisheye
cameras are a rigid multiview system, and the IMU transform should be solved as
one rig-level alignment problem.

## Runtime Environment And Docker

The current production path does not require ROS or a ROS/Kalibr Docker image.
It only needs:

```text
repo C++ camera_calibration binary
Python runtime with numpy, pyyaml, mcap, and OpenCV/scipy where used by helper scripts
MCAP files with synchronized image and IMU timestamps
existing KB8 camera-camera rig prior
```

The current native pipeline is:

```text
MCAP / extracted images
  -> repo C++ full-board detector
  -> cam*_features.bin
  -> Python MCAP + IMU + VI SE(3) solver
  -> Seeker KB8 YAML, summary JSON, HTML report
```

A lightweight Docker image is sufficient if a containerized run is desired. It
should contain the repo build dependencies plus the Python MCAP stack. It does
not need ROS unless the goal is to run upstream Kalibr as a separate baseline.

A development container can be used as a convenient Python and MCAP execution
environment. The final native SE(3) result does not call ROS nodes, `rosbag`, or
Kalibr's joint optimizer.

Use a full ROS Noetic / Kalibr image only for optional research comparison:

```text
MCAP -> ROS1 bag -> Kalibr AprilGrid / camchain / imu YAML -> Kalibr optimizer
```

That optional path is heavier, depends on ROS1, and is not the source of the
current Seeker KB8 VI calibration product.

## Native VI Run Order

The native VI path has three explicit steps:

1. Extract full-board features with the repo C++ detector.

```text
input: MCAP / extracted image streams and repo calibration pattern
output: cam0_features.bin, cam1_features.bin, cam2_features.bin, cam3_features.bin
```

2. Run gyro rotation alignment.

```bash
python applications/camera_calibration/scripts/fisheye_calibration/calibrate_visual_imu_rotation_from_full_board.py \
  --prior-yaml <stage1_kb8_rig_prior.yaml> \
  --feature-root <directory_with_cam_features> \
  --down-mcap <down_pair_vi_capture.mcap> \
  --up-mcap <up_pair_vi_capture.mcap> \
  --output-yaml <rig_aligned_rotation_yaml> \
  --raw-output-yaml <raw_per_camera_rotation_yaml> \
  --summary-json <rotation_summary.json> \
  --report-html <rotation_report.html>
```

3. Run accelerometer SE(3) alignment.

```bash
python applications/camera_calibration/scripts/fisheye_calibration/calibrate_visual_imu_se3_from_full_board.py \
  --prior-yaml <stage1_kb8_rig_prior.yaml> \
  --rotation-yaml <rig_aligned_rotation_yaml> \
  --rotation-summary <rotation_summary.json> \
  --feature-root <directory_with_cam_features> \
  --down-mcap <down_pair_vi_capture.mcap> \
  --up-mcap <up_pair_vi_capture.mcap> \
  --output-yaml <seeker_kb8_full_board_vi_se3.yaml> \
  --summary-json <full_board_vi_se3_summary.json> \
  --report-html <full_board_vi_se3_report.html>
```

The feature extraction step is native-pattern specific. The Python VI scripts
expect `cam*_features.bin`; they do not detect the board directly from images.

## Input Assumptions

The visual-inertial capture must keep the A4 board static and move the headset
camera-IMU rig. Moving the board in front of a stationary headset is not
observable for camera-IMU extrinsics because the visual target moves but the IMU
does not measure the same motion.

The current tested board is:

```text
pattern_resolution_17x24_segments_16_apriltag_0
```

The solver inputs are:

```text
<down_pair_vi_capture.mcap>
<up_pair_vi_capture.mcap>
<stage1_kb8_rig_prior.yaml>
<feature_root>/cam{0,1,2,3}_features.bin
```

Important: the old `kalibr_cam_chain_kb8_generated_20260526.yaml` field named
`T_cam_imu` is not a measured physical IMU prior. It is a camera-camera rig /
pseudo-rig prior using cam0 as the reference frame. The VI scripts use it only to
keep the four-camera rig fixed while solving the physical IMU rotation and
translation.

## Solvers

`calibrate_visual_imu_rotation_from_full_board.py` estimates camera-to-physical
IMU rotation from full-board visual poses and gyro angular velocity. It exports a
rig-consistent KB8 YAML and a raw per-camera diagnostic YAML. Translation is
still inherited from the pseudo-rig prior in this rotation-only pass.

`calibrate_visual_imu_se3_from_full_board.py` uses the rotation result,
full-board visual second derivatives, accelerometer samples, and the fixed
camera-camera rig prior to solve:

```text
t_cam0_imu
g_down, g_up
accel_bias_down, accel_bias_up
```

The SE(3) pass writes physical `T_cam_imu` and inverse `T_imu_cam` into the
driver-readable KB8 YAML. Time offset remains fixed at `0.0 s` in this pass.

## Output Artifacts

The SE(3) run writes three files with different audiences:

```text
seeker_kb8_full_board_vi_se3.yaml
  Final driver-facing calibration. Contains KB8 intrinsics, distortion, image
  resolution, rostopic, T_cam_imu, T_imu_cam, T_cn_cnm1, and timeshift_cam_imu.

full_board_vi_se3_summary.json
  Machine-readable audit record. Contains pose counts, motion sample counts,
  rank, condition number, gravity norms, accelerometer residuals, warnings, and
  output paths.

full_board_vi_se3_report.html
  Human-readable operator report. Use it to inspect visual coverage and VI solve
  quality before publishing the YAML.

seeker_vi_simple_three_scene/index.html
  Optional compact Three.js scene generated from the final KB8 YAML. Use it for
  visual sanity checks of the physical IMU frame, camera frustums, OpenCV camera
  axes, and driver-facing `T_cam_imu` / `T_imu_cam` conventions.
```

The rotation-only run also writes:

```text
seeker_kb8_full_board_vi_rig_aligned.yaml
  Intermediate YAML with physical IMU rotation alignment and fixed prior
  translation. Use it as input to the SE(3) pass, not as the final product.

seeker_kb8_full_board_vi_raw_per_camera.yaml
  Diagnostic per-camera rotation output before rig-level consistency alignment.
```

To generate the optional scene:

```bash
python applications/camera_calibration/scripts/fisheye_calibration/generate_seeker_vi_simple_three_scene.py \
  --camchain-yaml <seeker_kb8_full_board_vi_se3.yaml> \
  --output-dir <report_root>/seeker_vi_simple_three_scene \
  --viewer-assets-dir <directory_with_threejs_assets>
```

## Generic Reproduction

Run the SE(3) solve from the checked-out repo after full-board features have
been extracted:

```bash
OUT=<output_root>/full_board_vi_se3
mkdir -p "$OUT"
python applications/camera_calibration/scripts/fisheye_calibration/calibrate_visual_imu_se3_from_full_board.py \
  --prior-yaml <stage1_kb8_rig_prior.yaml> \
  --rotation-yaml <rig_aligned_rotation_yaml> \
  --rotation-summary <rotation_summary.json> \
  --feature-root <directory_with_cam_features> \
  --down-mcap <down_pair_vi_capture.mcap> \
  --up-mcap <up_pair_vi_capture.mcap> \
  --output-yaml "$OUT/seeker_kb8_full_board_vi_se3.yaml" \
  --summary-json "$OUT/full_board_vi_se3_summary.json" \
  --report-html "$OUT/full_board_vi_se3_report.html"
```

Expected 2026-05-31 SE(3) metrics:

```text
rank = 15 / 15
sample_count = 2852
sample_inlier_count = 2566
condition ~= 13.3247
gravity_norm down/up ~= 9.996 / 9.592 m/s^2
inlier accel residual median/p95 ~= 0.211 / 0.373 m/s^2
T_cam0_imu translation ~= [0.138360, -0.076104, 0.006933] m
```

## Data And Git Hygiene

Keep code and generated artifacts separate.

Commit these:

```text
Python solver scripts
README / setup notes
small tests
lightweight config templates
```

Do not commit these:

```text
MCAP captures
cam*_features.bin
per-run YAML/JSON/HTML outputs
large extracted image directories
local or remote `.local` output trees
```

When a calibration result matters, record:

```text
git commit or tag
input MCAP paths
feature-root path
prior YAML path
output YAML path
summary JSON path
report HTML path
key metrics from the summary JSON
```

## Output Checks

Before publishing a YAML:

- Confirm `T_cam_imu` and `T_imu_cam` are numerical inverses for every camera.
- Confirm the summary rank is full, the condition number is modest, and gravity
  norms are physically plausible.
- Confirm accelerometer residuals are close to the expected values above for
  this dataset.
- Confirm the report clearly states that the camera-camera rig prior is fixed
  and that this is not yet a full Kalibr-style joint nonlinear visual-inertial
  bundle adjustment.

Quality signals to inspect:

```text
valid_poses / dataset_imagesets
  How often the full board produced a usable camera pose.

motion_samples
  How many pose timestamps survived acceleration / angular-velocity filtering.

pose_rmse_median_px and pose_rmse_p95_px
  Visual frontend quality. High values usually mean detection, blur, or
  intrinsic/model mismatch problems.

rank
  The SE(3) linear system should be full rank for the enabled unknowns.

condition
  Lower is better. Very large values indicate weak motion excitation or
  degenerate geometry.

gravity_norm
  Should be close to physical gravity after allowing for accelerometer bias and
  solver noise.

inlier_residual_norm_m_s2
  Accelerometer consistency after the lever-arm solve. Compare against the
  known-good dataset metrics rather than treating one absolute threshold as
  universal.
```

## Residual Risk

The current product is a native fixed-intrinsics, fixed-camera-rig VI SE(3)
solve. It is useful as a measured physical camera-IMU transform for the current
Seeker KB8 product, but it is not yet a full continuous-time joint optimizer
over intrinsics, camera rig extrinsics, IMU bias/noise, time offset, and target
trajectory.

Known limitations:

- Intrinsics are fixed from the vision-only stage.
- Camera-camera rig extrinsics are fixed from the vision-only stage.
- `timeshift_cam_imu` is fixed at `0.0 s`.
- IMU noise parameters are not jointly estimated.
- The current solver estimates one rig-level physical IMU alignment; it does not
  independently re-fit four unrelated camera-IMU transforms.
- KB8 is the current Seeker driver-facing camera model. Omni/radtan and Kalibr
  formats are compatibility paths, not the primary product for this headset
  fisheye pipeline.
