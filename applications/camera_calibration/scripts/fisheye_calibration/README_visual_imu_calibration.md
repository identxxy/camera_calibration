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

## Input Assumptions

The visual-inertial capture must keep the A4 board static and move the headset
camera-IMU rig. Moving the board in front of a stationary headset is not
observable for camera-IMU extrinsics because the visual target moves but the IMU
does not measure the same motion.

The current tested board is:

```text
pattern_resolution_17x24_segments_16_apriltag_0
```

The current tested T0 inputs are:

```text
/home/ubuntu/camera_calibration/.local/seeker_vi_kalibr_20260530/data/seeker_20260530_down_vi_calib.mcap
/home/ubuntu/camera_calibration/.local/seeker_vi_kalibr_20260530/data/seeker_20260530_up_vi_calib.mcap
/home/ubuntu/camera_calibration/.local/seeker_vi_kalibr_20260530/inputs/kalibr_cam_chain_kb8_generated_20260526.yaml
/home/ubuntu/camera_calibration/.local/seeker_vi_kalibr_20260530/outputs/full_board_features_20260531/cam{0,1,2,3}_features.bin
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

## Reproduction On T0

Run the SE(3) solve from the checked-out repo while mounting the existing T0
feature and MCAP data:

```bash
BASE=/home/ubuntu/camera_calibration/.local/seeker_vi_kalibr_20260530
OUT=$BASE/outputs/full_board_vi_se3_20260531_merge_verify
mkdir -p "$OUT"
docker run --rm --entrypoint python3 \
  -v /home/ubuntu/camera_calibration:/workspace/camera_calibration \
  -w /workspace/camera_calibration \
  seeker-kalibr-mcap:noetic \
  applications/camera_calibration/scripts/fisheye_calibration/calibrate_visual_imu_se3_from_full_board.py \
    --prior-yaml /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/inputs/kalibr_cam_chain_kb8_generated_20260526.yaml \
    --rotation-yaml /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/outputs/full_board_vi_rotation_20260531/seeker_kb8_full_board_vi_rig_aligned.yaml \
    --rotation-summary /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/outputs/full_board_vi_rotation_20260531/full_board_vi_summary.json \
    --feature-root /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/outputs/full_board_features_20260531 \
    --down-mcap /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/data/seeker_20260530_down_vi_calib.mcap \
    --up-mcap /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/data/seeker_20260530_up_vi_calib.mcap \
    --output-yaml /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/outputs/full_board_vi_se3_20260531_merge_verify/seeker_kb8_full_board_vi_se3.yaml \
    --summary-json /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/outputs/full_board_vi_se3_20260531_merge_verify/full_board_vi_se3_summary.json \
    --report-html /workspace/camera_calibration/.local/seeker_vi_kalibr_20260530/outputs/full_board_vi_se3_20260531_merge_verify/full_board_vi_se3_report.html
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

## Residual Risk

The current product is a native fixed-intrinsics, fixed-camera-rig VI SE(3)
solve. It is useful as a measured physical camera-IMU transform for the current
Seeker KB8 product, but it is not yet a full continuous-time joint optimizer
over intrinsics, camera rig extrinsics, IMU bias/noise, time offset, and target
trajectory.
