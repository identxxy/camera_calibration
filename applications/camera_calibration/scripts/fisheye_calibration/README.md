# Fisheye Calibration Utilities

This directory contains the Seeker four-fisheye calibration tooling built around
MCAP capture preparation, first-pass intrinsic export, Seeker-driver YAML export,
assumed rig composition, and static QA/report generation.

## End-To-End Headset Calibration Flow

The headset product uses two calibration stages. These are two solve stages, not
necessarily two physical captures.

Stage 1 is vision-only four-fisheye calibration. Use frames with enough
per-camera board coverage, pose diversity, and enough multi-camera constraints
to solve:

```text
per-camera KB8 intrinsics
four-camera rig extrinsics
camera-camera rig prior
```

This stage does not require IMU data. The current Seeker driver-facing product
uses KB8 YAMLs generated from this stage.

Stage 2 is visual-inertial alignment. Use a capture where the calibration board
is static and the entire headset rig moves so the four fisheye cameras and the
IMU undergo the same physical motion. The VI stage keeps the Stage 1 intrinsics
and camera-camera rig fixed, then solves the physical IMU transform:

```text
T_cam0_imu
T_cam_i_imu for all four cameras through the fixed camera-camera rig
T_imu_cam inverses
```

The VI stage is documented in `README_visual_imu_calibration.md`. It uses the
repo C++ full-board detector as the visual frontend and Python MCAP/IMU solvers
as the inertial backend. It does not require ROS or a ROS/Kalibr Docker image
for the current native product.

Overall:

```text
One well-designed capture can be used twice:
  static board, moving headset rig
    -> Stage 1: vision-only four-fisheye intrinsics and rig
    -> Stage 2: physical camera-IMU SE(3)

Separate captures are optional:
  use them only if one capture cannot provide both vision coverage and VI motion
  excitation.
```

For a single-capture workflow, the static-board moving-rig dataset must still
contain enough board observations for every camera and enough simultaneous or
rig-connected observations to constrain the four-camera rig. If the VI capture
has good IMU motion but poor visual coverage, collect a separate vision-only
capture for Stage 1 and use the VI capture only for Stage 2.

## Capture Quality Checklist

For a single-capture vision+VI dataset:

- Keep the board fixed in the world. Move the whole headset rig, not the board.
- Keep the board visible and sharp in every fisheye camera for many viewpoints.
- Cover the image center and edges; avoid only fronto-parallel centered views.
- Include enough motion for the IMU: rotations around multiple axes, angular
  acceleration, and moderate translations. A purely slow pose sweep is weaker
  for IMU translation.
- Avoid motion so aggressive that the board is blurred or only briefly visible.
- Preserve image and IMU timestamps from the same clock source in MCAP.
- Make sure the capture contains enough rig-connected observations to constrain
  the four-camera rig, not only independent single-camera board sightings.

If one capture cannot satisfy both visual coverage and IMU excitation, split the
job into a high-coverage vision-only capture and a static-board moving-rig VI
capture.

## Output Artifacts

The driver-facing product is the Seeker KB8 YAML. Auxiliary JSON and HTML files
are for audit and operator feedback:

```text
*.yaml   final calibration consumed by driver / downstream systems
*.json   machine-readable metrics, run parameters, and quality gates
*.html   human-readable calibration report
```

Do not commit large captures or generated run artifacts such as MCAP files,
`cam*_features.bin`, or per-run output directories. Keep them under a run/output
root and record the path plus git commit/tag when the result is important.

## Vision-Only Main Flow

1. Extract and screen per-camera frames from a packed MCAP:

```bash
python applications/camera_calibration/scripts/fisheye_calibration/prepare_fisheye_intrinsics_from_mcap.py \
  --mcap /tmp/seeker_20260526_large.mcap \
  --output-root /tmp/camera_calibration_mcap/seeker_large_intrinsics_full
```

2. Run the generated calibration commands in the output directory.

3. Export native KB8 intrinsics:

```bash
python applications/camera_calibration/scripts/fisheye_calibration/export_kb8_intrinsics.py \
  --calibration-root /tmp/camera_calibration_mcap/seeker_large_intrinsics_full/calibration_kb8_firstpass \
  --output /tmp/camera_calibration_mcap/seeker_large_intrinsics_full/kb8_intrinsics_firstpass.json
```

4. Generate Seeker-compatible camchain YAMLs and QA reports:

```bash
python applications/camera_calibration/scripts/fisheye_calibration/generate_seeker_kb8_yaml.py ...
python applications/camera_calibration/scripts/fisheye_calibration/generate_seeker_kalibr_yaml.py ...
conda run -n base python applications/camera_calibration/scripts/fisheye_calibration/build_seeker_calibration_report.py ...
```

5. Generate the initial four-fisheye rig viewer:

```bash
python applications/camera_calibration/scripts/fisheye_calibration/generate_fisheye_initial_rig_viewer.py \
  --camchain-yaml <stage1_kb8_rig_prior.yaml> \
  --output-dir <report_root>/fisheye_initial_rig_viewer \
  --viewer-assets-dir <directory_with_threejs_assets>
```

The default viewer layout uses the Seeker hardware slot order `cam0=left-up`,
`cam1=left-down`, `cam2=right-down`, and `cam3=right-up`. It places the upper
pair in the upper row and the lower pair in the lower row, while preserving each
camera's orientation from the KB8 camchain. Camera local axes are interpreted as
OpenCV/CV axes: `+X` image right, `+Y` image down, and `+Z` optical forward. The
viewer converts CV coordinates to Three.js coordinates with `[x, y, z] -> [x,
-y, -z]`, then applies one global viewer-basis rotation so the lower-to-upper
camera slot offset becomes the configured viewer world-up direction. For this
fisheye schematic, `world_up_three` is `[0, -1, 0]`. Use `--layout metric` to
show the metric centers from the camchain instead.

## Files

- `prepare_fisheye_intrinsics_from_mcap.py`: MCAP frame extraction, blur/board/motion screening, calibration command generation.
- `apriltag_detect_pnm.c`: small C helper compiled against the vendored AprilTag sources.
- `export_kb8_intrinsics.py`: exports first-pass `CentralThinPrismFisheyeModel` intrinsics as KB8 JSON.
- `generate_seeker_kb8_yaml.py`: writes native KB8 Seeker-driver camchain YAML.
- `generate_seeker_kalibr_yaml.py`: writes compatibility `omni+radtan` camchain YAML.
- `compose_assumed_four_fisheye_rig.py`: builds the assumed four-camera rig from observed pair transforms and back-to-back constraints.
- `prepare_synced_pair_dirs.py`: prepares synchronized pair datasets for relative extrinsic calibration.
- `build_seeker_calibration_report.py`: generates per-capture static HTML reports and an index page.
- `generate_fisheye_initial_rig_viewer.py`: generates a Three.js four-fisheye rig viewer from a KB8 camchain.
- `test_prepare_fisheye_intrinsics_from_mcap.py`: focused unit tests for layout, sharpness, and motion gating.

In the integrated studio/headset calibration codebase, the core calibration
binary uses `--max_ba_iterations` as the generic BA speed/quality knob, plus
`--skip_bundle_adjustment` for initialization-only smoke runs.
