# Fisheye Calibration Utilities

This directory contains the Seeker four-fisheye calibration tooling built around
MCAP capture preparation, first-pass intrinsic export, Seeker-driver YAML export,
assumed rig composition, and static QA/report generation.

## Main Flow

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

## Files

- `prepare_fisheye_intrinsics_from_mcap.py`: MCAP frame extraction, blur/board/motion screening, calibration command generation.
- `apriltag_detect_pnm.c`: small C helper compiled against the vendored AprilTag sources.
- `export_kb8_intrinsics.py`: exports first-pass `CentralThinPrismFisheyeModel` intrinsics as KB8 JSON.
- `generate_seeker_kb8_yaml.py`: writes native KB8 Seeker-driver camchain YAML.
- `generate_seeker_kalibr_yaml.py`: writes compatibility `omni+radtan` camchain YAML.
- `compose_assumed_four_fisheye_rig.py`: builds the assumed four-camera rig from observed pair transforms and back-to-back constraints.
- `prepare_synced_pair_dirs.py`: prepares synchronized pair datasets for relative extrinsic calibration.
- `build_seeker_calibration_report.py`: generates per-capture static HTML reports and an index page.
- `test_prepare_fisheye_intrinsics_from_mcap.py`: focused unit tests for layout, sharpness, and motion gating.

In the integrated studio/headset calibration codebase, the core calibration
binary uses `--max_ba_iterations` as the generic BA speed/quality knob, plus
`--skip_bundle_adjustment` for initialization-only smoke runs.
