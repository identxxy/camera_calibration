# Studio Calibration Operation Commands

This is the short command map for the current t0 studio calibration workflow.
It is intentionally smaller than the full runbook in
`README_studio_calibration_pipeline.md`.

Use the panel for human-triggered runs:

```text
http://192.168.2.0:9898/
```

Use the report root for published results:

```text
http://192.168.2.0:9899/
```

## Canonical Result Paths

```text
/home/ubuntu/calib_data/current_calibration/artifacts/studio_32_cameras.yaml
/home/ubuntu/calib_data/current_calibration/reports/01_3d_viewer/index.html
```

## One Command Per Operation

Run from the repo root on t0:

```bash
cd /home/ubuntu/camera_calibration
```

### 1. Full All32 Regression

Runs outer tower refine, outer intrinsic report refresh, inner/outer bridge,
unified YAML export, correspondence JSON export, and current report publishing.

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/run_studio_calibration_pipeline.py \
  --run-tag <run_tag> \
  --output-root /home/ubuntu/calib_data/studio_calibration_runs/<run_tag> \
  --run-small-quality \
  --publish-current \
  --force
```

Dry-run the same operation without executing stage commands:

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/run_studio_calibration_pipeline.py \
  --run-tag <run_tag> \
  --output-root /home/ubuntu/calib_data/studio_calibration_runs/<run_tag> \
  --run-small-quality \
  --publish-current \
  --dry-run
```

### 2. Whole / Outer Cage Only

Use this when the outer tower capture changed and the bridge data should not be
rerun yet. The production preset is `wide200_then_gate6` and must consume the
black-tile physical-corner dataset.

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/run_studio_calibration_pipeline.py \
  --outer-only \
  --run-tag <run_tag> \
  --output-root /home/ubuntu/calib_data/studio_calibration_runs/<run_tag> \
  --outer-preset wide200_then_gate6 \
  --force
```

### 3. Large Marker / All32 Bridge

Use this after collecting `large_marker` and after a trusted outer result is
available. This is the fast path when the inner ring moved but outer cage and
outer intrinsics are still trusted.

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/run_studio_calibration_pipeline.py \
  --bridge-only \
  --run-tag <run_tag> \
  --output-root /home/ubuntu/calib_data/studio_calibration_runs/<run_tag> \
  --outer-final-pose-yaml <outer_tower_result>/camera_tr_rig_delta_refined.yaml \
  --outer-final-intrinsics-dir <outer_tower_result>/intrinsics_refined \
  --run-small-quality \
  --publish-current \
  --force
```

If the bridge run should use the default trusted outer prior already encoded in
the wrapper, omit `--outer-final-pose-yaml` and `--outer-final-intrinsics-dir`.

The bridge run also publishes the all32 camera-origin projection diagnostic:

```text
http://192.168.2.0:9899/current_calibration/reports/09_bridge_result_large_marker/camera_origin_projection/index.html
```

It projects every calibrated camera optical center into every large-marker view
image. The default overlay shows all 32 target cameras, with outer-only and
inner-only filters for debugging.

### 4. Small Marker / Inner Quality Only

Use this for inner8 quality probing without rerunning outer tower refinement.
The output is diagnostic unless you explicitly decide to promote a new inner
state.

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/run_inner_bridge_recalib_pipeline.py \
  --data-root /home/ubuntu/calib_data/calib_2026_05_31_v3 \
  --output-root /home/ubuntu/calib_data/studio_calibration_runs/<run_tag>/inner_bridge \
  --small-marker-sequence small_marker_inner8 \
  --run-stage small-fixed-rig-quality \
  --run-stage reports \
  --run-tag <run_tag> \
  --force
```

### 5. Outer Large Marker / Outer Intrinsics Refresh

This operation is infrequent. It is needed when outer lens, focus, resolution,
or distortion convention changes. It has four explicit commands because data
quality filtering and passing-image staging must run before expensive C++ board
feature extraction.

1. Run/collect W3/W4 distributed QC.

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/server_run_distributed_clients.py \
  --config <outer_large_marker_distributed_qc_config.json> \
  --output-dir /home/ubuntu/calib_data/<outer_large_marker_qc_run> \
  --run \
  --collect
```

2. Aggregate the worker output into a passing-images staging root. This is the
   step that writes the `<passing_images>/image_directories.txt` and
   `<passing_images>/manifest.tsv` consumed by the next two commands.

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/distributed_apriltag_quality_filter.py aggregate \
  --worker-output /home/ubuntu/calib_data/<outer_large_marker_qc_run> \
  --output-dir <passing_images> \
  --marker outer_large_marker \
  --stage-mode passing-images \
  --min-tags 1 \
  --link-mode copy \
  --overwrite
```

3. Extract low-density A4 board features from the staged passing images.

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/parallel_extract_features.py \
  --binary /home/ubuntu/camera_calibration/build_t0_current/applications/camera_calibration/camera_calibration \
  --repo-root /home/ubuntu/camera_calibration \
  --image-directories-file <passing_images>/image_directories.txt \
  --pattern-files applications/camera_calibration/patterns/pattern_resolution_17x24_segments_16_apriltag_0.yaml \
  --output-dataset <outer_large_marker_features.bin> \
  --work-dir <outer_large_marker_feature_shards> \
  --jobs 8 \
  --resume
```

4. Solve per-camera OpenCV intrinsics.

```bash
/home/ubuntu/miniconda3/bin/python scripts/calib/calibrate_tower_intrinsics_opencv.py \
  --dataset <outer_large_marker_features.bin> \
  --manifest <passing_images>/manifest.tsv \
  --output-dir <outer24_opencv_intrinsics_large_marker>
```

Only pass `--points-yaml <points.yaml>` when the input dataset does not already
carry version-1 known 3D point geometry. Do not pass the pattern YAML there; it
is a board-generation/extraction config, not a feature-id-to-3D-point map.

Pass the resulting intrinsics directory into later full/outer runs with
`--outer-frame-face-intrinsics-dir`.

### 6. Publish Current Reports From An Existing Run

Use this after manually regenerating viewer/report artifacts inside a run
directory.

```bash
/home/ubuntu/miniconda3/bin/python scripts/ops/publish_t0_clean_calib_reports.py \
  --root /home/ubuntu/calib_data \
  --base-url http://192.168.2.0:9899 \
  --run-tag <run_tag> \
  --current-dir /home/ubuntu/calib_data/current_calibration \
  --outer-large-intrinsic-report <run_root>/reports/outer_intrinsics_outer_large_marker \
  --outer-large-qc-root <outer_large_marker_qc_root> \
  --whole-qc-root <whole_qc_root> \
  --outer-frame-face-report-root <run_root>/outer_tower/frame_face_refine_wide200_then_gate6
```

## Command Boundary

Keep these concepts separate:

- `run_studio_calibration_pipeline.py`: orchestration for production all32
  reruns and current publishing.
- `run_outer_tower_recalib_pipeline.py`: lower-level whole/tower bootstrap and
  diagnostic wrapper.
- `run_inner_bridge_recalib_pipeline.py`: lower-level inner/large/small board
  wrapper.
- `publish_t0_clean_calib_reports.py`: the only current homepage publisher.

Do not reintroduce `report_registry.json` or old standalone report indexes as
human-facing entrypoints.
