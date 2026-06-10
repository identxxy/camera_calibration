# Calibration Panel Server

This panel is an operator UI for t0 calibration runs. It exposes a small
whitelist of run modes and stores every job under a dedicated run directory with
`job.json` and `run.log`.

## Start on t0

The long-running t0 user service is:

```text
camera-calibration-panel.service
```

It binds to `0.0.0.0:9898` and is linked from the curated report dashboard on
`9899`.

```bash
systemctl --user status camera-calibration-panel.service
systemctl --user restart camera-calibration-panel.service
```

Manual foreground start for debugging:

```bash
cd /home/ubuntu/camera_calibration
/home/ubuntu/miniconda3/bin/python scripts/calib/calibration_panel_server.py \
  --repo-root /home/ubuntu/camera_calibration \
  --runs-root /home/ubuntu/calib_data/panel_runs \
  --host 0.0.0.0 \
  --port 9898
```

Open the panel:

```text
http://<t0-ip>:9898/
```

## Report Dashboard Interface

The report dashboard is a separate read-only service:

```text
http://<t0-ip>:9899/
```

On the current t0 camera LAN this is:

```text
http://192.168.2.0:9899/
```

The dashboard publishes complete `http://192.168.2.0:9899/...` report links so
operators can copy URLs directly from the page. The operation panel defaults to:

```text
http://192.168.2.0:9898/
```

The final report interface is intentionally compact:

1. Final YAML: `current_calibration/artifacts/studio_32_cameras.yaml`.
2. Overall 3D viewer: `current_calibration/reports/01_3d_viewer/index.html`.
3. Inner data capture report: `02_inner_capture_small_marker`.
4. Inner intrinsic report: `03_inner_intrinsics_small_marker`.
5. Inner extrinsic report: `04_inner_extrinsics_small_marker`.
6. Outer data capture report: `05_outer_capture_outer_large_marker_whole`.
7. Outer intrinsic report: `06_outer_intrinsics_outer_large_marker`.
8. Outer extrinsic report: `07_outer_extrinsics_whole`.
9. Bridge result report: `09_bridge_result_large_marker`.

The root dashboard should not promote dated scratch reports, raw pipeline
directories, source/debug viewers, standalone correspondence viewers,
registry/debug JSON files, or operation buttons as extra homepage groups.
Operation launch is handled by the 9898 panel.

Production pipeline entrypoint:

- Studio 32-camera production pipeline:
  `http://192.168.2.0:9898/?mode=run_studio_calibration_pipeline`
  calls `scripts/calib/run_studio_calibration_pipeline.py`. It is the preferred
  operator path after data has been QC/staged, because it runs the current
  outer frame-face refine, large-marker bridge, unified 32-camera export, and
  optional 9899 publication in one provenance-tracked wrapper.

Per-operation and diagnostic entrypoints:

- Production whole outer cage panel:
  `http://192.168.2.0:9898/?mode=operate_whole_outer_cage`
- Production large-marker bridge panel:
  `http://192.168.2.0:9898/?mode=operate_large_marker_bridge`
- Production small-marker inner panel:
  `http://192.168.2.0:9898/?mode=operate_small_marker_inner`
- Current public reports:
  `http://192.168.2.0:9899/`
- Diagnostic inner/bridge wrapper panel:
  `http://192.168.2.0:9898/?mode=run_inner_bridge_recalib_pipeline`
- Diagnostic/full outer tower panel:
  `http://192.168.2.0:9898/?mode=run_outer_tower_recalib_pipeline`

Historical direct links under dated run directories, including the 2026-05-26
fast inner/bridge and outer tower reports, are legacy diagnostics only. The
operator-facing current result is always the curated 9899 root.

## Run Modes

- `run_studio_calibration_pipeline`: calls
  `scripts/calib/run_studio_calibration_pipeline.py`. This is the current
  production wrapper for reproducible all32 calibration runs. It defaults to
  whole data under
  `/home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1`, inner/bridge
  data under `/home/ubuntu/calib_data/calib_2026_05_31_v3`, and the
  `wide200_then_gate6` outer preset. Production whole/tower BA must use the
  black-tile physical-corner dataset
  `opencv_tower_dataset_black_tile_red_scale_edge.bin`; raw OpenCV AprilTag
  detector corners are legacy diagnostics only. Panel dry-run is enabled by
  default, and publication to the 9899 current entry requires the explicit
  `Publish current 9899 entry` field.
- `run_inner_bridge_recalib_pipeline`: calls
  `scripts/calib/run_inner_bridge_recalib_pipeline.py`. It writes to
  `/home/ubuntu/calib_data/studio_calibration_runs/latest_inner_bridge`
  by default and passes `--dry-run` unless the operator disables that field.
  Its default production path is
  `large-inner init + small fixed-rig quality + all32 bridge solve`: solve the
  final fixed-intrinsic inner extrinsic baseline from `large_marker_inner8` at
  frame stride 1, then run a `small_marker_inner8` fixed-intrinsic rig estimate
  only as a quality probe, and evaluate/solve the `large_marker_bridge_all32`
  bridge. The probe emits `camera_pnp_summary.tsv`; disconnected cameras are
  report flags and do not replace the large-inner baseline. Legacy `fixed`
  localize-only, `joint`, and `fixed_then_joint` small-marker refinement modes
  are exposed only as explicit diagnostics. `joint` now first builds a
  fixed-localize warm-start for the current small-marker dataset; direct joint
  BA from a stale state is avoided because it can segfault. The old `fixed`
  localize-only path can stall in LM bad-cost rejection on this dataset and is
  not the default. Joint diagnostics default to 3 BA iterations; longer runs
  were slow and produced unphysical high-order OpenCV distortion on the
  2026-05-26 capture.
  The bridge path uses `large_marker_bridge_all32` with the manifest convention
  outer cameras `0..23` followed by inner cameras `24..31`. The original inner
  camera products still use compact inner indices `0..7`; the pipeline remaps
  those inner intrinsics/poses to bridge indices `24..31` only for the all32
  bridge. The top-down anchor cameras are `4-1`, `4-2`, and `4-3`, corresponding
  to bridge indices `9`, `10`, and `11`, but they are not the only bridge
  anchors. The current bridge product is an all32 PnP initializer followed by
  direct all32 joint BA with known board points fixed. The final bridge quality
  comes from the BA correspondence residual TSV; the top-down bridge metric is
  kept only as a legacy diagnostic.
- `operate_whole_outer_cage`: calls
  `scripts/calib/run_outer_tower_recalib_pipeline.py` through the production
  operation alias. Its browser form defaults match the current production whole
  operation: frame-face refine with preset `wide200_then_gate6`, quality/final
  reports enabled, and COLMAP vote, side-prior, old tag-refine, and per-stage
  viewer disabled unless explicitly requested by the operator.
- `run_outer_tower_recalib_pipeline`: calls
  `scripts/calib/run_outer_tower_recalib_pipeline.py`. It writes to
  `/home/ubuntu/calib_data/studio_calibration_runs/latest_outer_tower`
  by default and passes `--dry-run` unless the operator disables that field.
  This mode remains a diagnostic/full bootstrap entrypoint: its browser form
  exposes and defaults to the older COLMAP vote, side-prior, tag-refine, and
  viewer stages so an operator can deliberately rerun the bootstrap family.
  Use `operate_whole_outer_cage` for the current production whole operation.
  The panel exposes `tag_intrinsics_refine_mode` for diagnostic outer
  intrinsic+extrinsic joint probes (`fixed`, `shared_fxfy`,
  `per_camera_fxfy`, `per_camera_fxfycxcy`). The default remains `fixed`;
  intrinsic updates are accepted per camera only if their prior gates pass. The
  default sparse-camera tag thresholds are `min use = 16` and `min delta = 10`,
  matching the current best weighted tag-refine result.
- `stage_data`: runs `scripts/ops/t0_stage_current_calib_data.py`.
- `distributed_qc`: runs `scripts/calib/server_run_distributed_clients.py`.
- `inner_warm_start_refine`: extracts small-marker features, grid-subsamples the
  dataset, refines from the saved inner warm-start state, then generates
  reprojection, rig, and interactive Three.js reports.
- `report_only`: builds reprojection, rig, and interactive Three.js reports from
  an existing dataset and state.

Use the panel's `Dry run` checkbox first. Dry runs write the exact argv commands
to `run.log` without executing them.

Pipeline reports also write exact run provenance into each `summary.json` and
HTML report: command line, Python executable, cwd, git branch, git commit, and
dirty working-tree status. When an operator compares two reports, check this
block first.

`report_only` and `inner_warm_start_refine` need the Three.js viewer assets
`three.min.js`, `OrbitControls.js`, and `TransformControls.js`. The default
asset directory points at the saved t0 interactive report snapshot:

```text
/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3/final_inner8_calibration_v1/reports/interactive_rig_viewer_v1
```

If that snapshot is moved, set `Three.js assets dir` in the panel to any
directory containing those three files.

## Safety Model

- The checked-in service binds to `0.0.0.0` on the private calibration network.
- Browser requests can start only named run modes. They cannot submit shell text.
- Commands are executed as argv lists with `shell=False`.
- Each job has its own run directory:

```text
<runs-root>/<timestamp>_<mode>_<id>/
  job.json
  run.log
```

Generated reports are linked from the job detail panel. The server also proxies
report files back through localhost so relative HTML assets work from the panel.
