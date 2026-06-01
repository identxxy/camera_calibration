# t0 Calibration Report Contract

This document defines the human-facing report entry and the producer contract for
t0 calibration reports.

## Human Entry

Use only this URL as the stable report entry:

```text
http://192.168.2.0:9899/
```

The root URL is served from the implementation file
`/home/ubuntu/calib_data/current_calibration/index.html`. The explicit
`/current_calibration/index.html` path remains valid as a compatibility URL,
but user-facing replies should point to the root URL.

The page is organized only by canonical human report categories:

1. `inner capture QC`: `small_marker` and `large_marker` calib board data
   collection reports.
2. `inner solve result`: inner8 3D viewer and reprojection/final solve report.
3. `outer capture QC`: `whole` / tower AprilTag data collection reports.
4. `outer solve diagnostics/result`: outer tower frame-face refine and COLMAP
   audit diagnostics.
5. `combined bridge / 32-camera result`: unified 3D viewer and
   `studio_32_cameras.yaml`.

The report entry must not add extra top-level groups such as dated scratch
reports, raw pipeline directories, source/debug viewers, or ad hoc operation
buttons. Operation pages remain registered in `report_registry.json`, but they
are supporting backend entries rather than report categories.

Each category can contain multiple concrete artifacts, but every promoted link
must answer one of the canonical category questions above. For example,
`whole` distributed QC logs belong under outer capture QC as diagnostics; they
must not become their own homepage group.

The final viewer area is separate. Its target contract is one canonical 3D
viewport with three modes:

- combined inner + outer
- inner only
- outer only

The current implementation uses one canonical combined viewer. Inner-only and
outer-only are UI modes inside that same viewer; source/debug viewers should not
be promoted as top-level human report links.

## Current Registry

The machine-readable registry lives at:

```text
/home/ubuntu/calib_data/current_calibration/report_registry.json
```

It is the source of truth for the current human entry. UI code should read this
registry instead of globbing arbitrary report files.

Current schema:

- `canonical_report_categories`: the five homepage categories, in display
  order.
- `report_groups`: compatibility alias for `canonical_report_categories`.
- `operation_entries`: supporting post-capture operation pages and 9898 panel
  modes.
- `final_viewer`: canonical unified viewer and
  `studio_32_cameras.yaml` URLs.

## Operation Contract

Every capture data type must have one human Operation page:

- `whole`: process the whole capture into outer camera cage calibration.
- `large marker`: process large-marker captures into the inner/outer bridge.
- `small marker`: process small-marker captures into inner camera calibration.

The Operation page is allowed to show buttons, but command execution belongs to
the controlled backend panel, currently served from:

```text
http://192.168.2.0:9898/
```

The report HTML must not embed arbitrary shell commands. Backend execution must
go through whitelisted CLI modes owned by the panel/server code.

The root index must expose the three processing entries directly:

```text
http://192.168.2.0:9898/?mode=operate_whole_outer_cage
http://192.168.2.0:9898/?mode=operate_large_marker_bridge
http://192.168.2.0:9898/?mode=operate_small_marker_inner
```

These entries are independent. A user who only re-captures `whole` should use
the `whole` operation; a user who only re-captures `large_marker` should use the
large-marker bridge operation; a user who only re-captures `small_marker` should
use the small-marker inner operation.

Target clean CLI shape:

```text
t0-calib operate whole --capture-root <whole_capture_root> --output-root <run_output_root> --publish-current
t0-calib operate large-marker --inner-sequence <large_marker_inner8> --bridge-sequence <large_marker_bridge_all32> --publish-current
t0-calib operate small-marker --inner-sequence <small_marker_inner8> --output-root <run_output_root> --publish-current
```

Current backend mapping:

- `whole` panel mode `operate_whole_outer_cage` uses `scripts/calib/run_outer_tower_recalib_pipeline.py`.
- `large marker` panel mode `operate_large_marker_bridge` currently uses `scripts/calib/run_inner_bridge_recalib_pipeline.py`.
- `small marker` panel mode `operate_small_marker_inner` currently uses `scripts/calib/run_inner_bridge_recalib_pipeline.py`.

The current one-command operator wrapper is:

```text
python3 scripts/calib/run_studio_calibration_pipeline.py --run-tag latest --force --publish-current
```

It calls the outer tower wrapper, the inner/bridge wrapper, then this current
entry builder with dynamic `--current-bridge-run-rel` and
`--current-outer-run-rel` values.

The last two mappings are transitional. The clean backend should split
small-marker inner calibration and large-marker bridge into separate user-facing
CLI modes, even if they share internal code.

Operation pages are not homepage report categories. The stable homepage should
point a human to report conclusions first; backend execution details live in
the registry, README, and 9898 panel.

## Producer Rule

Calibration algorithms may write outputs inside their own pipeline/run
directories, for example:

```text
/home/ubuntu/calib_data/<capture_root>/recalib_pipelines/<pipeline_id>/runs/<run_id>/
```

or an existing compatibility path such as:

```text
/home/ubuntu/calib_data/<capture_root>/recalib_pipelines/<pipeline_id>/latest/
```

They should not add arbitrary human-facing links to the main t0 entry.

Each production-capable run should write an artifact manifest with these fields:

```json
{
  "pipeline_id": "fast_inner_bridge",
  "run_id": "latest",
  "created_at": "2026-06-01T00:00:00Z",
  "input_datasets": {},
  "artifacts": {},
  "quality_gates": {},
  "recommended_for_humans": true
}
```

Only the report/UI owner should promote artifacts from pipeline run directories
into `current_calibration/report_registry.json`.

## Repository Hygiene

Generated report HTML should not live under source/script directories in this
repository. Acceptable locations are:

- t0 report output roots such as `/home/ubuntu/calib_data/...`.
- pipeline run directories under those report roots.
- repo-local research archives under `studio/exp/` or `studio/archive/` when a
  local generated artifact must be preserved for discussion.

If a generated HTML report is found under `scripts/` or
`applications/.../scripts/`, move it to an appropriate `studio/exp` archive
instead of deleting it, unless the owner explicitly says it is disposable.
Static application assets, such as an operator panel's checked-in
`panel_static/index.html`, are not generated calibration reports.

## 2026-06-01 Cleanup Boundary

The active t0 report server root is `http://192.168.2.0:9899/`, backed by
`/home/ubuntu/calib_data/current_calibration/index.html` and
`/home/ubuntu/calib_data/current_calibration/report_registry.json`.

Historical report clutter from report audits, panel dry-runs, smoke tests,
timed regressions, and early outer-tower experiments was moved to:

```text
/home/ubuntu/calib_data/archive/report_artifacts_20260601/
```

This archive intentionally excludes raw capture data, staged current data,
`current_calibration`, and the promoted `studio_32_cameras.yaml`.

## Current Semantics

`whole` means the whole-studio / outer-tower capture path. Its primary purpose
is to calibrate the overall studio cage, namely the `outer24` camera cage.

`large marker` means the large board path whose primary purpose is to bridge
inner cameras and outer cameras. The current implementation contains:

- `large_marker_inner8`: inner baseline with fixed intrinsics.
- `large_marker_bridge_all32`: outer24 + inner8 bridge. The current bridge
  contract is outer camera indices `0..23`, inner camera indices `24..31`, with
  top-down bridge anchors `4-1`, `4-2`, and `4-3` at indices `9`, `10`, `11`.

`small marker` means inner-only calibration/quality. Its primary purpose is to
calibrate the inner cameras. It must not be used as an outer bridge input unless
a future contract explicitly changes this.

## Standard Final Report Draft

The following is the current requirement-side draft. It is intentionally short
and should be tightened into quality gates once the human review loop settles.

### Whole / Outer Cage

- Per-machine and per-outer-camera capture counts, tag detection rates, and
  accepted frame set.
- Outer24 optimized pose/intrinsics version, reprojection residuals, rejected
  frames, and rejected cameras.
- Outer cage geometry sanity checks: camera directions, top-down cameras,
  ring consistency, and final outer-only viewer.

### Small Marker / Inner

- Inner8 per-camera small-marker coverage, corner counts, and accepted frames.
- Inner intrinsics/extrinsics, distortion sanity, and per-camera reprojection
  residuals.
- Explicit conclusion for weak or disconnected cameras, plus final inner-only
  viewer.

### Large Marker / Bridge

- Bridge input contract: outer/inner camera index order, bridge anchors, and
  accepted frames.
- Inner-to-outer transform, top-down/outer anchor vote counts, and metric
  residual gate.
- Bridge pass/fail hard gates, caveats, and final combined viewer.
