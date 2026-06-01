# Fast Inner/Bridge Recalibration Pipeline Wrapper

`run_inner_bridge_recalib_pipeline.py` is a thin orchestration wrapper for the
studio inner-camera plus large-marker bridge recalibration path. It is designed
for the calibration panel first: dry-run must be cheap, must not run bundle
adjustment, and must still leave stable report entrypoints.

Default t0 paths:

```bash
/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3
http://192.168.2.0:9899
```

Panel-compatible dry-run:

```bash
python3 scripts/calib/run_inner_bridge_recalib_pipeline.py \
  --stage-root /home/ubuntu/calib_data/calib_2026_05_26_jpg_v3 \
  --output-root /home/ubuntu/calib_data/calib_2026_05_26_jpg_v3/recalib_pipelines/fast_inner_bridge/latest \
  --small-marker-sequence small_marker_inner8 \
  --large-inner-marker-sequence large_marker_inner8 \
  --large-marker-sequence large_marker_bridge_all32 \
  --run-tag latest \
  --dry-run \
  --force
```

The production bridge sequence is `large_marker_bridge_all32`. The older
`large_marker_bridge_4topdown_v1` products may still be linked as historical
diagnostics, but the reproducible pipeline is now fixed around the 24 outer + 8
inner all32 manifest order.

## Bridge Capture Protocol

When the goal is to bind the top-down outer cameras (`4-1`, `4-2`, `4-3`) to
the movable inner-ring rig, use the larger low-density calibration board instead
of the AprilTag tower. The tower mainly constrains the horizontally inward
outer cameras; the top-down cameras see little or none of the vertical tower
faces, so they need a tabletop board bridge.

Capture the board around the inner working volume with these requirements:

- Keep `4-1`, `4-2`, and `4-3` active in the same synchronized sequence as the
  inner 8 cameras.
- Move the board through the desktop/workspace area, not only at the center.
- Include yaw rotation around gravity and a few moderate roll/pitch tilts so
  the bridge is not close to a planar fronto-parallel degeneracy.
- Ensure many frames are jointly visible by at least one top-down camera and
  at least three inner cameras. More joint visibility is better than simply
  accumulating single-camera detections.
- Keep the board fully or mostly inside the image for a subset of frames, but
  partial board observations are acceptable if the detector returns enough
  corners and the frame has cross-camera overlap.
- Use this bridge sequence to estimate/update extrinsics. Reuse existing
  intrinsics unless resolution, focus, lens, or distortion convention changed.

The intended solve is fixed-intrinsic first: use the current refined inner
intrinsics, the current outer/top-down intrinsics, and estimate the relative
transform between the inner rig and the outer/top-down studio frame. Joint
intrinsic/extrinsic refinement is a later validation stage, not the default
bridge solve.

## Bridge all32 camera convention

The all32 bridge manifest order is part of the contract:

- Bridge indices `0..23` are the 24 outer-ring cameras in `OUTER_CAMERAS`
  order.
- Bridge indices `24..31` are the original inner cameras `inner0..inner7`.
- Original inner camera indices remain `0..7` in the compact inner-only
  calibration products and intrinsics directory.
- Top-down outer bridge anchors are `4-1`, `4-2`, and `4-3`, which correspond
  to bridge / outer indices `9`, `10`, and `11`.

For all32 fixed-intrinsic PnP, the wrapper prepares a combined intrinsics
directory under:

```text
<output-root>/planned_inputs/bridge_all32_fixed_intrinsics
```

It copies outer intrinsics `intrinsics0.yaml..intrinsics23.yaml` from:

```text
<data-root>/whole_outer_tower/fixed_intrinsic_pnp_colmap_fallback_v1
```

and remaps compact inner intrinsics `intrinsics0_<id>.yaml..intrinsics7_<id>.yaml`
from:

```text
<data-root>/final_inner8_calibration_v1/intrinsics/small_marker_opencv_grid4_pattern3_v2
```

to `intrinsics24.yaml..intrinsics31.yaml`. This keeps the C++
`--estimate_fixed_intrinsic_rig` camera indices aligned with the all32 dataset
instead of accidentally treating the bridge as an 8-camera subset.

Inner-only fixed-intrinsic stages require the full canonical 8-camera order. If
a middle inner camera is excluded, the wrapper blocks the large-inner
initializer and small fixed-rig quality probe instead of compacting camera
indices and accidentally pairing camera `i+1` with `intrinsics{i}.yaml`.

## Design

The wrapper keeps heavy compute opt-in:

1. Reuse the current final inner8 fixed intrinsics by default.
2. Solve the production inner extrinsic baseline from `large_marker_inner8`
   with `LARGE_MARKER_PATTERN` and the fixed small-marker OpenCV intrinsics.
   When this initializer is requested or an existing wrapper output is present,
   it is the final inner baseline for downstream reports/bridge evaluation.
3. Plan a small-marker fixed-intrinsic rig estimate as a quality probe from the
   grid/subsampled small-marker dataset. It writes `camera_pnp_summary.tsv` and
   is allowed to flag disconnected cameras; it does not replace the large-inner
   baseline.
4. Plan a large-marker all32 fixed-intrinsic PnP solve, then evaluate the bridge
   using compact inner poses remapped to bridge indices `24..31` and top-down
   outer anchors `4-1/4-2/4-3` at indices `9/10/11`.
5. Generate stable entrypoint reports for panel links.

The bridge stage is currently a fixed-intrinsic PnP plus top-down anchor
evaluation/check. Full combined inner+outer bundle adjustment/refinement is
deliberately left as the next stage once this data flow and camera-index
contract are stable.

## Fast Recalib Usage

The fast path is intentionally fixed-intrinsic by default. It reuses the
existing inner intrinsics from:

```text
<data-root>/final_inner8_calibration_v1/intrinsics/small_marker_opencv_grid4_pattern3_v2
```

and only updates the inner rig extrinsics when `--run-large-inner-init` (or
`--run-stage large-inner-init`) is requested. This is the normal quick
recalibration path when lenses, resolution, focus, and distortion convention did
not change and the goal is to re-anchor moved inner cameras with a fresh
`large_marker_inner8` capture.

Use `small_marker_inner8` as a quality probe rather than the production baseline
unless you explicitly enter one of the diagnostic small-marker refinement modes.
The default `fixed_rig` probe checks connectivity and writes
`camera_pnp_summary.tsv`, but the final inner baseline remains the large-marker
fixed-intrinsic state or the configured existing prior.

When the inner rig is already acceptable and only the relation to the outer
studio frame needs refresh, run just the all32 bridge path:

```bash
python3 scripts/calib/run_inner_bridge_recalib_pipeline.py \
  --data-root /home/ubuntu/calib_data/calib_2026_05_26_jpg_v3 \
  --output-root /home/ubuntu/calib_data/calib_2026_05_26_jpg_v3/recalib_pipelines/fast_inner_bridge/latest \
  --large-marker-sequence large_marker_bridge_all32 \
  --run-large-bridge \
  --run-reports \
  --force
```

This keeps inner intrinsics fixed, uses the selected inner extrinsic prior, and
evaluates the bridge against the top-down outer anchors. Re-run
`--run-large-inner-init` first when the inner cameras were physically moved
relative to each other or the prior inner extrinsics are suspect.

The bridge evaluator now writes machine-readable quality gates into
`bridge_summary.json`:

- `quality_gates.metric_bridge`: production signal based on large-board PnP vote
  stability in metric inner-rig space. Defaults require at least 50 inner-board
  frames, median inner support of 3 cameras, each top-down outer anchor with at
  least 10 votes, max center p90 `<= 0.25 m`, max rotation p90 `<= 5 deg`, and
  non-degenerate top-down triangle area `>= 0.02 m^2`.
- `quality_gates.colmap_prior_diagnostic`: diagnostic signal only. A
  three-camera COLMAP Sim(3) alignment is weakly constrained, so a weak COLMAP
  diagnostic does not invalidate a passing metric bridge gate. It warns when the
  first-frame COLMAP top-down anchors have too few tracks, inconsistent pairwise
  scale, or large orientation residuals.

The fast pipeline surfaces these gate results in both `summary.json` and the
quality/final HTML reports.

Without `--dry-run`, heavy stages still do not execute unless an explicit switch
is passed:

- `--run-small-fixed-rig-quality`
- diagnostic `--run-small-refine`
- `--run-large-inner-init`
- `--run-large-bridge`
- `--run-reports`
- `--run-all`

The equivalent lower-level form is repeatable `--run-stage large-inner-init`,
`--run-stage small-fixed-rig-quality`, `--run-stage small-refine`,
`--run-stage large-bridge`, and `--run-stage reports`.

Requested stages use input/command fingerprints under `stage_fingerprints/`.
If a requested stage's existing output was produced with different inputs,
stride, prior paths, or command arguments, the wrapper recomputes it instead of
silently reusing the stale artifact. `--force` still forces recomputation
regardless of fingerprint state. This matters for stable `latest` roots where a
new capture may reuse the same output path.

Allow-failure diagnostic stages, such as small fixed-rig quality and bridge PnP,
write a fresh fingerprint only when the command succeeds. If the command fails
while stale output files happen to exist, the report keeps the failure status and
does not certify those stale files as current.

Inner refinement modes:

- `--inner-refine-mode fixed_rig` (default): fixed-intrinsic PnP rig quality
  probe for `small_marker_inner8`. This uses the same
  `--estimate_fixed_intrinsic_rig` binary path as the large-inner initializer,
  emits `pnp_views.tsv` and `camera_pnp_summary.tsv`, and is not the final
  inner extrinsic product.
- `--inner-refine-mode fixed`: legacy fixed-intrinsic localize-only diagnostic.
  This uses `--localize_only`; t0 tests can enter LM bad-cost rejection with
  cost 0 and fail to update, so it is no longer the default production path.
- `--inner-refine-mode joint`: fixed-intrinsic localization init on the current
  small-marker dataset, followed by joint intrinsic/extrinsic BA. Direct joint
  BA from a stale prior state is intentionally not used because it can attach
  mismatched `image_used` / dataset bookkeeping and segfault.
- `--inner-refine-mode fixed_then_joint`: explicit alias for the same two-stage
  path. Treat joint output as a diagnostic/probe until an intrinsics sanity gate
  accepts it; on the 2026-05-26 data, 3 BA iterations reduced cost but drove
  high-order OpenCV distortion terms to unphysical values.

Fast recalib frame subsampling:

- `--small-frame-stride 4` is the default. A 320-frame `small_marker` capture is
  reduced to about 80 synchronized frames before the small fixed-rig quality
  probe, which keeps the comparison practical while preserving marker motion
  coverage.
- `--large-inner-frame-stride 1` is the default for the large-marker inner
  initializer. Feature extraction is already full-frame and the estimator is
  fast; full-frame estimation improves weak-camera stability.
- `--large-frame-stride 1` is the default for all32 bridge PnP. The bridge graph
  is sensitive to camera0 support; stride 2 reduced the current 305-frame bridge
  capture to 153 frames and disconnected almost all cameras in fixed-intrinsic
  PnP.
- `--inner-joint-max-ba-iterations 3` is the default for joint probes. More
  iterations were slow on t0 and entered repeated bad-cost rejection on the
  current small-marker capture.
- `--inner-schur-mode sparse_onthefly` is the default for BA. The dense default
  can segfault in warm-start joint BA for this codebase/state combination.

## Data Quality Checks

For both small and large marker sequences, `summary.json` records:

- manifest path and existence
- manifest camera count
- scanned camera count
- frame-count min/max/spread
- usable/unusable camera counts
- common-frame count
- warnings for missing manifests, interior frame gaps, and frame-count spread
  beyond the tail-trim budget

The policy is the studio capture rule: tail-only short stops up to two frames are
acceptable and become a common-prefix trim case. Interior gaps or a spread larger
than two frames indicate likely per-camera drop, so the affected camera/sequence
should be excluded rather than partially retained.

## Outputs

Dry-run and plan-only runs always create:

- `summary.json`
- `run_manifest.json`
- `index.html`
- `quality_report/index.html`
- `final_report/index.html`
- `viewer/index.html`
- `planned_inputs/*_usable_image_directories.txt`
- `planned_inputs/*_usable_manifest.tsv`

The three report subdirectories may initially be alias/link pages. They exist so
the calibration panel can present stable links before heavy stages are run.

`summary.json` also records a `provenance` block with the exact wrapper command
line, Python executable, cwd, git branch, git commit, and dirty working-tree
status. Use it to compare recalib runs and avoid mixing calibration metrics from
different code revisions or parameter sets.

`run_manifest.json` is the compact timing source for repeated runs. It records
run start/finish time, total runtime, recalib inputs, and every stage's
`started_at`, `finished_at`, `duration_s`, command, status, log path, and notes.
The final report has a “Run Timing / Recalib Inputs” section that mirrors the
key fields so the calibration panel can quickly show which data root, small /
large-inner / bridge sequences, outer source, and stage timings produced the
current `latest` output.

For a local or t0 run, the main report entrypoints are:

```text
<output-root>/index.html
<output-root>/final_report/index.html
<output-root>/quality_report/index.html
<output-root>/viewer/index.html
<output-root>/run_manifest.json
```

When outputs are under `/home/ubuntu/calib_data`, report URLs use:

```bash
http://192.168.2.0:9899
```

Use `--http-root` and `--report-url-base` only if the t0 HTTP server mapping
changes.
