# Outer Tower Recalibration Pipeline Wrapper

`run_outer_tower_recalib_pipeline.py` is a thin orchestration layer around the
current outer AprilTag-tower recalibration tools:

1. `dataset_coverage_report.py`
2. `run_outer_colmap_frame_vote.py`
3. `vote_outer_colmap_runs.py`
4. `complete_outer_rig_side_prior.py`
5. `refine_outer_tower_frame_face_planes.py`
6. `generate_outer_frame_face_report.py`
7. `generate_outer_colmap_scene_viewer.py` for archived bootstrap audits

Default paths target the current t0 dataset:

```bash
/home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1
```

If that path is missing, the wrapper still writes `summary.json` and
`index.html` with missing-input status instead of failing during path inference.

Dry-run locally:

```bash
python3 scripts/calib/run_outer_tower_recalib_pipeline.py \
  --dry-run \
  --sample-count 2 \
  --run-colmap-vote \
  --run-side-prior \
  --run-tag-refine
```

Run on t0 with the existing default data layout:

```bash
python3 scripts/calib/run_outer_tower_recalib_pipeline.py \
  --sample-count 32 \
  --colmap-jobs 1 \
  --run-colmap-vote \
  --run-side-prior \
  --run-tag-refine
```

Heavy recomputation is opt-in. Without `--run-colmap-vote`,
`--run-side-prior`, or `--run-tag-refine`, the wrapper reports existing inferred
outputs when present and skips those stages.

## Distributed Whole-Tower QC

The current 2026-05-29 whole-tower capture should use the hybrid distributed
QC config:

```text
configs/distributed_whole_2026_05_29_filter_hybrid.json
```

Most cameras use OpenCV AprilTag detection at `--resize-factor 0.5` for speed.
The oblique/weak tower views `4-3` and `22587611` are explicit exceptions and
must use `--resize-factor 1.0`. A full-resolution probe recovered enough
observations for both cameras (`4-3`: 132 frames with at least four tags;
`22587611`: 41 frames), while the 0.5x pass reported zero qualifying frames.

When merging worker outputs, the aggregate step chooses the best metric row for
each `(camera, time, frame)` before computing frame gates and per-camera stats.
This keeps temporary full-resolution rescans from double-counting frames if a
half-resolution output is also passed for the same camera.

Keep each worker `detections.jsonl` in production full-resolution runs; do not
pass `--skip-detections-jsonl` unless the run is QC-only. After
`distributed_apriltag_quality_filter.py aggregate` writes the staged
`manifest.tsv`, `image_directories.txt`, `selected_frames.tsv`, and
`selected_images.tsv`, build the BA `calib_data` binary from cached detections
while re-reading only the staged/selected original-resolution images on t0 for
black-tile outer-corner refinement. This avoids re-running AprilTag detection on
t0, but still replaces the OpenCV inner detector corners with physical 8 cm
black-tile outer corners:

```bash
python3 scripts/calib/build_apriltag_tower_dataset_opencv.py \
  --image-directories-file /home/ubuntu/calib_data/<stage>/image_directories.txt \
  --input-detections-jsonl /home/ubuntu/calib_data/<worker-output>/detections.jsonl \
  --selected-images-tsv /home/ubuntu/calib_data/<stage>/selected_images.tsv \
  --black-tile-corner-refine red-scale-edge \
  --output-dataset /home/ubuntu/calib_data/<stage>/opencv_tower_dataset_black_tile_red_scale_edge.bin \
  --tower-config applications/camera_calibration/patterns/apriltag_tower_8faces_2x16_8cm.yaml \
  --summary-json /home/ubuntu/calib_data/<stage>/opencv_tower_dataset_black_tile_red_scale_edge_summary.json \
  --per-camera-tsv /home/ubuntu/calib_data/<stage>/opencv_tower_dataset_black_tile_red_scale_edge_per_camera.tsv \
  --detections-tsv /home/ubuntu/calib_data/<stage>/opencv_tower_dataset_black_tile_red_scale_edge_detections.tsv
```

The builder filters detections to tower-valid tag IDs, uses OpenCV corners only
as red-box scale priors, keeps partial corner observations instead of fabricating
missing tag corners, and writes the same version-1 dataset format as the rest of
the calibration tools. `build_apriltag_tower_dataset_from_detections.py` remains
useful for QC-only / legacy raw-corner diagnostics, but should not be the
production whole BA dataset builder.

On the 2026-05-31 whole recapture, a full-resolution QC probe recovered many
more candidate frames than the hybrid QC cache: 780 selected synchronized frames
and 8148 passing images, versus the older 249 selected frames. Rebuilding the
tower dataset from t0 by re-reading SMB 4K images took about 27.8 minutes for
24 cameras x 780 frames, so production runs should keep worker
`detections.jsonl` and use the direct-detections builder above.

The same probe also shows why the initialization BA gate must be loose when a
coarse-but-not-perfect outer prior is available. With the required
`opencv_tower_dataset_black_tile_red_scale_edge.bin` physical-corner dataset,
`wide50_then_gate6` kept only 4856 observations in the initialization pass and
1966 observations in the final `6 px` pass. `wide200_then_gate6` kept 68235
initial observations, retained 3583 final observations, and reached final
median/p90 `0.43 / 2.88 px` on the 2026-05-31 whole capture. The root issue was
not the final `6 px` quality gate; it was the earlier `50 px` gate deleting
valid same-ID observations before BA had a chance to correct the coarse pose.

Production outer-tower BA must therefore use:

- `opencv_tower_dataset_black_tile_red_scale_edge.bin`.
- `wide200_then_gate6` as the default refine preset.
- `wide50_*` presets only as legacy/diagnostic comparisons.

AprilTag corner localization is expected to be subpixel. The Python/OpenCV
distributed QC and dataset builder paths default to ArUco subpixel refinement
and then run `cv2.cornerSubPix()` on the original full-resolution grayscale
image after resize scaling. The C++ `AprilTagTowerDetector` uses AprilTag
`refine_edges = 1` and now also supports a conservative OpenCV
`cv::cornerSubPix()` post-refine controlled by the tower YAML keys
`corner_subpixel_refinement`, `corner_subpixel_window_half_extent`,
`corner_subpixel_max_iterations`, `corner_subpixel_epsilon`, and
`corner_subpixel_max_shift_px`.

Do not assume that turning on a larger subpixel window automatically improves
the final BA. On the 2026-05-29 616-frame dataset, the naive Python/OpenCV
`cornerSubPix` full extraction reached median/p90 `3.49 / 8.24 px`, while the
C++ AprilTag edge-refined dataset with the same independent frame-face model and
`10 px` gate reached `3.18 / 6.88 px`. The next accuracy path is
detector-consistent edge-line fitting and stricter per-tag quality gates, not
just more BA iterations or a larger `cornerSubPix` window.

For the current 2026-05-31 all32 recapture, the default production path is the
rigid yaw-45 tower refine with the `wide200_then_gate6` preset:

```bash
python3 scripts/calib/run_outer_tower_recalib_pipeline.py \
  --data-root /home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1 \
  --output-root /home/ubuntu/calib_data/studio_calibration_runs/<run-tag>/outer_tower_wrapper \
  --frame-face-output-dir /home/ubuntu/calib_data/studio_calibration_runs/<run-tag>/outer_tower/frame_face_refine_wide200_then_gate6 \
  --run-frame-face-refine \
  --frame-face-refine-preset wide200_then_gate6
```

This preset keeps the same synchronized frame/face model but uses a two-stage
observation gate. The first pass keeps observations within `200 px` of the
coarse-prior projection so weak-but-valid cameras can get an SE(3) delta. The
second pass re-gates from that warm start at `6 px` and writes the accepted
output. The model is no longer an unconstrained independent plane per
`(frame, face)`: each synchronized frame owns one `rig_tr_tower` pose, all faces
share one tower-local vertical axis, and adjacent face normals are fixed at
`360/8 = 45` degrees. The within-face corner coordinates use physical black-tile
outer corners: AprilTag detection provides the tag ID and red-box scale prior,
then the production dataset builder refines the 8 cm black-tile outer corners
with local corner/edge support. The less reliable physical face width is only
used to place different faces around the tower; within-face tag coordinates are
defined by the measured `8 cm` tag size and `2 cm` spacing.

The current studio tower has a critical geometry distinction:

- physical printed black-tile footprint: `8 cm` with `2 cm` tile gap;
- OpenCV AprilTag detector corner square: an inner square used only for ID
  detection and red-box scale prior;
- production BA geometry: `--tower_tag_size_m 0.08` and
  `--tower_tag_spacing_m 0.02`.

The older detector-square ratio `detected tag side / adjacent tag center pitch`
has median `0.671040859` in the real 2026-05-31 detections, corresponding to
`0.06710408594834662 / 0.03289591405165339`. That geometry is legacy
diagnostic-only for datasets that still store raw OpenCV inner detector corners.
Do not use it with the red-scale-edge black-tile outer-corner dataset.

The older independent frame-face model remains useful as a diagnostic/fallback
because it can expose low per-plane reprojection error, but it is not a valid
final geometry by itself: different faces in the same synchronized frame may
drift apart unless the rigid tower constraint is enabled.

On the 2026-06-10 black-tile rerun over the 2026-05-31 full-resolution whole
dataset, `wide200_then_gate6` kept 3583 accepted observations at median/p90
`0.43 / 2.88 px`, optimized face width to about `0.246 m`, and kept 20/24 outer
cameras active. The inactive cameras were `4-1`, `4-2`, `5-3`, and `7-3`; the
top-down `4-*` cameras are bridge-only for tower data, while `5-3` and `7-3`
remain weakly supported in the strict `6 px` accepted set. A bounded flexible
face-geometry probe kept 3742 observations at median/p90 `0.50 / 3.24 px`, so
the rigid yaw-45 model remains the production default and the flexible model is
diagnostic.

`--run-colmap-vote` can be parallelized with `--colmap-jobs N`. The default is
`1` for conservative memory/CPU usage because each job loads 24 synchronized 4K
images and runs SIFT/matching/mapper. On t0, `run_outer_colmap_frame_vote.py`
auto-detects `/home/ubuntu/miniconda3/envs/colmap4/bin/colmap` when `colmap` is
not in the non-interactive SSH `PATH`.

For tag refinement, the current default is
`--tag-intrinsics-mode colmap_fixed`. On the 2026-05-26 sparse `whole` capture,
this keeps the same active frames as the mixed CentralOpenCV sparse tower
intrinsics, accepts one more camera, and reduces p90 residuals. Use
`--tag-intrinsics-mode central_opencv` only when the tower capture has enough
coverage to trust per-camera tower intrinsics.

Intrinsic refinement inside the tag stage is now a separate opt-in switch. The
default remains fixed intrinsics:

```bash
--tag-intrinsics-refine-mode fixed
```

For a lightweight joint intrinsic+extrinsic pass around an existing outer pose
prior, use one of:

```bash
--tag-intrinsics-refine-mode shared_fxfy
--tag-intrinsics-refine-mode per_camera_fxfy
--tag-intrinsics-refine-mode per_camera_fxfycxcy
```

These modes are diagnostic or initialization modes when distortion is not also
being optimized. The production wrapper keeps the default final selection on
fixed-intrinsic frame-face refinement unless a non-fixed diagnostic result is
explicitly promoted after passing the quality gates.

The lower-level refine script also has a full per-camera OpenCV5 diagnostic
mode:

```bash
--tag-intrinsics-refine-mode per_camera_opencv5
```

This refines per-camera `fx/fy/cx/cy/k1/k2/p1/p2/k3` together with camera SE(3)
extrinsics, synchronized tower poses, and optionally the global tower face width.
It is a controlled experiment path, not the normal production final result. The
update is strongly regularized by default with
`--tag-intrinsics-focal-sigma-frac 0.01`,
`--tag-intrinsics-principal-sigma-px 8.0`, and
`--tag-intrinsics-distortion-sigma 0.05`. The per-block step limits
(`--tag-intrinsics-max-focal-step-frac`,
`--tag-intrinsics-max-principal-step-px`,
`--tag-intrinsics-max-distortion-step`) bound a single intrinsic optimizer
update, not the total drift accumulated across outer/post-refine iterations or
per-camera blocks. Joint intrinsic modes therefore also use total trust regions:
`--tag-intrinsics-max-total-focal-delta-frac 0.02` hard-clamps the log-focal
delta so final `fx/fy` stay within 2% of the prior,
`--tag-intrinsics-max-total-principal-delta-px 16.0` hard-clamps `cx/cy` delta,
and `--tag-intrinsics-max-total-distortion-delta` can hard-clamp each OpenCV5
distortion delta when nonzero.
The standalone refine script keeps these total clamps disabled by default
(`0`) unless they are explicitly passed.

Legacy raw OpenCV AprilTag tower datasets had an additional corner-convention
gate. The fixed-intrinsic PnP stage could test `--fixed_rig_corner_id_offset
0..3`, but `refine_outer_tower_delta_prior.py` had no runtime corner-offset
argument: it used `feature_id -> known_points[feature_id]` directly. Therefore
the selected corner offset had to be materialized before tag refine:

```bash
python3 scripts/calib/remap_apriltag_tower_dataset_corners.py \
  --input-dataset opencv_tower_dataset_fullres.bin \
  --output-dataset opencv_tower_dataset_fullres_corner_offset2.bin \
  --corner-id-offset 2
```

Do not apply this remap to the production
`opencv_tower_dataset_black_tile_red_scale_edge.bin` dataset. The black-tile
builder writes the physical 8 cm tile corners directly with the correct
`feature_id = tag_id * 4 + corner_id` convention.

Do not use a PnP-generated `camera_tr_rig.yaml` as the tag-refine camera prior.
The PnP output is a diagnostic source for per-frame tower poses and corner
offset selection. The camera prior for delta refine must come from the stable
coarse rig family, currently the RANSAC/side-prior result, and final reports
must visualize `camera_tr_rig_delta_refined_accepted.yaml` rather than the
ungated candidate pose file.

Cameras whose pose or intrinsic delta fails acceptance keep their prior
intrinsics in `intrinsics_refined_accepted/`. Diagnostics are written to
`diagnostics/camera_intrinsics.tsv`, and `summary.json` records intrinsic delta
maxima plus accepted/fallback cameras. The final HTML per-camera table joins the
intrinsic acceptance fields so pose fallback and intrinsic fallback can be
inspected together.

On the 2026-05-26 sparse `whole` capture, intrinsic refinement is not promoted
to the default final result. A shared-focal probe from the default 4915.2 px
prior improved the unconstrained residual but requested a ~3.9% focal change and
was rejected by the intrinsic acceptance gate. A per-camera focal probe accepted
9 camera intrinsics, but the accepted-output p90 residual was worse than the
fixed-intrinsic baseline. Keep these modes as diagnostic/probe stages until a
stronger tower capture constrains intrinsics better.

The wrapper enforces that policy in final selection: a non-`fixed`
`--tag-intrinsics-refine-mode` result is listed as a diagnostic tag-refine stage
and is not promoted to the final outer rig unless
`--promote-diagnostic-tag-refine` is explicitly passed. `--force` clears the
requested stage's declared outputs before rerunning it, so a failed requested
stage cannot silently publish stale `latest` artifacts as current outputs.

When the fast all32 bridge report has a passing `metric_bridge` quality gate,
the outer wrapper also injects the full bridge poses for top-down cameras
`4-1,4-2,4-3` into both `complete_outer_rig_side_prior.py` and
`refine_outer_tower_delta_prior.py`. This is controlled by
`--bridge-prior-override-policy gate` and
`--bridge-prior-override-labels 4-1,4-2,4-3`. The override is deliberately
gate-protected: the bridge YAML is no longer used only for anchor centers, but
only after the large-marker metric bridge has enough votes and stable
per-frame PnP residuals. Set the policy to `never` for diagnostics, or
`always` only for controlled experiments.

The current default also applies
`--tag-observation-residual-gate-px 600`, `--tag-accept-camera-p90-px 450`,
`--tag-accept-max-delta-translation-m 0.35`, and
`--tag-accept-max-delta-rotation-deg 6.5`. It uses
`--tag-min-camera-observations-for-use 16` and
`--tag-min-camera-observations-for-delta 10` so sparse but self-consistent
cameras can be optimized while final acceptance gates still protect the output.
The observation gate removes tag-corner
observations that are inconsistent with the current side-prior/tower-PnP
initialization before delta optimization. The p90 and delta acceptance gates
prevent a sparse local optimum from replacing a usable prior pose. Reports
include gated residuals, raw residuals, and final accepted-output residuals so
the operator can distinguish optimizer quality from capture outliers.

Before tag delta refinement, the wrapper now runs a per-frame PnP pose consensus
filter by default:

```bash
python3 scripts/calib/filter_pnp_views_by_pose_consensus.py \
  --pnp-views /path/to/fixed_intrinsic_pnp/pnp_views.tsv \
  --camera-prior-pose-yaml /path/to/side_prior/camera_tr_rig_side_prior.yaml \
  --output-pnp-views /path/to/pnp_pose_consensus/pnp_views_consensus.tsv \
  --center-threshold-m 0.35 \
  --rotation-threshold-deg 15 \
  --max-median-error-px 8
```

The filter converts every solved `camera_tr_tower` PnP view into a
`rig_tr_tower` vote through the current camera prior, then keeps the dominant
same-frame pose cluster. This is required for sparse tower captures because
individual planar or near-planar PnP views can have low per-view reprojection
error while representing mutually incompatible tower pose branches. Disable it
only for diagnostics with `--no-tag-pnp-pose-consensus`.

To inspect the residual tail after a tag-refine run without changing the solve,
run the standalone diagnostic:

```bash
python3 scripts/calib/analyze_outer_tag_residual_tail.py \
  /path/to/tag_refine_robust \
  --output-dir /path/to/residual_tail_diagnostics
```

It writes `residual_tail_summary.json` and `residual_tail_report.html`, ranks
worst cameras by p90/max/under-300 fraction, and reports accepted-refined versus
prior-only groups. Current refine runs export `diagnostics/observation_residuals.tsv`,
so the report also surfaces worst camera/frame/tag/corner/face records and
whether each row survived the final gate.

Refine diagnostics preserve the legacy candidate files:
`diagnostics/camera_reprojection.tsv` and `diagnostics/observation_residuals.tsv`
are computed with the refined candidate pose/intrinsics before final acceptance
fallback. Accepted-output counterparts are written separately as
`diagnostics/camera_reprojection_accepted.tsv` and
`diagnostics/observation_residuals_accepted.tsv`; these use the final
`camera_tr_rig_delta_refined_accepted.yaml` poses plus
`intrinsics_refined_accepted/`.
The final operator-facing refine summary reads the accepted-output
reprojection table and reports only final per-camera reprojection residuals and
observation counts. Delta rotation/translation remains available in
`diagnostics/camera_delta.tsv` for debugging, but it is intentionally omitted
from final reports because it mostly measures how far the solution moved from
the coarse prior.

The legacy 2026-05-26 diagnostic outer result used all32 bridge-anchored COLMAP
frame vote plus side-prior as the coarse rig, applied bridge full-pose overrides
to `4-1,4-2,4-3`, then ran median-error weighted tower PnP initialization and
tag delta refinement. On the sparse 2026-05-26 `whole` capture it accepted
20/24 camera deltas. The wrapper then applied a default post-refine observation trim
(`--tag-post-refine-observation-residual-gate-px 190`,
`--tag-post-refine-outer-iterations 2`), removing 180/1841 initially kept
observations and warm-starting a short second pass. The final accepted-output
gated median/p90 is `15.75/92.29 px`, with raw median/p90
`84.81/995.55 px`. The remaining prior-output cameras are
`4-1,4-2,4-3,8-1`; the three `4-*` top-down poses come from the passing bridge
full-pose override.

The outer manifest-level frame alignment field is intentionally labeled as a
frame-count-only check. It does not prove per-camera frame-id contiguity unless
the upstream staging report has already validated frame ids; use it as a fast
gate, not as proof that there were no interior drops.

For distributed whole-tower staging, the synchronization key is
`frame_key = <time>::<frame_id>`. QC metrics decide which frame keys pass the
configured visibility gate; staging then copies or symlinks every active camera
for the same `(time, frame_id)` into the filtered dataset with a shared output
name (`000000.jpg`, `000001.jpg`, ...). A camera does not need to see a tag to
be included in a selected frame; it may have `tag_count=0` in
`selected_images.tsv`. Before using a staged dataset for calibration, verify
that each selected `out_frame` has exactly the expected camera indices, one
unique `time`, one unique `frame_id`, and that symlink targets still point to
source files whose filename suffix matches `frame_id`.

`summary.json` reports `prior_only` as all cameras whose accepted output pose is
the prior pose, including inactive cameras and active cameras rejected by the
acceptance gates. More specific buckets such as `inactive_prior_only`,
`rejected_to_prior`, and `output_prior_pose` remain available for diagnosis.

Standalone `refine_outer_tower_delta_prior.py` uses the same safe defaults. If
all acceptance gates are deliberately disabled, the script keeps prior poses in
the `*_accepted.yaml` output unless `--allow_ungated_accepted_output` is passed.
This avoids accidentally publishing an ungated all-refined rig as an accepted
outer calibration.

The default bridge anchor for outer reports now comes from the fast all32 bridge
result when available:

```text
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260610_black_tile_wide200_pipeline_v2/inner_bridge/large_marker_bridge_all32/fixed_points_joint_ba_stride1_dense_v1/camera_tr_rig.yaml
```

For that all32 bridge YAML, top-down anchors use
`--anchor-label-to-pose-index 4-1:9,4-2:10,4-3:11`. The legacy
`large_marker_bridge_4topdown_v1` YAML still uses `4-1:8,4-2:9,4-3:10`.
The standalone `refine_outer_tower_delta_prior.py` default now follows the
all32 convention; pass the legacy map explicitly only when using legacy bridge
artifacts.

Important outputs:

- `summary.json`: machine-readable stage/input/status summary.
- `run_manifest.json`: per-run audit record with wrapper start/finish time,
  total runtime, input/output roots, final pose source, and each requested
  stage's command, duration, log path, return codes, and missing inputs. Use it
  to compare reruns, trace which stage consumed time, and recover the exact
  recalibration inputs without scraping the HTML report.
- `index.html`: compact report index with capture quality, coverage gate,
  side-prior status, tag-refine accepted/prior-only cameras, final YAML, and
  viewer link.
- `side_prior/summary.json`: records `bridge_pose_override_count` and per-camera
  center/rotation deltas when bridge full-pose overrides are applied.
- `tag_refine_robust/summary.json`: records `bridge_prior_overrides` before tag
  delta refinement starts.
- `viewer/index.html`: generated Three.js COLMAP scene/final-rig viewer
  when its inputs exist.

`summary.json` includes a `provenance` block with the exact wrapper command
line, Python executable, cwd, git branch, git commit, and dirty working-tree
status. Use it before comparing outer rigs: the current tower path has several
diagnostic modes that should not be mixed with the stable fixed-intrinsic
baseline without checking this record.
