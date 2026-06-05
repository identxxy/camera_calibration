# Studio 32-Camera Calibration Pipeline Runbook

Last updated: 2026-06-01

本文档固化 studio 32-camera calibration 的可复现操作路径。它覆盖三类
capture data 的 Operation 入口、今晚回归测试的 stage 顺序、当前 integration
wrapper 覆盖范围，以及最终 32-camera YAML artifact 的发布位置。

如果需要先理解硬件布局、外圈 / 内圈职责、AprilTag tower、large / small
marker 的语义，以及本仓库相对原始 calibration tool 的改造范围，先读：

```text
scripts/calib/README_studio_32_camera_system.md
```

## Current Canonical Artifact

当前正式 machine-readable 32-camera YAML:

```text
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml
```

Backup:

```text
/home/ubuntu/calib_data/calibration_backups/studio_32_cameras/studio_32_cameras_20260531_165339_recalib_20260531_193215_v2_outer_wide50.yaml
```

SHA256:

```text
d806d5509750a8832ebfbb045398b36657db61c9d66a75a8852460b6bcea46f0
```

Operator-facing report index:

```text
http://192.168.2.0:9899/
```

当前 integration wrapper:

```text
scripts/calib/run_studio_calibration_pipeline.py
```

Panel entry:

```text
http://192.168.2.0:9898/?mode=run_studio_calibration_pipeline
```

默认数据根是 `/home/ubuntu/calib_data/calib_2026_05_31_v3`。默认 prior 则有意
来自已经人工检查过的稳定结果：inner prior 使用 2026-05-26 的 refined inner8
state，outer COLMAP prior 使用 2026-05-26 的 fixed-K first-frame COLMAP，
outer frame-face delta prior/intrinsics 使用 2026-05-31 fullres raw gate6
发布结果。不要把这些 prior 自动改到最新 data root，除非最新 root 已经明确发布了
等价的 bootstrap artifact。

```text
/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3/final_inner8_calibration_v1/states/final_small_marker_grid4_refine_v1
/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3/colmap_outer24_firstframe_colmap404_v3/fixed_intrinsics/sparse_txt_final24_fixedK_ba/images.txt
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/outer_tower/frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1/camera_tr_rig_delta_refined.yaml
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/outer_tower/frame_face_refine_fullres_raw_ransac1000_wide50_gate6_v1/intrinsics_refined
```

## Operation Entries

三类数据是独立 Operation，不要把它们强行理解成同一个入口的三个按钮。采完哪一类
数据，就进入哪一类处理入口；只有做 full regression 时才把多个入口串起来。

```text
outer_large_marker -> W3/W4 distributed QC + passing-images staging
whole              -> http://192.168.2.0:9898/?mode=operate_whole_outer_cage
large_marker       -> http://192.168.2.0:9898/?mode=operate_large_marker_bridge
small_marker       -> http://192.168.2.0:9898/?mode=operate_small_marker_inner
```

- `outer_large_marker`: low-density board capture for fixed outer24 intrinsics.
  Run W3/W4 local QC first, then aggregate on t0 with
  `distributed_apriltag_quality_filter.py aggregate --stage-mode passing-images`
  so only per-camera tag-positive images are staged for the expensive C++ board
  detector. The resulting per-camera rough intrinsics are stored and reused by
  the outer tower / bridge pipeline.
- `whole`: AprilTag tower / whole-studio capture。产品目标是 fixed outer24 /
  studio cage 的 outer extrinsics refine 和 outer report。它主要约束水平向内看的
  outer cameras；`4-1`,`4-2`,`4-3` top-down cameras 通常要靠
  `large_marker` bridge 绑定。
- `large_marker`: low-density large board capture。产品目标是 inner-to-outer
  bridge，当前 production sequence 是 `large_marker_bridge_all32`，并用
  `large_marker_inner8` 作为 fixed-intrinsic inner baseline/init。all32 index
  contract 是 outer `0..23`、inner `24..31`，top-down bridge anchors
  `4-1`,`4-2`,`4-3` 对应 indices `9,10,11`。
- `small_marker`: high-density small board capture。产品目标是 inner8
  calibration / fixed-rig quality / diagnostic refine。默认不要把 small marker
  当作 outer bridge 输入；它只回答 inner intrinsics/extrinsics 是否可信。

## Full Regression Order

Production-capable regression should run in this order. Data QC/staging and
outer-intrinsic initialization are deliberately separate from the all32 solve;
do not skip them and then claim the run is reproducible.

0. Optional / infrequent `outer_large_marker` intrinsic refresh
   - Needed when outer lens/focus/resolution changes, or when old outer
     intrinsics are known bad. It is not required for routine inner-camera
     movement recalib.
   - Run Windows distributed QC on W3/W4 first.
   - Aggregate on t0 with `--stage-mode passing-images`; this stages only
     per-camera tag-positive images for per-camera intrinsic initialization.
   - Run `parallel_extract_features.py --pattern-files
     applications/camera_calibration/patterns/pattern_resolution_17x24_segments_16_apriltag_0.yaml`
     on the passing-images staging, then
     `calibrate_tower_intrinsics_opencv.py --points-yaml <large_marker points.yaml>`.
   - Store the resulting per-camera `intrinsics*.yaml` directory and pass it to
     `run_studio_calibration_pipeline.py --outer-frame-face-intrinsics-dir`.

1. Distributed QC/filter for `whole`
   - 推荐 config:
     `configs/distributed_whole_2026_05_31_filter_hybrid.json`
   - 用 `scripts/calib/server_run_distributed_clients.py` 调度/collect w1-w4，
     或通过 panel 的 `distributed_qc` mode 运行。
   - QC 输出要保留 `distributed_summary.json`、`index.html`、worker metrics；
     weak views `4-3` 和 inner serial `22587611`/`7611` 需要 full-resolution
     detect，不要用 0.5x 结果直接判死。

2. t0 collect/stage selected images
   - 用 `scripts/ops/t0_stage_current_calib_data.py` 或
     `scripts/calib/distributed_apriltag_quality_filter.py aggregate` 把选中的
     `(capture_time, frame_id)` staging 成统一 image-id。
   - `frame_key = <time>::<frame_id>` 是同步主键。允许 sequence tail 少
     `1-2` 帧并取 common prefix；如果单个 camera 中间掉帧，该 camera 在该
     sequence 中整体排除。
   - staged `whole_*` directory 必须包含 manifest、`image_directories.txt`、
     `selected_frames.tsv`、`selected_images.tsv` 和 data-quality HTML report。

3. Outer refine from existing extrinsics
   - 使用已接受的 outer prior / sane K，不再默认跑早期 COLMAP/RANSAC bootstrap。
   - 当前推荐 preset 是 `wide50_then_gate6`。它直接使用 full-resolution raw tag corners，先用 `50 px` initial gate
     在已有粗外参附近恢复弱共视 camera，再用 `6 px` final gate 做 accepted
     output。这个 preset 是 2026-06-01 后的默认 production preset。
   - 当前默认不会使用正八面体 face-width 或旧的
     `pnp_inlier_filter_facewidth...` cache；这些只保留作历史诊断。
   - 产物应包含 outer final report、viewer、`camera_tr_rig_delta_refined.yaml`
     和 `intrinsics_refined/`。

4. Large-marker bridge
   - 用第 3 步的 outer pose/intrinsics 跑 all32 bridge。
   - 默认复用内参；只有 inner rig 相机相对位置发生变化时才加
     `--run-large-inner-init`。
   - 必须检查 `bridge_summary.json` 的 metric bridge gate，尤其是 top-down
     anchors `4-1`,`4-2`,`4-3` 的 vote count、center/rotation residual 和
     triangle degeneracy。

5. Small-marker quality/refine
   - 正常回归至少跑 fixed-rig quality probe，即
     `--run-small-fixed-rig-quality` / wrapper 的 `--run-small-quality`。
   - `small_marker` joint refine 只在镜头、resolution、focus 或 distortion
     convention 改变时进入；否则它是 diagnostic，不替代当前 trusted inner
     intrinsics。

6. Export unified 32-camera YAML/report
   - 运行 `export_combined_studio_extrinsics.py` 生成
     `<output-root>/calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml`。
   - 生成/检查 combined 32-camera viewer、pipeline `summary.json`、`index.html`。
   - 只有质量门和人工检查通过后才用 `--publish-current` 更新
     `/home/ubuntu/calib_data/current_calibration/`。

## Integration Wrapper

先做 dry-run，确认命令里所有 staged path 指向本轮数据：

```bash
python3 scripts/calib/run_studio_calibration_pipeline.py \
  --whole-data-root /home/ubuntu/calib_data/calib_2026_05_31_v3 \
  --inner-data-root /home/ubuntu/calib_data/calib_2026_05_31_v3 \
  --run-tag regression_20260531 \
  --run-small-quality \
  --dry-run \
  --publish-current
```

正式回归：

```bash
python3 scripts/calib/run_studio_calibration_pipeline.py \
  --whole-data-root /home/ubuntu/calib_data/calib_2026_05_31_v3 \
  --inner-data-root /home/ubuntu/calib_data/calib_2026_05_31_v3 \
  --run-tag regression_20260531 \
  --force \
  --run-small-quality \
  --publish-current
```

如果 inner cameras 发生了相对位姿变化，再加：

```bash
--run-large-inner-init
```

wrapper 内部 stage 名称固定为 7 项：

1. `outer_tower`
2. `inner_bridge`
3. `export_unified_cameras`
4. `export_large_marker_correspondences`
5. `export_small_marker_correspondences`
6. `generate_advanced_correspondence_viewer`
7. `publish_current`，仅 `--publish-current` 且 bridge 被请求时执行

默认输出：

```text
/home/ubuntu/calib_data/studio_calibration_runs/<run-tag>/
```

关键产物：

```text
summary.json
index.html
outer_tower_wrapper/index.html
outer_tower/frame_face_refine_wide50_then_gate6/camera_tr_rig_delta_refined.yaml
outer_tower/frame_face_refine_wide50_then_gate6/intrinsics_refined/
inner_bridge/final_report/index.html
inner_bridge/combined_studio_rig_viewer_v1/index.html
marker_correspondences/large_marker_correspondences.tsv
marker_correspondences/small_marker_correspondences.tsv
advanced_correspondence_viewer_v1/index.html
calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml
```

## Useful Partial Runs

Outer only，适合 whole 重新采集但 bridge 未变：

```bash
python3 scripts/calib/run_studio_calibration_pipeline.py \
  --outer-only \
  --run-tag outer_regression_20260531 \
  --force
```

Bridge only，适合 outer 固定、只刷新 large-marker bridge：

```bash
python3 scripts/calib/run_studio_calibration_pipeline.py \
  --bridge-only \
  --run-tag bridge_regression_20260531 \
  --force \
  --run-small-quality \
  --publish-current
```

注意：`--bridge-only` 仍要求同一 `--output-root` 下已有
`outer_tower/frame_face_refine_wide50_then_gate6/camera_tr_rig_delta_refined.yaml`
和 `intrinsics_refined/`，否则 bridge 没有 outer source。

## Current 2026-05-31 v2 Result

The current published run is:

```text
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50
```

Key report URLs:

```text
http://192.168.2.0:9899/
http://192.168.2.0:9899/studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/inner_bridge/combined_studio_rig_viewer_v1/index.html
http://192.168.2.0:9899/studio_calibration_runs/recalib_20260531_193215_v2_outer_wide50/index.html
```

The 2026-06-04 run with outer intrinsics from `outer_large_marker` is the current
reference:

```text
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260604_outer_large_intrinsics_v1
http://192.168.2.0:9899/
http://192.168.2.0:9899/studio_calibration_runs/recalib_20260604_outer_large_intrinsics_v1/calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml
```

Key metrics from that run:

- outer24 intrinsic solve: `24/24` cameras, fx range `3628.51 - 3659.18 px`;
- outer whole residual median/p90: `2.264 / 4.956 px`;
- bridge metric gate: pass, top-down max center p90 `0.0339 m`, max rotation
  p90 `1.569 deg`;
- small-marker inner residual median/p90: `0.488 / 1.232 px`.

## Fast vs Full Recalib Policy

- Fast recalib: lenses/resolution/distortion convention 不变，复用 inner/outer
  intrinsics。outer cage 未动时只跑 `large_marker` bridge 和 small-marker quality；
  outer cage 轻微变化时从 existing outer extrinsics 做 frame-face delta refine。
  不需要重新采 `outer_large_marker`。
- Full outer recalib: outer cage 被动过、塔重新采集并且旧 prior 视觉上不可信、
  或 topological/camera-order 发生变化时，才回到 `whole` 全流程。即便如此，
  COLMAP first-frame / multi-frame voting 和 RANSAC rig voting 也只是 bootstrap
  fallback，不是默认 production path。
- Inner full recalib: 只有 lens/focus/resolution/distortion model 变化时，才把
  `small_marker` joint intrinsic/extrinsic refine 推成候选 production baseline。

## Acceptance Checklist

每次 production-capable run 至少记录：

- capture root、staged root、output root、run tag
- exact command 和 dry-run command
- distributed QC summary path 和 staged selected frame count
- outer active/prior-only camera list、accepted-output median/p90 residual
- bridge metric gate 状态和 top-down anchor residual
- small-marker fixed-rig quality 状态
- large/small correspondence TSV summary 和 advanced correspondence viewer path
- unified YAML path、SHA256、backup path
- 是否更新 `current_calibration`

## Calibration Validation Beyond BA Residual

3DGS reconstruction quality is an end-to-end sanity check, but it is not a pure
calibration metric. It mixes calibration, image undistortion, coordinate
conventions, scene texture, FOV overlap, exposure, masks, synchronization, and
training hyperparameters. Use the following validation ladder before blaming the
YAML.

1. Target holdout reprojection
   - Keep some `whole`, `large_marker`, or `small_marker` frames out of bundle
     adjustment.
   - Project known AprilTag / board corners with the final YAML.
   - Report per-camera median/p90/max pixel residual and overlay images.
   - This validates intrinsics, distortion, extrinsics, and target geometry
     without involving 3DGS.

2. Natural-feature epipolar validation
   - On synchronized non-calibration frames, match SIFT/SuperPoint features.
   - Undistort points with the same model used by the downstream pipeline.
   - Compute Sampson or point-to-epipolar-line distance for every camera pair.
   - A few-pixel median is plausible; tens or hundreds of pixels usually means
     import convention, undistortion, resize/crop/flip, or sync is wrong.

3. Fixed-calibration triangulation validation
   - Keep final `K`, distortion, and `T_camera_studio` fixed.
   - Triangulate matched static scene points and reproject them to all observing
     cameras.
   - Report residual by camera and by pair, plus track length / triangulation
     angle histograms.
   - This separates multi-view geometric consistency from 3DGS optimization.

4. COLMAP import parity test
   - Export the YAML to a COLMAP model using the exact downstream convention.
   - Run COLMAP point triangulation with cameras fixed.
   - Compare against a COLMAP self-BA model on the same undistorted images.
   - If COLMAP self-BA is good but YAML-fixed triangulation is bad, calibration
     or import is suspect. If both are sparse/bad, the capture/FOV/texture is the
     likely limit.

5. Downstream convention smoke test
   - Verify whether the downstream code expects world-to-camera or
     camera-to-world.
   - Current YAML stores `T_camera_studio` as world-to-camera:
     `p_camera = R @ p_studio + t`.
   - If the target system uses camera-to-world, invert every pose before use.
   - If images are resized, scale `fx`, `fy`, `cx`, `cy`; if cropped, shift
     `cx`, `cy`; if flipped/rotated, do not reuse K/extrinsics unchanged.

6. 3DGS staged checks
   - Run outer-only, inner-only, and all32 separately.
   - Inspect sparse point count, track length, ray-angle coverage, and visual
     quality before the full training run.
   - Small-FOV outer cameras can be well calibrated and still weak for 3DGS if
     the object occupies too few pixels or multi-view overlap is poor.

更详细的实验日志放在 `studio/exp/`；持久结论更新到
`studio/knowledge/studio_calibration_bootstrap_and_fast_recalib.md`。

当前回归测试耗时和产物记录：

```text
studio/exp/studio_calibration_pipeline_timing_2026_05_31.md
```
