# Studio 32-Camera Calibration Pipeline Runbook

Last updated: 2026-06-01

本文档固化 studio 32-camera calibration 的可复现操作路径。它覆盖三类
capture data 的 Operation 入口、今晚回归测试的 stage 顺序、当前 integration
wrapper 覆盖范围，以及最终 32-camera YAML artifact 的发布位置。

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
whole        -> http://192.168.2.0:9898/?mode=operate_whole_outer_cage
large_marker -> http://192.168.2.0:9898/?mode=operate_large_marker_bridge
small_marker -> http://192.168.2.0:9898/?mode=operate_small_marker_inner
```

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

2026-05-31 晚间回归测试按下面顺序执行。前两步是数据质量和 staging，后四步是
solve/report/export。`run_studio_calibration_pipeline.py` 只覆盖第 3-6 步，所以
不要跳过第 1-2 步直接宣称新数据可复现。

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

wrapper 内部 stage 名称固定为：

1. `outer_tower`
2. `inner_bridge`
3. `export_unified_cameras`
4. `publish_current`，仅 `--publish-current` 且 bridge 被请求时执行

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

Outer tower accepted residuals for the current promoted fullres raw gate6 run are
median/p90 `2.54 / 5.02 px` with `4884` accepted tag-corner observations and
`21/24` cameras receiving SE(3) deltas. Remaining prior-only cameras are the
top-down `4-1,4-2,4-3`, which are expected to come from the large-marker bridge.

The large-marker bridge is connected but narrowly fails the strict metric gate:
center residual p90 passes (`0.159 m`), vote count passes, but top-down anchor
rotation residual p90 is `5.52 deg` against a `5 deg` threshold. Treat the
current unified YAML as a usable candidate with a weak top-down bridge warning,
not as a final high-confidence production baseline.

## Fast vs Full Recalib Policy

- Fast recalib: lenses/resolution/distortion convention 不变，复用 inner/outer
  intrinsics。outer cage 未动时只跑 `large_marker` bridge 和 small-marker quality；
  outer cage 轻微变化时从 existing outer extrinsics 做 frame-face delta refine。
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
- unified YAML path、SHA256、backup path
- 是否更新 `current_calibration`

更详细的实验日志放在 `studio/exp/`；持久结论更新到
`studio/knowledge/studio_calibration_bootstrap_and_fast_recalib.md`。

当前回归测试耗时和产物记录：

```text
studio/exp/studio_calibration_pipeline_timing_2026_05_31.md
```
