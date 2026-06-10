# Studio 32-Camera Calibration Pipeline Runbook

Last updated: 2026-06-10

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
/home/ubuntu/calib_data/current_calibration/artifacts/studio_32_cameras.yaml
```

当前发布 run source:

```text
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260610_black_tile_wide200_pipeline_v2/calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml
```

SHA256:

```text
9dbc7c4908966c0550c3dd3561349e7f9597be738fc697e4644879275d956b6b
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

默认 whole data root 是
`/home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1`，因为 production
whole BA 必须使用 black-tile physical-corner dataset。默认 inner/bridge data
root 仍是 `/home/ubuntu/calib_data/calib_2026_05_31_v3`。默认 prior 则有意来自
已经人工检查过的稳定结果：inner prior 使用 2026-05-26 的 refined inner8
state，outer COLMAP prior 使用 2026-05-26 的 fixed-K first-frame COLMAP，outer
frame-face delta prior / intrinsics 使用 2026-06-10 black-tile wide200 accepted
outer result。production outer camera intrinsics 必须来自 2026-06-04
`outer_large_marker` 这类 low-density board 数据流，而不是 tower AprilTag raw
detector corners。不要把这些 prior 自动改到最新 data root，除非最新 root 已经明确
发布了等价的 bootstrap artifact。

```text
/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3/final_inner8_calibration_v1/states/final_small_marker_grid4_refine_v1
/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3/colmap_outer24_firstframe_colmap404_v3/fixed_intrinsics/sparse_txt_final24_fixedK_ba/images.txt
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260610_black_tile_wide200_pipeline_v2/outer_tower/frame_face_refine_wide200_then_gate6/camera_tr_rig_delta_refined.yaml
/home/ubuntu/calib_data/calib_2026_06_04_outer_large_marker_v2/outer_large_marker_20260604_passing_images_only_min1_bycam/outer24_intrinsics_large_marker_v1
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
  contract 是 outer `0..23`、inner `24..31`。`4-1`,`4-2`,`4-3`
  是 top-down hardware/layout metadata 和 legacy diagnostics，不是当前
  production bridge 的唯一 anchors。
- `small_marker`: high-density small board capture。产品目标是 inner8
  calibration / fixed-rig quality / diagnostic refine。默认不要把 small marker
  当作 outer bridge 输入；它只回答 inner intrinsics/extrinsics 是否可信。

## Human Operator Procedure

日常操作时先判断硬件变化范围，再选择最小足够的数据和 pipeline。不要因为有
一个 full wrapper 就默认每次重采 / 重跑所有内容。

### A. Only Inner Ring Moved

这是最常见的 fast recalib case。

1. 采集 `large_marker`
   - 用低密度 A4 board 在桌面 / bridge 区域移动。
   - 目标是让 inner8、top-down outer cameras、若干水平 outer cameras 同时看到
     board，形成 inner-to-outer bridge。
2. 采集 `small_marker`
   - 用高密度 A4 board 覆盖 inner8 视野，尤其补足图像边缘。
   - 目标是验证 inner intrinsics / fixed-rig quality；如果镜头没变，通常不把它
     推成新的 production intrinsics。
3. 跑 large-marker bridge operation
   - 复用当前 trusted outer intrinsics/extrinsics 和 inner intrinsics。
   - 如果 inner rig 相对位姿也需要重新初始化，加 `--run-large-inner-init`。
4. 跑 small-marker quality operation
   - 检查 inner per-camera residual、corner coverage、failed/inactive camera。
5. 发布 current
   - 只有 bridge residual、viewer 物理布局、YAML 坐标系都通过人工检查后，才
     更新 `/home/ubuntu/calib_data/current_calibration/`。

### B. Outer Cage Or Tower Calibration Needs Refresh

只有外圈 cage 被动过、相机标签/拓扑可能变了、旧 outer prior 视觉上不可信，
或需要重新验证 outer tower 数据时才走这条。

1. 采集 `whole`
   - 推 AprilTag tower 经过 studio 中心和外圈可见区域。
   - 尽量不要快速旋转 tower；目标是提供足够多清晰、同步、跨相机的 tag corner
     observations。
2. 运行 distributed tower QC
   - w1-w4 本地并行检测；weak views 用 full-resolution preset。
   - t0 只 collect/stage 通过 QC 的同步 frame。
3. 运行 outer frame-face refine
   - 从 trusted outer prior 开始，只优化合理 delta。
   - production outer solve 使用 rigid yaw-45 tower model：同一同步帧只有一个
     `rig_tr_tower`，8 个 face 共用竖直轴，相邻面 yaw 固定 `45 deg`。
   - printed black tile 是 8 cm + 2 cm gap；OpenCV AprilTag 红框只用于
     ID 和 red-box scale prior，生产 BA 使用 red-scale-edge 识别出的
     black-tile 外角：`tag_size_m=0.08`, `tag_spacing_m=0.02`；
     `face_width` 只作为可优化的弱先验，不再当作精确硬编码尺寸。
4. 再采 / 跑 `large_marker`
   - 用 all32 bridge 把 top-down cameras 和 inner rig 绑定进 final studio frame。
5. 发布 current
   - outer-only residual 好不等于 all32 可用；必须同时看 bridge residual 和
     unified viewer。

### C. Intrinsics Changed

只有 lens、focus、resolution、crop/resize、distortion convention 改变时才重新
做内参。

1. Outer intrinsics changed
   - 采集 `outer_large_marker`，只需要 W3/W4 外圈相机。
   - distributed QC 先筛选每台相机能看到 board/tag 的图，再把 passing images
     stage 回 t0。
   - 用 low-density A4 board corners 做 per-camera rough intrinsic solve，并把
     `intrinsics*.yaml` 目录存为下一轮 outer/bridge prior。
2. Inner intrinsics changed
   - 采集 `small_marker`。
   - 先单独做 inner per-camera intrinsic/distortion；必要时再做 fixed-rig refine。
3. Intrinsics 更新后必须重跑对应 extrinsic/bridge
   - 更新 outer K 后，重跑 `whole` outer refine 或至少重新检查 outer residual。
   - 更新 inner K 后，重跑 `large_marker` bridge。

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
     `calibrate_tower_intrinsics_opencv.py --dataset <features.bin>
     --manifest <passing_images>/manifest.tsv`. Only add
     `--points-yaml <points.yaml>` if the dataset does not already carry
     version-1 known 3D point geometry; do not pass the pattern YAML there.
   - The concise command map is
     `scripts/calib/README_studio_operation_commands.md`.
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
   - 当前 production preset 是 `wide200_then_gate6`。它必须使用
     `opencv_tower_dataset_black_tile_red_scale_edge.bin`：AprilTag 只提供 ID
     和 red-box scale prior，BA 使用重新检测的 8 cm black-tile physical
     corners。
   - `200 px` initial gate 用于在已有粗外参附近保留同 ID 的弱共视观测；
     `6 px` final gate 写 accepted output。旧 `wide50_*` preset 和 raw detector
     corner dataset 只保留作历史诊断。
   - 产物应包含 outer final report、viewer、`camera_tr_rig_delta_refined.yaml`
     和 `intrinsics_refined/`。

4. Large-marker bridge
   - 用第 3 步的 outer pose/intrinsics 跑 all32 bridge。
   - 默认复用内参；只有 inner rig 相机相对位置发生变化时才加
     `--run-large-inner-init`。
   - 必须检查 all32 fixed-known-point BA 导出的 correspondence residual
     summary：`ok_count`、median/p90/max residual。旧 `bridge_summary.json`
     的 top-down metric gate 只作为 diagnostic，不再作为 production gate。

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

## Optimization Strategy Details

当前 production pipeline 的核心原则是：可信 prior 限制 gauge / outlier，
high-quality correspondences 决定最终残差，最终发布的是物理 studio frame 下的
`camera_tr_studio_rig`。

### Outer Tower Refine

- Input geometry:
  - physical printed tile footprint 是 `0.08 m`,tile gap 是 `0.02 m`,
    center pitch 是 `0.10 m`。
  - Python/OpenCV detector corners 落在内部 AprilTag square 上，所以只用于
    tag ID 和 red-box scale prior；BA 使用 red-scale-edge 得到的 black-tile
    外角观测，几何为 `tag_size_m=0.08`, `tag_spacing_m=0.02`。
  - 不把八个 face 强制成理想八棱柱；不把不精确的 physical face width 当作
    BA 约束。
- Synchronization:
  - `frame_key = <capture_time>::<frame_id>` 是同一时刻多相机观测的唯一连接。
  - 不同 time 或不同 frame id 的 detections 不得互相组成同一 tower pose。
- Initialization:
  - 从 accepted outer prior 开始。COLMAP/RANSAC 只用于早期 bootstrap 或 prior
    失效时重建粗外参。
  - 每个同步 frame 中，每个 visible face 有独立 plane pose；PnP 只作为局部
    初始化和 gating，不是最终报告指标。
- Gating / robustification:
  - `wide200_then_gate6` 先用 `200 px` loose initial gate 保留同 ID 支持，再用
    `6 px` final gate 写 accepted output。
  - 报告中应区分 raw observations、accepted observations、final residual；
    不要用 delta magnitude 作为最终质量指标。
- Output:
  - per-camera final pixel reprojection residual、obs count、active/prior-only
    状态。
  - `camera_tr_rig_delta_refined.yaml` 和 `intrinsics_refined/` 供 bridge 使用。

### Large-Marker Bridge

- Input geometry:
  - 使用 low-density A4 board `_0` 的 known board points。
  - 当前 production bridge 使用全部可见相机和全部可接受 correspondences，不再只
    依赖 `4-*` top-down anchors。
- Initialization:
  - outer cameras 使用 outer tower refine 结果。
  - inner cameras 使用 trusted inner prior；如果 inner rig 已移动，加
    `--run-large-inner-init` 从 large board 初始化 inner poses。
- Optimization:
  - known board points 固定，优化 all32 camera poses；intrinsics 默认复用 trusted
    values。
  - 当明确要刷新内参时才允许 joint intrinsic/extrinsic BA，并必须检查 K、principal
    point、distortion 是否被 BA 拉到物理不合理区域。
- Quality:
  - bridge 主指标是 all32 fixed-known-point BA 的 pixel reprojection residual：
    median、p90、max、accepted correspondence count。
  - legacy top-down bridge metric 只作 diagnostic，不是 production gate。

### Small-Marker Inner Quality

- 使用 high-density A4 board `_3` 检查 inner8 的 feature coverage、intrinsic
  residual、fixed-rig residual。
- 如果 inner lens/focus/resolution 没变，small marker 主要作为 quality probe。
- 如果要更新 inner intrinsics，必须在更新后重新跑 large-marker bridge，不能只
  替换 K 而沿用旧 bridge。

### Canonical Studio Frame Export

- Final YAML uses `camera_tr_studio_rig`:
  `p_camera = R_camera_studio @ p_studio + t_camera_studio`。
- The exported `studio_rig` frame is physical, not cam0:
  - origin: mean center of non-4 `*-2` outer cameras;
  - `+Y`: vertical down, from `*-1` top layer to `*-3` bottom layer;
  - `+Z`: forward, opposite the missing `4-2` side gap;
  - `+X`: completes a right-handed frame.
- Viewer and YAML must use the same alignment transform. If a downstream system
  expects camera-to-world, invert `camera_tr_studio_rig` before use.

## Integration Wrapper

先做 dry-run，确认命令里所有 staged path 指向本轮数据：

```bash
python3 scripts/calib/run_studio_calibration_pipeline.py \
  --whole-data-root /home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1 \
  --inner-data-root /home/ubuntu/calib_data/calib_2026_05_31_v3 \
  --run-tag regression_20260531 \
  --run-small-quality \
  --dry-run \
  --publish-current
```

正式回归：

```bash
python3 scripts/calib/run_studio_calibration_pipeline.py \
  --whole-data-root /home/ubuntu/calib_data/calib_2026_05_31_fullres_probe_v1 \
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

wrapper 内部 stage 名称固定为 9 项：

1. `outer_tower`
2. `generate_outer_intrinsic_report`
3. `inner_bridge`
4. `export_unified_cameras`
5. `generate_bridge_camera_origin_projection`
6. `export_large_marker_correspondences`
7. `export_small_marker_correspondences`
8. `generate_advanced_correspondence_viewer`
9. `publish_current`，仅 `--publish-current` 且 bridge 被请求时执行

默认输出：

```text
/home/ubuntu/calib_data/studio_calibration_runs/<run-tag>/
```

关键产物：

```text
summary.json
index.html
outer_tower_wrapper/index.html
outer_tower/frame_face_refine_wide200_then_gate6/camera_tr_rig_delta_refined.yaml
outer_tower/frame_face_refine_wide200_then_gate6/intrinsics_refined/
inner_bridge/final_report/index.html
inner_bridge/combined_studio_rig_viewer_v1/index.html
marker_correspondences/large_marker_correspondences.tsv
marker_correspondences/small_marker_correspondences.tsv
advanced_correspondence_viewer_v1/correspondence_data.json
calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml
```

`advanced_correspondence_viewer_v1/correspondence_data.json` is loaded by the
unified 3D viewer. Do not promote the standalone advanced correspondence HTML as
a final report.

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
`outer_tower/frame_face_refine_wide200_then_gate6/camera_tr_rig_delta_refined.yaml`
和 `intrinsics_refined/`，否则 bridge 没有 outer source。

## Current 2026-06-10 Result

The current published run is:

```text
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260610_black_tile_wide200_pipeline_v2
```

Key report URLs:

```text
http://192.168.2.0:9899/
http://192.168.2.0:9899/current_calibration/artifacts/studio_32_cameras.yaml
http://192.168.2.0:9899/current_calibration/reports/01_3d_viewer/index.html
```

Key metrics from that run:

- outer24 intrinsic solve: `24/24` cameras from `outer_large_marker`;
- outer whole black-tile residual median/p90: `0.428 / 2.876 px`;
- large-marker all32 BA residual median/p90/max:
  `0.058 / 0.135 / 4.712 px`, `650948` accepted correspondences;
- small-marker inner fixed-rig residual median/p90/max:
  `0.488 / 1.232 / 3.294 px`, `71696` accepted correspondences.

The 9899 homepage must keep only the final YAML, one unified 3D viewer, and the
seven curated reports described in `scripts/ops/README_t0_report_contract.md`.
Do not promote old top-down anchor diagnostics, raw dated report folders, or
standalone correspondence viewers to the root page.

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
- all32 bridge BA correspondence median/p90/max residual
- legacy bridge metric gate 状态，仅作为 diagnostic
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
