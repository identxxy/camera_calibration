# Prompt for Main Calibration Conversation

你接下来负责 t0 上的 camera calibration report / operation 系统。请先遵守下面的上下文和标准，不要随意新增人类入口、乱放 HTML、或者让算法 pipeline 自己决定最终展示路径。

## 当前稳定入口

人类用户只从这个入口开始看：

```text
http://192.168.2.0:9899/
```

t0 文件路径：

```text
/home/ubuntu/calib_data/current_calibration/
```

当前发布 run source:

```text
/home/ubuntu/calib_data/studio_calibration_runs/recalib_20260610_black_tile_wide200_pipeline_v2
```

当前 production whole/tower BA 必须使用
`opencv_tower_dataset_black_tile_red_scale_edge.bin`。OpenCV AprilTag detector
corners 只用于 tag ID 和 red-box scale prior；BA 使用 8 cm black-tile physical
outer corners 和 2 cm spacing。`wide200_then_gate6` 是当前 production preset：
200 px initial gate 保留同 ID 候选，6 px final gate 写 accepted residual/report。

`/current_calibration/index.html` 是根入口背后的实现文件和兼容 URL；对用户展示时优先给
`http://192.168.2.0:9899/`。

## 已经完成的整理

- 删除了一个确认有 bug、打开后空白的 inner viewer：
  `calib_2026_05_26_jpg_v3/recalib_pipelines/fast_inner_bridge/latest/reports/interactive_inner_viewer/index.html`
- 保留其他历史 report，不继续物理删除；完整 inventory / cleanup audit 作为 debug 入口保留：
  `http://192.168.2.0:9899/report_audit_20260529_current/index.html`
- 新建了当前 clean entry：
  `http://192.168.2.0:9899/`
- 新建了 report contract：
  `scripts/ops/README_t0_report_contract.md`
- 新建/更新了入口生成脚本：
  `scripts/ops/build_t0_current_calib_entry.py`
- 更新了 9898 Operation Panel 后端：
  `scripts/calib/calibration_panel_server.py`

## 当前首页结构

9899 根入口只放当前 production artifacts，不再把 operation、dated scratch HTML、
source/debug viewer、registry/debug JSON、standalone correspondence viewer 或 raw
pipeline directories 提升成首页组：

1. final YAML:
   `http://192.168.2.0:9899/current_calibration/artifacts/studio_32_cameras.yaml`
2. overall 3D viewer:
   `http://192.168.2.0:9899/current_calibration/reports/01_3d_viewer/index.html`
3. inner 数据采集报告:
   `02_inner_capture_small_marker`
4. inner 内参报告:
   `03_inner_intrinsics_small_marker`
5. inner 外参报告:
   `04_inner_extrinsics_small_marker`
6. outer 数据采集报告:
   `05_outer_capture_outer_large_marker_whole`
7. outer 内参报告:
   `06_outer_intrinsics_outer_large_marker`
8. outer 外参报告:
   `07_outer_extrinsics_whole`
9. bridge 结果报告:
   `09_bridge_result_large_marker`

## 三类数据采集语义

Operation/backend 仍按 capture data type 分成三类：

1. `whole`
   - 主要目的：标定整体 studio cage，也就是 outer cameras / outer24 camera cage。
   - 不要把 whole 混成 inner/outer bridge 的主入口。

2. `large marker`
   - 主要目的：bridge inner cameras 和 outer cameras。
   - 当前包含 `large_marker_inner8` 和 `large_marker_bridge_all32`。
   - 当前 all32 contract：outer cameras indices `0..23`，inner cameras indices `24..31`。
   - `4-1`, `4-2`, `4-3` 是 top-down hardware/layout metadata 和 legacy
     diagnostics，不再是 production bridge 的唯一 anchors。

3. `small marker`
   - 主要目的：标定 inner cameras。
   - 只服务 inner cameras，不用于 outer bridge。

## 首页与 Operation 分层

首页报告分类回答“当前标定结果和数据质量在哪里看”。Operation 页面回答“采集后如何
启动受控处理”。两者必须分层：

- 首页只列 final YAML、one unified 3D viewer、seven curated reports。
- Operation 页面不作为首页报告链接；需要启动处理时去 9898 panel。
- 报告 HTML 不直接执行命令，也不嵌入任意 shell command。
- 后端只能通过 panel/server 白名单 mode 调用 CLI。

## Operation Panel 标准

9898 panel 入口：

```text
http://192.168.2.0:9898/
```

已经新增三个用户语义 mode：

```text
operate_whole_outer_cage
operate_large_marker_bridge
operate_small_marker_inner
```

当前底层实现仍复用已有 wrapper：

- `operate_whole_outer_cage` -> `scripts/calib/run_outer_tower_recalib_pipeline.py`
- `operate_large_marker_bridge` -> `scripts/calib/run_inner_bridge_recalib_pipeline.py`
- `operate_small_marker_inner` -> `scripts/calib/run_inner_bridge_recalib_pipeline.py`

长期目标是收敛成干净 CLI：

```text
t0-calib operate whole --capture-root <whole_capture_root> --output-root <run_output_root> --publish-current
t0-calib operate large-marker --inner-sequence <large_marker_inner8> --bridge-sequence <large_marker_bridge_all32> --publish-current
t0-calib operate small-marker --inner-sequence <small_marker_inner8> --output-root <run_output_root> --publish-current
```

## 最终 3D Viewer 标准

最终目标不是散落多个 viewer，而是一个 canonical 3D viewport，支持三个 mode：

- `inner + outer`
- `inner only`
- `outer only`

当前过渡状态：

- canonical current combined viewer:
  `http://192.168.2.0:9899/current_calibration/reports/01_3d_viewer/index.html`
- inner-only / outer-only:
  use the toggles inside the canonical combined viewer instead of linking separate legacy viewers.
- canonical machine-readable 32-camera artifact:
  `http://192.168.2.0:9899/current_calibration/artifacts/studio_32_cameras.yaml`

后续如果生成新 viewer，必须通过 `publish_t0_clean_calib_reports.py` promotion 到
`current_calibration`，而不是直接把新 HTML 链到主入口。

## 当前 re-calib audit 结论

当前 2026-06-10 发布版已完整跑通：

- outer whole black-tile residual median/p90: `0.428 / 2.876 px`;
- large-marker all32 bridge residual median/p90/max:
  `0.058 / 0.135 / 4.712 px`, `650948` accepted correspondences;
- small-marker inner fixed-rig residual median/p90/max:
  `0.488 / 1.232 / 3.294 px`, `71696` accepted correspondences;
- full pipeline runtime: `1241.65 s`.

inner re-calib pipeline 当前是 `usable_with_caveats`：

- `small_marker_inner8` 数据采集质量 OK：8/8 cameras，322 common frames，spread 0。
- `large_marker_inner8` 数据采集质量 OK：8/8 cameras，305 common frames，spread 0。
- `large_marker_bridge_all32` 数据采集质量 OK：32/32 cameras，305 common frames，spread 0。
- large-inner fixed-intrinsic initializer 成功，8/8 connected。
- all32 bridge 的 primary quality 来自 fixed-known-point joint BA 后的
  correspondence residual summary：
  - ok_count / median / p90 / max residual 必须从
    `large_marker_bridge_all32/fixed_points_joint_ba_*/correspondence_residual_summary.json`
    或 wrapper `bridge_correspondence_quality` 读取。
  - legacy top-down metric bridge gate 只作为 diagnostic。
- caveat：small-marker fixed-rig quality probe 中 camera `22463691` 较弱 / disconnected。
- caveat：all32 PnP initializer connectivity 只说明初始化覆盖率；最终报告不能用
  PnP initializer residual 冒充 BA residual。
- caveat：COLMAP prior / legacy bridge evaluator diagnostic weak/inconsistent 时，不自动否定
  all32 BA residual，但必须看 bridge-to-outer alignment diagnostic。

## 标准报告需求草案

### Whole / Outer Cage Final Report

必须回答：

- 每台机器、每个 outer camera 的采集帧数、TAG 检出率、accepted frame set。
- 哪些 frames / cameras 被拒绝，原因是什么。
- outer24 优化后的 pose / intrinsics 版本。
- per-camera reprojection residual、p50/p90/max、异常相机。
- outer cage geometry sanity：相机朝向、top-down cameras、环形一致性。
- 最终 outer-only / final viewer 链接。
- 明确 pass/fail gates。

### Small Marker / Inner Final Report

必须回答：

- inner8 每个 camera 的 small marker 覆盖率、corner count、accepted frames。
- inner intrinsics / extrinsics / distortion sanity。
- per-camera reprojection residual、异常相机。
- 弱相机或 disconnected camera 的明确结论。
- 最终 inner-only viewer 链接。
- 明确 pass/fail gates。

### Large Marker / Bridge Final Report

必须回答：

- bridge input contract：outer / inner camera index order、accepted frames、all32 correspondence count。
- all32 PnP initializer、fixed-known-point joint BA、post-BA reprojection residual。
- bridge-to-outer alignment diagnostic：center/rotation RMS、scale/gauge caveat。
- 哪些 cameras 没有参与或失败，是否影响 bridge。
- 最终 combined viewer 链接。
- 明确 pass/fail gates 和 caveats。

## Producer Rule

算法 pipeline 可以把产物写在自己的 run directory，例如：

```text
/home/ubuntu/calib_data/<capture_root>/recalib_pipelines/<pipeline_id>/runs/<run_id>/
```

或兼容路径：

```text
/home/ubuntu/calib_data/<capture_root>/recalib_pipelines/<pipeline_id>/latest/
```

但不能随意把 HTML 链接塞进主入口。production-capable run 应输出 artifact
manifest，并由 report/UI owner 通过 `publish_t0_clean_calib_reports.py`
promote 到：

```text
/home/ubuntu/calib_data/current_calibration/
```

artifact manifest 至少包含：

```json
{
  "pipeline_id": "...",
  "run_id": "...",
  "created_at": "...",
  "input_datasets": {},
  "artifacts": {},
  "quality_gates": {},
  "recommended_for_humans": true
}
```

## 对后续 Codex 的要求

- 不要创建新的零散 report 首页；更新 curated `current_calibration`。
- 不要物理删除历史产物，除非用户明确确认。
- 新 homepage report 必须维持 final YAML + one unified viewer + seven curated
  reports 的结构；backend operation 才按 `whole`、`large marker`、`small marker`
  三类组织。
- 新 operation 必须通过 9898 panel 白名单 mode 或未来 `t0-calib operate ...` CLI。
- 新 viewer 必须服务最终 canonical viewer contract。
- repo 内不要在 source/script 目录保留 generated report HTML；需要保留时归档到
  `studio/exp` 或 `studio/archive` 并说明来源。
- 面向人类的页面要简单、文字清晰，不做复杂营销式前端。
