# Studio 32-Camera Calibration System Overview

本文档说明本仓库在原始 `camera_calibration` 工具基础上，为手物交互
studio 的 32 相机联合标定所做的系统级改造、硬件约定、三类数据采集语义，
以及外圈 / 内圈 / bridge 的标定职责边界。

更偏向“怎么跑”的操作 runbook 见：

```text
scripts/calib/README_studio_calibration_pipeline.md
```

## Hardware Layout

目标场景是一个半径约 `2.5 m` 的手物交互数据采集棚，包含两圈相机：

```text
outer ring: 24 cameras, fixed on the studio cage
inner ring:  8 cameras, near-field/tabletop rig, may move between sessions
total:      32 synchronized cameras
```

外圈 24 台相机是固定 studio cage 的一部分，主要水平向内看。特殊情况是
`4-1`, `4-2`, `4-3`：这三台外圈相机是 top-down view，安装在上方，
通常不能被 AprilTag tower 的侧面观测很好约束，需要依赖 `large_marker`
bridge 绑定到整体坐标系。

内圈 8 台相机用于桌面范围内的手物交互近景采集。内圈相机可能重新安装、
移动或调整朝向，因此内圈的外参需要能够快速重标。内参通常可以复用，只有
镜头、焦距、分辨率、focus 或 distortion convention 改变时才需要重新做
完整内参标定。

## Machine And Data Layout

采集端有四台 Windows 机器，处理端是 t0：

```text
w1, w2: inner 8 cameras, each machine stores 4 camera streams
w3, w4: outer 24 cameras, each machine stores 12 camera streams
t0:     calibration processing, report service, panel service
```

原始数据在 Windows 机器上按如下格式落盘：

```text
D:/output/calib/<whole|large_marker|small_marker>/<time>/<camera_id>/<sn>_<imageid>.jpg
```

t0 通过 Windows 的 `Dshare` 网络共享挂载这些数据。项目脚本中旧的
`calib_mount` 命名已经逐步替换为更通用的 `cameras_mount` 语义，因为
calibration 只是这些相机数据的一种使用方式。

同步采集的 frame 对齐规则：

- 四路相机组在序列尾部少 `1-2` 帧是正常 tail-trim case。处理时取共同前缀
  或共同 image-id 交集。
- 如果单个相机因为连接不稳在序列中间丢帧，该相机在该 sequence 中整体无效，
  不要试图保留部分观测。

## What We Added To The Original Calibration Tool

原始仓库提供的是通用几何相机标定工具：它能够检测 repo 自带的 tagged
calibration board，求解单相机 / 固定 rig 的 intrinsic、distortion 和
extrinsic，并生成原生报告。

本分支保留原始能力，并增加了 studio 32-camera 产品所需的几类能力。

### Native C++ Extensions

这些改动扩展了原始 `camera_calibration` binary：

- AprilTag tower detector：通过 `--apriltag_tower_config` 直接从图像中提取
  tower tag corner observations。
- Dataset format compatibility：dataset parser 支持包含 3D known geometry 的
  版本，同时保持旧版 dataset 兼容。
- Sparse tower intrinsic tool：用于从 AprilTag tower sparse observations 做
  per-camera intrinsic 初始化 / 检查。
- Fixed-intrinsic rig estimation：在内参固定时用 2D-3D observations 解固定
  rig 外参。
- Dataset merge / subsample / intersection utilities：用于把多机多相机的
  feature extraction 结果整理成同步 multi-camera dataset。

这些 C++ 改动仍然是 native build 路径的一部分。t0 上的生产 binary 是本地
CMake/Ninja 编译出来的，不依赖 ROS Docker。

### Python Pipeline And Reports

这些脚本负责自动化、QC、报告和 viewer：

- Windows distributed AprilTag QC：w1-w4 在本地并行检测 tower tags，输出
  数据质量报告和可用图片清单。
- t0 staging：把 QC 选中的同步 frame 收集到 t0 的统一 staged dataset。
- OpenCV AprilTag tower dataset builder：把 tower tag corners 和 tower
  plane geometry 写成 calibration dataset。
- Outer tower frame-face refine：基于已有 outer prior，在每个同步 frame 中
  为每个 observed tower face 建独立平面位姿，用 tag corners refine outer
  camera extrinsics。
- Inner / bridge recalib wrapper：用 large / small marker 数据解内圈和
  inner-to-outer bridge。
- Unified 32-camera export：输出统一的 32 camera YAML，包含每台相机的
  intrinsic、distortion、extrinsic 和 canonical studio frame。
- HTML reports and Three.js viewers：包含数据采集 QC、最终 residual report、
  32-camera rig viewer、correspondence viewer、camera frustum image overlay。
- Calibration panel / report server：t0 上的 panel 服务用于从浏览器触发
  pipeline；9899 report service 用于展示 HTML 报告。

## Calibration Targets

我们实际使用三类标定目标，它们的职责不同，不应混用。

### AprilTag Tower For `whole`

`whole` 数据使用八棱柱 AprilTag 标定塔。当前硬件参数：

```text
faces:          8
tags per face:  32
layout:         2 columns x 16 rows
tag family:     tag36h11
tag size:       0.08 m
tag gap:        0.02 m
face id ranges: 0-31, 32-63, 64-95, ..., 224-255
```

每个面的 ID 排列约定：

```text
bottom row: small IDs, left-to-right, e.g. 0 1
top row:    large IDs, e.g. 30 31
```

打印 / 贴装后的 tag 图案相对原始 tag image 旋转了 `180 deg`。生成 tower
preview 时必须使用这个方向，否则 tag corner ordering 会和实物不一致。

八个面从 top-down view 看，ID 由小到大按逆时针排列。

重要设计选择：当前 production outer refine 不再依赖“理想正八面体”或精确
`face_width_m`。实物 tower 的 face width 和接缝工艺不够精确，强行把八个面
绑定成理想八棱柱会把几何误差注入 BA。现在默认把每个 observed frame-face
视为独立平面位姿，tower 的作用是提供稳定、可复用的 tag corner
correspondence，而不是作为刚性理想八面体模型。

`whole` 的主要目标：

- 对 fixed outer 24 cameras 做数据质量检查。
- 用已有 outer prior + tag corner observations refine outer extrinsics。
- 生成 outer tower report 和 outer rig viewer。
- 对水平向内看的 outer cameras 约束最强。

`4-1`, `4-2`, `4-3` top-down cameras 通常不是 `whole` tower 的主要约束对象；
它们应该通过 large-marker bridge 进入统一 32-camera rig。

### Low-Density A4 Board For `large_marker`

`large_marker` 使用 repo 自带 A4 calibration board 中最低密度的 pattern，
也就是 `_0` 结尾的 board。这里的 “large marker” 指的是较大的 cell / 较低
feature density，而不是另一套 tower。

它的作用是 bridge：

```text
outer fixed cage + top-down cameras + inner near-field rig
  -> shared observations of the low-density board
  -> all32 PnP initializer + all32 direct BA bridge
```

选择低密度 board 的原因：

- 单个 feature 更大，更容易被外圈 / top-down / 内圈在较大视角变化下同时看到。
- 对 bridge 来说，跨相机共视和几何稳定性比极高 feature density 更重要。
- 它可以覆盖桌面范围，适合把内圈 8 相机和外圈 top-down / side cameras 绑定。

当前 all32 bridge 的 index contract：

```text
outer cameras: indices 0..23
inner cameras: indices 24..31
top-down cameras 4-1, 4-2, 4-3: outer indices 9, 10, 11
```

`large_marker` 的主要目标：

- 在内参固定的前提下求 inner-to-outer bridge。
- 为 inner/outer 全部可见相机提供比 tower 更可靠的共享 board 约束。
- 在内圈相机位置改变时，快速刷新内圈外参和整体 rig 对齐。

### High-Density A4 Board For `small_marker`

`small_marker` 使用 repo 自带 A4 calibration board 中最高密度的 pattern，
也就是 `_3` 结尾的 board。

它的作用是 inner calibration / inner quality：

- 主要服务于内圈 8 台近景相机。
- 当 lens/focus/resolution/distortion convention 改变时，用它做 inner
  intrinsics + distortion 的高精度标定。
- 在日常 fast recalib 中，它更多作为 fixed-rig quality probe，验证当前
  inner intrinsics / extrinsics 是否仍可信。

不要默认把 `small_marker` 当作 outer bridge 输入。它 feature density 高，
适合近景高分辨率内圈相机，但对外圈远景 / top-down bridge 不如低密度 board
稳定。

## Three Data Types And Their Responsibilities

完整 studio 标定围绕四类 capture，其中 `outer_large_marker` 是外圈内参刷新，
日常 inner move recalib 通常不需要重采：

```text
outer_large_marker low-density A4 board captured by outer W3/W4 cameras
whole              AprilTag tower moving through the studio
large_marker       low-density A4 board moving over tabletop/workspace
small_marker       high-density A4 board moving over tabletop/workspace
```

推荐理解为几条 operation，而不是一个 monolithic command 的几个开关：

```text
outer_large_marker -> outer24 intrinsic initialization / refresh
whole              -> outer tower QC / outer extrinsic refine
large_marker       -> inner-to-outer bridge / top-down camera binding
small_marker       -> inner8 calibration or fixed-rig quality probe
```

最终统一 32-camera artifact 来自这些结果的组合：

```text
outer intrinsics from outer_large_marker or trusted prior
outer prior/refine from whole
inner intrinsics/extrinsics from small_marker or trusted prior
inner-to-outer bridge from large_marker
  -> studio_32_cameras.yaml
  -> unified 3D viewer
  -> final HTML report
```

## Fast Recalibration Policy

外圈 24 相机通常固定，因此它们不需要每次都完整重标。内圈 8 相机可能移动，
因此 fast recalib 主要围绕 large / small marker。

推荐策略：

- Inner cameras moved, lens unchanged:
  - 复用内参。
  - 用 `large_marker` 更新 inner-to-outer bridge。
  - 用 `small_marker` 做 inner fixed-rig quality probe。
- Outer lens/focus/resolution changed:
  - 用 `outer_large_marker` 重新估计 outer24 intrinsics。
  - 后续 `whole` refine / `large_marker` bridge 使用新的 outer intrinsics。
- Inner lens/focus/resolution changed:
  - 用 `small_marker` 重新标 inner intrinsics / distortion。
  - 再用 `large_marker` bridge 到 outer frame。
- Outer cage untouched:
  - 不重跑 `whole`。
  - 只刷新 bridge / inner quality。
- Outer cage moved or tower calibration became untrusted:
  - 重新采 `whole`。
  - 从已有 outer prior 开始做 tower frame-face refine。
  - 再用 `large_marker` bridge。

## Canonical Outputs

最终机器可读标定文件是：

```text
<run-root>/calibration_artifacts/studio_32_cameras_current/studio_32_cameras.yaml
```

当前 t0 发布版固定复制到稳定路径：

```text
/home/ubuntu/calib_data/current_calibration/artifacts/studio_32_cameras.yaml
http://192.168.2.0:9899/current_calibration/artifacts/studio_32_cameras.yaml
```

用户入口固定为：

```text
http://192.168.2.0:9899/
```

该 YAML 应包含：

- camera label / user id
- image resolution
- intrinsic matrix / distortion convention
- camera extrinsic in the canonical studio frame
- coordinate-frame metadata
- source artifact paths and run metadata where available

### Final YAML Coordinate Frame

`studio_32_cameras.yaml` 的外参是 `camera_tr_studio_rig`：

```text
studio rig point -> camera coordinates
right multiplication
meters
camera frame: OpenCV, +x right, +y down, +z forward
```

OpenCV convention 只用于每台相机自己的 camera frame。最终发布的 `studio_rig`
不是 cam0 frame，也不是 OpenCV camera frame，而是物理 studio/world frame。
导出时会用非 `4-*` 外圈相机估计一个 canonical studio frame，并把所有相机外参都
变换到这个坐标系：

```text
origin: mean center of non-4 *-2 cameras
+Y: vertical down direction, oriented from *-1 layer toward *-3 layer
+Z: forward direction, opposite the missing 4-2 side gap
-Z: backward direction, toward the missing 4-2 side gap
+X: right-handed completion, so X x Y = Z
```

这里的 `*-1/*-2/*-3` 指同一侧钢架上的上/中/下三层 outer cameras。也就是说
`+Y` 从上层 `*-1` 指向下层 `*-3`，与物理向下/重力加速度方向一致。
`4-2` 缺口方向定义为 backward，因此 `+Z forward` 是
`4-2` 缺口的反方向，`-Z` 才指向该缺口。`4-1`, `4-2`, `4-3` 是 top-down
cameras，不参与该 canonical frame 的三层平面估计。

YAML 内部会保留 `coordinate_transform` block，包括：

```text
source_coordinate_frame
aligned_coordinate_frame
point_transform
origin_source
aligned_from_source_rotation
source_from_aligned_rotation
axes_source
axis_meaning
positive_z_forward_direction_source
origin_level2_labels
negative_z_gap_labels
```

如果下游算法只消费最终 YAML，直接使用 `camera_tr_studio_rig` 即可；只有需要追溯到
pre-alignment source rig 时才需要读 `coordinate_transform`。

### Importing Into Other Coordinate Systems

YAML 中每台相机的外参记为：

```text
T_camera_studio = camera_tr_studio_rig
p_camera = T_camera_studio * p_studio
```

其中 `p_camera` 在 OpenCV camera frame 中，`p_studio` 在上面定义的 physical
studio frame 中。

#### 1. OpenCV / Multi-View Projection

如果下游只是用 OpenCV 投影 studio/world 点，不需要改坐标系：

```python
R = camera_tr_studio_rig[:3, :3]
t = camera_tr_studio_rig[:3, 3]
p_cam = R @ p_studio + t
u = project_with_K_and_distortion(p_cam, intrinsics)
```

最终 `studio_rig` 也使用 `+Y down`，但它仍然不是 OpenCV camera frame；OpenCV
camera frame 的 `+z` 是每台相机的 optical forward，而 `studio_rig +Z` 是场地
forward，即 `4-2` 缺口的反方向。

#### 2. Camera Pose / Camera Center In Studio Frame

如果下游需要 camera pose，而不是 world-to-camera extrinsic：

```python
R_cw = camera_tr_studio_rig[:3, :3]
t_cw = camera_tr_studio_rig[:3, 3]
R_wc = R_cw.T
C_w = -R_cw.T @ t_cw
T_studio_camera = inverse(T_camera_studio)
```

`C_w` 就是相机中心在 final `studio_rig` 坐标系下的位置。

#### 3. Convert The Rig To A Target World Frame

如果目标系统定义了另一个 world frame，并且：

```text
p_target = T_target_studio * p_studio
```

那么新外参应为：

```text
T_camera_target = T_camera_studio * inverse(T_target_studio)
```

也就是说，坐标系转换要右乘到现有 `camera_tr_studio_rig` 的后面，因为
`camera_tr_studio_rig` 是 `studio -> camera`。

#### 4. Convert From The Previous +Y-Up Studio Frame

旧版 2026-06-03 之前的 `studio_rig_level2_gravity_aligned` 是 `+Y up`、
`+Z forward`。新版 `studio_rig_y_down_z_forward` 等价于在旧版上做右手系两轴
翻转：

```text
p_new = diag(-1, -1, 1) * p_old

new +Y = old -Y  # vertical down
new +Z = old +Z  # forward, opposite 4-2 gap
```

如果要把旧版外参转换到新版 world frame：

```python
T_target_studio = np.eye(4)
T_target_studio[:3, :3] = np.diag([-1.0, -1.0, 1.0])
T_camera_target = T_camera_studio @ np.linalg.inv(T_target_studio)
```

这个变换不改变相机间相对位姿，也不改变重投影；它只是换了 world coordinate
parameterization。

最终用户检查入口是 HTML report 和 unified Three.js viewer：

```text
http://192.168.2.0:9899/
```

触发 t0 pipeline 的 panel 入口是：

```text
http://192.168.2.0:9898/
```

## Relationship To Headset Fisheye VI Work

Headset fisheye / visual-inertial calibration 是同一仓库里的另一条产品线，
文档在：

```text
applications/camera_calibration/scripts/fisheye_calibration/README.md
applications/camera_calibration/scripts/fisheye_calibration/README_visual_imu_calibration.md
```

它和 studio 32-camera pipeline 共享部分基础设施，例如原生
`camera_calibration` binary、calibration board detector、HTML/Three.js report
风格，但数据、相机模型和最终 YAML 产品不同。不要把 headset fisheye 的
`T_cam_imu` / KB8 输出和 studio 32-camera pinhole/OpenCV 输出混用。

## Main Entry Points

System overview:

```text
scripts/calib/README_studio_32_camera_system.md
```

Reproducible pipeline runbook:

```text
scripts/calib/README_studio_calibration_pipeline.md
```

Outer tower details:

```text
scripts/calib/README_outer_tower_recalib_pipeline.md
```

Inner / bridge details:

```text
scripts/calib/README_inner_bridge_recalib_pipeline.md
```

Panel / report server:

```text
scripts/calib/README_calibration_panel.md
scripts/ops/README_t0_report_contract.md
```
