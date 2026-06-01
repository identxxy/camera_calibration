# T0 Kalibr / ROS Noetic Setup

Date: 2026-05-31

This document records the Kalibr environment prepared on the T0 server for the
head-mounted fisheye visual-inertial calibration path.

## Remote Host

- SSH alias: `t0`
- Hostname observed on T0: `SLAI-4090-48G`
- Docker version: `29.1.3`
- Docker root: `/var/lib/docker`
- Disk for Docker and repo: `/dev/nvme0n1p2`
- Free space after setup: about `1.5T` available on `/`

No host-level ROS installation was performed. ROS Noetic and Kalibr are isolated
inside a Docker image.

## Source Location On T0

Kalibr source was cloned on T0 under the synchronized repository:

```bash
/home/ubuntu/camera_calibration/applications/camera_calibration/scripts/fisheye_calibration/ThirdParty/kalibr
```

Source repo and revision:

```text
https://github.com/ethz-asl/kalibr.git
1f60227442d25e36365ef5f72cd80b9666d73467
```

## Docker Image

Image tag:

```bash
seeker-kalibr:noetic
```

Image id:

```text
sha256:634869646a3d19a441a536c8c848c2a4f9227db190bd28e89670bf1a238132d1
```

Observed Docker disk usage:

```text
seeker-kalibr:noetic: 8.12GB docker disk usage
```

The image was built from Kalibr's upstream `Dockerfile_ros1_20_04`, which uses
`osrf/ros:noetic-desktop-full` as the base image.

Build command:

```bash
ssh t0 'set -e; cd /home/ubuntu/camera_calibration/applications/camera_calibration/scripts/fisheye_calibration/ThirdParty/kalibr; docker build -t seeker-kalibr:noetic -f Dockerfile_ros1_20_04 .'
```

Build result:

```text
catkin build: all 37 packages succeeded
warnings: 15 packages succeeded with compiler warnings
failed: 0 packages
```

The warnings are upstream C++ compiler warnings and GTK/GDK warnings from a
headless environment; no build failure was observed.

## Smoke Tests

ROS and `rosbag` are available in the container:

```bash
ssh t0 'docker run --rm --entrypoint /bin/bash seeker-kalibr:noetic -lc "source /catkin_ws/devel/setup.bash && rosversion -d && python3 -c \"import rosbag; print(\\\"rosbag import ok\\\")\""'
```

Observed output:

```text
noetic
rosbag import ok
```

Kalibr camera calibration CLI starts:

```bash
ssh t0 'docker run --rm --entrypoint /bin/bash seeker-kalibr:noetic -lc "source /catkin_ws/devel/setup.bash && rosrun kalibr kalibr_calibrate_cameras --help | head -60"'
```

Kalibr camera-IMU calibration CLI starts:

```bash
ssh t0 'docker run --rm --entrypoint /bin/bash seeker-kalibr:noetic -lc "source /catkin_ws/devel/setup.bash && rosrun kalibr kalibr_calibrate_imu_camera --help"'
```

Python package smoke test:

```bash
ssh t0 'docker run --rm --entrypoint /bin/bash seeker-kalibr:noetic -lc "source /catkin_ws/devel/setup.bash && python3 -c \"import kalibr_common; print(\\\"kalibr_common import ok\\\")\""'
```

Observed output:

```text
kalibr_common import ok
```

Headless runs print GTK/GDK display warnings such as `Unable to init server`.
Those warnings are expected because no GUI display is attached. Use Kalibr's
non-interactive flags such as `--dont-show-report` and avoid `--show-extraction`
for server-side runs.

## How To Run Kalibr On T0

Use `--entrypoint /bin/bash` so the command is not swallowed by Kalibr's upstream
Docker `ENTRYPOINT`.

Example camera intrinsic / multi-camera calibration command:

```bash
ssh t0 'docker run --rm --entrypoint /bin/bash \
  -v /home/ubuntu/camera_calibration:/workspace/camera_calibration \
  seeker-kalibr:noetic -lc "\
    source /catkin_ws/devel/setup.bash && \
    rosrun kalibr kalibr_calibrate_cameras \
      --bag /workspace/camera_calibration/path/to/capture.bag \
      --target /workspace/camera_calibration/path/to/aprilgrid.yaml \
      --models omni-radtan omni-radtan omni-radtan omni-radtan \
      --topics /cam0/image_raw /cam1/image_raw /cam2/image_raw /cam3/image_raw \
      --dont-show-report"'
```

Example visual-inertial calibration command:

```bash
ssh t0 'docker run --rm --entrypoint /bin/bash \
  -v /home/ubuntu/camera_calibration:/workspace/camera_calibration \
  seeker-kalibr:noetic -lc "\
    source /catkin_ws/devel/setup.bash && \
    rosrun kalibr kalibr_calibrate_imu_camera \
      --bag /workspace/camera_calibration/path/to/capture.bag \
      --cam /workspace/camera_calibration/path/to/camchain.yaml \
      --imu /workspace/camera_calibration/path/to/imu.yaml \
      --target /workspace/camera_calibration/path/to/aprilgrid.yaml \
      --dont-show-report"'
```

## Remaining Work Before MCAP-Based Calibration

Kalibr consumes ROS1 `.bag` files, not MCAP directly. The remaining bridge is:

1. Read Seeker MCAP image topics and IMU topic.
2. Write a ROS1 bag with:
   - `sensor_msgs/Image` for each fisheye stream.
   - `sensor_msgs/Imu` for the IMU stream.
   - timestamps preserved from MCAP.
3. Generate Kalibr input YAMLs:
   - `aprilgrid.yaml`
   - `imu.yaml`
   - initial or previously estimated `camchain.yaml`
4. Run `kalibr_calibrate_cameras` first if a Kalibr-native camchain is not yet
   available.
5. Run `kalibr_calibrate_imu_camera` for `T_cam_imu` and optional time offset.

For our current direction, T0 is ready for Kalibr execution once the MCAP-to-ROS1
bag conversion and YAML inputs are prepared.
