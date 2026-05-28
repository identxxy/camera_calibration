# Dockerized build environment

The upstream README targets Ubuntu 14.04 / 18.04 + CUDA 10.1, which no longer
works on Ada-Lovelace GPUs (CUDA 10.1 max compute capability is sm_75). This
container pins:

- Ubuntu 22.04
- CUDA 12.4 (devel image with nvcc)
- gcc-11 (CUDA 12 host-compiler upper bound)
- Qt 5 / Eigen 3 / Boost / GLEW / SuiteSparse / libpng from Ubuntu apt
- OpenGV built from the README-pinned commit

CUDA 13 is **deliberately avoided** — it drops headers the project depends on.

## Build the image

    ./docker/build_image.sh
    # tags camcalib:dev

## Build the project

    ./docker/run_shell.sh                      # mount repo, no data
    # or, with a host data directory mounted read-only at /data
    ./docker/run_shell.sh /home/vox/d/THAND

Inside the container shell:

    CUDA_ARCH=89 ./docker/build_project.sh

The binary will be at `build_docker/applications/camera_calibration/camera_calibration`.

## GPU arch override

    CUDA_ARCH="86;89" ./docker/build_project.sh   # RTX 30xx + 40xx fat binary
    CUDA_ARCH=75     ./docker/build_project.sh   # 20xx-series machine

## GUI / visualizations

`--show_visualizations` and the recording GUI need an X server. `run_shell.sh`
forwards `$DISPLAY` and mounts the X11 socket; the headless pipeline (image
folder → features → calibration) does not need any of that.
