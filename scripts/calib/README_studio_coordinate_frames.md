# Studio Coordinate Frame Hierarchy

This note defines the coordinate-frame layers used by the studio 24+8
calibration pipeline. Keep these layers separate; most viewer bugs come from
applying a transform at the wrong layer.

## Frames

### OpenCV Camera Frame

Each camera extrinsic is a world-to-camera transform:

```text
p_camera = T_camera_world * p_world
```

The camera frame is OpenCV:

```text
+x: image right
+y: image down
+z: optical forward
```

The YAML fields named `camera_tr_*` always mean world/rig points transformed
into this OpenCV camera frame.

### Dataset Solver Rig Frames

Each calibration solve has its own internal metric rig frame:

- `whole` / outer tower solve: `outer_tower_rig`
- `large_marker` all32 bridge solve: `bridge_rig`
- `small_marker` inner8 solve: `small_inner_rig`

These frames are not automatically interchangeable, even when they contain the
same camera labels. A `rig_tr_frame_face.yaml` from the `whole` solve is in
`outer_tower_rig`; it is not in the final studio YAML frame.

### Final Studio YAML Frame

The final published YAML is:

```text
/home/ubuntu/calib_data/current_calibration/artifacts/studio_32_cameras.yaml
```

Its `camera_tr_studio_rig` transforms are already expressed in the final
canonical studio frame:

```text
coordinate_frame: studio_rig_y_down_z_forward
origin: mean center of non-4 outer *-2 cameras
+Y: physical vertical down, from *-1 layer toward *-3 layer
+Z: physical forward, opposite the missing 4-2 side gap
-Z: toward the missing 4-2 side gap
+X: right-handed completion, +X cross +Y = +Z
```

Downstream consumers should use `camera_tr_studio_rig` directly.

### YAML `coordinate_transform`

The final YAML also stores a `coordinate_transform` block:

```text
p_aligned = R_aligned_from_source @ (p_source - origin_source)
```

This block is provenance for how the pre-canonical bridge/source frame was
converted into `studio_rig_y_down_z_forward` during export. It is not a general
viewer transform, and it must not be applied to:

- points already in `studio_rig_y_down_z_forward`;
- `rig_tr_frame_face.yaml` from the outer tower solve;
- marker correspondences exported with `--reference-studio32-yaml`.

Applying it to those already-final or unrelated frames double-transforms the
data and can rotate the displayed tower by roughly 90 degrees.

### Three.js Display Frame

The unified viewer uses a display-only mapping from metric final studio
coordinates to Three.js coordinates:

```text
p_three = [x, -y, -z]
```

This is only for rendering. It is not written back into calibration YAML, and it
does not change the OpenCV camera extrinsic convention.

## Whole / Tower Correspondence Display

The outer tower BA still writes model diagnostics in its own solver frame:

```text
rig_tr_frame_face.yaml
camera_tr_rig_delta_refined.yaml
```

Both files are in `outer_tower_rig`. They are valid for checking that particular
outer-tower solve, but they are not the final unified studio frame. The unified
viewer must not use `rig_tr_frame_face.yaml` as the 3D source for raw
correspondence endpoints.

For the final unified viewer, `whole` raw correspondence endpoints are generated
directly in `studio_rig_y_down_z_forward`:

1. Group detections by synchronized frame and tag-corner id.
2. Keep tracks seen by at least two cameras.
3. Back-project each 2D corner through the final camera intrinsics and
   `camera_tr_studio_rig`.
4. Triangulate the shared track in the final studio YAML frame.
5. Draw rays from final camera centers to that triangulated point.
6. Compute reprojection residuals against the same final camera model.

This path uses tag ids only to establish feature identity. It does not use the
outer-tower face pose as a world-coordinate source, and it does not apply the
YAML `coordinate_transform`.

The older model-based frame-face poses may still be shown in dedicated
outer-tower BA diagnostics, but they should not be mixed into the final 32-camera
correspondence overlay.

## Large / Small Marker Correspondences

The large/small marker correspondence TSVs generated with
`--reference-studio32-yaml` already store world points in the final studio YAML
frame. Load them directly and only apply the Three.js display mapping.
