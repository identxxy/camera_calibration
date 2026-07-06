#!/usr/bin/env python3
"""Generate a compact Three.js scene for a Seeker visual-IMU KB8 camchain."""

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


CAMERA_LABELS = {
    "cam0": "cam0 left-up",
    "cam1": "cam1 left-down",
    "cam2": "cam2 right-down",
    "cam3": "cam3 right-up",
}

CAMERA_COLORS = {
    "cam0": "#22d3ee",
    "cam1": "#a3e635",
    "cam2": "#facc15",
    "cam3": "#fb7185",
}

LABEL_OFFSETS = {
    "cam0": np.array([-1.0, -1.0, 0.0], dtype=float),
    "cam1": np.array([-1.0, 1.0, 0.0], dtype=float),
    "cam2": np.array([1.0, 1.0, 0.0], dtype=float),
    "cam3": np.array([1.0, -1.0, 0.0], dtype=float),
}

CV_TO_THREE = np.diag([1.0, -1.0, -1.0])
WORLD_UP_THREE = np.array([0.0, -1.0, 0.0], dtype=float)


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def skew(v):
    x, y, z = [float(x) for x in v]
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=float)


def rotation_between(source, target):
    source = normalize(source)
    target = normalize(target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=float)
    if dot < -1.0 + 1e-10:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=float))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=float))
        axis = normalize(axis)
        return -np.eye(3, dtype=float) + 2.0 * np.outer(axis, axis)
    cross = np.cross(source, target)
    K = skew(cross)
    return np.eye(3, dtype=float) + K + K @ K * ((1.0 - dot) / float(cross @ cross))


def invert_transform(T):
    out = np.eye(4, dtype=float)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def load_camchain(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cameras = {}
    for name, entry in sorted(raw.items()):
        if "T_imu_cam" in entry:
            T_imu_cam = np.asarray(entry["T_imu_cam"], dtype=float)
        elif "T_cam_imu" in entry:
            T_imu_cam = invert_transform(np.asarray(entry["T_cam_imu"], dtype=float))
        else:
            raise KeyError(f"{name} has neither T_imu_cam nor T_cam_imu")
        cameras[name] = {
            "entry": entry,
            "T_imu_cam": T_imu_cam,
            "center_imu": T_imu_cam[:3, 3],
            "R_imu_cam": T_imu_cam[:3, :3],
        }
    return cameras


def display_alignment(cameras):
    upper = []
    for name in ("cam0", "cam3"):
        if name in cameras:
            R = cameras[name]["R_imu_cam"]
            upper.append(CV_TO_THREE @ (R @ np.array([0.0, 0.0, 1.0], dtype=float)))
    if upper:
        source = normalize(np.mean(upper, axis=0))
        source_name = "mean upper-camera optical axis"
    else:
        source = np.array([0.0, -1.0, 0.0], dtype=float)
        source_name = "fallback"
    return rotation_between(source, WORLD_UP_THREE), {
        "source": source_name,
        "source_vector_before_alignment": source.tolist(),
        "target_world_up_three": WORLD_UP_THREE.tolist(),
        "note": "Viewer-only global rotation. It preserves all relative calibration transforms.",
    }


def to_display(point, align):
    return (align @ (CV_TO_THREE @ np.asarray(point, dtype=float))).tolist()


def vec_to_display(vector, align):
    return normalize(align @ (CV_TO_THREE @ np.asarray(vector, dtype=float))).tolist()


def build_camera_data(name, camera, align, args):
    entry = camera["entry"]
    R = camera["R_imu_cam"]
    center = camera["center_imu"]
    axes_cv = {
        "x": np.array([1.0, 0.0, 0.0], dtype=float),
        "y": np.array([0.0, 1.0, 0.0], dtype=float),
        "z": np.array([0.0, 0.0, 1.0], dtype=float),
    }
    basis_imu = {axis: R @ value for axis, value in axes_cv.items()}
    axis_lines = {
        axis: [to_display(center, align), to_display(center + basis_imu[axis] * args.axis_length, align)]
        for axis in ("x", "y", "z")
    }

    d = float(args.frustum_depth)
    hw = float(args.frustum_half_width)
    hh = float(args.frustum_half_height)
    local_corners = [
        np.array([-hw, -hh, d], dtype=float),
        np.array([hw, -hh, d], dtype=float),
        np.array([hw, hh, d], dtype=float),
        np.array([-hw, hh, d], dtype=float),
    ]
    corners = [center + R @ corner for corner in local_corners]
    corner_display = [to_display(corner, align) for corner in corners]
    center_display = to_display(center, align)
    frustum_lines = []
    for corner in corner_display:
        frustum_lines.append([center_display, corner])
    for i in range(4):
        frustum_lines.append([corner_display[i], corner_display[(i + 1) % 4]])

    intrinsics = entry.get("intrinsics", [])
    distortion = entry.get("distortion_coeffs", [])
    label_offset = LABEL_OFFSETS.get(name, np.zeros(3, dtype=float)) * float(args.label_offset)
    return {
        "key": name,
        "label": CAMERA_LABELS.get(name, name),
        "color": CAMERA_COLORS.get(name, "#94a3b8"),
        "center": center_display,
        "label_position": (np.asarray(center_display, dtype=float) + label_offset).tolist(),
        "center_imu": center.tolist(),
        "basis": {axis: vec_to_display(basis_imu[axis], align) for axis in ("x", "y", "z")},
        "optical_arrow": {
            "start": center_display,
            "direction": vec_to_display(basis_imu["z"], align),
            "length": float(args.frustum_depth) * 1.18,
        },
        "axis_lines": axis_lines,
        "frustum_corners": corner_display,
        "frustum_lines": frustum_lines,
        "intrinsics": intrinsics,
        "distortion_coeffs": distortion,
        "resolution": entry.get("resolution", []),
        "rostopic": entry.get("rostopic", ""),
    }


def build_scene_data(camchain_path, args):
    cameras = load_camchain(camchain_path)
    align, alignment_meta = display_alignment(cameras)
    camera_data = [
        build_camera_data(name, camera, align, args)
        for name, camera in cameras.items()
    ]
    imu_axes = {}
    origin = to_display([0.0, 0.0, 0.0], align)
    for axis, vector in {
        "x": [args.imu_axis_length, 0.0, 0.0],
        "y": [0.0, args.imu_axis_length, 0.0],
        "z": [0.0, 0.0, args.imu_axis_length],
    }.items():
        imu_axes[axis] = [origin, to_display(vector, align)]

    points = [origin]
    for camera in camera_data:
        points.append(camera["center"])
        for line in camera["frustum_lines"]:
            points.extend(line)
    points_np = np.asarray(points, dtype=float)
    bounds_min = points_np.min(axis=0)
    bounds_max = points_np.max(axis=0)
    center = ((bounds_min + bounds_max) * 0.5).tolist()
    radius = max(float(np.linalg.norm(bounds_max - bounds_min) * 0.5), 0.1)

    return {
        "title": args.title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_yaml": str(Path(camchain_path).resolve()),
        "coordinate_note": (
            "Input poses use T_imu_cam/T_cam_imu. Camera local frames are OpenCV: "
            "+X image-right, +Y image-down, +Z optical-forward. The viewer applies "
            "[x, y, z] -> [x, -y, -z] plus one global scene rotation for display only."
        ),
        "scene_alignment": alignment_meta,
        "world_up_three": WORLD_UP_THREE.tolist(),
        "bounds": {"center": center, "radius": radius},
        "imu": {
            "label": "IMU frame",
            "origin": origin,
            "axes": imu_axes,
        },
        "cameras": camera_data,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #111315;
      --panel: #f7f7f4;
      --line: #c9c9c3;
      --ink: #202124;
      --muted: #5f6368;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #ececea;
      color: var(--ink);
    }}
    header {{
      padding: 18px 24px 14px;
      background: #202124;
      color: #fff;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      line-height: 1.25;
      font-weight: 680;
    }}
    header p {{
      margin: 2px 0;
      color: #c9d1d9;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    main {{ padding: 18px 24px 24px; }}
    .stage {{
      position: relative;
      height: min(78vh, 820px);
      min-height: 540px;
      border: 1px solid #222;
      background: var(--bg);
      overflow: hidden;
    }}
    #viewport {{ position: absolute; inset: 0; }}
    .panel {{
      position: absolute;
      top: 14px;
      right: 14px;
      width: 330px;
      max-height: calc(100% - 28px);
      overflow: auto;
      padding: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 16px 44px rgba(0,0,0,.26);
    }}
    .panel h2 {{
      margin: 0 0 8px;
      font-size: 17px;
      line-height: 1.25;
    }}
    .note {{
      margin: 0 0 12px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 7px;
      margin-bottom: 12px;
    }}
    button {{
      min-height: 30px;
      border: 1px solid #b8b8b2;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-size: 12px;
      cursor: pointer;
    }}
    button:hover {{ border-color: #6b7280; }}
    button.active {{
      border-color: var(--accent);
      background: #dbeafe;
      color: #1d4ed8;
    }}
    .legend {{
      display: grid;
      gap: 5px;
      margin: 0 0 12px;
      font-size: 12px;
    }}
    .legend span {{
      display: inline-flex;
      gap: 7px;
      align-items: center;
    }}
    .swatch {{
      width: 16px;
      height: 3px;
      display: inline-block;
    }}
    .camera-list {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    .camera-card {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 8px 9px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .camera-card strong {{ display: block; font-size: 13px; margin-bottom: 3px; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      color: #374151;
    }}
    .label {{
      position: absolute;
      padding: 2px 6px;
      border-radius: 5px;
      background: rgba(255,255,255,.88);
      color: #111827;
      font-size: 12px;
      white-space: nowrap;
      pointer-events: none;
      transform: translate(-50%, -50%);
      border: 1px solid rgba(0,0,0,.12);
    }}
    .error {{
      position: absolute;
      top: 18px;
      left: 18px;
      max-width: 620px;
      padding: 12px 14px;
      background: #fff4ef;
      border: 1px solid #e5a088;
      color: #422318;
      font-size: 13px;
      line-height: 1.45;
      display: none;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>{source_yaml}</p>
    <p>{coordinate_note}</p>
  </header>
  <main>
    <section class="stage">
      <div id="viewport"></div>
      <div id="labels"></div>
      <div id="error" class="error">WebGL renderer failed. Try opening the file in a normal Chrome window with hardware acceleration enabled.</div>
      <aside class="panel">
        <h2>Scene Controls</h2>
        <p class="note">Global display alignment is viewer-only; relative camera and IMU transforms are read from the YAML.</p>
        <div class="controls">
          <button id="reset">Reset</button>
          <button id="toggle-frustum" class="active">Frustum</button>
          <button id="toggle-axes" class="active">Axes</button>
          <button id="toggle-labels" class="active">Labels</button>
          <button id="view-top">Top</button>
          <button id="view-front">Front</button>
        </div>
        <div class="legend">
          <span><i class="swatch" style="background:#f87171"></i>CV +X image right</span>
          <span><i class="swatch" style="background:#34d399"></i>CV +Y image down</span>
          <span><i class="swatch" style="background:#60a5fa"></i>CV +Z optical forward</span>
          <span><i class="swatch" style="background:#fbbf24"></i>IMU axes</span>
        </div>
        <p class="note"><code id="alignment"></code></p>
        <div id="camera-list" class="camera-list"></div>
      </aside>
    </section>
  </main>
  <script src="./three.min.js"></script>
  <script src="./OrbitControls.js"></script>
  <script>
    const SCENE_DATA = {scene_data_json};
  </script>
  <script>
    const viewport = document.getElementById("viewport");
    const errorBox = document.getElementById("error");
    const labelRoot = document.getElementById("labels");
    const cameraList = document.getElementById("camera-list");
    const alignment = document.getElementById("alignment");
    const labels = [];
    let showLabels = true;

    function v3(values) {{
      return new THREE.Vector3(values[0], values[1], values[2]);
    }}

    function makeLine(points, color, opacity = 1, linewidth = 1) {{
      const flat = [];
      for (const p of points) flat.push(p[0], p[1], p[2]);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(flat, 3));
      const material = new THREE.LineBasicMaterial({{
        color,
        transparent: opacity < 1,
        opacity,
        linewidth,
      }});
      return new THREE.LineSegments(geometry, material);
    }}

    function makeFrustumMesh(center, corners, color) {{
      const c = center;
      const p = corners;
      const triangles = [
        c, p[0], p[1],
        c, p[1], p[2],
        c, p[2], p[3],
        c, p[3], p[0],
        p[0], p[1], p[2],
        p[0], p[2], p[3],
      ];
      const flat = [];
      for (const point of triangles) flat.push(point[0], point[1], point[2]);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(flat, 3));
      geometry.computeVertexNormals();
      const material = new THREE.MeshBasicMaterial({{
        color,
        transparent: true,
        opacity: 0.13,
        side: THREE.DoubleSide,
        depthWrite: false,
      }});
      return new THREE.Mesh(geometry, material);
    }}

    function makeArrow(start, direction, length, color) {{
      const arrow = new THREE.ArrowHelper(
        v3(direction).normalize(),
        v3(start),
        length,
        color,
        length * 0.20,
        length * 0.09
      );
      arrow.line.material.depthTest = false;
      arrow.cone.material.depthTest = false;
      return arrow;
    }}

    function makeSphere(position, color, radius) {{
      const geometry = new THREE.SphereGeometry(radius, 24, 12);
      const material = new THREE.MeshStandardMaterial({{ color, roughness: 0.65, metalness: 0.05 }});
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(v3(position));
      return mesh;
    }}

    function addLabel(text, position, color) {{
      const el = document.createElement("div");
      el.className = "label";
      el.textContent = text;
      el.style.color = color || "#111827";
      labelRoot.appendChild(el);
      labels.push({{ el, position: v3(position) }});
    }}

    function projectLabels(camera) {{
      for (const item of labels) {{
        const p = item.position.clone().project(camera);
        const visible = showLabels && p.z > -1 && p.z < 1;
        item.el.style.display = visible ? "block" : "none";
        if (!visible) continue;
        item.el.style.left = `${{(p.x * 0.5 + 0.5) * viewport.clientWidth}}px`;
        item.el.style.top = `${{(-p.y * 0.5 + 0.5) * viewport.clientHeight}}px`;
      }}
    }}

    function setButtonState(id, enabled) {{
      document.getElementById(id).classList.toggle("active", enabled);
    }}

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
    if (!renderer.getContext()) {{
      errorBox.style.display = "block";
    }}
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x111315, 1);
    viewport.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x111315, SCENE_DATA.bounds.radius * 2.2, SCENE_DATA.bounds.radius * 6.2);
    const camera = new THREE.PerspectiveCamera(48, 1, 0.005, 50);
    camera.up.set(...SCENE_DATA.world_up_three);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const boundsCenter = v3(SCENE_DATA.bounds.center);
    const radius = SCENE_DATA.bounds.radius;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x1f2937, 1.5);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(radius * 1.5, -radius * 2.0, radius * 1.2);
    scene.add(dir);

    const grid = new THREE.GridHelper(radius * 4.0, 16, 0x5b5f66, 0x2f3338);
    grid.position.copy(boundsCenter);
    scene.add(grid);

    const frustumGroup = new THREE.Group();
    const axesGroup = new THREE.Group();
    const markerGroup = new THREE.Group();
    scene.add(frustumGroup, axesGroup, markerGroup);

    for (const [axis, line] of Object.entries(SCENE_DATA.imu.axes)) {{
      const color = axis === "x" ? 0xf87171 : axis === "y" ? 0x34d399 : 0x60a5fa;
      axesGroup.add(makeLine(line, color, 1));
    }}
    markerGroup.add(makeSphere(SCENE_DATA.imu.origin, 0xfbbf24, radius * 0.018));
    addLabel("IMU", SCENE_DATA.imu.origin, "#d97706");

    for (const cam of SCENE_DATA.cameras) {{
      frustumGroup.add(makeFrustumMesh(cam.center, cam.frustum_corners, cam.color));
      frustumGroup.add(makeLine(cam.frustum_lines, cam.color, 1));
      markerGroup.add(makeSphere(cam.center, cam.color, radius * 0.016));
      axesGroup.add(makeLine([cam.axis_lines.x], 0xf87171, 1));
      axesGroup.add(makeLine([cam.axis_lines.y], 0x34d399, 1));
      axesGroup.add(makeLine([cam.axis_lines.z], 0x60a5fa, 1));
      axesGroup.add(makeArrow(cam.optical_arrow.start, cam.optical_arrow.direction, cam.optical_arrow.length, 0x60a5fa));
      addLabel(cam.label, cam.label_position, cam.color);

      const center = cam.center_imu.map(v => Number(v).toFixed(4)).join(", ");
      const forward = cam.basis.z.map(v => Number(v).toFixed(3)).join(", ");
      const fx = cam.intrinsics.length ? Number(cam.intrinsics[0]).toFixed(2) : "-";
      const fy = cam.intrinsics.length ? Number(cam.intrinsics[1]).toFixed(2) : "-";
      const card = document.createElement("div");
      card.className = "camera-card";
      card.innerHTML = `<strong style="color:${{cam.color}}">${{cam.label}}</strong>` +
        `<div>center_imu: <code>[${{center}}]</code></div>` +
        `<div>viewer +Z: <code>[${{forward}}]</code></div>` +
        `<div>fx/fy: <code>${{fx}} / ${{fy}}</code></div>` +
        `<div>${{cam.rostopic || ""}}</div>`;
      cameraList.appendChild(card);
    }}

    alignment.textContent = `${{SCENE_DATA.scene_alignment.source}} -> world up ${{JSON.stringify(SCENE_DATA.scene_alignment.target_world_up_three)}}`;

    function resetView() {{
      controls.target.copy(boundsCenter);
      camera.position.copy(boundsCenter).add(new THREE.Vector3(radius * 1.8, -radius * 1.45, radius * 1.75));
      camera.lookAt(boundsCenter);
      controls.update();
    }}

    function viewTop() {{
      controls.target.copy(boundsCenter);
      camera.position.copy(boundsCenter).add(new THREE.Vector3(0, -radius * 3.0, 0.001));
      camera.lookAt(boundsCenter);
      controls.update();
    }}

    function viewFront() {{
      controls.target.copy(boundsCenter);
      camera.position.copy(boundsCenter).add(new THREE.Vector3(0, 0.001, radius * 3.0));
      camera.lookAt(boundsCenter);
      controls.update();
    }}

    function resize() {{
      const rect = viewport.getBoundingClientRect();
      renderer.setSize(rect.width, rect.height, false);
      camera.aspect = rect.width / Math.max(rect.height, 1);
      camera.updateProjectionMatrix();
    }}

    document.getElementById("reset").addEventListener("click", resetView);
    document.getElementById("view-top").addEventListener("click", viewTop);
    document.getElementById("view-front").addEventListener("click", viewFront);
    document.getElementById("toggle-frustum").addEventListener("click", (event) => {{
      frustumGroup.visible = !frustumGroup.visible;
      setButtonState("toggle-frustum", frustumGroup.visible);
    }});
    document.getElementById("toggle-axes").addEventListener("click", () => {{
      axesGroup.visible = !axesGroup.visible;
      setButtonState("toggle-axes", axesGroup.visible);
    }});
    document.getElementById("toggle-labels").addEventListener("click", () => {{
      showLabels = !showLabels;
      setButtonState("toggle-labels", showLabels);
    }});
    window.addEventListener("resize", resize);

    function animate() {{
      controls.update();
      renderer.render(scene, camera);
      projectLabels(camera);
      requestAnimationFrame(animate);
    }}

    resetView();
    resize();
    animate();
    window.__seekerViScene = {{ scene, camera, renderer, controls, data: SCENE_DATA }};
  </script>
</body>
</html>
"""


def copy_asset(output_dir, assets_dir, name):
    src = Path(assets_dir) / name
    if not src.is_file():
        raise FileNotFoundError(src)
    shutil.copy2(src, Path(output_dir) / name)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a compact Three.js Seeker VI calibration scene.")
    parser.add_argument("--camchain-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--viewer-assets-dir", required=True)
    parser.add_argument("--title", default="Seeker VI Calibration Simple Three.js Scene")
    parser.add_argument("--frustum-depth", type=float, default=0.18)
    parser.add_argument("--frustum-half-width", type=float, default=0.12)
    parser.add_argument("--frustum-half-height", type=float, default=0.088)
    parser.add_argument("--axis-length", type=float, default=0.055)
    parser.add_argument("--imu-axis-length", type=float, default=0.085)
    parser.add_argument("--label-offset", type=float, default=0.026)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_asset(output_dir, args.viewer_assets_dir, "three.min.js")
    copy_asset(output_dir, args.viewer_assets_dir, "OrbitControls.js")
    scene_data = build_scene_data(args.camchain_yaml, args)
    html = HTML_TEMPLATE.format(
        title=args.title,
        source_yaml=scene_data["source_yaml"],
        coordinate_note=scene_data["coordinate_note"],
        scene_data_json=json.dumps(scene_data, separators=(",", ":")),
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    (output_dir / "scene_data.json").write_text(json.dumps(scene_data, indent=2), encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
