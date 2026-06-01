#!/usr/bin/env python3
"""Generate a Three.js viewer for many aligned outer COLMAP scene models."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

import generate_threejs_rig_viewer as rig_viewer
from generate_combined_studio_rig_viewer import estimate_tower_up_from_pose_yaml
import run_outer_colmap_frame_vote as vote_base
import vote_outer_colmap_runs as vote_runs


DEFAULT_VIEWER_ASSETS_DIR = Path(
    "/home/ubuntu/calib_data/calib_2026_05_26_jpg_v3/"
    "final_inner8_calibration_v1/reports/interactive_rig_viewer_v1"
)


def parse_colmap_points3d(path, max_points):
    points = []
    if not Path(path).exists():
        return points
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        rgb = [int(parts[4]), int(parts[5]), int(parts[6])]
        error = float(parts[7])
        points.append({"xyz": xyz, "rgb": rgb, "error": error})
    if max_points > 0 and len(points) > max_points:
        step = max(1, math.ceil(len(points) / max_points))
        points = points[::step][:max_points]
    return points


def copy_viewer_assets(output_dir, assets_dir):
    output_dir = Path(output_dir)
    assets_dir = Path(assets_dir)
    required = ["three.min.js", "OrbitControls.js"]
    for name in required:
        src = assets_dir / name
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, output_dir / name)


def camera_geometry_from_rig_tr_camera(rig_tr_camera, args):
    camera_tr_rig = vote_base.invert_pose(rig_tr_camera)
    return rig_viewer.build_camera_geometry(
        camera_tr_rig,
        args.frustum_depth,
        args.frustum_half_width,
        args.frustum_half_height,
        args.axis_length,
    )


def aligned_scene_from_summary(summary, manifest_rows, anchor_centers, args):
    txt_dir = Path(summary["best_txt_dir"])
    images = vote_base.load_colmap_images(txt_dir / "images.txt")
    anchor_labels = list(anchor_centers)
    missing_anchors = [label for label in anchor_labels if label not in images]
    if missing_anchors:
        return {
            "frame": summary["frame"],
            "status": "missing_anchors",
            "missing_anchors": missing_anchors,
            "registered_count": summary.get("registered_count", 0),
            "points3d_count": summary.get("points3d_count", 0),
            "anchor_rms_m": None,
            "sim3_scale": None,
            "cameras": [],
            "points": {"positions": [], "colors": [], "count": 0},
        }

    source = np.asarray([images[label]["center_world"] for label in anchor_labels], dtype=np.float64)
    target = np.asarray([anchor_centers[label] for label in anchor_labels], dtype=np.float64)
    scale, rotation, translation, singular_values, residuals = vote_base.umeyama_similarity(source, target)
    anchor_rms = float(np.sqrt(np.mean(residuals ** 2)))
    aligned = math.isfinite(scale) and scale > 0
    accepted = aligned and anchor_rms <= args.max_anchor_rms_m

    cameras = []
    skipped_far_cameras = 0
    for row in manifest_rows:
        label = row["camera_id"]
        image = images.get(label)
        if image is None:
            continue
        center = scale * rotation @ image["center_world"] + translation
        if args.max_camera_norm_m > 0 and np.linalg.norm(center) > args.max_camera_norm_m:
            skipped_far_cameras += 1
            continue
        rig_r_camera = rotation @ image["world_tr_camera"][:3, :3]
        rig_tr_camera = vote_base.pose_matrix(rig_r_camera, center)
        geometry = camera_geometry_from_rig_tr_camera(rig_tr_camera, args)
        cameras.append({
            "index": row["camera_index"],
            "label": label,
            "center": geometry["center"],
            "basis": geometry["basis"],
            "frustum_lines": geometry["frustum_lines"],
            "metric_center": [float(v) for v in center],
            "tracks": image["triangulated_point_count"],
            "point2d_count": image["point2d_count"],
            "registered": True,
        })

    point_positions = []
    point_colors = []
    skipped_far_points = 0
    if aligned:
        for point in parse_colmap_points3d(txt_dir / "points3D.txt", args.max_points_per_scene):
            xyz = np.asarray(point["xyz"], dtype=np.float64)
            metric_xyz = scale * rotation @ xyz + translation
            if args.max_point_norm_m > 0 and np.linalg.norm(metric_xyz) > args.max_point_norm_m:
                skipped_far_points += 1
                continue
            point_positions.extend(rig_viewer.to_three(metric_xyz))
            point_colors.extend([max(0.0, min(1.0, value / 255.0)) for value in point["rgb"]])

    return {
        "frame": summary["frame"],
        "status": "accepted" if accepted else "bad_anchor_alignment",
        "registered_count": summary.get("registered_count", 0),
        "display_camera_count": len(cameras),
        "skipped_far_camera_count": skipped_far_cameras,
        "points3d_count": summary.get("points3d_count", 0),
        "anchor_rms_m": anchor_rms,
        "sim3_scale": float(scale),
        "sim3_singular_values": [float(v) for v in singular_values],
        "cameras": cameras,
        "points": {
            "positions": point_positions,
            "colors": point_colors,
            "count": len(point_positions) // 3,
            "skipped_far_count": skipped_far_points,
        },
    }


def read_metrics(path):
    if not path or not Path(path).exists():
        return {}
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            result[row["camera_id"]] = row
    return result


def load_final_rig(pose_yaml, metrics_tsv, manifest_rows, args):
    display_label = args.final_rig_label.strip() or "final rig"
    selection_source = args.final_rig_source.strip()
    if not pose_yaml or not Path(pose_yaml).exists():
        return {
            "source": "",
            "selection_source": selection_source,
            "display_label": display_label,
            "cameras": [],
        }
    used, poses = rig_viewer.load_poses(pose_yaml)
    metrics = read_metrics(metrics_tsv)
    cameras = []
    for row in manifest_rows:
        index = row["camera_index"]
        label = row["camera_id"]
        if index >= len(poses) or not used[index]:
            continue
        geometry = rig_viewer.build_camera_geometry(
            poses[index],
            args.frustum_depth,
            args.frustum_half_width,
            args.frustum_half_height,
            args.axis_length,
        )
        metric = metrics.get(label, {})
        cameras.append({
            "index": index,
            "label": label,
            "center": geometry["center"],
            "basis": geometry["basis"],
            "frustum_lines": geometry["frustum_lines"],
            "metrics": metric,
        })
    return {
        "source": str(Path(pose_yaml).resolve()),
        "selection_source": selection_source,
        "display_label": display_label,
        "metrics_tsv": str(Path(metrics_tsv).resolve()) if metrics_tsv else "",
        "cameras": cameras,
    }


def compute_bounds(scenes, final_rig):
    camera_points = []
    for scene in scenes:
        for cam in scene.get("cameras", []):
            camera_points.append(cam["center"])
    for cam in final_rig.get("cameras", []):
        camera_points.append(cam["center"])
    if not camera_points:
        return {"center": [0.0, 0.0, 0.0], "radius": 3.0}

    anchor = np.asarray(camera_points, dtype=np.float64)
    center = np.median(anchor, axis=0)
    camera_dist = np.linalg.norm(anchor - center[None, :], axis=1)
    radius = max(1.2, min(12.0, float(np.percentile(camera_dist, 98)) * 1.35 + 0.5))
    return {
        "center": [float(v) for v in center],
        "radius": radius,
    }


def build_viewer_data(args):
    manifest = vote_base.read_manifest(args.manifest)
    frames = vote_runs.parse_frames(args.frames)
    summaries = vote_runs.discover_completed_runs(args.runs_root, frames=frames, max_runs=args.max_runs)
    label_to_pose_index = vote_base.parse_label_pose_indices(args.anchor_label_to_pose_index)
    anchor_centers = vote_base.load_anchor_centers(args.anchor_pose_yaml, label_to_pose_index)

    scenes = [
        aligned_scene_from_summary(summary, manifest, anchor_centers, args)
        for summary in summaries
    ]
    final_rig = load_final_rig(args.final_pose_yaml, args.final_metrics_tsv, manifest, args)
    bounds = compute_bounds(scenes, final_rig)
    accepted = [scene for scene in scenes if scene["status"] == "accepted"]
    tower_up_alignment = estimate_tower_up_from_pose_yaml(args.tower_pose_yaml)
    return {
        "title": args.title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_runs_root": str(Path(args.runs_root).resolve()),
        "anchor_pose_yaml": str(Path(args.anchor_pose_yaml).resolve()),
        "coordinate_note": "COLMAP scenes are Sim(3)-aligned to bridge anchors 4-1/4-2/4-3. Display coordinates map metric x,y,z to Three x,-y,-z.",
        "viewer_options": {
            "default_reference_up_vector_three": (
                tower_up_alignment["display_up_vector"] if tower_up_alignment else None
            ),
            "up_alignment": tower_up_alignment,
        },
        "settings": {
            "max_anchor_rms_m": args.max_anchor_rms_m,
            "max_points_per_scene": args.max_points_per_scene,
            "frustum_depth": args.frustum_depth,
            "frustum_half_width": args.frustum_half_width,
            "frustum_half_height": args.frustum_half_height,
        },
        "summary": {
            "scene_count": len(scenes),
            "accepted_scene_count": len(accepted),
            "final_rig_camera_count": len(final_rig.get("cameras", [])),
            "total_display_points": sum(scene.get("points", {}).get("count", 0) for scene in scenes),
        },
        "manifest": [
            {"index": row["camera_index"], "label": row["camera_id"]}
            for row in manifest
        ],
        "final_rig": final_rig,
        "scenes": scenes,
        "bounds": bounds,
    }


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Outer COLMAP Scene Viewer</title>
  <style>
    :root {
      --bg: #151515;
      --panel: #f4f5f6;
      --line: #c9ced6;
      --ink: #202124;
      --muted: #667085;
      --blue: #1a73e8;
      --green: #1e8e3e;
      --amber: #b06000;
      --red: #b3261e;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #f7f8fa; }
    header { padding: 18px 22px 12px; border-bottom: 1px solid var(--line); background: #ffffff; }
    h1 { margin: 0 0 6px; font-size: 22px; }
    p { margin: 0; color: var(--muted); line-height: 1.45; }
    main { display: grid; grid-template-columns: 1fr 380px; min-height: calc(100vh - 78px); }
    #viewport { position: relative; min-height: calc(100vh - 78px); background: var(--bg); }
    canvas { display: block; outline: none; }
    aside { border-left: 1px solid var(--line); background: var(--panel); padding: 14px; overflow: auto; max-height: calc(100vh - 78px); }
    .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
    .metric { background: #ffffff; border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .metric strong { display: block; font-size: 18px; }
    .metric span { color: var(--muted); font-size: 12px; }
    label { display: grid; gap: 5px; margin: 9px 0; color: var(--muted); font-size: 12px; }
    select, input[type="range"] { width: 100%; }
    select { min-height: 32px; border: 1px solid var(--line); border-radius: 5px; background: #ffffff; }
    .buttons { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; margin: 10px 0; }
    button { border: 1px solid #adb5bd; background: #ffffff; color: var(--ink); border-radius: 5px; min-height: 31px; font-size: 12px; cursor: pointer; }
    button.active { background: #dfe9ff; border-color: var(--blue); color: #174ea6; }
    #scene-info { background: #ffffff; border: 1px solid var(--line); border-radius: 6px; padding: 9px; font-size: 12px; line-height: 1.55; margin: 10px 0; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; background: #ffffff; border: 1px solid var(--line); }
    th, td { padding: 6px 7px; border-bottom: 1px solid #e1e5ea; text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    tr { cursor: pointer; }
    tr.selected { background: #dfe9ff; }
    tr.rejected td:nth-child(2) { color: var(--amber); }
    .legend { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; margin: 8px 0 12px; font-size: 12px; color: var(--muted); }
    .swatch { width: 14px; height: 4px; display: inline-block; margin-right: 5px; vertical-align: middle; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { max-height: none; border-left: 0; border-top: 1px solid var(--line); }
      #viewport { min-height: 70vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Outer COLMAP 32-Scene Viewer</h1>
    <p>Single-frame COLMAP models are aligned to the metric bridge anchors. Select a scene to inspect its sparse point cloud and camera poses; the <span data-final-rig-label>final rig</span> is overlaid in gold.</p>
  </header>
  <main>
    <section id="viewport"></section>
    <aside>
      <div class="metrics">
        <div class="metric"><strong id="metric-scenes">-</strong><span>COLMAP scenes</span></div>
        <div class="metric"><strong id="metric-accepted">-</strong><span>accepted aligned</span></div>
        <div class="metric"><strong id="metric-points">-</strong><span>selected sparse points</span></div>
        <div class="metric"><strong id="metric-final">-</strong><span id="metric-final-label">final rig cameras</span></div>
      </div>
      <label>Scene
        <select id="scene-select"></select>
      </label>
      <div class="buttons">
        <button id="toggle-scene-cameras" class="active">Scene cams</button>
        <button id="toggle-points" class="active">Points</button>
        <button id="toggle-final" class="active">Final rig</button>
        <button id="toggle-labels" class="active">Labels</button>
        <button id="toggle-grid" class="active">Grid</button>
        <button id="top-view">Top</button>
        <button id="front-view">Front</button>
        <button id="fit-view">Fit</button>
      </div>
      <label>Point size
        <input id="point-size" type="range" min="0.004" max="0.04" step="0.002" value="0.012">
      </label>
      <div class="legend">
        <span><i class="swatch" style="background:#65a8ff"></i>selected COLMAP scene</span>
        <span><i class="swatch" style="background:#ffd166"></i><span data-final-rig-label>final rig</span></span>
        <span><i class="swatch" style="background:#ffffff"></i>sparse points</span>
        <span><i class="swatch" style="background:#8fd694"></i>bridge anchors</span>
      </div>
      <div id="scene-info"></div>
      <table>
        <thead><tr><th>Frame</th><th>Status</th><th>Reg</th><th>Pts</th><th>RMS</th></tr></thead>
        <tbody id="scene-table"></tbody>
      </table>
    </aside>
  </main>
  <script src="./three.min.js"></script>
  <script src="./OrbitControls.js"></script>
  <script>
const DATA = __VIEWER_DATA__;
function displayText(value, fallback) {
  const text = String(value || "").trim();
  return text.length ? text : fallback;
}
function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}
const finalRigLabel = displayText(DATA.final_rig && DATA.final_rig.display_label, "final rig");
const finalRigSource = displayText(DATA.final_rig && DATA.final_rig.selection_source, "");
const viewport = document.getElementById("viewport");
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x151515, 1);
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 200);
camera.up.set(0, 1, 0);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;

const root = new THREE.Group();
scene.add(root);
const sceneCameraGroup = new THREE.Group();
const pointGroup = new THREE.Group();
const finalRigGroup = new THREE.Group();
const labelGroup = new THREE.Group();
root.add(sceneCameraGroup, pointGroup, finalRigGroup, labelGroup);

const bounds = DATA.bounds || {center: [0, 0, 0], radius: 3};
const boundsCenter = vec(bounds.center);
const boundsRadius = Math.max(1, bounds.radius || 3);
const worldFromReferenceQuat = initialWorldFromReferenceQuaternion();
root.quaternion.copy(worldFromReferenceQuat);
const grid = new THREE.GridHelper(boundsRadius * 3.0, 18, 0x9a9a9a, 0x404040);
grid.material.transparent = true;
grid.material.opacity = 0.42;
scene.add(grid);
scene.add(new THREE.AxesHelper(boundsRadius * 0.22));
const light = new THREE.HemisphereLight(0xffffff, 0x202020, 1.8);
scene.add(light);

let selectedSceneIndex = 0;
let pointSize = 0.012;
const sceneRows = new Map();

function vec(values) { return new THREE.Vector3(values[0], values[1], values[2]); }
function initialWorldFromReferenceQuaternion() {
  const options = DATA.viewer_options || {};
  const quat = options.default_world_from_reference_quaternion_xyzw;
  if (Array.isArray(quat) && quat.length === 4) {
    return new THREE.Quaternion(quat[0], quat[1], quat[2], quat[3]).normalize();
  }
  const up = options.default_reference_up_vector_three;
  if (Array.isArray(up) && up.length === 3) {
    const from = vec(up);
    if (from.lengthSq() > 0) {
      return new THREE.Quaternion().setFromUnitVectors(from.normalize(), new THREE.Vector3(0, 1, 0));
    }
  }
  return new THREE.Quaternion();
}
function referenceToWorldPoint(point) {
  return vec(point).applyQuaternion(worldFromReferenceQuat);
}
function fmt(value, digits=3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(digits);
}
function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose && m.dispose());
      else child.material.dispose && child.material.dispose();
    }
  }
}
function makeLineSegments(lines, color, opacity=1, linewidth=1) {
  const values = [];
  lines.forEach((line) => values.push(...line[0], ...line[1]));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(values, 3));
  const material = new THREE.LineBasicMaterial({color, transparent: opacity < 1, opacity, linewidth});
  return new THREE.LineSegments(geometry, material);
}
function makeLabel(text, position, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "28px system-ui, sans-serif";
  ctx.fillStyle = color;
  ctx.fillText(text, 8, 42);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({map: texture, transparent: true, depthTest: false});
  const sprite = new THREE.Sprite(material);
  sprite.position.copy(vec(position)).add(new THREE.Vector3(0, 0.055, 0));
  sprite.scale.set(0.28, 0.07, 1);
  return sprite;
}
function addCameraSet(group, cameras, color, opacity, labelPrefix) {
  cameras.forEach((cam) => {
    group.add(makeLineSegments(cam.frustum_lines, color, opacity));
    const center = vec(cam.center);
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.026, 12, 8),
      new THREE.MeshBasicMaterial({color})
    );
    sphere.position.copy(center);
    group.add(sphere);
    const zAxis = vec(cam.basis.z).normalize();
    const ray = makeLineSegments([[cam.center, center.clone().addScaledVector(zAxis, 0.22).toArray()]], color, opacity);
    group.add(ray);
    labelGroup.add(makeLabel(`${labelPrefix}${cam.label}`, cam.center, color === 0xffd166 ? "#ffd166" : "#8fc7ff"));
  });
}
function buildPointCloud(points) {
  if (!points || !points.positions || !points.positions.length) return;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points.positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(points.colors, 3));
  const material = new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
    sizeAttenuation: true,
  });
  pointGroup.add(new THREE.Points(geometry, material));
}
function sceneStatusClass(sceneData) {
  return sceneData.status === "accepted" ? "" : "rejected";
}
function renderSceneTable() {
  const table = document.getElementById("scene-table");
  table.innerHTML = "";
  DATA.scenes.forEach((sceneData, index) => {
    const row = document.createElement("tr");
    row.className = sceneStatusClass(sceneData);
    row.innerHTML = `<td>${sceneData.frame}</td><td>${sceneData.status}</td><td>${sceneData.registered_count}</td><td>${sceneData.points3d_count}</td><td>${fmt(sceneData.anchor_rms_m, 3)}</td>`;
    row.addEventListener("click", () => selectScene(index));
    table.appendChild(row);
    sceneRows.set(index, row);
  });
}
function renderSceneOptions() {
  const select = document.getElementById("scene-select");
  select.innerHTML = "";
  DATA.scenes.forEach((sceneData, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `frame ${sceneData.frame} | ${sceneData.status} | ${sceneData.registered_count} cams`;
    select.appendChild(option);
  });
  select.addEventListener("change", () => selectScene(Number(select.value)));
}
function renderFinalRig() {
  clearGroup(finalRigGroup);
  const cameras = (DATA.final_rig && DATA.final_rig.cameras) || [];
  addCameraSet(finalRigGroup, cameras, 0xffd166, 0.92, "V ");
}
function updateSceneInfo(sceneData) {
  document.getElementById("metric-points").textContent = String((sceneData.points || {}).count || 0);
  const finalRigInfo = finalRigSource
    ? `<br>final rig: ${escapeHTML(finalRigLabel)} (${escapeHTML(finalRigSource)})`
    : `<br>final rig: ${escapeHTML(finalRigLabel)}`;
  document.getElementById("scene-info").innerHTML =
    `<strong>Frame ${sceneData.frame}</strong><br>` +
    `status: ${sceneData.status}<br>` +
    `registered cameras: ${sceneData.registered_count}<br>` +
    `COLMAP points3D: ${sceneData.points3d_count}<br>` +
    `displayed points: ${((sceneData.points || {}).count || 0)}<br>` +
    `anchor RMS: ${fmt(sceneData.anchor_rms_m, 4)} m<br>` +
    `Sim3 scale: ${fmt(sceneData.sim3_scale, 4)}` +
    finalRigInfo;
}
function selectScene(index) {
  selectedSceneIndex = index;
  const sceneData = DATA.scenes[index];
  document.getElementById("scene-select").value = String(index);
  sceneRows.forEach((row, rowIndex) => row.classList.toggle("selected", rowIndex === index));
  clearGroup(sceneCameraGroup);
  clearGroup(pointGroup);
  [...labelGroup.children].forEach((child) => {
    if (child.material && child.material.map) child.material.map.dispose();
  });
  clearGroup(labelGroup);
  addCameraSet(sceneCameraGroup, sceneData.cameras || [], 0x65a8ff, sceneData.status === "accepted" ? 0.78 : 0.38, "");
  renderFinalRig();
  buildPointCloud(sceneData.points);
  updateSceneInfo(sceneData);
}
function fitView(mode="iso") {
  const c = referenceToWorldPoint(bounds.center);
  const r = boundsRadius;
  let offset = new THREE.Vector3(r * 1.15, r * 0.82, r * 1.15);
  if (mode === "top") offset = new THREE.Vector3(0, r * 2.0, 0.001);
  if (mode === "front") offset = new THREE.Vector3(0, r * 0.25, r * 2.0);
  camera.position.copy(c.clone().add(offset));
  controls.target.copy(c);
  camera.near = Math.max(0.01, r / 2000);
  camera.far = Math.max(20, r * 20);
  camera.updateProjectionMatrix();
  controls.update();
}
function resize() {
  const rect = viewport.getBoundingClientRect();
  renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
  camera.aspect = Math.max(1, rect.width) / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
}
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
function setToggle(id, group) {
  const button = document.getElementById(id);
  button.addEventListener("click", () => {
    group.visible = !group.visible;
    button.classList.toggle("active", group.visible);
  });
}
function applyFinalRigText() {
  document.querySelectorAll("[data-final-rig-label]").forEach((element) => {
    element.textContent = finalRigLabel;
  });
  document.getElementById("metric-final-label").textContent = `${finalRigLabel} cameras`;
  document.getElementById("toggle-final").textContent = finalRigLabel;
}

document.getElementById("metric-scenes").textContent = String(DATA.summary.scene_count);
document.getElementById("metric-accepted").textContent = `${DATA.summary.accepted_scene_count}/${DATA.summary.scene_count}`;
document.getElementById("metric-final").textContent = String(DATA.summary.final_rig_camera_count);
document.getElementById("point-size").addEventListener("input", (event) => {
  pointSize = Number(event.target.value);
  selectScene(selectedSceneIndex);
});
document.getElementById("top-view").addEventListener("click", () => fitView("top"));
document.getElementById("front-view").addEventListener("click", () => fitView("front"));
document.getElementById("fit-view").addEventListener("click", () => fitView("iso"));
setToggle("toggle-scene-cameras", sceneCameraGroup);
setToggle("toggle-points", pointGroup);
setToggle("toggle-final", finalRigGroup);
setToggle("toggle-labels", labelGroup);
setToggle("toggle-grid", grid);

applyFinalRigText();
renderSceneOptions();
renderSceneTable();
selectScene(0);
fitView("iso");
resize();
window.addEventListener("resize", resize);
animate();

window.viewerDebug = {
  sceneCount: DATA.scenes.length,
  acceptedSceneCount: DATA.summary.accepted_scene_count,
  finalRigCameraCount: DATA.summary.final_rig_camera_count,
  selectedScene: () => DATA.scenes[selectedSceneIndex],
};
  </script>
</body>
</html>
"""


def write_viewer(output_dir, viewer_data):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_text = HTML_TEMPLATE.replace("__VIEWER_DATA__", json.dumps(viewer_data))
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    (output_dir / "scene_data.json").write_text(json.dumps(viewer_data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--anchor-pose-yaml", required=True, type=Path)
    parser.add_argument("--anchor-label-to-pose-index", default="4-1:8,4-2:9,4-3:10")
    parser.add_argument("--final-pose-yaml", default="")
    parser.add_argument("--final-metrics-tsv", default="")
    parser.add_argument("--final-rig-label", default="final rig")
    parser.add_argument("--final-rig-source", default="")
    parser.add_argument("--tower-pose-yaml", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--viewer-assets-dir", type=Path, default=DEFAULT_VIEWER_ASSETS_DIR)
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-runs", type=int, default=32)
    parser.add_argument("--max-anchor-rms-m", type=float, default=0.35)
    parser.add_argument("--max-points-per-scene", type=int, default=4500)
    parser.add_argument(
        "--max-camera-norm-m",
        type=float,
        default=20.0,
        help="Do not draw per-scene COLMAP camera poses farther than this from the metric origin. Use <=0 to disable.",
    )
    parser.add_argument(
        "--max-point-norm-m",
        type=float,
        default=12.0,
        help="Drop aligned sparse points farther than this from the metric origin. Use <=0 to disable.",
    )
    parser.add_argument("--title", default="Outer COLMAP 32-Scene Viewer")
    parser.add_argument("--frustum-depth", type=float, default=0.25)
    parser.add_argument("--frustum-half-width", type=float, default=0.14)
    parser.add_argument("--frustum-half-height", type=float, default=0.09)
    parser.add_argument("--axis-length", type=float, default=0.16)
    args = parser.parse_args()

    viewer_data = build_viewer_data(args)
    write_viewer(args.output_dir, viewer_data)
    copy_viewer_assets(args.output_dir, args.viewer_assets_dir)
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
