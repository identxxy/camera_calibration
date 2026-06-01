#!/usr/bin/env python3
"""Complete unstable outer side columns using relative-pose topology priors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import run_outer_colmap_frame_vote as base
import vote_outer_colmap_runs as vote_runs


def parse_side_pairs(text):
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            target, reference = item.split(":", 1)
        elif "=" in item:
            target, reference = item.split("=", 1)
        else:
            raise ValueError(f"Invalid pair {item!r}; use target:reference")
        pairs.append((target.strip(), reference.strip()))
    return pairs


def rotation_residual_deg(rotation_a, rotation_b):
    return base.rotation_angle_deg(rotation_a @ rotation_b.T)


def average_pose(votes):
    translations = np.asarray([vote["relative"][:3, 3] for vote in votes], dtype=np.float64)
    translation = np.median(translations, axis=0)
    rotation = base.average_rotations([vote["relative"][:3, :3] for vote in votes])
    return base.pose_matrix(rotation, translation)


def ransac_relative_pose(votes, args):
    if len(votes) < args.min_relative_votes:
        return None

    translations = np.asarray([vote["relative"][:3, 3] for vote in votes], dtype=np.float64)
    best = None
    for hypothesis_index, hypothesis in enumerate(votes):
        hypothesis_t = hypothesis["relative"][:3, 3]
        hypothesis_r = hypothesis["relative"][:3, :3]
        translation_errors = np.linalg.norm(translations - hypothesis_t[None, :], axis=1)
        rotation_errors = np.asarray([
            rotation_residual_deg(vote["relative"][:3, :3], hypothesis_r)
            for vote in votes
        ], dtype=np.float64)
        inlier_indices = [
            index
            for index, (translation_error, rotation_error) in enumerate(zip(translation_errors, rotation_errors))
            if translation_error <= args.relative_translation_threshold_m
            and rotation_error <= args.relative_rotation_threshold_deg
        ]
        if len(inlier_indices) < args.min_relative_votes:
            continue
        score = (
            len(inlier_indices),
            -float(np.median(translation_errors[inlier_indices])),
            -float(np.median(rotation_errors[inlier_indices])),
            float(sum(votes[index]["track_score"] for index in inlier_indices)),
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "hypothesis_index": hypothesis_index,
                "inlier_indices": inlier_indices,
            }

    if best is None:
        return None

    inliers = [votes[index] for index in best["inlier_indices"]]
    relative = average_pose(inliers)
    translation = relative[:3, 3]
    rotation = relative[:3, :3]
    translation_errors = [
        float(np.linalg.norm(vote["relative"][:3, 3] - translation))
        for vote in inliers
    ]
    rotation_errors = [
        rotation_residual_deg(vote["relative"][:3, :3], rotation)
        for vote in inliers
    ]
    return {
        "relative": relative,
        "raw_votes": len(votes),
        "inlier_votes": len(inliers),
        "hypothesis_frame": votes[best["hypothesis_index"]]["frame"],
        "inlier_frames": [vote["frame"] for vote in inliers],
        "translation_median_residual_m": float(np.median(translation_errors)),
        "translation_p90_residual_m": float(np.percentile(translation_errors, 90)),
        "translation_max_residual_m": float(np.max(translation_errors)),
        "rotation_median_residual_deg": float(np.median(rotation_errors)),
        "rotation_p90_residual_deg": float(np.percentile(rotation_errors, 90)),
        "rotation_max_residual_deg": float(np.max(rotation_errors)),
        "track_median": float(np.median([vote["track_score"] for vote in inliers])),
    }


def aligned_scene_poses(summary, anchor_centers, args):
    images = base.load_colmap_images(Path(summary["best_txt_dir"]) / "images.txt")
    anchor_labels = list(anchor_centers)
    missing = [label for label in anchor_labels if label not in images]
    if missing:
        return None, {
            "frame": summary["frame"],
            "status": "missing_anchors",
            "missing_anchors": ",".join(missing),
            "anchor_rms_m": "",
            "sim3_scale": "",
            "camera_count": len(images),
        }

    source = np.asarray([images[label]["center_world"] for label in anchor_labels], dtype=np.float64)
    target = np.asarray([anchor_centers[label] for label in anchor_labels], dtype=np.float64)
    scale, rotation, translation, singular_values, residuals = base.umeyama_similarity(source, target)
    anchor_rms = float(np.sqrt(np.mean(residuals ** 2)))
    if not math.isfinite(scale) or scale <= 0 or anchor_rms > args.max_anchor_rms_m:
        return None, {
            "frame": summary["frame"],
            "status": "bad_anchor_alignment",
            "missing_anchors": "",
            "anchor_rms_m": f"{anchor_rms:.8g}",
            "sim3_scale": f"{scale:.8g}",
            "camera_count": len(images),
        }

    poses = {}
    for label, image in images.items():
        if image["triangulated_point_count"] < args.min_tracks_per_vote:
            continue
        center = scale * rotation @ image["center_world"] + translation
        if args.max_center_norm_m > 0 and np.linalg.norm(center) > args.max_center_norm_m:
            continue
        rig_r_camera = rotation @ image["world_tr_camera"][:3, :3]
        poses[label] = {
            "rig_tr_camera": base.pose_matrix(rig_r_camera, center),
            "tracks": image["triangulated_point_count"],
            "point2d_count": image["point2d_count"],
        }
    return poses, {
        "frame": summary["frame"],
        "status": "accepted",
        "missing_anchors": "",
        "anchor_rms_m": f"{anchor_rms:.8g}",
        "sim3_scale": f"{scale:.8g}",
        "camera_count": len(poses),
    }


def collect_scene_poses(args):
    frames = vote_runs.parse_frames(args.frames)
    summaries = vote_runs.discover_completed_runs(args.runs_root, frames=frames, max_runs=args.max_runs)
    label_to_pose_index = base.parse_label_pose_indices(args.anchor_label_to_pose_index)
    anchor_centers = base.load_anchor_centers(args.anchor_pose_yaml, label_to_pose_index)

    accepted = []
    rows = []
    for summary in summaries:
        poses, row = aligned_scene_poses(summary, anchor_centers, args)
        rows.append(row)
        if poses:
            accepted.append({
                "frame": summary["frame"],
                "poses": poses,
            })
    return summaries, accepted, rows


def collect_relative_votes(accepted_scenes, pairs):
    votes_by_pair = {pair: [] for pair in pairs}
    for scene in accepted_scenes:
        poses = scene["poses"]
        for target, reference in pairs:
            target_pose = poses.get(target)
            reference_pose = poses.get(reference)
            if not target_pose or not reference_pose:
                continue
            relative = base.invert_pose(reference_pose["rig_tr_camera"]) @ target_pose["rig_tr_camera"]
            votes_by_pair[(target, reference)].append({
                "frame": scene["frame"],
                "relative": relative,
                "track_score": min(target_pose["tracks"], reference_pose["tracks"]),
                "target_tracks": target_pose["tracks"],
                "reference_tracks": reference_pose["tracks"],
            })
    return votes_by_pair


def write_tsv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_direct_status(path):
    status = {}
    if not path or not Path(path).exists():
        return status
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            status[row["camera_id"]] = row
    return status


def center_from_camera_tr_rig(camera_tr_rig):
    return base.invert_pose(camera_tr_rig)[:3, 3]


def pose_rotation_delta_deg(a, b):
    return base.rotation_angle_deg(a[:3, :3].T @ b[:3, :3])


def apply_bridge_pose_overrides(completed_poses, manifest, anchor_pose_yaml, label_to_pose_index, labels):
    labels = [label.strip() for label in str(labels or "").split(",") if label.strip()]
    if not labels:
        return [], completed_poses
    bridge_poses = base.load_pose_yaml(anchor_pose_yaml)
    label_to_manifest_index = {row["camera_id"]: row["camera_index"] for row in manifest}
    updated = list(completed_poses)
    rows = []
    for label in labels:
        if label not in label_to_manifest_index:
            raise ValueError(f"Bridge pose override label {label} is not in the outer manifest")
        if label not in label_to_pose_index:
            raise ValueError(f"Bridge pose override label {label} has no anchor pose index mapping")
        manifest_index = label_to_manifest_index[label]
        bridge_index = label_to_pose_index[label]
        if bridge_index >= len(bridge_poses) or bridge_poses[bridge_index] is None:
            raise ValueError(f"Bridge pose override {label}:{bridge_index} missing in {anchor_pose_yaml}")
        before = updated[manifest_index]
        after = bridge_poses[bridge_index]
        rows.append({
            "camera_id": label,
            "outer_manifest_index": manifest_index,
            "bridge_pose_index": bridge_index,
            "center_delta_m": None if before is None else float(
                np.linalg.norm(center_from_camera_tr_rig(before) - center_from_camera_tr_rig(after))),
            "rotation_delta_deg": None if before is None else float(pose_rotation_delta_deg(before, after)),
        })
        updated[manifest_index] = after
    return rows, updated


def write_outputs(
        args,
        manifest,
        base_poses,
        completed_poses,
        pair_results,
        frame_rows,
        direct_status,
        bridge_override_rows):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base.write_pose_yaml(output_root / "camera_tr_rig_side_prior.yaml", completed_poses)

    bridge_override_by_label = {
        row["camera_id"]: row for row in bridge_override_rows
    }
    camera_rows = []
    for row in manifest:
        index = row["camera_index"]
        label = row["camera_id"]
        pose = completed_poses[index] if index < len(completed_poses) else None
        base_pose = base_poses[index] if index < len(base_poses) else None
        direct = direct_status.get(label, {})
        if pose is None:
            status = "missing"
            source = "none"
            center = [None, None, None]
        elif label in bridge_override_by_label:
            status = "bridge_metric_override"
            source = "bridge_full_pose"
            center = center_from_camera_tr_rig(pose)
        elif base_pose is not None:
            status = "direct_ransac"
            source = "absolute_pose_ransac"
            center = center_from_camera_tr_rig(pose)
        else:
            status = "side_prior_completed"
            source = "relative_side_prior"
            center = center_from_camera_tr_rig(pose)
        camera_rows.append({
            "camera_index": index,
            "camera_id": label,
            "status": status,
            "source": source,
            "direct_raw_votes": direct.get("raw_votes", ""),
            "direct_inlier_votes": direct.get("inlier_votes", ""),
            "direct_inlier_fraction": direct.get("inlier_fraction", ""),
            "center_x_m": "" if center[0] is None else f"{center[0]:.8g}",
            "center_y_m": "" if center[1] is None else f"{center[1]:.8g}",
            "center_z_m": "" if center[2] is None else f"{center[2]:.8g}",
        })

    pair_rows = []
    for (target, reference), result in pair_results.items():
        if result is None:
            pair_rows.append({
                "target_camera": target,
                "reference_camera": reference,
                "status": "insufficient_or_unstable_relative_votes",
                "raw_votes": 0,
                "inlier_votes": 0,
                "inlier_fraction": "",
                "hypothesis_frame": "",
                "inlier_frames": "",
                "translation_x_m": "",
                "translation_y_m": "",
                "translation_z_m": "",
                "translation_median_residual_m": "",
                "translation_p90_residual_m": "",
                "rotation_median_residual_deg": "",
                "rotation_p90_residual_deg": "",
                "track_median": "",
            })
            continue
        translation = result["relative"][:3, 3]
        pair_rows.append({
            "target_camera": target,
            "reference_camera": reference,
            "status": "relative_ransac_voted",
            "raw_votes": result["raw_votes"],
            "inlier_votes": result["inlier_votes"],
            "inlier_fraction": f"{result['inlier_votes'] / result['raw_votes']:.8g}",
            "hypothesis_frame": result["hypothesis_frame"],
            "inlier_frames": ",".join(str(frame) for frame in result["inlier_frames"]),
            "translation_x_m": f"{translation[0]:.8g}",
            "translation_y_m": f"{translation[1]:.8g}",
            "translation_z_m": f"{translation[2]:.8g}",
            "translation_median_residual_m": f"{result['translation_median_residual_m']:.8g}",
            "translation_p90_residual_m": f"{result['translation_p90_residual_m']:.8g}",
            "rotation_median_residual_deg": f"{result['rotation_median_residual_deg']:.8g}",
            "rotation_p90_residual_deg": f"{result['rotation_p90_residual_deg']:.8g}",
            "track_median": f"{result['track_median']:.8g}",
        })

    write_tsv(output_root / "camera_side_prior_summary.tsv", camera_rows, [
        "camera_index", "camera_id", "status", "source",
        "direct_raw_votes", "direct_inlier_votes", "direct_inlier_fraction",
        "center_x_m", "center_y_m", "center_z_m",
    ])
    write_tsv(output_root / "relative_pair_summary.tsv", pair_rows, [
        "target_camera", "reference_camera", "status", "raw_votes", "inlier_votes", "inlier_fraction",
        "hypothesis_frame", "inlier_frames",
        "translation_x_m", "translation_y_m", "translation_z_m",
        "translation_median_residual_m", "translation_p90_residual_m",
        "rotation_median_residual_deg", "rotation_p90_residual_deg",
        "track_median",
    ])
    write_tsv(output_root / "run_alignment_summary.tsv", frame_rows, [
        "frame", "status", "missing_anchors", "anchor_rms_m", "sim3_scale", "camera_count",
    ])
    summary = {
        "source_base_pose_yaml": str(Path(args.base_pose_yaml).resolve()),
        "runs_root": str(Path(args.runs_root).resolve()),
        "output_pose_yaml": str((output_root / "camera_tr_rig_side_prior.yaml").resolve()),
        "camera_count": len(manifest),
        "base_pose_count": sum(1 for pose in base_poses if pose is not None),
        "completed_pose_count": sum(1 for pose in completed_poses if pose is not None),
        "side_prior_completed_count": sum(1 for row in camera_rows if row["status"] == "side_prior_completed"),
        "relative_pair_success_count": sum(1 for row in pair_rows if row["status"] == "relative_ransac_voted"),
        "relative_pair_count": len(pair_rows),
        "bridge_pose_override_count": len(bridge_override_rows),
        "bridge_pose_overrides": bridge_override_rows,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def complete_with_side_prior(args):
    manifest = base.read_manifest(args.manifest)
    label_to_index = {row["camera_id"]: row["camera_index"] for row in manifest}
    base_poses = base.load_pose_yaml(args.base_pose_yaml)
    completed_poses = list(base_poses)
    if len(completed_poses) < len(manifest):
        completed_poses.extend([None] * (len(manifest) - len(completed_poses)))

    _summaries, accepted_scenes, frame_rows = collect_scene_poses(args)
    pairs = parse_side_pairs(args.side_pairs)
    votes_by_pair = collect_relative_votes(accepted_scenes, pairs)
    pair_results = {
        pair: ransac_relative_pose(votes, args)
        for pair, votes in votes_by_pair.items()
    }

    for target, reference in pairs:
        target_index = label_to_index[target]
        reference_index = label_to_index[reference]
        result = pair_results[(target, reference)]
        if result is None:
            continue
        reference_pose = completed_poses[reference_index]
        if reference_pose is None:
            continue
        rig_tr_reference = base.invert_pose(reference_pose)
        rig_tr_target = rig_tr_reference @ result["relative"]
        completed_poses[target_index] = base.invert_pose(rig_tr_target)

    bridge_override_rows, completed_poses = apply_bridge_pose_overrides(
        completed_poses,
        manifest,
        args.anchor_pose_yaml,
        base.parse_label_pose_indices(args.anchor_label_to_pose_index),
        args.bridge_pose_override_labels)
    direct_status = load_direct_status(args.base_metrics_tsv)
    return write_outputs(
        args,
        manifest,
        base_poses,
        completed_poses,
        pair_results,
        frame_rows,
        direct_status,
        bridge_override_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--anchor-pose-yaml", required=True, type=Path)
    parser.add_argument("--anchor-label-to-pose-index", default="4-1:8,4-2:9,4-3:10")
    parser.add_argument("--bridge-pose-override-labels", default="")
    parser.add_argument("--base-pose-yaml", required=True, type=Path)
    parser.add_argument("--base-metrics-tsv", default="")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-runs", type=int, default=32)
    parser.add_argument("--max-anchor-rms-m", type=float, default=0.35)
    parser.add_argument("--max-center-norm-m", type=float, default=8.0)
    parser.add_argument("--min-tracks-per-vote", type=int, default=10)
    parser.add_argument("--min-relative-votes", type=int, default=4)
    parser.add_argument("--relative-translation-threshold-m", type=float, default=0.65)
    parser.add_argument("--relative-rotation-threshold-deg", type=float, default=20.0)
    parser.add_argument(
        "--side-pairs",
        default="6-1:7-1,6-2:7-2,6-3:7-3,5-1:6-1,5-2:6-2,5-3:6-3",
        help="Comma-separated target:reference pairs. They are applied in order.",
    )
    args = parser.parse_args()
    summary = complete_with_side_prior(args)
    print(summary["output_pose_yaml"])


if __name__ == "__main__":
    main()
