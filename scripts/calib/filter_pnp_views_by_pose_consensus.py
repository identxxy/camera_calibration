#!/usr/bin/env python3
"""Filter per-view PnP poses to the dominant per-frame pose consensus."""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def load_refine_module():
    module_path = Path(__file__).resolve().parent / "refine_outer_tower_delta_prior.py"
    spec = importlib.util.spec_from_file_location("refine_outer_tower_delta_prior", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pnp_row_pose(refine, row):
    return refine.pose_matrix(
        refine.quat_xyzw_to_matrix(
            float(row["qx"]),
            float(row["qy"]),
            float(row["qz"]),
            float(row["qw"])),
        [float(row["tx"]), float(row["ty"]), float(row["tz"])])


def parse_int_field(row, key, default=0):
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_candidate_views(
        refine,
        pnp_views_path,
        camera_priors,
        max_median_error_px,
        min_points,
        min_inliers):
    all_rows = []
    candidates = []
    with Path(pnp_views_path).open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fieldnames = reader.fieldnames
        for row in reader:
            all_rows.append(row)
            if row.get("status") != "solved":
                continue
            try:
                median_error = float(row["median_error_px"])
            except (KeyError, TypeError, ValueError):
                continue
            if median_error > max_median_error_px:
                continue
            point_count = parse_int_field(row, "points")
            inlier_count = parse_int_field(row, "inliers")
            if point_count < min_points or inlier_count < min_inliers:
                continue
            try:
                frame_index = int(row["imageset_index"])
                camera_index = int(row["camera_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if camera_index < 0 or camera_index >= len(camera_priors):
                continue
            camera_prior = camera_priors[camera_index]
            if camera_prior is None:
                continue
            camera_tr_target = pnp_row_pose(refine, row)
            rig_tr_target = refine.invert_pose(camera_prior) @ camera_tr_target
            candidates.append({
                "frame_index": frame_index,
                "camera_index": camera_index,
                "median_error_px": median_error,
                "points": point_count,
                "inliers": inlier_count,
                "row": row,
                "rig_tr_target": rig_tr_target,
            })
    if not fieldnames:
        raise ValueError(f"Empty PnP views TSV: {pnp_views_path}")
    return fieldnames, all_rows, candidates


def cluster_pose_votes(refine, votes, center_threshold_m, rotation_threshold_deg):
    if len(votes) <= 1:
        return list(votes)

    best_cluster = []
    best_score = (-1, float("-inf"))
    for seed in votes:
        cluster = []
        cost = 0.0
        for vote in votes:
            translation_error = float(np.linalg.norm(
                vote["rig_tr_target"][:3, 3] - seed["rig_tr_target"][:3, 3]))
            rotation_error = float(refine.pose_rotation_delta_deg(
                seed["rig_tr_target"],
                vote["rig_tr_target"]))
            if translation_error <= center_threshold_m and rotation_error <= rotation_threshold_deg:
                cluster.append(vote)
                # Keep count as the primary score; use a tiny geometric tie-breaker.
                cost += translation_error + 0.02 * rotation_error
        score = (len(cluster), -cost)
        if score > best_score:
            best_score = score
            best_cluster = cluster
    return best_cluster


def run(args):
    refine = load_refine_module()
    camera_priors = refine.load_pose_yaml(args.camera_prior_pose_yaml)
    fieldnames, all_rows, candidates = load_candidate_views(
        refine,
        args.pnp_views,
        camera_priors,
        args.max_median_error_px,
        args.min_points,
        args.min_inliers)

    by_frame = collections.defaultdict(list)
    for candidate in candidates:
        by_frame[candidate["frame_index"]].append(candidate)

    keep_keys = set()
    per_camera = collections.Counter()
    per_frame_rows = []
    single_vote_frames = 0
    multi_vote_consensus_frames = 0
    rejected_candidate_views = 0

    for frame_index in sorted(by_frame):
        votes = by_frame[frame_index]
        cluster = cluster_pose_votes(
            refine,
            votes,
            args.center_threshold_m,
            args.rotation_threshold_deg)
        if len(votes) == 1:
            single_vote_frames += 1
        elif len(cluster) >= max(2, args.min_consensus_votes):
            multi_vote_consensus_frames += 1
        rejected_candidate_views += len(votes) - len(cluster)

        cluster_translation_spread_m = ""
        cluster_rotation_spread_deg = ""
        if len(cluster) >= 2:
            average_pose = refine.robust_weighted_average_poses(
                [vote["rig_tr_target"] for vote in cluster],
                [vote["median_error_px"] for vote in cluster])
            cluster_translation_spread_m = max(
                float(np.linalg.norm(vote["rig_tr_target"][:3, 3] - average_pose[:3, 3]))
                for vote in cluster)
            cluster_rotation_spread_deg = max(
                float(refine.pose_rotation_delta_deg(average_pose, vote["rig_tr_target"]))
                for vote in cluster)

        if len(cluster) >= args.min_consensus_votes:
            for vote in cluster:
                keep_keys.add((vote["frame_index"], vote["camera_index"]))
                per_camera[vote["camera_index"]] += 1

        per_frame_rows.append({
            "frame_index": frame_index,
            "input_candidate_views": len(votes),
            "kept_views": len(cluster) if len(cluster) >= args.min_consensus_votes else 0,
            "cluster_translation_spread_m": cluster_translation_spread_m,
            "cluster_rotation_spread_deg": cluster_rotation_spread_deg,
        })

    args.output_pnp_views.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pnp_views.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            output = dict(row)
            try:
                key = (int(row["imageset_index"]), int(row["camera_index"]))
            except (KeyError, TypeError, ValueError):
                key = None
            if row.get("status") == "solved" and key not in keep_keys:
                output["status"] = "rejected_pose_consensus"
            writer.writerow(output)

    if args.per_frame_tsv:
        args.per_frame_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_frame_tsv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                delimiter="\t",
                fieldnames=[
                    "frame_index",
                    "input_candidate_views",
                    "kept_views",
                    "cluster_translation_spread_m",
                    "cluster_rotation_spread_deg",
                ])
            writer.writeheader()
            writer.writerows(per_frame_rows)

    summary = {
        "mode": "pnp_pose_consensus_filter",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_pnp_views": str(args.pnp_views),
        "output_pnp_views": str(args.output_pnp_views),
        "camera_prior_pose_yaml": str(args.camera_prior_pose_yaml),
        "center_threshold_m": float(args.center_threshold_m),
        "rotation_threshold_deg": float(args.rotation_threshold_deg),
        "max_median_error_px": float(args.max_median_error_px),
        "min_points": int(args.min_points),
        "min_inliers": int(args.min_inliers),
        "min_consensus_votes": int(args.min_consensus_votes),
        "input_candidate_views": int(len(candidates)),
        "kept_candidate_views": int(len(keep_keys)),
        "rejected_candidate_views": int(rejected_candidate_views),
        "frames_with_candidates": int(len(by_frame)),
        "frames_with_kept_views": int(sum(1 for row in per_frame_rows if row["kept_views"] > 0)),
        "single_vote_frames": int(single_vote_frames),
        "multi_vote_consensus_frames": int(multi_vote_consensus_frames),
        "camera_count_with_kept_views": int(len(per_camera)),
        "kept_views_per_camera": {
            str(camera_index): int(count)
            for camera_index, count in sorted(per_camera.items())
        },
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reject per-view PnP poses that do not agree with the dominant "
            "same-frame target pose under an existing camera rig prior."
        ))
    parser.add_argument("--pnp-views", required=True, type=Path)
    parser.add_argument("--camera-prior-pose-yaml", required=True, type=Path)
    parser.add_argument("--output-pnp-views", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--per-frame-tsv", type=Path)
    parser.add_argument("--center-threshold-m", type=float, default=0.35)
    parser.add_argument("--rotation-threshold-deg", type=float, default=15.0)
    parser.add_argument("--max-median-error-px", type=float, default=8.0)
    parser.add_argument("--min-points", type=int, default=0)
    parser.add_argument("--min-inliers", type=int, default=0)
    parser.add_argument("--min-consensus-votes", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
