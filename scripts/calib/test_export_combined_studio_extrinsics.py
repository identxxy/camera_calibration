#!/usr/bin/env python3
"""Tests for combined studio extrinsics export."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_combined_studio_extrinsics as export_extrinsics  # noqa: E402
import studio_canonical_frame  # noqa: E402


def write_pose_yaml(path, count, offset):
    lines = [
        f"pose_count: {count}",
        "poses:",
    ]
    for index in range(count):
        lines.extend([
            f"  - index: {index}",
            f"    tx: {offset + index:.6f}",
            f"    ty: {offset + index + 0.1:.6f}",
            f"    tz: {offset + index + 0.2:.6f}",
            "    qx: 0",
            "    qy: 0",
            "    qz: 0",
            "    qw: 1",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_intrinsics_yaml(path, fx):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "type: CentralOpenCVModel",
            "width: 4096",
            "height: 3000",
            f"parameters: [{fx}, {fx + 1}, 2048, 1500, 0.1, 0.2]",
            "",
        ]),
        encoding="utf-8",
    )


def pose_from_center(center):
    pose = export_extrinsics.np.eye(4)
    pose[:3, 3] = -export_extrinsics.np.asarray(center, dtype=export_extrinsics.np.float64)
    return pose


class ExportCombinedStudioExtrinsicsTest(unittest.TestCase):
    def test_reads_inner_camera_ids_from_camera_labels_tsv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera_labels.tsv"
            path.write_text(
                "index\tlabel\tgroup\tcamera_id\tsource_yaml\tsource_index\n"
                "0\t1-1\touter\t1-1\touter.yaml\t0\n"
                "24\tinner0\tinner\t22463688\tinner.yaml\t24\n"
                "25\tinner1\tinner\t22463690\tinner.yaml\t25\n",
                encoding="utf-8",
            )

            self.assertEqual(
                export_extrinsics.read_inner_camera_ids(path),
                {0: "22463688", 1: "22463690"},
            )

    def test_export_uses_outer_0_to_23_and_inner_24_to_31(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outer_yaml = root / "outer.yaml"
            inner_yaml = root / "inner_bridge.yaml"
            manifest_tsv = root / "manifest.tsv"
            output_dir = root / "out"
            write_pose_yaml(outer_yaml, 24, 100)
            write_pose_yaml(inner_yaml, 32, 200)
            manifest_tsv.write_text(
                "camera_index\tcamera_id\n"
                + "".join(f"{index}\tsn{index}\n" for index in range(8)),
                encoding="utf-8",
            )

            args = type("Args", (), {
                "outer_final_pose_yaml": outer_yaml,
                "inner_bridge_pose_yaml": inner_yaml,
                "inner_manifest": manifest_tsv,
                "inner_bridge_indices": "24,25,26,27,28,29,30,31",
                "intrinsics_dir": root / "intrinsics",
                "output_dir": output_dir,
                "viewer_url": "http://example/viewer",
                "run_tag": "test",
            })()

            poses, rows = export_extrinsics.build_combined_poses(args)
            export_extrinsics.write_pose_yaml(output_dir / "camera_tr_studio_rig.yaml", poses)
            export_extrinsics.write_label_tsv(output_dir / "camera_labels.tsv", rows)
            export_extrinsics.write_manifest(
                output_dir / "manifest.json",
                args,
                output_dir / "camera_tr_studio_rig.yaml",
                output_dir / "camera_labels.tsv",
                output_dir / "studio_32_cameras.yaml",
                rows,
            )

            self.assertEqual(len(poses), 32)
            self.assertAlmostEqual(poses[0][0, 3], 100.0)
            self.assertAlmostEqual(poses[23][0, 3], 123.0)
            self.assertAlmostEqual(poses[24][0, 3], 224.0)
            self.assertAlmostEqual(poses[31][0, 3], 231.0)
            self.assertEqual(rows[0]["label"], "1-1")
            self.assertEqual(rows[23]["label"], "8-3")
            self.assertEqual(rows[24]["label"], "inner0")
            self.assertEqual(rows[24]["camera_id"], "sn0")
            self.assertEqual(rows[31]["label"], "inner7")

            text = (output_dir / "camera_tr_studio_rig.yaml").read_text(encoding="utf-8")
            self.assertIn("pose_count: 32", text)
            self.assertIn("- index: 31", text)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pose_convention"]["transform"], "camera_tr_studio_rig")
            self.assertEqual(manifest["inputs"]["viewer_url"], "http://example/viewer")
            self.assertEqual(len(manifest["cameras"]), 32)

    def test_writes_unified_intrinsics_and_extrinsics_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            intrinsics_dir = root / "intrinsics"
            rows = []
            poses = []
            for index in range(32):
                write_intrinsics_yaml(intrinsics_dir / f"intrinsics{index}.yaml", 4000 + index)
                pose = export_extrinsics.np.eye(4)
                pose[:3, 3] = [index, index + 0.1, index + 0.2]
                poses.append(pose)
                rows.append({
                    "index": index,
                    "label": f"cam{index}",
                    "group": "outer" if index < 24 else "inner",
                    "camera_id": f"id{index}",
                    "source_yaml": "pose_source.yaml",
                    "source_index": index,
                })

            unified_path = output_dir / "studio_32_cameras.yaml"
            export_extrinsics.write_unified_camera_yaml(
                unified_path,
                poses,
                rows,
                intrinsics_dir,
            )

            text = unified_path.read_text(encoding="utf-8")
            self.assertIn("schema_version: 1", text)
            self.assertIn("camera_count: 32", text)
            self.assertIn("camera_tr_studio_rig:", text)
            self.assertIn("T_camera_studio = camera_tr_studio_rig", text)
            self.assertIn("p_camera = R @ p_studio + t", text)
            self.assertIn("studio_rig +Y: physical vertical down", text)
            self.assertIn("studio_rig +Z: physical forward", text)
            self.assertIn("C_studio = -R.T @ t", text)
            self.assertIn("intrinsics:", text)
            self.assertIn("model: CentralOpenCVModel", text)
            self.assertIn("parameters: [4000", text)
            self.assertIn('label: "cam31"', text)

    def test_canonical_frame_transform_sets_origin_y_and_gap_axes(self):
        poses = [export_extrinsics.np.eye(4) for _ in range(32)]
        rows = []
        for index, label in enumerate(export_extrinsics.OUTER_CAMERA_LABELS):
            side, level = [int(part) for part in label.split("-")]
            theta = 2.0 * math.pi * (side - 1) / 8.0
            center = [math.cos(theta), float(level - 2), math.sin(theta)]
            if side == 4:
                center = [4.0, 4.0 + float(level), 4.0]
            poses[index] = pose_from_center(center)
            rows.append({
                "index": index,
                "label": label,
                "group": "outer",
                "camera_id": label,
                "source_yaml": "outer.yaml",
                "source_index": index,
            })
        for index in range(24, 32):
            rows.append({
                "index": index,
                "label": f"inner{index - 24}",
                "group": "inner",
                "camera_id": f"inner{index - 24}",
                "source_yaml": "inner.yaml",
                "source_index": index,
            })

        frame = export_extrinsics.estimate_frame_from_camera_poses(poses, rows)
        transformed = [export_extrinsics.transform_pose_to_aligned(pose, frame) for pose in poses]

        centers = {
            row["label"]: studio_canonical_frame.camera_center_from_camera_tr_rig(transformed[row["index"]])
            for row in rows
            if row["label"] in export_extrinsics.OUTER_CAMERA_LABELS
        }
        level2 = [centers[f"{side}-2"] for side in range(1, 9) if side != 4]
        origin = export_extrinsics.np.mean(export_extrinsics.np.asarray(level2), axis=0)
        self.assertLess(float(export_extrinsics.np.linalg.norm(origin)), 1e-9)
        for side in (1, 2, 3, 5, 6, 7, 8):
            self.assertLess(float(centers[f"{side}-3"][1] - centers[f"{side}-1"][1]), 0.0)
        gap_mid = 0.5 * (centers["3-2"] + centers["5-2"])
        self.assertLess(float(gap_mid[2]), -0.1)


if __name__ == "__main__":
    unittest.main()
