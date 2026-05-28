#!/usr/bin/env python3
"""Export CentralThinPrismFisheye intrinsics as KB8 JSON."""

import argparse
import ast
import json
import re
from pathlib import Path


def read_field(text, name):
    match = re.search(rf"^{re.escape(name)}\s*:\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing field {name}")
    return match.group(1).strip()


def load_intrinsics(path):
    text = Path(path).read_text(encoding="utf-8")
    model_type = read_field(text, "type")
    if model_type != "CentralThinPrismFisheyeModel":
        raise ValueError(f"Expected CentralThinPrismFisheyeModel, got {model_type}")

    params = ast.literal_eval(read_field(text, "parameters"))
    if len(params) < 8:
        raise ValueError(f"Expected at least 8 parameters, got {len(params)}")

    return {
        "source_file": str(path),
        "source_model": model_type,
        "width": int(read_field(text, "width")),
        "height": int(read_field(text, "height")),
        "model": "KB8",
        "projection": "Kannala-Brandt equidistant",
        "parameter_order": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
        "params": [float(v) for v in params[:8]],
        "ignored_source_params": {
            "p1": float(params[8]) if len(params) > 8 else None,
            "p2": float(params[9]) if len(params) > 9 else None,
            "sx1": float(params[10]) if len(params) > 10 else None,
            "sy1": float(params[11]) if len(params) > 11 else None,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intrinsics", nargs="+", required=True)
    parser.add_argument("--camera-names", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.camera_names and len(args.camera_names) != len(args.intrinsics):
        raise ValueError("--camera-names must match --intrinsics length")

    cameras = {}
    for index, path in enumerate(args.intrinsics):
        name = args.camera_names[index] if args.camera_names else f"cam{index}"
        cameras[name] = load_intrinsics(path)

    output = {
        "format": "KB8",
        "convention": "params = [fx, fy, cx, cy, k1, k2, k3, k4]",
        "notes": [
            "Source model is CentralThinPrismFisheyeModel.",
            "Only the four radial Kannala-Brandt terms are exported.",
            "Thin-prism/tangential source terms are listed under ignored_source_params.",
        ],
        "cameras": cameras,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
