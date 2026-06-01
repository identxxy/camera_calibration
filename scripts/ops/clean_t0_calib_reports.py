#!/usr/bin/env python3
"""Delete selected t0 calibration report clutter from an audit inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


DEFAULT_ROOT = "/home/ubuntu/calib_data"
DEFAULT_INVENTORY = "/home/ubuntu/calib_data/report_audit_20260529_current/report_inventory.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--selector",
        choices=["delete-known-bad", "placeholder-viewers"],
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def select_paths(root: Path, inventory: dict, selector: str) -> list[Path]:
    paths = []
    for entry in inventory["entries"]:
        rel_path = entry["rel_path"]
        if selector == "delete-known-bad" and entry["category"] == "delete_known_bad":
            group = entry.get("delete_group") or rel_path
            paths.append(root / group)
        elif (
            selector == "placeholder-viewers"
            and entry["category"] == "delete_candidate"
            and entry["reason"].startswith("viewer is an explicit placeholder")
        ):
            paths.append((root / rel_path).parent)
    return sorted(set(path.resolve() for path in paths))


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    inventory_path = Path(args.inventory).resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    paths = select_paths(root, inventory, args.selector)
    root_prefix = str(root) + "/"
    for path in paths:
        if not str(path).startswith(root_prefix):
            raise RuntimeError(f"refusing to delete outside root: {path}")

    action = "delete" if args.execute else "dry-run"
    print(f"{action}: {len(paths)} paths selected")
    for path in paths:
        rel = path.relative_to(root)
        print(rel)
        if args.execute and path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
