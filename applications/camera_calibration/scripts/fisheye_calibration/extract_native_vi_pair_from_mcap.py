#!/usr/bin/env python3
import argparse
import io
import json
from pathlib import Path

from PIL import Image as PilImage

from mcap_to_kalibr_bag import CAMERA_ROWS, camera_tiles, parse_cameras, parse_compressed_image, stamp_ns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract synchronized two-camera image files from a Seeker packed MCAP.")
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cameras", required=True, help="Two packed camera row indices, e.g. 1,2 or 0,3.")
    parser.add_argument("--pair-name", required=True)
    parser.add_argument("--image-topic", default="/seeker/image/packed/compressed")
    parser.add_argument("--layout", default="vertical4")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def write_tsv(path, frames):
    with Path(path).open("w", encoding="utf-8") as f:
        f.write("index\tsrc\ttimestamp_ns\tcam_a_image\tcam_b_image\n")
        for frame in frames:
            f.write(
                f"{frame['index']}\t{frame['src']}\t{frame['timestamp_ns']}\t"
                f"{frame['cam_a_image']}\t{frame['cam_b_image']}\n")


def main():
    from mcap.reader import make_reader

    args = parse_args()
    cameras = parse_cameras(args.cameras)
    if len(cameras) != 2:
        raise ValueError(f"This script expects exactly two cameras, got {cameras}")

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    image_dirs = {}
    for camera_index in cameras:
        image_dirs[camera_index] = images_dir / f"cam{camera_index}"
        image_dirs[camera_index].mkdir(parents=True, exist_ok=True)

    frames = []
    seen = 0
    written = 0
    with open(args.mcap, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(
                topics=[args.image_topic], log_time_order=True):
            if channel.topic != args.image_topic:
                continue
            if args.stride > 1 and seen % args.stride != 0:
                seen += 1
                continue
            if args.max_frames > 0 and written >= args.max_frames:
                break

            parsed = parse_compressed_image(message.data)
            timestamp_ns = stamp_ns(parsed["sec"], parsed["nsec"])
            packed = PilImage.open(io.BytesIO(parsed["data"])).convert("L")
            tiles = camera_tiles(packed.width, packed.height, args.layout)
            paths = {}
            for camera_index in cameras:
                crop = packed.crop(tiles[camera_index])
                image_path = image_dirs[camera_index] / f"{written:06d}_src{seen:06d}_{timestamp_ns}.png"
                crop.save(image_path)
                paths[camera_index] = str(image_path)

            frames.append({
                "index": written,
                "src": seen,
                "timestamp_ns": timestamp_ns,
                "cam_a_image": paths[cameras[0]],
                "cam_b_image": paths[cameras[1]],
                "cam_a_stamp_ns": timestamp_ns,
                "cam_b_stamp_ns": timestamp_ns,
                "cam_a_tag_count": None,
                "cam_b_tag_count": None,
            })
            written += 1
            seen += 1
            if written % 100 == 0:
                print(f"written={written} pair={args.pair_name}", flush=True)

    manifest = {
        "format": "seeker_native_vi_pair_manifest_v0",
        "pair_name": args.pair_name,
        "mcap": str(Path(args.mcap).resolve()),
        "image_topic": args.image_topic,
        "cameras": cameras,
        "camera_rows": {f"cam{i}": CAMERA_ROWS.get(i, "unknown") for i in cameras},
        "frames": frames,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(output_dir / "manifest.tsv", frames)
    print(json.dumps({
        "pair_name": args.pair_name,
        "frames": len(frames),
        "manifest": str((output_dir / "manifest.json").resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
