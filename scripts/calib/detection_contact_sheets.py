#!/usr/bin/env python3
"""Create per-camera detection contact sheets from a calibration dataset."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from dataset_coverage_report import read_dataset, read_manifest


COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def parse_image_directories(args):
    if args.image_directories_file:
        text = Path(args.image_directories_file).read_text().strip()
    else:
        text = args.image_directories or ""
    dirs = [Path(item).expanduser() for item in text.split(",") if item.strip()]
    if not dirs:
        raise SystemExit("No image directories were provided.")
    for path in dirs:
        if not path.is_dir():
            raise SystemExit(f"Image directory does not exist: {path}")
    return dirs


def choose_frame_indices(dataset, camera_index, max_samples):
    scored = []
    for frame_index, imageset in enumerate(dataset["imagesets"]):
        features = imageset["features"][camera_index]
        if features:
            scored.append((frame_index, len(features)))
    if not scored:
        return []

    candidate_indices = {
        scored[0][0],
        scored[len(scored) // 2][0],
        scored[-1][0],
        max(scored, key=lambda item: item[1])[0],
    }

    if max_samples > len(candidate_indices):
        step = max(1, len(scored) // max_samples)
        for offset in range(0, len(scored), step):
            candidate_indices.add(scored[offset][0])
            if len(candidate_indices) >= max_samples:
                break

    return sorted(candidate_indices)


def draw_overlay(image, features, scale, label):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    by_tag = {}
    for x, y, feature_id in features:
        tag_id = feature_id // 4
        corner_id = feature_id % 4
        by_tag.setdefault(tag_id, {})[corner_id] = (x * scale, y * scale)

    for tag_id, corners in by_tag.items():
        color = COLORS[tag_id % len(COLORS)]
        if len(corners) >= 4:
            pts = [corners[index] for index in sorted(corners)]
            draw.line(pts + [pts[0]], fill=color, width=3)
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            draw.text((cx + 4, cy + 4), str(tag_id), fill=color, font=small_font)
        for corner_id, point in corners.items():
            x, y = point
            r = 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
            draw.text((x + 4, y - 6), str(corner_id), fill=color, font=small_font)

    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255), font=font)


def make_tile(image_path, imageset, camera_index, tile_width):
    image = Image.open(image_path).convert("RGB")
    scale = tile_width / image.width
    tile_height = max(1, int(round(image.height * scale)))
    image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
    features = imageset["features"][camera_index]
    label = f"{imageset['filename']} | {len(features)} corners | {len({f[2] // 4 for f in features})} tags"
    draw_overlay(image, features, scale, label)
    return image


def save_contact_sheet(path, tiles, columns):
    if not tiles:
        return
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (245, 247, 250))
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(tile, (x, y))
    sheet.save(path, quality=92)


def write_index(path, rows):
    cards = []
    for row in rows:
        cards.append(f"""
<a class="card {html.escape(row['status'])}" href="{html.escape(row['filename'])}">
  <strong>{html.escape(row['user_id'])}</strong>
  <span>{html.escape(row['stage_name'])}</span>
  <small>{row['sample_count']} samples · {row['positive_frames']} positive frames</small>
</a>""")
    path.write_text(f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Detection Contact Sheets</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #17202a; font-family: Inter, system-ui, sans-serif; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 10px; }}
    p {{ color: #667085; }}
    .grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 18px; }}
    .card {{ display: block; text-decoration: none; color: inherit; background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 12px; }}
    .card strong, .card span, .card small {{ display: block; }}
    .card span, .card small {{ color: #667085; font-size: 12px; }}
    .empty {{ opacity: .58; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(3, 1fr); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Detection Contact Sheets</h1>
    <p>Representative positive frames with tag corner overlays.</p>
    <div class="grid">{''.join(cards)}</div>
  </div>
</body>
</html>
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--image-directories")
    parser.add_argument("--image-directories-file", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--tile-width", type=int, default=640)
    args = parser.parse_args()

    dataset = read_dataset(args.dataset)
    image_dirs = parse_image_directories(args)
    if len(image_dirs) != dataset["camera_count"]:
        raise SystemExit(
            f"Got {len(image_dirs)} image directories for {dataset['camera_count']} dataset cameras")
    manifest = read_manifest(args.manifest, dataset["camera_count"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for camera_index in range(dataset["camera_count"]):
        sample_indices = choose_frame_indices(dataset, camera_index, args.max_samples)
        tiles = []
        for frame_index in sample_indices:
            imageset = dataset["imagesets"][frame_index]
            image_path = image_dirs[camera_index] / imageset["filename"]
            if not image_path.is_file():
                continue
            tiles.append(make_tile(image_path, imageset, camera_index, args.tile_width))

        entry = manifest[camera_index]
        filename = f"camera_{camera_index:02d}_{entry['user_id']}.jpg"
        if tiles:
            save_contact_sheet(args.output_dir / filename, tiles, args.columns)
            status = "ok"
        else:
            status = "empty"
        positive_frames = sum(
            1 for imageset in dataset["imagesets"]
            if imageset["features"][camera_index])
        index_rows.append({
            "filename": filename,
            "status": status,
            "sample_count": len(tiles),
            "positive_frames": positive_frames,
            "user_id": entry["user_id"],
            "stage_name": entry["stage_name"],
        })
        print(f"camera {camera_index:02d} {entry['user_id']}: {len(tiles)} samples")

    write_index(args.output_dir / "index.html", index_rows)


if __name__ == "__main__":
    raise SystemExit(main())
