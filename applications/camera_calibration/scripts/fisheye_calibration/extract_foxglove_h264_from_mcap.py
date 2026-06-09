#!/usr/bin/env python3
"""Extract Foxglove ROS1 CompressedVideo H264 payloads from an MCAP file."""

import argparse
import json
import struct
from pathlib import Path

from mcap.reader import make_reader


DEFAULT_TOPICS = {
    "left_up": "/camera/left_up/h264",
    "left_down": "/camera/left_down/h264",
    "right_down": "/camera/right_down/h264",
    "right_up": "/camera/right_up/h264",
}


def read_ros1_string(data, offset):
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    value = data[offset:offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def parse_compressed_video_ros1(data):
    data = bytes(data)
    if len(data) < 8:
        raise ValueError("CompressedVideo message is too short")
    sec, nsec = struct.unpack_from("<II", data, 0)
    offset = 8
    frame_id, offset = read_ros1_string(data, offset)
    payload_len = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    payload = data[offset:offset + payload_len]
    offset += payload_len
    fmt, offset = read_ros1_string(data, offset)
    if offset != len(data):
        raise ValueError(f"Trailing bytes in CompressedVideo message: parsed={offset}, size={len(data)}")
    return {
        "timestamp_ns": int(sec) * 1_000_000_000 + int(nsec),
        "frame_id": frame_id,
        "format": fmt,
        "data": bytes(payload),
    }


def parse_topic_map(raw):
    if not raw:
        return DEFAULT_TOPICS
    data = json.loads(raw)
    out = dict(DEFAULT_TOPICS)
    out.update({str(k): str(v) for k, v in data.items()})
    return out


def extract_streams(mcap_path, output_root, topics):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    topic_to_name = {topic: name for name, topic in topics.items()}
    handles = {
        name: (output_root / f"{name}.h264").open("wb")
        for name in topics
    }
    summary = {
        name: {
            "topic": topic,
            "frames": 0,
            "first_timestamp_ns": None,
            "last_timestamp_ns": None,
            "formats": {},
            "path": str((output_root / f"{name}.h264").resolve()),
        }
        for name, topic in topics.items()
    }
    try:
        with Path(mcap_path).open("rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                name = topic_to_name.get(channel.topic)
                if name is None:
                    continue
                if schema and schema.name and schema.name != "foxglove_msgs/CompressedVideo":
                    continue
                decoded = parse_compressed_video_ros1(message.data)
                handles[name].write(decoded["data"])
                item = summary[name]
                item["frames"] += 1
                item["first_timestamp_ns"] = item["first_timestamp_ns"] or decoded["timestamp_ns"]
                item["last_timestamp_ns"] = decoded["timestamp_ns"]
                item["formats"][decoded["format"]] = item["formats"].get(decoded["format"], 0) + 1
    finally:
        for handle in handles.values():
            handle.close()
    for item in summary.values():
        first = item["first_timestamp_ns"]
        last = item["last_timestamp_ns"]
        frames = item["frames"]
        if first is not None and last is not None and frames > 1:
            item["duration_s"] = (last - first) / 1_000_000_000.0
            item["fps_estimate"] = (frames - 1) / max(1e-9, item["duration_s"])
        else:
            item["duration_s"] = 0.0
            item["fps_estimate"] = 0.0
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--topic-map-json",
                        help="Optional JSON object mapping camera names to MCAP topics.")
    parser.add_argument("--summary-json")
    args = parser.parse_args()

    topics = parse_topic_map(args.topic_map_json)
    summary = extract_streams(args.mcap, args.output_root, topics)
    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output_root) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
