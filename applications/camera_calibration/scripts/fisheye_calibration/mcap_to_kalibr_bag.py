#!/usr/bin/env python3
import argparse
import io
import json
import math
import struct
from pathlib import Path

from PIL import Image as PilImage


CAMERA_ROWS = {
    0: "left_up",
    1: "left_down",
    2: "right_down",
    3: "right_up",
}


def align(offset, size):
    return (offset + size - 1) & ~(size - 1)


class CdrReader:
    def __init__(self, data):
        self.data = data
        self.offset = 4

    def read_i32(self):
        self.offset = align(self.offset, 4)
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_u32(self):
        self.offset = align(self.offset, 4)
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_f64(self):
        self.offset = align(self.offset, 8)
        value = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return value

    def read_string(self):
        size = self.read_u32()
        raw = self.data[self.offset:self.offset + size]
        self.offset += size
        self.offset = align(self.offset, 4)
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")

    def read_header(self):
        sec = self.read_i32()
        nsec = self.read_u32()
        frame_id = self.read_string()
        return sec, nsec, frame_id

    def read_f64_array(self, count):
        return [self.read_f64() for _ in range(count)]

    def read_u8_sequence(self):
        size = self.read_u32()
        raw = self.data[self.offset:self.offset + size]
        self.offset += size
        return bytes(raw)


def parse_compressed_image(data):
    reader = CdrReader(data)
    sec, nsec, frame_id = reader.read_header()
    image_format = reader.read_string()
    payload = reader.read_u8_sequence()
    return {
        "sec": sec,
        "nsec": nsec,
        "frame_id": frame_id,
        "format": image_format,
        "data": payload,
    }


def parse_imu(data):
    reader = CdrReader(data)
    sec, nsec, frame_id = reader.read_header()
    orientation = reader.read_f64_array(4)
    orientation_covariance = reader.read_f64_array(9)
    angular_velocity = reader.read_f64_array(3)
    angular_velocity_covariance = reader.read_f64_array(9)
    linear_acceleration = reader.read_f64_array(3)
    linear_acceleration_covariance = reader.read_f64_array(9)
    return {
        "sec": sec,
        "nsec": nsec,
        "frame_id": frame_id,
        "orientation": orientation,
        "orientation_covariance": orientation_covariance,
        "angular_velocity": angular_velocity,
        "angular_velocity_covariance": angular_velocity_covariance,
        "linear_acceleration": linear_acceleration,
        "linear_acceleration_covariance": linear_acceleration_covariance,
    }


def make_ros_time(sec, nsec):
    import rospy

    return rospy.Time(int(sec), int(nsec))


def stamp_ns(sec, nsec):
    return int(sec) * 1_000_000_000 + int(nsec)


def camera_tiles(width, height, layout):
    if layout != "vertical4":
        raise ValueError(f"Unsupported layout: {layout}")
    if height % 4 != 0:
        raise ValueError(f"Expected vertical4 packed height divisible by 4, got {width}x{height}")
    tile_h = height // 4
    return [(0, index * tile_h, width, (index + 1) * tile_h) for index in range(4)]


def finite_or_zero(value):
    return float(value) if math.isfinite(float(value)) else 0.0


def write_image_message(bag, topic, stamp, frame_id, gray_image):
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = gray_image.height
    msg.width = gray_image.width
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = gray_image.width
    msg.data = gray_image.tobytes()
    bag.write(topic, msg, t=stamp)


def write_imu_message(bag, topic, parsed):
    from sensor_msgs.msg import Imu

    stamp = make_ros_time(parsed["sec"], parsed["nsec"])
    msg = Imu()
    msg.header.stamp = stamp
    msg.header.frame_id = parsed["frame_id"] or "imu"
    msg.orientation.x = finite_or_zero(parsed["orientation"][0])
    msg.orientation.y = finite_or_zero(parsed["orientation"][1])
    msg.orientation.z = finite_or_zero(parsed["orientation"][2])
    msg.orientation.w = finite_or_zero(parsed["orientation"][3])
    msg.orientation_covariance = [finite_or_zero(v) for v in parsed["orientation_covariance"]]
    msg.angular_velocity.x = finite_or_zero(parsed["angular_velocity"][0])
    msg.angular_velocity.y = finite_or_zero(parsed["angular_velocity"][1])
    msg.angular_velocity.z = finite_or_zero(parsed["angular_velocity"][2])
    msg.angular_velocity_covariance = [finite_or_zero(v) for v in parsed["angular_velocity_covariance"]]
    msg.linear_acceleration.x = finite_or_zero(parsed["linear_acceleration"][0])
    msg.linear_acceleration.y = finite_or_zero(parsed["linear_acceleration"][1])
    msg.linear_acceleration.z = finite_or_zero(parsed["linear_acceleration"][2])
    msg.linear_acceleration_covariance = [finite_or_zero(v) for v in parsed["linear_acceleration_covariance"]]
    bag.write(topic, msg, t=stamp)


def parse_cameras(value):
    cameras = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index > 3:
            raise ValueError(f"Camera index must be in [0, 3], got {index}")
        cameras.append(index)
    if not cameras:
        raise ValueError("No cameras selected")
    return cameras


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Seeker packed ROS2 MCAP into a Kalibr-compatible ROS1 bag.")
    parser.add_argument("--mcap", required=True)
    parser.add_argument("--output-bag", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--image-topic", default="/seeker/image/packed/compressed")
    parser.add_argument("--imu-topic-in", default="/seeker/imu")
    parser.add_argument("--imu-topic-out", default="/imu0")
    parser.add_argument("--cameras", required=True,
                        help="Comma-separated packed camera row indices, e.g. 1,2 or 0,3.")
    parser.add_argument("--topic-prefix", default="/cam")
    parser.add_argument("--layout", default="vertical4")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-imu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from mcap.reader import make_reader
    import rosbag

    selected_cameras = parse_cameras(args.cameras)
    output_bag = Path(args.output_bag)
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else output_bag.with_suffix(".summary.json")

    image_counts = {str(index): 0 for index in selected_cameras}
    first_stamp_ns = None
    last_stamp_ns = None
    imu_count = 0
    packed_count = 0
    image_topic_names = {
        index: f"{args.topic_prefix}{index}/image_raw"
        for index in selected_cameras
    }

    with open(args.mcap, "rb") as f, rosbag.Bag(str(output_bag), "w", compression="lz4") as bag:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(
                topics=[args.image_topic, args.imu_topic_in], log_time_order=True):
            if channel.topic == args.imu_topic_in:
                if args.skip_imu:
                    continue
                parsed_imu = parse_imu(message.data)
                write_imu_message(bag, args.imu_topic_out, parsed_imu)
                imu_count += 1
                ns = stamp_ns(parsed_imu["sec"], parsed_imu["nsec"])
                first_stamp_ns = ns if first_stamp_ns is None else min(first_stamp_ns, ns)
                last_stamp_ns = ns if last_stamp_ns is None else max(last_stamp_ns, ns)
                continue

            if channel.topic != args.image_topic:
                continue
            if args.stride > 1 and packed_count % args.stride != 0:
                packed_count += 1
                continue
            if args.max_frames > 0 and packed_count >= args.max_frames:
                break

            parsed_image = parse_compressed_image(message.data)
            stamp = make_ros_time(parsed_image["sec"], parsed_image["nsec"])
            packed = PilImage.open(io.BytesIO(parsed_image["data"])).convert("L")
            tiles = camera_tiles(packed.width, packed.height, args.layout)
            for camera_index in selected_cameras:
                tile = tiles[camera_index]
                crop = packed.crop(tile)
                frame_id = f"cam{camera_index}_{CAMERA_ROWS.get(camera_index, 'unknown')}"
                write_image_message(bag, image_topic_names[camera_index], stamp, frame_id, crop)
                image_counts[str(camera_index)] += 1
            ns = stamp_ns(parsed_image["sec"], parsed_image["nsec"])
            first_stamp_ns = ns if first_stamp_ns is None else min(first_stamp_ns, ns)
            last_stamp_ns = ns if last_stamp_ns is None else max(last_stamp_ns, ns)
            packed_count += 1
            if packed_count % 100 == 0:
                print(f"packed_frames={packed_count} image_counts={image_counts} imu_count={imu_count}", flush=True)

    summary = {
        "mcap": str(args.mcap),
        "output_bag": str(output_bag),
        "image_topic_in": args.image_topic,
        "imu_topic_in": args.imu_topic_in,
        "imu_topic_out": args.imu_topic_out,
        "selected_cameras": selected_cameras,
        "camera_rows": {str(k): CAMERA_ROWS.get(k, "unknown") for k in selected_cameras},
        "image_topics": {str(k): v for k, v in image_topic_names.items()},
        "image_counts": image_counts,
        "imu_count": imu_count,
        "packed_frames_written": packed_count,
        "first_stamp_ns": first_stamp_ns,
        "last_stamp_ns": last_stamp_ns,
        "duration_s": None if first_stamp_ns is None or last_stamp_ns is None else (last_stamp_ns - first_stamp_ns) / 1e9,
        "image_encoding": "mono8",
        "bag_compression": "lz4",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
