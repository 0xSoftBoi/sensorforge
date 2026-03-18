#!/usr/bin/env python3
"""
Merge multi-device recording sessions into a single aligned timeline.

Aligns iPhone (SensorForge) and Jetson (Qualia + voice assistant) data
using clock sync offsets from the WiFi bridge protocol.

Usage:
    python merge_sessions.py \\
        --iphone /path/to/sensorforge/session \\
        --jetson /path/to/jetson/session \\
        --output /path/to/merged

Output structure (LeRobot v3.0 compatible):
    merged/
    ├── data/
    │   └── episode_000000.parquet
    ├── videos/
    │   └── observation.video/
    │       └── episode_000000.mp4
    ├── meta/
    │   ├── info.json
    │   ├── tasks.jsonl
    │   └── episodes.jsonl
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


def read_csv(path):
    """Read CSV file into list of dicts with numeric conversion."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v) if "." in v else int(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows


def align_timestamps(data, clock_offset_ns, ts_key="timestamp_ns"):
    """Apply clock offset to align timestamps across devices."""
    for row in data:
        if ts_key in row:
            row[ts_key] = int(row[ts_key]) + clock_offset_ns
    return data


def interpolate_nearest(data, timeline_ns, ts_key="timestamp_ns"):
    """Nearest-neighbor interpolation to match target timeline."""
    if not data:
        return [{}] * len(timeline_ns)

    result = []
    idx = 0
    for t in timeline_ns:
        while (idx < len(data) - 1 and
               abs(data[idx + 1].get(ts_key, 0) - t) < abs(data[idx].get(ts_key, 0) - t)):
            idx += 1
        row = dict(data[idx])
        row.pop(ts_key, None)
        result.append(row)
    return result


def merge_sessions(iphone_dir, jetson_dir, output_dir, clock_offset_ns=0):
    """Merge iPhone and Jetson session data into unified timeline."""
    iphone_dir = Path(iphone_dir) if iphone_dir else None
    jetson_dir = Path(jetson_dir) if jetson_dir else None
    output_dir = Path(output_dir)

    # ── Read iPhone data ─────────────────────────────────────────
    iphone_poses = []
    iphone_imu = []
    iphone_gps = []
    iphone_mag = []
    iphone_baro = []

    if iphone_dir and iphone_dir.exists():
        iphone_poses = read_csv(iphone_dir / "poses.csv")
        iphone_imu = read_csv(iphone_dir / "imu.csv")
        iphone_gps = read_csv(iphone_dir / "gps.csv")
        iphone_mag = read_csv(iphone_dir / "magnetometer.csv")
        iphone_baro = read_csv(iphone_dir / "barometer.csv")

        # Apply clock offset to iPhone timestamps
        if clock_offset_ns != 0:
            for dataset in [iphone_poses, iphone_imu, iphone_gps, iphone_mag, iphone_baro]:
                align_timestamps(dataset, clock_offset_ns)

        print(f"iPhone: {len(iphone_poses)} poses, {len(iphone_imu)} IMU, "
              f"{len(iphone_gps)} GPS, {len(iphone_baro)} baro")

    # ── Read Jetson data ─────────────────────────────────────────
    jetson_telemetry = []
    qualia_beliefs = []
    qualia_lore = []

    if jetson_dir and jetson_dir.exists():
        jetson_telemetry = read_csv(jetson_dir / "telemetry.csv")
        qualia_beliefs = read_csv(jetson_dir / "qualia_beliefs.csv")
        qualia_lore = read_csv(jetson_dir / "qualia_lore.csv")

        print(f"Jetson: {len(jetson_telemetry)} telemetry, "
              f"{len(qualia_beliefs)} beliefs, {len(qualia_lore)} lore")

    # ── Build master timeline ────────────────────────────────────
    # Use iPhone poses as primary timeline (30Hz), fall back to Jetson telemetry
    if iphone_poses:
        timeline_ns = [row["timestamp_ns"] for row in iphone_poses]
        timeline_source = "iphone_pose"
    elif jetson_telemetry:
        timeline_ns = [row.get("timestamp_ns", 0) for row in jetson_telemetry]
        timeline_source = "jetson_telemetry"
    elif qualia_beliefs:
        timeline_ns = [row.get("timestamp_ns", 0) for row in qualia_beliefs]
        timeline_source = "qualia_beliefs"
    else:
        print("Error: No timestamped data found from any device.")
        sys.exit(1)

    print(f"Timeline: {len(timeline_ns)} frames from {timeline_source}")

    # ── Interpolate all data to master timeline ──────────────────
    imu_interp = interpolate_nearest(iphone_imu, timeline_ns)
    gps_interp = interpolate_nearest(iphone_gps, timeline_ns)
    mag_interp = interpolate_nearest(iphone_mag, timeline_ns)
    baro_interp = interpolate_nearest(iphone_baro, timeline_ns)
    telemetry_interp = interpolate_nearest(jetson_telemetry, timeline_ns)
    beliefs_interp = interpolate_nearest(qualia_beliefs, timeline_ns)

    # ── Merge into unified rows ──────────────────────────────────
    merged = []
    for i, t in enumerate(timeline_ns):
        row = {"timestamp_ns": int(t), "frame_index": i}

        # iPhone pose
        if i < len(iphone_poses):
            pose = iphone_poses[i]
            for k in ["x", "y", "z", "qx", "qy", "qz", "qw", "pitch", "yaw", "roll"]:
                if k in pose:
                    row[f"iphone.pose.{k}"] = pose[k]

        # iPhone IMU
        for k, v in imu_interp[i].items():
            row[f"iphone.imu.{k}"] = v

        # iPhone GPS
        for k, v in gps_interp[i].items():
            row[f"iphone.gps.{k}"] = v

        # iPhone magnetometer
        for k, v in mag_interp[i].items():
            row[f"iphone.mag.{k}"] = v

        # iPhone barometer
        for k, v in baro_interp[i].items():
            row[f"iphone.baro.{k}"] = v

        # Jetson telemetry
        for k, v in telemetry_interp[i].items():
            row[f"jetson.{k}"] = v

        # Qualia beliefs
        for k, v in beliefs_interp[i].items():
            row[f"qualia.{k}"] = v

        merged.append(row)

    # ── Write output ─────────────────────────────────────────────
    data_dir = output_dir / "data"
    video_dir = output_dir / "videos" / "observation.video"
    meta_dir = output_dir / "meta"
    for d in [data_dir, video_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    episode_name = "episode_000000"

    if HAS_PARQUET and merged:
        table = pa.Table.from_pylist(merged)
        pq.write_table(table, data_dir / f"{episode_name}.parquet")
        print(f"Wrote {len(merged)} rows to {episode_name}.parquet")
    elif merged:
        csv_path = data_dir / f"{episode_name}.csv"
        keys = merged[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(merged)
        print(f"Wrote {len(merged)} rows to {episode_name}.csv")

    # Copy video from iPhone
    if iphone_dir:
        video_src = iphone_dir / "video.mp4"
        if video_src.exists():
            shutil.copy2(video_src, video_dir / f"{episode_name}.mp4")
            print("Copied iPhone video")

    # Copy LORE as supplementary data
    if qualia_lore:
        lore_dir = output_dir / "data" / "lore"
        lore_dir.mkdir(parents=True, exist_ok=True)
        with open(lore_dir / f"{episode_name}_lore.jsonl", "w") as f:
            for entry in qualia_lore:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(qualia_lore)} LORE entries")

    # Write metadata
    features = {}
    if merged:
        for key in merged[0].keys():
            val = merged[0][key]
            if isinstance(val, int) and key != "frame_index":
                features[key] = {"dtype": "int64", "shape": [1]}
            elif isinstance(val, float):
                features[key] = {"dtype": "float64", "shape": [1]}

    devices = []
    if iphone_poses:
        devices.append("iphone")
    if jetson_telemetry or qualia_beliefs:
        devices.append("jetson")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "sensorforge_multi_device",
        "total_episodes": 1,
        "total_frames": len(merged),
        "fps": 30,
        "features": features,
        "devices": devices,
        "clock_offset_ns": clock_offset_ns,
        "timeline_source": timeline_source,
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "multi_device_capture"}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        f.write(json.dumps({
            "episode_index": 0,
            "task_index": 0,
            "length": len(merged),
            "devices": devices,
        }) + "\n")

    print(f"\nMerged dataset: {output_dir}")
    print(f"  Frames: {len(merged)}")
    print(f"  Devices: {', '.join(devices)}")
    col_count = len(merged[0]) if merged else 0
    print(f"  Columns: {col_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multi-device sessions into unified LeRobot dataset"
    )
    parser.add_argument("--iphone", help="iPhone SensorForge session directory")
    parser.add_argument("--jetson", help="Jetson session directory (telemetry + qualia)")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--clock-offset", type=int, default=0,
                        help="Clock offset in nanoseconds (server - client)")
    args = parser.parse_args()

    if not args.iphone and not args.jetson:
        print("Error: provide at least one of --iphone or --jetson")
        sys.exit(1)

    merge_sessions(args.iphone, args.jetson, args.output, args.clock_offset)


if __name__ == "__main__":
    main()
