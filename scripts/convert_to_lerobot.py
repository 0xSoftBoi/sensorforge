#!/usr/bin/env python3
"""
Convert a SensorForge session to LeRobot v3.0 dataset format.

Usage:
    python convert_to_lerobot.py /path/to/session [--output /path/to/output]

Reads all CSVs from a SensorForge session directory, merges them to a 30Hz
timeline aligned to pose timestamps, and outputs in LeRobot v3.0 format.

Output structure:
    output/
    ├── data/
    │   └── episode_000000.parquet  (or .csv fallback)
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
    print("Warning: pyarrow not installed. Will output CSV instead of Parquet.")


def read_csv(path):
    """Read a CSV file and return list of dicts with numeric conversion."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
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


def interpolate_to_timeline(data, timeline_ns, ts_key="timestamp_ns"):
    """Nearest-neighbor interpolation of data to match timeline timestamps."""
    if not data:
        return [{}] * len(timeline_ns)

    result = []
    data_idx = 0
    for t in timeline_ns:
        # Advance to nearest sample
        while (data_idx < len(data) - 1 and
               abs(data[data_idx + 1][ts_key] - t) < abs(data[data_idx][ts_key] - t)):
            data_idx += 1
        row = dict(data[data_idx])
        row.pop(ts_key, None)
        result.append(row)
    return result


def convert_session(session_dir, output_dir):
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)

    # Read session metadata
    meta_path = session_dir / "session.json"
    if meta_path.exists():
        with open(meta_path) as f:
            session_meta = json.load(f)
    else:
        session_meta = {}

    # Read all sensor CSVs
    poses = read_csv(session_dir / "poses.csv")
    imu = read_csv(session_dir / "imu.csv")
    gps = read_csv(session_dir / "gps.csv")
    mag = read_csv(session_dir / "magnetometer.csv")
    baro = read_csv(session_dir / "barometer.csv")
    ble = read_csv(session_dir / "ble_telemetry.csv")

    if not poses:
        print("Error: No pose data found. Cannot build timeline.")
        sys.exit(1)

    # Use pose timestamps as the master 30Hz timeline
    timeline_ns = [row["timestamp_ns"] for row in poses]

    # Interpolate all sensors to pose timeline
    imu_interp = interpolate_to_timeline(imu, timeline_ns)
    gps_interp = interpolate_to_timeline(gps, timeline_ns)
    mag_interp = interpolate_to_timeline(mag, timeline_ns)
    baro_interp = interpolate_to_timeline(baro, timeline_ns)

    # Read Qualia beliefs if present (exported by qualia_bridge)
    qualia_beliefs = read_csv(session_dir / "qualia_beliefs.csv")

    # Interpolate Qualia data to pose timeline
    qualia_interp = interpolate_to_timeline(qualia_beliefs, timeline_ns) if qualia_beliefs else None

    # Merge into unified rows
    merged = []
    for i, t in enumerate(timeline_ns):
        row = {"timestamp_ns": int(t), "frame_index": i}

        # Pose
        pose = poses[i]
        for k in ["x", "y", "z", "qx", "qy", "qz", "qw", "pitch", "yaw", "roll", "tracking"]:
            if k in pose:
                row[f"pose.{k}"] = pose[k]

        # IMU
        for k, v in imu_interp[i].items():
            row[f"imu.{k}"] = v

        # GPS
        for k, v in gps_interp[i].items():
            row[f"gps.{k}"] = v

        # Magnetometer
        for k, v in mag_interp[i].items():
            row[f"mag.{k}"] = v

        # Barometer
        for k, v in baro_interp[i].items():
            row[f"baro.{k}"] = v

        # Qualia beliefs (7 layers x VFE + compression)
        if qualia_interp and qualia_interp[i]:
            for k, v in qualia_interp[i].items():
                row[f"qualia.{k}"] = v

        merged.append(row)

    # Create output directory structure
    data_dir = output_dir / "data"
    video_dir = output_dir / "videos" / "observation.video"
    meta_dir = output_dir / "meta"
    for d in [data_dir, video_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Write data
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

    # Copy video
    video_src = session_dir / "video.mp4"
    if video_src.exists():
        shutil.copy2(video_src, video_dir / f"{episode_name}.mp4")
        print("Copied video.mp4")

    # Copy BLE telemetry (raw, not interpolated — variable-rate data)
    ble_src = session_dir / "ble_telemetry.csv"
    if ble_src.exists() and ble:
        ble_dir = output_dir / "data" / "ble"
        ble_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ble_src, ble_dir / f"{episode_name}_ble.csv")
        print(f"Copied BLE telemetry ({len(ble)} rows)")

    # Copy audio
    audio_src = session_dir / "audio.wav"
    if audio_src.exists():
        audio_dir = output_dir / "data" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_src, audio_dir / f"{episode_name}.wav")
        print("Copied audio.wav")

    # Write info.json (LeRobot v3.0 schema)
    # Build features from first merged row
    features = {}
    if merged:
        for key in merged[0].keys():
            val = merged[0][key]
            if isinstance(val, int) and key != "frame_index":
                features[key] = {"dtype": "int64", "shape": [1]}
            elif isinstance(val, float):
                features[key] = {"dtype": "float64", "shape": [1]}
            elif isinstance(val, str):
                features[key] = {"dtype": "string", "shape": [1]}

    # Get video resolution from session metadata, fall back to defaults
    video_resolution = [1080, 1920, 3]
    sensors_meta = session_meta.get("sensors", {})
    video_fps = sensors_meta.get("video_fps", 30)
    video_codec = sensors_meta.get("video_codec", "hevc")

    features["observation.video"] = {
        "dtype": "video",
        "shape": video_resolution,
        "video_info": {
            "video.fps": video_fps,
            "video.codec": video_codec,
        }
    }

    info = {
        "codebase_version": "v3.0",
        "robot_type": session_meta.get("device", {}).get("model", "iphone"),
        "total_episodes": 1,
        "total_frames": len(merged),
        "fps": 30,
        "features": features,
        "data_path": f"data/{episode_name}.parquet" if HAS_PARQUET else f"data/{episode_name}.csv",
        "video_path": f"videos/observation.video/{episode_name}.mp4",
        "splits": {"train": f"0:{len(merged)}"},
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Wrote meta/info.json")

    # Write tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        task = {"task_index": 0, "task": "multi_modal_capture"}
        f.write(json.dumps(task) + "\n")

    # Write episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        ep = {
            "episode_index": 0,
            "task_index": 0,
            "length": len(merged),
            "session_start": session_meta.get("session_start", ""),
            "session_end": session_meta.get("session_end", ""),
            "device": session_meta.get("device", {}),
        }
        f.write(json.dumps(ep) + "\n")

    print(f"\nDone! LeRobot dataset written to: {output_dir}")
    print(f"  Episodes: 1")
    print(f"  Frames: {len(merged)}")
    sensors = ["pose", "imu", "gps", "magnetometer", "barometer", "video"]
    if qualia_beliefs:
        sensors.append("qualia")
    print(f"  Sensors: {', '.join(sensors)}")
    if ble:
        print(f"  BLE telemetry rows: {len(ble)} (raw, not interpolated)")
    if qualia_beliefs:
        print(f"  Qualia belief rows: {len(qualia_beliefs)} (interpolated to timeline)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SensorForge session to LeRobot v3.0 format"
    )
    parser.add_argument("session_dir", help="Path to SensorForge session directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: session_dir_lerobot)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.session_dir.rstrip("/") + "_lerobot"

    convert_session(args.session_dir, args.output)


if __name__ == "__main__":
    main()
