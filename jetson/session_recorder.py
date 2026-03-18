#!/usr/bin/env python3
"""
Session Recorder — records Qualia beliefs + camera video for training data.

Writes a session directory compatible with convert_to_lerobot.py:
    ~/training-data/sessions/<timestamp>/
    ├── qualia_beliefs.csv    (beliefs at 10Hz)
    ├── video.mp4             (640x480 from USB camera)
    └── session.json          (metadata)

Usage:
    python3 session_recorder.py [--output-dir DIR] [--camera DEVICE] [--no-video]
"""

import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent dir so we can import qualia_bridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qualia_bridge import QualiaBridge, NUM_LAYERS, STATE_DIM


def make_csv_header():
    """Build CSV header matching what convert_to_lerobot.py expects."""
    cols = ["timestamp_ns"]
    for i in range(NUM_LAYERS):
        cols.extend([
            f"l{i}_vfe",
            f"l{i}_compression",
            f"l{i}_confirm_streak",
            f"l{i}_cycle_us",
        ])
    cols.extend(["scene", "activity", "num_objects"])
    for i in range(STATE_DIM):
        cols.append(f"embedding_{i}")
    return cols


def sample_row(bridge):
    """Read one snapshot from SHM and return a dict for the CSV."""
    ts = int(time.time_ns())
    row = {"timestamp_ns": ts}

    for i in range(NUM_LAYERS):
        belief = bridge.read_layer_belief(i)
        row[f"l{i}_vfe"] = f"{belief.vfe:.6f}"
        row[f"l{i}_compression"] = belief.compression
        row[f"l{i}_confirm_streak"] = belief.confirm_streak
        row[f"l{i}_cycle_us"] = belief.cycle_us

    world = bridge.read_world_model()
    row["scene"] = world.scene.replace(",", ";")  # escape commas for CSV
    row["activity"] = world.activity.replace(",", ";")
    row["num_objects"] = world.num_objects
    for i in range(STATE_DIM):
        row[f"embedding_{i}"] = f"{world.scene_embedding[i]:.6f}"

    return row


def start_video_recorder(output_path, camera_device="/dev/video0"):
    """Start ffmpeg to record video from USB camera."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "v4l2",
        "-video_size", "640x480",
        "-i", camera_device,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        str(output_path),
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Record Qualia beliefs + video for training")
    parser.add_argument("--output-dir", default=os.path.expanduser("~/training-data/sessions"))
    parser.add_argument("--camera", default="/dev/video0", help="V4L2 camera device")
    parser.add_argument("--no-video", action="store_true", help="Skip video recording")
    parser.add_argument("--hz", type=int, default=10, help="Sample rate (default: 10)")
    args = parser.parse_args()

    # Connect to Qualia SHM
    bridge = QualiaBridge()
    if not bridge.open():
        print("ERROR: Cannot open Qualia SHM. Is qualia-watch running?")
        sys.exit(1)

    # Create session directory
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc).isoformat()
    sample_interval = 1.0 / args.hz
    sample_count = 0
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Start video recording
    ffmpeg_proc = None
    if not args.no_video:
        video_path = session_dir / "video.mp4"
        ffmpeg_proc = start_video_recorder(video_path, args.camera)
        print(f"Recording video: {video_path}")

    # Open CSV
    csv_path = session_dir / "qualia_beliefs.csv"
    header = make_csv_header()
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=header)
    writer.writeheader()

    print(f"Session: {session_dir}")
    print(f"Recording beliefs at {args.hz}Hz... (Ctrl-C to stop)")

    try:
        while running:
            t0 = time.monotonic()

            row = sample_row(bridge)
            writer.writerow(row)
            sample_count += 1

            if sample_count % (args.hz * 10) == 0:
                csv_file.flush()
                print(f"  {sample_count} samples recorded ({sample_count // args.hz}s)")

            elapsed = time.monotonic() - t0
            sleep_time = sample_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        csv_file.close()

        # Stop video
        if ffmpeg_proc:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=5)

        end_time = datetime.now(timezone.utc).isoformat()

        # Write session metadata
        meta = {
            "session_start": start_time,
            "session_end": end_time,
            "samples": sample_count,
            "sample_hz": args.hz,
            "device": {
                "model": "jetson-orin-nano",
                "camera": args.camera,
            },
            "sensors": {
                "video_fps": 30,
                "video_codec": "h264",
                "video_resolution": [480, 640, 3],
            },
        }
        with open(session_dir / "session.json", "w") as f:
            json.dump(meta, f, indent=2)

        bridge.close()
        print(f"\nSession saved to {session_dir}")
        print(f"  Samples: {sample_count} ({sample_count / args.hz:.1f}s)")
        print(f"  CSV: {csv_path}")
        if ffmpeg_proc:
            print(f"  Video: {session_dir / 'video.mp4'}")
        print(f"\nConvert: python3 scripts/convert_to_lerobot.py {session_dir}")


if __name__ == "__main__":
    main()
