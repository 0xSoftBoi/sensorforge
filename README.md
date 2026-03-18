# SensorForge

Unified robotics monorepo: iOS sensor capture, Jetson voice assistant, and Qualia active inference engine.

## Architecture

```
┌─────────────┐    WiFi/TCP    ┌──────────────────────────────────────┐
│  iPhone      │◄─────────────►│  Jetson Orin Nano                    │
│  SensorForge │    sensors +   │                                      │
│  (ios/)      │    commands    │  ┌──────────────┐  SHM  ┌─────────┐ │
└─────────────┘               │  │ Voice Assist. │◄────►│ Qualia  │ │
                               │  │ (jetson/)     │       │ Engine  │ │
                               │  └──────────────┘       └─────────┘ │
                               │         │                     │      │
                               │         └──────┬──────────────┘      │
                               │                │                     │
                               │         ┌──────▼──────┐              │
                               │         │ Web Dashboard│ :8080       │
                               │         └─────────────┘              │
                               └──────────────────────────────────────┘
```

The iPhone captures 11 sensor streams and sends decimated data over WiFi to the Jetson. The Jetson runs a fully offline voice assistant (wake word → STT → LLM → TTS) with 24 tools for system monitoring, robot control, and Qualia brain inspection. The Qualia engine is a 7-layer active inference hierarchy running on GPU (Metal on Mac, CUDA on Jetson) that learns to predict its sensory input, with Gemini providing semantic grounding.

## Components

| Directory | Stack | Description | Docs |
|-----------|-------|-------------|------|
| [`ios/`](ios/) | Swift / ARKit | iPhone sensor capture (11 sensors), BLE bridge, WiFi streaming | [ios/README.md](ios/README.md) |
| [`jetson/`](jetson/) | Python | Offline voice assistant with wake word, STT, LLM, TTS, 24 tools | [jetson/README.md](jetson/README.md) |
| [`qualia/`](qualia/) | Rust + CUDA/Metal | Active inference engine — 7-layer belief hierarchy, SHM IPC, web dashboard | [qualia/README.md](qualia/README.md) |
| [`protocol/`](protocol/) | JSON + Python | WiFi bridge message protocol | [protocol/README.md](protocol/README.md) |
| [`scripts/`](scripts/) | Python | Data conversion tools (LeRobot format, session merging) | — |

## Prerequisites

### Hardware

- **iPhone 12 Pro or newer** — LiDAR sensor required for depth capture
- **Jetson Orin Nano** (8GB recommended) — runs voice assistant + Qualia engine
- **USB microphone + speaker** — for voice assistant audio I/O
- **UGV chassis** (optional) — Traxxas or similar for motor control

### Software

| Component | Requires |
|-----------|----------|
| iOS app | macOS + Xcode 15+, iOS 17+ device |
| Jetson assistant | Python 3.10+, Ollama, whisper.cpp, Piper TTS |
| Qualia (Mac) | Rust 1.75+, Xcode CLT (Metal SDK), ffmpeg |
| Qualia (Jetson) | Rust 1.75+, CUDA toolkit, JetPack 6+ |

## Quick Start

### iOS (SensorForge)

```bash
open ios/SensorForge.xcodeproj
# Build and run on iPhone 12 Pro or newer
# Grant permissions: Camera, Mic, Location, Bluetooth
```

The app auto-discovers the Jetson via Bonjour on the local network. See [ios/README.md](ios/README.md) for detailed setup.

### Jetson Voice Assistant

```bash
cd jetson
bash setup.sh                              # Install all dependencies
python3 voice_assistant.py                 # Run assistant
python3 voice_assistant.py --test-tools    # Test all 24 tools
```

Requires Ollama running locally (`curl -fsSL https://ollama.com/install.sh | sh && ollama pull gemma3:1b`). See [jetson/README.md](jetson/README.md) for full setup.

### Qualia Engine (Mac — Metal)

```bash
cd qualia
export GEMINI_API_KEY="your-key-here"      # Optional: runs offline without it
cargo build --release
cargo run --release -p qualia-watch        # TUI supervisor
```

### Qualia Engine (Jetson — CUDA)

```bash
cd qualia
cargo build --release --features cuda
cargo run --release -p qualia-watch        # TUI supervisor
```

### Web Dashboard

```bash
cargo run --release -p qualia-agent        # http://localhost:8080
```

See [qualia/README.md](qualia/README.md) for architecture details, TUI controls, and environment variables.

## Sensors Captured (iOS)

| Sensor | Rate | Source | Output File |
|--------|------|--------|-------------|
| 4K RGB Video | 30fps | ARKit | `video.mp4` |
| LiDAR Depth | 30fps | ARKit sceneDepth | `depth/*.bin` |
| 6-DoF Camera Pose | 60Hz | ARKit VIO | `poses.csv` |
| 3D Scene Mesh | Real-time | ARKit | `surfaces.jsonl` |
| IMU (Accel + Gyro) | 100-200Hz | CoreMotion | `imu.csv` |
| Magnetometer | 50Hz | CoreMotion | `magnetometer.csv` |
| GPS (L1+L5) | 1Hz | CoreLocation | `gps.csv` |
| Barometer | 1Hz | CMAltimeter | `barometer.csv` |
| Spatial Audio | 48kHz | AVAudioEngine | `audio.wav` |
| Ambient Light | 30Hz | ARKit | (in sensor_frame) |
| BLE Device Telemetry | Varies | CoreBluetooth | (via WiFi bridge) |

## WiFi Bridge

The iPhone and Jetson communicate over TCP (port 9876) using length-prefixed JSON messages. The iPhone discovers the Jetson via Bonjour (`_sensorforge._tcp`) on the local network.

**iPhone → Jetson**: Decimated sensor data at ~10Hz (IMU, pose, GPS, barometer, light)
**Jetson → iPhone**: Voice state, Qualia layer status, scene description, UGV battery

See [protocol/README.md](protocol/README.md) for the full message specification.

## IPC via Shared Memory

Qualia uses POSIX shared memory (`/dev/shm/qualia_body`, 64 MiB) for zero-copy IPC between all Jetson processes. The Python voice assistant reads Qualia beliefs via `mmap` + `struct.unpack`, matching the `repr(C)` layout defined in `qualia/crates/types/src/lib.rs`.

## LeRobot Export

Convert recorded sessions to [LeRobot](https://github.com/huggingface/lerobot) v3.0 format:

```bash
python3 scripts/convert_to_lerobot.py path/to/session output/
python3 scripts/merge_sessions.py session1/ session2/ merged_output/
```

## Development

Each component can be developed independently:

- **iOS**: Open `ios/SensorForge.xcodeproj` in Xcode. No external dependencies.
- **Jetson**: Edit Python files in `jetson/`. Test tools with `--test-tools` flag.
- **Qualia**: `cargo build --release` in `qualia/`. Use `RUST_LOG=debug` for verbose output. `cargo test` to run tests.
- **Protocol**: Shared between iOS (Swift) and Jetson (Python). Edit `protocol/schema.json` first, then update `protocol/messages.py` and `ios/SensorForge/WiFiBridge.swift` to match.

See [CONTRIBUTING.md](CONTRIBUTING.md) for code style and PR guidelines.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No LiDAR depth data | Ensure you're using an iPhone with LiDAR (12 Pro, 13 Pro, 14 Pro, 15 Pro, or newer) |
| iPhone can't find Jetson | Both devices must be on the same WiFi network. Check that port 9876 isn't blocked |
| Qualia runs without Gemini | This is expected — without `GEMINI_API_KEY`, it runs offline with synthetic data and hash-based embeddings |
| Shared memory permission denied | Qualia-watch must be run first (it creates the SHM region). On Linux, check `/dev/shm` permissions |
| whisper.cpp build fails on Jetson | Ensure cmake and build-essential are installed: `sudo apt install cmake build-essential` |
| Ollama not responding | Check it's running: `curl localhost:11434/api/tags`. Start with `ollama serve` |
| Voice assistant no audio | Verify ALSA devices: `arecord -l` (mic) and `aplay -l` (speaker). Adjust device paths in `voice_assistant.py` |

## License

MIT

Built by [@0xsoftboi](https://github.com/0xsoftboi)
