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

## Components

| Directory | Stack | Description |
|-----------|-------|-------------|
| `ios/` | Swift / ARKit | iPhone sensor capture (11+ sensors), BLE bridge, LeRobot export |
| `jetson/` | Python | Voice assistant with wake word, STT, LLM, TTS, 24 tools, UGV control |
| `qualia/` | Rust + CUDA/Metal | Active inference engine — 7-layer belief hierarchy, SHM IPC, web dashboard |
| `scripts/` | Python | Data conversion tools (LeRobot format) |
| `protocol/` | JSON + Python | WiFi bridge message schemas |

## Quick Start

### iOS (SensorForge)
```bash
open ios/SensorForge.xcodeproj
# Build and run on iPhone 12 Pro or newer
# Grant permissions: Camera, Mic, Location, Bluetooth
```

### Jetson Voice Assistant
```bash
cd jetson && bash setup.sh
python3 voice_assistant.py              # Run assistant
python3 voice_assistant.py --test-tools # Test all tools
```

### Qualia Engine (Mac — Metal)
```bash
cd qualia && cargo build --release
cargo run --release -p qualia-watch     # TUI supervisor
```

### Qualia Engine (Jetson — CUDA)
```bash
cd qualia && cargo build --release --features cuda
cargo run --release -p qualia-watch     # TUI supervisor
```

### Web Dashboard
```bash
cargo run --release -p qualia-agent     # http://localhost:8080
```

## Sensors Captured (iOS)

| Sensor | Rate | Source |
|--------|------|--------|
| 4K RGB Video | 30fps | ARKit |
| LiDAR Depth | 30fps | ARKit sceneDepth |
| 6-DoF Camera Pose | 60Hz | ARKit VIO |
| 3D Scene Mesh | Real-time | ARKit |
| IMU (Accel + Gyro) | 100-200Hz | CoreMotion |
| Magnetometer | 50Hz | CoreMotion |
| GPS (L1+L5) | 1Hz | CoreLocation |
| Barometer | 1Hz | CMAltimeter |
| Spatial Audio | 48kHz | AVAudioEngine |
| Ambient Light | 30Hz | ARKit |
| BLE Device Telemetry | Varies | CoreBluetooth |

## IPC via Shared Memory

Qualia uses POSIX shared memory (`/dev/shm/qualia_body`, 64 MiB) for zero-copy IPC between all Jetson processes. The Python voice assistant reads Qualia beliefs via `mmap` + `struct.unpack`, matching the `repr(C)` layout defined in `qualia/crates/types/src/lib.rs`.

## LeRobot Export

```bash
python3 scripts/convert_to_lerobot.py path/to/session output/
```

## License

MIT

Built by [@0xsoftboi](https://github.com/0xsoftboi)
