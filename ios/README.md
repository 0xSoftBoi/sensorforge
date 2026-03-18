# SensorForge iOS

iPhone sensor capture app that records 11 sensor streams simultaneously and streams decimated data to the Jetson over WiFi.

## Requirements

- **iPhone 12 Pro or newer** (LiDAR sensor required for depth capture)
- **iOS 17.0+**
- **Xcode 15+** on macOS
- Permissions: Camera, Microphone, Location (Always or When In Use), Bluetooth

## Setup & Build

```bash
open SensorForge.xcodeproj
```

1. Select your development team under Signing & Capabilities
2. Connect your iPhone and select it as the build target
3. Build and run (Cmd+R)
4. Grant all permission prompts on first launch

No external package dependencies — the app uses only Apple frameworks.

## Architecture

| File | Role |
|------|------|
| `SensorForgeApp.swift` | App entry point, SwiftUI lifecycle |
| `CaptureEngine.swift` | Orchestrates all 11 sensor streams, writes output files |
| `WiFiBridge.swift` | TCP connection to Jetson with Bonjour discovery |
| `BLEBridge.swift` | Bluetooth peripheral scanning and pairing |
| `BLEPairingView.swift` | UI for BLE device management |
| `MainCaptureView.swift` | Main recording UI |
| `SyncClock.swift` | Monotonic nanosecond clock for cross-device timestamp alignment |
| `HeadlessStart.swift` | Countdown starter, shake-to-record, Siri shortcuts, screen keep-awake |

## Sensors

| Sensor | Rate | Framework | Output |
|--------|------|-----------|--------|
| 4K RGB Video | 30fps | ARKit | `video.mp4` |
| LiDAR Depth | 30fps | ARKit sceneDepth | `depth/*.bin` (float16 per pixel) |
| 6-DoF Camera Pose | 60Hz | ARKit VIO | `poses.csv` |
| 3D Scene Mesh | Real-time | ARKit | `surfaces.jsonl` |
| Accelerometer + Gyroscope | 100-200Hz | CoreMotion | `imu.csv` |
| Magnetometer | 50Hz | CoreMotion | `magnetometer.csv` |
| GPS (L1+L5) | 1Hz | CoreLocation | `gps.csv` |
| Barometer | 1Hz | CMAltimeter | `barometer.csv` |
| Spatial Audio | 48kHz / 16-bit | AVAudioEngine | `audio.wav` |
| Ambient Light | 30Hz | ARKit lightEstimate | (sent via WiFi bridge) |
| BLE Telemetry | Varies | CoreBluetooth | (sent via WiFi bridge) |

## Session Output Format

Each recording session creates a timestamped directory:

```
SensorForge_20240115_143022/
├── session.json          # Metadata: device, start/end time, sensor config
├── video.mp4             # 4K RGB video
├── audio.wav             # Spatial audio
├── imu.csv               # timestamp_ns, ax, ay, az, gx, gy, gz
├── gps.csv               # timestamp_ns, lat, lon, alt, accuracy
├── magnetometer.csv      # timestamp_ns, mx, my, mz
├── barometer.csv         # timestamp_ns, pressure_hpa, relative_alt_m
├── poses.csv             # timestamp_ns, px, py, pz, qx, qy, qz, qw
├── surfaces.jsonl        # One JSON object per mesh update
└── depth/
    ├── 000000.bin        # Float16 depth map per frame
    ├── 000001.bin
    └── ...
```

All timestamps use monotonic nanoseconds from `SyncClock` for cross-device alignment.

## WiFi Bridge

The app discovers the Jetson on the local network via Bonjour and establishes a TCP connection for real-time sensor streaming.

- **Discovery**: mDNS service type `_sensorforge._tcp`, port 9876
- **Transport**: Length-prefixed JSON (4-byte big-endian length + UTF-8 payload)
- **Send rate**: ~10Hz (decimated from full sensor rates)

### Data sent to Jetson

- IMU (accel, gyro, magnetometer)
- 6-DoF camera pose (position + quaternion rotation)
- GPS coordinates
- Barometer (pressure + relative altitude)
- Ambient light level

### Data received from Jetson

- Voice assistant state (idle, listening, processing, speaking)
- Qualia layer status (VFE, compression, challenged flag per layer)
- Current scene description and directive
- UGV battery voltage

Connection state is published as `ConnectionState` (disconnected, browsing, connecting, connected) and displayed in the UI.

## BLE Bridge

The app scans for and pairs with Bluetooth Low Energy devices, relaying their telemetry to the Jetson.

### Supported device types

| Type | Detection | Examples |
|------|-----------|----------|
| Traxxas | Service UUID match | RC vehicle telemetry |
| Arduino | Name prefix "Arduino" / "HM-10" | Custom sensor boards |
| ESP32 | Name prefix "ESP32" / "ESP_" | IoT sensor nodes |
| OBD-II | Name contains "OBD" / "ELM" / "Viecar" | Vehicle diagnostics |
| Raspberry Pi | Name prefix "RPi" / "RaspberryPi" | Custom peripherals |

Paired devices show signal quality (RSSI-based bars), device type icon, and connection status.

## Headless Modes

The app supports hands-free operation:

- **Countdown start**: Configurable delay before recording begins (audio ticks + vibration)
- **Shake to toggle**: Shake the phone to start/stop recording (2-second cooldown)
- **Siri Shortcuts**: "Start SensorForge" / "Stop SensorForge" voice commands
- **Screen keep-awake**: Prevents display from dimming during recording
