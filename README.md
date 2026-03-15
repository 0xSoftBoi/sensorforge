# SensorForge

**Turn any iPhone into a research-grade robotics data capture platform.**

One tap. Every sensor. Any robot. Robotics-native export.

## The Problem

Robotics researchers need synchronized multi-modal sensor data. Current options cost $5K+ in dedicated hardware, take days to configure, and still miss sensors like spatial audio and barometric pressure.

**Your iPhone already has everything:** 48MP camera, LiDAR, spatial audio (3 mics), 200Hz IMU, dual-frequency GPS, barometer, magnetometer — and a Neural Engine for on-device ML.

Nobody has built an app that captures ALL of these simultaneously, synchronized to one clock, and exports in formats robotics researchers actually use.

## What It Records (all simultaneous)

|Sensor                |Rate     |Source                               |
|----------------------|---------|-------------------------------------|
|4K RGB Video          |30fps    |ARKit                                |
|LiDAR Depth           |30fps    |ARKit sceneDepth                     |
|6-DoF Camera Pose     |60Hz     |ARKit VIO                            |
|3D Scene Mesh         |Real-time|ARKit                                |
|Surface Classification|Real-time|ARKit (floor/wall/ceiling/table/seat)|
|IMU (Accel + Gyro)    |100-200Hz|CoreMotion                           |
|Magnetometer          |50Hz     |CoreMotion                           |
|GPS (L1+L5)           |1Hz      |CoreLocation                         |
|Barometer             |1Hz      |CMAltimeter                          |
|Spatial Audio         |48kHz    |AVAudioEngine                        |
|Ambient Light         |30Hz     |ARKit                                |
|BLE Device Telemetry  |Varies   |CoreBluetooth                        |

Plus on-device YOLOv8 annotation and audio scene classification via CoreML.

## Mount It On Anything

SensorForge connects to external hardware via Bluetooth, WiFi, or SDK and logs their telemetry to the same synchronized clock:

**Flying Drones** — DJI Mini/Air/Mavic via Mobile SDK (GPS, gimbal, altitude, velocity)

**RC Cars** — Traxxas via BLE (speed, RPM, motor temp, battery voltage)

**DIY Robots** — Arduino/RPi/ESP32 via BLE or WiFi (encoders, sensors, motor commands)

**Cars** — OBD-II via BLE dongle (vehicle speed, RPM, throttle, engine data)

**Quadrupeds** — Unitree Go2 via WiFi SDK (joint states, foot contacts)

**Or just handheld** — walk, hike, bike with phone in hand or chest mount

## The “Phone Is Taped to a Drone” Problem

When your phone is strapped to a robot, how do you press Start?

- **⏱ Countdown Timer** — tap “Start in 30s”, mount phone, go (recommended)
- **🗣 “Hey Siri, SensorForge”** — voice command, works when screen is off
- **📳 Shake 3x** — shake phone to toggle recording
- **⌚ Apple Watch** — start/stop from your wrist (coming soon)

## Export Formats

- **MCAP** — Industry standard robotics logging (Foxglove compatible, default ROS2 bag format)
- **LeRobot v3.0** — HuggingFace robot learning datasets (Parquet + MP4)
- **Raw CSV + Media** — Universal. Open in pandas, Excel, MATLAB, anything
- **ROS2 Bag** — Standard ROS2 message types
- **HDF5** — Scientific computing format

## Why This Matters

- Robotics raised **$40.7B in 2025** (74% YoY increase)
- Data scarcity is the **#1 bottleneck** cited by Scale AI, NVIDIA, Epoch AI
- **Zero public datasets** exist with synchronized RGB + LiDAR + spatial audio + GPS + environmental from a mobile platform
- LeRobot v0.4 officially added phone support as a core feature
- Apple’s own EgoDex captured 829 hours via ARKit for manipulation research
- 3 YC W26 startups are building phone-based robotics data collection

## Architecture

```
┌────────────────────────────────────────────┐
│           Phone Capture Layer              │
│  iOS (ARKit, CoreMotion, CoreLocation,     │
│       AVAudioEngine, CoreML)               │
│  Android (ARCore, SensorManager, Camera2)  │
└──────────────────┬─────────────────────────┘
                   │
┌──────────────────┼─────────────────────────┐
│          Device Bridge Layer               │
│  BLE │ WiFi/WS │ DJI SDK │ MAVLink │ ROS2 │
└──────────────────┬─────────────────────────┘
                   │
┌──────────────────┼─────────────────────────┐
│          Export Pipeline                   │
│  MCAP │ LeRobot │ CSV │ ROS2 Bag │ HDF5   │
└────────────────────────────────────────────┘
```

## Roadmap

- [x] **v0.1** — All iPhone sensors → CSV, BLE bridge, countdown start, shake detection
- [ ] **v0.2** — MCAP native export, DJI SDK, HuggingFace upload, Apple Watch
- [ ] **v0.3** — Android app (Kotlin/ARCore), shared bridge protocols
- [ ] **v0.4** — MAVLink, ROS2 bridge, multi-device sync
- [ ] **v0.5** — On-device YOLOv8, audio classification, privacy controls
- [ ] **v1.0** — App Store + Play Store launch

## Contributing

PRs welcome. Priority needs:

1. **iOS devs** — Core capture pipeline, Siri Shortcuts, MCAP writer
1. **Android devs** — ARCore capture, Kotlin port of bridge layer
1. **ML engineers** — CoreML/TFLite model optimization, annotation pipeline
1. **Robot owners** — Test on your hardware, validate BLE bridge
1. **3D designers** — Phone mount STLs for common platforms

## Quick Start

```bash
git clone https://github.com/0xsoftboi/sensorforge.git
# Open in Xcode, build to iPhone 12 Pro or newer
# Grant permissions: Camera, Mic, Location, Bluetooth
# Tap the big red button. That's it.
```

## License

MIT

-----

Built by [@0xsoftboi](https://github.com/0xsoftboi)

*The best sensor rig is the one already in your pocket.*
