# SensorForge WiFi Bridge Protocol

Communication protocol between the iPhone (SensorForge app) and the Jetson (voice assistant + Qualia engine).

## Transport

- **TCP** over WiFi on **port 9876**
- **Framing**: 4-byte big-endian length prefix + UTF-8 JSON payload
- **Max message size**: 1 MB (enforced by receiver)

```
┌──────────────┬──────────────────────────────────┐
│ 4 bytes      │ N bytes                          │
│ big-endian   │ UTF-8 JSON                       │
│ length (N)   │ payload                          │
└──────────────┴──────────────────────────────────┘
```

## Discovery

The Jetson advertises itself via **mDNS/Bonjour**:

- Service type: `_sensorforge._tcp.local.`
- Port: `9876`

The iPhone app browses for this service using `NWBrowser` and connects automatically when found. Both devices must be on the same WiFi network.

## Connection Flow

```
iPhone                              Jetson
  │                                    │
  │──── Browse _sensorforge._tcp ────►│
  │◄─── Service found ────────────────│
  │                                    │
  │──── TCP connect :9876 ───────────►│
  │                                    │
  │──── Handshake ───────────────────►│
  │◄─── HandshakeAck ────────────────│
  │                                    │
  │──── ClockSync ───────────────────►│
  │◄─── ClockSync (with server_ns) ──│
  │                                    │
  │──── SensorFrame (~10Hz) ─────────►│
  │──── SensorFrame ─────────────────►│
  │      ...                           │
  │                                    │
  │──── StatusRequest ───────────────►│
  │◄─── StatusResponse ──────────────│
  │                                    │
  │──── Command ─────────────────────►│
  │◄─── CommandAck ──────────────────│
```

## Message Types

### `handshake` (iPhone → Jetson)

Initial connection message identifying the device.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"handshake"` |
| `device_name` | string | iPhone model name |
| `device_id` | string | UUID for this device |
| `protocol_version` | int | Must be `1` |
| `sensors_available` | string[] | List of available sensor types |
| `timestamp_ns` | int | Device monotonic nanoseconds |

### `handshake_ack` (Jetson → iPhone)

Acknowledges the connection and reports server capabilities.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"handshake_ack"` |
| `server_name` | string | Jetson hostname |
| `qualia_active` | bool | Whether the Qualia engine is running |
| `clock_offset_ns` | int | Estimated clock offset (server - client) |
| `timestamp_ns` | int | Server monotonic nanoseconds |

### `sensor_frame` (iPhone → Jetson)

Decimated sensor data sent at ~10Hz.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"sensor_frame"` |
| `seq` | int | Frame sequence number |
| `timestamp_ns` | int | Device monotonic nanoseconds |
| `imu.accel` | float[3] | Accelerometer [x, y, z] m/s^2 |
| `imu.gyro` | float[3] | Gyroscope [x, y, z] rad/s |
| `imu.mag` | float[3] | Magnetometer [x, y, z] uT |
| `pose.position` | float[3] | Camera position [x, y, z] meters |
| `pose.rotation` | float[4] | Camera rotation quaternion [x, y, z, w] |
| `pose.tracking_state` | string | `"normal"`, `"limited"`, or `"notAvailable"` |
| `gps.lat` | float | Latitude (degrees) |
| `gps.lon` | float | Longitude (degrees) |
| `gps.alt` | float | Altitude (meters) |
| `gps.accuracy` | float | Horizontal accuracy (meters) |
| `barometer.pressure_hpa` | float | Atmospheric pressure (hPa) |
| `barometer.relative_altitude_m` | float | Relative altitude change (meters) |
| `ambient_light` | float | Ambient light intensity (lux) |

### `status_request` (iPhone → Jetson)

Request current robot and Qualia status. No fields beyond `type`.

### `status_response` (Jetson → iPhone)

Current state of the voice assistant, Qualia engine, and UGV.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"status_response"` |
| `timestamp_ns` | int | Server monotonic nanoseconds |
| `voice_state` | string | `"idle"`, `"listening"`, `"processing"`, or `"speaking"` |
| `qualia.active` | bool | Whether Qualia is running |
| `qualia.layers[]` | array | Per-layer status objects |
| `qualia.layers[].id` | int | Layer number (0-6) |
| `qualia.layers[].vfe` | float | Variational free energy (prediction error) |
| `qualia.layers[].compression` | int | Compression level (habit formation) |
| `qualia.layers[].challenged` | bool | Whether the layer is currently challenged |
| `qualia.scene` | string | Current scene description from Gemini |
| `qualia.directive` | string | Current top-level directive |
| `ugv.battery_v` | float | UGV battery voltage |
| `ugv.moving` | bool | Whether the UGV is currently moving |

### `command` (iPhone → Jetson)

Send a command to the robot.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"command"` |
| `action` | string | One of: `set_directive`, `move_forward`, `move_backward`, `turn_left`, `turn_right`, `stop`, `speak` |
| `params` | object | Action-specific parameters |

### `command_ack` (Jetson → iPhone)

Acknowledges command execution.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"command_ack"` |
| `action` | string | Echoed from the command |
| `success` | bool | Whether the command succeeded |
| `result` | string | Result or error message |

### `clock_sync` (bidirectional)

Timestamp alignment between devices. The iPhone sends with `client_ns` set; the Jetson responds with both `client_ns` (echoed) and `server_ns` filled in.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"clock_sync"` |
| `client_ns` | int | Client monotonic nanoseconds when sent |
| `server_ns` | int | Server monotonic nanoseconds when received (response only) |

## Implementation

- **Schema**: [`schema.json`](schema.json) — Machine-readable protocol definition
- **Python**: [`messages.py`](messages.py) — Dataclass implementations with `encode_message()` / `decode_message()` / `read_message()` helpers
- **Swift**: `ios/SensorForge/WiFiBridge.swift` — NWConnection-based client with Bonjour discovery

### Python usage

```python
from protocol.messages import (
    SensorFrame, StatusResponse, Command,
    encode_message, read_message
)

# Send a message
frame = SensorFrame(seq=1)
raw = encode_message(frame.to_dict())
conn.sendall(raw)

# Read a message
msg = read_message(conn.makefile("rb"))
if msg and msg["type"] == "sensor_frame":
    frame = SensorFrame.from_dict(msg)
```
