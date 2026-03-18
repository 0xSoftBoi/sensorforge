"""
SensorForge WiFi Bridge — Message types.

Length-prefixed JSON over TCP. Each message is:
  [4 bytes big-endian length][UTF-8 JSON payload]

All timestamps are monotonic nanoseconds from the sending device.
"""

import json
import struct
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


PROTOCOL_VERSION = 1
BRIDGE_PORT = 9876
MDNS_SERVICE_TYPE = "_sensorforge._tcp.local."

# ── Framing ──────────────────────────────────────────────────────────

def encode_message(msg: dict) -> bytes:
    """Encode a message dict as length-prefixed JSON bytes."""
    payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def decode_message(data: bytes) -> dict:
    """Decode a JSON payload (without length prefix)."""
    return json.loads(data.decode("utf-8"))


def read_message(reader) -> Optional[dict]:
    """Read one length-prefixed message from a stream reader.
    Returns None on EOF or error."""
    header = _read_exact(reader, 4)
    if header is None:
        return None
    length = struct.unpack(">I", header)[0]
    if length > 1_000_000:  # 1MB sanity limit
        return None
    payload = _read_exact(reader, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _read_exact(reader, n: int) -> Optional[bytes]:
    """Read exactly n bytes from a file-like object."""
    buf = bytearray()
    while len(buf) < n:
        chunk = reader.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ── Monotonic nanosecond clock ───────────────────────────────────────

def mono_ns() -> int:
    """Monotonic nanoseconds (matches SyncClock.swift)."""
    return time.monotonic_ns()


# ── Message dataclasses ─────────────────────────────────────────────

@dataclass
class Handshake:
    device_name: str
    device_id: str
    sensors_available: list = field(default_factory=list)
    protocol_version: int = PROTOCOL_VERSION
    timestamp_ns: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = "handshake"
        if not d["timestamp_ns"]:
            d["timestamp_ns"] = mono_ns()
        return d


@dataclass
class HandshakeAck:
    server_name: str
    qualia_active: bool = False
    clock_offset_ns: int = 0
    timestamp_ns: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = "handshake_ack"
        if not d["timestamp_ns"]:
            d["timestamp_ns"] = mono_ns()
        return d


@dataclass
class IMUData:
    accel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gyro: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mag: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class PoseData:
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    tracking_state: str = "normal"


@dataclass
class GPSData:
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    accuracy: float = 0.0


@dataclass
class BarometerData:
    pressure_hpa: float = 0.0
    relative_altitude_m: float = 0.0


@dataclass
class SensorFrame:
    seq: int = 0
    timestamp_ns: int = 0
    imu: IMUData = field(default_factory=IMUData)
    pose: PoseData = field(default_factory=PoseData)
    gps: GPSData = field(default_factory=GPSData)
    barometer: BarometerData = field(default_factory=BarometerData)
    ambient_light: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = "sensor_frame"
        if not d["timestamp_ns"]:
            d["timestamp_ns"] = mono_ns()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SensorFrame":
        return cls(
            seq=d.get("seq", 0),
            timestamp_ns=d.get("timestamp_ns", 0),
            imu=IMUData(**d.get("imu", {})),
            pose=PoseData(**d.get("pose", {})),
            gps=GPSData(**d.get("gps", {})),
            barometer=BarometerData(**d.get("barometer", {})),
            ambient_light=d.get("ambient_light", 0.0),
        )


@dataclass
class QualiaLayerStatus:
    id: int = 0
    vfe: float = 0.0
    compression: int = 0
    challenged: bool = False


@dataclass
class QualiaStatus:
    active: bool = False
    layers: list = field(default_factory=list)
    scene: str = ""
    directive: str = ""


@dataclass
class UGVStatus:
    battery_v: float = 0.0
    moving: bool = False


@dataclass
class StatusResponse:
    voice_state: str = "idle"
    qualia: QualiaStatus = field(default_factory=QualiaStatus)
    ugv: UGVStatus = field(default_factory=UGVStatus)
    timestamp_ns: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = "status_response"
        if not d["timestamp_ns"]:
            d["timestamp_ns"] = mono_ns()
        return d


@dataclass
class Command:
    action: str = ""
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = "command"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Command":
        return cls(action=d.get("action", ""), params=d.get("params", {}))


@dataclass
class CommandAck:
    action: str = ""
    success: bool = True
    result: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = "command_ack"
        return d


@dataclass
class ClockSync:
    client_ns: int = 0
    server_ns: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = "clock_sync"
        return d
