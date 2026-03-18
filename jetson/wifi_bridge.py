"""
SensorForge WiFi Bridge — Jetson TCP Server.

Receives sensor data from iPhone, serves robot/Qualia status,
and handles remote commands. Advertises via Avahi/mDNS.

Usage:
    python3 wifi_bridge.py                  # Run server
    python3 wifi_bridge.py --test           # Test without mDNS
    python3 wifi_bridge.py --port 9876      # Custom port
"""

import json
import logging
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

# Add parent dir to path for protocol imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from protocol.messages import (
    BRIDGE_PORT,
    PROTOCOL_VERSION,
    HandshakeAck,
    StatusResponse,
    QualiaStatus,
    QualiaLayerStatus,
    UGVStatus,
    CommandAck,
    ClockSync,
    SensorFrame,
    Command,
    encode_message,
    mono_ns,
)

log = logging.getLogger("wifi-bridge")

# ── mDNS Advertisement (Avahi on Linux, dns-sd on macOS) ─────────

_mdns_proc: Optional[subprocess.Popen] = None


def start_mdns(port: int):
    """Advertise _sensorforge._tcp via mDNS."""
    global _mdns_proc
    hostname = socket.gethostname()

    if sys.platform == "darwin":
        # macOS: use dns-sd
        _mdns_proc = subprocess.Popen(
            ["dns-sd", "-R", hostname, "_sensorforge._tcp", "local", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # Linux (Jetson): use avahi-publish
        _mdns_proc = subprocess.Popen(
            [
                "avahi-publish",
                "-s",
                hostname,
                "_sensorforge._tcp",
                str(port),
                f"version={PROTOCOL_VERSION}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    log.info("mDNS: advertising %s._sensorforge._tcp on port %d", hostname, port)


def stop_mdns():
    global _mdns_proc
    if _mdns_proc:
        _mdns_proc.terminate()
        _mdns_proc.wait(timeout=3)
        _mdns_proc = None


# ── Sensor Data Store ────────────────────────────────────────────

class SensorStore:
    """Thread-safe storage for the latest iPhone sensor data."""

    def __init__(self):
        self._lock = threading.Lock()
        self.latest_frame: Optional[dict] = None
        self.frame_count: int = 0
        self.connected_device: Optional[str] = None
        self.clock_offset_ns: int = 0

    def update(self, frame: dict):
        with self._lock:
            self.latest_frame = frame
            self.frame_count += 1

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            return self.latest_frame

    @property
    def is_connected(self) -> bool:
        return self.connected_device is not None


# Global store
sensor_store = SensorStore()


# ── Qualia Status Helper ─────────────────────────────────────────

def get_qualia_status() -> QualiaStatus:
    """Read Qualia status via the bridge module."""
    try:
        from qualia_bridge import get_bridge
        bridge = get_bridge()
        if bridge is None:
            return QualiaStatus(active=False)

        layers = bridge.read_all_layers()
        world = bridge.read_world_model()

        return QualiaStatus(
            active=True,
            layers=[
                QualiaLayerStatus(
                    id=s.layer_id,
                    vfe=round(s.vfe, 4),
                    compression=s.compression,
                    challenged=s.is_challenged,
                )
                for s in layers
            ],
            scene=world.scene[:200],
            directive=world.directive[:200],
        )
    except Exception:
        return QualiaStatus(active=False)


# ── Client Handler ───────────────────────────────────────────────

def handle_client(conn: socket.socket, addr: tuple):
    """Handle a single iPhone client connection."""
    log.info("Client connected: %s:%d", addr[0], addr[1])
    conn.settimeout(30.0)

    try:
        while True:
            # Read 4-byte length header
            header = _recv_exact(conn, 4)
            if header is None:
                break
            length = struct.unpack(">I", header)[0]
            if length > 1_000_000:
                log.warning("Message too large: %d bytes", length)
                break

            payload = _recv_exact(conn, length)
            if payload is None:
                break

            msg = json.loads(payload.decode("utf-8"))
            msg_type = msg.get("type", "")

            if msg_type == "handshake":
                _handle_handshake(conn, msg)
            elif msg_type == "sensor_frame":
                _handle_sensor_frame(msg)
            elif msg_type == "status_request":
                _handle_status_request(conn)
            elif msg_type == "command":
                _handle_command(conn, msg)
            elif msg_type == "clock_sync":
                _handle_clock_sync(conn, msg)
            else:
                log.warning("Unknown message type: %s", msg_type)

    except socket.timeout:
        log.info("Client %s:%d timed out", addr[0], addr[1])
    except (ConnectionResetError, BrokenPipeError):
        log.info("Client %s:%d disconnected", addr[0], addr[1])
    except json.JSONDecodeError as e:
        log.error("JSON decode error from %s: %s", addr[0], e)
    finally:
        sensor_store.connected_device = None
        conn.close()
        log.info("Client %s:%d closed", addr[0], addr[1])


def _handle_handshake(conn: socket.socket, msg: dict):
    device_name = msg.get("device_name", "unknown")
    device_id = msg.get("device_id", "")
    version = msg.get("protocol_version", 0)
    client_ts = msg.get("timestamp_ns", 0)

    log.info("Handshake from %s (id=%s, v=%d)", device_name, device_id[:8], version)

    sensor_store.connected_device = device_name

    # Estimate clock offset
    server_ts = mono_ns()
    clock_offset = server_ts - client_ts if client_ts else 0
    sensor_store.clock_offset_ns = clock_offset

    qualia_active = False
    try:
        from qualia_bridge import get_bridge
        qualia_active = get_bridge() is not None
    except Exception:
        pass

    ack = HandshakeAck(
        server_name=socket.gethostname(),
        qualia_active=qualia_active,
        clock_offset_ns=clock_offset,
    )
    _send_message(conn, ack.to_dict())


def _handle_sensor_frame(msg: dict):
    sensor_store.update(msg)
    if sensor_store.frame_count % 100 == 0:
        log.debug("Received %d sensor frames", sensor_store.frame_count)


def _handle_status_request(conn: socket.socket):
    qualia = get_qualia_status()
    ugv = UGVStatus(battery_v=0.0, moving=False)

    # Try to get real UGV status
    try:
        from ugv_driver import UGVDriver
        driver = UGVDriver()
        status = driver.read_status()
        if status:
            ugv.battery_v = status.get("battery_v", 0.0)
    except Exception:
        pass

    resp = StatusResponse(
        voice_state="idle",
        qualia=qualia,
        ugv=ugv,
    )
    _send_message(conn, resp.to_dict())


def _handle_command(conn: socket.socket, msg: dict):
    cmd = Command.from_dict(msg)
    log.info("Command: %s params=%s", cmd.action, cmd.params)

    result = "Unknown command"
    success = False

    try:
        if cmd.action == "set_directive":
            from qualia_bridge import get_bridge
            bridge = get_bridge()
            if bridge:
                bridge.write_directive(cmd.params.get("text", ""))
                result = "Directive set"
                success = True
            else:
                result = "Qualia not running"

        elif cmd.action in ("move_forward", "move_backward", "turn_left", "turn_right", "stop"):
            from voice_assistant import execute_tool
            tool_name = cmd.action if cmd.action != "stop" else "stop_robot"
            result = execute_tool(tool_name, cmd.params)
            success = True

        elif cmd.action == "speak":
            text = cmd.params.get("text", "")
            log.info("Remote speak: %s", text[:50])
            result = f"Speaking: {text[:50]}"
            success = True

    except Exception as e:
        result = str(e)

    ack = CommandAck(action=cmd.action, success=success, result=result)
    _send_message(conn, ack.to_dict())


def _handle_clock_sync(conn: socket.socket, msg: dict):
    sync = ClockSync(
        client_ns=msg.get("client_ns", 0),
        server_ns=mono_ns(),
    )
    _send_message(conn, sync.to_dict())


# ── Network helpers ──────────────────────────────────────────────

def _recv_exact(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _send_message(conn: socket.socket, msg: dict):
    data = encode_message(msg)
    conn.sendall(data)


# ── Server ───────────────────────────────────────────────────────

def run_server(port: int = BRIDGE_PORT, use_mdns: bool = True):
    """Run the WiFi bridge TCP server."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(2)

    log.info("WiFi bridge listening on 0.0.0.0:%d", port)

    if use_mdns:
        start_mdns(port)

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        stop_mdns()
        server.close()


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SensorForge WiFi Bridge")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT, help="TCP port")
    parser.add_argument("--test", action="store_true", help="Test mode (no mDNS)")
    args = parser.parse_args()

    run_server(port=args.port, use_mdns=not args.test)
