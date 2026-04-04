"""
SensorForge WiFi Bridge — Jetson TCP Server.

Receives sensor data from iPhone, serves robot/Qualia status,
and handles remote commands. Advertises via Avahi/mDNS.

Usage:
    python3 wifi_bridge.py                  # Run server
    python3 wifi_bridge.py --test           # Test without mDNS
    python3 wifi_bridge.py --port 9876      # Custom port
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
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

# ── HMAC Authentication ──────────────────────────────────────────

_SECRET = os.environ.get("SENSORFORGE_SECRET", "").encode()
if not _SECRET:
    log.critical(
        "SENSORFORGE_SECRET is not set — running in dev mode with NO authentication!"
    )

# Tracks which socket file descriptors have completed the HMAC handshake.
_authenticated_sockets: set = set()
_auth_lock = threading.Lock()

# ── Lazy UGV Driver Singleton ────────────────────────────────────

_ugv_driver = None
_ugv_driver_attempted = False


def _get_ugv_driver():
    global _ugv_driver, _ugv_driver_attempted
    if _ugv_driver_attempted:
        return _ugv_driver
    _ugv_driver_attempted = True
    try:
        from ugv_driver import UGVDriver
        _ugv_driver = UGVDriver()
        log.info("UGV driver initialized")
    except Exception as e:
        log.warning("UGV driver unavailable: %s", e)
    return _ugv_driver


def _read_ugv_status() -> "UGVStatus":
    """Read real UGV telemetry, falling back to zeros."""
    ugv = UGVStatus(battery_v=0.0, moving=False)
    driver = _get_ugv_driver()
    if driver:
        try:
            v = driver.get_battery_voltage()
            if v is not None:
                ugv.battery_v = round(v, 2)
        except Exception:
            pass
        try:
            left, right = driver.get_wheel_speeds()
            ugv.wheel_speed_left = round(left, 3)
            ugv.wheel_speed_right = round(right, 3)
            ugv.moving = abs(left) > 0.01 or abs(right) > 0.01
        except Exception:
            pass
        try:
            imu = driver.get_imu_cached()
            if imu and "yaw" in imu:
                ugv.heading_deg = round(imu["yaw"], 1)
        except Exception:
            pass
    return ugv


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
    sock_fd = conn.fileno()

    try:
        # ── HMAC challenge-response ──────────────────────────────
        if _SECRET:
            nonce = secrets.token_bytes(32)
            _send_message(conn, {"type": "challenge", "nonce": nonce.hex()})

            # Read the auth response (first message must be auth)
            header = _recv_exact(conn, 4)
            if header is None:
                log.warning("Auth: client %s disconnected before auth", addr[0])
                return
            length = struct.unpack(">I", header)[0]
            if length > 1_000_000:
                log.warning("Auth: oversized message from %s", addr[0])
                return
            payload = _recv_exact(conn, length)
            if payload is None:
                return
            auth_msg = json.loads(payload.decode("utf-8"))
            if auth_msg.get("type") != "auth":
                log.warning("Auth: expected auth message from %s, got %s", addr[0], auth_msg.get("type"))
                _send_message(conn, {"type": "auth_fail", "reason": "expected auth message"})
                return
            expected_hmac = hmac.new(_SECRET, nonce, hashlib.sha256).hexdigest()
            provided_hmac = auth_msg.get("hmac", "")
            if not hmac.compare_digest(expected_hmac, provided_hmac):
                log.warning("Auth: HMAC mismatch from %s — rejecting", addr[0])
                _send_message(conn, {"type": "auth_fail", "reason": "invalid hmac"})
                return
            with _auth_lock:
                _authenticated_sockets.add(sock_fd)
            log.info("Auth: client %s:%d authenticated", addr[0], addr[1])
            _send_message(conn, {"type": "auth_ok"})
        else:
            # Dev mode: no secret set, skip auth
            with _auth_lock:
                _authenticated_sockets.add(sock_fd)

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

            # Reject unauthenticated sockets (belt-and-suspenders)
            with _auth_lock:
                is_auth = sock_fd in _authenticated_sockets
            if not is_auth:
                log.warning("Rejected message from unauthenticated socket %s", addr[0])
                break

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
        with _auth_lock:
            _authenticated_sockets.discard(sock_fd)
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
    ugv = _read_ugv_status()

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

    if cmd.action == "robot_status":
        _handle_status_request(conn)
        return

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


# ── HTTP Status Server (for web dashboard) ──────────────────────

HTTP_PORT = BRIDGE_PORT + 1  # 9877


def run_http_status_server(port: int = HTTP_PORT):
    """Lightweight HTTP server for the web dashboard to fetch robot telemetry."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    def _cors_origin(origin: str) -> str:
        """Return the origin if it is localhost, otherwise empty string."""
        import re
        if origin and re.match(r'^http://localhost(:\d+)?$', origin):
            return origin
        return ""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/status":
                qualia = get_qualia_status()
                frame = sensor_store.get_latest()

                ugv = _read_ugv_status()
                status = {
                    "iphone": {
                        "connected": sensor_store.is_connected,
                        "device": sensor_store.connected_device,
                        "frames": sensor_store.frame_count,
                        "latest": frame,
                    },
                    "qualia": {
                        "active": qualia.active,
                        "layers": [
                            {"id": l.id, "vfe": l.vfe, "compression": l.compression, "challenged": l.challenged}
                            for l in qualia.layers
                        ],
                        "scene": qualia.scene,
                        "directive": qualia.directive,
                    },
                    "voice_state": "idle",
                    "ugv": {
                        "battery_v": ugv.battery_v,
                        "moving": ugv.moving,
                        "wheel_speed_left": ugv.wheel_speed_left,
                        "wheel_speed_right": ugv.wheel_speed_right,
                        "heading_deg": ugv.heading_deg,
                    },
                }

                body = json.dumps(status).encode()
                origin = self.headers.get("Origin", "")
                allowed = _cors_origin(origin)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if allowed:
                    self.send_header("Access-Control-Allow-Origin", allowed)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_OPTIONS(self):
            origin = self.headers.get("Origin", "")
            allowed = _cors_origin(origin)
            self.send_response(200)
            if allowed:
                self.send_header("Access-Control-Allow-Origin", allowed)
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress access logs

    httpd = HTTPServer(("0.0.0.0", port), Handler)
    log.info("HTTP status server on http://0.0.0.0:%d/status", port)
    httpd.serve_forever()


# ── Server ───────────────────────────────────────────────────────

def run_server(port: int = BRIDGE_PORT, use_mdns: bool = True):
    """Run the WiFi bridge TCP server + HTTP status server."""
    # Start HTTP status server in background thread
    http_thread = threading.Thread(
        target=run_http_status_server, args=(port + 1,), daemon=True
    )
    http_thread.start()

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
