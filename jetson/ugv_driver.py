"""
Waveshare UGV Serial Driver — JSON commands to ESP32 lower controller.

Protocol: Send JSON lines with a "T" type field over serial at 115200 baud.
  T:11  — direct PWM control: {"T":11,"L":128,"R":128}  (int, -255 to +255)
  T:1   — closed-loop speed:  {"T":1,"L":0.3,"R":0.3}   (float, -0.5 to +0.5 m/s)

The ESP32 has a ~3s heartbeat timeout — if no command arrives, motors stop.
This driver maintains a keepalive thread to prevent unexpected stops.

Speed range (T:11 PWM mode): -255 (full reverse) to 255 (full forward) per motor.
Serial port: /dev/ttyTHS1 (Tegra hardware UART to ESP32).
"""

import json
import logging
import threading
import time

log = logging.getLogger(__name__)

# Safety limits
MAX_SPEED = 128          # Half of 255 max — safe for indoor use
MAX_DURATION = 5.0       # Max seconds per single movement command
HEARTBEAT_INTERVAL = 2.0 # Seconds between keepalive commands


class UGVDriver:
    """Waveshare UGV serial driver — JSON motor commands to ESP32."""

    def __init__(self, port="/dev/ttyTHS1", baud=115200):
        import serial
        self.serial = serial.Serial(port, baud, timeout=1)
        time.sleep(0.5)  # ESP32 reset delay
        self._drain_rx()

        self._running = True
        self._current_cmd = {"T": 11, "L": 0, "R": 0}  # Heartbeat sends this instead of hardcoded stop
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, daemon=True,
        )
        self._heartbeat.start()
        self._lock = threading.Lock()
        log.info(f"UGV driver connected: {port}@{baud}")

    def _drain_rx(self):
        """Read and discard any buffered data from ESP32."""
        while self.serial.in_waiting:
            self.serial.readline()

    def _heartbeat_loop(self):
        """Resend current motor command every 2s to keep ESP32 alive (it auto-stops after ~3s)."""
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            if self._running:
                try:
                    self._send_raw(self._current_cmd)
                except Exception as e:
                    log.warning(f"Heartbeat send failed: {e}")

    def _send_raw(self, cmd):
        """Send a JSON command to the ESP32."""
        with self._lock:
            line = json.dumps(cmd) + "\n"
            self.serial.write(line.encode())
            self.serial.flush()
            log.debug(f"TX: {line.strip()}")

    def _read_response(self, timeout=0.5):
        """Read a line from ESP32 (if any)."""
        self.serial.timeout = timeout
        try:
            line = self.serial.readline().decode().strip()
            if line:
                return json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return None

    # ─── Movement Primitives ────────────────────────────────────

    def move(self, left_speed, right_speed):
        """Set motor speeds. Range: -255 to 255 per side."""
        left_speed = max(-MAX_SPEED, min(MAX_SPEED, int(left_speed)))
        right_speed = max(-MAX_SPEED, min(MAX_SPEED, int(right_speed)))
        cmd = {"T": 11, "L": left_speed, "R": right_speed}
        self._current_cmd = cmd
        self._send_raw(cmd)

    def forward(self, speed=100, duration=1.0):
        """Drive forward for duration seconds."""
        speed = min(abs(speed), MAX_SPEED)
        duration = min(duration, MAX_DURATION)
        self.move(speed, speed)
        time.sleep(duration)
        self.stop()

    def backward(self, speed=100, duration=1.0):
        """Drive backward for duration seconds."""
        speed = min(abs(speed), MAX_SPEED)
        duration = min(duration, MAX_DURATION)
        self.move(-speed, -speed)
        time.sleep(duration)
        self.stop()

    def turn_left(self, speed=80, duration=0.5):
        """Turn left in place (right motor forward, left motor backward)."""
        speed = min(abs(speed), MAX_SPEED)
        duration = min(duration, MAX_DURATION)
        self.move(-speed, speed)
        time.sleep(duration)
        self.stop()

    def turn_right(self, speed=80, duration=0.5):
        """Turn right in place (left motor forward, right motor backward)."""
        speed = min(abs(speed), MAX_SPEED)
        duration = min(duration, MAX_DURATION)
        self.move(speed, -speed)
        time.sleep(duration)
        self.stop()

    def stop(self):
        """Immediate stop."""
        cmd = {"T": 11, "L": 0, "R": 0}
        self._current_cmd = cmd
        self._send_raw(cmd)

    # ─── Sensor Reads ───────────────────────────────────────────

    def get_imu(self):
        """Request IMU data from ESP32 (if supported)."""
        self._send_raw({"cmd": "imu"})
        return self._read_response(timeout=1.0)

    def get_battery(self):
        """Request battery voltage from ESP32 (if supported)."""
        self._send_raw({"cmd": "battery"})
        return self._read_response(timeout=1.0)

    def get_status(self):
        """Get combined robot status string."""
        parts = []
        batt = self.get_battery()
        if batt and isinstance(batt, dict):
            voltage = batt.get("voltage", batt.get("v"))
            if voltage:
                parts.append(f"Battery: {voltage}V")
        imu = self.get_imu()
        if imu and isinstance(imu, dict):
            parts.append(f"IMU: {imu}")
        return ". ".join(parts) if parts else "Robot connected (no sensor data available)"

    # ─── Lifecycle ──────────────────────────────────────────────

    def close(self):
        """Shutdown: stop motors and close serial."""
        self._running = False
        try:
            self.stop()
            self.serial.close()
        except Exception:
            pass
        log.info("UGV driver closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read_telemetry(self, timeout=0.5):
        """Read T:1001 telemetry response from ESP32.

        Returns dict with keys like: T, L, R, ax, ay, az, gx, gy, gz, v (battery mV),
        odl, odr (odometry ticks), or None if no data.
        """
        resp = self._read_response(timeout=timeout)
        if resp and isinstance(resp, dict) and resp.get("T") == 1001:
            return resp
        return resp

    @staticmethod
    def probe(port="/dev/ttyTHS1", baud=115200):
        """Test serial connection. Returns (success, info_string)."""
        try:
            import serial
        except ImportError:
            return False, "pyserial not installed (pip install pyserial)"
        try:
            s = serial.Serial(port, baud, timeout=2)
            time.sleep(0.5)
            # Drain startup messages
            startup = []
            while s.in_waiting:
                line = s.readline().decode(errors="replace").strip()
                if line:
                    startup.append(line)
            # Send stop command (safe)
            s.write(json.dumps({"T": 11, "L": 0, "R": 0}).encode() + b"\n")
            time.sleep(0.5)
            # Read response
            responses = []
            while s.in_waiting:
                line = s.readline().decode(errors="replace").strip()
                if line:
                    responses.append(line)
            s.close()
            info = f"Port: {port}@{baud}"
            if startup:
                info += f" | Startup: {startup}"
            if responses:
                info += f" | Response: {responses}"
            return True, info
        except Exception as e:
            return False, f"Failed: {e}"
