#!/usr/bin/env python3
"""
Manual Keyboard Drive — teleop with live VFE display.

Drive the robot with WASD/arrows while viewing real-time Qualia layer beliefs.
Session recorder picks up motor state automatically for training data.

Controls:
    W / Up      Forward
    S / Down    Backward
    A / Left    Turn left
    D / Right   Turn right
    Space       Emergency stop
    +/-         Adjust speed (30-128)
    Q / Esc     Quit

Usage:
    python3 manual_drive.py [--port /dev/ttyACM0] [--speed 60]

    # SSH with terminal allocation for keyboard input:
    ssh -t jetson 'cd ~/sensorforge && python3 jetson/manual_drive.py'
"""

import json
import logging
import os
import signal
import sys
import termios
import time
import tty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ugv_driver import UGVDriver, MAX_SPEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s manual: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MOTOR_STATE_FILE = "/tmp/qualia_motor_state.json"
MIN_SPEED = 30
SPEED_STEP = 10

# Key codes
KEY_W = ord("w")
KEY_A = ord("a")
KEY_S = ord("s")
KEY_D = ord("d")
KEY_Q = ord("q")
KEY_SPACE = ord(" ")
KEY_PLUS = ord("+")
KEY_EQUALS = ord("=")  # unshifted +
KEY_MINUS = ord("-")
KEY_ESC = 27
KEY_UP = 65     # after ESC [ sequence
KEY_DOWN = 66
KEY_RIGHT = 67
KEY_LEFT = 68


def acquire_singleton(name):
    """Ensure only one drive instance runs."""
    pidfile = f"/tmp/qualia_{name}.pid"
    try:
        with open(pidfile) as f:
            old_pid = int(f.read().strip())
        os.kill(old_pid, 0)
        log.warning(f"Killing stale {name} process (PID {old_pid})")
        os.kill(old_pid, signal.SIGTERM)
        time.sleep(1)
        try:
            os.kill(old_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        pass
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))


def write_motor_state(left: int, right: int):
    try:
        state = {"motor_left": left, "motor_right": right, "ts": time.time()}
        tmp = MOTOR_STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, MOTOR_STATE_FILE)
    except OSError:
        pass


def read_qualia_vfe():
    """Read VFE from qualia_bridge if available. Returns list of (layer, vfe, compression) or None."""
    try:
        from qualia_bridge import QualiaBridge, NUM_LAYERS
        bridge = read_qualia_vfe._bridge
        if bridge is None or not bridge.is_open:
            bridge = QualiaBridge()
            if not bridge.open():
                return None
            read_qualia_vfe._bridge = bridge
        result = []
        for i in range(NUM_LAYERS):
            b = bridge.read_layer_belief(i)
            result.append((i, b.vfe, b.compression))
        return result
    except Exception:
        return None

read_qualia_vfe._bridge = None


def get_key(fd):
    """Read a single keypress, handling arrow key escape sequences."""
    ch = os.read(fd, 1)
    if not ch:
        return None
    b = ch[0]
    if b == KEY_ESC:
        # Could be arrow key: ESC [ A/B/C/D
        ch2 = os.read(fd, 1) if select_ready(fd, 0.05) else b""
        if ch2 and ch2[0] == ord("["):
            ch3 = os.read(fd, 1) if select_ready(fd, 0.05) else b""
            if ch3:
                return ("arrow", ch3[0])
        return ("key", KEY_ESC)
    return ("key", b)


def select_ready(fd, timeout):
    """Check if fd has data ready within timeout."""
    import select
    r, _, _ = select.select([fd], [], [], timeout)
    return bool(r)


def format_direction(left, right):
    """Human-readable direction from motor speeds."""
    if left == 0 and right == 0:
        return "STOPPED"
    if left > 0 and right > 0:
        return "FORWARD"
    if left < 0 and right < 0:
        return "REVERSE"
    if left < 0 and right > 0:
        return "LEFT"
    if left > 0 and right < 0:
        return "RIGHT"
    return f"L={left} R={right}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manual keyboard drive with live VFE display")
    parser.add_argument("--port", default="/dev/ttyACM0", help="UGV serial port")
    parser.add_argument("--speed", type=int, default=60, help="Initial speed (30-128)")
    args = parser.parse_args()

    speed = max(MIN_SPEED, min(MAX_SPEED, args.speed))

    # Singleton lock (conflicts with explorer)
    acquire_singleton("explorer")

    # Connect UGV
    try:
        ugv = UGVDriver(port=args.port)
    except Exception as e:
        print(f"ERROR: Cannot connect to UGV at {args.port}: {e}")
        sys.exit(1)

    # Save/restore terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    motor_left = 0
    motor_right = 0
    running = True

    def cleanup():
        nonlocal running
        running = False
        ugv.stop()
        write_motor_state(0, 0)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        ugv.close()
        # Clear singleton
        try:
            os.remove("/tmp/qualia_explorer.pid")
        except OSError:
            pass

    def handle_signal(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Switch to raw terminal mode
    tty.setraw(fd)

    def set_motors(left, right):
        nonlocal motor_left, motor_right
        motor_left = left
        motor_right = right
        ugv.move(left, right)
        write_motor_state(left, right)

    def stop():
        set_motors(0, 0)

    def refresh_display():
        """Redraw the status line."""
        direction = format_direction(motor_left, motor_right)
        line = f"\r\x1b[K  Speed: {speed:3d}  |  {direction:8s}  |  L={motor_left:+4d} R={motor_right:+4d}"

        # Append VFE info if available
        vfe_data = read_qualia_vfe()
        if vfe_data:
            vfes = [f"L{i}:{v:.4f}" for i, v, _ in vfe_data[:3]]
            comp = [f"{c}" for _, _, c in vfe_data[:3]]
            line += f"  |  VFE: {' '.join(vfes)}  |  Comp: {'/'.join(comp)}"

        sys.stdout.write(line)
        sys.stdout.flush()

    # Print banner (raw mode: \r\n for newlines)
    banner = [
        "",
        "=== Manual Drive Mode ===",
        "  W/Up=fwd  S/Down=rev  A/Left=left  D/Right=right",
        "  Space=stop  +/-=speed  Q/Esc=quit",
        "",
    ]
    for line in banner:
        sys.stdout.write(line + "\r\n")
    sys.stdout.flush()

    try:
        while running:
            if not select_ready(fd, 0.1):
                # No key pressed — if motors are moving, keep refreshing display
                refresh_display()
                continue

            event = get_key(fd)
            if event is None:
                continue

            kind, val = event

            if kind == "key":
                v = val
                if v == KEY_W:
                    set_motors(speed, speed)
                elif v == KEY_S:
                    set_motors(-speed, -speed)
                elif v == KEY_A:
                    set_motors(-speed, speed)
                elif v == KEY_D:
                    set_motors(speed, -speed)
                elif v == KEY_SPACE:
                    stop()
                elif v in (KEY_PLUS, KEY_EQUALS):
                    speed = min(MAX_SPEED, speed + SPEED_STEP)
                elif v == KEY_MINUS:
                    speed = max(MIN_SPEED, speed - SPEED_STEP)
                elif v == KEY_Q or v == KEY_ESC:
                    break
                else:
                    continue

            elif kind == "arrow":
                if val == KEY_UP:
                    set_motors(speed, speed)
                elif val == KEY_DOWN:
                    set_motors(-speed, -speed)
                elif val == KEY_LEFT:
                    set_motors(-speed, speed)
                elif val == KEY_RIGHT:
                    set_motors(speed, -speed)

            refresh_display()

    finally:
        cleanup()
        print("\r\n\r\nManual drive stopped.\r\n")


if __name__ == "__main__":
    main()
