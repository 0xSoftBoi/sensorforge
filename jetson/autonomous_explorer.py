#!/usr/bin/env python3
"""
Autonomous Explorer — curiosity-driven robot exploration using Qualia VFE.

Reads layer VFE/compression from Qualia SHM and drives the robot:
  - Move forward when VFE is low (boring) → seek novelty
  - Pause when VFE spikes (something new) → let layers learn
  - Turn randomly when VFE stays high → try new direction
  - Emergency stop if VFE exceeds safety threshold

Writes motor state to /tmp/qualia_motor_state.json for session recorder.

Usage:
    python3 autonomous_explorer.py [--port /dev/ttyACM0] [--speed 60]
"""

import json
import logging
import os
import random
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qualia_bridge import QualiaBridge, NUM_LAYERS
from ugv_driver import UGVDriver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s explorer: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MOTOR_STATE_FILE = "/tmp/qualia_motor_state.json"

# Exploration parameters
BASE_SPEED = 60
PAUSE_SPEED = 0
VFE_SPIKE_THRESHOLD = 3.0    # VFE > baseline * this = spike
VFE_SAFETY_THRESHOLD = 10.0  # VFE > baseline * this for > 5s = emergency stop
VFE_BORED_THRESHOLD = 0.5    # VFE < baseline * this = bored, seek novelty
PAUSE_DURATION = 2.0         # Seconds to pause on VFE spike (let layers learn)
TURN_DURATION = 0.4          # Seconds for a random turn
FORWARD_CHUNK = 1.0          # Move forward in 1-second chunks
SAFETY_WINDOW = 5.0          # Seconds of sustained high VFE before emergency stop


def write_motor_state(left: int, right: int):
    """Write current motor speeds to shared file for session recorder."""
    try:
        state = {"motor_left": left, "motor_right": right, "ts": time.time()}
        tmp = MOTOR_STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, MOTOR_STATE_FILE)
    except OSError:
        pass


def get_mean_vfe(bridge: QualiaBridge, layers: list = None) -> float:
    """Get mean VFE across specified layers (default: all)."""
    if layers is None:
        layers = list(range(NUM_LAYERS))
    total = 0.0
    count = 0
    for i in layers:
        belief = bridge.read_layer_belief(i)
        total += belief.vfe
        count += 1
    return total / count if count > 0 else 0.0


def get_layer_vfes(bridge: QualiaBridge) -> list:
    """Get VFE for each layer."""
    return [bridge.read_layer_belief(i).vfe for i in range(NUM_LAYERS)]


class AutonomousExplorer:
    def __init__(self, port="/dev/ttyACM0", speed=BASE_SPEED):
        self.bridge = QualiaBridge()
        self.ugv = None
        self.port = port
        self.speed = speed
        self.running = True
        self.baseline_vfe = None
        self.high_vfe_start = None
        self.motor_left = 0
        self.motor_right = 0

    def start(self):
        if not self.bridge.open():
            log.error("Cannot open Qualia SHM. Is qualia-watch running?")
            return False

        try:
            self.ugv = UGVDriver(port=self.port)
        except Exception as e:
            log.error(f"Cannot connect to UGV at {self.port}: {e}")
            self.bridge.close()
            return False

        log.info(f"Connected: SHM + UGV at {self.port}")
        return True

    def _set_motors(self, left: int, right: int):
        self.motor_left = left
        self.motor_right = right
        self.ugv.move(left, right)
        write_motor_state(left, right)

    def _stop(self):
        self._set_motors(0, 0)

    def _forward(self):
        self._set_motors(self.speed, self.speed)

    def _turn_random(self):
        if random.random() < 0.5:
            log.info("Turning left (seeking novelty)")
            self._set_motors(-self.speed, self.speed)
        else:
            log.info("Turning right (seeking novelty)")
            self._set_motors(self.speed, -self.speed)
        time.sleep(TURN_DURATION)
        self._stop()

    def calibrate_baseline(self, samples=10, interval=0.2):
        """Sample VFE for a couple seconds to establish baseline."""
        log.info("Calibrating VFE baseline...")
        readings = []
        for _ in range(samples):
            readings.append(get_mean_vfe(self.bridge))
            time.sleep(interval)
        self.baseline_vfe = max(sum(readings) / len(readings), 0.001)
        log.info(f"Baseline VFE: {self.baseline_vfe:.6f}")

    def run(self):
        self.calibrate_baseline()

        log.info("Starting curiosity-driven exploration")
        log.info(f"  Speed: {self.speed}, Spike: {VFE_SPIKE_THRESHOLD}x, "
                 f"Bored: {VFE_BORED_THRESHOLD}x, Safety: {VFE_SAFETY_THRESHOLD}x")

        while self.running:
            vfe = get_mean_vfe(self.bridge)
            ratio = vfe / self.baseline_vfe if self.baseline_vfe > 0 else 1.0

            # Safety check: sustained extreme VFE
            if ratio > VFE_SAFETY_THRESHOLD:
                if self.high_vfe_start is None:
                    self.high_vfe_start = time.monotonic()
                elif time.monotonic() - self.high_vfe_start > SAFETY_WINDOW:
                    log.warning(f"SAFETY STOP: VFE {vfe:.4f} ({ratio:.1f}x baseline) "
                                f"sustained for {SAFETY_WINDOW}s")
                    self._stop()
                    time.sleep(5.0)
                    self.high_vfe_start = None
                    continue
            else:
                self.high_vfe_start = None

            # VFE spike → pause and let layers learn
            if ratio > VFE_SPIKE_THRESHOLD:
                log.info(f"VFE spike: {vfe:.4f} ({ratio:.1f}x baseline) — pausing to learn")
                self._stop()
                time.sleep(PAUSE_DURATION)
                # Update baseline with exponential moving average
                self.baseline_vfe = self.baseline_vfe * 0.9 + vfe * 0.1
                continue

            # Low VFE → bored, turn to seek novelty
            if ratio < VFE_BORED_THRESHOLD:
                log.info(f"Low VFE: {vfe:.4f} ({ratio:.1f}x baseline) — seeking novelty")
                self._turn_random()
                # Move forward briefly after turn
                self._forward()
                time.sleep(FORWARD_CHUNK)
                self._stop()
                continue

            # Normal VFE → move forward
            self._forward()
            time.sleep(FORWARD_CHUNK)
            self._stop()

            # Slowly adapt baseline
            self.baseline_vfe = self.baseline_vfe * 0.95 + vfe * 0.05

            # Brief pause between chunks for stability
            time.sleep(0.1)

    def shutdown(self):
        self.running = False
        if self.ugv:
            self.ugv.close()
        if self.bridge.is_open:
            self.bridge.close()
        write_motor_state(0, 0)
        log.info("Explorer shutdown complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Curiosity-driven autonomous exploration")
    parser.add_argument("--port", default="/dev/ttyACM0", help="UGV serial port")
    parser.add_argument("--speed", type=int, default=BASE_SPEED, help="Base motor speed (0-128)")
    args = parser.parse_args()

    explorer = AutonomousExplorer(port=args.port, speed=args.speed)

    def handle_signal(sig, frame):
        explorer.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not explorer.start():
        sys.exit(1)

    try:
        explorer.run()
    except Exception as e:
        log.error(f"Explorer error: {e}")
    finally:
        explorer.shutdown()


if __name__ == "__main__":
    main()
