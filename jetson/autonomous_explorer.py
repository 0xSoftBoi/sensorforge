#!/usr/bin/env python3
"""
Autonomous Explorer — Expected Free Energy (EFE) driven exploration.

Phase 3.1: Replaces random left/right with EFE minimization:
  EFE(action) = -epistemic_value - pragmatic_value + novelty_cost

Maintains a circular buffer of (action, pre_vfe, post_vfe, embedding) tuples.
For each candidate action, predicts outcome from history and picks lowest EFE.

Phase 3.2: Records actions to SHM ActionHistory so layers can learn
sensorimotor contingencies.

Usage:
    python3 autonomous_explorer.py [--port /dev/ttyACM0] [--speed 60]
"""

import json
import logging
import math
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

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


def acquire_singleton(name):
    """Ensure only one instance runs. Kill stale instance if found."""
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
    log.info(f"Singleton lock acquired: {pidfile} (PID {os.getpid()})")

# Exploration parameters
BASE_SPEED = 60
VFE_SAFETY_THRESHOLD = 10.0   # emergency stop multiplier
SAFETY_WINDOW = 5.0           # seconds of sustained danger
ACTION_DURATION = 1.0         # seconds per action execution
TURN_DURATION = 0.4           # seconds for a turn action
EMBEDDING_DIM = 16            # compressed embedding for action history

# EFE parameters
EPISTEMIC_WEIGHT = 1.0        # weight for information gain
PRAGMATIC_WEIGHT = 0.5        # weight for goal alignment
NOVELTY_WEIGHT = 0.3          # weight for novelty seeking
HISTORY_SIZE = 200            # action-outcome buffer size
MIN_HISTORY = 10              # minimum samples before using EFE

# Action codes
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_REVERSE = 4

ACTION_NAMES = {0: "stop", 1: "forward", 2: "left", 3: "right", 4: "reverse"}


@dataclass
class ActionOutcome:
    """Record of an action and its effect on the system."""
    action: int
    speed: int
    pre_vfe: np.ndarray     # VFE per layer before action
    post_vfe: np.ndarray    # VFE per layer after action
    pre_embedding: np.ndarray  # 16-dim compressed scene embedding before
    post_embedding: np.ndarray # 16-dim compressed scene embedding after
    timestamp: float


def write_motor_state(left: int, right: int):
    try:
        state = {"motor_left": left, "motor_right": right, "ts": time.time()}
        tmp = MOTOR_STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, MOTOR_STATE_FILE)
    except OSError:
        pass


def get_layer_vfes(bridge: QualiaBridge) -> np.ndarray:
    """Get VFE array for all layers."""
    return np.array([bridge.read_layer_belief(i).vfe for i in range(NUM_LAYERS)])


def get_layer_zscores(bridge: QualiaBridge) -> np.ndarray:
    """Get z-scores for all layers (Phase 1.2)."""
    return np.array([bridge.vfe_zscore(i) for i in range(NUM_LAYERS)])


def get_scene_embedding(bridge: QualiaBridge, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Get compressed scene embedding from world model."""
    try:
        wm = bridge.read_world_model()
        full = np.array(wm.scene_embedding, dtype=np.float32)
        # Average-pool from 64 to dim
        if len(full) > dim:
            bin_size = len(full) / dim
            compressed = np.array([
                full[int(i * bin_size):int((i + 1) * bin_size)].mean()
                for i in range(dim)
            ])
            return compressed
        return full[:dim]
    except Exception:
        return np.zeros(dim, dtype=np.float32)


def get_directive_embedding(bridge: QualiaBridge, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Get the directive embedding (goal state) for pragmatic value."""
    try:
        wm = bridge.read_world_model()
        # Use a hash-based embedding of the directive text as the goal vector
        directive = wm.directive
        if not directive:
            return np.zeros(dim, dtype=np.float32)
        # Simple deterministic embedding from text
        emb = np.zeros(dim, dtype=np.float32)
        for i, ch in enumerate(directive.encode("utf-8")):
            emb[i % dim] += math.sin(ch * 0.1 + i * 0.3) * 0.1
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    except Exception:
        return np.zeros(dim, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EFEPolicy:
    """Expected Free Energy policy for action selection.

    EFE(action) = -epistemic_value(action) - pragmatic_value(action) + novelty_cost(action)

    Lower EFE = better action.
    - Epistemic: how much VFE change do we expect? (information gain)
    - Pragmatic: how close to the directive goal will we get?
    - Novelty: prefer states dissimilar to recent history
    """

    def __init__(self, history_size: int = HISTORY_SIZE):
        self.history: deque = deque(maxlen=history_size)
        self.candidate_actions = [ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT]

    def record(self, outcome: ActionOutcome):
        """Record an action-outcome pair."""
        self.history.append(outcome)

    def predict_outcome(self, action: int) -> tuple:
        """Predict (vfe_change, embedding_change) for an action from history.
        Returns (mean_vfe_delta, predicted_post_embedding, confidence)."""
        matching = [h for h in self.history if h.action == action]

        if len(matching) < 3:
            # Not enough data — return neutral prediction with low confidence
            return 0.0, None, 0.0

        # Weighted by recency (more recent = more weight)
        weights = np.exp(np.linspace(-2, 0, len(matching)))
        weights /= weights.sum()

        vfe_deltas = np.array([
            (h.post_vfe.mean() - h.pre_vfe.mean()) for h in matching
        ])
        mean_delta = float(np.average(vfe_deltas, weights=weights))

        # Predicted post-embedding: weighted average of historical post-embeddings
        post_embeddings = np.array([h.post_embedding for h in matching])
        predicted_emb = np.average(post_embeddings, axis=0, weights=weights)

        confidence = min(len(matching) / 20.0, 1.0)  # saturates at 20 samples

        return mean_delta, predicted_emb, confidence

    def compute_efe(self, action: int, current_vfes: np.ndarray,
                    current_embedding: np.ndarray,
                    directive_embedding: np.ndarray,
                    recent_embeddings: list) -> float:
        """Compute Expected Free Energy for an action.
        Lower = better (we minimize EFE)."""

        vfe_delta, predicted_emb, confidence = self.predict_outcome(action)

        # 1. Epistemic value: expected VFE reduction (information gain)
        # Negative delta = VFE decreased = good = positive epistemic value
        epistemic = -vfe_delta * confidence  # flip sign: decrease is good

        # 2. Pragmatic value: predicted alignment with directive
        pragmatic = 0.0
        if predicted_emb is not None and np.linalg.norm(directive_embedding) > 0.01:
            pragmatic = cosine_similarity(predicted_emb, directive_embedding) * confidence

        # 3. Novelty: prefer states dissimilar to recent history
        novelty = 0.0
        if predicted_emb is not None and recent_embeddings:
            similarities = [
                cosine_similarity(predicted_emb, past_emb)
                for past_emb in recent_embeddings[-20:]
            ]
            mean_sim = np.mean(similarities) if similarities else 0.0
            novelty = 1.0 - mean_sim  # high novelty when low similarity

        # EFE = -epistemic - pragmatic - novelty (we minimize, so negate the goods)
        efe = -(EPISTEMIC_WEIGHT * epistemic
                + PRAGMATIC_WEIGHT * pragmatic
                + NOVELTY_WEIGHT * novelty)

        return efe

    def select_action(self, current_vfes: np.ndarray,
                      current_embedding: np.ndarray,
                      directive_embedding: np.ndarray,
                      recent_embeddings: list) -> tuple:
        """Select the action with lowest EFE. Returns (action, efe_scores)."""

        if len(self.history) < MIN_HISTORY:
            # Not enough data — explore randomly
            action = random.choice(self.candidate_actions)
            return action, {}

        efe_scores = {}
        for action in self.candidate_actions:
            efe = self.compute_efe(
                action, current_vfes, current_embedding,
                directive_embedding, recent_embeddings,
            )
            efe_scores[action] = efe

        # Select action with lowest EFE (with small random tiebreaker)
        best_action = min(efe_scores, key=lambda a: efe_scores[a] + random.gauss(0, 0.01))

        return best_action, efe_scores


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
        self.policy = EFEPolicy()
        self.recent_embeddings: list = []

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

    def _execute_action(self, action: int):
        """Execute a motor action."""
        if action == ACTION_FORWARD:
            self._set_motors(self.speed, self.speed)
            time.sleep(ACTION_DURATION)
        elif action == ACTION_LEFT:
            self._set_motors(-self.speed, self.speed)
            time.sleep(TURN_DURATION)
        elif action == ACTION_RIGHT:
            self._set_motors(self.speed, -self.speed)
            time.sleep(TURN_DURATION)
        elif action == ACTION_REVERSE:
            self._set_motors(-self.speed, -self.speed)
            time.sleep(ACTION_DURATION * 0.5)
        self._stop()

    def calibrate_baseline(self, samples=10, interval=0.2):
        """Sample VFE for a couple seconds to establish baseline."""
        log.info("Calibrating VFE baseline...")
        readings = []
        for _ in range(samples):
            vfes = get_layer_vfes(self.bridge)
            readings.append(float(vfes.mean()))
            time.sleep(interval)
        self.baseline_vfe = max(sum(readings) / len(readings), 0.001)
        log.info(f"Baseline VFE: {self.baseline_vfe:.6f}")

    def run(self):
        self.calibrate_baseline()

        directive_emb = get_directive_embedding(self.bridge)
        log.info("Starting EFE-driven exploration")
        log.info(f"  Speed: {self.speed}, Safety: {VFE_SAFETY_THRESHOLD}x, "
                 f"History: {HISTORY_SIZE}, EFE weights: "
                 f"ep={EPISTEMIC_WEIGHT} pr={PRAGMATIC_WEIGHT} nv={NOVELTY_WEIGHT}")

        action_count = 0

        while self.running:
            # Read current state
            current_vfes = get_layer_vfes(self.bridge)
            current_zscores = get_layer_zscores(self.bridge)
            current_emb = get_scene_embedding(self.bridge)
            mean_vfe = float(current_vfes.mean())
            ratio = mean_vfe / self.baseline_vfe if self.baseline_vfe > 0 else 1.0
            max_z = float(np.max(np.abs(current_zscores)))

            # Safety check: sustained extreme VFE
            if ratio > VFE_SAFETY_THRESHOLD:
                if self.high_vfe_start is None:
                    self.high_vfe_start = time.monotonic()
                elif time.monotonic() - self.high_vfe_start > SAFETY_WINDOW:
                    log.warning(f"SAFETY STOP: VFE {mean_vfe:.4f} ({ratio:.1f}x baseline) "
                                f"sustained for {SAFETY_WINDOW}s")
                    self._stop()
                    time.sleep(5.0)
                    self.high_vfe_start = None
                    continue
            else:
                self.high_vfe_start = None

            # High surprise (z > 3): pause to let layers learn
            if max_z > 3.0 and ratio > 2.0:
                log.info(f"Surprise z={max_z:.1f}, VFE={mean_vfe:.4f} — pausing to learn")
                self._stop()
                time.sleep(2.0)
                self.baseline_vfe = self.baseline_vfe * 0.9 + mean_vfe * 0.1
                continue

            # Record pre-action state
            pre_vfes = current_vfes.copy()
            pre_emb = current_emb.copy()

            # Select action via EFE policy
            action, efe_scores = self.policy.select_action(
                current_vfes, current_emb, directive_emb, self.recent_embeddings,
            )

            # Execute action
            self._execute_action(action)
            action_count += 1

            # Wait briefly for layers to process
            time.sleep(0.3)

            # Measure post-action state
            post_vfes = get_layer_vfes(self.bridge)
            post_emb = get_scene_embedding(self.bridge)

            # Record outcome
            outcome = ActionOutcome(
                action=action,
                speed=self.speed,
                pre_vfe=pre_vfes,
                post_vfe=post_vfes,
                pre_embedding=pre_emb,
                post_embedding=post_emb,
                timestamp=time.time(),
            )
            self.policy.record(outcome)
            self.recent_embeddings.append(post_emb.copy())
            if len(self.recent_embeddings) > 50:
                self.recent_embeddings.pop(0)

            # Phase 3.2: Write action to SHM for layers to learn
            self.bridge.write_action(
                action=action,
                speed_left=self.motor_left,
                speed_right=self.motor_right,
                pre_vfe=float(pre_vfes.mean()),
                post_vfe=float(post_vfes.mean()),
                embedding=post_emb.tolist(),
            )

            # Slowly adapt baseline
            self.baseline_vfe = self.baseline_vfe * 0.95 + mean_vfe * 0.05

            # Log
            vfe_delta = float(post_vfes.mean() - pre_vfes.mean())
            if efe_scores:
                scores_str = " ".join(
                    f"{ACTION_NAMES.get(a, '?')}={s:.3f}" for a, s in efe_scores.items()
                )
                log.info(f"#{action_count} {ACTION_NAMES.get(action, '?')}: "
                         f"dVFE={vfe_delta:+.4f}, z_max={max_z:.1f}, EFE=[{scores_str}]")
            else:
                log.info(f"#{action_count} {ACTION_NAMES.get(action, '?')} (random): "
                         f"dVFE={vfe_delta:+.4f}, z_max={max_z:.1f}, "
                         f"history={len(self.policy.history)}/{MIN_HISTORY}")

            # Periodically refresh directive embedding
            if action_count % 50 == 0:
                directive_emb = get_directive_embedding(self.bridge)

            # Brief pause for stability
            time.sleep(0.1)

    def shutdown(self):
        self.running = False
        if self.ugv:
            self.ugv.close()
        if self.bridge.is_open:
            self.bridge.close()
        write_motor_state(0, 0)
        log.info(f"Explorer shutdown (history: {len(self.policy.history)} entries)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EFE-driven autonomous exploration")
    parser.add_argument("--port", default="/dev/ttyACM0", help="UGV serial port")
    parser.add_argument("--speed", type=int, default=BASE_SPEED, help="Base motor speed (0-128)")
    args = parser.parse_args()

    acquire_singleton("explorer")

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
