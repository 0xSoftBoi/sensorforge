#!/usr/bin/env python3
"""
Autonomous Explorer — Expected Free Energy (EFE) driven exploration.

Phase 3.1: Replaces random left/right with EFE minimization:
  EFE(action) = -epistemic_value - pragmatic_value + novelty_cost

Phase 3.2: Records actions to SHM ActionHistory so layers can learn
sensorimotor contingencies.

Phase 6.4: Boltzmann softmax action selection with adaptive temperature,
TD(λ)-style outcome prediction, and information-theoretic curiosity bonus
(Pathak et al. 2017 ICM-inspired).

Usage:
    python3 autonomous_explorer.py [--port /dev/ttyTHS1] [--speed 60]
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
CURIOSITY_WEIGHT = 0.4        # weight for prediction-error curiosity (Phase 6.4)
HISTORY_SIZE = 200            # action-outcome buffer size
MIN_HISTORY = 10              # minimum samples before using EFE

# Boltzmann softmax parameters (Phase 6.4)
SOFTMAX_TEMP_INIT = 1.0       # initial temperature (higher = more random)
SOFTMAX_TEMP_MIN = 0.1        # minimum temperature (near-greedy)
SOFTMAX_TEMP_DECAY = 0.995    # decay per action (anneal toward exploitation)

# TD learning parameters (Phase 6.4)
TD_LAMBDA = 0.7               # eligibility trace decay
TD_GAMMA = 0.95               # discount factor for future VFE reduction
TD_LR = 0.05                  # learning rate for value estimates

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
    """Expected Free Energy policy with Boltzmann softmax selection,
    TD(λ)-style value prediction, and curiosity bonus.

    Phase 6.4 upgrades:
    - Boltzmann softmax: stochastic selection proportional to exp(-EFE/τ)
      with adaptive temperature τ that decays from exploration to exploitation
    - TD(λ) prediction: exponentially-weighted temporal difference learning
      for more accurate VFE outcome predictions (replaces simple averaging)
    - Curiosity bonus: reward for prediction errors (Pathak et al. 2017 ICM)
      — prefer actions whose outcomes are hard to predict (high epistemic value)
    """

    def __init__(self, history_size: int = HISTORY_SIZE):
        self.history: deque = deque(maxlen=history_size)
        self.candidate_actions = [ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT]
        self.temperature = SOFTMAX_TEMP_INIT

        # TD(λ) value estimates: V(action) = expected cumulative VFE reduction
        self.action_values = {a: 0.0 for a in self.candidate_actions}
        # Eligibility traces for TD(λ)
        self.eligibility = {a: 0.0 for a in self.candidate_actions}
        self.last_action = None
        self.last_vfe = None

        # Prediction error tracking for curiosity bonus
        self.prediction_errors = {a: deque(maxlen=50) for a in self.candidate_actions}

    def record(self, outcome: ActionOutcome):
        """Record an action-outcome pair and update TD value estimates."""
        self.history.append(outcome)

        vfe_delta = float(outcome.post_vfe.mean() - outcome.pre_vfe.mean())

        # TD(λ) update: reward = -vfe_delta (VFE decrease is good)
        reward = -vfe_delta
        if self.last_action is not None and self.last_vfe is not None:
            # TD error: δ = r + γ·V(a') - V(a)
            td_error = (reward + TD_GAMMA * self.action_values.get(outcome.action, 0.0)
                        - self.action_values.get(self.last_action, 0.0))

            # Update all action values proportional to eligibility
            for a in self.candidate_actions:
                self.action_values[a] += TD_LR * td_error * self.eligibility[a]
                # Decay eligibility traces
                self.eligibility[a] *= TD_GAMMA * TD_LAMBDA

        # Set eligibility for current action
        self.eligibility[outcome.action] = 1.0
        self.last_action = outcome.action
        self.last_vfe = float(outcome.post_vfe.mean())

        # Track prediction error for curiosity (Phase 6.4)
        predicted_delta, predicted_emb, conf = self.predict_outcome(outcome.action)
        actual_delta = vfe_delta
        pred_error = abs(actual_delta - predicted_delta)
        if predicted_emb is not None:
            emb_error = float(np.linalg.norm(outcome.post_embedding - predicted_emb))
            pred_error += emb_error
        self.prediction_errors[outcome.action].append(pred_error)

        # Decay temperature (anneal from exploration to exploitation)
        self.temperature = max(SOFTMAX_TEMP_MIN, self.temperature * SOFTMAX_TEMP_DECAY)

    def predict_outcome(self, action: int) -> tuple:
        """Predict (vfe_change, embedding_change) for an action from history.
        Uses recency-weighted averaging with TD-bootstrapped value correction.
        Returns (mean_vfe_delta, predicted_post_embedding, confidence)."""
        matching = [h for h in self.history if h.action == action]

        if len(matching) < 3:
            return 0.0, None, 0.0

        # Weighted by recency (more recent = more weight)
        weights = np.exp(np.linspace(-2, 0, len(matching)))
        weights /= weights.sum()

        vfe_deltas = np.array([
            (h.post_vfe.mean() - h.pre_vfe.mean()) for h in matching
        ])
        mean_delta = float(np.average(vfe_deltas, weights=weights))

        # Blend in TD value estimate for multi-step lookahead
        td_value = self.action_values.get(action, 0.0)
        confidence = min(len(matching) / 20.0, 1.0)
        # TD correction: shift predicted delta toward learned value
        mean_delta = mean_delta * 0.7 + (-td_value) * 0.3 * confidence

        post_embeddings = np.array([h.post_embedding for h in matching])
        predicted_emb = np.average(post_embeddings, axis=0, weights=weights)

        return mean_delta, predicted_emb, confidence

    def _curiosity_bonus(self, action: int) -> float:
        """Information-theoretic curiosity: reward for high prediction error.
        Actions whose outcomes we can't predict well have high epistemic value."""
        errors = self.prediction_errors.get(action)
        if not errors or len(errors) < 3:
            return 1.0  # maximum curiosity for unexplored actions
        mean_error = float(np.mean(list(errors)))
        # Normalize: high error = high curiosity, but cap to prevent runaway
        return min(mean_error, 5.0) / 5.0

    def compute_efe(self, action: int, current_vfes: np.ndarray,
                    current_embedding: np.ndarray,
                    directive_embedding: np.ndarray,
                    recent_embeddings: list) -> float:
        """Compute Expected Free Energy for an action.
        Lower = better (we minimize EFE)."""

        vfe_delta, predicted_emb, confidence = self.predict_outcome(action)

        # 1. Epistemic value: expected VFE reduction (information gain)
        epistemic = -vfe_delta * confidence

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
            novelty = 1.0 - mean_sim

        # 4. Curiosity bonus (Phase 6.4): reward unpredictable outcomes
        curiosity = self._curiosity_bonus(action)

        efe = -(EPISTEMIC_WEIGHT * epistemic
                + PRAGMATIC_WEIGHT * pragmatic
                + NOVELTY_WEIGHT * novelty
                + CURIOSITY_WEIGHT * curiosity)

        return efe

    def select_action(self, current_vfes: np.ndarray,
                      current_embedding: np.ndarray,
                      directive_embedding: np.ndarray,
                      recent_embeddings: list) -> tuple:
        """Select action via Boltzmann softmax over EFE scores.
        Returns (action, efe_scores)."""

        if len(self.history) < MIN_HISTORY:
            action = random.choice(self.candidate_actions)
            return action, {}

        efe_scores = {}
        for action in self.candidate_actions:
            efe = self.compute_efe(
                action, current_vfes, current_embedding,
                directive_embedding, recent_embeddings,
            )
            efe_scores[action] = efe

        # Boltzmann softmax: P(a) ∝ exp(-EFE(a) / τ)
        # Lower EFE = higher probability, temperature controls exploration
        actions = list(efe_scores.keys())
        efes = np.array([efe_scores[a] for a in actions])

        # Numerical stability: subtract max before exp
        efes_shifted = -(efes - efes.min()) / max(self.temperature, 1e-6)
        exp_efes = np.exp(efes_shifted)
        probs = exp_efes / exp_efes.sum()

        # Sample action from distribution
        chosen_idx = np.random.choice(len(actions), p=probs)
        best_action = actions[chosen_idx]

        return best_action, efe_scores


class AutonomousExplorer:
    def __init__(self, port="/dev/ttyTHS1", speed=BASE_SPEED):
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
        log.info("Starting EFE-driven exploration (Phase 6.4: Boltzmann + TD + Curiosity)")
        log.info(f"  Speed: {self.speed}, Safety: {VFE_SAFETY_THRESHOLD}x, "
                 f"History: {HISTORY_SIZE}, EFE weights: "
                 f"ep={EPISTEMIC_WEIGHT} pr={PRAGMATIC_WEIGHT} "
                 f"nv={NOVELTY_WEIGHT} cu={CURIOSITY_WEIGHT}")
        log.info(f"  Softmax τ₀={SOFTMAX_TEMP_INIT} τ_min={SOFTMAX_TEMP_MIN} "
                 f"decay={SOFTMAX_TEMP_DECAY}, TD λ={TD_LAMBDA} γ={TD_GAMMA}")

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
                td_str = " ".join(
                    f"{ACTION_NAMES.get(a, '?')}={v:.3f}"
                    for a, v in self.policy.action_values.items()
                )
                log.info(f"#{action_count} {ACTION_NAMES.get(action, '?')}: "
                         f"dVFE={vfe_delta:+.4f}, z_max={max_z:.1f}, "
                         f"τ={self.policy.temperature:.3f}, "
                         f"EFE=[{scores_str}], TD=[{td_str}]")
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
    parser.add_argument("--port", default="/dev/ttyTHS1", help="UGV serial port")
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
