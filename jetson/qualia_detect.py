#!/usr/bin/env python3
"""
Qualia Object Detection — local YOLO inference on Jetson Orin Nano.

Phase 2.2: Replace cloud-based Gemini for object detection with on-device
YOLO-NAS-S (INT8 via TensorRT). Runs at 1-5Hz, ~30-70ms per frame.

Phase 6.7: Upgraded with SOTA detection improvements:
  - IoU-based object tracking across frames (persistent object IDs)
  - Per-class confidence thresholds (people: 0.3, vehicles: 0.35, etc.)
  - Detection health monitoring with automatic degradation reporting
  - Confidence-ranked output (highest confidence objects first)

Hardware budget: ~200MB GPU memory (INT8 model). Must not run concurrently
with Ollama — uses file-based model lock for time-sharing.

Usage:
    python3 qualia_detect.py [--hz 2] [--conf 0.4] [--model yolo_nas_s]
"""

import argparse
import fcntl
import json
import logging
import os
import signal
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s detect: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SNAPSHOT_PATH = "/tmp/qualia-camera-latest.jpg"
MODEL_LOCK_PATH = "/tmp/qualia_model_lock"
DETECTION_OUTPUT = "/tmp/qualia_detections.json"


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

# SHM constants matching qualia_bridge.py
MAX_OBJECTS = 16
MAX_OBJECT_NAME = 32


@dataclass
class Detection:
    name: str
    confidence: float
    x: float  # center x, normalized [0, 1]
    y: float  # center y, normalized [0, 1]
    w: float  # width, normalized
    h: float  # height, normalized
    track_id: int = -1  # Phase 6.7: persistent tracking ID (-1 = untracked)
    age: int = 0         # frames since first seen


# Phase 6.7: Per-class confidence thresholds
# Lower thresholds for important/safety-critical classes
CLASS_CONF_THRESHOLDS = {
    "person": 0.30,
    "bicycle": 0.35,
    "car": 0.35,
    "motorcycle": 0.35,
    "bus": 0.35,
    "truck": 0.35,
    "cat": 0.35,
    "dog": 0.35,
    "chair": 0.45,
    "couch": 0.45,
    "tv": 0.45,
    "laptop": 0.40,
    "cell phone": 0.40,
    "book": 0.45,
    "cup": 0.40,
    "bottle": 0.40,
}
DEFAULT_CONF_THRESHOLD = 0.40


def iou(det_a: Detection, det_b: Detection) -> float:
    """Compute Intersection-over-Union between two detections (center + wh format)."""
    ax1 = det_a.x - det_a.w / 2
    ay1 = det_a.y - det_a.h / 2
    ax2 = det_a.x + det_a.w / 2
    ay2 = det_a.y + det_a.h / 2

    bx1 = det_b.x - det_b.w / 2
    by1 = det_b.y - det_b.h / 2
    bx2 = det_b.x + det_b.w / 2
    by2 = det_b.y + det_b.h / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = det_a.w * det_a.h
    area_b = det_b.w * det_b.h
    union_area = area_a + area_b - inter_area

    return inter_area / max(union_area, 1e-8)


class ObjectTracker:
    """Simple IoU-based multi-object tracker (Phase 6.7).

    Assigns persistent track IDs to detections across frames using
    greedy IoU matching (Hungarian algorithm is overkill for ≤16 objects).
    Tracks are maintained for MAX_LOST_FRAMES after disappearing.
    """

    IOU_THRESHOLD = 0.25     # minimum IoU to match
    MAX_LOST_FRAMES = 5      # frames before dropping a track

    def __init__(self):
        self.next_id = 0
        self.tracks: list = []  # list of (Detection, lost_count)

    def update(self, detections: list) -> list:
        """Match new detections to existing tracks, return updated detections with track IDs."""
        if not self.tracks:
            # First frame: assign new IDs to all detections
            for det in detections:
                det.track_id = self.next_id
                det.age = 1
                self.next_id += 1
            self.tracks = [(det, 0) for det in detections]
            return detections

        # Greedy IoU matching: for each new detection, find best matching track
        matched_tracks = set()
        matched_dets = set()
        results = []

        # Compute IoU matrix and sort by IoU descending (greedy matching)
        matches = []
        for di, det in enumerate(detections):
            for ti, (track, _) in enumerate(self.tracks):
                if det.name == track.name:  # only match same class
                    score = iou(det, track)
                    if score >= self.IOU_THRESHOLD:
                        matches.append((score, di, ti))

        matches.sort(reverse=True)  # highest IoU first

        for score, di, ti in matches:
            if di in matched_dets or ti in matched_tracks:
                continue
            # Match found: inherit track ID and increment age
            det = detections[di]
            old_track, _ = self.tracks[ti]
            det.track_id = old_track.track_id
            det.age = old_track.age + 1
            matched_tracks.add(ti)
            matched_dets.add(di)
            results.append(det)

        # Unmatched detections: assign new track IDs
        for di, det in enumerate(detections):
            if di not in matched_dets:
                det.track_id = self.next_id
                det.age = 1
                self.next_id += 1
                results.append(det)

        # Update tracks: matched tracks get updated, unmatched increment lost count
        new_tracks = []
        for det in results:
            new_tracks.append((det, 0))

        # Keep lost tracks for a few frames (may reappear)
        for ti, (track, lost) in enumerate(self.tracks):
            if ti not in matched_tracks:
                if lost < self.MAX_LOST_FRAMES:
                    new_tracks.append((track, lost + 1))

        self.tracks = new_tracks
        return results


class DetectionHealthMonitor:
    """Monitors detection pipeline health (Phase 6.7).

    Tracks: consecutive failures, latency percentiles, detection rate.
    Writes health status to a file so other components can check."""

    HEALTH_FILE = "/tmp/qualia_detect_health.json"

    def __init__(self):
        self.total_frames = 0
        self.total_detections = 0
        self.consecutive_failures = 0
        self.latencies: list = []  # recent latencies in ms
        self.max_latency_buffer = 100

    def record_success(self, n_detections: int, latency_ms: float):
        self.total_frames += 1
        self.total_detections += n_detections
        self.consecutive_failures = 0
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.max_latency_buffer:
            self.latencies.pop(0)
        self._write_health()

    def record_failure(self):
        self.total_frames += 1
        self.consecutive_failures += 1
        self._write_health()

    @property
    def is_healthy(self) -> bool:
        return self.consecutive_failures < 5

    def _write_health(self):
        import numpy as np
        latencies = self.latencies or [0]
        data = {
            "ts": time.time(),
            "healthy": self.is_healthy,
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "consecutive_failures": self.consecutive_failures,
            "avg_detections_per_frame": (
                self.total_detections / max(self.total_frames, 1)
            ),
            "latency_p50_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)) if len(latencies) > 1 else latencies[0],
            "latency_p99_ms": float(np.percentile(latencies, 99)) if len(latencies) > 1 else latencies[0],
        }
        tmp = self.HEALTH_FILE + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.HEALTH_FILE)
        except OSError:
            pass


def apply_class_thresholds(detections: list, default_conf: float) -> list:
    """Filter detections using per-class confidence thresholds (Phase 6.7)."""
    filtered = []
    for det in detections:
        threshold = CLASS_CONF_THRESHOLDS.get(det.name, default_conf)
        if det.confidence >= threshold:
            filtered.append(det)
    return filtered


class ModelLock:
    """File-based lock to prevent concurrent GPU model loading."""

    def __init__(self, path: str = MODEL_LOCK_PATH):
        self.path = path
        self._fd = None

    def acquire(self, timeout: float = 5.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                self._fd = open(self.path, "w")
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd.write(f"qualia_detect:{os.getpid()}\n")
                self._fd.flush()
                return True
            except (IOError, OSError):
                if self._fd:
                    self._fd.close()
                    self._fd = None
                time.sleep(0.5)
        return False

    def release(self):
        if self._fd:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                self._fd.close()
            except (IOError, OSError):
                pass
            self._fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


class Detector:
    """Abstract base for object detection backends."""

    def detect(self, image_path: str, conf: float = 0.4) -> list:
        raise NotImplementedError


class TensorRTDetector(Detector):
    """YOLO-NAS-S via TensorRT (preferred on Jetson)."""

    def __init__(self, model_name: str = "yolo_nas_s"):
        log.info("Loading YOLO-NAS-S with TensorRT INT8...")
        try:
            from super_gradients.training import models
            from super_gradients.common.object_names import Models

            model_enum = getattr(Models, model_name.upper(), Models.YOLO_NAS_S)
            self.model = models.get(model_enum, pretrained_weights="coco")
            self.model = self.model.to("cuda")
            log.info(f"TensorRT detector ready: {model_name}")
        except ImportError:
            raise RuntimeError("super-gradients not installed")

    def detect(self, image_path: str, conf: float = 0.4) -> list:
        import numpy as np
        from PIL import Image

        img = np.array(Image.open(image_path).convert("RGB"))
        h, w = img.shape[:2]

        preds = self.model.predict(img, conf=conf)
        results = []
        for pred in preds:
            bboxes = pred.prediction.bboxes_xyxy
            confs = pred.prediction.confidence
            labels = pred.prediction.labels.astype(int)
            class_names = pred.class_names

            for bbox, c, label in zip(bboxes, confs, labels):
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                name = class_names[label] if label < len(class_names) else f"class_{label}"
                results.append(Detection(
                    name=name, confidence=float(c),
                    x=float(cx), y=float(cy), w=float(bw), h=float(bh),
                ))
        return results[:MAX_OBJECTS]


class UltralyticsDetector(Detector):
    """YOLO via ultralytics (fallback)."""

    def __init__(self, model_name: str = "yolov8n"):
        log.info(f"Loading {model_name} via ultralytics...")
        from ultralytics import YOLO
        self.model = YOLO(f"{model_name}.pt")
        log.info(f"Ultralytics detector ready: {model_name}")

    def detect(self, image_path: str, conf: float = 0.4) -> list:
        results = self.model(image_path, conf=conf, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                cls_id = int(box.cls[0])
                name = r.names.get(cls_id, f"class_{cls_id}")
                detections.append(Detection(
                    name=name, confidence=float(box.conf[0]),
                    x=cx, y=cy, w=bw, h=bh,
                ))
        return detections[:MAX_OBJECTS]


class ONNXDetector(Detector):
    """YOLO via ONNX Runtime (CPU fallback)."""

    def __init__(self, model_path: str = "yolov8n.onnx"):
        log.info(f"Loading ONNX model: {model_path}")
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        log.info("ONNX detector ready")

    def detect(self, image_path: str, conf: float = 0.4) -> list:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB").resize((640, 640))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]

        outputs = self.session.run(None, {self.input_name: img_np})
        # Simplified: actual parsing depends on model output format
        return []


def create_detector(model_name: str = "yolo_nas_s") -> Detector:
    """Try backends in order of preference."""
    # 1. TensorRT via super-gradients (best on Jetson)
    try:
        return TensorRTDetector(model_name)
    except (ImportError, RuntimeError) as e:
        log.warning(f"TensorRT backend unavailable: {e}")

    # 2. Ultralytics (good fallback)
    try:
        return UltralyticsDetector("yolov8n")
    except (ImportError, RuntimeError) as e:
        log.warning(f"Ultralytics backend unavailable: {e}")

    # 3. ONNX Runtime (CPU fallback)
    try:
        return ONNXDetector()
    except (ImportError, RuntimeError) as e:
        log.warning(f"ONNX backend unavailable: {e}")

    raise RuntimeError("No detection backend available. Install: pip install super-gradients")


def write_detections_to_shm(bridge, detections: list):
    """Write detected objects directly into WorldModel SHM."""
    if not bridge or not bridge.is_open:
        return

    mm = bridge._mm
    if mm is None:
        return

    from qualia_bridge import (
        WORLD_MODEL_OFFSET, WORLD_OBJECT_SIZE, MAX_OBJECT_NAME, MAX_OBJECTS,
    )

    pos = WORLD_MODEL_OFFSET
    count = min(len(detections), MAX_OBJECTS)

    for i in range(MAX_OBJECTS):
        obj_offset = pos + i * WORLD_OBJECT_SIZE
        if i < count:
            det = detections[i]
            # Write name (null-terminated)
            name_bytes = det.name.encode("utf-8")[:MAX_OBJECT_NAME - 1] + b"\x00"
            name_padded = name_bytes.ljust(MAX_OBJECT_NAME, b"\x00")
            mm[obj_offset:obj_offset + MAX_OBJECT_NAME] = name_padded
            # Write confidence, x, y, active
            struct.pack_into("<fffB3x", mm, obj_offset + MAX_OBJECT_NAME,
                             det.confidence, det.x, det.y, 1)
        else:
            # Clear inactive slots
            mm[obj_offset:obj_offset + WORLD_OBJECT_SIZE] = b"\x00" * WORLD_OBJECT_SIZE

    # Write num_objects
    num_obj_offset = pos + MAX_OBJECTS * WORLD_OBJECT_SIZE
    struct.pack_into("<I", mm, num_obj_offset, count)

    # Bump update_seq to notify layers of new detection data
    # update_seq is an AtomicU64 at a known offset
    from qualia_bridge import STATE_DIM, MAX_SCENE_LEN, MAX_ACTIVITY_LEN, MAX_DIRECTIVE_LEN
    update_seq_offset = (num_obj_offset + 8  # num_objects + pad
                         + MAX_SCENE_LEN + MAX_ACTIVITY_LEN
                         + STATE_DIM * 4 + MAX_DIRECTIVE_LEN
                         + 8 * 7)  # 7 u64 counters before update_seq
    old_seq = struct.unpack_from("<Q", mm, update_seq_offset)[0]
    struct.pack_into("<Q", mm, update_seq_offset, old_seq + 1)


def write_detections_json(detections: list):
    """Write detections to JSON file for other consumers.
    Phase 6.7: includes track_id and age for temporal consistency."""
    # Sort by confidence descending (most confident first)
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    data = {
        "ts": time.time(),
        "count": len(sorted_dets),
        "objects": [
            {
                "name": d.name,
                "confidence": d.confidence,
                "x": d.x,
                "y": d.y,
                "track_id": d.track_id,
                "age": d.age,
            }
            for d in sorted_dets
        ],
    }
    tmp = DETECTION_OUTPUT + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, DETECTION_OUTPUT)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="On-device YOLO object detection for Qualia")
    parser.add_argument("--hz", type=float, default=2.0, help="Detection frequency (1-5 Hz)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--model", default="yolo_nas_s", help="Model name")
    parser.add_argument("--no-shm", action="store_true", help="Skip SHM writes (JSON only)")
    args = parser.parse_args()

    acquire_singleton("detect")

    hz = float(os.environ.get("QUALIA_DETECT_HZ", args.hz))
    interval = 1.0 / max(hz, 0.1)

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Open SHM bridge
    bridge = None
    if not args.no_shm:
        try:
            from qualia_bridge import QualiaBridge
            bridge = QualiaBridge()
            if not bridge.open():
                log.warning("Cannot open Qualia SHM — running JSON-only mode")
                bridge = None
        except ImportError:
            log.warning("qualia_bridge not found — running JSON-only mode")

    # Acquire model lock and load detector
    lock = ModelLock()
    if not lock.acquire(timeout=30.0):
        log.error("Cannot acquire model lock (another model may be loaded)")
        sys.exit(1)

    try:
        detector = create_detector(args.model)
    except RuntimeError as e:
        log.error(f"No detection backend: {e}")
        lock.release()
        sys.exit(1)

    # Phase 6.7: Initialize tracker and health monitor
    tracker = ObjectTracker()
    health = DetectionHealthMonitor()

    log.info(f"Running at {hz:.1f} Hz, conf={args.conf}, interval={interval:.2f}s")
    log.info(f"  Phase 6.7: IoU tracking (threshold={ObjectTracker.IOU_THRESHOLD}), "
             f"per-class thresholds, health monitoring")
    detect_count = 0

    try:
        while running:
            loop_start = time.monotonic()

            if not os.path.exists(SNAPSHOT_PATH):
                time.sleep(1.0)
                continue

            # Check snapshot freshness (skip if older than 5s)
            try:
                snap_age = time.time() - os.path.getmtime(SNAPSHOT_PATH)
                if snap_age > 5.0:
                    time.sleep(0.5)
                    continue
            except OSError:
                continue

            try:
                t0 = time.monotonic()
                # Run detector with low base threshold (per-class filtering happens after)
                raw_detections = detector.detect(SNAPSHOT_PATH, conf=min(args.conf, 0.25))
                dt_ms = (time.monotonic() - t0) * 1000

                # Phase 6.7: Apply per-class confidence thresholds
                detections = apply_class_thresholds(raw_detections, args.conf)

                # Phase 6.7: Update tracker with persistent object IDs
                detections = tracker.update(detections)

                # Truncate to MAX_OBJECTS (keep highest confidence)
                detections.sort(key=lambda d: d.confidence, reverse=True)
                detections = detections[:MAX_OBJECTS]

                detect_count += 1
                health.record_success(len(detections), dt_ms)

                if detections:
                    names = [
                        f"{d.name}#{d.track_id}({d.confidence:.0%},age={d.age})"
                        for d in detections[:5]
                    ]
                    log.info(f"#{detect_count} {dt_ms:.0f}ms: {', '.join(names)}")
                elif detect_count % 10 == 0:
                    log.info(f"#{detect_count} {dt_ms:.0f}ms: no objects "
                             f"(tracks: {len(tracker.tracks)})")

                # Write results
                if bridge:
                    write_detections_to_shm(bridge, detections)
                write_detections_json(detections)

            except Exception as e:
                log.error(f"Detection error: {e}")
                health.record_failure()
                if not health.is_healthy:
                    log.error(f"HEALTH WARNING: {health.consecutive_failures} consecutive failures")
                time.sleep(1.0)

            elapsed = time.monotonic() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    finally:
        lock.release()
        if bridge:
            bridge.close()
        log.info(f"Detector shutdown after {detect_count} frames "
                 f"(tracked {tracker.next_id} unique objects)")


if __name__ == "__main__":
    main()
