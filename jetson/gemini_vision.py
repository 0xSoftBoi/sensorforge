"""
Unified Gemini Vision Service — shared by voice assistant and Qualia.

Captures frames on a schedule, answers Qualia layer questions (LORE),
answers voice assistant queries ("what do you see?"), and injects
scene embeddings into Qualia's WorldModel via SHM.

Deduplicates API calls by caching results for a configurable TTL.

Usage:
    python3 gemini_vision.py                # Run as service
    python3 gemini_vision.py --once         # Single capture + describe
    python3 gemini_vision.py --budget 100   # Set max API calls
"""

import base64
import json
import logging
import os
import subprocess
import threading
import time
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger("gemini-vision")

# ── Config ───────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBEDDING_MODEL}:embedContent"

CAMERA_DEVICE = os.environ.get("CAMERA_DEVICE", "/dev/video0")
CAPTURE_W = 640
CAPTURE_H = 480

DEFAULT_INTERVAL_SECS = 30
DEFAULT_MAX_CALLS = 50
CACHE_TTL_SECS = 10  # Don't re-call Gemini within this window


# ── Cache ────────────────────────────────────────────────────────

class VisionCache:
    """Thread-safe cache for the most recent vision result."""

    def __init__(self, ttl: float = CACHE_TTL_SECS):
        self._lock = threading.Lock()
        self.ttl = ttl
        self.scene: str = ""
        self.activity: str = ""
        self.objects: list = []
        self.embedding: list = []
        self.timestamp: float = 0
        self.calls_made: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.embed_tokens: int = 0

    def is_fresh(self) -> bool:
        return (time.time() - self.timestamp) < self.ttl

    def update(self, scene: str, activity: str, objects: list,
               embedding: list, input_tok: int, output_tok: int, embed_tok: int):
        with self._lock:
            self.scene = scene
            self.activity = activity
            self.objects = objects
            self.embedding = embedding
            self.timestamp = time.time()
            self.calls_made += 1
            self.input_tokens += input_tok
            self.output_tokens += output_tok
            self.embed_tokens += embed_tok

    def get_summary(self) -> str:
        with self._lock:
            if not self.scene:
                return "No vision data yet."
            obj_str = ", ".join(f"{o['name']}({o['confidence']:.0%})" for o in self.objects[:5])
            return f"Scene: {self.scene}. Activity: {self.activity}. Objects: {obj_str or 'none'}."

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls_made,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "embed_tokens": self.embed_tokens,
                "age_secs": round(time.time() - self.timestamp, 1) if self.timestamp else None,
            }


# Global cache
cache = VisionCache()


# ── Frame Capture ────────────────────────────────────────────────

def capture_frame_jpeg(device: str = CAMERA_DEVICE, timeout: float = 10.0) -> Optional[bytes]:
    """Capture a single JPEG frame from camera via ffmpeg."""
    try:
        # Detect platform for input format
        import sys
        if sys.platform == "darwin":
            input_args = ["-f", "avfoundation", "-framerate", "30",
                          "-video_size", f"{CAPTURE_W}x{CAPTURE_H}", "-i", "0"]
        else:
            input_args = ["-f", "v4l2", "-video_size", f"{CAPTURE_W}x{CAPTURE_H}",
                          "-i", device]

        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            + input_args
            + ["-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "-q:v", "5", "pipe:1"],
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0 or not result.stdout:
            log.error("ffmpeg capture failed: %s", result.stderr.decode()[:200])
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        log.error("ffmpeg capture timed out (%.0fs)", timeout)
        return None
    except FileNotFoundError:
        log.error("ffmpeg not found")
        return None


# ── Gemini API Calls ─────────────────────────────────────────────

def call_gemini_vision(image_b64: str, directive: str = "",
                       questions: list = None) -> Optional[dict]:
    """Call Gemini Vision API. Returns parsed response dict or None."""
    if not GEMINI_API_KEY:
        return None

    prompt = (
        "You are the visual cortex of an autonomous robot. "
        "Analyze this image and respond with ONLY valid JSON (no markdown):\n"
        '{"scene": "<one sentence>", "activity": "<what is happening>", '
        '"objects": [{"name": "<obj>", "confidence": 0.0-1.0, "x": 0.0-1.0, "y": 0.0-1.0}]'
    )

    if questions:
        prompt += ', "lore_answers": ['
        for i, q in enumerate(questions):
            if i > 0:
                prompt += ", "
            prompt += f'"<answer: {q}>"'
        prompt += "]"

    prompt += "}"
    if directive:
        prompt += f"\n\nDirective: {directive}"

    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.1},
    }).encode()

    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    req = Request(url, data=body, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        clean = text.strip().strip("`").removeprefix("json").strip()

        result = json.loads(clean)
        result["_usage"] = {
            "input": data.get("usageMetadata", {}).get("promptTokenCount", 0),
            "output": data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
        }
        return result

    except (URLError, json.JSONDecodeError, KeyError) as e:
        log.error("Gemini vision error: %s", e)
        return None


def call_gemini_embedding(text: str) -> Optional[tuple]:
    """Call Gemini Embedding API. Returns (embedding_list, token_count) or None."""
    if not GEMINI_API_KEY:
        return None

    body = json.dumps({
        "model": f"models/{GEMINI_EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 64,
    }).encode()

    url = f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}"
    req = Request(url, data=body, headers={"Content-Type": "application/json"})

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        values = data["embedding"]["values"]
        tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
        return ([float(v) for v in values], tokens)

    except (URLError, json.JSONDecodeError, KeyError) as e:
        log.error("Gemini embedding error: %s", e)
        return None


# ── High-level API (used by voice assistant) ─────────────────────

def describe_scene(force: bool = False) -> str:
    """Get a description of the current scene. Uses cache if fresh."""
    if cache.is_fresh() and not force:
        return cache.get_summary()

    jpeg = capture_frame_jpeg()
    if jpeg is None:
        return "Camera not available."

    b64 = base64.b64encode(jpeg).decode()
    result = call_gemini_vision(b64)
    if result is None:
        return "Gemini API not available."

    usage = result.get("_usage", {})
    embed_result = call_gemini_embedding(
        f"{result.get('scene', '')} {result.get('activity', '')}"
    )

    embed_tokens = 0
    embedding = []
    if embed_result:
        embedding, embed_tokens = embed_result

    cache.update(
        scene=result.get("scene", ""),
        activity=result.get("activity", ""),
        objects=result.get("objects", []),
        embedding=embedding,
        input_tok=usage.get("input", 0),
        output_tok=usage.get("output", 0),
        embed_tok=embed_tokens,
    )

    # Inject into Qualia SHM if available
    _inject_into_qualia(result, embedding)

    return cache.get_summary()


def _inject_into_qualia(result: dict, embedding: list):
    """Write vision results into Qualia's WorldModel SHM."""
    try:
        from qualia_bridge import get_bridge
        bridge = get_bridge()
        if bridge is None:
            return

        # Write scene embedding if we have one
        if embedding and len(embedding) >= 64:
            import struct
            mm = bridge._mm
            if mm is None:
                return

            from qualia_bridge import (
                WORLD_MODEL_OFFSET, MAX_OBJECTS, WORLD_OBJECT_SIZE,
                MAX_SCENE_LEN, MAX_ACTIVITY_LEN, STATE_DIM,
            )

            # Write scene text
            scene_offset = WORLD_MODEL_OFFSET + MAX_OBJECTS * WORLD_OBJECT_SIZE + 8
            scene_bytes = result.get("scene", "").encode("utf-8")[:MAX_SCENE_LEN - 1] + b"\x00"
            padded = scene_bytes.ljust(MAX_SCENE_LEN, b"\x00")
            mm[scene_offset:scene_offset + MAX_SCENE_LEN] = padded

            # Write activity
            act_offset = scene_offset + MAX_SCENE_LEN
            act_bytes = result.get("activity", "").encode("utf-8")[:MAX_ACTIVITY_LEN - 1] + b"\x00"
            act_padded = act_bytes.ljust(MAX_ACTIVITY_LEN, b"\x00")
            mm[act_offset:act_offset + MAX_ACTIVITY_LEN] = act_padded

            # Write embedding
            emb_offset = act_offset + MAX_ACTIVITY_LEN
            emb_data = struct.pack(f"<{STATE_DIM}f", *embedding[:STATE_DIM])
            mm[emb_offset:emb_offset + STATE_DIM * 4] = emb_data

            log.debug("Injected vision into Qualia SHM")

    except Exception as e:
        log.debug("Qualia injection failed: %s", e)


# ── Service Loop ─────────────────────────────────────────────────

def run_service(interval: int = DEFAULT_INTERVAL_SECS, max_calls: int = DEFAULT_MAX_CALLS):
    """Run as a periodic vision service."""
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set")
        return

    log.info("Gemini vision service: interval=%ds, budget=%d calls", interval, max_calls)

    while cache.calls_made < max_calls:
        describe_scene(force=True)
        stats = cache.get_stats()
        log.info(
            "Vision #%d: %s (tokens: in=%d out=%d emb=%d)",
            stats["calls"], cache.scene[:60],
            stats["input_tokens"], stats["output_tokens"], stats["embed_tokens"],
        )
        time.sleep(interval)

    log.info("Budget exhausted (%d calls). Exiting.", max_calls)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Unified Gemini Vision Service")
    parser.add_argument("--once", action="store_true", help="Single capture + describe")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECS)
    parser.add_argument("--budget", type=int, default=DEFAULT_MAX_CALLS)
    args = parser.parse_args()

    if args.once:
        print(describe_scene(force=True))
        print(f"Stats: {cache.get_stats()}")
    else:
        run_service(interval=args.interval, max_calls=args.budget)
