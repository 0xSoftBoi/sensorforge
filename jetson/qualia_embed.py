#!/usr/bin/env python3
"""
Qualia Local Embeddings — on-device text embeddings via ONNX.

Phase 2.3: Replace cloud Gemini embeddings with local all-MiniLM-L6-v2
(~80MB ONNX, runs on CPU). Generates 384-dim embeddings from detection
descriptions, pools to 64-dim for the Qualia state space.

Phase 6.6: Upgraded embedding pipeline:
  - Confidence-weighted object descriptions (high-conf objects emphasized)
  - Raw spatial coordinates in text (x=0.35, y=0.52) for richer encoding
  - Temporal context buffer: embed current + recent 2 frames for change awareness
  - Audio-visual fusion: incorporate audio VAD/features when available

Usage:
    python3 qualia_embed.py [--interval 2.0] [--model all-MiniLM-L6-v2]
"""

import argparse
import json
import logging
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s embed: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Model cache directory
MODEL_DIR = Path.home() / ".cache" / "qualia" / "models"
DETECTION_JSON = "/tmp/qualia_detections.json"
STATE_DIM = 64


class LocalEmbedder:
    """Text embedder using all-MiniLM-L6-v2 ONNX model on CPU."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.session = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load ONNX model and tokenizer."""
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            raise RuntimeError(
                "Install: pip install onnxruntime tokenizers\n"
                "  or: pip install sentence-transformers  (includes everything)"
            )

        model_dir = MODEL_DIR / self.model_name
        onnx_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        if not onnx_path.exists():
            log.info(f"Downloading {self.model_name} ONNX model...")
            self._download_model(model_dir)

        if not onnx_path.exists():
            raise RuntimeError(f"Model not found at {onnx_path}")

        # CPU-only inference
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        if tokenizer_path.exists():
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            # Fallback: use transformers tokenizer
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    f"sentence-transformers/{self.model_name}",
                    cache_dir=str(MODEL_DIR),
                )
            except ImportError:
                raise RuntimeError("Install: pip install transformers")

        log.info(f"Loaded {self.model_name} ONNX model ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    def _download_model(self, model_dir: Path):
        """Download ONNX model from HuggingFace."""
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download

            repo_id = f"sentence-transformers/{self.model_name}"
            for fname in ["model.onnx", "tokenizer.json", "config.json"]:
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=fname,
                        local_dir=str(model_dir),
                    )
                except Exception as e:
                    log.warning(f"Could not download {fname}: {e}")
        except ImportError:
            log.warning("huggingface_hub not installed, trying sentence-transformers export...")
            self._export_from_sentence_transformers(model_dir)

    def _export_from_sentence_transformers(self, model_dir: Path):
        """Export model to ONNX from sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            model = SentenceTransformer(f"sentence-transformers/{self.model_name}")
            # Save tokenizer
            model.tokenizer.save_pretrained(str(model_dir))

            # Export to ONNX
            dummy_input = model.tokenizer(
                "test", return_tensors="pt", padding=True, truncation=True
            )
            torch.onnx.export(
                model[0].auto_model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                str(model_dir / "model.onnx"),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                },
                opset_version=14,
            )
            log.info("Exported model to ONNX")
        except Exception as e:
            raise RuntimeError(f"Cannot export model: {e}")

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a text string. Returns 384-dim vector."""
        if hasattr(self.tokenizer, "encode"):
            # tokenizers.Tokenizer
            encoded = self.tokenizer.encode(text)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        else:
            # transformers tokenizer
            encoded = self.tokenizer(
                text, return_tensors="np", padding=True, truncation=True, max_length=128
            )
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)

        # Run ONNX inference
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Handle optional token_type_ids
        input_names = [inp.name for inp in self.session.get_inputs()]
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, inputs)

        # Mean pooling over token dimension
        token_embeddings = outputs[0]  # (1, seq_len, hidden_dim)
        mask_expanded = attention_mask[..., np.newaxis].astype(np.float32)
        sum_embeddings = (token_embeddings * mask_expanded).sum(axis=1)
        sum_mask = mask_expanded.sum(axis=1).clip(min=1e-9)
        embedding = (sum_embeddings / sum_mask)[0]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_to_64(self, text: str) -> np.ndarray:
        """Generate embedding and pool to 64 dimensions for Qualia state space."""
        full = self.embed(text)  # 384-dim
        return pool_to_64(full)


def pool_to_64(embedding: np.ndarray) -> np.ndarray:
    """Average-pool a high-dim embedding down to 64 dimensions."""
    dim = len(embedding)
    if dim == STATE_DIM:
        return embedding
    if dim < STATE_DIM:
        result = np.zeros(STATE_DIM, dtype=np.float32)
        result[:dim] = embedding
        return result

    # Average pool: divide into 64 bins
    bin_size = dim / STATE_DIM
    result = np.zeros(STATE_DIM, dtype=np.float32)
    for i in range(STATE_DIM):
        start = int(i * bin_size)
        end = int((i + 1) * bin_size)
        result[i] = embedding[start:end].mean()

    # Re-normalize
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm

    return result


AUDIO_FEATURES_FILE = "/tmp/qualia_audio_features.json"


def build_scene_text(detections: list, audio_context: str = "") -> str:
    """Build a rich text description from detections for embedding.

    Phase 6.6 improvements:
    - Include raw (x, y) coordinates for precise spatial encoding
    - Sort by confidence (most confident first — embedding models weight early tokens)
    - Include spatial relationships between objects
    - Incorporate audio context (VAD, ambient description)
    """
    if not detections:
        base = "empty scene, no objects detected"
        return f"{base}. {audio_context}" if audio_context else base

    # Sort by confidence descending — embedding models give more weight to early tokens
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)

    obj_strs = []
    for d in sorted_dets:
        x, y = d.get("x", 0.5), d.get("y", 0.5)
        conf = d.get("confidence", 0)

        # Spatial description with both words AND raw coordinates
        h_pos = "left" if x < 0.33 else ("right" if x > 0.66 else "center")
        v_pos = "upper" if y < 0.33 else ("lower" if y > 0.66 else "middle")

        obj_strs.append(
            f"{d['name']} at {h_pos}-{v_pos} (x={x:.2f}, y={y:.2f}, {conf:.0%})"
        )

    # Add spatial relationships for top objects
    relationships = []
    if len(sorted_dets) >= 2:
        for i in range(min(len(sorted_dets) - 1, 3)):
            a, b = sorted_dets[i], sorted_dets[i + 1]
            dx = a.get("x", 0.5) - b.get("x", 0.5)
            if abs(dx) > 0.2:
                rel = "left of" if dx < 0 else "right of"
                relationships.append(f"{a['name']} is {rel} {b['name']}")

    parts = [f"Scene with {len(detections)} objects: {', '.join(obj_strs)}"]
    if relationships:
        parts.append(f"Layout: {'; '.join(relationships)}")
    if audio_context:
        parts.append(audio_context)

    return ". ".join(parts)


def get_audio_context() -> str:
    """Read audio features file and build audio context string for embedding."""
    try:
        if not os.path.exists(AUDIO_FEATURES_FILE):
            return ""
        with open(AUDIO_FEATURES_FILE) as f:
            data = json.load(f)
        if time.time() - data.get("ts", 0) > 5.0:
            return ""
        vad = data.get("vad", False)
        rms = data.get("rms", 0)
        if vad:
            return f"Audio: speech detected (loudness {rms:.3f})"
        elif rms > 0.01:
            return f"Audio: ambient sound (loudness {rms:.3f})"
        else:
            return "Audio: silence"
    except (json.JSONDecodeError, OSError):
        return ""


def write_embedding_to_shm(bridge, embedding: np.ndarray):
    """Write 64-dim embedding directly to WorldModel.scene_embedding in SHM."""
    if not bridge or not bridge.is_open or bridge._mm is None:
        return

    from qualia_bridge import (
        WORLD_MODEL_OFFSET, MAX_OBJECTS, WORLD_OBJECT_SIZE,
        MAX_SCENE_LEN, MAX_ACTIVITY_LEN, STATE_DIM,
    )

    # scene_embedding offset = WorldModel start + objects + num_objects + pad + scene + activity
    emb_offset = (WORLD_MODEL_OFFSET
                  + MAX_OBJECTS * WORLD_OBJECT_SIZE + 8  # objects + num_objects + pad
                  + MAX_SCENE_LEN + MAX_ACTIVITY_LEN)    # scene + activity

    # Write 64 floats
    for i in range(min(len(embedding), STATE_DIM)):
        struct.pack_into("<f", bridge._mm, emb_offset + i * 4, float(embedding[i]))

    # Bump update_seq
    from qualia_bridge import MAX_DIRECTIVE_LEN
    update_seq_offset = (emb_offset + STATE_DIM * 4 + MAX_DIRECTIVE_LEN + 8 * 7)
    old_seq = struct.unpack_from("<Q", bridge._mm, update_seq_offset)[0]
    struct.pack_into("<Q", bridge._mm, update_seq_offset, old_seq + 1)


def main():
    parser = argparse.ArgumentParser(description="Local text embeddings for Qualia")
    parser.add_argument("--interval", type=float, default=2.0, help="Update interval (seconds)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name")
    parser.add_argument("--no-shm", action="store_true", help="Skip SHM writes")
    args = parser.parse_args()

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Load embedder
    try:
        embedder = LocalEmbedder(args.model)
    except RuntimeError as e:
        log.error(f"Cannot load embedding model: {e}")
        sys.exit(1)

    # Quick self-test
    t0 = time.monotonic()
    test_emb = embedder.embed("test sentence")
    dt = (time.monotonic() - t0) * 1000
    log.info(f"Self-test: {len(test_emb)}d embedding in {dt:.0f}ms")

    # Open SHM bridge
    bridge = None
    if not args.no_shm:
        try:
            from qualia_bridge import QualiaBridge
            bridge = QualiaBridge()
            if not bridge.open():
                log.warning("Cannot open Qualia SHM")
                bridge = None
        except ImportError:
            log.warning("qualia_bridge not found")

    log.info(f"Running every {args.interval:.1f}s (Phase 6.6: temporal buffer + audio fusion)")
    update_count = 0
    prev_text = ""

    # Phase 6.6: Temporal embedding buffer — blend current with recent frames
    TEMPORAL_BUFFER_SIZE = 3
    temporal_embeddings: list = []  # list of recent 64-dim embeddings
    TEMPORAL_WEIGHTS = np.array([0.1, 0.25, 0.65])  # oldest → newest weighting

    try:
        while running:
            loop_start = time.monotonic()

            # Read latest detections from qualia_detect.py
            detections = []
            try:
                if os.path.exists(DETECTION_JSON):
                    with open(DETECTION_JSON) as f:
                        data = json.load(f)
                    detections = data.get("objects", [])
                    # Skip if stale (> 10s old)
                    if time.time() - data.get("ts", 0) > 10.0:
                        detections = []
            except (json.JSONDecodeError, OSError):
                pass

            # Also try reading scene from SHM world model
            scene_text = ""
            if bridge and bridge.is_open:
                try:
                    wm = bridge.read_world_model()
                    if wm.scene:
                        scene_text = wm.scene
                except Exception:
                    pass

            # Phase 6.6: Get audio context for multi-modal fusion
            audio_ctx = get_audio_context()

            # Build embedding text from detections + scene + audio
            det_text = build_scene_text(detections, audio_context=audio_ctx)
            full_text = f"{det_text}. {scene_text}" if scene_text else det_text

            # Skip if text hasn't changed
            if full_text == prev_text and update_count > 0:
                elapsed = time.monotonic() - loop_start
                if elapsed < args.interval:
                    time.sleep(args.interval - elapsed)
                continue

            prev_text = full_text

            # Generate embedding
            try:
                t0 = time.monotonic()
                current_emb = embedder.embed_to_64(full_text)
                dt_ms = (time.monotonic() - t0) * 1000

                # Phase 6.6: Temporal blending — weighted average of recent embeddings
                # This gives the embedding a sense of "what's been happening" not just "what is now"
                temporal_embeddings.append(current_emb.copy())
                if len(temporal_embeddings) > TEMPORAL_BUFFER_SIZE:
                    temporal_embeddings.pop(0)

                if len(temporal_embeddings) >= TEMPORAL_BUFFER_SIZE:
                    w = TEMPORAL_WEIGHTS
                    embedding_64 = sum(
                        w[i] * temporal_embeddings[i]
                        for i in range(TEMPORAL_BUFFER_SIZE)
                    )
                    # Re-normalize after blending
                    norm = np.linalg.norm(embedding_64)
                    if norm > 0:
                        embedding_64 = embedding_64 / norm
                else:
                    embedding_64 = current_emb

                update_count += 1
                if update_count % 10 == 1 or detections:
                    norm = float(np.linalg.norm(embedding_64))
                    log.info(
                        f"#{update_count} {dt_ms:.0f}ms: {len(detections)} obj, "
                        f"norm={norm:.3f}, buf={len(temporal_embeddings)}/{TEMPORAL_BUFFER_SIZE}, "
                        f"text={full_text[:60]}"
                    )

                # Write to SHM
                if bridge:
                    write_embedding_to_shm(bridge, embedding_64)

            except Exception as e:
                log.error(f"Embedding error: {e}")

            elapsed = time.monotonic() - loop_start
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)

    finally:
        if bridge:
            bridge.close()
        log.info(f"Embedder shutdown after {update_count} updates")


if __name__ == "__main__":
    main()
