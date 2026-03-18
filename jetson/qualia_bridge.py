"""
Qualia SHM Bridge — Python reader for Qualia's shared memory.

Reads all 7 layer beliefs, WorldModel, ThoughtBuffer, and LoreBuffer
directly from /dev/shm/qualia_body using mmap + struct.unpack.

Layout matches repr(C) types in qualia/crates/types/src/lib.rs exactly.
Offsets verified against qualia/crates/shm/src/lib.rs.
"""

import ctypes
import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Constants (must match Rust crate exactly) ────────────────────────

STATE_DIM = 64
NUM_LAYERS = 7
WEIGHT_COUNT = STATE_DIM * STATE_DIM  # 4096
SHM_MAGIC = 0x5155414C3141454E  # "QUAL1AEN"

# SHM layout offsets (from qualia-shm)
SHM_SIZE = 64 * 1024 * 1024  # 64 MiB
HEADER_SIZE = 48
LAYER_SLOTS_OFFSET = 4096
LAYER_SLOT_SIZE = 19136
LEDGER_OFFSET = 138048
WORLD_MODEL_OFFSET = 16915264
THOUGHT_BUFFER_OFFSET = 16917256
LORE_BUFFER_OFFSET = 17060624

# Struct sizes
BELIEF_SLOT_SIZE = 1088
WORLD_OBJECT_SIZE = 48
THOUGHT_ENTRY_SIZE = 280
LORE_ENTRY_SIZE = 800

# Sub-struct counts
MAX_OBJECTS = 16
MAX_SCENE_LEN = 512
MAX_DIRECTIVE_LEN = 256
MAX_ACTIVITY_LEN = 128
MAX_OBJECT_NAME = 32
MAX_THOUGHT_LEN = 256
MAX_THOUGHTS = 512
MAX_LORE_TEXT = 512
MAX_LORE_QUESTION = 256
MAX_LORE_ENTRIES = 128
MAX_QUESTION_TEXT = 256


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class BeliefSlot:
    mean: list  # [f32; 64]
    precision: list  # [f32; 64]
    vfe: float
    prediction: list  # [f32; 64]
    residual: list  # [f32; 64]
    challenge_vfe: float
    confirm_streak: int
    compression: int
    layer: int
    timestamp_ns: int
    cycle_us: int


@dataclass
class WorldObject:
    name: str
    confidence: float
    x: float
    y: float
    active: bool


@dataclass
class WorldModel:
    objects: list  # [WorldObject]
    num_objects: int
    scene: str
    activity: str
    scene_embedding: list  # [f32; 64]
    directive: str
    last_vision_ns: int
    last_llm_ns: int
    llm_call_count: int
    vision_frame_count: int
    gemini_input_tokens: int
    gemini_output_tokens: int
    gemini_embedding_tokens: int
    update_seq: int


@dataclass
class ThoughtEntry:
    text: str
    layer: int
    kind: int  # 0=observe, 1=predict, 2=surprise, 3=learn, 4=resolve, 5=escalate
    vfe: float
    timestamp_ns: int
    seq: int


@dataclass
class LoreEntry:
    question: str
    answer: str
    layer: int
    reason: int
    embedding_delta: float
    effectiveness: float
    timestamp_ns: int
    seq: int


@dataclass
class LayerSummary:
    """Compact summary of a layer's state for voice assistant tools."""
    layer_id: int
    vfe: float
    compression: int
    confirm_streak: int
    challenge_vfe: float
    cycle_us: int
    is_challenged: bool
    mean_magnitude: float  # L2 norm of mean vector


# ── Thought kind labels ──────────────────────────────────────────────

THOUGHT_KINDS = {
    0: "observe",
    1: "predict",
    2: "surprise",
    3: "learn",
    4: "resolve",
    5: "escalate",
}


# ── SHM Reader ───────────────────────────────────────────────────────

class QualiaBridge:
    """Read-only bridge to Qualia's shared memory region."""

    def __init__(self, shm_name: str = "/qualia_body"):
        self.shm_name = shm_name
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None

    def open(self) -> bool:
        """Open the shared memory region. Returns True on success."""
        try:
            shm_path = f"/dev/shm{self.shm_name}"
            if not os.path.exists(shm_path):
                # macOS uses shm_open which maps to /tmp/
                # Try POSIX shm_open via ctypes
                return self._open_posix()

            self._fd = os.open(shm_path, os.O_RDWR)
            self._mm = mmap.mmap(self._fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

            # Verify magic
            magic = struct.unpack_from("<Q", self._mm, 0)[0]
            if magic != SHM_MAGIC:
                self.close()
                return False
            return True
        except (OSError, ValueError):
            return False

    def _open_posix(self) -> bool:
        """Open via POSIX shm_open (works on macOS and Linux)."""
        try:
            libc = ctypes.CDLL("libc.dylib" if os.uname().sysname == "Darwin" else "libc.so.6")

            O_RDWR = 0x0002
            fd = libc.shm_open(
                self.shm_name.encode(),
                O_RDWR,
                0o600,
            )
            if fd < 0:
                return False

            self._fd = fd
            self._mm = mmap.mmap(fd, SHM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

            magic = struct.unpack_from("<Q", self._mm, 0)[0]
            if magic != SHM_MAGIC:
                self.close()
                return False
            return True
        except (OSError, ValueError, AttributeError):
            return False

    def close(self):
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    @property
    def is_open(self) -> bool:
        return self._mm is not None

    # ── BeliefSlot reader ────────────────────────────────────────────

    def _read_belief_slot(self, offset: int) -> BeliefSlot:
        """Read a BeliefSlot at the given byte offset."""
        mm = self._mm
        pos = offset

        # mean: [f32; 64] = 256 bytes
        mean = list(struct.unpack_from("<64f", mm, pos))
        pos += 256

        # precision: [f32; 64] = 256 bytes
        precision = list(struct.unpack_from("<64f", mm, pos))
        pos += 256

        # vfe: f32
        vfe = struct.unpack_from("<f", mm, pos)[0]
        pos += 4

        # prediction: [f32; 64] = 256 bytes
        prediction = list(struct.unpack_from("<64f", mm, pos))
        pos += 256

        # residual: [f32; 64] = 256 bytes
        residual = list(struct.unpack_from("<64f", mm, pos))
        pos += 256

        # challenge_vfe: f32, confirm_streak: u32, compression: u8, layer: u8, _pad: [u8;2]
        challenge_vfe, confirm_streak, compression, layer = struct.unpack_from("<fIBB", mm, pos)
        pos += 4 + 4 + 1 + 1 + 2  # includes 2-byte pad

        # timestamp_ns: u64, cycle_us: u32, _pad2: [u8;4]
        timestamp_ns, cycle_us = struct.unpack_from("<QI", mm, pos)

        return BeliefSlot(
            mean=mean,
            precision=precision,
            vfe=vfe,
            prediction=prediction,
            residual=residual,
            challenge_vfe=challenge_vfe,
            confirm_streak=confirm_streak,
            compression=compression,
            layer=layer,
            timestamp_ns=timestamp_ns,
            cycle_us=cycle_us,
        )

    # ── Layer readers ────────────────────────────────────────────────

    def read_layer_belief(self, layer: int) -> BeliefSlot:
        """Read the current (front) belief buffer for a layer."""
        assert 0 <= layer < NUM_LAYERS
        slot_offset = LAYER_SLOTS_OFFSET + layer * LAYER_SLOT_SIZE

        # Double buffer: read write_idx to find the front buffer.
        # write_idx is after: 2 * BeliefSlot + weights + bias
        # = 2 * 1088 + 4096*4 + 64*4 = 2176 + 16384 + 256 = 18816
        write_idx_offset = slot_offset + 2 * BELIEF_SLOT_SIZE + WEIGHT_COUNT * 4 + STATE_DIM * 4
        # write_idx is AtomicUsize = 8 bytes on 64-bit
        write_idx = struct.unpack_from("<Q", self._mm, write_idx_offset)[0] & 1

        # Front buffer is at write_idx
        belief_offset = slot_offset + write_idx * BELIEF_SLOT_SIZE
        return self._read_belief_slot(belief_offset)

    def read_layer_weights(self, layer: int) -> tuple:
        """Read a layer's weight matrix and bias. Returns (weights_64x64, bias_64)."""
        assert 0 <= layer < NUM_LAYERS
        slot_offset = LAYER_SLOTS_OFFSET + layer * LAYER_SLOT_SIZE
        weights_offset = slot_offset + 2 * BELIEF_SLOT_SIZE

        weights = list(struct.unpack_from(f"<{WEIGHT_COUNT}f", self._mm, weights_offset))
        bias = list(struct.unpack_from(f"<{STATE_DIM}f", self._mm, weights_offset + WEIGHT_COUNT * 4))
        return weights, bias

    def read_all_layers(self) -> list:
        """Read summary of all 7 layers."""
        summaries = []
        for i in range(NUM_LAYERS):
            b = self.read_layer_belief(i)
            mag = sum(v * v for v in b.mean) ** 0.5
            summaries.append(LayerSummary(
                layer_id=i,
                vfe=b.vfe,
                compression=b.compression,
                confirm_streak=b.confirm_streak,
                challenge_vfe=b.challenge_vfe,
                cycle_us=b.cycle_us,
                is_challenged=b.challenge_vfe > 0.05,  # approximate threshold
                mean_magnitude=mag,
            ))
        return summaries

    # ── WorldModel reader ────────────────────────────────────────────

    def read_world_model(self) -> WorldModel:
        """Read the WorldModel from SHM."""
        mm = self._mm
        pos = WORLD_MODEL_OFFSET

        # objects: [WorldObject; 16]
        objects = []
        for _ in range(MAX_OBJECTS):
            name_bytes = mm[pos:pos + MAX_OBJECT_NAME]
            name = _read_cstr(name_bytes)
            pos += MAX_OBJECT_NAME
            confidence, x, y, active = struct.unpack_from("<fffB", mm, pos)
            pos += 4 + 4 + 4 + 1 + 3  # includes 3-byte pad
            if active:
                objects.append(WorldObject(name=name, confidence=confidence, x=x, y=y, active=bool(active)))

        # num_objects: u32, _pad0: u32
        num_objects = struct.unpack_from("<I", mm, pos)[0]
        pos += 8

        # scene: [u8; 512]
        scene = _read_cstr(mm[pos:pos + MAX_SCENE_LEN])
        pos += MAX_SCENE_LEN

        # activity: [u8; 128]
        activity = _read_cstr(mm[pos:pos + MAX_ACTIVITY_LEN])
        pos += MAX_ACTIVITY_LEN

        # scene_embedding: [f32; 64]
        scene_embedding = list(struct.unpack_from("<64f", mm, pos))
        pos += STATE_DIM * 4

        # directive: [u8; 256]
        directive = _read_cstr(mm[pos:pos + MAX_DIRECTIVE_LEN])
        pos += MAX_DIRECTIVE_LEN

        # Timestamps and counters
        (last_vision_ns, last_llm_ns, llm_call_count, vision_frame_count,
         gemini_input_tokens, gemini_output_tokens, gemini_embedding_tokens,
         update_seq) = struct.unpack_from("<8Q", mm, pos)

        return WorldModel(
            objects=objects,
            num_objects=num_objects,
            scene=scene,
            activity=activity,
            scene_embedding=scene_embedding,
            directive=directive,
            last_vision_ns=last_vision_ns,
            last_llm_ns=last_llm_ns,
            llm_call_count=llm_call_count,
            vision_frame_count=vision_frame_count,
            gemini_input_tokens=gemini_input_tokens,
            gemini_output_tokens=gemini_output_tokens,
            gemini_embedding_tokens=gemini_embedding_tokens,
            update_seq=update_seq,
        )

    # ── Write directive into WorldModel ──────────────────────────────

    def write_directive(self, text: str):
        """Inject a directive into the WorldModel's directive field."""
        if not self._mm:
            return
        # directive offset = WORLD_MODEL_OFFSET + objects(16*48) + num_objects(4) + pad(4)
        #                   + scene(512) + activity(128) + embedding(256) = offset to directive
        directive_offset = WORLD_MODEL_OFFSET + MAX_OBJECTS * WORLD_OBJECT_SIZE + 8 + MAX_SCENE_LEN + MAX_ACTIVITY_LEN + STATE_DIM * 4
        encoded = text.encode("utf-8")[:MAX_DIRECTIVE_LEN - 1] + b"\x00"
        padded = encoded.ljust(MAX_DIRECTIVE_LEN, b"\x00")
        self._mm[directive_offset:directive_offset + MAX_DIRECTIVE_LEN] = padded

    # ── ThoughtBuffer reader ─────────────────────────────────────────

    def read_recent_thoughts(self, count: int = 20) -> list:
        """Read the N most recent thoughts from the ring buffer."""
        mm = self._mm
        pos = THOUGHT_BUFFER_OFFSET

        # write_seq: AtomicU64
        write_seq = struct.unpack_from("<Q", mm, pos)[0]
        pos += 8  # skip write_seq

        if write_seq == 0:
            return []

        thoughts = []
        start = max(0, write_seq - count)

        for seq in range(start, write_seq):
            idx = seq % MAX_THOUGHTS
            entry_offset = pos + idx * THOUGHT_ENTRY_SIZE
            text = _read_cstr(mm[entry_offset:entry_offset + MAX_THOUGHT_LEN])
            entry_pos = entry_offset + MAX_THOUGHT_LEN
            layer, kind = struct.unpack_from("<BB", mm, entry_pos)
            entry_pos += 4  # 2 bytes + 2 pad
            vfe = struct.unpack_from("<f", mm, entry_pos)[0]
            entry_pos += 4
            timestamp_ns, entry_seq = struct.unpack_from("<QQ", mm, entry_pos)

            if entry_seq == seq:
                thoughts.append(ThoughtEntry(
                    text=text, layer=layer, kind=kind,
                    vfe=vfe, timestamp_ns=timestamp_ns, seq=entry_seq,
                ))

        return thoughts

    # ── LoreBuffer reader ────────────────────────────────────────────

    def read_recent_lore(self, count: int = 10) -> list:
        """Read the N most recent lore entries."""
        mm = self._mm
        pos = LORE_BUFFER_OFFSET

        write_seq = struct.unpack_from("<Q", mm, pos)[0]
        pos += 8

        if write_seq == 0:
            return []

        lore = []
        start = max(0, write_seq - count)

        for seq in range(start, write_seq):
            idx = seq % MAX_LORE_ENTRIES
            entry_offset = pos + idx * LORE_ENTRY_SIZE

            question = _read_cstr(mm[entry_offset:entry_offset + MAX_LORE_QUESTION])
            entry_pos = entry_offset + MAX_LORE_QUESTION

            answer = _read_cstr(mm[entry_pos:entry_pos + MAX_LORE_TEXT])
            entry_pos += MAX_LORE_TEXT

            layer, reason = struct.unpack_from("<BB", mm, entry_pos)
            entry_pos += 2

            embedding_delta, effectiveness = struct.unpack_from("<ff", mm, entry_pos)
            entry_pos += 8

            entry_pos += 2  # _pad

            timestamp_ns, entry_seq = struct.unpack_from("<QQ", mm, entry_pos)

            if entry_seq == seq:
                lore.append(LoreEntry(
                    question=question, answer=answer,
                    layer=layer, reason=reason,
                    embedding_delta=embedding_delta,
                    effectiveness=effectiveness,
                    timestamp_ns=timestamp_ns, seq=entry_seq,
                ))

        return lore

    # ── High-level summaries for voice assistant ─────────────────────

    def get_beliefs_summary(self) -> str:
        """Human-readable summary of all layer beliefs."""
        layers = self.read_all_layers()
        lines = ["Qualia belief state:"]
        layer_names = [
            "L0 superposition", "L1 motor", "L2 local",
            "L3 visual", "L4 behavior", "L5 deep", "L6 sensor",
        ]
        for s in layers:
            status = "CHALLENGED" if s.is_challenged else f"stable (streak={s.confirm_streak})"
            lines.append(
                f"  {layer_names[s.layer_id]}: VFE={s.vfe:.4f}, comp={s.compression}, "
                f"{status}, {s.cycle_us}us/cycle"
            )
        return "\n".join(lines)

    def get_surprise_summary(self) -> str:
        """Which layers have high VFE (prediction errors)."""
        layers = self.read_all_layers()
        surprised = [s for s in layers if s.is_challenged]
        if not surprised:
            return "All layers are calm — no significant prediction errors."
        layer_names = [
            "L0 superposition", "L1 motor", "L2 local",
            "L3 visual", "L4 behavior", "L5 deep", "L6 sensor",
        ]
        lines = [f"{len(surprised)} layer(s) have high prediction error:"]
        for s in surprised:
            lines.append(
                f"  {layer_names[s.layer_id]}: VFE={s.vfe:.4f} "
                f"(challenge_vfe={s.challenge_vfe:.4f})"
            )
        return "\n".join(lines)

    def get_lore_summary(self) -> str:
        """Recent LORE Q&A entries."""
        entries = self.read_recent_lore(5)
        if not entries:
            return "No LORE entries yet — the system hasn't asked any questions."
        lines = [f"{len(entries)} recent LORE entries:"]
        for e in entries:
            kind = THOUGHT_KINDS.get(e.reason, "unknown")
            lines.append(f"  L{e.layer} ({kind}): Q: {e.question[:80]}")
            lines.append(f"    A: {e.answer[:100]}")
        return "\n".join(lines)

    def get_world_summary(self) -> str:
        """Current world model state."""
        w = self.read_world_model()
        lines = [f"Scene: {w.scene}"]
        lines.append(f"Activity: {w.activity}")
        if w.objects:
            obj_strs = [f"{o.name}({o.confidence:.0%})" for o in w.objects[:5]]
            lines.append(f"Objects: {', '.join(obj_strs)}")
        lines.append(f"Directive: {w.directive}")
        lines.append(f"Gemini calls: {w.llm_call_count}, frames: {w.vision_frame_count}")
        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────

def _read_cstr(buf: bytes) -> str:
    """Read a null-terminated C string from bytes."""
    try:
        end = buf.index(0)
        return buf[:end].decode("utf-8", errors="replace")
    except ValueError:
        return buf.decode("utf-8", errors="replace")


# ── Singleton for voice assistant integration ────────────────────────

_bridge: Optional[QualiaBridge] = None


def get_bridge() -> Optional[QualiaBridge]:
    """Get or create the global QualiaBridge instance. Returns None if SHM not available."""
    global _bridge
    if _bridge is not None and _bridge.is_open:
        return _bridge
    _bridge = QualiaBridge()
    if _bridge.open():
        return _bridge
    _bridge = None
    return None


# ── CLI test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    bridge = QualiaBridge()
    if not bridge.open():
        print("ERROR: Cannot open Qualia SHM. Is qualia-watch running?")
        print("  Start it: cd qualia && cargo run --release -p qualia-watch")
        raise SystemExit(1)

    print("=== Qualia SHM Bridge ===\n")
    print(bridge.get_beliefs_summary())
    print()
    print(bridge.get_surprise_summary())
    print()
    print(bridge.get_world_summary())
    print()
    print(bridge.get_lore_summary())
    print()

    thoughts = bridge.read_recent_thoughts(10)
    if thoughts:
        print(f"Recent thoughts ({len(thoughts)}):")
        for t in thoughts:
            kind = THOUGHT_KINDS.get(t.kind, "?")
            print(f"  L{t.layer} [{kind}] VFE={t.vfe:.4f}: {t.text[:80]}")

    bridge.close()
