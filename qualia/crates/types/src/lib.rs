use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};

pub const STATE_DIM: usize = 64;
pub const SHM_MAGIC: u64 = 0x5155414C3141454E; // "QUAL1AEN"
pub const NUM_LAYERS: usize = 7;

// ── World Model ────────────────────────────────────────────────────────

pub const MAX_OBJECTS: usize = 16;
pub const MAX_SCENE_LEN: usize = 512;
pub const MAX_DIRECTIVE_LEN: usize = 256;
pub const MAX_ACTIVITY_LEN: usize = 128;
pub const MAX_OBJECT_NAME: usize = 32;

/// A detected object in the world.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WorldObject {
    /// Null-terminated ASCII name (e.g. "person\0", "mug\0")
    pub name: [u8; MAX_OBJECT_NAME],
    pub confidence: f32,
    /// Normalized position in frame [0.0, 1.0]
    pub x: f32,
    pub y: f32,
    /// Whether this slot is active
    pub active: u8,
    pub _pad: [u8; 3],
}

/// The world model — the system's accumulated semantic understanding
/// of the environment it exists in. Lives in shared memory so every
/// layer and the TUI can read it.
#[repr(C)]
pub struct WorldModel {
    /// Detected objects in the scene
    pub objects: [WorldObject; MAX_OBJECTS],
    pub num_objects: u32,
    pub _pad0: u32,

    /// Scene description from LLM (null-terminated UTF-8)
    pub scene: [u8; MAX_SCENE_LEN],
    /// Current activity description (null-terminated UTF-8)
    pub activity: [u8; MAX_ACTIVITY_LEN],

    /// Semantic embedding of the scene — 64-dim vector derived from
    /// Gemini understanding. This gets injected into L4-L5 beliefs.
    pub scene_embedding: [f32; STATE_DIM],

    /// Prime directive — the system's top-level goal (null-terminated UTF-8)
    /// Set by the user, read by L5 during escalation.
    pub directive: [u8; MAX_DIRECTIVE_LEN],

    /// Timestamps and counters
    pub last_vision_ns: u64,
    pub last_llm_ns: u64,
    pub llm_call_count: u64,
    pub vision_frame_count: u64,

    /// Gemini API usage tracking
    pub gemini_input_tokens: u64,
    pub gemini_output_tokens: u64,
    pub gemini_embedding_tokens: u64,

    /// Sequence number — incremented on each update
    pub update_seq: AtomicU64,
}

// ── Thought Stream ───────────────────────────────────────────────────

pub const MAX_THOUGHT_LEN: usize = 256;
pub const MAX_THOUGHTS: usize = 512;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ThoughtEntry {
    pub text: [u8; MAX_THOUGHT_LEN],
    pub layer: u8,
    pub kind: u8,      // 0=observe, 1=predict, 2=surprise, 3=learn, 4=resolve, 5=escalate
    pub _pad: [u8; 2],
    pub vfe: f32,
    pub timestamp_ns: u64,
    pub seq: u64,
}

#[repr(C)]
pub struct ThoughtBuffer {
    pub write_seq: AtomicU64,
    pub entries: [ThoughtEntry; MAX_THOUGHTS],
}

// ── Lore: accumulated world-knowledge from questions the layers ask ──

pub const MAX_LORE_TEXT: usize = 512;
pub const MAX_LORE_QUESTION: usize = 256;
pub const MAX_LORE_ENTRIES: usize = 128;
pub const MAX_QUESTION_TEXT: usize = 256;

/// A question a layer poses to the outside world when it can't resolve
/// its own prediction errors. The vision runner harvests these and
/// batches them into Gemini calls.
#[repr(C)]
pub struct QuestionSlot {
    /// The question text (null-terminated UTF-8)
    pub text: [u8; MAX_QUESTION_TEXT],
    /// Which layer is asking
    pub layer: u8,
    /// Why: 0=high_vfe, 1=compression_plateau, 2=novel_pattern, 3=escalation
    pub reason: u8,
    pub _pad: [u8; 2],
    /// VFE at the time of asking
    pub vfe: f32,
    /// Sequence — vision runner reads and clears by setting to 0
    pub pending: AtomicBool,
    pub _pad2: [u8; 7],
    pub timestamp_ns: u64,
}

/// Accumulated lore — an answered question that becomes part of the
/// system's world-knowledge. Each entry is a question + Gemini's answer
/// + the embedding shift it caused. LORE PERSISTS — it's the system's
/// long-term semantic memory.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LoreEntry {
    /// The original question
    pub question: [u8; MAX_LORE_QUESTION],
    /// Gemini's answer
    pub answer: [u8; MAX_LORE_TEXT],
    /// Which layer asked
    pub layer: u8,
    /// Reason code (same as QuestionSlot)
    pub reason: u8,
    /// How much the embedding shifted when this lore was integrated
    pub embedding_delta: f32,
    /// Confidence: did this lore reduce VFE? (post-injection VFE / pre-injection VFE)
    pub effectiveness: f32,
    pub _pad: [u8; 2],
    pub timestamp_ns: u64,
    pub seq: u64,
}

/// Ring buffer of accumulated lore.
#[repr(C)]
pub struct LoreBuffer {
    pub write_seq: AtomicU64,
    pub entries: [LoreEntry; MAX_LORE_ENTRIES],
}

// ── Belief State ─────────────────────────────────────────────────────

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct BeliefSlot {
    pub mean: [f32; STATE_DIM],
    pub precision: [f32; STATE_DIM],
    pub vfe: f32,
    pub prediction: [f32; STATE_DIM],
    pub residual: [f32; STATE_DIM],
    pub challenge_vfe: f32,
    pub confirm_streak: u32,
    pub compression: u8,
    pub layer: u8,
    pub _pad: [u8; 2],
    pub timestamp_ns: u64,
    pub cycle_us: u32,
    pub _pad2: [u8; 4],
    /// VFE exponential moving average — tracks "normal" VFE for this layer.
    /// Used for z-score surprise detection: z = (vfe - vfe_ema) / sqrt(vfe_var)
    pub vfe_ema: f32,
    /// VFE variance (EMA of squared deviations). For z-score computation.
    pub vfe_var: f32,
    /// Information-theoretic compression ratio: H(prediction) / H(observation).
    /// Replaces the streak-based u8 compression counter for meaningful metric.
    pub compression_ratio: f32,
}

/// Weight dimensions for the generative model.
/// Each layer has a 64×64 weight matrix and 64-element bias vector
/// that map belief → prediction of the layer below.
/// These ARE the non-verbal learned representations.
pub const WEIGHT_COUNT: usize = STATE_DIM * STATE_DIM; // 4096

#[repr(C)]
pub struct LayerSlot {
    pub buffers: [BeliefSlot; 2],
    /// Generative model weights: W[i][j] = weights[i * STATE_DIM + j]
    /// Phase 7: GLU prediction = (W @ mean + bias) ⊙ σ(gate_w @ mean + gate_b)
    /// Stored in shared memory so every process can inspect them.
    pub weights: [f32; WEIGHT_COUNT],
    /// Generative model bias vector.
    pub bias: [f32; STATE_DIM],
    /// GLU gate weights (Phase 7): σ(gate_w @ mean + gate_b) controls information flow.
    pub gate_w: [f32; WEIGHT_COUNT],
    /// GLU gate bias.
    pub gate_b: [f32; STATE_DIM],
    /// Adam optimizer first moment (momentum) for weights.
    pub adam_m_w: [f32; WEIGHT_COUNT],
    /// Adam optimizer second moment (RMSProp) for weights.
    pub adam_v_w: [f32; WEIGHT_COUNT],
    /// Adam first moment for bias.
    pub adam_m_b: [f32; STATE_DIM],
    /// Adam second moment for bias.
    pub adam_v_b: [f32; STATE_DIM],
    /// Adam first moment for gate weights.
    pub adam_m_gw: [f32; WEIGHT_COUNT],
    /// Adam second moment for gate weights.
    pub adam_v_gw: [f32; WEIGHT_COUNT],
    /// Adam first moment for gate bias.
    pub adam_m_gb: [f32; STATE_DIM],
    /// Adam second moment for gate bias.
    pub adam_v_gb: [f32; STATE_DIM],
    /// Adam timestep counter (per-layer, incremented each weight update).
    pub adam_t: AtomicU64,
    pub write_idx: AtomicUsize,
    pub challenge_flag: AtomicBool,
    pub confirm_flag: AtomicBool,
    pub escalate_flag: AtomicBool,
    pub _pad: u8,
    pub confirm_total: AtomicU64,
    pub challenge_total: AtomicU64,
    /// Question slot — layer writes here when it needs outside help.
    pub question: QuestionSlot,
}

// ── Action History (Phase 3.2: sensorimotor contingencies) ──────────

pub const MAX_ACTION_ENTRIES: usize = 256;

/// An action the system took, with before/after VFE and embedding snapshot.
/// Used by EFE explorer to predict action outcomes and by layers to learn
/// sensorimotor contingencies.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ActionEntry {
    /// Action code: 0=stop, 1=forward, 2=left, 3=right, 4=reverse
    pub action: u8,
    pub _pad0: [u8; 3],
    /// Motor speeds at time of action
    pub speed_left: i16,
    pub speed_right: i16,
    /// Mean VFE across layers BEFORE the action
    pub pre_vfe: f32,
    /// Mean VFE across layers AFTER the action (measured ~1s later)
    pub post_vfe: f32,
    /// Scene embedding snapshot at time of action (compressed to 16 dims)
    pub embedding: [f32; 16],
    /// Timestamp when action was taken
    pub timestamp_ns: u64,
    /// Sequence number
    pub seq: u64,
}

/// Ring buffer of action history, written by explorer, readable by all layers.
#[repr(C)]
pub struct ActionHistory {
    pub write_seq: AtomicU64,
    pub entries: [ActionEntry; MAX_ACTION_ENTRIES],
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedgerEvent {
    Challenge = 0,
    Confirm = 1,
    Habit = 2,
    HabitDecay = 3,
    Escalate = 4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LedgerEntry {
    pub seq: u64,
    pub layer: u8,
    pub event: LedgerEvent,
    pub compression: u8,
    pub _pad: u8,
    pub vfe: f32,
    pub residual_norm: f32,
    pub belief_mean: [f32; STATE_DIM],
    pub timestamp_ns: u64,
}

#[repr(C)]
pub struct ShmHeader {
    pub magic: u64,
    pub version: u32,
    pub num_layers: u32,
    pub layer_slot_size: u64,
    pub ledger_offset: u64,
    pub ledger_capacity: u64,
    pub ledger_write_seq: AtomicU64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HealthReport {
    pub layer: u8,
    pub compression: u8,
    pub _pad: [u8; 2],
    pub vfe: f32,
    pub challenge_vfe: f32,
    pub confirm_streak: u32,
    pub cycle_us: u32,
    pub timestamp_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn belief_slot_is_cache_aligned() {
        assert_eq!(mem::align_of::<BeliefSlot>(), 64);
    }

    #[test]
    fn belief_slot_size_is_stable() {
        // 4 * 64 * 4 (mean, precision, prediction, residual) = 1024
        // + 2 floats (vfe, challenge_vfe) = 8
        // + u32 (confirm_streak) + u8 (compression) + u8 (layer) + 2 pad = 8
        // + u64 (timestamp_ns) = 8
        // + u32 (cycle_us) + 4 pad = 8
        // + 3 floats (vfe_ema, vfe_var, compression_ratio) = 12
        // total = 1068, rounded up to 64-byte alignment = 1088
        let size = mem::size_of::<BeliefSlot>();
        assert!(size % 64 == 0, "BeliefSlot size {size} not 64-byte aligned");
        assert_eq!(size, 1088, "BeliefSlot size changed — update CUDA kernel struct and Python bridge");
    }

    #[test]
    fn ledger_event_repr() {
        assert_eq!(LedgerEvent::Challenge as u8, 0);
        assert_eq!(LedgerEvent::Escalate as u8, 4);
    }

    #[test]
    fn health_report_is_pod() {
        // Ensure HealthReport can be safely transmitted as bytes
        let size = mem::size_of::<HealthReport>();
        assert!(size > 0);
        assert!(size <= 64);
    }
}
