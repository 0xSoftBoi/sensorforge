pub use qualia_types::*;

use metal::*;

const BELIEF_KERNEL_SRC: &str = include_str!("../../../kernels/belief_update.metal");

pub struct LayerParams {
    pub threshold: f32,
    pub learning_rate: f32,
    pub layer_id: u8,
    pub freq_hz: f64,
    pub weight_decay: f32,
    pub precision_eta: f32,
    pub surprise_threshold: f32,
    pub top_down_weight: f32,
}

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    params_buf: Buffer,
}

impl MetalContext {
    pub fn new(params: &LayerParams, has_above: bool) -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(BELIEF_KERNEL_SRC, &opts)
            .map_err(|e| format!("Metal compile error: {}", e))?;

        let function = library
            .get_function("belief_update", None)
            .map_err(|e| format!("Kernel function 'belief_update' not found: {}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline error: {}", e))?;

        let params_data: [f32; 8] = [
            params.threshold,
            params.learning_rate,
            params.layer_id as f32,
            params.weight_decay,
            if has_above { 1.0 } else { 0.0 },
            params.precision_eta,
            params.surprise_threshold,
            params.top_down_weight,
        ];
        let params_buf = device.new_buffer_with_data(
            params_data.as_ptr() as *const _,
            (8 * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device,
            queue,
            pipeline,
            params_buf,
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Run belief update kernel on GPU with full generative model.
    pub fn dispatch_belief_update(
        &self,
        belief: &mut BeliefSlot,
        below: &BeliefSlot,
        above: Option<&BeliefSlot>,
        weights: &mut [f32; WEIGHT_COUNT],
        bias: &mut [f32; STATE_DIM],
    ) {
        let belief_size = std::mem::size_of::<BeliefSlot>() as u64;
        let weights_size = (WEIGHT_COUNT * std::mem::size_of::<f32>()) as u64;
        let bias_size = (STATE_DIM * std::mem::size_of::<f32>()) as u64;

        let dummy_above: BeliefSlot = unsafe { std::mem::zeroed() };
        let above_ref = above.unwrap_or(&dummy_above);

        let belief_buf = self.device.new_buffer_with_data(
            belief as *const BeliefSlot as *const _,
            belief_size,
            MTLResourceOptions::StorageModeShared,
        );
        let below_buf = self.device.new_buffer_with_data(
            below as *const BeliefSlot as *const _,
            belief_size,
            MTLResourceOptions::StorageModeShared,
        );
        let above_buf = self.device.new_buffer_with_data(
            above_ref as *const BeliefSlot as *const _,
            belief_size,
            MTLResourceOptions::StorageModeShared,
        );
        let weights_buf = self.device.new_buffer_with_data(
            weights.as_ptr() as *const _,
            weights_size,
            MTLResourceOptions::StorageModeShared,
        );
        let bias_buf = self.device.new_buffer_with_data(
            bias.as_ptr() as *const _,
            bias_size,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&belief_buf), 0);
        enc.set_buffer(1, Some(&below_buf), 0);
        enc.set_buffer(2, Some(&above_buf), 0);
        enc.set_buffer(3, Some(&self.params_buf), 0);
        enc.set_buffer(4, Some(&weights_buf), 0);
        enc.set_buffer(5, Some(&bias_buf), 0);

        let grid_size = MTLSize::new(STATE_DIM as u64, 1, 1);
        let group_size = MTLSize::new(STATE_DIM as u64, 1, 1);
        enc.dispatch_threads(grid_size, group_size);

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        unsafe {
            std::ptr::copy_nonoverlapping(
                belief_buf.contents() as *const BeliefSlot,
                belief as *mut BeliefSlot,
                1,
            );
            std::ptr::copy_nonoverlapping(
                weights_buf.contents() as *const f32,
                weights.as_mut_ptr(),
                WEIGHT_COUNT,
            );
            std::ptr::copy_nonoverlapping(
                bias_buf.contents() as *const f32,
                bias.as_mut_ptr(),
                STATE_DIM,
            );
        }
    }
}

// ── Thought Theater: Factor Pressure GPU context ────────────────────

const PRESSURE_KERNEL_SRC: &str = include_str!("../../../kernels/factor_pressure.metal");

pub struct TheaterMetalContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    params_buf: Buffer,
}

impl TheaterMetalContext {
    pub fn new(age_decay: f32, support_weight: f32, contradict_weight: f32) -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(PRESSURE_KERNEL_SRC, &opts)
            .map_err(|e| format!("Metal compile error (pressure): {}", e))?;

        let function = library
            .get_function("factor_pressure", None)
            .map_err(|e| format!("Kernel 'factor_pressure' not found: {}", e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline error (pressure): {}", e))?;

        let params_data: [f32; 3] = [age_decay, support_weight, contradict_weight];
        let params_buf = device.new_buffer_with_data(
            params_data.as_ptr() as *const _,
            (3 * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self { device, queue, pipeline, params_buf })
    }

    pub fn dispatch_factor_pressure(
        &self,
        nodes: &mut [GraphNode],
        edges: &mut [GraphEdge],
        energies: &mut [f32],
        node_count: u32,
        edge_count: u32,
    ) {
        let node_size = (nodes.len() * std::mem::size_of::<GraphNode>()) as u64;
        let edge_size = (edges.len() * std::mem::size_of::<GraphEdge>()) as u64;
        let energy_size = (energies.len() * std::mem::size_of::<f32>()) as u64;

        let nodes_buf = self.device.new_buffer_with_data(
            nodes.as_ptr() as *const _,
            node_size,
            MTLResourceOptions::StorageModeShared,
        );
        let edges_buf = self.device.new_buffer_with_data(
            edges.as_ptr() as *const _,
            edge_size,
            MTLResourceOptions::StorageModeShared,
        );
        let energies_buf = self.device.new_buffer_with_data(
            energies.as_ptr() as *const _,
            energy_size,
            MTLResourceOptions::StorageModeShared,
        );

        let counts: [u32; 2] = [node_count, edge_count];
        let counts_buf = self.device.new_buffer_with_data(
            counts.as_ptr() as *const _,
            (2 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&nodes_buf), 0);
        enc.set_buffer(1, Some(&edges_buf), 0);
        enc.set_buffer(2, Some(&energies_buf), 0);
        enc.set_buffer(3, Some(&self.params_buf), 0);
        enc.set_buffer(4, Some(&counts_buf), 0);

        let grid_size = MTLSize::new(64, 1, 1);
        let group_size = MTLSize::new(64, 1, 1);
        enc.dispatch_threads(grid_size, group_size);

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        unsafe {
            std::ptr::copy_nonoverlapping(
                nodes_buf.contents() as *const GraphNode,
                nodes.as_mut_ptr(),
                nodes.len(),
            );
            std::ptr::copy_nonoverlapping(
                edges_buf.contents() as *const GraphEdge,
                edges.as_mut_ptr(),
                edges.len(),
            );
            std::ptr::copy_nonoverlapping(
                energies_buf.contents() as *const f32,
                energies.as_mut_ptr(),
                energies.len(),
            );
        }
    }
}

pub fn default_params(layer_id: u8) -> LayerParams {
    match layer_id {
        0 => LayerParams { threshold: 0.1, learning_rate: 0.01, layer_id: 0, freq_hz: 1000.0, weight_decay: 0.0001, precision_eta: 0.01, surprise_threshold: 2.0, top_down_weight: 0.3 },
        1 => LayerParams { threshold: 0.08, learning_rate: 0.008, layer_id: 1, freq_hz: 100.0, weight_decay: 0.0001, precision_eta: 0.01, surprise_threshold: 2.0, top_down_weight: 0.3 },
        2 => LayerParams { threshold: 0.05, learning_rate: 0.005, layer_id: 2, freq_hz: 100.0, weight_decay: 0.00005, precision_eta: 0.008, surprise_threshold: 2.0, top_down_weight: 0.3 },
        3 => LayerParams { threshold: 0.05, learning_rate: 0.005, layer_id: 3, freq_hz: 100.0, weight_decay: 0.00005, precision_eta: 0.008, surprise_threshold: 2.0, top_down_weight: 0.3 },
        4 => LayerParams { threshold: 0.03, learning_rate: 0.003, layer_id: 4, freq_hz: 1.0, weight_decay: 0.00001, precision_eta: 0.005, surprise_threshold: 2.5, top_down_weight: 0.4 },
        5 => LayerParams { threshold: 0.02, learning_rate: 0.002, layer_id: 5, freq_hz: 0.1, weight_decay: 0.00001, precision_eta: 0.003, surprise_threshold: 3.0, top_down_weight: 0.5 },
        6 => LayerParams { threshold: 0.1, learning_rate: 0.001, layer_id: 6, freq_hz: 30.0, weight_decay: 0.0001, precision_eta: 0.005, surprise_threshold: 2.0, top_down_weight: 0.0 },
        _ => LayerParams { threshold: 0.1, learning_rate: 0.01, layer_id, freq_hz: 10.0, weight_decay: 0.0001, precision_eta: 0.01, surprise_threshold: 2.0, top_down_weight: 0.3 },
    }
}

// Thought kind constants
const THOUGHT_OBSERVE: u8 = 0;
const THOUGHT_PREDICT: u8 = 1;
const THOUGHT_SURPRISE: u8 = 2;
const THOUGHT_LEARN: u8 = 3;
const THOUGHT_RESOLVE: u8 = 4;
const THOUGHT_ESCALATE: u8 = 5;

const LAYER_DESCRIPTIONS: [&str; 7] = [
    "raw sensation",         // L0: superposition — direct sensory contact
    "motor patterns",        // L1: body/motor — how things move
    "local structure",       // L2: local spatial — nearby arrangement
    "visual patterns",       // L3: visual — what things look like
    "short-term behavior",   // L4: behavioral — what's happening now
    "deep patterns",         // L5: deep behavioral — persistent regularities
    "senses",                // L6: sensor — raw camera input
];

/// Run the main loop for a single layer.
pub fn run_layer(layer_id: u8, name: &str) {
    use qualia_shm::{LayerReader, LayerWriter, ShmRegion};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    let shm_name =
        std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".to_string());
    eprintln!("qualia-{name}: opening shm '{shm_name}', layer {layer_id}");

    let shm = ShmRegion::open(&shm_name).unwrap_or_else(|e| {
        panic!("qualia-{name}: failed to open shm: {e}");
    });

    let params = default_params(layer_id);
    let has_above = (layer_id as usize) < NUM_LAYERS - 1;
    let metal = MetalContext::new(&params, has_above).unwrap_or_else(|e| {
        panic!("qualia-{name}: Metal init failed: {e}");
    });
    eprintln!("qualia-{name}: Metal device: {}", metal.device_name());

    let my_slot = shm.layer_slot(layer_id as usize);
    let writer = LayerWriter::new(my_slot);

    let below_slot = if layer_id > 0 {
        shm.layer_slot((layer_id - 1) as usize)
    } else {
        shm.layer_slot(NUM_LAYERS - 1) // L0 reads from sensor layer (L6)
    };
    let reader = LayerReader::new(below_slot);

    let above_reader = if has_above {
        Some(LayerReader::new(shm.layer_slot((layer_id + 1) as usize)))
    } else {
        None
    };

    let tick_duration = if params.freq_hz > 0.0 {
        Duration::from_secs_f64(1.0 / params.freq_hz)
    } else {
        Duration::from_secs(60)
    };

    // Initialize weights as identity matrix
    {
        let weights = unsafe {
            let slot_ptr = my_slot as *const qualia_types::LayerSlot
                as *mut qualia_types::LayerSlot;
            &mut (*slot_ptr).weights
        };
        let bias = unsafe {
            let slot_ptr = my_slot as *const qualia_types::LayerSlot
                as *mut qualia_types::LayerSlot;
            &mut (*slot_ptr).bias
        };
        for i in 0..STATE_DIM {
            for j in 0..STATE_DIM {
                weights[i * STATE_DIM + j] = if i == j { 1.0 } else { 0.0 };
            }
            bias[i] = 0.0;
        }
    }

    // Initialize belief
    {
        let buf = writer.back_buffer();
        buf.layer = layer_id;
        for i in 0..STATE_DIM {
            buf.mean[i] = 0.0;
            buf.precision[i] = 1.0;
            buf.prediction[i] = 0.0;
            buf.residual[i] = 0.0;
        }
        buf.vfe = 0.0;
        buf.challenge_vfe = 0.0;
        buf.confirm_streak = 0;
        buf.compression = 0;
        writer.publish();
    }

    let desc = LAYER_DESCRIPTIONS[layer_id as usize];
    shm.emit_thought(
        layer_id, THOUGHT_OBSERVE, 0.0,
        &format!("{} init: dim={}, freq={:.1}Hz", desc, STATE_DIM, params.freq_hz),
    );

    // Semantic injection strength: how much the LLM's world understanding
    // influences this layer's beliefs. Upper layers get more — they deal
    // in meaning. Lower layers stay grounded in raw sensation.
    // Gemini embeddings flow down from L5 (top processing layer).
    // L6 is sensor — no injection there. Gemini IS the semantic layer.
    let semantic_alpha: f32 = match layer_id {
        5 => 0.30,   // deep patterns — strongest, closest to Gemini's understanding
        4 => 0.15,   // short-term behavior — informed by scene understanding
        3 => 0.05,   // visual patterns — light semantic context
        2 => 0.02,   // local structure — subtle semantic prior
        _ => 0.0,    // L0-L1, L6: pure sensation, no injection
    };
    let has_injection = semantic_alpha > 0.0;

    if has_injection {
        eprintln!("qualia-{name}: semantic injection alpha={semantic_alpha}");
    }

    if has_above {
        eprintln!("qualia-{name}: top-down from L{} (weight={:.2})", layer_id + 1, params.top_down_weight);
    }
    eprintln!("qualia-{name}: running at {:.1} Hz, prec_eta={:.3}, surprise_gate={:.1}", params.freq_hz, params.precision_eta, params.surprise_threshold);

    // Thought rate limiting: don't spam, make them meaningful
    let mut last_thought = Instant::now();
    let mut last_question = Instant::now();
    let mut prev_vfe: f32 = 0.0;
    let mut prev_compression: u8 = 0;
    let mut prev_streak: u32 = 0;
    let mut cycle_count: u64 = 0;
    let mut last_world_seq: u64 = 0;
    let mut high_vfe_streak: u32 = 0;        // consecutive high-VFE cycles
    let mut compression_stall_count: u32 = 0; // cycles at same compression

    // Questions are SPARSE — don't waste Gemini calls.
    // Upper layers ask more often (they deal in meaning).
    let question_cooldown = Duration::from_secs(match layer_id {
        5 => 60,       // deep patterns — ask every minute
        4 => 120,      // behavior — every 2 min
        3 => 300,      // visual — every 5 min
        _ => 600,      // lower layers rarely need to ask
    });
    let thought_cooldown = Duration::from_millis(match layer_id {
        0 => 2000,      // L0 is fast, throttle more
        1..=3 => 1000,  // mid layers
        4..=5 => 500,   // behavioral layers — more reflective
        6 => 300,       // semantic — most articulate
        _ => 2000,
    });

    loop {
        let cycle_start = Instant::now();
        cycle_count += 1;

        let below = *reader.read();
        let above_belief = above_reader.as_ref().map(|r| *r.read());

        let (weights, bias) = unsafe {
            let slot_ptr =
                my_slot as *const qualia_types::LayerSlot as *mut qualia_types::LayerSlot;
            (&mut (*slot_ptr).weights, &mut (*slot_ptr).bias)
        };

        let buf = writer.back_buffer();
        buf.layer = layer_id;
        metal.dispatch_belief_update(buf, &below, above_belief.as_ref(), weights, bias);

        // ── Semantic injection ──────────────────────────────────
        // Blend the LLM's scene_embedding into this layer's beliefs.
        // This is how meaning flows down: the world model perturbs
        // upper layers, creating prediction errors that propagate
        // down through the hierarchy via normal weight learning.
        if has_injection {
            let world = shm.world_model();
            let world_seq = world.update_seq.load(Ordering::Relaxed);

            // Stronger injection when world model just updated (new LLM call)
            // to create a burst of prediction error that forces re-learning.
            let alpha = if world_seq != last_world_seq && world_seq > 0 {
                last_world_seq = world_seq;
                // New information — inject strongly to disrupt current beliefs.
                // This is "paying attention" — the system notices something changed.
                let burst_alpha = (semantic_alpha * 3.0).min(0.5);

                if last_thought.elapsed() >= thought_cooldown {
                    shm.emit_thought(layer_id, THOUGHT_SURPRISE, buf.vfe, &format!(
                        "{} semantic inject: alpha={:.2}, VFE={:.4}, seq={}",
                        desc, burst_alpha, buf.vfe, world.update_seq.load(Ordering::Relaxed)
                    ));
                    last_thought = Instant::now();
                }

                burst_alpha
            } else {
                semantic_alpha
            };

            // Blend: mean[i] = (1-alpha) * mean[i] + alpha * embedding[i]
            // The embedding is in [-1, 1], beliefs are in arbitrary range.
            // We scale the embedding by the layer's current precision —
            // confident layers resist change, uncertain layers absorb it.
            for i in 0..STATE_DIM {
                let emb_val = world.scene_embedding[i];
                let prec_scale = 1.0 / (1.0 + buf.precision[i] * 0.1);
                buf.mean[i] = buf.mean[i] * (1.0 - alpha * prec_scale)
                    + emb_val * alpha * prec_scale;
            }

            // Also boost precision on dimensions where we have strong
            // semantic signal — we're more confident about what we know.
            for i in 0..STATE_DIM {
                let emb_strength = world.scene_embedding[i].abs();
                if emb_strength > 0.3 {
                    buf.precision[i] = (buf.precision[i] * (1.0 + alpha * 0.1 * emb_strength))
                        .min(100.0);
                }
            }
        }

        let elapsed = cycle_start.elapsed();
        buf.cycle_us = elapsed.as_micros() as u32;
        buf.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let vfe = buf.vfe;
        let compression = buf.compression;
        let streak = buf.confirm_streak;
        let is_challenge = buf.challenge_vfe > params.threshold;

        if is_challenge {
            my_slot.challenge_flag.store(true, Ordering::Release);
            my_slot.challenge_total.fetch_add(1, Ordering::Relaxed);
        } else {
            my_slot.confirm_flag.store(true, Ordering::Release);
            my_slot.confirm_total.fetch_add(1, Ordering::Relaxed);
        }

        writer.publish();

        // ── Generate thoughts based on what just happened ──
        if last_thought.elapsed() >= thought_cooldown {
            let thought = generate_thought(
                layer_id, desc, vfe, prev_vfe, compression, prev_compression,
                streak, prev_streak, is_challenge, cycle_count, &params,
            );
            if let Some((kind, text)) = thought {
                shm.emit_thought(layer_id, kind, vfe, &text);
                last_thought = Instant::now();
            }
        }

        // ── Track confusion for question generation ──
        if is_challenge && vfe > params.threshold * 3.0 {
            high_vfe_streak += 1;
        } else {
            high_vfe_streak = 0;
        }
        if compression == prev_compression && compression > 0 {
            compression_stall_count += 1;
        } else {
            compression_stall_count = 0;
        }

        // ── Pose questions to the outside world (sparingly) ──
        if has_injection && last_question.elapsed() >= question_cooldown
            && !my_slot.question.pending.load(Ordering::Relaxed)
        {
            let question = generate_question(
                layer_id, desc, vfe, high_vfe_streak,
                compression, compression_stall_count,
                is_challenge, cycle_count, &params,
            );
            if let Some((reason, text)) = question {
                // Write question to our slot
                let q = unsafe {
                    let slot_ptr = my_slot as *const LayerSlot as *mut LayerSlot;
                    &mut (*slot_ptr).question
                };
                q.text = [0u8; MAX_QUESTION_TEXT];
                let bytes = text.as_bytes();
                let len = bytes.len().min(MAX_QUESTION_TEXT - 1);
                q.text[..len].copy_from_slice(&bytes[..len]);
                q.layer = layer_id;
                q.reason = reason;
                q.vfe = vfe;
                q.timestamp_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
                q.pending.store(true, Ordering::Release);

                shm.emit_thought(layer_id, THOUGHT_ESCALATE, vfe, &format!(
                    "QUESTION VFE={:.3}: {}", vfe, text
                ));
                last_question = Instant::now();
            }
        }

        prev_vfe = vfe;
        prev_compression = compression;
        prev_streak = streak;

        let compute_time = cycle_start.elapsed();
        if compute_time < tick_duration {
            std::thread::sleep(tick_duration - compute_time);
        }
    }
}

fn generate_thought(
    _layer_id: u8, desc: &str,
    vfe: f32, prev_vfe: f32,
    compression: u8, prev_compression: u8,
    streak: u32, prev_streak: u32,
    is_challenge: bool,
    cycle: u64,
    params: &LayerParams,
) -> Option<(u8, String)> {
    let vfe_delta = vfe - prev_vfe;
    let vfe_ratio = if prev_vfe > 0.001 { vfe / prev_vfe } else { 1.0 };

    // Sudden VFE spike
    if is_challenge && vfe_ratio > 2.0 && vfe > params.threshold * 2.0 {
        return Some((THOUGHT_SURPRISE, format!(
            "{} VFE {:.4} -> {:.4} ({:.1}x spike, thresh={:.4})",
            desc, prev_vfe, vfe, vfe_ratio, params.threshold
        )));
    }

    // Challenge after long confirm streak
    if is_challenge && prev_streak > 20 {
        return Some((THOUGHT_SURPRISE, format!(
            "{} broke {}-cycle streak, VFE={:.4}",
            desc, prev_streak, vfe
        )));
    }

    // Active challenge — VFE above threshold
    if is_challenge && vfe > params.threshold {
        return Some((THOUGHT_LEARN, format!(
            "{} VFE={:.4} (thresh={:.4}, {:.1}x), dW active",
            desc, vfe, params.threshold, vfe / params.threshold
        )));
    }

    // Compression changed
    if compression > prev_compression && compression % 5 == 0 {
        return Some((THOUGHT_RESOLVE, format!(
            "{} compression {} -> {}", desc, prev_compression, compression
        )));
    }

    // Confirm streak milestone
    if streak > 0 && prev_streak > 0 && streak % 100 == 0 && streak != prev_streak {
        return Some((THOUGHT_RESOLVE, format!(
            "{} streak={}, VFE={:.4}, comp={}", desc, streak, vfe, compression
        )));
    }

    // VFE dropped below threshold
    if vfe_delta < -0.01 && prev_vfe > params.threshold && vfe < params.threshold {
        return Some((THOUGHT_RESOLVE, format!(
            "{} VFE {:.4} -> {:.4} (below thresh={:.4})",
            desc, prev_vfe, vfe, params.threshold
        )));
    }

    // VFE extremely high
    if vfe > params.threshold * 10.0 && cycle % 10 == 0 {
        return Some((THOUGHT_ESCALATE, format!(
            "{} VFE={:.3} ({:.0}x thresh), cycle {}", desc, vfe, vfe / params.threshold, cycle
        )));
    }

    // Periodic status when calm
    if !is_challenge && streak > 50 && cycle % 200 == 0 {
        return Some((THOUGHT_OBSERVE, format!(
            "{} stable: streak={}, VFE={:.4}, comp={}", desc, streak, vfe, compression
        )));
    }

    // First few cycles — narrate waking up
    if cycle <= 3 {
        return Some((THOUGHT_PREDICT, format!(
            "Making my first predictions about {}. Everything starts as identity — I predict the layer below is what I believe.",
            desc
        )));
    }

    None
}

/// Generate a question for the outside world. Returns (reason_code, question_text).
/// Questions are the system's way of building LORE — they're asked sparingly
/// and only when the layer genuinely can't resolve its prediction errors alone.
fn generate_question(
    layer_id: u8, desc: &str,
    vfe: f32, high_vfe_streak: u32,
    compression: u8, compression_stall_count: u32,
    is_challenge: bool,
    cycle: u64,
    params: &LayerParams,
) -> Option<(u8, String)> {
    // Reason 0: persistent high VFE — the layer is stuck, can't learn the pattern
    if high_vfe_streak > 50 && vfe > params.threshold * 5.0 {
        let question = match layer_id {
            5 => format!(
                "I process {} but my predictions keep failing (VFE={:.4} for {} cycles). \
                 What deep pattern or regularity exists in this environment that I'm missing?",
                desc, vfe, high_vfe_streak
            ),
            4 => format!(
                "My {} predictions are wrong (VFE={:.4}, {} cycles stuck). \
                 What is currently happening that I haven't accounted for?",
                desc, vfe, high_vfe_streak
            ),
            3 => format!(
                "I can't predict what I'm seeing in {} (VFE={:.4}). \
                 What visual pattern should I expect here?",
                desc, vfe
            ),
            _ => format!(
                "Layer {} ({}) can't converge (VFE={:.4}, {} stuck cycles). \
                 What structure exists here that I should learn?",
                layer_id, desc, vfe, high_vfe_streak
            ),
        };
        return Some((0, question));
    }

    // Reason 1: compression plateau — learned something but can't simplify further
    if compression_stall_count > 200 && compression > 20 && is_challenge {
        let question = match layer_id {
            5 => format!(
                "I've learned a pattern in {} (compression={}) but can't simplify it. \
                 Is there a higher-level concept that unifies what I'm seeing?",
                desc, compression
            ),
            4 => format!(
                "My {} model plateaued at compression={}. \
                 What category or label describes this behavioral pattern?",
                desc, compression
            ),
            _ => format!(
                "Compression stalled at {} for {}. \
                 What abstraction am I missing?",
                compression, desc
            ),
        };
        return Some((1, question));
    }

    // Reason 2: novel pattern — was stable, then suddenly challenged
    if is_challenge && vfe > params.threshold * 8.0 && cycle > 100 {
        return Some((2, format!(
            "Something new appeared in {}. VFE spiked to {:.4}. \
             What changed in the environment?",
            desc, vfe
        )));
    }

    None
}
