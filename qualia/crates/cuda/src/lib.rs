pub use qualia_types::*;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const BELIEF_KERNEL_SRC: &str = include_str!("../../../kernels/belief_update.cu");

pub struct LayerParams {
    pub threshold: f32,
    pub learning_rate: f32,
    pub layer_id: u8,
    pub freq_hz: f64,
    pub weight_decay: f32,
}

pub struct CudaContext {
    device: Arc<CudaDevice>,
    params_dev: CudaSlice<f32>,
}

impl CudaContext {
    pub fn new(params: &LayerParams) -> Result<Self, String> {
        let device = CudaDevice::new(0).map_err(|e| format!("CUDA device init failed: {e}"))?;

        let ptx = Ptx::from_src(BELIEF_KERNEL_SRC);
        device
            .load_ptx(ptx, "belief", &["belief_update"])
            .map_err(|e| format!("PTX compile/load failed: {e}"))?;

        let params_data: [f32; 4] = [
            params.threshold,
            params.learning_rate,
            params.layer_id as f32,
            params.weight_decay,
        ];
        let params_dev = device
            .htod_copy(params_data.to_vec())
            .map_err(|e| format!("params copy failed: {e}"))?;

        Ok(Self { device, params_dev })
    }

    pub fn device_name(&self) -> String {
        format!("CUDA device 0")
    }

    /// Run belief update kernel on GPU with full generative model.
    pub fn dispatch_belief_update(
        &self,
        belief: &mut BeliefSlot,
        below: &BeliefSlot,
        weights: &mut [f32; WEIGHT_COUNT],
        bias: &mut [f32; STATE_DIM],
    ) {
        let belief_bytes = unsafe {
            std::slice::from_raw_parts(
                belief as *const BeliefSlot as *const u8,
                std::mem::size_of::<BeliefSlot>(),
            )
        };
        let below_bytes = unsafe {
            std::slice::from_raw_parts(
                below as *const BeliefSlot as *const u8,
                std::mem::size_of::<BeliefSlot>(),
            )
        };

        let belief_dev = self
            .device
            .htod_copy(belief_bytes.to_vec())
            .expect("belief htod");
        let below_dev = self
            .device
            .htod_copy(below_bytes.to_vec())
            .expect("below htod");
        let weights_dev = self
            .device
            .htod_copy(weights.to_vec())
            .expect("weights htod");
        let bias_dev = self.device.htod_copy(bias.to_vec()).expect("bias htod");

        let kernel = self
            .device
            .get_func("belief", "belief_update")
            .expect("kernel not found");

        // 1 block of 64 threads — matches Metal dispatch (64 threads per layer)
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (STATE_DIM as u32, 1, 1),
            shared_mem_bytes: 0, // shared memory is statically allocated in kernel
        };

        unsafe {
            kernel
                .launch(
                    cfg,
                    (&belief_dev, &below_dev, &self.params_dev, &weights_dev, &bias_dev),
                )
                .expect("kernel launch failed");
        }

        // Copy results back
        let belief_result = self.device.dtoh_sync_copy(&belief_dev).expect("belief dtoh");
        unsafe {
            std::ptr::copy_nonoverlapping(
                belief_result.as_ptr(),
                belief as *mut BeliefSlot as *mut u8,
                std::mem::size_of::<BeliefSlot>(),
            );
        }

        let weights_result = self
            .device
            .dtoh_sync_copy(&weights_dev)
            .expect("weights dtoh");
        weights.copy_from_slice(&weights_result);

        let bias_result = self.device.dtoh_sync_copy(&bias_dev).expect("bias dtoh");
        bias.copy_from_slice(&bias_result);
    }
}

pub fn default_params(layer_id: u8) -> LayerParams {
    match layer_id {
        0 => LayerParams {
            threshold: 0.1,
            learning_rate: 0.01,
            layer_id: 0,
            freq_hz: 1000.0,
            weight_decay: 0.0001,
        },
        1 => LayerParams {
            threshold: 0.08,
            learning_rate: 0.008,
            layer_id: 1,
            freq_hz: 100.0,
            weight_decay: 0.0001,
        },
        2 => LayerParams {
            threshold: 0.05,
            learning_rate: 0.005,
            layer_id: 2,
            freq_hz: 100.0,
            weight_decay: 0.00005,
        },
        3 => LayerParams {
            threshold: 0.05,
            learning_rate: 0.005,
            layer_id: 3,
            freq_hz: 100.0,
            weight_decay: 0.00005,
        },
        4 => LayerParams {
            threshold: 0.03,
            learning_rate: 0.003,
            layer_id: 4,
            freq_hz: 1.0,
            weight_decay: 0.00001,
        },
        5 => LayerParams {
            threshold: 0.02,
            learning_rate: 0.002,
            layer_id: 5,
            freq_hz: 0.1,
            weight_decay: 0.00001,
        },
        6 => LayerParams {
            threshold: 0.1,
            learning_rate: 0.0,
            layer_id: 6,
            freq_hz: 30.0,
            weight_decay: 0.0,
        },
        _ => LayerParams {
            threshold: 0.1,
            learning_rate: 0.01,
            layer_id,
            freq_hz: 10.0,
            weight_decay: 0.0001,
        },
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
    "raw sensation",
    "motor patterns",
    "local structure",
    "visual patterns",
    "short-term behavior",
    "deep patterns",
    "senses",
];

/// Run the main loop for a single layer (CUDA backend).
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
    let cuda = CudaContext::new(&params).unwrap_or_else(|e| {
        panic!("qualia-{name}: CUDA init failed: {e}");
    });
    eprintln!("qualia-{name}: {}", cuda.device_name());

    let my_slot = shm.layer_slot(layer_id as usize);
    let writer = LayerWriter::new(my_slot);

    let below_slot = if layer_id > 0 {
        shm.layer_slot((layer_id - 1) as usize)
    } else {
        shm.layer_slot(NUM_LAYERS - 1)
    };
    let reader = LayerReader::new(below_slot);

    let tick_duration = if params.freq_hz > 0.0 {
        Duration::from_secs_f64(1.0 / params.freq_hz)
    } else {
        Duration::from_secs(60)
    };

    // Initialize weights as identity matrix
    {
        let weights = unsafe {
            let slot_ptr =
                my_slot as *const qualia_types::LayerSlot as *mut qualia_types::LayerSlot;
            &mut (*slot_ptr).weights
        };
        let bias = unsafe {
            let slot_ptr =
                my_slot as *const qualia_types::LayerSlot as *mut qualia_types::LayerSlot;
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
        layer_id,
        THOUGHT_OBSERVE,
        0.0,
        &format!(
            "{} init: dim={}, freq={:.1}Hz, backend=CUDA",
            desc, STATE_DIM, params.freq_hz
        ),
    );

    let semantic_alpha: f32 = match layer_id {
        5 => 0.30,
        4 => 0.15,
        3 => 0.05,
        2 => 0.02,
        _ => 0.0,
    };
    let has_injection = semantic_alpha > 0.0;

    if has_injection {
        eprintln!("qualia-{name}: semantic injection alpha={semantic_alpha}");
    }

    eprintln!(
        "qualia-{name}: running at {:.1} Hz with 64x64 generative model (CUDA)",
        params.freq_hz
    );

    let mut last_thought = Instant::now();
    let mut last_question = Instant::now();
    let mut prev_vfe: f32 = 0.0;
    let mut prev_compression: u8 = 0;
    let mut prev_streak: u32 = 0;
    let mut cycle_count: u64 = 0;
    let mut last_world_seq: u64 = 0;
    let mut high_vfe_streak: u32 = 0;
    let mut compression_stall_count: u32 = 0;

    let question_cooldown = Duration::from_secs(match layer_id {
        5 => 60,
        4 => 120,
        3 => 300,
        _ => 600,
    });
    let thought_cooldown = Duration::from_millis(match layer_id {
        0 => 2000,
        1..=3 => 1000,
        4..=5 => 500,
        6 => 300,
        _ => 2000,
    });

    loop {
        let cycle_start = Instant::now();
        cycle_count += 1;

        let below = *reader.read();

        let (weights, bias) = unsafe {
            let slot_ptr =
                my_slot as *const qualia_types::LayerSlot as *mut qualia_types::LayerSlot;
            (&mut (*slot_ptr).weights, &mut (*slot_ptr).bias)
        };

        let buf = writer.back_buffer();
        buf.layer = layer_id;
        cuda.dispatch_belief_update(buf, &below, weights, bias);

        // Semantic injection (identical to Metal backend)
        if has_injection {
            let world = shm.world_model();
            let world_seq = world.update_seq.load(Ordering::Relaxed);

            let alpha = if world_seq != last_world_seq && world_seq > 0 {
                last_world_seq = world_seq;
                let burst_alpha = (semantic_alpha * 3.0).min(0.5);

                if last_thought.elapsed() >= thought_cooldown {
                    shm.emit_thought(
                        layer_id,
                        THOUGHT_SURPRISE,
                        buf.vfe,
                        &format!(
                            "{} semantic inject: alpha={:.2}, VFE={:.4}, seq={}",
                            desc,
                            burst_alpha,
                            buf.vfe,
                            world.update_seq.load(Ordering::Relaxed)
                        ),
                    );
                    last_thought = Instant::now();
                }

                burst_alpha
            } else {
                semantic_alpha
            };

            for i in 0..STATE_DIM {
                let emb_val = world.scene_embedding[i];
                let prec_scale = 1.0 / (1.0 + buf.precision[i] * 0.1);
                buf.mean[i] =
                    buf.mean[i] * (1.0 - alpha * prec_scale) + emb_val * alpha * prec_scale;
            }

            for i in 0..STATE_DIM {
                let emb_strength = world.scene_embedding[i].abs();
                if emb_strength > 0.3 {
                    buf.precision[i] =
                        (buf.precision[i] * (1.0 + alpha * 0.1 * emb_strength)).min(100.0);
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

        // Generate thoughts
        if last_thought.elapsed() >= thought_cooldown {
            let thought = generate_thought(
                layer_id,
                desc,
                vfe,
                prev_vfe,
                compression,
                prev_compression,
                streak,
                prev_streak,
                is_challenge,
                cycle_count,
                &params,
            );
            if let Some((kind, text)) = thought {
                shm.emit_thought(layer_id, kind, vfe, &text);
                last_thought = Instant::now();
            }
        }

        // Track confusion for question generation
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

        // Pose questions to the outside world
        if has_injection
            && last_question.elapsed() >= question_cooldown
            && !my_slot.question.pending.load(Ordering::Relaxed)
        {
            let question = generate_question(
                layer_id,
                desc,
                vfe,
                high_vfe_streak,
                compression,
                compression_stall_count,
                is_challenge,
                cycle_count,
                &params,
            );
            if let Some((reason, text)) = question {
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

                shm.emit_thought(
                    layer_id,
                    THOUGHT_ESCALATE,
                    vfe,
                    &format!("QUESTION VFE={:.3}: {}", vfe, text),
                );
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
    _layer_id: u8,
    desc: &str,
    vfe: f32,
    prev_vfe: f32,
    compression: u8,
    prev_compression: u8,
    streak: u32,
    prev_streak: u32,
    is_challenge: bool,
    cycle: u64,
    params: &LayerParams,
) -> Option<(u8, String)> {
    let vfe_ratio = if prev_vfe > 0.001 {
        vfe / prev_vfe
    } else {
        1.0
    };

    if is_challenge && vfe_ratio > 2.0 && vfe > params.threshold * 2.0 {
        return Some((
            THOUGHT_SURPRISE,
            format!(
                "{} VFE {:.4} -> {:.4} ({:.1}x spike, thresh={:.4})",
                desc, prev_vfe, vfe, vfe_ratio, params.threshold
            ),
        ));
    }

    if is_challenge && prev_streak > 20 {
        return Some((
            THOUGHT_SURPRISE,
            format!("{} broke {}-cycle streak, VFE={:.4}", desc, prev_streak, vfe),
        ));
    }

    if is_challenge && vfe > params.threshold {
        return Some((
            THOUGHT_LEARN,
            format!(
                "{} VFE={:.4} (thresh={:.4}, {:.1}x), dW active",
                desc,
                vfe,
                params.threshold,
                vfe / params.threshold
            ),
        ));
    }

    if compression > prev_compression && compression % 5 == 0 {
        return Some((
            THOUGHT_RESOLVE,
            format!(
                "{} compression {} -> {}",
                desc, prev_compression, compression
            ),
        ));
    }

    if streak > 0 && prev_streak > 0 && streak % 100 == 0 && streak != prev_streak {
        return Some((
            THOUGHT_RESOLVE,
            format!(
                "{} streak={}, VFE={:.4}, comp={}",
                desc, streak, vfe, compression
            ),
        ));
    }

    let vfe_delta = vfe - prev_vfe;
    if vfe_delta < -0.01 && prev_vfe > params.threshold && vfe < params.threshold {
        return Some((
            THOUGHT_RESOLVE,
            format!(
                "{} VFE {:.4} -> {:.4} (below thresh={:.4})",
                desc, prev_vfe, vfe, params.threshold
            ),
        ));
    }

    if vfe > params.threshold * 10.0 && cycle % 10 == 0 {
        return Some((
            THOUGHT_ESCALATE,
            format!(
                "{} VFE={:.3} ({:.0}x thresh), cycle {}",
                desc,
                vfe,
                vfe / params.threshold,
                cycle
            ),
        ));
    }

    if !is_challenge && streak > 50 && cycle % 200 == 0 {
        return Some((
            THOUGHT_OBSERVE,
            format!(
                "{} stable: streak={}, VFE={:.4}, comp={}",
                desc, streak, vfe, compression
            ),
        ));
    }

    if cycle <= 3 {
        return Some((
            THOUGHT_PREDICT,
            format!(
                "Making my first predictions about {}. Everything starts as identity — I predict the layer below is what I believe.",
                desc
            ),
        ));
    }

    None
}

fn generate_question(
    layer_id: u8,
    desc: &str,
    vfe: f32,
    high_vfe_streak: u32,
    compression: u8,
    compression_stall_count: u32,
    is_challenge: bool,
    cycle: u64,
    params: &LayerParams,
) -> Option<(u8, String)> {
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

    if is_challenge && vfe > params.threshold * 8.0 && cycle > 100 {
        return Some((
            2,
            format!(
                "Something new appeared in {}. VFE spiked to {:.4}. \
                 What changed in the environment?",
                desc, vfe
            ),
        ));
    }

    None
}
