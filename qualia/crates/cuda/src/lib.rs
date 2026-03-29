pub use qualia_types::*;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const BELIEF_KERNEL_SRC: &str = include_str!("../../../kernels/belief_update.cu");

/// Compile CUDA kernel to PTX, patching the ISA version if toolkit > driver.
///
/// NVRTC from toolkit 12.9 generates PTX with `.version 8.8` but the Jetson
/// driver (540.4.0 = CUDA 12.6) only supports up to 8.5. The PTX content is
/// compatible — only the version header is too new. We compile with NVRTC
/// targeting the GPU's compute capability, then query the driver's max
/// supported CUDA version and downgrade the PTX version header to match.
fn compile_kernel(device: &Arc<CudaDevice>) -> Result<cudarc::nvrtc::Ptx, String> {
    if let Ok(path) = std::env::var("QUALIA_PTX_FILE") {
        eprintln!("qualia-cuda: loading pre-compiled PTX from {path}");
        let src = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read PTX file {path}: {e}"))?;
        return Ok(cudarc::nvrtc::Ptx::from_src(src));
    }

    let (major, minor) = detect_gpu_arch(device);
    let arch = format!("compute_{}{}", major, minor);

    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some(Box::leak(arch.into_boxed_str())),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(BELIEF_KERNEL_SRC, opts)
        .map_err(|e| format!("NVRTC compile failed: {e}"))?;

    // Patch PTX version header to match driver's supported ISA
    let ptx_src = ptx.to_src();
    let patched = patch_ptx_version(&ptx_src);
    Ok(cudarc::nvrtc::Ptx::from_src(patched))
}

fn detect_gpu_arch(device: &Arc<CudaDevice>) -> (i32, i32) {
    use cudarc::driver::sys::CUdevice_attribute::*;
    let major = device
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(8);
    let minor = device
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(7);
    eprintln!("qualia-cuda: detected GPU compute_{}{}", major, minor);
    (major, minor)
}

/// Query the driver's CUDA version and compute the max PTX ISA it supports.
/// Then rewrite the `.version X.Y` directive in the PTX source to match.
fn patch_ptx_version(ptx_src: &str) -> String {
    // Map CUDA driver version → max PTX ISA version
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes
    let driver_ptx = driver_max_ptx_version();

    if let Some(target) = driver_ptx {
        // Find and replace the .version line
        let mut result = String::with_capacity(ptx_src.len());
        for line in ptx_src.lines() {
            if line.starts_with(".version ") {
                let orig = line.trim_start_matches(".version ").trim();
                eprintln!(
                    "qualia-cuda: patching PTX .version {} -> {} (driver limit)",
                    orig, target
                );
                result.push_str(&format!(".version {}", target));
            } else {
                result.push_str(line);
            }
            result.push('\n');
        }
        result
    } else {
        eprintln!("qualia-cuda: could not query driver version, using PTX as-is");
        ptx_src.to_string()
    }
}

fn driver_max_ptx_version() -> Option<&'static str> {
    // Allow override
    if let Ok(v) = std::env::var("QUALIA_PTX_VERSION") {
        return Some(Box::leak(v.into_boxed_str()));
    }

    // Query driver CUDA version via cuDriverGetVersion
    let mut driver_ver: i32 = 0;
    let rc = unsafe { cudarc::driver::sys::lib().cuDriverGetVersion(&mut driver_ver) };
    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return None;
    }

    // driver_ver format: major*1000 + minor*10 (e.g., 12060 = CUDA 12.6)
    let cuda_major = driver_ver / 1000;
    let cuda_minor = (driver_ver % 1000) / 10;
    eprintln!("qualia-cuda: driver supports CUDA {}.{}", cuda_major, cuda_minor);

    // CUDA version → max PTX ISA version mapping
    let ptx_ver = match (cuda_major, cuda_minor) {
        (12, 0) => "8.0",
        (12, 1) => "8.1",
        (12, 2) => "8.2",
        (12, 3) => "8.3",
        (12, 4) => "8.4",
        (12, 5 | 6) => "8.5",
        (12, 7 | 8) => "8.6",
        (12, _) => "8.8",
        _ => return None,
    };
    Some(ptx_ver)
}

pub struct LayerParams {
    pub threshold: f32,
    pub learning_rate: f32,
    pub layer_id: u8,
    pub freq_hz: f64,
    pub weight_decay: f32,
    /// Precision learning rate for free-energy gradient (Phase 1.1)
    pub precision_eta: f32,
    /// Z-score threshold for surprise-gated sparse updates (Phase 5.2)
    pub surprise_threshold: f32,
    /// Weight for top-down prediction errors (Phase 5.1)
    pub top_down_weight: f32,
    // Phase 8: Adam optimizer hyperparameters
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_epsilon: f32,
    // Phase 8: Langevin exploration noise
    pub langevin_sigma: f32,
    // Phase 8: Gradient clipping
    pub grad_clip_norm: f32,
    // Phase 8: LR scheduling
    pub warmup_steps: f32,
    pub total_steps: f32,
}

pub struct CudaContext {
    device: Arc<CudaDevice>,
    params_dev: CudaSlice<f32>,
}

impl CudaContext {
    pub fn new(params: &LayerParams, has_above: bool) -> Result<Self, String> {
        let device = CudaDevice::new(0).map_err(|e| format!("CUDA device init failed: {e}"))?;

        let ptx = compile_kernel(&device)?;
        device
            .load_ptx(ptx, "belief", &["belief_update"])
            .map_err(|e| format!("PTX load failed: {e}"))?;

        let params_data: [f32; 15] = [
            params.threshold,        // [0]
            params.learning_rate,    // [1]
            params.layer_id as f32,  // [2]
            params.weight_decay,     // [3]
            if has_above { 1.0 } else { 0.0 }, // [4] has_above
            params.precision_eta,    // [5]
            params.surprise_threshold, // [6]
            params.top_down_weight,  // [7]
            params.adam_beta1,       // [8]
            params.adam_beta2,       // [9]
            params.adam_epsilon,     // [10]
            params.langevin_sigma,   // [11]
            params.grad_clip_norm,   // [12]
            params.warmup_steps,     // [13]
            params.total_steps,      // [14]
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
    /// Phase 8: passes all 17 buffers including GLU gates and Adam optimizer state.
    pub fn dispatch_belief_update(
        &self,
        belief: &mut BeliefSlot,
        below: &BeliefSlot,
        above: Option<&BeliefSlot>,
        weights: &mut [f32; WEIGHT_COUNT],
        bias: &mut [f32; STATE_DIM],
        gate_w: &mut [f32; WEIGHT_COUNT],
        gate_b: &mut [f32; STATE_DIM],
        adam_m_w: &mut [f32; WEIGHT_COUNT],
        adam_v_w: &mut [f32; WEIGHT_COUNT],
        adam_m_b: &mut [f32; STATE_DIM],
        adam_v_b: &mut [f32; STATE_DIM],
        adam_m_gw: &mut [f32; WEIGHT_COUNT],
        adam_v_gw: &mut [f32; WEIGHT_COUNT],
        adam_m_gb: &mut [f32; STATE_DIM],
        adam_v_gb: &mut [f32; STATE_DIM],
        adam_t: &mut u32,
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

        let dummy_above: BeliefSlot = unsafe { std::mem::zeroed() };
        let above_ref = above.unwrap_or(&dummy_above);
        let above_bytes = unsafe {
            std::slice::from_raw_parts(
                above_ref as *const BeliefSlot as *const u8,
                std::mem::size_of::<BeliefSlot>(),
            )
        };

        let belief_dev = self.device.htod_copy(belief_bytes.to_vec()).expect("belief htod");
        let below_dev = self.device.htod_copy(below_bytes.to_vec()).expect("below htod");
        let above_dev = self.device.htod_copy(above_bytes.to_vec()).expect("above htod");
        let weights_dev = self.device.htod_copy(weights.to_vec()).expect("weights htod");
        let bias_dev = self.device.htod_copy(bias.to_vec()).expect("bias htod");
        // Phase 8: GLU gate buffers
        let gate_w_dev = self.device.htod_copy(gate_w.to_vec()).expect("gate_w htod");
        let gate_b_dev = self.device.htod_copy(gate_b.to_vec()).expect("gate_b htod");
        // Phase 8: Adam optimizer moment buffers
        let adam_m_w_dev = self.device.htod_copy(adam_m_w.to_vec()).expect("adam_m_w htod");
        let adam_v_w_dev = self.device.htod_copy(adam_v_w.to_vec()).expect("adam_v_w htod");
        let adam_m_b_dev = self.device.htod_copy(adam_m_b.to_vec()).expect("adam_m_b htod");
        let adam_v_b_dev = self.device.htod_copy(adam_v_b.to_vec()).expect("adam_v_b htod");
        let adam_m_gw_dev = self.device.htod_copy(adam_m_gw.to_vec()).expect("adam_m_gw htod");
        let adam_v_gw_dev = self.device.htod_copy(adam_v_gw.to_vec()).expect("adam_v_gw htod");
        let adam_m_gb_dev = self.device.htod_copy(adam_m_gb.to_vec()).expect("adam_m_gb htod");
        let adam_v_gb_dev = self.device.htod_copy(adam_v_gb.to_vec()).expect("adam_v_gb htod");
        let adam_t_dev = self.device.htod_copy(vec![*adam_t]).expect("adam_t htod");

        let kernel = self.device.get_func("belief", "belief_update").expect("kernel not found");

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (STATE_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Phase 8: 17 buffers exceeds cudarc's LaunchAsync tuple limit,
        // so we build the raw kernel parameter pointer array manually.
        unsafe {
            use cudarc::driver::DevicePtr;
            let mut params_ptrs: Vec<*mut std::ffi::c_void> = vec![
                *belief_dev.device_ptr() as *mut std::ffi::c_void,
                *below_dev.device_ptr() as *mut std::ffi::c_void,
                *above_dev.device_ptr() as *mut std::ffi::c_void,
                *self.params_dev.device_ptr() as *mut std::ffi::c_void,
                *weights_dev.device_ptr() as *mut std::ffi::c_void,
                *bias_dev.device_ptr() as *mut std::ffi::c_void,
                *gate_w_dev.device_ptr() as *mut std::ffi::c_void,
                *gate_b_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_m_w_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_v_w_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_m_b_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_v_b_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_m_gw_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_v_gw_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_m_gb_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_v_gb_dev.device_ptr() as *mut std::ffi::c_void,
                *adam_t_dev.device_ptr() as *mut std::ffi::c_void,
            ];
            kernel
                .launch(cfg, &mut params_ptrs)
                .expect("kernel launch failed");
        }

        // Copy back all mutable buffers
        let belief_result = self.device.dtoh_sync_copy(&belief_dev).expect("belief dtoh");
        unsafe {
            std::ptr::copy_nonoverlapping(
                belief_result.as_ptr(),
                belief as *mut BeliefSlot as *mut u8,
                std::mem::size_of::<BeliefSlot>(),
            );
        }

        weights.copy_from_slice(&self.device.dtoh_sync_copy(&weights_dev).expect("weights dtoh"));
        bias.copy_from_slice(&self.device.dtoh_sync_copy(&bias_dev).expect("bias dtoh"));
        gate_w.copy_from_slice(&self.device.dtoh_sync_copy(&gate_w_dev).expect("gate_w dtoh"));
        gate_b.copy_from_slice(&self.device.dtoh_sync_copy(&gate_b_dev).expect("gate_b dtoh"));
        adam_m_w.copy_from_slice(&self.device.dtoh_sync_copy(&adam_m_w_dev).expect("adam_m_w dtoh"));
        adam_v_w.copy_from_slice(&self.device.dtoh_sync_copy(&adam_v_w_dev).expect("adam_v_w dtoh"));
        adam_m_b.copy_from_slice(&self.device.dtoh_sync_copy(&adam_m_b_dev).expect("adam_m_b dtoh"));
        adam_v_b.copy_from_slice(&self.device.dtoh_sync_copy(&adam_v_b_dev).expect("adam_v_b dtoh"));
        adam_m_gw.copy_from_slice(&self.device.dtoh_sync_copy(&adam_m_gw_dev).expect("adam_m_gw dtoh"));
        adam_v_gw.copy_from_slice(&self.device.dtoh_sync_copy(&adam_v_gw_dev).expect("adam_v_gw dtoh"));
        adam_m_gb.copy_from_slice(&self.device.dtoh_sync_copy(&adam_m_gb_dev).expect("adam_m_gb dtoh"));
        adam_v_gb.copy_from_slice(&self.device.dtoh_sync_copy(&adam_v_gb_dev).expect("adam_v_gb dtoh"));
        *adam_t = self.device.dtoh_sync_copy(&adam_t_dev).expect("adam_t dtoh")[0];
    }
}

/// Phase 8: Common Adam/Langevin/LR defaults shared across layers.
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPSILON: f32 = 1e-8;
const LANGEVIN_SIGMA: f32 = 0.01;
const GRAD_CLIP_NORM: f32 = 1.0;
const WARMUP_STEPS: f32 = 100.0;
const TOTAL_STEPS: f32 = 10000.0;

pub fn default_params(layer_id: u8) -> LayerParams {
    let base = |threshold, learning_rate, lid, freq_hz, weight_decay, precision_eta, surprise_threshold, top_down_weight| {
        LayerParams {
            threshold, learning_rate, layer_id: lid, freq_hz, weight_decay, precision_eta,
            surprise_threshold, top_down_weight,
            adam_beta1: ADAM_BETA1, adam_beta2: ADAM_BETA2, adam_epsilon: ADAM_EPSILON,
            langevin_sigma: LANGEVIN_SIGMA, grad_clip_norm: GRAD_CLIP_NORM,
            warmup_steps: WARMUP_STEPS, total_steps: TOTAL_STEPS,
        }
    };
    match layer_id {
        0 => base(0.1, 0.01, 0, 1000.0, 0.0001, 0.01, 2.0, 0.3),
        1 => base(0.08, 0.008, 1, 100.0, 0.0001, 0.01, 2.0, 0.3),
        2 => base(0.05, 0.005, 2, 100.0, 0.00005, 0.008, 2.0, 0.3),
        3 => base(0.05, 0.005, 3, 100.0, 0.00005, 0.008, 2.0, 0.3),
        4 => base(0.03, 0.003, 4, 1.0, 0.00001, 0.005, 2.5, 0.4),
        5 => base(0.02, 0.002, 5, 0.1, 0.00001, 0.003, 3.0, 0.5),
        6 => base(0.1, 0.001, 6, 30.0, 0.0001, 0.005, 2.0, 0.0),
        _ => base(0.1, 0.01, layer_id, 10.0, 0.0001, 0.01, 2.0, 0.3),
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

/// Phase 8: Total floats in a full checkpoint (weights + bias + GLU gates + Adam state).
const CHECKPOINT_FLOATS: usize = WEIGHT_COUNT * 6 + STATE_DIM * 6; // 6 weight arrays + 6 bias arrays

/// Load layer weights from a checkpoint file into raw slot pointer. Returns true if loaded.
/// Phase 8: loads full state including GLU gates and Adam optimizer moments.
/// Uses raw pointer to avoid &mut reference casting lint.
fn load_weights(
    dir: &str,
    layer_id: u8,
    slot: *mut qualia_types::LayerSlot,
) -> bool {
    let path = format!("{}/layer_{}_weights.bin", dir, layer_id);
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(_) => return false,
    };

    // Support legacy checkpoints (weights + bias only)
    let legacy_len = (WEIGHT_COUNT + STATE_DIM) * std::mem::size_of::<f32>();
    let full_len = CHECKPOINT_FLOATS * std::mem::size_of::<f32>() + std::mem::size_of::<u64>();

    unsafe {
        if data.len() == legacy_len {
            let floats: &[f32] =
                std::slice::from_raw_parts(data.as_ptr() as *const f32, WEIGHT_COUNT + STATE_DIM);
            (*slot).weights.copy_from_slice(&floats[..WEIGHT_COUNT]);
            (*slot).bias.copy_from_slice(&floats[WEIGHT_COUNT..]);
            eprintln!("qualia-l{}: loaded LEGACY checkpoint from {} (no GLU/Adam)", layer_id, path);
            return true;
        }

        if data.len() == full_len {
            let floats: &[f32] =
                std::slice::from_raw_parts(data.as_ptr() as *const f32, CHECKPOINT_FLOATS);
            let mut off = 0;
            (*slot).weights.copy_from_slice(&floats[off..off + WEIGHT_COUNT]); off += WEIGHT_COUNT;
            (*slot).bias.copy_from_slice(&floats[off..off + STATE_DIM]); off += STATE_DIM;
            (*slot).gate_w.copy_from_slice(&floats[off..off + WEIGHT_COUNT]); off += WEIGHT_COUNT;
            (*slot).gate_b.copy_from_slice(&floats[off..off + STATE_DIM]); off += STATE_DIM;
            (*slot).adam_m_w.copy_from_slice(&floats[off..off + WEIGHT_COUNT]); off += WEIGHT_COUNT;
            (*slot).adam_v_w.copy_from_slice(&floats[off..off + WEIGHT_COUNT]); off += WEIGHT_COUNT;
            (*slot).adam_m_b.copy_from_slice(&floats[off..off + STATE_DIM]); off += STATE_DIM;
            (*slot).adam_v_b.copy_from_slice(&floats[off..off + STATE_DIM]); off += STATE_DIM;
            (*slot).adam_m_gw.copy_from_slice(&floats[off..off + WEIGHT_COUNT]); off += WEIGHT_COUNT;
            (*slot).adam_v_gw.copy_from_slice(&floats[off..off + WEIGHT_COUNT]); off += WEIGHT_COUNT;
            (*slot).adam_m_gb.copy_from_slice(&floats[off..off + STATE_DIM]); off += STATE_DIM;
            (*slot).adam_v_gb.copy_from_slice(&floats[off..off + STATE_DIM]);
            let t_bytes = &data[CHECKPOINT_FLOATS * std::mem::size_of::<f32>()..];
            let adam_t = u64::from_le_bytes(t_bytes[..8].try_into().unwrap());
            (*slot).adam_t.store(adam_t, std::sync::atomic::Ordering::Relaxed);
            eprintln!("qualia-l{}: loaded FULL checkpoint from {} (adam_t={})", layer_id, path, adam_t);
            return true;
        }
    }

    eprintln!(
        "qualia-l{}: checkpoint {} has wrong size ({}, expected {} or {}), skipping",
        layer_id, path, data.len(), legacy_len, full_len
    );
    false
}

/// Save layer weights to a checkpoint file (atomic: write .tmp then rename).
/// Phase 8: saves full state including GLU gates and Adam optimizer moments.
fn save_weights(
    dir: &str,
    layer_id: u8,
    slot: &qualia_types::LayerSlot,
) {
    let path = format!("{}/layer_{}_weights.bin", dir, layer_id);
    let tmp = format!("{}.tmp", path);

    let total_bytes = CHECKPOINT_FLOATS * std::mem::size_of::<f32>() + std::mem::size_of::<u64>();
    let mut data = Vec::with_capacity(total_bytes);

    for arr in [slot.weights.as_slice(), slot.bias.as_slice(),
                slot.gate_w.as_slice(), slot.gate_b.as_slice(),
                slot.adam_m_w.as_slice(), slot.adam_v_w.as_slice(),
                slot.adam_m_b.as_slice(), slot.adam_v_b.as_slice(),
                slot.adam_m_gw.as_slice(), slot.adam_v_gw.as_slice(),
                slot.adam_m_gb.as_slice(), slot.adam_v_gb.as_slice()] {
        for &v in arr {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    let adam_t = slot.adam_t.load(std::sync::atomic::Ordering::Relaxed);
    data.extend_from_slice(&adam_t.to_le_bytes());

    if let Err(e) = std::fs::write(&tmp, &data) {
        eprintln!("qualia-l{}: failed to write checkpoint {}: {}", layer_id, tmp, e);
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        eprintln!("qualia-l{}: failed to rename checkpoint: {}", layer_id, e);
        return;
    }
    eprintln!("qualia-l{}: saved FULL checkpoint to {} ({} bytes, adam_t={})", layer_id, path, data.len(), adam_t);
}

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
    // Phase 5.1: layers 0-5 have a layer above; L6 (sensor) is the top
    let has_above = (layer_id as usize) < NUM_LAYERS - 1;
    let cuda = CudaContext::new(&params, has_above).unwrap_or_else(|e| {
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

    // Phase 5.1: reader for the layer ABOVE (for top-down prediction errors)
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

    let checkpoint_dir = std::env::var("QUALIA_CHECKPOINT_DIR").ok();

    // Initialize weights — load checkpoint or fall back to identity matrix
    // Phase 8: also initialize GLU gates and Adam state
    {
        let slot_raw = my_slot as *const qualia_types::LayerSlot as *mut qualia_types::LayerSlot;

        let loaded = checkpoint_dir
            .as_deref()
            .map(|dir| load_weights(dir, layer_id, slot_raw))
            .unwrap_or(false);

        if !loaded {
            unsafe {
                for i in 0..STATE_DIM {
                    for j in 0..STATE_DIM {
                        let idx = i * STATE_DIM + j;
                        (*slot_raw).weights[idx] = if i == j { 1.0 } else { 0.0 };
                        (*slot_raw).gate_w[idx] = 0.0;
                        (*slot_raw).adam_m_w[idx] = 0.0;
                        (*slot_raw).adam_v_w[idx] = 0.0;
                        (*slot_raw).adam_m_gw[idx] = 0.0;
                        (*slot_raw).adam_v_gw[idx] = 0.0;
                    }
                    (*slot_raw).bias[i] = 0.0;
                    (*slot_raw).gate_b[i] = 0.0;
                    (*slot_raw).adam_m_b[i] = 0.0;
                    (*slot_raw).adam_v_b[i] = 0.0;
                    (*slot_raw).adam_m_gb[i] = 0.0;
                    (*slot_raw).adam_v_gb[i] = 0.0;
                }
                (*slot_raw).adam_t.store(0, std::sync::atomic::Ordering::Relaxed);
            }
            eprintln!("qualia-{name}: initialized weights as identity + GLU/Adam zeroed (no checkpoint)");
        }
    }

    // Register SIGTERM handler to save weights on shutdown
    let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let shutdown_flag = shutdown.clone();
        if let Err(e) = unsafe {
            signal_hook::low_level::register(signal_hook::consts::SIGTERM, move || {
                shutdown_flag.store(true, std::sync::atomic::Ordering::SeqCst);
            })
        } {
            eprintln!("qualia-{name}: failed to register SIGTERM handler: {e}");
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

    if has_above {
        eprintln!("qualia-{name}: top-down from L{} (weight={:.2})", layer_id + 1, params.top_down_weight);
    }

    eprintln!(
        "qualia-{name}: running at {:.1} Hz, prec_eta={:.3}, surprise_gate={:.1} (CUDA)",
        params.freq_hz, params.precision_eta, params.surprise_threshold
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
        if shutdown.load(std::sync::atomic::Ordering::SeqCst) {
            eprintln!("qualia-{name}: SIGTERM received, saving weights...");
            if let Some(ref dir) = checkpoint_dir {
                let slot_ref = unsafe {
                    &*(my_slot as *const qualia_types::LayerSlot)
                };
                save_weights(dir, layer_id, slot_ref);
            }
            break;
        }

        let cycle_start = Instant::now();
        cycle_count += 1;

        let below = *reader.read();

        // Phase 5.1: Read belief from the layer above (if it exists)
        let above_belief = above_reader.as_ref().map(|r| *r.read());

        // Phase 8: Extract all mutable buffers from the slot via raw pointer
        let slot_raw = my_slot as *const qualia_types::LayerSlot as *mut qualia_types::LayerSlot;
        let mut adam_t_val = unsafe { (*slot_raw).adam_t.load(std::sync::atomic::Ordering::Relaxed) as u32 };

        let buf = writer.back_buffer();
        buf.layer = layer_id;
        unsafe {
            cuda.dispatch_belief_update(
                buf, &below, above_belief.as_ref(),
                &mut (*slot_raw).weights, &mut (*slot_raw).bias,
                &mut (*slot_raw).gate_w, &mut (*slot_raw).gate_b,
                &mut (*slot_raw).adam_m_w, &mut (*slot_raw).adam_v_w,
                &mut (*slot_raw).adam_m_b, &mut (*slot_raw).adam_v_b,
                &mut (*slot_raw).adam_m_gw, &mut (*slot_raw).adam_v_gw,
                &mut (*slot_raw).adam_m_gb, &mut (*slot_raw).adam_v_gb,
                &mut adam_t_val,
            );
            // Write back the updated adam_t
            (*slot_raw).adam_t.store(adam_t_val as u64, std::sync::atomic::Ordering::Relaxed);
        }

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
