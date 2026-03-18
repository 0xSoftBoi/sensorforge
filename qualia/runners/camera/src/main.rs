use qualia_shm::{LayerWriter, ShmRegion};
use qualia_types::STATE_DIM;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::SystemTime;

/// Sensor layer — L6 holds raw sensory input that L0 reads from.
const SENSOR_LAYER: usize = 6;

/// Phase 2.1: Capture at 16×16 for DCT, take top 64 coefficients.
const GRID_SIZE: usize = 16; // 16×16 capture → DCT → 64 features

fn main() {
    let shm_name =
        std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".to_string());
    eprintln!("qualia-camera: opening shm '{shm_name}'");

    let shm = ShmRegion::open(&shm_name).unwrap_or_else(|e| {
        panic!("qualia-camera: failed to open shm: {e}");
    });

    let slot = shm.layer_slot(SENSOR_LAYER);
    let writer = LayerWriter::new(slot);

    // Initialize the sensor slot
    {
        let buf = writer.back_buffer();
        buf.layer = SENSOR_LAYER as u8;
        for i in 0..STATE_DIM {
            buf.precision[i] = 1.0;
        }
        writer.publish();
    }

    eprintln!("qualia-camera: starting webcam capture via ffmpeg...");
    eprintln!("qualia-camera: 16x16 grayscale → DCT → 64 features → L{SENSOR_LAYER}");

    // Try ffmpeg first, fall back to synthetic data
    match start_ffmpeg() {
        Ok(mut child) => {
            let stderr_pipe = child.stderr.take();
            let stdout = child.stdout.take().expect("ffmpeg stdout");
            run_camera_loop(&writer, stdout);
            if let Some(mut stderr) = stderr_pipe {
                let mut err_buf = String::new();
                let _ = stderr.read_to_string(&mut err_buf);
                if !err_buf.is_empty() {
                    eprintln!("qualia-camera: ffmpeg stderr: {}", err_buf.trim());
                }
            }
            let _ = child.kill();
            eprintln!("qualia-camera: ffmpeg stream ended, switching to synthetic...");
        }
        Err(e) => {
            eprintln!("qualia-camera: ffmpeg not available ({e})");
            if cfg!(target_os = "macos") {
                eprintln!("qualia-camera: install with: brew install ffmpeg");
            } else {
                eprintln!("qualia-camera: install with: sudo apt install ffmpeg");
            }
        }
    }

    eprintln!("qualia-camera: falling back to synthetic sensor data...");
    run_synthetic_loop(&writer);
}

fn start_ffmpeg() -> Result<std::process::Child, std::io::Error> {
    let device = std::env::var("CAMERA_DEVICE").unwrap_or_else(|_| {
        if cfg!(target_os = "macos") { "0".into() }
        else { "/dev/video0".into() }
    });

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error"]);
    if cfg!(target_os = "macos") {
        cmd.args(["-f", "avfoundation", "-framerate", "30", "-video_size", "640x480", "-i", &device]);
    } else {
        cmd.args(["-f", "v4l2", "-video_size", "640x480", "-i", &device]);
    }
    // Phase 2.1: Capture at 16×16 for DCT (was 8×8)
    // Two outputs from one device:
    //   1) 16x16 grayscale rawvideo → pipe:1 (SHM at 30fps)
    //   2) Full-res JPEG snapshot → /tmp/qualia-camera-latest.jpg (vision at 1fps)
    cmd.args([
        "-filter_complex", "[0:v]split=2[shm][snap];[shm]scale=16:16,format=gray[low]",
        "-map", "[low]", "-f", "rawvideo", "-r", "30", "pipe:1",
        "-map", "[snap]", "-r", "1", "-update", "1", "-q:v", "5",
        "/tmp/qualia-camera-latest.jpg",
    ])
       .stdout(Stdio::piped())
       .stderr(Stdio::piped());
    cmd.spawn()
}

/// Precompute DCT-II basis for NxN transform.
/// basis[u][x] = cos(π * (2*x + 1) * u / (2*N)) * norm
fn precompute_dct_basis<const N: usize>() -> [[f32; N]; N] {
    let mut basis = [[0.0f32; N]; N];
    let n = N as f32;
    for u in 0..N {
        let norm = if u == 0 {
            (1.0 / n).sqrt()
        } else {
            (2.0 / n).sqrt()
        };
        for x in 0..N {
            basis[u][x] = norm * (std::f32::consts::PI * (2.0 * x as f32 + 1.0) * u as f32 / (2.0 * n)).cos();
        }
    }
    basis
}

/// Apply 2D DCT to a 16×16 grayscale image, return top 64 coefficients in zigzag order.
/// This extracts frequency-domain features: edges, textures, spatial structure.
fn dct_16x16_to_64(pixels: &[f32; 256], basis: &[[f32; 16]; 16]) -> [f32; 64] {
    // Step 1: 2D DCT via separable 1D transforms
    // Row transform
    let mut row_dct = [0.0f32; 256];
    for y in 0..16 {
        for u in 0..16 {
            let mut sum = 0.0f32;
            for x in 0..16 {
                sum += pixels[y * 16 + x] * basis[u][x];
            }
            row_dct[y * 16 + u] = sum;
        }
    }

    // Column transform
    let mut dct = [0.0f32; 256];
    for u in 0..16 {
        for v in 0..16 {
            let mut sum = 0.0f32;
            for y in 0..16 {
                sum += row_dct[y * 16 + u] * basis[v][y];
            }
            dct[v * 16 + u] = sum;
        }
    }

    // Step 2: Zigzag scan to get top 64 coefficients (low-frequency first)
    let zigzag_order: [usize; 64] = [
         0,  1, 16,  2, 17, 32,  3, 18,
        33, 48,  4, 19, 34, 49, 64,  5,
        20, 35, 50, 65, 80,  6, 21, 36,
        51, 66, 81, 96,  7, 22, 37, 52,
        67, 82, 97,112,  8, 23, 38, 53,
        68, 83, 98,113,128,  9, 24, 39,
        54, 69, 84, 99,114,129,144, 10,
        25, 40, 55, 70, 85,100,115,130,
    ];

    let mut features = [0.0f32; 64];
    for (i, &idx) in zigzag_order.iter().enumerate() {
        if idx < 256 {
            features[i] = dct[idx];
        }
    }
    features
}

fn run_camera_loop(writer: &LayerWriter, mut reader: impl Read) {
    let frame_size = GRID_SIZE * GRID_SIZE; // 256 bytes per frame (16×16)
    let mut frame_buf = vec![0u8; frame_size];
    let mut frame_count: u64 = 0;

    // Precompute DCT basis once
    let basis = precompute_dct_basis::<16>();

    // EMA for variance tracking (Phase 1.3: dynamic precision)
    let mut pixel_ema = [0.0f32; 64];
    let mut pixel_var = [0.0f32; 64];

    eprintln!("qualia-camera: webcam streaming (16×16 → DCT → 64 features)...");

    loop {
        // Read one 16×16 grayscale frame from ffmpeg
        if reader.read_exact(&mut frame_buf).is_err() {
            eprintln!("qualia-camera: ffmpeg stream ended");
            break;
        }

        // Convert grayscale bytes to normalized f32 [0.0, 1.0]
        let mut pixels = [0.0f32; 256];
        for i in 0..256 {
            pixels[i] = frame_buf[i] as f32 / 255.0;
        }

        // Phase 2.1: Apply 2D DCT and extract top 64 frequency features
        let features = dct_16x16_to_64(&pixels, &basis);

        let buf = writer.back_buffer();
        buf.layer = SENSOR_LAYER as u8;

        for i in 0..STATE_DIM {
            buf.mean[i] = features[i];
        }

        // Phase 1.3: Dynamic precision from pixel variance
        // High variance → lower precision (uncertain), low variance → higher precision
        let var_alpha = 0.05f32;
        for i in 0..STATE_DIM {
            let val = features[i];
            pixel_ema[i] = pixel_ema[i] * (1.0 - var_alpha) + val * var_alpha;
            let dev = val - pixel_ema[i];
            pixel_var[i] = pixel_var[i] * (1.0 - var_alpha) + dev * dev * var_alpha;

            // Precision inversely proportional to variance
            // High variance → low precision (0.5), low variance → high precision (10.0)
            let var_clamped = pixel_var[i].max(0.001);
            buf.precision[i] = (1.0 / (var_clamped * 10.0 + 0.1)).clamp(0.5, 10.0);
        }

        buf.vfe = 0.0;
        buf.challenge_vfe = 0.0;
        buf.timestamp_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        buf.cycle_us = 0;

        writer.publish();

        frame_count += 1;
        if frame_count % 300 == 0 {
            let mean_var: f32 = pixel_var.iter().sum::<f32>() / STATE_DIM as f32;
            let mean_prec: f32 = (0..STATE_DIM).map(|i| buf.precision[i]).sum::<f32>() / STATE_DIM as f32;
            eprintln!(
                "qualia-camera: {} frames, mean_var={:.4}, mean_prec={:.2}",
                frame_count, mean_var, mean_prec
            );
        }
    }
}

fn run_synthetic_loop(writer: &LayerWriter) {
    use std::time::{Duration, Instant};

    eprintln!("qualia-camera: generating synthetic visual patterns (DCT)...");

    let basis = precompute_dct_basis::<16>();
    let mut t: f64 = 0.0;
    let tick = Duration::from_millis(33); // ~30fps

    let mut pixel_ema = [0.0f32; 64];
    let mut pixel_var = [0.0f32; 64];

    loop {
        let start = Instant::now();

        // Generate 16×16 synthetic scene
        let mut pixels = [0.0f32; 256];
        for i in 0..256 {
            let x = (i % 16) as f64 / 16.0;
            let y = (i / 16) as f64 / 16.0;

            let cx = 0.5 + 0.3 * (t * 0.5).sin();
            let cy = 0.5 + 0.3 * (t * 0.3).cos();
            let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            let blob = (1.0 - dist * 3.0).max(0.0);
            let edge = (1.0 - (x - 0.5 - 0.4 * (t * 0.2).sin()).abs() * 5.0).max(0.0);
            let noise = ((t * 100.0 + i as f64 * 7.3).sin() * 0.5 + 0.5) * 0.1;

            pixels[i] = (blob * 0.6 + edge * 0.3 + noise) as f32;
        }

        let features = dct_16x16_to_64(&pixels, &basis);

        let buf = writer.back_buffer();
        buf.layer = SENSOR_LAYER as u8;

        let var_alpha = 0.05f32;
        for i in 0..STATE_DIM {
            buf.mean[i] = features[i];
            let val = features[i];
            pixel_ema[i] = pixel_ema[i] * (1.0 - var_alpha) + val * var_alpha;
            let dev = val - pixel_ema[i];
            pixel_var[i] = pixel_var[i] * (1.0 - var_alpha) + dev * dev * var_alpha;
            let var_clamped = pixel_var[i].max(0.001);
            buf.precision[i] = (1.0 / (var_clamped * 10.0 + 0.1)).clamp(0.5, 10.0);
        }

        buf.vfe = 0.0;
        buf.challenge_vfe = 0.0;
        buf.timestamp_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        writer.publish();

        t += 0.033;
        let elapsed = start.elapsed();
        if elapsed < tick {
            std::thread::sleep(tick - elapsed);
        }
    }
}
