use qualia_shm::{LayerWriter, ShmRegion};
use qualia_types::STATE_DIM;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::SystemTime;

/// Sensor layer — L6 holds raw sensory input that L0 reads from.
const SENSOR_LAYER: usize = 6;

/// Phase 7.5: Multi-scale DCT pyramid.
/// Capture at 16×16, but extract features at two scales:
///   - 8×8 coarse (global structure, 16 coefficients)
///   - 16×16 fine (local detail, 48 coefficients)
/// Total: 16 + 48 = 64 features = STATE_DIM
const GRID_SIZE: usize = 16;
const COARSE_FEATURES: usize = 16;  // from 8×8 DCT
const FINE_FEATURES: usize = 48;    // from 16×16 DCT

fn main() {
    let shm_name =
        std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".to_string());
    eprintln!("qualia-camera: opening shm '{shm_name}'");

    let shm = ShmRegion::open(&shm_name).unwrap_or_else(|e| {
        panic!("qualia-camera: failed to open shm: {e}");
    });

    let slot = shm.layer_slot(SENSOR_LAYER);
    let writer = LayerWriter::new(slot);

    {
        let buf = writer.back_buffer();
        buf.layer = SENSOR_LAYER as u8;
        for i in 0..STATE_DIM {
            buf.precision[i] = 1.0;
        }
        writer.publish();
    }

    eprintln!("qualia-camera: Phase 7.5 multi-scale DCT pyramid (8×8 coarse + 16×16 fine → 64 features)");

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

/// Apply 2D DCT to a 16×16 image and return top N coefficients in zigzag order.
fn dct_2d_zigzag(pixels: &[f32; 256], basis: &[[f32; 16]; 16], n_coeffs: usize) -> Vec<f32> {
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

    // Zigzag scan for 16×16
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

    let mut features = vec![0.0f32; n_coeffs];
    for (i, &idx) in zigzag_order.iter().enumerate().take(n_coeffs) {
        if idx < 256 {
            features[i] = dct[idx];
        }
    }
    features
}

/// Phase 7.5: Downsample 16×16 to 8×8 by averaging 2×2 blocks.
fn downsample_8x8(pixels: &[f32; 256]) -> [f32; 64] {
    let mut coarse = [0.0f32; 64];
    for by in 0..8 {
        for bx in 0..8 {
            let mut sum = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    sum += pixels[(by * 2 + dy) * 16 + bx * 2 + dx];
                }
            }
            coarse[by * 8 + bx] = sum * 0.25;
        }
    }
    coarse
}

/// Apply 2D DCT to an 8×8 image and return top N coefficients in zigzag order.
fn dct_8x8_zigzag(pixels: &[f32; 64], basis: &[[f32; 8]; 8], n_coeffs: usize) -> Vec<f32> {
    let mut row_dct = [0.0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;
            for x in 0..8 {
                sum += pixels[y * 8 + x] * basis[u][x];
            }
            row_dct[y * 8 + u] = sum;
        }
    }

    let mut dct = [0.0f32; 64];
    for u in 0..8 {
        for v in 0..8 {
            let mut sum = 0.0f32;
            for y in 0..8 {
                sum += row_dct[y * 8 + u] * basis[v][y];
            }
            dct[v * 8 + u] = sum;
        }
    }

    // Zigzag scan for 8×8
    let zigzag_8x8: [usize; 64] = [
         0,  1,  8,  2,  9, 16,  3, 10,
        17, 24,  4, 11, 18, 25, 32,  5,
        12, 19, 26, 33, 40,  6, 13, 20,
        27, 34, 41, 48,  7, 14, 21, 28,
        35, 42, 49, 56, 15, 22, 29, 36,
        43, 50, 57, 23, 30, 37, 44, 51,
        58, 31, 38, 45, 52, 59, 39, 46,
        53, 60, 47, 54, 61, 55, 62, 63,
    ];

    let mut features = vec![0.0f32; n_coeffs];
    for (i, &idx) in zigzag_8x8.iter().enumerate().take(n_coeffs) {
        if idx < 64 {
            features[i] = dct[idx];
        }
    }
    features
}

/// Phase 7.5: Multi-scale DCT feature extraction.
/// Returns 64 features: [16 coarse (8×8 global) | 48 fine (16×16 detail)]
fn multiscale_dct_features(
    pixels: &[f32; 256],
    basis_16: &[[f32; 16]; 16],
    basis_8: &[[f32; 8]; 8],
) -> [f32; 64] {
    // Coarse scale: 8×8 (global structure — blob shapes, overall brightness gradient)
    let coarse_pixels = downsample_8x8(pixels);
    let coarse = dct_8x8_zigzag(&coarse_pixels, basis_8, COARSE_FEATURES);

    // Fine scale: 16×16 (local detail — edges, textures, fine structure)
    let fine = dct_2d_zigzag(pixels, basis_16, FINE_FEATURES);

    let mut features = [0.0f32; 64];
    for i in 0..COARSE_FEATURES {
        features[i] = coarse[i];
    }
    for i in 0..FINE_FEATURES {
        features[COARSE_FEATURES + i] = fine[i];
    }
    features
}

fn run_camera_loop(writer: &LayerWriter, mut reader: impl Read) {
    let frame_size = GRID_SIZE * GRID_SIZE;
    let mut frame_buf = vec![0u8; frame_size];
    let mut frame_count: u64 = 0;

    // Precompute DCT bases for both scales
    let basis_16 = precompute_dct_basis::<16>();
    let basis_8 = precompute_dct_basis::<8>();

    // EMA for variance tracking (Phase 1.3: dynamic precision)
    let mut pixel_ema = [0.0f32; 64];
    let mut pixel_var = [0.0f32; 64];

    // Phase 7.5: Frame-to-frame temporal difference for change detection
    let mut prev_features = [0.0f32; 64];

    eprintln!("qualia-camera: webcam streaming (multi-scale DCT → 64 features)...");

    loop {
        if reader.read_exact(&mut frame_buf).is_err() {
            eprintln!("qualia-camera: ffmpeg stream ended");
            break;
        }

        let mut pixels = [0.0f32; 256];
        for i in 0..256 {
            pixels[i] = frame_buf[i] as f32 / 255.0;
        }

        // Phase 7.5: Multi-scale DCT pyramid
        let features = multiscale_dct_features(&pixels, &basis_16, &basis_8);

        let buf = writer.back_buffer();
        buf.layer = SENSOR_LAYER as u8;

        for i in 0..STATE_DIM {
            buf.mean[i] = features[i];
        }

        // Phase 7.5: Adaptive precision from both variance AND temporal change
        let var_alpha = 0.05f32;
        for i in 0..STATE_DIM {
            let val = features[i];
            pixel_ema[i] = pixel_ema[i] * (1.0 - var_alpha) + val * var_alpha;
            let dev = val - pixel_ema[i];
            pixel_var[i] = pixel_var[i] * (1.0 - var_alpha) + dev * dev * var_alpha;

            // Temporal change: how much this feature changed from last frame
            let temporal_change = (val - prev_features[i]).abs();

            // Precision: inversely proportional to variance + temporal change
            // High variance or rapid change → low precision (uncertain)
            let var_clamped = pixel_var[i].max(0.001);
            let temporal_factor = 1.0 + temporal_change * 5.0;
            buf.precision[i] = (1.0 / (var_clamped * 10.0 * temporal_factor + 0.1)).clamp(0.5, 10.0);
        }

        prev_features = features;

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
                "qualia-camera: {} frames, mean_var={:.4}, mean_prec={:.2} (multi-scale)",
                frame_count, mean_var, mean_prec
            );
        }
    }
}

fn run_synthetic_loop(writer: &LayerWriter) {
    use std::time::{Duration, Instant};

    eprintln!("qualia-camera: generating synthetic visual patterns (multi-scale DCT)...");

    let basis_16 = precompute_dct_basis::<16>();
    let basis_8 = precompute_dct_basis::<8>();
    let mut t: f64 = 0.0;
    let tick = Duration::from_millis(33);

    let mut pixel_ema = [0.0f32; 64];
    let mut pixel_var = [0.0f32; 64];
    let mut prev_features = [0.0f32; 64];

    loop {
        let start = Instant::now();

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

        let features = multiscale_dct_features(&pixels, &basis_16, &basis_8);

        let buf = writer.back_buffer();
        buf.layer = SENSOR_LAYER as u8;

        let var_alpha = 0.05f32;
        for i in 0..STATE_DIM {
            buf.mean[i] = features[i];
            let val = features[i];
            pixel_ema[i] = pixel_ema[i] * (1.0 - var_alpha) + val * var_alpha;
            let dev = val - pixel_ema[i];
            pixel_var[i] = pixel_var[i] * (1.0 - var_alpha) + dev * dev * var_alpha;
            let temporal_change = (val - prev_features[i]).abs();
            let var_clamped = pixel_var[i].max(0.001);
            let temporal_factor = 1.0 + temporal_change * 5.0;
            buf.precision[i] = (1.0 / (var_clamped * 10.0 * temporal_factor + 0.1)).clamp(0.5, 10.0);
        }

        prev_features = features;

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
