use qualia_shm::{LayerWriter, ShmRegion};
use qualia_types::STATE_DIM;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::SystemTime;

/// Sensor layer — L6 holds raw sensory input that L0 reads from.
const SENSOR_LAYER: usize = 6;

/// Capture resolution before downsampling (ffmpeg handles the scaling)
const GRID_SIZE: usize = 8; // 8x8 = 64 = STATE_DIM

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
            buf.precision[i] = 1.0; // sensor data has high precision
        }
        writer.publish();
    }

    eprintln!("qualia-camera: starting webcam capture via ffmpeg...");
    eprintln!("qualia-camera: 8x8 grayscale → 64 floats → L{SENSOR_LAYER}");

    // Try ffmpeg first, fall back to synthetic data
    match start_ffmpeg() {
        Ok(mut child) => {
            let stdout = child.stdout.take().expect("ffmpeg stdout");
            run_camera_loop(&writer, stdout);
            let _ = child.kill();
            // ffmpeg stream ended (device busy, unplugged, etc) — fall through to synthetic
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

    // Always fall through to synthetic if we get here
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
    cmd.args(["-vf", "scale=8:8,format=gray", "-f", "rawvideo", "-r", "30", "pipe:1"])
       .stdout(Stdio::piped())
       .stderr(Stdio::piped());
    cmd.spawn()
}

fn run_camera_loop(writer: &LayerWriter, mut reader: impl Read) {
    let frame_size = GRID_SIZE * GRID_SIZE; // 64 bytes per frame
    let mut frame_buf = vec![0u8; frame_size];
    let mut frame_count: u64 = 0;

    eprintln!("qualia-camera: webcam streaming...");

    loop {
        // Read one 8x8 grayscale frame from ffmpeg
        if reader.read_exact(&mut frame_buf).is_err() {
            eprintln!("qualia-camera: ffmpeg stream ended");
            break;
        }

        let buf = writer.back_buffer();
        buf.layer = SENSOR_LAYER as u8;

        // Convert grayscale bytes to normalized f32 [0.0, 1.0]
        for i in 0..STATE_DIM {
            buf.mean[i] = frame_buf[i] as f32 / 255.0;
        }

        // Sensor precision: high (we trust what we see)
        for i in 0..STATE_DIM {
            buf.precision[i] = 10.0;
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
            eprintln!("qualia-camera: {} frames captured", frame_count);
        }
    }
}

fn run_synthetic_loop(writer: &LayerWriter) {
    use std::time::{Duration, Instant};

    eprintln!("qualia-camera: generating synthetic visual patterns...");

    let mut t: f64 = 0.0;
    let tick = Duration::from_millis(33); // ~30fps

    loop {
        let start = Instant::now();
        let buf = writer.back_buffer();
        buf.layer = SENSOR_LAYER as u8;

        // Generate moving patterns — like what a baby might see:
        // drifting light patches, edges, oscillating blobs
        for i in 0..STATE_DIM {
            let x = (i % 8) as f64 / 8.0;
            let y = (i / 8) as f64 / 8.0;

            // Slow-moving blob
            let cx = 0.5 + 0.3 * (t * 0.5).sin();
            let cy = 0.5 + 0.3 * (t * 0.3).cos();
            let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            let blob = (1.0 - dist * 3.0).max(0.0);

            // Moving edge
            let edge = (1.0 - (x - 0.5 - 0.4 * (t * 0.2).sin()).abs() * 5.0).max(0.0);

            // Flicker / noise
            let noise = ((t * 100.0 + i as f64 * 7.3).sin() * 0.5 + 0.5) * 0.1;

            buf.mean[i] = (blob * 0.6 + edge * 0.3 + noise) as f32;
            buf.precision[i] = 10.0;
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
