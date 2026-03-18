use qualia_shm::{ShmRegion, LayerReader, NUM_LAYERS, HealthReport};
use std::io::Write;
use std::time::{Duration, Instant};

fn main() {
    let shm_name = std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".to_string());
    eprintln!("qualia-health: opening shm '{shm_name}'");

    let shm = ShmRegion::open(&shm_name).expect("Failed to open shm");
    let mut stdout = std::io::stdout().lock();
    let tick = Duration::from_millis(100); // 10 Hz

    loop {
        let start = Instant::now();

        for layer in 0..NUM_LAYERS {
            let slot = shm.layer_slot(layer);
            let reader = LayerReader::new(slot);
            let belief = reader.read();

            let report = HealthReport {
                layer: layer as u8,
                compression: belief.compression,
                _pad: [0; 2],
                vfe: belief.vfe,
                challenge_vfe: belief.challenge_vfe,
                confirm_streak: belief.confirm_streak,
                cycle_us: belief.cycle_us,
                timestamp_ns: belief.timestamp_ns,
            };

            // Write as raw bytes
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &report as *const HealthReport as *const u8,
                    std::mem::size_of::<HealthReport>(),
                )
            };
            let _ = stdout.write_all(bytes);
        }
        let _ = stdout.flush();

        let elapsed = start.elapsed();
        if elapsed < tick {
            std::thread::sleep(tick - elapsed);
        }
    }
}
