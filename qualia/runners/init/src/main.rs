use qualia_shm::ShmRegion;
use qualia_ipc::ControlListener;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static RUNNING: AtomicBool = AtomicBool::new(true);

static RUNNER_NAMES: &[&str] = &[
    "qualia-camera",
    "qualia-l0-superposition",
    "qualia-l1-belief",
    "qualia-l2-belief",
    "qualia-l3-belief",
    "qualia-l4-behavior",
    "qualia-l5-behavior",
    "qualia-health",
    "qualia-vision",
    "qualia-coach",
];

fn main() {
    let shm_name = std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".to_string());
    let sock_path = std::env::var("QUALIA_SOCK_PATH")
        .unwrap_or_else(|_| "/tmp/qualia_body.sock".to_string());

    eprintln!("╔══════════════════════════════════════════╗");
    eprintln!("║         QUALIA ENGINE v0.1.0             ║");
    eprintln!("║         body_double @ apple_silicon      ║");
    eprintln!("╚══════════════════════════════════════════╝");
    eprintln!();

    // Always clean up stale shm first — makes restart painless
    cleanup_stale_shm(&shm_name);

    eprintln!("[init] Creating shared memory '{shm_name}'...");
    let _shm = ShmRegion::create(&shm_name).unwrap_or_else(|e| {
        panic!("[init] Failed to create shm: {e}");
    });
    eprintln!("[init] Shared memory created: 64 MB");

    eprintln!("[init] Binding control socket '{sock_path}'...");
    let control = ControlListener::bind(&sock_path).expect("Failed to bind control socket");

    // Catch SIGINT and SIGTERM for graceful shutdown
    unsafe {
        libc::signal(libc::SIGINT, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGTERM, signal_handler as libc::sighandler_t);
    }

    let self_path = std::env::current_exe().expect("Cannot get self path");
    let bin_dir = self_path.parent().expect("Cannot get bin dir");

    // Spawn all runners in our process group
    let mut children: Vec<(String, Child)> = Vec::new();
    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());

    for name in RUNNER_NAMES {
        let bin_path = bin_dir.join(name);
        eprintln!("[init] Spawning {name}...");

        let mut cmd = Command::new(&bin_path);
        cmd.env("QUALIA_SHM_NAME", &shm_name)
            .env("RUST_LOG", &rust_log)
            .stderr(Stdio::inherit());

        // Suppress stdout for runners that don't need it
        if *name == "qualia-health" || *name == "qualia-camera" || *name == "qualia-vision" {
            cmd.stdout(Stdio::null());
        }

        match cmd.spawn() {
            Ok(child) => {
                eprintln!("[init]   pid {}", child.id());
                children.push((name.to_string(), child));
            }
            Err(e) => {
                eprintln!("[init] WARNING: {name}: {e}");
            }
        }
    }

    eprintln!();
    eprintln!("[init] {} runners launched. Ctrl+C to shutdown.", children.len());
    eprintln!("[init] Run 'qualia-watch' in another terminal for the TUI.");
    eprintln!();

    // Supervisor loop — check every 50ms for fast Ctrl+C response
    while RUNNING.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(50));

        // Check control socket
        if let Some(mut stream) = control.try_accept() {
            if let Ok((msg, _)) = stream.recv() {
                match msg {
                    qualia_ipc::ControlMsg::Shutdown | qualia_ipc::ControlMsg::Estop => {
                        eprintln!("[init] {:?} received", msg);
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    // Fast shutdown: SIGTERM all children, then SIGKILL stragglers
    shutdown_children(&mut children);

    // Clean up shm so next restart is clean
    eprintln!("[init] Cleaning up...");
    drop(_shm); // unlinks shm
    let _ = std::fs::remove_file(&sock_path);
    eprintln!("[init] Done.");
}

fn shutdown_children(children: &mut Vec<(String, Child)>) {
    eprintln!();
    eprintln!("[init] Shutting down {} processes...", children.len());

    // First: SIGTERM all (gentle)
    for (name, child) in children.iter() {
        unsafe { libc::kill(child.id() as i32, libc::SIGTERM); }
        eprintln!("[init]   SIGTERM → {name} (pid {})", child.id());
    }

    // Wait up to 2 seconds for graceful exit
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let all_done = children.iter_mut().all(|(_, child)| {
            matches!(child.try_wait(), Ok(Some(_)))
        });
        if all_done || Instant::now() > deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // SIGKILL any stragglers
    for (name, mut child) in children.drain(..) {
        match child.try_wait() {
            Ok(Some(_)) => {} // already exited
            _ => {
                eprintln!("[init]   SIGKILL → {name}");
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    eprintln!("[init] All processes stopped.");
}

fn cleanup_stale_shm(name: &str) {
    let c_name = match std::ffi::CString::new(name.as_bytes()) {
        Ok(c) => c,
        Err(_) => return,
    };
    // Try to open — if it exists, unlink it
    let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDONLY, 0) };
    if fd >= 0 {
        unsafe { libc::close(fd); }
        unsafe { libc::shm_unlink(c_name.as_ptr()); }
        eprintln!("[init] Cleaned up stale shm '{name}'");
    }
}

extern "C" fn signal_handler(_sig: libc::c_int) {
    RUNNING.store(false, Ordering::Relaxed);
}
