use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    prelude::*,
    widgets::*,
};
use qualia_shm::{ShmRegion, LayerReader, MAX_LEDGER_ENTRIES};
use qualia_ipc::ControlListener;
use qualia_types::{BeliefSlot, LedgerEvent, NUM_LAYERS, STATE_DIM, WEIGHT_COUNT, MAX_OBJECTS, MAX_THOUGHTS};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static RUNNING: AtomicBool = AtomicBool::new(true);

/// Runners spawned by the supervisor. Does NOT include qualia-watch (that's us).
static RUNNER_NAMES: &[&str] = &[
    "qualia-camera",
    "qualia-l0-superposition",
    "qualia-l1-belief",
    "qualia-l2-belief",
    "qualia-l3-belief",
    "qualia-l4-behavior",
    "qualia-l5-behavior",
    "qualia-l6-semantic",
    "qualia-health",
    "qualia-vision",
    "qualia-agent",
];

const LAYER_NAMES: [&str; NUM_LAYERS] = [
    "superposition",
    "belief_motor",
    "belief_local",
    "belief_visual",
    "behavior_short",
    "behavior_deep",
    "sensor",
];

const LAYER_FREQ: [&str; NUM_LAYERS] = [
    "1000", "100", "100", "100", "1", "0.1", "30",
];

const MAX_DISPLAY_EVENTS: usize = 100;
const VFE_HISTORY_LEN: usize = 120; // ~6 seconds at 20fps

// ── View modes ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Overview,   // 1 — layer table + events (default)
    Detail,     // 2 — single layer deep dive: all 64 dims
    Hex,        // 3 — hex dump of raw belief memory
    Sparklines, // 4 — VFE history sparklines for all layers
    Residuals,  // 5 — residual heatmap across dimensions
    Weights,    // 6 — weight matrix heatmap
    World,      // 7 — world model: scene, objects, directive
}

const VIEW_LABELS: &[(&str, ViewMode)] = &[
    ("1:Overview", ViewMode::Overview),
    ("2:Detail", ViewMode::Detail),
    ("3:Hex", ViewMode::Hex),
    ("4:Spark", ViewMode::Sparklines),
    ("5:Residual", ViewMode::Residuals),
    ("6:Weights", ViewMode::Weights),
    ("7:World", ViewMode::World),
];

// ── App state ───────────────────────────────────────────────────────────

const MAX_DISPLAY_THOUGHTS: usize = 50;

struct ThoughtDisplay {
    text: String,
    layer: u8,
    kind: u8,
    _vfe: f32,
    _timestamp_ns: u64,
}

struct App {
    shm: ShmRegion,
    start_time: Instant,
    last_ledger_seq: u64,
    events: Vec<EventEntry>,
    shm_name: String,
    _sock_path: String,
    view: ViewMode,
    selected_layer: usize,
    detail_scroll: u16,
    hex_scroll: u16,
    vfe_history: [[f64; VFE_HISTORY_LEN]; NUM_LAYERS],
    vfe_hist_idx: usize,
    belief_snapshots: [BeliefSlot; NUM_LAYERS],
    last_thought_seq: u64,
    thoughts: Vec<ThoughtDisplay>,
    children: Vec<(String, Child)>,
    has_gemini_key: bool,
}

struct EventEntry {
    timestamp_ns: u64,
    layer: u8,
    event_type: LedgerEvent,
    detail: String,
}

// ── Entry point ─────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shm_name = std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".into());
    let sock_path = std::env::var("QUALIA_SOCK_PATH")
        .unwrap_or_else(|_| "/tmp/qualia_body.sock".to_string());
    let headless = std::env::var("QUALIA_HEADLESS").map(|v| v == "1").unwrap_or(false);

    // ── Supervisor: create SHM and spawn runners ──
    cleanup_stale_shm(&shm_name);
    let shm = ShmRegion::create(&shm_name).unwrap_or_else(|e| {
        panic!("Failed to create shm '{shm_name}': {e}");
    });

    let _control = ControlListener::bind(&sock_path).ok();

    // Catch SIGINT/SIGTERM
    unsafe {
        libc::signal(libc::SIGINT, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGTERM, signal_handler as libc::sighandler_t);
    }

    let children = spawn_runners(&shm_name);

    if headless {
        return run_headless(shm, children, sock_path);
    }

    // ── TUI setup ──
    enable_raw_mode()?;
    std::io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let zero_belief = unsafe { std::mem::zeroed::<BeliefSlot>() };
    let mut app = App {
        shm,
        start_time: Instant::now(),
        last_ledger_seq: 0,
        events: Vec::with_capacity(MAX_DISPLAY_EVENTS),
        shm_name: shm_name.clone(),
        _sock_path: sock_path.clone(),
        view: ViewMode::Overview,
        selected_layer: 0,
        detail_scroll: 0,
        hex_scroll: 0,
        vfe_history: [[0.0; VFE_HISTORY_LEN]; NUM_LAYERS],
        vfe_hist_idx: 0,
        belief_snapshots: [zero_belief; NUM_LAYERS],
        last_thought_seq: 0,
        thoughts: Vec::with_capacity(MAX_DISPLAY_THOUGHTS),
        children,
        has_gemini_key: std::env::var("GEMINI_API_KEY").map(|k| !k.is_empty()).unwrap_or(false),
    };

    // ── Main TUI loop ──
    while RUNNING.load(Ordering::Relaxed) {
        // Snapshot all layers
        for i in 0..NUM_LAYERS {
            let slot = app.shm.layer_slot(i);
            let reader = LayerReader::new(slot);
            app.belief_snapshots[i] = *reader.read();
        }

        // Record VFE history
        for i in 0..NUM_LAYERS {
            app.vfe_history[i][app.vfe_hist_idx] = app.belief_snapshots[i].vfe as f64;
        }
        app.vfe_hist_idx = (app.vfe_hist_idx + 1) % VFE_HISTORY_LEN;

        poll_ledger(&mut app);
        poll_thoughts(&mut app);

        terminal.draw(|frame| ui(frame, &app))?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('c')
                            if key.modifiers.contains(
                                crossterm::event::KeyModifiers::CONTROL,
                            ) => break,
                        KeyCode::Char('r') => {
                            shutdown_children(&mut app.children);
                            app.children = spawn_runners(&app.shm_name);
                        }
                        KeyCode::Char('1') => app.view = ViewMode::Overview,
                        KeyCode::Char('2') => app.view = ViewMode::Detail,
                        KeyCode::Char('3') => app.view = ViewMode::Hex,
                        KeyCode::Char('4') => app.view = ViewMode::Sparklines,
                        KeyCode::Char('5') => app.view = ViewMode::Residuals,
                        KeyCode::Char('6') => app.view = ViewMode::Weights,
                        KeyCode::Char('7') => app.view = ViewMode::World,
                        KeyCode::Up | KeyCode::Char('k') => {
                            if app.selected_layer > 0 {
                                app.selected_layer -= 1;
                                app.detail_scroll = 0;
                                app.hex_scroll = 0;
                            }
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if app.selected_layer < NUM_LAYERS - 1 {
                                app.selected_layer += 1;
                                app.detail_scroll = 0;
                                app.hex_scroll = 0;
                            }
                        }
                        KeyCode::Tab => {
                            app.view = match app.view {
                                ViewMode::Overview => ViewMode::Detail,
                                ViewMode::Detail => ViewMode::Hex,
                                ViewMode::Hex => ViewMode::Sparklines,
                                ViewMode::Sparklines => ViewMode::Residuals,
                                ViewMode::Residuals => ViewMode::Weights,
                                ViewMode::Weights => ViewMode::World,
                                ViewMode::World => ViewMode::Overview,
                            };
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // ── Shutdown: clean up terminal, then kill children ──
    cleanup_terminal()?;
    shutdown_children(&mut app.children);
    let _ = std::fs::remove_file(&sock_path);
    // shm drops automatically (owner=true unlinks)
    Ok(())
}

fn run_headless(
    shm: ShmRegion,
    mut children: Vec<(String, Child)>,
    sock_path: String,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("qualia-watch: headless mode (QUALIA_HEADLESS=1)");
    eprintln!("qualia-watch: spawned {} runners", children.len());

    let mut tick: u64 = 0;
    while RUNNING.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_secs(5));
        tick += 1;

        // Print layer VFE summary every 5s
        let mut vfes = Vec::new();
        for i in 0..NUM_LAYERS {
            let slot = shm.layer_slot(i);
            let reader = LayerReader::new(slot);
            let belief = reader.read();
            vfes.push(format!("L{}={:.4}", i, belief.vfe));
        }
        eprintln!("qualia-watch [{}]: {}", tick * 5, vfes.join(" "));

        // Check for dead children and respawn camera if needed
        let mut respawn_camera = false;
        children.retain_mut(|(name, child)| {
            if let Ok(Some(status)) = child.try_wait() {
                eprintln!("qualia-watch: {} exited with {}", name, status);
                if name == "qualia-camera" {
                    respawn_camera = true;
                }
                return false; // remove dead child
            }
            true
        });

        if respawn_camera {
            eprintln!("qualia-watch: respawning qualia-camera...");
            let self_path = std::env::current_exe().expect("Cannot get self path");
            let bin_dir = self_path.parent().expect("Cannot get bin dir");
            let bin_path = bin_dir.join("qualia-camera");
            let shm_name = std::env::var("QUALIA_SHM_NAME")
                .unwrap_or_else(|_| "/qualia_body".to_string());
            let log_path = "/tmp/qualia-camera.log";
            let log_stderr = std::fs::File::create(log_path)
                .map(Stdio::from)
                .unwrap_or_else(|_| Stdio::null());
            let mut cmd = Command::new(&bin_path);
            cmd.env("QUALIA_SHM_NAME", &shm_name)
                .stdout(Stdio::null())
                .stderr(log_stderr);
            for key in &["CAMERA_DEVICE", "RUST_LOG"] {
                if let Ok(val) = std::env::var(key) {
                    cmd.env(key, val);
                }
            }
            match cmd.spawn() {
                Ok(child) => {
                    eprintln!("qualia-watch: qualia-camera respawned (pid {})", child.id());
                    children.push(("qualia-camera".to_string(), child));
                }
                Err(e) => {
                    eprintln!("qualia-watch: failed to respawn qualia-camera: {}", e);
                }
            }
        }
    }

    shutdown_children(&mut children);
    let _ = std::fs::remove_file(&sock_path);
    Ok(())
}

fn cleanup_terminal() -> Result<(), Box<dyn std::error::Error>> {
    disable_raw_mode()?;
    std::io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

fn shutdown_children(children: &mut Vec<(String, Child)>) {
    eprintln!("\nShutting down {} processes...", children.len());
    for (_, child) in children.iter() {
        unsafe { libc::kill(child.id() as i32, libc::SIGTERM); }
    }
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
    for (_, mut child) in children.drain(..) {
        match child.try_wait() {
            Ok(Some(_)) => {}
            _ => { let _ = child.kill(); let _ = child.wait(); }
        }
    }
    eprintln!("All processes stopped.");
}

fn cleanup_stale_shm(name: &str) {
    let c_name = match std::ffi::CString::new(name.as_bytes()) {
        Ok(c) => c,
        Err(_) => return,
    };
    let fd = unsafe { libc::shm_open(c_name.as_ptr(), libc::O_RDONLY, 0) };
    if fd >= 0 {
        unsafe { libc::close(fd); }
        unsafe { libc::shm_unlink(c_name.as_ptr()); }
    }
}

extern "C" fn signal_handler(_sig: libc::c_int) {
    RUNNING.store(false, Ordering::Relaxed);
}

fn spawn_runners(shm_name: &str) -> Vec<(String, Child)> {
    let self_path = std::env::current_exe().expect("Cannot get self path");
    let bin_dir = self_path.parent().expect("Cannot get bin dir");
    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());

    let mut children: Vec<(String, Child)> = Vec::new();

    for name in RUNNER_NAMES {
        let bin_path = bin_dir.join(name);
        let mut cmd = Command::new(&bin_path);
        let log_path = format!("/tmp/{}.log", name);
        let log_stderr = std::fs::File::create(&log_path)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        cmd.env("QUALIA_SHM_NAME", shm_name)
            .env("RUST_LOG", &rust_log)
            .stdout(Stdio::null())
            .stderr(log_stderr);

        // Pass through env vars that runners need
        for key in &[
            "GEMINI_API_KEY",
            "CAMERA_DEVICE",
            "QUALIA_SOCK_PATH",
            "QUALIA_WEB_PORT",
            "QUALIA_LLM_INTERVAL",
            "QUALIA_LLM_MAX_CALLS",
        ] {
            if let Ok(val) = std::env::var(key) {
                cmd.env(key, val);
            }
        }

        if let Ok(child) = cmd.spawn() {
            children.push((name.to_string(), child));
        }
    }

    children
}


// ── Main UI ─────────────────────────────────────────────────────────────

fn ui(frame: &mut Frame, app: &App) {
    let area = frame.area();

    let (gemini_mark, gemini_color) = if app.has_gemini_key {
        ("✓ GEMINI", Color::Green)
    } else {
        ("✗ GEMINI", Color::Red)
    };
    let title = Line::from(vec![
        Span::styled(
            format!(" QUALIA ENGINE v0.1.0 | shm:{} | {} runners | ", app.shm_name, app.children.len()),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ),
        Span::styled(gemini_mark, Style::default().fg(gemini_color).add_modifier(Modifier::BOLD)),
        Span::styled(" ", Style::default()),
    ]);
    let outer_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(title)
        .title_alignment(Alignment::Left);

    let inner = outer_block.inner(area);
    frame.render_widget(outer_block, area);

    // Top: view tabs + status | Bottom: content
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // tab bar
            Constraint::Min(1),   // content
            Constraint::Length(1), // status bar
        ])
        .split(inner);

    render_tab_bar(frame, app, chunks[0]);

    match app.view {
        ViewMode::Overview => render_overview(frame, app, chunks[1]),
        ViewMode::Detail => render_detail(frame, app, chunks[1]),
        ViewMode::Hex => render_hex(frame, app, chunks[1]),
        ViewMode::Sparklines => render_sparklines(frame, app, chunks[1]),
        ViewMode::Residuals => render_residuals(frame, app, chunks[1]),
        ViewMode::Weights => render_weights(frame, app, chunks[1]),
        ViewMode::World => render_world(frame, app, chunks[1]),
    }

    render_status_bar(frame, app, chunks[2]);
}

// ── Tab bar ─────────────────────────────────────────────────────────────

fn render_tab_bar(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![Span::raw("  ")];
    for (i, (label, mode)) in VIEW_LABELS.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" │ ", Style::default().fg(Color::DarkGray)));
        }
        let style = if *mode == app.view {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        spans.push(Span::styled(*label, style));
    }
    spans.push(Span::styled(
        format!("    [L{} ▲▼]  [Tab]  [r:restart]  [q/^C:quit]", app.selected_layer),
        Style::default().fg(Color::DarkGray),
    ));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ═══════════════════════════════════════════════════════════════════════
// VIEW 1: Overview (original view)
// ═══════════════════════════════════════════════════════════════════════

fn render_overview(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(12),   // layer table
            Constraint::Length(12), // events
        ])
        .split(area);

    render_layer_table(frame, app, chunks[0]);
    render_events(frame, app, chunks[1]);
}

fn render_layer_table(frame: &mut Frame, app: &App, area: Rect) {
    let header_cells = ["Layer", "Name", "Hz", "VFE", "Comp", "Streak", "μs", "C/E", "Confirms", "Challenges"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));
    let header = Row::new(header_cells).height(1);

    let mut rows = Vec::new();
    for i in 0..NUM_LAYERS {
        let belief = &app.belief_snapshots[i];
        let slot = app.shm.layer_slot(i);

        let challenge = slot.challenge_flag.load(Ordering::Relaxed);
        let escalate = slot.escalate_flag.load(Ordering::Relaxed);
        let confirms = slot.confirm_total.load(Ordering::Relaxed);
        let challenges = slot.challenge_total.load(Ordering::Relaxed);

        let ce = match (challenge, escalate) {
            (true, true) => "C/E",
            (true, false) => "C",
            (false, true) => "E",
            (false, false) => "·/·",
        };
        let ce_style = match (challenge, escalate) {
            (true, _) => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            (_, true) => Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            _ => Style::default().fg(Color::DarkGray),
        };

        let (vfe_text, vfe_sty) = if belief.timestamp_ns == 0 {
            ("---".to_string(), Style::default().fg(Color::DarkGray))
        } else {
            (format!("{:.4}", belief.vfe), vfe_style(belief.vfe))
        };

        let (comp_text, streak_text, cycle_text) = if belief.timestamp_ns == 0 {
            ("---".into(), "---".into(), "---".into())
        } else {
            (
                compression_bar(belief.compression),
                format!("{}", belief.confirm_streak),
                format!("{}", belief.cycle_us),
            )
        };

        let row_style = if i == app.selected_layer {
            Style::default().bg(Color::DarkGray)
        } else {
            Style::default()
        };

        let row = Row::new(vec![
            Cell::from(format!("  {}", i)).style(Style::default().fg(Color::White)),
            Cell::from(LAYER_NAMES[i]).style(Style::default().fg(Color::White)),
            Cell::from(LAYER_FREQ[i]).style(Style::default().fg(Color::DarkGray)),
            Cell::from(vfe_text).style(vfe_sty),
            Cell::from(comp_text).style(Style::default().fg(Color::Blue)),
            Cell::from(streak_text).style(Style::default().fg(Color::White)),
            Cell::from(cycle_text).style(Style::default().fg(Color::White)),
            Cell::from(ce).style(ce_style),
            Cell::from(format!("{}", confirms)).style(Style::default().fg(Color::Green)),
            Cell::from(format!("{}", challenges)).style(Style::default().fg(Color::Red)),
        ]).style(row_style);
        rows.push(row);
    }

    let widths = [
        Constraint::Length(7),   // Layer
        Constraint::Length(16),  // Name
        Constraint::Length(6),   // Hz
        Constraint::Length(9),   // VFE
        Constraint::Length(6),   // Comp
        Constraint::Length(8),   // Streak
        Constraint::Length(6),   // μs
        Constraint::Length(5),   // C/E
        Constraint::Length(10),  // Confirms
        Constraint::Length(10),  // Challenges
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray)))
        .column_spacing(1);

    frame.render_widget(table, area);
}

fn render_events(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" Events ", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let visible = inner.height as usize;
    let start = app.events.len().saturating_sub(visible);

    let items: Vec<ListItem> = app.events[start..]
        .iter()
        .rev()
        .map(|e| {
            let ts = format_timestamp_ns(e.timestamp_ns);
            let ename = event_name(e.event_type);
            let style = event_style(e.event_type);
            ListItem::new(Line::from(vec![
                Span::styled(format!("  {}  ", ts), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("L{}  ", e.layer), Style::default().fg(Color::White)),
                Span::styled(format!("{:<12}", ename), style),
                Span::styled(e.detail.clone(), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();

    frame.render_widget(List::new(items), inner);
}

// ═══════════════════════════════════════════════════════════════════════
// VIEW 2: Detail — single layer deep dive
// ═══════════════════════════════════════════════════════════════════════

fn render_detail(frame: &mut Frame, app: &App, area: Rect) {
    let belief = &app.belief_snapshots[app.selected_layer];
    let layer = app.selected_layer;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            format!(" Layer {} — {} — Detail ", layer, LAYER_NAMES[layer]),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // scalars
            Constraint::Min(1),    // dimension table
        ])
        .split(inner);

    // Weight stats for this layer
    let slot = app.shm.layer_slot(layer);
    let diag_mean: f64 = (0..STATE_DIM)
        .map(|i| slot.weights[i * STATE_DIM + i] as f64)
        .sum::<f64>() / STATE_DIM as f64;
    let off_diag: f64 = (0..STATE_DIM)
        .flat_map(|i| (0..STATE_DIM).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .map(|(i, j)| (slot.weights[i * STATE_DIM + j] as f64).powi(2))
        .sum::<f64>()
        .sqrt();

    let scalars = vec![
        Line::from(vec![
            Span::styled("  VFE: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.6}", belief.vfe), vfe_style(belief.vfe)),
            Span::styled("    Challenge VFE: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.6}", belief.challenge_vfe), vfe_style(belief.challenge_vfe)),
            Span::styled("    Compression: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}/255", belief.compression), Style::default().fg(Color::Blue)),
            Span::styled(format!(" {}", compression_bar(belief.compression)), Style::default().fg(Color::Blue)),
        ]),
        Line::from(vec![
            Span::styled("  Confirm Streak: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", belief.confirm_streak), Style::default().fg(Color::Green)),
            Span::styled("    Cycle: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}μs", belief.cycle_us), Style::default().fg(Color::White)),
            Span::styled("    W diag: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.3}", diag_mean), Style::default().fg(Color::Green)),
            Span::styled("    W off-diag: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.3}", off_diag), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Dim   Mean        Precision   Prediction  Residual",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
    ];
    frame.render_widget(Paragraph::new(scalars), chunks[0]);

    // Per-dimension table
    let mut lines = Vec::new();
    for d in 0..STATE_DIM {
        let res_val = belief.residual[d];
        let res_style = if res_val.abs() < 0.001 {
            Style::default().fg(Color::Green)
        } else if res_val.abs() < 0.01 {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::Red)
        };

        lines.push(Line::from(vec![
            Span::styled(format!("  {:3}   ", d), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:>10.6}  ", belief.mean[d]), Style::default().fg(Color::White)),
            Span::styled(format!("{:>10.4}  ", belief.precision[d]), Style::default().fg(Color::Cyan)),
            Span::styled(format!("{:>10.6}  ", belief.prediction[d]), Style::default().fg(Color::Blue)),
            Span::styled(format!("{:>10.6}", res_val), res_style),
        ]));
    }
    let paragraph = Paragraph::new(lines).scroll((app.detail_scroll, 0));
    frame.render_widget(paragraph, chunks[1]);
}

// ═══════════════════════════════════════════════════════════════════════
// VIEW 3: Hex — raw memory dump
// ═══════════════════════════════════════════════════════════════════════

fn render_hex(frame: &mut Frame, app: &App, area: Rect) {
    let belief = &app.belief_snapshots[app.selected_layer];
    let layer = app.selected_layer;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            format!(" Layer {} — {} — Hex Dump ", layer, LAYER_NAMES[layer]),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Cast BeliefSlot to raw bytes
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            belief as *const BeliefSlot as *const u8,
            std::mem::size_of::<BeliefSlot>(),
        )
    };

    let bytes_per_line = 16;
    let mut lines = Vec::new();

    // Header showing field regions
    lines.push(Line::from(Span::styled(
        "  Offset   00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F   ASCII           Field",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(""));

    for (i, chunk) in bytes.chunks(bytes_per_line).enumerate() {
        let offset = i * bytes_per_line;

        // Determine which field this offset belongs to
        let field_name = offset_to_field(offset);
        let field_color = field_to_color(offset);

        // Hex bytes with gap at byte 8
        let mut hex_spans = vec![
            Span::styled(format!("  {:04X}   ", offset), Style::default().fg(Color::DarkGray)),
        ];

        for (j, &b) in chunk.iter().enumerate() {
            if j == 8 {
                hex_spans.push(Span::raw(" "));
            }
            let byte_style = if b == 0 {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default().fg(field_color)
            };
            hex_spans.push(Span::styled(format!("{:02X} ", b), byte_style));
        }
        // Pad if last line is short
        for j in chunk.len()..bytes_per_line {
            if j == 8 {
                hex_spans.push(Span::raw(" "));
            }
            hex_spans.push(Span::raw("   "));
        }

        // ASCII representation
        hex_spans.push(Span::raw("  "));
        let ascii: String = chunk
            .iter()
            .map(|&b| if (0x20..=0x7E).contains(&b) { b as char } else { '·' })
            .collect();
        hex_spans.push(Span::styled(
            format!("{:<16}", ascii),
            Style::default().fg(Color::DarkGray),
        ));

        // Field label
        hex_spans.push(Span::styled(
            format!("  {}", field_name),
            Style::default().fg(field_color).add_modifier(Modifier::DIM),
        ));

        lines.push(Line::from(hex_spans));
    }

    let paragraph = Paragraph::new(lines).scroll((app.hex_scroll, 0));
    frame.render_widget(paragraph, inner);
}

fn offset_to_field(offset: usize) -> &'static str {
    match offset {
        0..256 => "mean[64]",
        256..512 => "precision[64]",
        512..516 => "vfe",
        516..772 => "prediction[64]",
        772..1028 => "residual[64]",
        1028..1032 => "challenge_vfe",
        1032..1036 => "confirm_streak",
        1036..1037 => "compression",
        1037..1038 => "layer",
        1038..1040 => "_pad",
        1040..1048 => "timestamp_ns",
        1048..1052 => "cycle_us",
        _ => "_pad2",
    }
}

fn field_to_color(offset: usize) -> Color {
    match offset {
        0..256 => Color::Green,
        256..512 => Color::Cyan,
        512..516 => Color::Yellow,
        516..772 => Color::Blue,
        772..1028 => Color::Red,
        1028..1032 => Color::Yellow,
        1032..1036 => Color::Green,
        1036..1037 => Color::Magenta,
        1037..1038 => Color::White,
        1040..1048 => Color::DarkGray,
        1048..1052 => Color::DarkGray,
        _ => Color::DarkGray,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// VIEW 4: Sparklines — VFE history over time
// ═══════════════════════════════════════════════════════════════════════

fn render_sparklines(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " VFE History — All Layers ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Each layer gets equal vertical space
    let constraints: Vec<Constraint> = (0..NUM_LAYERS)
        .map(|_| Constraint::Ratio(1, NUM_LAYERS as u32))
        .collect();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(inner);

    for layer in 0..NUM_LAYERS {
        // Build ordered history from ring buffer
        let mut data = Vec::with_capacity(VFE_HISTORY_LEN);
        for i in 0..VFE_HISTORY_LEN {
            let idx = (app.vfe_hist_idx + i) % VFE_HISTORY_LEN;
            data.push(app.vfe_history[layer][idx]);
        }

        // Scale to u64 for sparkline (multiply by 10000 for precision)
        let max_val = data.iter().cloned().fold(0.001_f64, f64::max);
        let scaled: Vec<u64> = data
            .iter()
            .map(|&v| ((v / max_val) * 100.0) as u64)
            .collect();

        let color = match layer {
            0 => Color::Red,
            1 => Color::Yellow,
            2 => Color::Green,
            3 => Color::Cyan,
            4 => Color::Blue,
            5 => Color::Magenta,
            6 => Color::LightRed,
            _ => Color::DarkGray,
        };

        let belief = &app.belief_snapshots[layer];
        let label = format!(
            " L{} {} — VFE: {:.4} — max: {:.4} ",
            layer, LAYER_NAMES[layer], belief.vfe, max_val,
        );

        let sparkline = Sparkline::default()
            .block(Block::default()
                .title(Span::styled(label, Style::default().fg(color)))
                .borders(Borders::NONE))
            .data(&scaled)
            .style(Style::default().fg(color));

        frame.render_widget(sparkline, rows[layer]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// VIEW 5: Residuals — heatmap across all layers × dimensions
// ═══════════════════════════════════════════════════════════════════════

fn render_residuals(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Residual Heatmap — Layers × Dimensions ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // legend
            Constraint::Min(1),   // heatmap
        ])
        .split(inner);

    // Legend
    let legend = Line::from(vec![
        Span::raw("  "),
        Span::styled("·", Style::default().fg(Color::DarkGray)),
        Span::styled(" <0.001  ", Style::default().fg(Color::DarkGray)),
        Span::styled("░", Style::default().fg(Color::Green)),
        Span::styled(" <0.005  ", Style::default().fg(Color::DarkGray)),
        Span::styled("▒", Style::default().fg(Color::Yellow)),
        Span::styled(" <0.01   ", Style::default().fg(Color::DarkGray)),
        Span::styled("▓", Style::default().fg(Color::Red)),
        Span::styled(" <0.05   ", Style::default().fg(Color::DarkGray)),
        Span::styled("█", Style::default().fg(Color::LightRed)),
        Span::styled(" ≥0.05   ", Style::default().fg(Color::DarkGray)),
        Span::raw("   (absolute residual per dimension)"),
    ]);
    frame.render_widget(Paragraph::new(vec![Line::from(""), legend]), chunks[0]);

    // Heatmap: each layer is a row, each dimension is a character
    let mut lines = Vec::new();

    // Column header (dimension indices every 8)
    let mut dim_header_spans = vec![Span::styled("          ", Style::default().fg(Color::DarkGray))];
    for d in 0..STATE_DIM {
        if d % 8 == 0 {
            dim_header_spans.push(Span::styled(
                format!("{:<8}", d),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }
    lines.push(Line::from(dim_header_spans));

    for layer in 0..NUM_LAYERS {
        let belief = &app.belief_snapshots[layer];
        let mut spans = vec![Span::styled(
            format!("  L{} {:5} ", layer, &LAYER_NAMES[layer][..5.min(LAYER_NAMES[layer].len())]),
            Style::default().fg(Color::White),
        )];

        for d in 0..STATE_DIM {
            let abs_res = belief.residual[d].abs();
            let (ch, color) = if abs_res < 0.001 {
                ('·', Color::DarkGray)
            } else if abs_res < 0.005 {
                ('░', Color::Green)
            } else if abs_res < 0.01 {
                ('▒', Color::Yellow)
            } else if abs_res < 0.05 {
                ('▓', Color::Red)
            } else {
                ('█', Color::LightRed)
            };
            spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
        }

        // Append mean residual
        let mean_res: f32 = belief.residual.iter().map(|r| r.abs()).sum::<f32>() / STATE_DIM as f32;
        spans.push(Span::styled(
            format!("  avg={:.4}", mean_res),
            Style::default().fg(Color::DarkGray),
        ));

        lines.push(Line::from(spans));
    }

    frame.render_widget(Paragraph::new(lines), chunks[1]);
}

// ═══════════════════════════════════════════════════════════════════════
// VIEW 6: Weights — generative model matrix heatmap
// ═══════════════════════════════════════════════════════════════════════

fn render_weights(frame: &mut Frame, app: &App, area: Rect) {
    let layer = app.selected_layer;
    let slot = app.shm.layer_slot(layer);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            format!(" Layer {} — {} — Weight Matrix W[64×64] ", layer, LAYER_NAMES[layer]),
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // stats
            Constraint::Min(1),   // heatmap
        ])
        .split(inner);

    // Weight statistics
    let weights = &slot.weights;
    let bias = &slot.bias;

    let mut _w_sum: f64 = 0.0;
    let mut w_abs_sum: f64 = 0.0;
    let mut w_max: f32 = f32::MIN;
    let mut w_min: f32 = f32::MAX;
    let mut off_diag_norm: f64 = 0.0;

    for i in 0..STATE_DIM {
        for j in 0..STATE_DIM {
            let w = weights[i * STATE_DIM + j];
            _w_sum += w as f64;
            w_abs_sum += w.abs() as f64;
            if w > w_max { w_max = w; }
            if w < w_min { w_min = w; }
            if i != j {
                off_diag_norm += (w * w) as f64;
            }
        }
    }
    off_diag_norm = off_diag_norm.sqrt();
    let diag_mean: f64 = (0..STATE_DIM).map(|i| weights[i * STATE_DIM + i] as f64).sum::<f64>() / STATE_DIM as f64;
    let bias_norm: f64 = bias.iter().map(|b| (*b as f64) * (*b as f64)).sum::<f64>().sqrt();

    let stats = vec![
        Line::from(vec![
            Span::styled("  Diag mean: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.4}", diag_mean), Style::default().fg(Color::Green)),
            Span::styled("    Off-diag ‖W‖: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.4}", off_diag_norm), Style::default().fg(Color::Yellow)),
            Span::styled("    ‖bias‖: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.4}", bias_norm), Style::default().fg(Color::Cyan)),
            Span::styled("    range: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("[{:.3}, {:.3}]", w_min, w_max), Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  64×64 weight matrix — brighter = larger absolute value, color = sign (green=+, red=-)",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    frame.render_widget(Paragraph::new(stats), chunks[0]);

    // Weight heatmap: each row of W is one line, each column is one character
    let mut lines = Vec::new();

    // Column header
    let mut header_spans = vec![Span::styled("     ", Style::default().fg(Color::DarkGray))];
    for j in 0..STATE_DIM {
        if j % 8 == 0 {
            header_spans.push(Span::styled(
                format!("{:<8}", j),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }
    lines.push(Line::from(header_spans));

    let abs_max = w_abs_sum as f32 / WEIGHT_COUNT as f32 * 5.0; // scale relative to mean
    let abs_max = abs_max.max(0.01); // avoid division by zero

    for i in 0..STATE_DIM {
        let mut spans = vec![Span::styled(
            format!("{:3}  ", i),
            Style::default().fg(Color::DarkGray),
        )];

        for j in 0..STATE_DIM {
            let w = weights[i * STATE_DIM + j];
            let intensity = (w.abs() / abs_max).min(1.0);

            let (ch, color) = if i == j {
                // Diagonal — identity baseline
                if w > 0.5 {
                    ('█', Color::Green)
                } else if w > 0.0 {
                    ('▓', Color::Green)
                } else {
                    ('▓', Color::Red)
                }
            } else {
                // Off-diagonal — learned connections
                let ch = if intensity < 0.05 {
                    '·'
                } else if intensity < 0.2 {
                    '░'
                } else if intensity < 0.5 {
                    '▒'
                } else if intensity < 0.8 {
                    '▓'
                } else {
                    '█'
                };
                let color = if w >= 0.0 { Color::Green } else { Color::Red };
                (ch, color)
            };

            spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
        }

        // Row stats
        let row_norm: f32 = (0..STATE_DIM)
            .map(|j| {
                let w = weights[i * STATE_DIM + j];
                w * w
            })
            .sum::<f32>()
            .sqrt();
        spans.push(Span::styled(
            format!("  ‖{:.2}‖", row_norm),
            Style::default().fg(Color::DarkGray),
        ));

        lines.push(Line::from(spans));
    }

    let paragraph = Paragraph::new(lines).scroll((app.hex_scroll, 0));
    frame.render_widget(paragraph, chunks[1]);
}

// ═══════════════════════════════════════════════════════════════════════
// VIEW 7: World — world model + thought stream
// ═══════════════════════════════════════════════════════════════════════

fn render_world(frame: &mut Frame, app: &App, area: Rect) {
    let world = app.shm.world_model();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " World Model + Thought Stream ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Split: left = world model, right = thought stream
    let halves = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(45), // world model
            Constraint::Percentage(55), // thoughts
        ])
        .split(inner);

    // ══════════════════════════════════════════
    // LEFT: World model
    // ══════════════════════════════════════════
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),  // directive + scene + activity
            Constraint::Length(8),  // objects
            Constraint::Min(1),    // embedding heatmap
        ])
        .split(halves[0]);

    let scene = read_cstr_display(&world.scene);
    let activity = read_cstr_display(&world.activity);
    let directive = read_cstr_display(&world.directive);
    let llm_calls = world.llm_call_count;
    let vision_frames = world.vision_frame_count;

    let info_lines = vec![
        Line::from(vec![
            Span::styled(" DIRECTIVE ", Style::default().fg(Color::Black).bg(Color::Magenta).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" {}", directive), Style::default().fg(Color::Magenta)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Scene: ", Style::default().fg(Color::Cyan)),
            Span::styled(&scene, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Activity: ", Style::default().fg(Color::Yellow)),
            Span::styled(&activity, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(format!("  {} obj", world.num_objects), Style::default().fg(Color::White)),
            Span::styled(format!("  {} frames", vision_frames), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("  {} LLM", llm_calls), Style::default().fg(Color::Green)),
        ]),
    ];
    frame.render_widget(Paragraph::new(info_lines), left_chunks[0]);

    // Objects
    let obj_block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" Objects ", Style::default().fg(Color::White)));
    let obj_inner = obj_block.inner(left_chunks[1]);
    frame.render_widget(obj_block, left_chunks[1]);

    let mut obj_lines = Vec::new();
    if world.num_objects == 0 {
        obj_lines.push(Line::from(Span::styled("  (none)", Style::default().fg(Color::DarkGray))));
    } else {
        for i in 0..world.num_objects.min(MAX_OBJECTS as u32) as usize {
            let obj = &world.objects[i];
            if obj.active == 0 { continue; }
            let name = read_cstr_display(&obj.name);
            let conf_color = if obj.confidence > 0.8 { Color::Green }
                else if obj.confidence > 0.5 { Color::Yellow }
                else { Color::Red };
            obj_lines.push(Line::from(vec![
                Span::styled(format!("  {:.0}% ", obj.confidence * 100.0), Style::default().fg(conf_color)),
                Span::styled(name, Style::default().fg(Color::White)),
                Span::styled(format!(" ({:.1},{:.1})", obj.x, obj.y), Style::default().fg(Color::DarkGray)),
            ]));
        }
    }
    frame.render_widget(Paragraph::new(obj_lines), obj_inner);

    // Embedding 8x8 heatmap
    let emb_block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" Embedding [64] → L3-L6 ", Style::default().fg(Color::White)));
    let emb_inner = emb_block.inner(left_chunks[2]);
    frame.render_widget(emb_block, left_chunks[2]);

    let mut grid_lines = Vec::new();
    for row in 0..8 {
        if row >= emb_inner.height as usize { break; }
        let mut spans = vec![Span::raw("  ")];
        for col in 0..8 {
            let val = world.scene_embedding[row * 8 + col];
            let (ch, color) = embedding_char(val);
            spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
            spans.push(Span::raw(" "));
        }
        // Show first 4 values numerically
        let idx = row * 8;
        spans.push(Span::styled(
            format!("  [{:2}] {:.2} {:.2} {:.2} {:.2}",
                idx,
                world.scene_embedding[idx],
                world.scene_embedding[idx+1],
                world.scene_embedding[idx+2],
                world.scene_embedding[idx+3],
            ),
            Style::default().fg(Color::DarkGray),
        ));
        grid_lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(grid_lines), emb_inner);

    // ══════════════════════════════════════════
    // RIGHT: Thought stream
    // ══════════════════════════════════════════
    let thought_block = Block::default()
        .borders(Borders::LEFT)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Thoughts ",
            Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
        ));
    let thought_inner = thought_block.inner(halves[1]);
    frame.render_widget(thought_block, halves[1]);

    let visible = thought_inner.height as usize;
    let start = app.thoughts.len().saturating_sub(visible);

    let thought_lines: Vec<ListItem> = app.thoughts[start..]
        .iter()
        .map(|t| {
            let layer_label = if t.layer == 255 {
                "VIS".to_string()
            } else {
                format!("L{}", t.layer)
            };

            let kind_label = match t.kind {
                0 => "see",
                1 => "predict",
                2 => "surprise!",
                3 => "learn",
                4 => "resolve",
                5 => "escalate",
                _ => "?",
            };

            let kind_color = match t.kind {
                0 => Color::Cyan,
                1 => Color::Blue,
                2 => Color::Red,
                3 => Color::Yellow,
                4 => Color::Green,
                5 => Color::Magenta,
                _ => Color::DarkGray,
            };

            ListItem::new(Line::from(vec![
                Span::styled(
                    format!(" {:>3} ", layer_label),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{:<10} ", kind_label),
                    Style::default().fg(kind_color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    t.text.clone(),
                    Style::default().fg(Color::White),
                ),
            ]))
        })
        .collect();

    if thought_lines.is_empty() {
        let empty = vec![ListItem::new(Line::from(Span::styled(
            "  Waiting for thoughts...",
            Style::default().fg(Color::DarkGray),
        )))];
        frame.render_widget(List::new(empty), thought_inner);
    } else {
        frame.render_widget(List::new(thought_lines), thought_inner);
    }
}

fn embedding_char(val: f32) -> (char, Color) {
    let abs = val.abs();
    if abs < 0.05 {
        ('·', Color::DarkGray)
    } else if abs < 0.2 {
        ('░', if val > 0.0 { Color::Green } else { Color::Red })
    } else if abs < 0.5 {
        ('▒', if val > 0.0 { Color::Green } else { Color::Red })
    } else if abs < 0.8 {
        ('▓', if val > 0.0 { Color::Green } else { Color::Red })
    } else {
        ('█', if val > 0.0 { Color::Green } else { Color::Red })
    }
}

fn read_cstr_display(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    if end == 0 { return "(empty)".to_string(); }
    String::from_utf8_lossy(&buf[..end]).to_string()
}

// ── Status bar ──────────────────────────────────────────────────────────

fn render_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let elapsed = app.start_time.elapsed();
    let secs = elapsed.as_secs();
    let uptime = format!("{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60);
    let ledger_seq = app.shm.ledger_seq();

    let bar = Line::from(vec![
        Span::styled(format!("  SHM: {} ", app.shm_name), Style::default().fg(Color::Cyan)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled("GPU: Apple Silicon ", Style::default().fg(Color::White)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("Uptime: {} ", uptime), Style::default().fg(Color::Green)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("Ledger: {} ", ledger_seq), Style::default().fg(Color::DarkGray)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("Layer: {} {}", app.selected_layer, LAYER_NAMES[app.selected_layer]),
            Style::default().fg(Color::Yellow),
        ),
    ]);

    frame.render_widget(Paragraph::new(bar), area);
}

// ── Ledger polling ──────────────────────────────────────────────────────

fn poll_ledger(app: &mut App) {
    let current_seq = app.shm.ledger_seq();
    if current_seq <= app.last_ledger_seq {
        return;
    }

    let start_seq = if current_seq - app.last_ledger_seq > MAX_LEDGER_ENTRIES as u64 {
        current_seq - MAX_LEDGER_ENTRIES as u64
    } else {
        app.last_ledger_seq
    };

    for seq in (start_seq + 1)..=current_seq {
        let idx = (seq as usize) % MAX_LEDGER_ENTRIES;
        let entry = app.shm.ledger_entry(idx);
        if entry.seq != seq {
            continue;
        }
        let detail = format_event_detail(entry.event, entry.vfe, entry.residual_norm, entry.compression);
        app.events.push(EventEntry {
            timestamp_ns: entry.timestamp_ns,
            layer: entry.layer,
            event_type: entry.event,
            detail,
        });
    }

    if app.events.len() > MAX_DISPLAY_EVENTS {
        let drain = app.events.len() - MAX_DISPLAY_EVENTS;
        app.events.drain(..drain);
    }

    app.last_ledger_seq = current_seq;
}

// ── Thought polling ─────────────────────────────────────────────────

fn poll_thoughts(app: &mut App) {
    let tb = app.shm.thought_buffer();
    let current_seq = tb.write_seq.load(Ordering::Acquire);
    if current_seq <= app.last_thought_seq {
        return;
    }

    let start_seq = if current_seq - app.last_thought_seq > MAX_THOUGHTS as u64 {
        current_seq - MAX_THOUGHTS as u64
    } else {
        app.last_thought_seq
    };

    for seq in start_seq..current_seq {
        let idx = (seq as usize) % MAX_THOUGHTS;
        let entry = &tb.entries[idx];
        if entry.seq == seq {
            app.thoughts.push(ThoughtDisplay {
                text: read_cstr_display(&entry.text),
                layer: entry.layer,
                kind: entry.kind,
                _vfe: entry.vfe,
                _timestamp_ns: entry.timestamp_ns,
            });
        }
    }

    // Trim
    if app.thoughts.len() > MAX_DISPLAY_THOUGHTS {
        let drain = app.thoughts.len() - MAX_DISPLAY_THOUGHTS;
        app.thoughts.drain(..drain);
    }

    app.last_thought_seq = current_seq;
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn compression_bar(value: u8) -> String {
    let blocks = (value as usize * 8) / 255;
    let full = blocks / 2;
    let half = blocks % 2;
    let mut bar = "█".repeat(full);
    if half > 0 { bar.push('▌'); }
    while bar.chars().count() < 4 { bar.push(' '); }
    bar
}

fn vfe_style(vfe: f32) -> Style {
    if vfe < 0.01 {
        Style::default().fg(Color::Green)
    } else if vfe < 0.05 {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Red)
    }
}

fn event_name(event: LedgerEvent) -> &'static str {
    match event {
        LedgerEvent::Challenge => "CHALLENGE",
        LedgerEvent::Confirm => "CONFIRM",
        LedgerEvent::Habit => "HABIT",
        LedgerEvent::HabitDecay => "HABIT_DECAY",
        LedgerEvent::Escalate => "ESCALATE",
    }
}

fn event_style(event: LedgerEvent) -> Style {
    match event {
        LedgerEvent::Challenge => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        LedgerEvent::Confirm => Style::default().fg(Color::Green),
        LedgerEvent::Habit => Style::default().fg(Color::Cyan),
        LedgerEvent::HabitDecay => Style::default().fg(Color::Yellow),
        LedgerEvent::Escalate => Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
    }
}

fn format_event_detail(event: LedgerEvent, vfe: f32, residual_norm: f32, compression: u8) -> String {
    match event {
        LedgerEvent::Challenge => format!("vfe={:.3}  residual={:.3}", vfe, residual_norm),
        LedgerEvent::Confirm => format!("vfe={:.3}", vfe),
        LedgerEvent::Habit => format!("compression={}", compression),
        LedgerEvent::HabitDecay => format!("compression={}", compression),
        LedgerEvent::Escalate => format!("vfe={:.3}  residual={:.3}", vfe, residual_norm),
    }
}

fn format_timestamp_ns(ns: u64) -> String {
    let total_ms = ns / 1_000_000;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = (total_mins / 60) % 24;
    format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
}
