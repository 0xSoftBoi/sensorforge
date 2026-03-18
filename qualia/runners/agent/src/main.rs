use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::Html,
    routing::get,
    Router,
};
use qualia_shm::{LayerReader, ShmRegion};
use qualia_types::*;
use serde::Serialize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tower_http::cors::CorsLayer;

const DASHBOARD_HTML: &str = include_str!("../static/index.html");

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("QUALIA_WEB_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let app = Router::new()
        .route("/", get(serve_dashboard))
        .route("/ws", get(ws_handler))
        .layer(CorsLayer::permissive());

    let addr = format!("0.0.0.0:{port}");
    eprintln!("qualia-agent: web dashboard at http://localhost:{port}");
    eprintln!("qualia-agent: WebSocket at ws://localhost:{port}/ws");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn serve_dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

async fn ws_handler(ws: WebSocketUpgrade) -> axum::response::Response {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    let shm_name =
        std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".to_string());

    let shm = match ShmRegion::open(&shm_name) {
        Ok(s) => s,
        Err(e) => {
            let _ = socket
                .send(Message::Text(format!(
                    r#"{{"error":"Cannot open shm: {e}"}}"#
                ).into()))
                .await;
            return;
        }
    };

    eprintln!("qualia-agent: WebSocket client connected");

    let mut last_thought_seq: u64 = shm.thought_buffer().write_seq.load(Ordering::Acquire);
    let mut last_lore_seq: u64 = shm.lore_buffer().write_seq.load(Ordering::Acquire);

    loop {
        let snapshot = build_snapshot(&shm, &mut last_thought_seq, &mut last_lore_seq);
        let json = serde_json::to_string(&snapshot).unwrap_or_default();

        if socket.send(Message::Text(json.into())).await.is_err() {
            eprintln!("qualia-agent: WebSocket client disconnected");
            break;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn build_snapshot(shm: &ShmRegion, last_thought_seq: &mut u64, last_lore_seq: &mut u64) -> EngineSnapshot {
    let mut layers = Vec::with_capacity(NUM_LAYERS);

    for i in 0..NUM_LAYERS {
        let slot = shm.layer_slot(i);
        let reader = LayerReader::new(slot);
        let belief = *reader.read();

        let weights = &slot.weights;
        let bias = &slot.bias;

        // Weight stats
        let diag_mean: f64 = (0..STATE_DIM)
            .map(|d| weights[d * STATE_DIM + d] as f64)
            .sum::<f64>()
            / STATE_DIM as f64;
        let off_diag_norm: f64 = (0..STATE_DIM)
            .flat_map(|r| (0..STATE_DIM).map(move |c| (r, c)))
            .filter(|(r, c)| r != c)
            .map(|(r, c)| (weights[r * STATE_DIM + c] as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let bias_norm: f64 = bias
            .iter()
            .map(|b| (*b as f64) * (*b as f64))
            .sum::<f64>()
            .sqrt();

        // Weight matrix — downsample to 16x16 for bandwidth
        let mut weight_grid = [[0.0f32; 16]; 16];
        for r in 0..16 {
            for c in 0..16 {
                let mut sum = 0.0f32;
                for dr in 0..4 {
                    for dc in 0..4 {
                        sum += weights[(r * 4 + dr) * STATE_DIM + (c * 4 + dc)];
                    }
                }
                weight_grid[r][c] = sum / 16.0;
            }
        }

        layers.push(LayerSnapshot {
            id: i as u8,
            name: LAYER_NAMES[i].to_string(),
            freq: LAYER_FREQ[i].to_string(),
            vfe: belief.vfe,
            challenge_vfe: belief.challenge_vfe,
            compression: belief.compression,
            confirm_streak: belief.confirm_streak,
            cycle_us: belief.cycle_us,
            timestamp_ns: belief.timestamp_ns,
            challenge: slot.challenge_flag.load(Ordering::Relaxed),
            escalate: slot.escalate_flag.load(Ordering::Relaxed),
            confirms: slot.confirm_total.load(Ordering::Relaxed),
            challenges: slot.challenge_total.load(Ordering::Relaxed),
            mean: belief.mean.to_vec(),
            precision: belief.precision.to_vec(),
            prediction: belief.prediction.to_vec(),
            residual: belief.residual.to_vec(),
            diag_mean,
            off_diag_norm,
            bias_norm,
            weight_grid,
        });
    }

    // World model
    let world = shm.world_model();
    let scene = read_cstr(&world.scene);
    let activity = read_cstr(&world.activity);
    let directive = read_cstr(&world.directive);

    let mut objects = Vec::new();
    for i in 0..world.num_objects.min(MAX_OBJECTS as u32) as usize {
        let obj = &world.objects[i];
        if obj.active == 0 {
            continue;
        }
        objects.push(ObjectSnapshot {
            name: read_cstr(&obj.name),
            confidence: obj.confidence,
            x: obj.x,
            y: obj.y,
        });
    }

    let world_snapshot = WorldSnapshot {
        scene,
        activity,
        directive,
        num_objects: world.num_objects,
        objects,
        scene_embedding: world.scene_embedding.to_vec(),
        vision_frame_count: world.vision_frame_count,
        llm_call_count: world.llm_call_count,
        update_seq: world.update_seq.load(Ordering::Relaxed),
        gemini_input_tokens: world.gemini_input_tokens,
        gemini_output_tokens: world.gemini_output_tokens,
        gemini_embedding_tokens: world.gemini_embedding_tokens,
    };

    // Read new thoughts since last snapshot
    let tb = shm.thought_buffer();
    let current_seq = tb.write_seq.load(Ordering::Acquire);
    let mut thoughts = Vec::new();

    if current_seq > *last_thought_seq {
        let start = if current_seq - *last_thought_seq > MAX_THOUGHTS as u64 {
            current_seq - MAX_THOUGHTS as u64
        } else {
            *last_thought_seq
        };

        for seq in start..current_seq {
            let idx = (seq as usize) % MAX_THOUGHTS;
            let entry = &tb.entries[idx];
            if entry.seq == seq {
                thoughts.push(ThoughtSnapshot {
                    text: read_cstr(&entry.text),
                    layer: entry.layer,
                    kind: entry.kind,
                    vfe: entry.vfe,
                    timestamp_ns: entry.timestamp_ns,
                    seq: entry.seq,
                });
            }
        }
        *last_thought_seq = current_seq;
    }

    // Read new lore since last snapshot
    let lb = shm.lore_buffer();
    let current_lore_seq = lb.write_seq.load(Ordering::Acquire);
    let mut lore = Vec::new();

    if current_lore_seq > *last_lore_seq {
        let start = if current_lore_seq - *last_lore_seq > MAX_LORE_ENTRIES as u64 {
            current_lore_seq - MAX_LORE_ENTRIES as u64
        } else {
            *last_lore_seq
        };

        for seq in start..current_lore_seq {
            let idx = (seq as usize) % MAX_LORE_ENTRIES;
            let entry = &lb.entries[idx];
            if entry.seq == seq {
                lore.push(LoreSnapshot {
                    question: read_cstr(&entry.question),
                    answer: read_cstr(&entry.answer),
                    layer: entry.layer,
                    reason: entry.reason,
                    embedding_delta: entry.embedding_delta,
                    effectiveness: entry.effectiveness,
                    timestamp_ns: entry.timestamp_ns,
                    seq: entry.seq,
                });
            }
        }
        *last_lore_seq = current_lore_seq;
    }

    EngineSnapshot {
        layers,
        world: world_snapshot,
        thoughts,
        lore,
        lore_total: current_lore_seq,
        ledger_seq: shm.ledger_seq(),
    }
}

fn read_cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).to_string()
}

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

// ── Serialization types ──────────────────────────────────────────────────

#[derive(Serialize)]
struct EngineSnapshot {
    layers: Vec<LayerSnapshot>,
    world: WorldSnapshot,
    thoughts: Vec<ThoughtSnapshot>,
    lore: Vec<LoreSnapshot>,
    lore_total: u64,
    ledger_seq: u64,
}

#[derive(Serialize)]
struct LayerSnapshot {
    id: u8,
    name: String,
    freq: String,
    vfe: f32,
    challenge_vfe: f32,
    compression: u8,
    confirm_streak: u32,
    cycle_us: u32,
    timestamp_ns: u64,
    challenge: bool,
    escalate: bool,
    confirms: u64,
    challenges: u64,
    mean: Vec<f32>,
    precision: Vec<f32>,
    prediction: Vec<f32>,
    residual: Vec<f32>,
    diag_mean: f64,
    off_diag_norm: f64,
    bias_norm: f64,
    weight_grid: [[f32; 16]; 16],
}

#[derive(Serialize)]
struct WorldSnapshot {
    scene: String,
    activity: String,
    directive: String,
    num_objects: u32,
    objects: Vec<ObjectSnapshot>,
    scene_embedding: Vec<f32>,
    vision_frame_count: u64,
    llm_call_count: u64,
    update_seq: u64,
    gemini_input_tokens: u64,
    gemini_output_tokens: u64,
    gemini_embedding_tokens: u64,
}

#[derive(Serialize)]
struct ObjectSnapshot {
    name: String,
    confidence: f32,
    x: f32,
    y: f32,
}

#[derive(Serialize)]
struct ThoughtSnapshot {
    text: String,
    layer: u8,
    kind: u8,
    vfe: f32,
    timestamp_ns: u64,
    seq: u64,
}

#[derive(Serialize)]
struct LoreSnapshot {
    question: String,
    answer: String,
    layer: u8,
    reason: u8,
    embedding_delta: f32,
    effectiveness: f32,
    timestamp_ns: u64,
    seq: u64,
}
