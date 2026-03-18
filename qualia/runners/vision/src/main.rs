use qualia_shm::ShmRegion;
use qualia_types::*;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
#[allow(unused_imports)]
use qualia_types::{MAX_QUESTION_TEXT, MAX_LORE_ENTRIES};
use std::time::{Duration, Instant, SystemTime};

/// How often to call Gemini API (seconds).
/// Override with QUALIA_LLM_INTERVAL env var.
const LLM_INTERVAL_SECS_DEFAULT: u64 = 30;

/// Maximum LLM calls per session. After this, fall back to offline.
/// Override with QUALIA_LLM_MAX_CALLS env var.
const LLM_MAX_CALLS_DEFAULT: u64 = 50;

/// Capture resolution for LLM (higher than the 8x8 sensor feed).
const CAPTURE_W: u32 = 640;
const CAPTURE_H: u32 = 480;

/// Gemini model for vision (scene understanding).
const GEMINI_VISION_MODEL: &str = "gemini-2.5-flash";

/// Gemini model for text embeddings → 64-dim scene vector.
const GEMINI_EMBEDDING_MODEL: &str = "gemini-embedding-2-preview";

fn main() {
    let shm_name =
        std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".to_string());
    let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();

    let llm_interval = std::env::var("QUALIA_LLM_INTERVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(LLM_INTERVAL_SECS_DEFAULT);

    let llm_max_calls = std::env::var("QUALIA_LLM_MAX_CALLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(LLM_MAX_CALLS_DEFAULT);

    eprintln!("qualia-vision: opening shm '{shm_name}'");

    let shm = ShmRegion::open(&shm_name).unwrap_or_else(|e| {
        panic!("qualia-vision: failed to open shm: {e}");
    });

    if api_key.is_empty() {
        eprintln!("qualia-vision: WARNING: GEMINI_API_KEY not set");
        eprintln!("qualia-vision: Running in offline mode — synthetic world model only");
        run_offline_loop(&shm);
    } else {
        eprintln!("qualia-vision: Gemini API enabled (vision + embeddings)");
        eprintln!("qualia-vision: Interval: {}s, Max calls: {}", llm_interval, llm_max_calls);
        run_vision_loop(&shm, &api_key, llm_interval, llm_max_calls);
    }
}

/// Offline mode: generate a synthetic world model from sensor data patterns.
fn run_offline_loop(shm: &ShmRegion) {
    let world = shm.world_model_mut();
    set_default_directive(world);

    shm.emit_thought(255, 0, 0.0, "vision: offline mode, no GEMINI_API_KEY");

    let mut prev_brightness: f32 = 0.0;
    let mut prev_num_objects: u32 = 0;
    let mut tick: u64 = 0;
    loop {
        // Read L6 sensor data to derive basic scene info
        let sensor = {
            let slot = shm.layer_slot(NUM_LAYERS - 1); // sensor layer
            let reader = qualia_shm::LayerReader::new(slot);
            *reader.read()
        };

        let brightness: f32 = sensor.mean.iter().sum::<f32>() / STATE_DIM as f32;
        let variance: f32 = sensor.mean.iter()
            .map(|v| (v - brightness).powi(2))
            .sum::<f32>() / STATE_DIM as f32;
        let edge_energy: f32 = sensor.mean.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f32>() / (STATE_DIM - 1) as f32;

        for i in 0..STATE_DIM {
            world.scene_embedding[i] = match i % 8 {
                0 => brightness,
                1 => variance,
                2 => edge_energy,
                3 => sensor.mean[i],
                4 => (brightness - 0.5).abs(),
                5 => if variance > 0.01 { 1.0 } else { 0.0 },
                6 => sensor.precision[i] * 0.01,
                _ => (tick as f32 * 0.001).sin() * 0.1,
            };
        }

        world.num_objects = 0;

        let scene_desc = format!(
            "offline: bright={:.2} var={:.3} edge={:.3}\0",
            brightness, variance, edge_energy
        );
        let bytes = scene_desc.as_bytes();
        let len = bytes.len().min(MAX_SCENE_LEN - 1);
        world.scene[..len].copy_from_slice(&bytes[..len]);
        world.scene[len] = 0;

        world.activity = [0u8; MAX_ACTIVITY_LEN];
        if variance > 0.02 {
            world.activity[..16].copy_from_slice(b"motion detected\0");
        } else if brightness > 0.3 {
            world.activity[..18].copy_from_slice(b"static scene, lit\0");
        } else {
            world.activity[..10].copy_from_slice(b"dark/idle\0");
        }

        world.last_vision_ns = now_ns();
        world.vision_frame_count += 1;
        world.update_seq.store(
            world.update_seq.load(Ordering::Relaxed) + 1,
            Ordering::Release,
        );

        tick += 1;

        if tick % 30 == 0 {
            let brightness_change = (brightness - prev_brightness).abs();
            if brightness_change > 0.1 || world.num_objects != prev_num_objects || variance > 0.02 {
                shm.emit_thought(255, 0, brightness_change, &format!(
                    "sensor: bright={:.2} var={:.3} edge={:.3} obj={} delta_b={:.2}",
                    brightness, variance, edge_energy, world.num_objects, brightness_change
                ));
            } else if tick % 150 == 0 {
                shm.emit_thought(255, 0, 0.0, &format!(
                    "sensor: bright={:.2} var={:.3} obj={} tick={}",
                    brightness, world.num_objects, variance, tick
                ));
            }
        }
        prev_brightness = brightness;
        prev_num_objects = world.num_objects;

        if tick % 60 == 0 {
            eprintln!(
                "qualia-vision: offline tick {}, {} objects, brightness={:.2}",
                tick, world.num_objects, brightness
            );
        }

        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Online mode: capture frames, call Gemini Vision + Embedding APIs.
fn run_vision_loop(shm: &ShmRegion, api_key: &str, interval_secs: u64, max_calls: u64) {
    let world = shm.world_model_mut();
    set_default_directive(world);

    shm.emit_thought(255, 0, 0.0, &format!(
        "vision: gemini online, budget={}, interval={}s",
        max_calls, interval_secs
    ));

    let mut last_llm = Instant::now() - Duration::from_secs(interval_secs + 1);
    let mut calls_made: u64 = 0;
    let mut budget_exhausted = false;

    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_llm).as_secs();

        if !budget_exhausted
            && elapsed >= interval_secs
        {
            eprintln!("qualia-vision: triggering capture ({}s since last)", elapsed);
            last_llm = now;

            let _remaining = max_calls - calls_made;

            match capture_frame() {
                Ok(jpeg_data) => {
                    let b64 = base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        &jpeg_data,
                    );

                    let directive = read_cstr(&world.directive);

                    // Step 0: Harvest pending questions from layers
                    let pending_questions = harvest_questions(shm);

                    // Step 1: Gemini Vision — understand the scene
                    eprintln!("qualia-vision: calling Gemini Vision API ({}B image)...", b64.len());
                    match call_gemini_vision(api_key, &b64, &directive, &pending_questions) {
                        Ok((response, vision_usage)) => {
                            // Track vision API tokens
                            world.gemini_input_tokens += vision_usage.input_tokens;
                            world.gemini_output_tokens += vision_usage.output_tokens;

                            let obj_names: Vec<&str> = response.objects.iter()
                                .map(|o| o.name.as_str())
                                .collect();
                            shm.emit_thought(255, 0, 0.0, &format!(
                                "gemini vision #{}: {} obj={} [{}]",
                                calls_made + 1,
                                &response.scene[..response.scene.len().min(80)],
                                response.objects.len(),
                                obj_names.join(",")
                            ));

                            apply_vision_response(world, &response);

                            // Step 2: Gemini Embedding — generate a real semantic
                            // embedding from the scene description. This is what
                            // teaches the lower layers what to expect.
                            let embed_text = format!(
                                "{} {} {}",
                                response.scene, response.activity,
                                obj_names.join(" ")
                            );
                            match call_gemini_embedding(api_key, &embed_text) {
                                Ok((embedding, embed_tokens)) => {
                                    // Track embedding API tokens
                                    world.gemini_embedding_tokens += embed_tokens;

                                    // Project the embedding into our 64-dim space
                                    project_embedding_to_scene(world, &embedding);
                                    let emb_norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
                                    shm.emit_thought(255, 3, 0.0, &format!(
                                        "embedding: {}d, norm={:.3}, projected to 64d",
                                        embedding.len(), emb_norm
                                    ));
                                }
                                Err(e) => {
                                    // Fall back to hash-based embedding
                                    eprintln!("qualia-vision: embedding API error: {e}");
                                    generate_hash_embedding(world, &response);
                                }
                            }

                            // Step 3: Answer layer questions → LORE
                            if !pending_questions.is_empty() && !response.lore_answers.is_empty() {
                                let old_emb: [f32; STATE_DIM] = world.scene_embedding;
                                for (i, answer) in response.lore_answers.iter().enumerate() {
                                    if i < pending_questions.len() {
                                        let (layer, reason, question) = &pending_questions[i];
                                        let emb_delta: f32 = (0..STATE_DIM)
                                            .map(|d| (world.scene_embedding[d] - old_emb[d]).powi(2))
                                            .sum::<f32>()
                                            .sqrt();
                                        shm.emit_lore(question, answer, *layer, *reason, emb_delta, 0.0);
                                        shm.emit_thought(255, 3, 0.0, &format!(
                                            "lore L{} r={}: Q={} A={}",
                                            layer, reason,
                                            if question.len() > 50 { &question[..50] } else { question },
                                            if answer.len() > 60 { &answer[..60] } else { answer },
                                        ));
                                    }
                                }
                            }

                            world.last_llm_ns = now_ns();
                            world.llm_call_count += 1;
                            calls_made += 1;
                            world.update_seq.store(
                                world.update_seq.load(Ordering::Relaxed) + 1,
                                Ordering::Release,
                            );

                            eprintln!(
                                "qualia-vision: Gemini call #{}/{} — {} objects",
                                calls_made, max_calls, world.num_objects
                            );

                            if calls_made >= max_calls {
                                budget_exhausted = true;
                                shm.emit_thought(255, 5, 0.0, &format!(
                                    "budget exhausted: {}/{} calls, sensor-only mode",
                                    calls_made, max_calls
                                ));
                                eprintln!(
                                    "qualia-vision: Gemini budget exhausted ({}/{}). Continuing with sensor data only.",
                                    calls_made, max_calls
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("qualia-vision: Gemini vision error: {}", e);
                            shm.emit_thought(255, 5, 0.0, &format!(
                                "vision err: {}", e
                            ));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("qualia-vision: capture error: {}", e);
                    shm.emit_thought(255, 5, 0.0, &format!(
                        "capture err: {}", e
                    ));
                }
            }
        }

        // Always update embedding from live sensor data between Gemini calls
        update_embedding_from_sensor(shm, world);

        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Read the latest JPEG snapshot written by qualia-camera.
/// The camera runner writes /tmp/qualia-camera-latest.jpg at ~1fps via ffmpeg.
/// Falls back to direct ffmpeg capture if the snapshot isn't available.
fn capture_frame() -> Result<Vec<u8>, String> {
    let snapshot_path = "/tmp/qualia-camera-latest.jpg";

    // Primary: read snapshot from camera runner (no device conflict)
    if let Ok(data) = std::fs::read(snapshot_path) {
        if data.len() > 100 && data[0] == 0xFF && data[1] == 0xD8 {
            eprintln!("qualia-vision: read snapshot from camera ({}B)", data.len());
            return Ok(data);
        }
    }

    // Fallback: direct capture (will fail if camera runner holds the device)
    let device = std::env::var("CAMERA_DEVICE").unwrap_or_else(|_| {
        if cfg!(target_os = "macos") { "0".into() }
        else { "/dev/video0".into() }
    });

    eprintln!("qualia-vision: no snapshot, trying direct ffmpeg capture...");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-hide_banner", "-loglevel", "error"]);
    if cfg!(target_os = "macos") {
        cmd.args(["-f", "avfoundation", "-framerate", "30",
                  "-video_size", &format!("{}x{}", CAPTURE_W, CAPTURE_H), "-i", &device]);
    } else {
        cmd.args(["-f", "v4l2",
                  "-video_size", &format!("{}x{}", CAPTURE_W, CAPTURE_H), "-i", &device]);
    }
    cmd.args(["-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "-q:v", "5", "pipe:1"])
       .stdin(Stdio::null())
       .stdout(Stdio::piped())
       .stderr(Stdio::piped());
    let output = cmd.output().map_err(|e| format!("ffmpeg: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg failed: {stderr}"));
    }

    if output.stdout.is_empty() {
        return Err("ffmpeg produced no output".into());
    }

    Ok(output.stdout)
}

// ── Gemini Vision API ───────────────────────────────────────────────

/// Token usage from a Gemini API call.
struct GeminiUsage {
    input_tokens: u64,
    output_tokens: u64,
}

/// Harvest pending questions from all layer slots. Returns (layer, reason, question_text).
fn harvest_questions(shm: &ShmRegion) -> Vec<(u8, u8, String)> {
    let mut questions = Vec::new();
    for i in 0..NUM_LAYERS {
        let slot = shm.layer_slot(i);
        if slot.question.pending.load(Ordering::Acquire) {
            let text = read_cstr(&slot.question.text);
            if !text.is_empty() {
                questions.push((slot.question.layer, slot.question.reason, text));
            }
            // Clear the pending flag
            slot.question.pending.store(false, Ordering::Release);
        }
    }
    questions
}

/// Call Gemini Vision to understand a camera frame.
/// If layers have pending questions, include them in the prompt for LORE generation.
fn call_gemini_vision(
    api_key: &str,
    image_b64: &str,
    directive: &str,
    questions: &[(u8, u8, String)],
) -> Result<(VisionResponse, GeminiUsage), String> {
    let mut prompt = format!(
        "You are the visual cortex of an autonomous system. \
         Analyze this image and respond with ONLY valid JSON (no markdown, no code fences):\n\
         {{\n\
           \"scene\": \"<one sentence describing what you see>\",\n\
           \"activity\": \"<what is happening>\",\n\
           \"objects\": [\n\
             {{\"name\": \"<object>\", \"confidence\": 0.0-1.0, \"x\": 0.0-1.0, \"y\": 0.0-1.0}}\n\
           ]"
    );

    if !questions.is_empty() {
        prompt.push_str(",\n           \"lore_answers\": [\n");
        for (i, (layer, _reason, q)) in questions.iter().enumerate() {
            if i > 0 { prompt.push_str(",\n"); }
            prompt.push_str(&format!(
                "             \"<answer L{}'s question: {}>\"",
                layer, q
            ));
        }
        prompt.push_str("\n           ]");
    }

    prompt.push_str(&format!(
        "\n         }}\n\n\
         System directive: {directive}\n\
         Keep descriptions factual and concise."
    ));

    if !questions.is_empty() {
        prompt.push_str(&format!(
            "\n\nThe system's processing layers have {} question(s) about what they're perceiving. \
             Answer each one concisely based on what you see in the image. \
             These answers become LORE — accumulated world-knowledge the system uses to build predictions.",
            questions.len()
        ));
    }

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        GEMINI_VISION_MODEL, api_key
    );

    let body = serde_json::json!({
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64
                    }
                },
                {
                    "text": prompt
                }
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.1
        }
    });

    let resp = ureq::post(&url)
        .set("content-type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("Gemini vision request failed: {e}"))?;

    let resp_text = resp.into_string()
        .map_err(|e| format!("Response read error: {e}"))?;
    let resp_body: serde_json::Value = serde_json::from_str(&resp_text)
        .map_err(|e| format!("JSON parse error: {e}"))?;

    // Extract text from Gemini response
    let text = resp_body["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| format!("No text in Gemini response: {}", resp_text))?;

    // Strip markdown code fences if present
    let clean = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let vision: VisionResponse = serde_json::from_str(clean)
        .map_err(|e| format!("Vision JSON parse error: {e}\nRaw: {text}"))?;

    // Extract token usage
    let usage = GeminiUsage {
        input_tokens: resp_body["usageMetadata"]["promptTokenCount"].as_u64().unwrap_or(0),
        output_tokens: resp_body["usageMetadata"]["candidatesTokenCount"].as_u64().unwrap_or(0),
    };

    Ok((vision, usage))
}

// ── Gemini Embedding API ────────────────────────────────────────────

/// Call Gemini Embedding API to get a real semantic embedding vector.
/// This is the key innovation: instead of a hash-based fake embedding,
/// we get genuine semantic representations that teach the lower layers
/// what patterns to expect in the world.
fn call_gemini_embedding(api_key: &str, text: &str) -> Result<(Vec<f32>, u64), String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent?key={}",
        GEMINI_EMBEDDING_MODEL, api_key
    );

    let body = serde_json::json!({
        "model": format!("models/{}", GEMINI_EMBEDDING_MODEL),
        "content": {
            "parts": [{
                "text": text
            }]
        },
        "outputDimensionality": STATE_DIM
    });

    let resp = ureq::post(&url)
        .set("content-type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("Gemini embedding request failed: {e}"))?;

    let resp_text = resp.into_string()
        .map_err(|e| format!("Response read error: {e}"))?;
    let resp_body: serde_json::Value = serde_json::from_str(&resp_text)
        .map_err(|e| format!("JSON parse error: {e}"))?;

    let values = resp_body["embedding"]["values"]
        .as_array()
        .ok_or_else(|| format!("No embedding values in response: {}", resp_text))?;

    let embedding: Vec<f32> = values.iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    if embedding.is_empty() {
        return Err("Empty embedding returned".into());
    }

    let token_count = resp_body["usageMetadata"]["totalTokenCount"].as_u64().unwrap_or(0);

    Ok((embedding, token_count))
}

/// Project a Gemini embedding (potentially > 64 dims) into our 64-dim scene space.
/// If Gemini returns exactly 64 dims (via outputDimensionality), this is a direct copy.
/// Otherwise we average-pool to fit.
fn project_embedding_to_scene(world: &mut WorldModel, embedding: &[f32]) {
    if embedding.len() == STATE_DIM {
        // Direct copy — ideal case
        world.scene_embedding.copy_from_slice(embedding);
    } else if embedding.len() > STATE_DIM {
        // Average-pool: divide embedding into STATE_DIM bins
        let bin_size = embedding.len() as f32 / STATE_DIM as f32;
        for i in 0..STATE_DIM {
            let start = (i as f32 * bin_size) as usize;
            let end = ((i + 1) as f32 * bin_size) as usize;
            let end = end.min(embedding.len());
            let sum: f32 = embedding[start..end].iter().sum();
            let count = (end - start) as f32;
            world.scene_embedding[i] = sum / count;
        }
    } else {
        // Embedding is shorter — copy what we have, zero-pad
        for i in 0..STATE_DIM {
            world.scene_embedding[i] = if i < embedding.len() {
                embedding[i]
            } else {
                0.0
            };
        }
    }
}

// ── Shared types and helpers ────────────────────────────────────────

#[derive(serde::Deserialize)]
struct VisionResponse {
    scene: String,
    activity: String,
    objects: Vec<VisionObject>,
    /// Answers to layer questions — becomes LORE.
    #[serde(default)]
    lore_answers: Vec<String>,
}

#[derive(serde::Deserialize)]
struct VisionObject {
    name: String,
    confidence: f32,
    #[serde(default)]
    x: f32,
    #[serde(default)]
    y: f32,
}

/// Apply Gemini Vision response to the WorldModel in shared memory.
fn apply_vision_response(world: &mut WorldModel, resp: &VisionResponse) {
    // Scene description
    let scene_bytes = resp.scene.as_bytes();
    let len = scene_bytes.len().min(MAX_SCENE_LEN - 1);
    world.scene[..len].copy_from_slice(&scene_bytes[..len]);
    world.scene[len] = 0;

    // Activity
    let act_bytes = resp.activity.as_bytes();
    let act_len = act_bytes.len().min(MAX_ACTIVITY_LEN - 1);
    world.activity[..act_len].copy_from_slice(&act_bytes[..act_len]);
    world.activity[act_len] = 0;

    // Objects
    world.num_objects = 0;
    for obj in resp.objects.iter().take(MAX_OBJECTS) {
        let idx = world.num_objects as usize;
        let name_bytes = obj.name.as_bytes();
        let name_len = name_bytes.len().min(MAX_OBJECT_NAME - 1);
        world.objects[idx].name = [0u8; MAX_OBJECT_NAME];
        world.objects[idx].name[..name_len].copy_from_slice(&name_bytes[..name_len]);
        world.objects[idx].confidence = obj.confidence;
        world.objects[idx].x = obj.x;
        world.objects[idx].y = obj.y;
        world.objects[idx].active = 1;
        world.num_objects += 1;
    }
}

/// Fallback: hash-based embedding when Gemini embedding API fails.
fn generate_hash_embedding(world: &mut WorldModel, resp: &VisionResponse) {
    let scene_full = format!("{} {}", resp.scene, resp.activity);
    for i in 0..STATE_DIM {
        let mut val: f32 = 0.0;
        for (j, ch) in scene_full.bytes().enumerate() {
            val += ((ch as f32) * ((i * 7 + j * 13) as f32).sin()) * 0.01;
        }
        for obj in resp.objects.iter().take(MAX_OBJECTS) {
            val += obj.confidence * ((obj.x * (i as f32 + 1.0)).sin() * 0.1);
        }
        world.scene_embedding[i] = val.tanh();
    }
}

/// Between Gemini calls, update the scene embedding with live sensor data.
fn update_embedding_from_sensor(shm: &ShmRegion, world: &mut WorldModel) {
    let slot = shm.layer_slot(NUM_LAYERS - 1); // sensor layer
    let reader = qualia_shm::LayerReader::new(slot);
    let sensor = reader.read();

    // Blend sensor data into the scene embedding (low-pass filter)
    let alpha = 0.05_f32;
    for i in 0..STATE_DIM {
        let sensor_val = sensor.mean[i] * 2.0 - 1.0; // normalize to [-1, 1]
        world.scene_embedding[i] = world.scene_embedding[i] * (1.0 - alpha) + sensor_val * alpha;
    }

    world.last_vision_ns = now_ns();
    world.vision_frame_count += 1;
}

fn set_default_directive(world: &mut WorldModel) {
    let directive = b"Observe and understand the environment. Report what you see.\0";
    world.directive[..directive.len()].copy_from_slice(directive);
}

fn read_cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).to_string()
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}
