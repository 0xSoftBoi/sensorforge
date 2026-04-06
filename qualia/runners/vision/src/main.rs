use qualia_shm::ShmRegion;
use qualia_types::*;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
#[allow(unused_imports)]
use qualia_types::{MAX_QUESTION_TEXT, MAX_LORE_ENTRIES};
use std::time::{Duration, Instant, SystemTime};

/// How often to call Gemini API (seconds).
/// Override with QUALIA_LLM_INTERVAL env var.
const LLM_INTERVAL_SECS_DEFAULT: u64 = 300;

/// Maximum LLM calls per session. After this, fall back to offline.
/// Override with QUALIA_LLM_MAX_CALLS env var.
const LLM_MAX_CALLS_DEFAULT: u64 = 10;

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

/// Online mode: Gemini demoted to LORE oracle (Phase 2.4).
///
/// Primary perception is handled by local YOLO + local embeddings
/// (qualia_detect.py + qualia_embed.py). Gemini is called ONLY when:
/// 1. Layers pose questions via QuestionSlot (hard semantic reasoning)
/// 2. Minimum interval has passed since last call
/// 3. Budget hasn't been exhausted
///
/// This dramatically reduces API costs while keeping Gemini available
/// for the questions that local models can't answer.
fn run_vision_loop(shm: &ShmRegion, api_key: &str, interval_secs: u64, max_calls: u64) {
    let world = shm.world_model_mut();
    set_default_directive(world);

    // Check if local detection/embedding services are running
    let local_detect_available = std::path::Path::new("/tmp/qualia_detections.json").exists();
    if local_detect_available {
        eprintln!("qualia-vision: local detection active — Gemini demoted to LORE oracle");
    } else {
        eprintln!("qualia-vision: no local detection — Gemini handles both perception + LORE");
    }

    shm.emit_thought(255, 0, 0.0, &format!(
        "vision: gemini=LORE oracle, budget={}, local_detect={}",
        max_calls, local_detect_available
    ));

    let mut last_llm = Instant::now() - Duration::from_secs(interval_secs + 1);
    let mut calls_made: u64 = 0;
    let mut budget_exhausted = false;
    // Phase 8: adaptive embedding filter state
    let mut prev_sensor = [0.0_f32; STATE_DIM];
    let mut innovation_ema: f32 = 0.01;
    let mut last_detect_check = Instant::now();

    loop {
        let now = Instant::now();
        let elapsed_since_llm = now.duration_since(last_llm).as_secs();

        // Phase 2.4: Check for local detections (written by qualia_detect.py)
        if now.duration_since(last_detect_check).as_millis() > 500 {
            last_detect_check = now;
            read_local_detections(world);
        }

        // Harvest pending questions from layers
        let pending_questions = harvest_questions(shm);
        let has_questions = !pending_questions.is_empty();

        // Phase 2.4: Gemini called ONLY when layers have questions
        // OR when there are no local services and the interval has passed
        let should_call_gemini = !budget_exhausted && elapsed_since_llm >= interval_secs
            && (has_questions || !local_detect_available);

        if should_call_gemini {
            eprintln!(
                "qualia-vision: Gemini call ({}s elapsed, {} questions, local={})",
                elapsed_since_llm, pending_questions.len(), local_detect_available
            );
            last_llm = Instant::now();

            match capture_frame() {
                Ok(jpeg_data) => {
                    let b64 = base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        &jpeg_data,
                    );

                    let directive = read_cstr(&world.directive);

                    // Call Gemini Vision
                    match call_gemini_vision(api_key, &b64, &directive, &pending_questions) {
                        Ok((response, vision_usage)) => {
                            world.gemini_input_tokens += vision_usage.input_tokens;
                            world.gemini_output_tokens += vision_usage.output_tokens;

                            let obj_names: Vec<&str> = response.objects.iter()
                                .map(|o| o.name.as_str())
                                .collect();

                            // Only update scene/objects from Gemini if no local detection
                            if !local_detect_available {
                                apply_vision_response(world, &response);
                            }

                            shm.emit_thought(255, 0, 0.0, &format!(
                                "gemini LORE #{}: {} [{}] q={}",
                                calls_made + 1,
                                &response.scene[..response.scene.len().min(60)],
                                obj_names.join(","),
                                pending_questions.len()
                            ));

                            // Embedding: use Gemini only if no local embedder
                            if !std::path::Path::new("/tmp/qualia_detections.json").exists() {
                                let embed_text = format!(
                                    "{} {} {}",
                                    response.scene, response.activity,
                                    obj_names.join(" ")
                                );
                                match call_gemini_embedding(api_key, &embed_text) {
                                    Ok((embedding, embed_tokens)) => {
                                        world.gemini_embedding_tokens += embed_tokens;
                                        project_embedding_to_scene(world, &embedding);
                                    }
                                    Err(e) => {
                                        eprintln!("qualia-vision: embedding API error: {e}");
                                        generate_hash_embedding(world, &response);
                                    }
                                }
                            }

                            // Answer layer questions → LORE (the primary reason for calling)
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
                                "qualia-vision: Gemini #{}/{} — {} LORE answers",
                                calls_made, max_calls, response.lore_answers.len()
                            );

                            if calls_made >= max_calls {
                                budget_exhausted = true;
                                shm.emit_thought(255, 5, 0.0, &format!(
                                    "budget exhausted: {}/{} calls",
                                    calls_made, max_calls
                                ));
                            }
                        }
                        Err(e) => {
                            eprintln!("qualia-vision: Gemini error: {}", e);
                            shm.emit_thought(255, 5, 0.0, &format!("gemini err: {}", e));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("qualia-vision: capture error: {}", e);
                }
            }
        }

        // Always update embedding from live sensor data (Phase 8: adaptive alpha)
        update_embedding_from_sensor(shm, world, &mut prev_sensor, &mut innovation_ema);

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

// ── Local Detection Reader (Phase 2.4) ──────────────────────────────

/// Read local YOLO detections from /tmp/qualia_detections.json
/// (written by qualia_detect.py at 1-5Hz).
fn read_local_detections(world: &mut WorldModel) {
    let path = "/tmp/qualia_detections.json";
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return,
    };

    let json: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Check freshness (skip if older than 5 seconds)
    if let Some(ts) = json["ts"].as_f64() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        if now - ts > 5.0 {
            return;
        }
    }

    // Update objects in world model
    if let Some(objects) = json["objects"].as_array() {
        world.num_objects = 0;
        for obj in objects.iter().take(MAX_OBJECTS) {
            let idx = world.num_objects as usize;
            if let Some(name) = obj["name"].as_str() {
                let name_bytes = name.as_bytes();
                let name_len = name_bytes.len().min(MAX_OBJECT_NAME - 1);
                world.objects[idx].name = [0u8; MAX_OBJECT_NAME];
                world.objects[idx].name[..name_len].copy_from_slice(&name_bytes[..name_len]);
                world.objects[idx].confidence = obj["confidence"].as_f64().unwrap_or(0.0) as f32;
                world.objects[idx].x = obj["x"].as_f64().unwrap_or(0.5) as f32;
                world.objects[idx].y = obj["y"].as_f64().unwrap_or(0.5) as f32;
                world.objects[idx].active = 1;
                world.num_objects += 1;
            }
        }

        // Build scene description from detections
        let names: Vec<String> = objects.iter()
            .filter_map(|o| o["name"].as_str().map(|s| s.to_string()))
            .collect();
        if !names.is_empty() {
            let desc = format!("local detect: {}\0", names.join(", "));
            let bytes = desc.as_bytes();
            let len = bytes.len().min(MAX_SCENE_LEN - 1);
            world.scene[..len].copy_from_slice(&bytes[..len]);
            world.scene[len] = 0;

            world.activity = [0u8; MAX_ACTIVITY_LEN];
            let act = format!("{} objects detected\0", names.len());
            let act_bytes = act.as_bytes();
            let act_len = act_bytes.len().min(MAX_ACTIVITY_LEN - 1);
            world.activity[..act_len].copy_from_slice(&act_bytes[..act_len]);
        }

        world.vision_frame_count += 1;
        world.update_seq.store(
            world.update_seq.load(Ordering::Relaxed) + 1,
            Ordering::Release,
        );
    }
}

// ── Gemini Vision API ───────────────────────────────────────────────

/// Token usage from a Gemini API call.
struct GeminiUsage {
    input_tokens: u64,
    output_tokens: u64,
}

/// Harvest pending questions from all layer slots. Returns (layer, reason, question_text).
/// Phase 8: questions are sorted by urgency — higher VFE layers and persistent-confusion
/// (reason=0) rank first, so the most critical questions get Gemini's attention.
fn harvest_questions(shm: &ShmRegion) -> Vec<(u8, u8, String)> {
    let mut questions = Vec::new();
    for i in 0..NUM_LAYERS {
        let slot = shm.layer_slot(i);
        if slot.question.pending.load(Ordering::Acquire) {
            let text = read_cstr(&slot.question.text);
            if !text.is_empty() {
                let vfe = slot.question.vfe;
                let reason = slot.question.reason;
                let layer = slot.question.layer;
                // Urgency: higher VFE = more urgent, reason 0 (persistent confusion) = most urgent
                let urgency = vfe * match reason {
                    0 => 3.0,  // persistent high VFE — can't learn alone
                    2 => 2.0,  // novel pattern spike
                    1 => 1.5,  // compression plateau
                    _ => 1.0,
                };
                questions.push((layer, reason, text, urgency));
            }
            slot.question.pending.store(false, Ordering::Release);
        }
    }
    // Sort by urgency descending — most critical questions first
    questions.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    // Drop urgency from the output
    questions.into_iter().map(|(l, r, t, _)| (l, r, t)).collect()
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
            "maxOutputTokens": 4096,
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
/// Phase 8: adaptive alpha — when sensor data changes rapidly (high innovation),
/// increase alpha to track faster; when stable, decrease to reduce noise.
/// This is a simplified Kalman-inspired gain: alpha ∝ innovation / (innovation + noise).
fn update_embedding_from_sensor(shm: &ShmRegion, world: &mut WorldModel,
                                 prev_sensor: &mut [f32; STATE_DIM],
                                 innovation_ema: &mut f32) {
    let slot = shm.layer_slot(NUM_LAYERS - 1); // sensor layer
    let reader = qualia_shm::LayerReader::new(slot);
    let sensor = reader.read();

    // Compute innovation: how much did the sensor change since last read?
    let mut innovation: f32 = 0.0;
    for i in 0..STATE_DIM {
        let sensor_val = sensor.mean[i] * 2.0 - 1.0;
        let diff = sensor_val - prev_sensor[i];
        innovation += diff * diff;
        prev_sensor[i] = sensor_val;
    }
    innovation = innovation.sqrt() / STATE_DIM as f32;

    // EMA of innovation for smooth adaptation
    *innovation_ema = *innovation_ema * 0.9 + innovation * 0.1;

    // Adaptive alpha: high innovation → alpha up to 0.3, low → down to 0.02
    // Kalman-style gain: K = innovation / (innovation + measurement_noise)
    let noise_floor = 0.005_f32;
    let alpha = (*innovation_ema / (*innovation_ema + noise_floor)).clamp(0.02, 0.3);

    for i in 0..STATE_DIM {
        world.scene_embedding[i] = world.scene_embedding[i] * (1.0 - alpha) + prev_sensor[i] * alpha;
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
