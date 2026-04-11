// Qualia Coach Runner — Thought Theater graph pipeline.
//
// Observe → Propose → Factor Pressure → Coach Trim → Small Graph → Operational Slice
//
// Runs at ~10 Hz, reads all layer beliefs and the world model, maintains the
// Big Graph (speculative proposals) and distils it into the Small Graph
// (trusted core) and Operational Slice (planner compact state).

use qualia_shm::{LayerReader, ShmRegion};
use qualia_types::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static RUNNING: AtomicBool = AtomicBool::new(true);

const COACH_HZ: f64 = 10.0;

// Coach tuning parameters
const PROPOSE_CONFIDENCE_THRESHOLD: f32 = 0.3;
const VFE_FACTOR_THRESHOLD_MULT: f32 = 2.0;
const TRIM_ENERGY_THRESHOLD: f32 = 5.0;
const TRIM_MIN_GENERATION: u8 = 3;
const MERGE_COSINE_THRESHOLD: f32 = 0.9;
const PROMOTE_CONFIDENCE_THRESHOLD: f32 = 0.5;
const PROMOTE_MAX_ENERGY: f32 = 2.0;

// Factor pressure kernel parameters
const AGE_DECAY: f32 = 0.1;
#[cfg(any(feature = "metal", feature = "cuda"))]
const SUPPORT_WEIGHT: f32 = 1.0;
#[cfg(any(feature = "metal", feature = "cuda"))]
const CONTRADICT_WEIGHT: f32 = 1.5;

const THOUGHT_OBSERVE: u8 = 0;

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn main() {
    let shm_name =
        std::env::var("QUALIA_SHM_NAME").unwrap_or_else(|_| "/qualia_body".to_string());
    eprintln!("qualia-coach: opening shm '{shm_name}'");

    let shm = ShmRegion::open(&shm_name).unwrap_or_else(|e| {
        panic!("qualia-coach: failed to open shm: {e}");
    });

    // Register SIGTERM handler for graceful shutdown
    unsafe {
        let _ = signal_hook::low_level::register(signal_hook::consts::SIGTERM, || {
            RUNNING.store(false, Ordering::SeqCst);
        });
    }

    // Initialise theater header
    {
        let th_ptr = shm.as_ptr();
        unsafe {
            let th = &mut *(th_ptr.add(qualia_shm::THEATER_OFFSET) as *mut TheaterHeader);
            th.magic = THEATER_MAGIC;
        }
    }

    // Initialise graph capacities
    {
        let bg = shm.big_graph_mut();
        bg.header.node_capacity = MAX_BIG_NODES as u16;
        bg.header.edge_capacity = MAX_BIG_EDGES as u16;
        bg.header.node_count = 0;
        bg.header.edge_count = 0;

        let sg = shm.small_graph_mut();
        sg.header.node_capacity = MAX_SMALL_NODES as u16;
        sg.header.edge_capacity = MAX_SMALL_EDGES as u16;
        sg.header.node_count = 0;
        sg.header.edge_count = 0;
    }

    // GPU context for factor pressure (optional — falls back to CPU)
    #[cfg(feature = "metal")]
    let gpu = qualia_metal::TheaterMetalContext::new(AGE_DECAY, SUPPORT_WEIGHT, CONTRADICT_WEIGHT).ok();
    #[cfg(feature = "cuda")]
    let gpu = qualia_cuda::TheaterCudaContext::new(AGE_DECAY, SUPPORT_WEIGHT, CONTRADICT_WEIGHT).ok();
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let gpu: Option<()> = None;

    if gpu.is_some() {
        eprintln!("qualia-coach: GPU factor pressure enabled");
    } else {
        eprintln!("qualia-coach: running factor pressure on CPU (fallback)");
    }

    shm.emit_thought(
        255, // coach uses layer 255
        THOUGHT_OBSERVE,
        0.0,
        "coach: init, Thought Theater online",
    );

    eprintln!("qualia-coach: running at {COACH_HZ} Hz");

    let tick = Duration::from_secs_f64(1.0 / COACH_HZ);
    let mut next_id: u16 = 0;

    loop {
        if !RUNNING.load(Ordering::SeqCst) {
            eprintln!("qualia-coach: shutting down");
            break;
        }

        let cycle_start = Instant::now();

        // ── 1. OBSERVE: snapshot layer beliefs and world model ──
        let mut layer_vfe = [0.0f32; NUM_LAYERS];
        let mut layer_threshold = [0.0f32; NUM_LAYERS];
        let mut layer_beliefs: [Option<BeliefSlot>; NUM_LAYERS] = [None; NUM_LAYERS];
        for i in 0..NUM_LAYERS {
            let slot = shm.layer_slot(i);
            let reader = LayerReader::new(slot);
            let belief = *reader.read();
            layer_vfe[i] = belief.vfe;
            layer_beliefs[i] = Some(belief);

            #[cfg(feature = "metal")]
            { layer_threshold[i] = qualia_metal::default_params(i as u8).threshold; }
            #[cfg(feature = "cuda")]
            { layer_threshold[i] = qualia_cuda::default_params(i as u8).threshold; }
            #[cfg(not(any(feature = "metal", feature = "cuda")))]
            { layer_threshold[i] = 0.1; }
        }

        let world = shm.world_model();

        // ── 2. PROPOSE: inject new nodes into Big Graph ──
        let bg = shm.big_graph_mut();

        // Age existing nodes
        let nc = bg.header.node_count as usize;
        for i in 0..nc {
            if bg.nodes[i].kind as u8 != 0 {
                bg.nodes[i].generation = bg.nodes[i].generation.saturating_add(1);
            }
        }

        // Propose Object nodes from WorldModel
        for i in 0..world.num_objects.min(MAX_OBJECTS as u32) as usize {
            let obj = &world.objects[i];
            if obj.active == 0 || obj.confidence < PROPOSE_CONFIDENCE_THRESHOLD {
                continue;
            }
            if has_similar_node(bg, &obj.name, NodeKind::Object) {
                continue;
            }
            if let Some(slot) = find_empty_node(bg) {
                let node = &mut bg.nodes[slot];
                *node = empty_node();
                node.kind = NodeKind::Object;
                node.id = next_id;
                next_id = next_id.wrapping_add(1);
                node.label = obj.name;
                node.position = [obj.x, obj.y, 0.0];
                node.confidence = obj.confidence;
                node.timestamp_ns = now_ns();
                // Project scene embedding to NODE_EMBED_DIM by averaging groups of 4
                for d in 0..NODE_EMBED_DIM {
                    let base = d * (STATE_DIM / NODE_EMBED_DIM);
                    let mut sum = 0.0f32;
                    for k in 0..4 {
                        sum += world.scene_embedding[base + k];
                    }
                    node.embedding[d] = sum * 0.25;
                }
                if bg.header.node_count < bg.header.node_capacity {
                    bg.header.node_count = (slot as u16 + 1).max(bg.header.node_count);
                }
            }
        }

        // Propose Factor nodes from high-VFE layers
        for i in 0..NUM_LAYERS {
            if layer_vfe[i] > layer_threshold[i] * VFE_FACTOR_THRESHOLD_MULT {
                if has_factor_for_layer(bg, i as u8) {
                    continue;
                }
                if let Some(slot) = find_empty_node(bg) {
                    let node = &mut bg.nodes[slot];
                    *node = empty_node();
                    node.kind = NodeKind::Factor;
                    node.id = next_id;
                    next_id = next_id.wrapping_add(1);
                    write_label(&mut node.label, &format!("vfe_L{}", i));
                    node.source_layer = i as u8;
                    node.confidence = (layer_vfe[i] / layer_threshold[i]).min(10.0) * 0.1;
                    node.timestamp_ns = now_ns();
                    if let Some(ref belief) = layer_beliefs[i] {
                        for d in 0..NODE_EMBED_DIM {
                            let base = d * (STATE_DIM / NODE_EMBED_DIM);
                            let mut sum = 0.0f32;
                            for k in 0..4 {
                                sum += belief.residual[base + k].abs();
                            }
                            node.embedding[d] = sum * 0.25;
                        }
                    }
                    if bg.header.node_count < bg.header.node_capacity {
                        bg.header.node_count = (slot as u16 + 1).max(bg.header.node_count);
                    }
                }
            }
        }

        // Propose Region node from scene description (if non-empty and not already present)
        let scene_str = read_cstr(&world.scene);
        if !scene_str.is_empty() && !has_kind(bg, NodeKind::Region) {
            if let Some(slot) = find_empty_node(bg) {
                let node = &mut bg.nodes[slot];
                *node = empty_node();
                node.kind = NodeKind::Region;
                node.id = next_id;
                next_id = next_id.wrapping_add(1);
                write_label(&mut node.label, "scene");
                node.confidence = 0.6;
                node.timestamp_ns = now_ns();
                for d in 0..NODE_EMBED_DIM {
                    let base = d * (STATE_DIM / NODE_EMBED_DIM);
                    let mut sum = 0.0f32;
                    for k in 0..4 {
                        sum += world.scene_embedding[base + k];
                    }
                    node.embedding[d] = sum * 0.25;
                }
                if bg.header.node_count < bg.header.node_capacity {
                    bg.header.node_count = (slot as u16 + 1).max(bg.header.node_count);
                }
            }
        }

        // Auto-create edges between objects and regions (Contains)
        let nc = bg.header.node_count as usize;
        ensure_containment_edges(bg, nc);

        // Auto-create edges between factors and objects they relate to (Causes)
        ensure_factor_edges(bg, nc);

        // ── 3. FACTOR PRESSURE: compute energies ──
        let nc = bg.header.node_count as usize;
        #[allow(unused_variables)]
        let ec = bg.header.edge_count as usize;
        let pb = shm.pressure_buffer_mut();

        #[allow(unused_variables, unused_mut)]
        let mut used_gpu = false;

        #[cfg(feature = "metal")]
        if let Some(ref gpu_ctx) = gpu {
            if nc > 0 {
                gpu_ctx.dispatch_factor_pressure(
                    &mut bg.nodes[..nc],
                    &mut bg.edges[..ec.max(1)],
                    &mut pb.energies[..nc],
                    nc as u32,
                    ec as u32,
                );
                used_gpu = true;
            }
        }

        #[cfg(feature = "cuda")]
        if let Some(ref gpu_ctx) = gpu {
            if nc > 0 {
                gpu_ctx.dispatch_factor_pressure(
                    &mut bg.nodes[..nc],
                    &mut bg.edges[..ec.max(1)],
                    &mut pb.energies[..nc],
                    nc as u32,
                    ec as u32,
                );
                used_gpu = true;
            }
        }

        if !used_gpu && nc > 0 {
            // CPU fallback: simple energy from age + confidence
            for i in 0..nc {
                let node = &bg.nodes[i];
                if node.kind as u8 == 0 {
                    pb.energies[i] = 0.0;
                } else {
                    pb.energies[i] = AGE_DECAY * node.generation as f32
                        - node.confidence * 0.5;
                }
            }
        }
        pb.update_seq.fetch_add(1, Ordering::Release);

        // ── 4. COACH TRIM ──
        let mut trimmed = 0u32;
        let mut merged = 0u32;
        let tm = shm.trim_mask_mut();

        // Reset mask to all-keep
        for b in tm.bits.iter_mut() {
            *b = 0xFF;
        }

        for i in 0..nc {
            let node = &bg.nodes[i];
            if node.kind as u8 == 0 {
                continue;
            }

            // Trim: high energy + old enough
            if pb.energies[i] > TRIM_ENERGY_THRESHOLD && node.generation > TRIM_MIN_GENERATION {
                clear_bit(&mut tm.bits, i);
                bg.nodes[i].kind = NodeKind::Empty;
                bg.nodes[i].flags |= 0x04; // trimmed flag
                trimmed += 1;
                continue;
            }

            // Merge: find duplicate by embedding similarity
            for j in (i + 1)..nc {
                let other = &bg.nodes[j];
                if other.kind as u8 == 0 || other.kind != node.kind {
                    continue;
                }
                let sim = cosine_similarity(&node.embedding, &other.embedding);
                if sim > MERGE_COSINE_THRESHOLD {
                    // Keep the one with higher confidence
                    if node.confidence >= other.confidence {
                        clear_bit(&mut tm.bits, j);
                        bg.nodes[j].kind = NodeKind::Empty;
                        bg.nodes[j].flags |= 0x02; // merged flag
                    } else {
                        clear_bit(&mut tm.bits, i);
                        bg.nodes[i].kind = NodeKind::Empty;
                        bg.nodes[i].flags |= 0x02;
                    }
                    merged += 1;
                    break;
                }
            }
        }
        tm.update_seq.fetch_add(1, Ordering::Release);

        // ── 5. BUILD SMALL GRAPH ──
        let sg = shm.small_graph_mut();
        sg.header.node_count = 0;
        sg.header.edge_count = 0;
        let mut promoted = 0u32;

        // Map from big graph index → small graph index
        let mut big_to_small = [0xFFFFu16; MAX_BIG_NODES];

        for i in 0..nc {
            let node = &bg.nodes[i];
            if node.kind as u8 == 0 {
                continue;
            }
            // WhatIf nodes never promoted
            if node.kind == NodeKind::WhatIf {
                continue;
            }

            let qualifies = node.confidence >= PROMOTE_CONFIDENCE_THRESHOLD
                && pb.energies[i] < PROMOTE_MAX_ENERGY
                && node.generation >= 1;

            if !qualifies {
                continue;
            }

            let sg_idx = sg.header.node_count as usize;
            if sg_idx >= MAX_SMALL_NODES {
                break;
            }

            let mut promoted_node = bg.nodes[i];
            // Route → Path on promotion
            if promoted_node.kind == NodeKind::Route {
                promoted_node.kind = NodeKind::Path;
            }
            promoted_node.flags |= 0x01; // promoted flag
            sg.nodes[sg_idx] = promoted_node;
            big_to_small[i] = sg_idx as u16;
            sg.header.node_count += 1;
            promoted += 1;
        }

        // Copy edges that connect two promoted nodes
        let ec = bg.header.edge_count as usize;
        for i in 0..ec {
            let edge = &bg.edges[i];
            if edge.kind as u8 == 0 {
                continue;
            }
            let from_sg = big_to_small[edge.from as usize];
            let to_sg = big_to_small[edge.to as usize];
            if from_sg != 0xFFFF && to_sg != 0xFFFF {
                let sg_ei = sg.header.edge_count as usize;
                if sg_ei >= MAX_SMALL_EDGES {
                    break;
                }
                let mut sg_edge = *edge;
                sg_edge.from = from_sg;
                sg_edge.to = to_sg;
                sg.edges[sg_ei] = sg_edge;
                sg.header.edge_count += 1;
            }
        }

        sg.header.generation.fetch_add(1, Ordering::Release);

        // ── 6. DERIVE OPERATIONAL SLICE ──
        let ops = shm.operational_slice_mut();
        let sg_nc = sg.header.node_count as usize;

        // Pose: from the most confident Region node
        let mut best_region_conf = 0.0f32;
        for i in 0..sg_nc {
            let n = &sg.nodes[i];
            if n.kind == NodeKind::Region && n.confidence > best_region_conf {
                best_region_conf = n.confidence;
                ops.pose[0] = n.position[0];
                ops.pose[1] = n.position[1];
                ops.pose[2] = n.position[2];
                // roll/pitch/yaw stay 0 (no orientation info from graph)
            }
        }

        // Goal: from directive + Path nodes
        let directive = read_cstr(&world.directive);
        ops.goal_text = [0u8; MAX_GOAL_TEXT];
        let d_bytes = directive.as_bytes();
        let d_len = d_bytes.len().min(MAX_GOAL_TEXT - 1);
        ops.goal_text[..d_len].copy_from_slice(&d_bytes[..d_len]);

        // Average Path node positions as goal target
        let mut path_count = 0u32;
        let mut goal_sum = [0.0f32; 3];
        for i in 0..sg_nc {
            if sg.nodes[i].kind == NodeKind::Path {
                goal_sum[0] += sg.nodes[i].position[0];
                goal_sum[1] += sg.nodes[i].position[1];
                goal_sum[2] += sg.nodes[i].position[2];
                path_count += 1;
            }
        }
        if path_count > 0 {
            let inv = 1.0 / path_count as f32;
            ops.goal = [goal_sum[0] * inv, goal_sum[1] * inv, goal_sum[2] * inv];
        }

        // Hazards: Factor nodes with high energy in the Small Graph
        ops.hazard_count = 0;
        for i in 0..sg_nc {
            let n = &sg.nodes[i];
            if n.kind == NodeKind::Factor && n.energy > 1.0 {
                let hi = ops.hazard_count as usize;
                if hi >= MAX_HAZARDS {
                    break;
                }
                ops.hazards[hi] = Hazard {
                    kind: 3, // unknown
                    severity: (n.energy * 25.0).min(255.0) as u8,
                    _pad: [0; 2],
                    x: n.position[0],
                    y: n.position[1],
                    z: n.position[2],
                    radius: 0.5,
                };
                ops.hazard_count += 1;
            }
        }

        ops.confidence = if sg_nc > 0 {
            let total_conf: f32 = (0..sg_nc).map(|i| sg.nodes[i].confidence).sum();
            total_conf / sg_nc as f32
        } else {
            0.0
        };
        ops.timestamp_ns = now_ns();
        ops.update_seq.fetch_add(1, Ordering::Release);

        // ── 7. EMIT THOUGHTS ──
        let big_nodes_active = count_active(bg);
        let small_nodes_active = sg.header.node_count;
        shm.emit_thought(
            255,
            THOUGHT_OBSERVE,
            0.0,
            &format!(
                "coach: +{} proposed, -{} trimmed, ~{} merged, ^{} promoted | big={}/{} small={}/{}",
                count_new_nodes(bg),
                trimmed,
                merged,
                promoted,
                big_nodes_active,
                MAX_BIG_NODES,
                small_nodes_active,
                MAX_SMALL_NODES,
            ),
        );

        // Compact Big Graph: recount actual nodes (remove trailing empties)
        recount_nodes(bg);

        let elapsed = cycle_start.elapsed();
        if elapsed < tick {
            std::thread::sleep(tick - elapsed);
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────────

fn empty_node() -> GraphNode {
    GraphNode {
        kind: NodeKind::Empty,
        flags: 0,
        id: 0,
        parent_id: 0xFFFF,
        edge_start: 0,
        edge_count: 0,
        _pad0: [0; 2],
        label: [0; MAX_NODE_LABEL],
        position: [0.0; 3],
        embedding: [0.0; NODE_EMBED_DIM],
        energy: 0.0,
        confidence: 0.0,
        timestamp_ns: 0,
        source_layer: 0,
        generation: 0,
        _pad1: [0; 6],
    }
}

fn write_label(label: &mut [u8; MAX_NODE_LABEL], text: &str) {
    let bytes = text.as_bytes();
    let len = bytes.len().min(MAX_NODE_LABEL - 1);
    label[..len].copy_from_slice(&bytes[..len]);
}

fn read_cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).to_string()
}

fn has_similar_node(bg: &BigGraph, label: &[u8; MAX_NODE_LABEL], kind: NodeKind) -> bool {
    let nc = bg.header.node_count as usize;
    for i in 0..nc {
        let n = &bg.nodes[i];
        if n.kind == kind && n.label == *label {
            return true;
        }
    }
    false
}

fn has_factor_for_layer(bg: &BigGraph, layer: u8) -> bool {
    let nc = bg.header.node_count as usize;
    for i in 0..nc {
        let n = &bg.nodes[i];
        if n.kind == NodeKind::Factor && n.source_layer == layer {
            return true;
        }
    }
    false
}

fn has_kind(bg: &BigGraph, kind: NodeKind) -> bool {
    let nc = bg.header.node_count as usize;
    for i in 0..nc {
        if bg.nodes[i].kind == kind {
            return true;
        }
    }
    false
}

fn find_empty_node(bg: &BigGraph) -> Option<usize> {
    // First try within current count
    let nc = bg.header.node_count as usize;
    for i in 0..nc {
        if bg.nodes[i].kind as u8 == 0 {
            return Some(i);
        }
    }
    // Then try one past the end
    if nc < MAX_BIG_NODES {
        return Some(nc);
    }
    None
}

fn cosine_similarity(a: &[f32; NODE_EMBED_DIM], b: &[f32; NODE_EMBED_DIM]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..NODE_EMBED_DIM {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        dot / denom
    }
}

fn clear_bit(bits: &mut [u8; MAX_BIG_NODES / 8], index: usize) {
    let byte = index / 8;
    let bit = index % 8;
    if byte < bits.len() {
        bits[byte] &= !(1 << bit);
    }
}

fn count_active(bg: &BigGraph) -> u16 {
    let nc = bg.header.node_count as usize;
    let mut count = 0u16;
    for i in 0..nc {
        if bg.nodes[i].kind as u8 != 0 {
            count += 1;
        }
    }
    count
}

fn count_new_nodes(bg: &BigGraph) -> u16 {
    let nc = bg.header.node_count as usize;
    let mut count = 0u16;
    for i in 0..nc {
        if bg.nodes[i].kind as u8 != 0 && bg.nodes[i].generation == 0 {
            count += 1;
        }
    }
    count
}

fn recount_nodes(bg: &mut BigGraph) {
    // Find the last non-empty node and set node_count
    let mut last = 0u16;
    for i in 0..bg.header.node_count as usize {
        if bg.nodes[i].kind as u8 != 0 {
            last = i as u16 + 1;
        }
    }
    bg.header.node_count = last;
}

fn ensure_containment_edges(bg: &mut BigGraph, nc: usize) {
    // For each Object node, ensure there's a Contains edge from a Region
    for i in 0..nc {
        if bg.nodes[i].kind != NodeKind::Object {
            continue;
        }
        // Check if this object already has a Contains edge
        let has_contains = (0..bg.header.edge_count as usize).any(|e| {
            let edge = &bg.edges[e];
            edge.kind == EdgeKind::Contains
                && (edge.to == i as u16 || edge.from == i as u16)
        });
        if has_contains {
            continue;
        }
        // Find the nearest Region node
        let mut best_region: Option<usize> = None;
        let mut best_dist = f32::MAX;
        for j in 0..nc {
            if bg.nodes[j].kind == NodeKind::Region {
                let dx = bg.nodes[i].position[0] - bg.nodes[j].position[0];
                let dy = bg.nodes[i].position[1] - bg.nodes[j].position[1];
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best_region = Some(j);
                }
            }
        }
        if let Some(r) = best_region {
            let ei = bg.header.edge_count as usize;
            if ei < MAX_BIG_EDGES {
                bg.edges[ei] = GraphEdge {
                    from: r as u16,
                    to: i as u16,
                    kind: EdgeKind::Contains,
                    _pad: 0,
                    weight: 1.0,
                    tension: 0.0,
                };
                bg.header.edge_count += 1;
            }
        }
    }
}

fn ensure_factor_edges(bg: &mut BigGraph, nc: usize) {
    // Link Factor nodes to Object nodes via Causes edges
    for i in 0..nc {
        if bg.nodes[i].kind != NodeKind::Factor {
            continue;
        }
        let has_edge = (0..bg.header.edge_count as usize).any(|e| {
            let edge = &bg.edges[e];
            edge.from == i as u16 || edge.to == i as u16
        });
        if has_edge {
            continue;
        }
        // Connect to the most similar Object node by embedding
        let mut best_obj: Option<usize> = None;
        let mut best_sim = -1.0f32;
        for j in 0..nc {
            if bg.nodes[j].kind == NodeKind::Object {
                let sim = cosine_similarity(&bg.nodes[i].embedding, &bg.nodes[j].embedding);
                if sim > best_sim {
                    best_sim = sim;
                    best_obj = Some(j);
                }
            }
        }
        if let Some(obj) = best_obj {
            let ei = bg.header.edge_count as usize;
            if ei < MAX_BIG_EDGES {
                bg.edges[ei] = GraphEdge {
                    from: i as u16,
                    to: obj as u16,
                    kind: EdgeKind::Causes,
                    _pad: 0,
                    weight: best_sim.max(0.1),
                    tension: 0.0,
                };
                bg.header.edge_count += 1;
            }
        }
    }
}
