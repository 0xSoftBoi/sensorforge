#include <metal_stdlib>
using namespace metal;

// Factor pressure kernel — computes per-node energy from edge tensions.
// Part of the Thought Theater: Big Graph → Factor Pressure → Coach Trim.
// Must match Rust GraphNode/GraphEdge repr(C) layout.

struct GraphNodeGpu {
    uchar kind;
    uchar flags;
    ushort id;
    ushort parent_id;
    ushort edge_start;
    ushort edge_count;
    uchar _pad0[2];
    uchar label[32];
    float position[3];
    float embedding[16];
    float energy;
    float confidence;
    ulong timestamp_ns;
    uchar source_layer;
    uchar generation;
    uchar _pad1[6];
};

struct GraphEdgeGpu {
    ushort from;
    ushort to;
    uchar kind;
    uchar _pad;
    float weight;
    float tension;
};

// Dot product of two NODE_EMBED_DIM=16 embeddings
float dot16(const device float* a, const device float* b) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// params[0] = age_decay
// params[1] = support_weight
// params[2] = contradict_weight
kernel void factor_pressure(
    device GraphNodeGpu*  nodes     [[buffer(0)]],
    device GraphEdgeGpu*  edges     [[buffer(1)]],
    device float*         energies  [[buffer(2)]],
    const device float*   params    [[buffer(3)]],
    const device uint*    counts    [[buffer(4)]],  // [0]=node_count, [1]=edge_count
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= 64) return;

    uint node_count = counts[0];
    uint edge_count = counts[1];

    float age_decay      = params[0];
    float support_weight = params[1];
    float contra_weight  = params[2];

    uint batch = (node_count + 63) / 64;
    uint start = tid * batch;
    uint end   = start + batch;
    if (end > node_count) end = node_count;

    for (uint n = start; n < end; n++) {
        device GraphNodeGpu* node = &nodes[n];
        if (node->kind == 0) {
            energies[n] = 0.0f;
            continue;
        }

        float total_tension = 0.0f;

        uint e_start = node->edge_start;
        uint e_end   = e_start + node->edge_count;
        if (e_end > edge_count) e_end = edge_count;

        for (uint e = e_start; e < e_end; e++) {
            device GraphEdgeGpu* edge = &edges[e];
            if (edge->kind == 0) continue;

            uint other_id = (edge->from == n) ? edge->to : edge->from;
            if (other_id >= node_count) continue;

            device GraphNodeGpu* other = &nodes[other_id];
            if (other->kind == 0) continue;

            float sim = dot16(node->embedding, other->embedding);
            float edge_w = max(edge->weight, 0.01f);
            float t = 0.0f;

            switch (edge->kind) {
                case 2:  // Supports
                    t = -support_weight * edge_w * sim;
                    break;
                case 3:  // Contradicts
                    t = contra_weight * edge_w * (1.0f - sim);
                    break;
                case 1:  // Contains
                    t = -0.5f * edge_w * sim;
                    break;
                case 4:  // LeadsTo
                    t = -0.3f * edge_w * sim;
                    break;
                case 5:  // Causes
                    t = -0.4f * edge_w * sim;
                    break;
                case 6:  // Alternative
                    t = 0.2f * edge_w * (1.0f - sim);
                    break;
                default:
                    break;
            }

            edge->tension = t;
            total_tension += t;
        }

        float age_penalty = age_decay * float(node->generation);
        float energy = total_tension + age_penalty;
        energy = clamp(energy, -100.0f, 100.0f);

        node->energy = energy;
        energies[n] = energy;
    }
}
