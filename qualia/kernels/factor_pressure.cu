// Factor pressure kernel — computes per-node energy from edge tensions.
// Part of the Thought Theater: Big Graph → Factor Pressure → Coach Trim.
//
// 64 threads, each handles ceil(node_count / 64) nodes.
// Must match Rust GraphNode/GraphEdge repr(C) layout.

struct GraphNodeGpu {
    unsigned char kind;
    unsigned char flags;
    unsigned short id;
    unsigned short parent_id;
    unsigned short edge_start;
    unsigned short edge_count;
    unsigned char _pad0[2];
    unsigned char label[32];
    float position[3];
    float embedding[16];
    float energy;
    float confidence;
    unsigned long long timestamp_ns;
    unsigned char source_layer;
    unsigned char generation;
    unsigned char _pad1[6];
};

struct GraphEdgeGpu {
    unsigned short from;
    unsigned short to;
    unsigned char kind;
    unsigned char _pad;
    float weight;
    float tension;
};

// Dot product of two NODE_EMBED_DIM=16 embeddings
__device__ float dot16(const float* a, const float* b) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// params[0] = age_decay        (energy penalty per generation)
// params[1] = support_weight   (scale for supporting edges)
// params[2] = contradict_weight (scale for contradicting edges)
extern "C" __global__ void factor_pressure(
    GraphNodeGpu*  nodes,
    GraphEdgeGpu*  edges,
    float*         energies,
    const float*   params,
    const unsigned int node_count,
    const unsigned int edge_count
)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64) return;

    float age_decay       = params[0];
    float support_weight  = params[1];
    float contra_weight   = params[2];

    // Each thread handles a batch of nodes
    unsigned int batch = (node_count + 63) / 64;
    unsigned int start = tid * batch;
    unsigned int end   = start + batch;
    if (end > node_count) end = node_count;

    for (unsigned int n = start; n < end; n++) {
        GraphNodeGpu* node = &nodes[n];
        if (node->kind == 0) {  // Empty
            energies[n] = 0.0f;
            continue;
        }

        float total_tension = 0.0f;

        // Iterate this node's edges
        unsigned int e_start = node->edge_start;
        unsigned int e_end   = e_start + node->edge_count;
        if (e_end > edge_count) e_end = edge_count;

        for (unsigned int e = e_start; e < e_end; e++) {
            GraphEdgeGpu* edge = &edges[e];
            if (edge->kind == 0) continue;  // Empty edge

            // Find the other node
            unsigned int other_id = (edge->from == n) ? edge->to : edge->from;
            if (other_id >= node_count) continue;

            GraphNodeGpu* other = &nodes[other_id];
            if (other->kind == 0) continue;

            float sim = dot16(node->embedding, other->embedding);
            float edge_w = fmaxf(edge->weight, 0.01f);
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

        // Age decay: older nodes accumulate more energy (pressure to trim)
        float age_penalty = age_decay * (float)node->generation;

        float energy = total_tension + age_penalty;
        // Clamp to prevent runaway
        energy = fminf(fmaxf(energy, -100.0f), 100.0f);

        node->energy = energy;
        energies[n] = energy;
    }
}
