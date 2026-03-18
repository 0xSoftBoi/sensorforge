#include <metal_stdlib>
using namespace metal;

// Must match Rust BeliefSlot repr(C, align(64)) layout.
// STATE_DIM = 64
struct BeliefSlot {
    float mean[64];
    float precision[64];
    float vfe;
    float prediction[64];
    float residual[64];
    float challenge_vfe;
    uint  confirm_streak;
    uchar compression;
    uchar layer;
    uchar _pad[2];
    ulong timestamp_ns;
    uint  cycle_us;
    uchar _pad2[4];
};

// params[0] = threshold
// params[1] = learning_rate (belief update)
// params[2] = layer_id
// params[3] = weight_decay

kernel void belief_update(
    device BeliefSlot*       me      [[buffer(0)]],
    const device BeliefSlot* below   [[buffer(1)]],
    const device float*      params  [[buffer(2)]],
    device float*            weights [[buffer(3)]],  // 64×64 generative model
    device float*            bias    [[buffer(4)]],  // 64 bias
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= 64) return;

    float threshold    = params[0];
    float lr           = params[1];
    float weight_decay = params[3];
    float lr_w         = lr * 0.1;  // weight learning rate (10x slower than belief)

    // ── Cache my mean in threadgroup memory for matrix multiply ──
    threadgroup float shared_mean[64];
    shared_mean[tid] = me->mean[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 1. PREDICT: prediction = W @ mean + bias ──
    // Each thread computes one element of the prediction vector.
    // This IS the generative model — W encodes what this layer
    // has learned about how its beliefs map to the layer below.
    float pred = bias[tid];
    for (uint j = 0; j < 64; j++) {
        pred += weights[tid * 64 + j] * shared_mean[j];
    }
    me->prediction[tid] = pred;

    // ── 2. RESIDUAL (prediction error) ──
    // The surprise: what the world actually looks like vs what we predicted.
    float residual = below->mean[tid] - pred;
    me->residual[tid] = residual;

    // ── 3. PRECISION-WEIGHTED SQUARED ERROR ──
    float prec = me->precision[tid];
    float clamped_res = clamp(residual, -10.0f, 10.0f);
    float weighted = clamped_res * min(prec, 10.0f) * clamped_res;

    // ── 4. VFE reduction (sum across all dimensions) ──
    threadgroup float shared_vfe[64];
    shared_vfe[tid] = weighted;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            shared_vfe[tid] += shared_vfe[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float total_vfe = shared_vfe[0];

    // ── 5. Thread 0: update scalar fields ──
    if (tid == 0) {
        me->challenge_vfe = total_vfe;
        me->vfe = total_vfe;

        if (total_vfe <= threshold) {
            me->confirm_streak += 1;
            if (me->compression < 255 && me->confirm_streak > 100) {
                me->compression += 1;
            }
        } else {
            me->confirm_streak = 0;
            if (me->compression > 0) {
                me->compression -= 1;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 6. BELIEF UPDATE: gradient descent on free energy ──
    // Move mean toward the sensory evidence, scaled by precision.
    // Clamp the effective precision to prevent explosive updates when
    // a sudden embedding injection hits a high-precision layer.
    if (total_vfe > threshold) {
        float eff_prec = min(prec, 1.0f);  // cap effective precision for update
        float delta = lr * eff_prec * residual;
        delta = clamp(delta, -0.1f, 0.1f);  // max step per cycle
        me->mean[tid] += delta;
        me->mean[tid] = clamp(me->mean[tid], -10.0f, 10.0f);
    }

    // ── 7. WEIGHT LEARNING: update generative model ──
    // W += lr_w * outer(precision * residual, mean)
    // Each thread updates its row of W (64 weights).
    // Uses the CACHED pre-update mean — the belief that generated the prediction.
    if (total_vfe > threshold) {
        float eff_prec_w = min(prec, 1.0f);
        float grad_scale = clamp(lr_w * eff_prec_w * residual, -0.01f, 0.01f);
        for (uint j = 0; j < 64; j++) {
            // Weight decay prevents unbounded growth
            weights[tid * 64 + j] *= (1.0f - weight_decay);
            // Gradient step
            weights[tid * 64 + j] += grad_scale * shared_mean[j];
            // Clamp to prevent explosion
            weights[tid * 64 + j] = clamp(weights[tid * 64 + j], -10.0f, 10.0f);
        }
        bias[tid] *= (1.0f - weight_decay);
        bias[tid] += clamp(lr_w * eff_prec_w * residual, -0.01f, 0.01f);
        bias[tid] = clamp(bias[tid], -10.0f, 10.0f);
    }

    // ── NaN guard: if VFE is NaN, reset this dimension ──
    if (isnan(me->mean[tid]) || isinf(me->mean[tid])) {
        me->mean[tid] = below->mean[tid];  // reset to sensory input
        me->precision[tid] = 0.01f;        // reset precision
    }

    // ── 8. PRECISION UPDATE: adapt confidence based on prediction accuracy ──
    float abs_res = abs(residual);
    if (abs_res < 0.01f) {
        me->precision[tid] = min(prec * 1.001f, 10.0f);  // lower cap: 10 not 100
    } else {
        me->precision[tid] = max(prec * 0.999f, 0.01f);
    }
}
