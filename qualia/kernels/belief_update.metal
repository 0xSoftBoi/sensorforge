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
    float vfe_ema;
    float vfe_var;
    float compression_ratio;
};

// Sigmoid for precision-scaled learning
float sigmoid_fn(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Simple hash-based PRNG for Langevin noise
uint xorshift32(uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float rand_uniform_metal(uint state) {
    return float(state & 0x7FFFFFFF) / float(0x7FFFFFFF) * 2.0f - 1.0f;
}

// params[0] = threshold
// params[1] = learning_rate (belief update)
// params[2] = layer_id
// params[3] = weight_decay
// params[4] = has_above (1.0 if above layer exists, 0.0 otherwise)
// params[5] = precision_eta (precision learning rate)
// params[6] = surprise_threshold (z-score for sparse gating)
// params[7] = top_down_weight
// params[8] = adam_beta1 (default 0.9)
// params[9] = adam_beta2 (default 0.999)
// params[10] = adam_epsilon (default 1e-8)
// params[11] = langevin_sigma (exploration noise, 0=disabled)
// params[12] = grad_clip_norm (max gradient L2 norm, default 1.0)

kernel void belief_update(
    device BeliefSlot*       me      [[buffer(0)]],
    const device BeliefSlot* below   [[buffer(1)]],
    const device BeliefSlot* above   [[buffer(2)]],
    const device float*      params  [[buffer(3)]],
    device float*            weights [[buffer(4)]],
    device float*            bias    [[buffer(5)]],
    device float*            adam_m_w [[buffer(6)]],
    device float*            adam_v_w [[buffer(7)]],
    device float*            adam_m_b [[buffer(8)]],
    device float*            adam_v_b [[buffer(9)]],
    device atomic_uint*      adam_t   [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= 64) return;

    float threshold     = params[0];
    float lr            = params[1];
    float weight_decay  = params[3];
    float has_above     = params[4];
    float prec_eta      = params[5];
    float surprise_gate = params[6];
    float td_weight     = params[7];
    float beta1         = params[8];
    float beta2         = params[9];
    float adam_eps       = params[10];
    float langevin_sigma = params[11];
    float grad_clip_max = params[12];
    float lr_w          = lr * 0.1f;

    threadgroup float shared_mean[64];
    shared_mean[tid] = me->mean[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. PREDICT
    float pred = bias[tid];
    for (uint j = 0; j < 64; j++) {
        pred += weights[tid * 64 + j] * shared_mean[j];
    }
    me->prediction[tid] = pred;

    // 2. BOTTOM-UP RESIDUAL
    float residual = below->mean[tid] - pred;
    me->residual[tid] = residual;

    // 2b. TOP-DOWN ERROR (Phase 5.1)
    float error_down = 0.0f;
    if (has_above > 0.5f) {
        error_down = above->prediction[tid] - me->mean[tid];
    }

    // 3. PRECISION-WEIGHTED SQUARED ERROR
    float prec = me->precision[tid];
    float clamped_res = clamp(residual, -10.0f, 10.0f);
    float weighted = clamped_res * min(prec, 10.0f) * clamped_res;

    // 4. VFE reduction
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

    // 4b. Compression ratio entropy proxies
    threadgroup float shared_pred_abs[64];
    threadgroup float shared_obs_abs[64];
    shared_pred_abs[tid] = abs(pred);
    shared_obs_abs[tid] = abs(below->mean[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pred_abs[tid] += shared_pred_abs[tid + s];
            shared_obs_abs[tid] += shared_obs_abs[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 5. Thread 0: scalar fields
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

        // VFE EMA + variance (Phase 1.2)
        float ema_alpha = 0.02f;
        float old_ema = me->vfe_ema;
        me->vfe_ema = old_ema + ema_alpha * (total_vfe - old_ema);
        float vfe_dev = total_vfe - me->vfe_ema;
        me->vfe_var = me->vfe_var + ema_alpha * (vfe_dev * vfe_dev - me->vfe_var);

        // Compression ratio (Phase 1.4)
        float pred_energy = shared_pred_abs[0];
        float obs_e = shared_obs_abs[0];
        if (obs_e > 0.001f) {
            float ratio = pred_energy / obs_e;
            me->compression_ratio = me->compression_ratio * 0.95f + ratio * 0.05f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Surprise gating (Phase 5.2)
    float vfe_std = sqrt(max(me->vfe_var, 1e-8f));
    float z_score = (total_vfe - me->vfe_ema) / vfe_std;
    bool is_surprising = (z_score > surprise_gate) || (total_vfe > threshold);

    // 6. BELIEF UPDATE with Langevin exploration noise (Phase 6.3)
    if (total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid_fn(prec - 1.0f);
        float eff_lr = lr * (0.2f + 0.8f * prec_gate);

        float delta_up = eff_lr * residual;
        float delta_down = 0.0f;
        if (has_above > 0.5f) {
            float td_prec_gate = sigmoid_fn(prec - 1.0f);
            delta_down = eff_lr * td_weight * td_prec_gate * error_down;
        }

        float delta = delta_up + delta_down;

        // Phase 6.3: Langevin noise scaled by inverse sqrt(precision)
        if (langevin_sigma > 0.0f) {
            uint seed = tid * 2654435761u;
            seed ^= uint(me->timestamp_ns & 0xFFFFFFFF);
            seed = xorshift32(seed);
            float noise = rand_uniform_metal(seed) * langevin_sigma / max(sqrt(prec), 0.1f);
            delta += noise;
        }

        delta = clamp(delta, -0.1f, 0.1f);
        me->mean[tid] += delta;
        me->mean[tid] = clamp(me->mean[tid], -10.0f, 10.0f);
    }

    // 7. WEIGHT LEARNING with Adam optimizer (Phase 6.1)
    if (is_surprising && total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid_fn(prec - 1.0f);
        float eff_lr_w = lr_w * (0.2f + 0.8f * prec_gate);

        // Phase 6.2: Gradient norm clipping
        threadgroup float shared_mean_sq[64];
        shared_mean_sq[tid] = shared_mean[tid] * shared_mean[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 32; s > 0; s >>= 1) {
            if (tid < s) shared_mean_sq[tid] += shared_mean_sq[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float mean_norm_sq = shared_mean_sq[0];
        float grad_norm = abs(residual) * sqrt(max(mean_norm_sq, 1e-8f));
        float clip_scale = (grad_norm > grad_clip_max) ? (grad_clip_max / grad_norm) : 1.0f;

        float clipped_res = residual * clip_scale;

        // Increment Adam timestep (thread 0 only)
        if (tid == 0) {
            atomic_fetch_add_explicit(adam_t, 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float t_f = float(atomic_load_explicit(adam_t, memory_order_relaxed));
        float bc1 = 1.0f - pow(beta1, t_f);
        float bc2 = 1.0f - pow(beta2, t_f);

        // Adam update for weights
        for (uint j = 0; j < 64; j++) {
            uint idx = tid * 64 + j;
            float g = eff_lr_w * clipped_res * shared_mean[j];

            // AdamW decoupled weight decay
            weights[idx] *= (1.0f - weight_decay);

            adam_m_w[idx] = beta1 * adam_m_w[idx] + (1.0f - beta1) * g;
            adam_v_w[idx] = beta2 * adam_v_w[idx] + (1.0f - beta2) * g * g;

            float m_hat = adam_m_w[idx] / max(bc1, 1e-8f);
            float v_hat = adam_v_w[idx] / max(bc2, 1e-8f);

            weights[idx] += m_hat / (sqrt(v_hat) + adam_eps);
            weights[idx] = clamp(weights[idx], -10.0f, 10.0f);
        }

        // Adam update for bias
        float g_b = eff_lr_w * clipped_res;
        bias[tid] *= (1.0f - weight_decay);
        adam_m_b[tid] = beta1 * adam_m_b[tid] + (1.0f - beta1) * g_b;
        adam_v_b[tid] = beta2 * adam_v_b[tid] + (1.0f - beta2) * g_b * g_b;
        float m_hat_b = adam_m_b[tid] / max(bc1, 1e-8f);
        float v_hat_b = adam_v_b[tid] / max(bc2, 1e-8f);
        bias[tid] += m_hat_b / (sqrt(v_hat_b) + adam_eps);
        bias[tid] = clamp(bias[tid], -10.0f, 10.0f);
    }

    // NaN guard
    if (isnan(me->mean[tid]) || isinf(me->mean[tid])) {
        me->mean[tid] = below->mean[tid];
        me->precision[tid] = 0.01f;
    }

    for (uint j = 0; j < 64; j++) {
        if (isnan(weights[tid * 64 + j]) || isinf(weights[tid * 64 + j])) {
            weights[tid * 64 + j] = (tid == j) ? 1.0f : 0.0f;
        }
    }
    if (isnan(bias[tid]) || isinf(bias[tid])) {
        bias[tid] = 0.0f;
    }

    // 8. PRECISION UPDATE (Phase 1.1): free-energy gradient
    float res_sq = clamped_res * clamped_res;
    float inv_prec = 1.0f / max(prec, 0.01f);
    float prec_grad = inv_prec - res_sq;
    float new_prec = prec + prec_eta * prec_grad;
    me->precision[tid] = clamp(new_prec, 0.01f, 10.0f);
}
