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

float sigmoid_fn(float x) {
    return 1.0f / (1.0f + exp(-x));
}

uint xorshift32(uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float rand_uniform_metal(uint state) {
    return float(state & 0x7FFFFFFF) / float(0x7FFFFFFF) * 2.0f - 1.0f;
}

// Phase 7.2: Cosine annealing with warmup
float lr_schedule_metal(float t, float warmup_steps, float total_steps) {
    if (t < warmup_steps) {
        return t / max(warmup_steps, 1.0f);
    }
    float progress = (t - warmup_steps) / max(total_steps - warmup_steps, 1.0f);
    progress = min(progress, 1.0f);
    return 0.1f + 0.9f * 0.5f * (1.0f + cos(3.14159265f * progress));
}

// params[0]  = threshold
// params[1]  = learning_rate
// params[2]  = layer_id
// params[3]  = weight_decay
// params[4]  = has_above
// params[5]  = precision_eta
// params[6]  = surprise_threshold
// params[7]  = top_down_weight
// params[8]  = adam_beta1
// params[9]  = adam_beta2
// params[10] = adam_epsilon
// params[11] = langevin_sigma
// params[12] = grad_clip_norm
// params[13] = warmup_steps (Phase 7.2)
// params[14] = total_steps  (Phase 7.2)

kernel void belief_update(
    device BeliefSlot*       me       [[buffer(0)]],
    const device BeliefSlot* below    [[buffer(1)]],
    const device BeliefSlot* above    [[buffer(2)]],
    const device float*      params   [[buffer(3)]],
    device float*            weights  [[buffer(4)]],
    device float*            bias     [[buffer(5)]],
    device float*            gate_w   [[buffer(6)]],
    device float*            gate_b   [[buffer(7)]],
    device float*            adam_m_w [[buffer(8)]],
    device float*            adam_v_w [[buffer(9)]],
    device float*            adam_m_b [[buffer(10)]],
    device float*            adam_v_b [[buffer(11)]],
    device float*            adam_m_gw [[buffer(12)]],
    device float*            adam_v_gw [[buffer(13)]],
    device float*            adam_m_gb [[buffer(14)]],
    device float*            adam_v_gb [[buffer(15)]],
    device atomic_uint*      adam_t    [[buffer(16)]],
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
    float warmup_steps  = params[13];
    float total_steps   = params[14];

    // Phase 7.2: LR schedule
    float t_f = float(atomic_load_explicit(adam_t, memory_order_relaxed));
    float lr_mult = (total_steps > 0.0f) ? lr_schedule_metal(t_f, warmup_steps, total_steps) : 1.0f;
    float lr_w = lr * 0.1f * lr_mult;
    float eff_base_lr = lr * lr_mult;

    threadgroup float shared_mean[64];
    shared_mean[tid] = me->mean[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. GLU PREDICT (Phase 7.1): pred = (W@mean+b) ⊙ σ(Wg@mean+bg)
    float linear = bias[tid];
    float gate_val = gate_b[tid];
    for (uint j = 0; j < 64; j++) {
        linear += weights[tid * 64 + j] * shared_mean[j];
        gate_val += gate_w[tid * 64 + j] * shared_mean[j];
    }
    float gate = sigmoid_fn(gate_val);
    float pred = linear * gate;
    me->prediction[tid] = pred;

    // 2. BOTTOM-UP RESIDUAL
    float residual = below->mean[tid] - pred;
    me->residual[tid] = residual;

    // 2b. TOP-DOWN ERROR
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
        if (tid < s) shared_vfe[tid] += shared_vfe[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_vfe = shared_vfe[0];

    // 4b. Compression ratio
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
            if (me->compression < 255 && me->confirm_streak > 100)
                me->compression += 1;
        } else {
            me->confirm_streak = 0;
            if (me->compression > 0) me->compression -= 1;
        }
        float ema_alpha = 0.02f;
        float old_ema = me->vfe_ema;
        me->vfe_ema = old_ema + ema_alpha * (total_vfe - old_ema);
        float vfe_dev = total_vfe - me->vfe_ema;
        me->vfe_var = me->vfe_var + ema_alpha * (vfe_dev * vfe_dev - me->vfe_var);
        float pred_energy = shared_pred_abs[0];
        float obs_e = shared_obs_abs[0];
        if (obs_e > 0.001f) {
            float ratio = pred_energy / obs_e;
            me->compression_ratio = me->compression_ratio * 0.95f + ratio * 0.05f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Surprise gating
    float vfe_std = sqrt(max(me->vfe_var, 1e-8f));
    float z_score = (total_vfe - me->vfe_ema) / vfe_std;
    bool is_surprising = (z_score > surprise_gate) || (total_vfe > threshold);

    // 6. BELIEF UPDATE with Langevin noise
    if (total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid_fn(prec - 1.0f);
        float eff_lr = eff_base_lr * (0.2f + 0.8f * prec_gate);
        float delta_up = eff_lr * residual;
        float delta_down = 0.0f;
        if (has_above > 0.5f) {
            float td_prec_gate = sigmoid_fn(prec - 1.0f);
            delta_down = eff_lr * td_weight * td_prec_gate * error_down;
        }
        float delta = delta_up + delta_down;
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

    // 7. WEIGHT + GATE LEARNING with Adam (Phase 7.1 GLU)
    if (is_surprising && total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid_fn(prec - 1.0f);
        float eff_lr_w = lr_w * (0.2f + 0.8f * prec_gate);

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

        if (tid == 0) atomic_fetch_add_explicit(adam_t, 1u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float t_now = float(atomic_load_explicit(adam_t, memory_order_relaxed));
        float bc1 = 1.0f - pow(beta1, t_now);
        float bc2 = 1.0f - pow(beta2, t_now);

        // GLU gradient: d_linear = d_pred * gate, d_gate_pre = d_pred * linear * gate*(1-gate)
        float d_pred = clipped_res;
        float d_linear = d_pred * gate;
        float d_gate_pre = d_pred * linear * gate * (1.0f - gate);

        for (uint j = 0; j < 64; j++) {
            uint idx = tid * 64 + j;

            // Main weights
            float g_w = eff_lr_w * d_linear * shared_mean[j];
            weights[idx] *= (1.0f - weight_decay);
            adam_m_w[idx] = beta1 * adam_m_w[idx] + (1.0f - beta1) * g_w;
            adam_v_w[idx] = beta2 * adam_v_w[idx] + (1.0f - beta2) * g_w * g_w;
            weights[idx] += (adam_m_w[idx] / max(bc1, 1e-8f)) / (sqrt(adam_v_w[idx] / max(bc2, 1e-8f)) + adam_eps);
            weights[idx] = clamp(weights[idx], -10.0f, 10.0f);

            // Gate weights
            float g_gw = eff_lr_w * d_gate_pre * shared_mean[j];
            gate_w[idx] *= (1.0f - weight_decay);
            adam_m_gw[idx] = beta1 * adam_m_gw[idx] + (1.0f - beta1) * g_gw;
            adam_v_gw[idx] = beta2 * adam_v_gw[idx] + (1.0f - beta2) * g_gw * g_gw;
            gate_w[idx] += (adam_m_gw[idx] / max(bc1, 1e-8f)) / (sqrt(adam_v_gw[idx] / max(bc2, 1e-8f)) + adam_eps);
            gate_w[idx] = clamp(gate_w[idx], -10.0f, 10.0f);
        }

        // Biases
        float g_b = eff_lr_w * d_linear;
        bias[tid] *= (1.0f - weight_decay);
        adam_m_b[tid] = beta1 * adam_m_b[tid] + (1.0f - beta1) * g_b;
        adam_v_b[tid] = beta2 * adam_v_b[tid] + (1.0f - beta2) * g_b * g_b;
        bias[tid] += (adam_m_b[tid] / max(bc1, 1e-8f)) / (sqrt(adam_v_b[tid] / max(bc2, 1e-8f)) + adam_eps);
        bias[tid] = clamp(bias[tid], -10.0f, 10.0f);

        float g_gb = eff_lr_w * d_gate_pre;
        gate_b[tid] *= (1.0f - weight_decay);
        adam_m_gb[tid] = beta1 * adam_m_gb[tid] + (1.0f - beta1) * g_gb;
        adam_v_gb[tid] = beta2 * adam_v_gb[tid] + (1.0f - beta2) * g_gb * g_gb;
        gate_b[tid] += (adam_m_gb[tid] / max(bc1, 1e-8f)) / (sqrt(adam_v_gb[tid] / max(bc2, 1e-8f)) + adam_eps);
        gate_b[tid] = clamp(gate_b[tid], -10.0f, 10.0f);
    }

    // NaN guard
    if (isnan(me->mean[tid]) || isinf(me->mean[tid])) {
        me->mean[tid] = below->mean[tid];
        me->precision[tid] = 0.01f;
    }
    for (uint j = 0; j < 64; j++) {
        uint idx = tid * 64 + j;
        if (isnan(weights[idx]) || isinf(weights[idx]))
            weights[idx] = (tid == j) ? 1.0f : 0.0f;
        if (isnan(gate_w[idx]) || isinf(gate_w[idx]))
            gate_w[idx] = 0.0f;
    }
    if (isnan(bias[tid]) || isinf(bias[tid])) bias[tid] = 0.0f;
    if (isnan(gate_b[tid]) || isinf(gate_b[tid])) gate_b[tid] = 0.0f;

    // 8. PRECISION UPDATE
    float res_sq = clamped_res * clamped_res;
    float inv_prec = 1.0f / max(prec, 0.01f);
    float prec_grad = inv_prec - res_sq;
    float new_prec = prec + prec_eta * prec_grad;
    me->precision[tid] = clamp(new_prec, 0.01f, 10.0f);
}
