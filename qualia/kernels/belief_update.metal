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
    // New fields (fit within 1088-byte alignment padding)
    float vfe_ema;
    float vfe_var;
    float compression_ratio;
};

// Sigmoid for precision-scaled learning
float sigmoid_fn(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// params[0] = threshold
// params[1] = learning_rate (belief update)
// params[2] = layer_id
// params[3] = weight_decay
// params[4] = has_above (1.0 if above layer exists, 0.0 otherwise)
// params[5] = precision_eta (precision learning rate)
// params[6] = surprise_threshold (z-score for sparse gating)
// params[7] = top_down_weight

kernel void belief_update(
    device BeliefSlot*       me      [[buffer(0)]],
    const device BeliefSlot* below   [[buffer(1)]],
    const device BeliefSlot* above   [[buffer(2)]],
    const device float*      params  [[buffer(3)]],
    device float*            weights [[buffer(4)]],
    device float*            bias    [[buffer(5)]],
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

    // 6. BELIEF UPDATE with sigmoid precision (Phase 1.1)
    if (total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid_fn(prec - 1.0f);
        float eff_lr = lr * (0.2f + 0.8f * prec_gate);

        float delta_up = eff_lr * residual;
        float delta_down = 0.0f;
        if (has_above > 0.5f) {
            float td_prec_gate = sigmoid_fn(prec - 1.0f);
            delta_down = eff_lr * td_weight * td_prec_gate * error_down;
        }

        float delta = clamp(delta_up + delta_down, -0.1f, 0.1f);
        me->mean[tid] += delta;
        me->mean[tid] = clamp(me->mean[tid], -10.0f, 10.0f);
    }

    // 7. WEIGHT LEARNING (surprise-gated)
    if (is_surprising && total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid_fn(prec - 1.0f);
        float eff_lr_w = lr_w * (0.2f + 0.8f * prec_gate);
        float grad_scale = clamp(eff_lr_w * residual, -0.01f, 0.01f);
        for (uint j = 0; j < 64; j++) {
            weights[tid * 64 + j] *= (1.0f - weight_decay);
            weights[tid * 64 + j] += grad_scale * shared_mean[j];
            weights[tid * 64 + j] = clamp(weights[tid * 64 + j], -10.0f, 10.0f);
        }
        bias[tid] *= (1.0f - weight_decay);
        bias[tid] += clamp(eff_lr_w * residual, -0.01f, 0.01f);
        bias[tid] = clamp(bias[tid], -10.0f, 10.0f);
    }

    // NaN guard
    if (isnan(me->mean[tid]) || isinf(me->mean[tid])) {
        me->mean[tid] = below->mean[tid];
        me->precision[tid] = 0.01f;
    }

    // 8. PRECISION UPDATE (Phase 1.1): free-energy gradient
    float res_sq = clamped_res * clamped_res;
    float inv_prec = 1.0f / max(prec, 0.01f);
    float prec_grad = inv_prec - res_sq;
    float new_prec = prec + prec_eta * prec_grad;
    me->precision[tid] = clamp(new_prec, 0.01f, 10.0f);
}
