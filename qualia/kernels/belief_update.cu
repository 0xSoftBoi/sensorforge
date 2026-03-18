// CUDA belief update kernel — predictive coding with active inference.
// Must match Rust BeliefSlot repr(C, align(64)) layout.
// STATE_DIM = 64
//
// Phases implemented:
//   1.1 — Adaptive learning rates via sigmoid(precision)
//   1.2 — VFE surprise detection (EMA + variance tracking)
//   1.4 — Information-theoretic compression ratio
//   5.1 — Top-down prediction errors from layer above
//   5.2 — Surprise-gated sparse updates

struct BeliefSlot {
    float mean[64];
    float precision[64];
    float vfe;
    float prediction[64];
    float residual[64];
    float challenge_vfe;
    unsigned int  confirm_streak;
    unsigned char compression;
    unsigned char layer;
    unsigned char _pad[2];
    unsigned long long timestamp_ns;
    unsigned int  cycle_us;
    unsigned char _pad2[4];
    // New fields (fit within 1088-byte alignment padding)
    float vfe_ema;
    float vfe_var;
    float compression_ratio;
};

// Sigmoid activation for precision-scaled learning
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// params[0] = threshold
// params[1] = learning_rate (belief update)
// params[2] = layer_id
// params[3] = weight_decay
// params[4] = has_above (1.0 if above layer exists, 0.0 otherwise)
// params[5] = precision_eta (precision learning rate, default 0.01)
// params[6] = surprise_threshold (z-score threshold for sparse gating, default 2.0)
// params[7] = top_down_weight (weight for top-down errors, default 0.3)

extern "C" __global__ void belief_update(
    BeliefSlot*       me,
    const BeliefSlot* below,
    const BeliefSlot* above,    // layer above (NULL-safe: check params[4])
    const float*      params,
    float*            weights,  // 64x64 generative model
    float*            bias      // 64 bias
)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64) return;

    float threshold     = params[0];
    float lr            = params[1];
    float weight_decay  = params[3];
    float has_above     = params[4];
    float prec_eta      = params[5];
    float surprise_gate = params[6];
    float td_weight     = params[7];
    float lr_w          = lr * 0.1f;  // weight learning rate (10x slower than belief)

    // -- Cache my mean in shared memory for matrix multiply --
    __shared__ float shared_mean[64];
    shared_mean[tid] = me->mean[tid];
    __syncthreads();

    // -- 1. PREDICT: prediction = W @ mean + bias --
    float pred = bias[tid];
    for (unsigned int j = 0; j < 64; j++) {
        pred += weights[tid * 64 + j] * shared_mean[j];
    }
    me->prediction[tid] = pred;

    // -- 2. BOTTOM-UP RESIDUAL (prediction error from layer below) --
    float residual = below->mean[tid] - pred;
    me->residual[tid] = residual;

    // -- 2b. TOP-DOWN ERROR (Phase 5.1): prediction from layer above --
    // The above layer's prediction field contains what it predicts we should be.
    float error_down = 0.0f;
    if (has_above > 0.5f) {
        error_down = above->prediction[tid] - me->mean[tid];
    }

    // -- 3. PRECISION-WEIGHTED SQUARED ERROR (bottom-up) --
    float prec = me->precision[tid];
    float clamped_res = fminf(fmaxf(residual, -10.0f), 10.0f);
    float weighted = clamped_res * fminf(prec, 10.0f) * clamped_res;

    // -- 4. VFE reduction (sum across all dimensions) --
    __shared__ float shared_vfe[64];
    shared_vfe[tid] = weighted;
    __syncthreads();

    for (unsigned int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            shared_vfe[tid] += shared_vfe[tid + s];
        }
        __syncthreads();
    }

    float total_vfe = shared_vfe[0];

    // -- 4b. Compute prediction entropy for compression ratio (Phase 1.4) --
    // Entropy proxy: mean absolute prediction across dims (higher = more structure)
    __shared__ float shared_pred_abs[64];
    shared_pred_abs[tid] = fabsf(pred);
    __syncthreads();

    float obs_energy = fabsf(below->mean[tid]);
    __shared__ float shared_obs_abs[64];
    shared_obs_abs[tid] = obs_energy;
    __syncthreads();

    // Reduce for entropy proxies (thread 0 only needs totals)
    for (unsigned int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pred_abs[tid] += shared_pred_abs[tid + s];
            shared_obs_abs[tid] += shared_obs_abs[tid + s];
        }
        __syncthreads();
    }

    // -- 5. Thread 0: update scalar fields --
    if (tid == 0) {
        me->challenge_vfe = total_vfe;
        me->vfe = total_vfe;

        // Legacy compression counter (kept for backward compatibility)
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

        // Phase 1.2: VFE EMA and variance tracking for z-score surprise
        float ema_alpha = 0.02f;  // slow tracking (~50 sample window)
        float old_ema = me->vfe_ema;
        me->vfe_ema = old_ema + ema_alpha * (total_vfe - old_ema);
        float vfe_dev = total_vfe - me->vfe_ema;
        me->vfe_var = me->vfe_var + ema_alpha * (vfe_dev * vfe_dev - me->vfe_var);

        // Phase 1.4: Information-theoretic compression ratio
        float pred_energy = shared_pred_abs[0];
        float obs_e = shared_obs_abs[0];
        if (obs_e > 0.001f) {
            float ratio = pred_energy / obs_e;
            // EMA smooth the compression ratio
            me->compression_ratio = me->compression_ratio * 0.95f + ratio * 0.05f;
        }
    }

    __syncthreads();

    // -- Phase 5.2: Surprise-gated sparse updates --
    // Compute z-score from VFE EMA/variance. Skip weight updates when not surprised.
    float vfe_std = sqrtf(fmaxf(me->vfe_var, 1e-8f));
    float z_score = (total_vfe - me->vfe_ema) / vfe_std;
    bool is_surprising = (z_score > surprise_gate) || (total_vfe > threshold);

    // -- 6. BELIEF UPDATE: gradient descent on free energy --
    // Phase 1.1: Adaptive learning rate via sigmoid(precision)
    // High precision → sigmoid ~1 → slower learning (confident, small corrections)
    // Low precision → sigmoid ~0.5 → faster learning (uncertain, big corrections)
    // Inverted: we want LESS update when precision is HIGH (predictions are good)
    if (total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid(prec - 1.0f);  // high prec → small gate
        float eff_lr = lr * (0.2f + 0.8f * prec_gate);   // floor at 20% of base lr

        // Bottom-up update (precision-weighted)
        float delta_up = eff_lr * residual;

        // Top-down update (Phase 5.1)
        float delta_down = 0.0f;
        if (has_above > 0.5f) {
            float td_prec_gate = sigmoid(prec - 1.0f);  // high prec → trust top-down more
            delta_down = eff_lr * td_weight * td_prec_gate * error_down;
        }

        float delta = delta_up + delta_down;
        delta = fminf(fmaxf(delta, -0.1f), 0.1f);  // max step per cycle
        me->mean[tid] += delta;
        me->mean[tid] = fminf(fmaxf(me->mean[tid], -10.0f), 10.0f);
    }

    // -- 7. WEIGHT LEARNING: update generative model --
    // Phase 5.2: Only update weights when there's genuine surprise (z-score gating)
    if (is_surprising && total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid(prec - 1.0f);
        float eff_lr_w = lr_w * (0.2f + 0.8f * prec_gate);
        float grad_scale = fminf(fmaxf(eff_lr_w * residual, -0.01f), 0.01f);
        for (unsigned int j = 0; j < 64; j++) {
            weights[tid * 64 + j] *= (1.0f - weight_decay);
            weights[tid * 64 + j] += grad_scale * shared_mean[j];
            weights[tid * 64 + j] = fminf(fmaxf(weights[tid * 64 + j], -10.0f), 10.0f);
        }
        bias[tid] *= (1.0f - weight_decay);
        bias[tid] += fminf(fmaxf(eff_lr_w * residual, -0.01f), 0.01f);
        bias[tid] = fminf(fmaxf(bias[tid], -10.0f), 10.0f);
    }

    // -- NaN guard: if mean is NaN/Inf, reset to sensory input --
    if (isnan(me->mean[tid]) || isinf(me->mean[tid])) {
        me->mean[tid] = below->mean[tid];
        me->precision[tid] = 0.01f;
    }

    // -- Also reset weights if they went NaN --
    for (unsigned int j = 0; j < 64; j++) {
        if (isnan(weights[tid * 64 + j]) || isinf(weights[tid * 64 + j])) {
            weights[tid * 64 + j] = (tid == j) ? 1.0f : 0.0f;  // reset to identity
        }
    }
    if (isnan(bias[tid]) || isinf(bias[tid])) {
        bias[tid] = 0.0f;
    }

    // -- 8. PRECISION UPDATE (Phase 1.1): proper free-energy gradient --
    // Free energy w.r.t. precision: dF/dπ = 1/π - residual²
    // When residual² < 1/π: precision should increase (predictions are good)
    // When residual² > 1/π: precision should decrease (predictions are bad)
    float res_sq = clamped_res * clamped_res;
    float inv_prec = 1.0f / fmaxf(prec, 0.01f);
    float prec_grad = inv_prec - res_sq;

    // EMA-smoothed precision update to prevent oscillation
    float new_prec = prec + prec_eta * prec_grad;
    me->precision[tid] = fminf(fmaxf(new_prec, 0.01f), 10.0f);
}
