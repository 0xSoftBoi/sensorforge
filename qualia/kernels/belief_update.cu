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
//   6.1 — Adam optimizer for weight learning (Kingma & Ba 2015)
//   6.2 — Gradient norm clipping (Pascanu et al. 2013)
//   6.3 — Langevin exploration noise in belief space (Welling & Teh 2011)

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
    float vfe_ema;
    float vfe_var;
    float compression_ratio;
};

// Sigmoid activation for precision-scaled learning
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Simple xorshift64 PRNG for Langevin noise (per-thread deterministic)
__device__ unsigned long long xorshift64(unsigned long long state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// Convert u64 to float in [-1, 1]
__device__ float rand_uniform(unsigned long long state) {
    return (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

// params[0] = threshold
// params[1] = learning_rate (belief update)
// params[2] = layer_id
// params[3] = weight_decay
// params[4] = has_above (1.0 if above layer exists, 0.0 otherwise)
// params[5] = precision_eta (precision learning rate, default 0.01)
// params[6] = surprise_threshold (z-score threshold for sparse gating, default 2.0)
// params[7] = top_down_weight (weight for top-down errors, default 0.3)
// params[8] = adam_beta1 (default 0.9)
// params[9] = adam_beta2 (default 0.999)
// params[10] = adam_epsilon (default 1e-8)
// params[11] = langevin_sigma (exploration noise scale, default 0.001, 0=disabled)
// params[12] = grad_clip_norm (max gradient L2 norm, default 1.0)

extern "C" __global__ void belief_update(
    BeliefSlot*       me,
    const BeliefSlot* below,
    const BeliefSlot* above,
    const float*      params,
    float*            weights,
    float*            bias,
    float*            adam_m_w,   // first moment for weights  [4096]
    float*            adam_v_w,   // second moment for weights [4096]
    float*            adam_m_b,   // first moment for bias     [64]
    float*            adam_v_b,   // second moment for bias    [64]
    unsigned long long* adam_t    // timestep counter          [1]
)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64) return;

    float threshold     = params[0];
    float lr            = params[1];
    float layer_id      = params[2];
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
    __shared__ float shared_pred_abs[64];
    shared_pred_abs[tid] = fabsf(pred);
    __syncthreads();

    float obs_energy = fabsf(below->mean[tid]);
    __shared__ float shared_obs_abs[64];
    shared_obs_abs[tid] = obs_energy;
    __syncthreads();

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
        float ema_alpha = 0.02f;
        float old_ema = me->vfe_ema;
        me->vfe_ema = old_ema + ema_alpha * (total_vfe - old_ema);
        float vfe_dev = total_vfe - me->vfe_ema;
        me->vfe_var = me->vfe_var + ema_alpha * (vfe_dev * vfe_dev - me->vfe_var);

        // Phase 1.4: Information-theoretic compression ratio
        float pred_energy = shared_pred_abs[0];
        float obs_e = shared_obs_abs[0];
        if (obs_e > 0.001f) {
            float ratio = pred_energy / obs_e;
            me->compression_ratio = me->compression_ratio * 0.95f + ratio * 0.05f;
        }
    }

    __syncthreads();

    // -- Phase 5.2: Surprise-gated sparse updates --
    float vfe_std = sqrtf(fmaxf(me->vfe_var, 1e-8f));
    float z_score = (total_vfe - me->vfe_ema) / vfe_std;
    bool is_surprising = (z_score > surprise_gate) || (total_vfe > threshold);

    // -- 6. BELIEF UPDATE with top-down + Langevin noise (Phase 6.3) --
    if (total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid(prec - 1.0f);
        float eff_lr = lr * (0.2f + 0.8f * prec_gate);

        float delta_up = eff_lr * residual;

        float delta_down = 0.0f;
        if (has_above > 0.5f) {
            float td_prec_gate = sigmoid(prec - 1.0f);
            delta_down = eff_lr * td_weight * td_prec_gate * error_down;
        }

        float delta = delta_up + delta_down;

        // Phase 6.3: Langevin exploration noise — inject Gaussian noise
        // scaled by inverse precision (uncertain dims explore more)
        if (langevin_sigma > 0.0f) {
            unsigned long long seed = (unsigned long long)(tid * 2654435761u)
                ^ me->timestamp_ns ^ (unsigned long long)(total_vfe * 1e6f);
            seed = xorshift64(seed);
            float noise = rand_uniform(seed) * langevin_sigma / fmaxf(sqrtf(prec), 0.1f);
            delta += noise;
        }

        delta = fminf(fmaxf(delta, -0.1f), 0.1f);
        me->mean[tid] += delta;
        me->mean[tid] = fminf(fmaxf(me->mean[tid], -10.0f), 10.0f);
    }

    // -- 7. WEIGHT LEARNING with Adam optimizer (Phase 6.1) --
    if (is_surprising && total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid(prec - 1.0f);
        float eff_lr_w = lr_w * (0.2f + 0.8f * prec_gate);

        // Phase 6.2: Compute per-thread gradient norm for clipping
        // grad_w[tid][j] = residual * shared_mean[j] (outer product)
        // grad_norm² = residual² * ||mean||²
        __shared__ float shared_mean_sq[64];
        shared_mean_sq[tid] = shared_mean[tid] * shared_mean[tid];
        __syncthreads();

        for (unsigned int s = 32; s > 0; s >>= 1) {
            if (tid < s) shared_mean_sq[tid] += shared_mean_sq[tid + s];
            __syncthreads();
        }
        float mean_norm_sq = shared_mean_sq[0];
        float grad_norm = fabsf(residual) * sqrtf(fmaxf(mean_norm_sq, 1e-8f));
        float clip_scale = (grad_norm > grad_clip_max)
            ? (grad_clip_max / grad_norm) : 1.0f;

        float clipped_res = residual * clip_scale;

        // Increment Adam timestep (thread 0 only, atomic)
        if (tid == 0) {
            atomicAdd(adam_t, 1ULL);
        }
        __syncthreads();

        float t_f = (float)(*adam_t);
        float bc1 = 1.0f - powf(beta1, t_f);  // bias correction for first moment
        float bc2 = 1.0f - powf(beta2, t_f);  // bias correction for second moment

        // Adam update for weights: W[tid][j]
        for (unsigned int j = 0; j < 64; j++) {
            unsigned int idx = tid * 64 + j;
            float g = eff_lr_w * clipped_res * shared_mean[j];

            // Include weight decay as decoupled (AdamW style)
            weights[idx] *= (1.0f - weight_decay);

            // Update moments
            adam_m_w[idx] = beta1 * adam_m_w[idx] + (1.0f - beta1) * g;
            adam_v_w[idx] = beta2 * adam_v_w[idx] + (1.0f - beta2) * g * g;

            // Bias-corrected estimates
            float m_hat = adam_m_w[idx] / fmaxf(bc1, 1e-8f);
            float v_hat = adam_v_w[idx] / fmaxf(bc2, 1e-8f);

            // Update weight
            float step = m_hat / (sqrtf(v_hat) + adam_eps);
            weights[idx] += step;
            weights[idx] = fminf(fmaxf(weights[idx], -10.0f), 10.0f);
        }

        // Adam update for bias
        float g_b = eff_lr_w * clipped_res;
        bias[tid] *= (1.0f - weight_decay);
        adam_m_b[tid] = beta1 * adam_m_b[tid] + (1.0f - beta1) * g_b;
        adam_v_b[tid] = beta2 * adam_v_b[tid] + (1.0f - beta2) * g_b * g_b;
        float m_hat_b = adam_m_b[tid] / fmaxf(bc1, 1e-8f);
        float v_hat_b = adam_v_b[tid] / fmaxf(bc2, 1e-8f);
        bias[tid] += m_hat_b / (sqrtf(v_hat_b) + adam_eps);
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
            weights[tid * 64 + j] = (tid == j) ? 1.0f : 0.0f;
        }
    }
    if (isnan(bias[tid]) || isinf(bias[tid])) {
        bias[tid] = 0.0f;
    }

    // -- 8. PRECISION UPDATE (Phase 1.1): proper free-energy gradient --
    float res_sq = clamped_res * clamped_res;
    float inv_prec = 1.0f / fmaxf(prec, 0.01f);
    float prec_grad = inv_prec - res_sq;
    float new_prec = prec + prec_eta * prec_grad;
    me->precision[tid] = fminf(fmaxf(new_prec, 0.01f), 10.0f);
}
