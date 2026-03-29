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
//   7.1 — GLU nonlinear prediction (Dauphin et al. 2017)
//   7.2 — Learning rate warmup + cosine annealing (Loshchilov & Hutter 2017)

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

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ unsigned long long xorshift64(unsigned long long state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

__device__ float rand_uniform(unsigned long long state) {
    return (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF * 2.0f - 1.0f;
}

// Phase 7.2: Cosine annealing with warmup
// Returns learning rate multiplier in [0, 1]
__device__ float lr_schedule(float t, float warmup_steps, float total_steps) {
    if (t < warmup_steps) {
        // Linear warmup
        return t / fmaxf(warmup_steps, 1.0f);
    }
    // Cosine annealing to 10% of peak
    float progress = (t - warmup_steps) / fmaxf(total_steps - warmup_steps, 1.0f);
    progress = fminf(progress, 1.0f);
    return 0.1f + 0.9f * 0.5f * (1.0f + cosf(3.14159265f * progress));
}

// params[0]  = threshold
// params[1]  = learning_rate (belief update)
// params[2]  = layer_id
// params[3]  = weight_decay
// params[4]  = has_above (1.0 if above layer exists, 0.0 otherwise)
// params[5]  = precision_eta
// params[6]  = surprise_threshold (z-score)
// params[7]  = top_down_weight
// params[8]  = adam_beta1
// params[9]  = adam_beta2
// params[10] = adam_epsilon
// params[11] = langevin_sigma
// params[12] = grad_clip_norm
// params[13] = warmup_steps (Phase 7.2, default 100)
// params[14] = total_steps (Phase 7.2, default 10000, 0=no schedule)

extern "C" __global__ void belief_update(
    BeliefSlot*       me,
    const BeliefSlot* below,
    const BeliefSlot* above,
    const float*      params,
    float*            weights,
    float*            bias,
    float*            gate_w,      // GLU gate weights  [4096] (Phase 7.1)
    float*            gate_b,      // GLU gate bias     [64]   (Phase 7.1)
    float*            adam_m_w,
    float*            adam_v_w,
    float*            adam_m_b,
    float*            adam_v_b,
    float*            adam_m_gw,   // Adam moments for gate weights (Phase 7.1)
    float*            adam_v_gw,
    float*            adam_m_gb,
    float*            adam_v_gb,
    unsigned long long* adam_t
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
    float beta1         = params[8];
    float beta2         = params[9];
    float adam_eps       = params[10];
    float langevin_sigma = params[11];
    float grad_clip_max = params[12];
    float warmup_steps  = params[13];
    float total_steps   = params[14];

    // Phase 7.2: Apply LR schedule
    float t_f = (float)(*adam_t);
    float lr_mult = (total_steps > 0.0f) ? lr_schedule(t_f, warmup_steps, total_steps) : 1.0f;
    float lr_w = lr * 0.1f * lr_mult;
    float eff_base_lr = lr * lr_mult;

    // -- Cache mean in shared memory --
    __shared__ float shared_mean[64];
    shared_mean[tid] = me->mean[tid];
    __syncthreads();

    // -- 1. GLU PREDICT (Phase 7.1): pred = (W @ mean + b) ⊙ σ(Wg @ mean + bg) --
    // Linear path
    float linear = bias[tid];
    for (unsigned int j = 0; j < 64; j++) {
        linear += weights[tid * 64 + j] * shared_mean[j];
    }
    // Gate path
    float gate_val = gate_b[tid];
    for (unsigned int j = 0; j < 64; j++) {
        gate_val += gate_w[tid * 64 + j] * shared_mean[j];
    }
    float gate = sigmoid(gate_val);
    // GLU output: element-wise product of linear and gate
    float pred = linear * gate;
    me->prediction[tid] = pred;

    // -- 2. BOTTOM-UP RESIDUAL --
    float residual = below->mean[tid] - pred;
    me->residual[tid] = residual;

    // -- 2b. TOP-DOWN ERROR (Phase 5.1) --
    float error_down = 0.0f;
    if (has_above > 0.5f) {
        error_down = above->prediction[tid] - me->mean[tid];
    }

    // -- 3. PRECISION-WEIGHTED SQUARED ERROR --
    float prec = me->precision[tid];
    float clamped_res = fminf(fmaxf(residual, -10.0f), 10.0f);
    float weighted = clamped_res * fminf(prec, 10.0f) * clamped_res;

    // -- 4. VFE reduction --
    __shared__ float shared_vfe[64];
    shared_vfe[tid] = weighted;
    __syncthreads();

    for (unsigned int s = 32; s > 0; s >>= 1) {
        if (tid < s) shared_vfe[tid] += shared_vfe[tid + s];
        __syncthreads();
    }
    float total_vfe = shared_vfe[0];

    // -- 4b. Compression ratio --
    __shared__ float shared_pred_abs[64];
    __shared__ float shared_obs_abs[64];
    shared_pred_abs[tid] = fabsf(pred);
    shared_obs_abs[tid] = fabsf(below->mean[tid]);
    __syncthreads();

    for (unsigned int s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pred_abs[tid] += shared_pred_abs[tid + s];
            shared_obs_abs[tid] += shared_obs_abs[tid + s];
        }
        __syncthreads();
    }

    // -- 5. Thread 0: scalar fields --
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
    __syncthreads();

    // -- Surprise gating --
    float vfe_std = sqrtf(fmaxf(me->vfe_var, 1e-8f));
    float z_score = (total_vfe - me->vfe_ema) / vfe_std;
    bool is_surprising = (z_score > surprise_gate) || (total_vfe > threshold);

    // -- 6. BELIEF UPDATE with Langevin noise --
    if (total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid(prec - 1.0f);
        float eff_lr = eff_base_lr * (0.2f + 0.8f * prec_gate);

        float delta_up = eff_lr * residual;
        float delta_down = 0.0f;
        if (has_above > 0.5f) {
            float td_prec_gate = sigmoid(prec - 1.0f);
            delta_down = eff_lr * td_weight * td_prec_gate * error_down;
        }

        float delta = delta_up + delta_down;

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

    // -- 7. WEIGHT + GATE LEARNING with Adam (Phase 7.1 GLU) --
    if (is_surprising && total_vfe > threshold) {
        float prec_gate = 1.0f - sigmoid(prec - 1.0f);
        float eff_lr_w = lr_w * (0.2f + 0.8f * prec_gate);

        // Gradient norm clipping
        __shared__ float shared_mean_sq[64];
        shared_mean_sq[tid] = shared_mean[tid] * shared_mean[tid];
        __syncthreads();
        for (unsigned int s = 32; s > 0; s >>= 1) {
            if (tid < s) shared_mean_sq[tid] += shared_mean_sq[tid + s];
            __syncthreads();
        }
        float mean_norm_sq = shared_mean_sq[0];
        float grad_norm = fabsf(residual) * sqrtf(fmaxf(mean_norm_sq, 1e-8f));
        float clip_scale = (grad_norm > grad_clip_max) ? (grad_clip_max / grad_norm) : 1.0f;
        float clipped_res = residual * clip_scale;

        // Increment Adam timestep
        if (tid == 0) atomicAdd(adam_t, 1ULL);
        __syncthreads();

        float t_now = (float)(*adam_t);
        float bc1 = 1.0f - powf(beta1, t_now);
        float bc2 = 1.0f - powf(beta2, t_now);

        // GLU gradient decomposition (Phase 7.1):
        // pred = linear * gate, where linear = W@x+b, gate = σ(Wg@x+bg)
        // d_loss/d_W = d_loss/d_pred * gate * x^T
        // d_loss/d_Wg = d_loss/d_pred * linear * gate*(1-gate) * x^T
        float d_pred = clipped_res;  // d_loss/d_pred for MSE-like loss
        float d_linear = d_pred * gate;
        float d_gate_pre = d_pred * linear * gate * (1.0f - gate);  // sigmoid derivative

        // Adam update for main weights (W)
        for (unsigned int j = 0; j < 64; j++) {
            unsigned int idx = tid * 64 + j;

            // Main weights
            float g_w = eff_lr_w * d_linear * shared_mean[j];
            weights[idx] *= (1.0f - weight_decay);
            adam_m_w[idx] = beta1 * adam_m_w[idx] + (1.0f - beta1) * g_w;
            adam_v_w[idx] = beta2 * adam_v_w[idx] + (1.0f - beta2) * g_w * g_w;
            float mh_w = adam_m_w[idx] / fmaxf(bc1, 1e-8f);
            float vh_w = adam_v_w[idx] / fmaxf(bc2, 1e-8f);
            weights[idx] += mh_w / (sqrtf(vh_w) + adam_eps);
            weights[idx] = fminf(fmaxf(weights[idx], -10.0f), 10.0f);

            // Gate weights (Phase 7.1)
            float g_gw = eff_lr_w * d_gate_pre * shared_mean[j];
            gate_w[idx] *= (1.0f - weight_decay);
            adam_m_gw[idx] = beta1 * adam_m_gw[idx] + (1.0f - beta1) * g_gw;
            adam_v_gw[idx] = beta2 * adam_v_gw[idx] + (1.0f - beta2) * g_gw * g_gw;
            float mh_gw = adam_m_gw[idx] / fmaxf(bc1, 1e-8f);
            float vh_gw = adam_v_gw[idx] / fmaxf(bc2, 1e-8f);
            gate_w[idx] += mh_gw / (sqrtf(vh_gw) + adam_eps);
            gate_w[idx] = fminf(fmaxf(gate_w[idx], -10.0f), 10.0f);
        }

        // Adam update for biases
        float g_b = eff_lr_w * d_linear;
        bias[tid] *= (1.0f - weight_decay);
        adam_m_b[tid] = beta1 * adam_m_b[tid] + (1.0f - beta1) * g_b;
        adam_v_b[tid] = beta2 * adam_v_b[tid] + (1.0f - beta2) * g_b * g_b;
        bias[tid] += (adam_m_b[tid] / fmaxf(bc1, 1e-8f)) / (sqrtf(adam_v_b[tid] / fmaxf(bc2, 1e-8f)) + adam_eps);
        bias[tid] = fminf(fmaxf(bias[tid], -10.0f), 10.0f);

        // Gate bias
        float g_gb = eff_lr_w * d_gate_pre;
        gate_b[tid] *= (1.0f - weight_decay);
        adam_m_gb[tid] = beta1 * adam_m_gb[tid] + (1.0f - beta1) * g_gb;
        adam_v_gb[tid] = beta2 * adam_v_gb[tid] + (1.0f - beta2) * g_gb * g_gb;
        gate_b[tid] += (adam_m_gb[tid] / fmaxf(bc1, 1e-8f)) / (sqrtf(adam_v_gb[tid] / fmaxf(bc2, 1e-8f)) + adam_eps);
        gate_b[tid] = fminf(fmaxf(gate_b[tid], -10.0f), 10.0f);
    }

    // -- NaN guard --
    if (isnan(me->mean[tid]) || isinf(me->mean[tid])) {
        me->mean[tid] = below->mean[tid];
        me->precision[tid] = 0.01f;
    }
    for (unsigned int j = 0; j < 64; j++) {
        unsigned int idx = tid * 64 + j;
        if (isnan(weights[idx]) || isinf(weights[idx]))
            weights[idx] = (tid == j) ? 1.0f : 0.0f;
        if (isnan(gate_w[idx]) || isinf(gate_w[idx]))
            gate_w[idx] = 0.0f;
    }
    if (isnan(bias[tid]) || isinf(bias[tid])) bias[tid] = 0.0f;
    if (isnan(gate_b[tid]) || isinf(gate_b[tid])) gate_b[tid] = 0.0f;

    // -- 8. PRECISION UPDATE --
    float res_sq = clamped_res * clamped_res;
    float inv_prec = 1.0f / fmaxf(prec, 0.01f);
    float prec_grad = inv_prec - res_sq;
    float new_prec = prec + prec_eta * prec_grad;
    me->precision[tid] = fminf(fmaxf(new_prec, 0.01f), 10.0f);
}
