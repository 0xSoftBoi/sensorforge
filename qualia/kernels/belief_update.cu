// CUDA port of belief_update.metal
// Must match Rust BeliefSlot repr(C, align(64)) layout.
// STATE_DIM = 64

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
};

// params[0] = threshold
// params[1] = learning_rate (belief update)
// params[2] = layer_id
// params[3] = weight_decay

extern "C" __global__ void belief_update(
    BeliefSlot*       me,
    const BeliefSlot* below,
    const float*      params,
    float*            weights,  // 64x64 generative model
    float*            bias      // 64 bias
)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64) return;

    float threshold    = params[0];
    float lr           = params[1];
    float weight_decay = params[3];
    float lr_w         = lr * 0.1f;  // weight learning rate (10x slower than belief)

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

    // -- 2. RESIDUAL (prediction error) --
    float residual = below->mean[tid] - pred;
    me->residual[tid] = residual;

    // -- 3. PRECISION-WEIGHTED SQUARED ERROR --
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
    }

    __syncthreads();

    // -- 6. BELIEF UPDATE: gradient descent on free energy --
    if (total_vfe > threshold) {
        float eff_prec = fminf(prec, 1.0f);
        float delta = lr * eff_prec * residual;
        delta = fminf(fmaxf(delta, -0.1f), 0.1f);  // max step per cycle
        me->mean[tid] += delta;
        me->mean[tid] = fminf(fmaxf(me->mean[tid], -10.0f), 10.0f);
    }

    // -- 7. WEIGHT LEARNING: update generative model --
    if (total_vfe > threshold) {
        float eff_prec_w = fminf(prec, 1.0f);
        float grad_scale = fminf(fmaxf(lr_w * eff_prec_w * residual, -0.01f), 0.01f);
        for (unsigned int j = 0; j < 64; j++) {
            weights[tid * 64 + j] *= (1.0f - weight_decay);
            weights[tid * 64 + j] += grad_scale * shared_mean[j];
            weights[tid * 64 + j] = fminf(fmaxf(weights[tid * 64 + j], -10.0f), 10.0f);
        }
        bias[tid] *= (1.0f - weight_decay);
        bias[tid] += fminf(fmaxf(lr_w * eff_prec_w * residual, -0.01f), 0.01f);
        bias[tid] = fminf(fmaxf(bias[tid], -10.0f), 10.0f);
    }

    // -- NaN guard: if mean is NaN/Inf, reset to sensory input --
    if (isnan(me->mean[tid]) || isinf(me->mean[tid])) {
        me->mean[tid] = below->mean[tid];
        me->precision[tid] = 0.01f;
    }

    // -- Also reset weights if they went NaN (fixes the recurring NaN bug) --
    for (unsigned int j = 0; j < 64; j++) {
        if (isnan(weights[tid * 64 + j]) || isinf(weights[tid * 64 + j])) {
            weights[tid * 64 + j] = (tid == j) ? 1.0f : 0.0f;  // reset to identity
        }
    }
    if (isnan(bias[tid]) || isinf(bias[tid])) {
        bias[tid] = 0.0f;
    }

    // -- 8. PRECISION UPDATE: adapt confidence based on prediction accuracy --
    float abs_res = fabsf(residual);
    if (abs_res < 0.01f) {
        me->precision[tid] = fminf(prec * 1.001f, 10.0f);
    } else {
        me->precision[tid] = fmaxf(prec * 0.999f, 0.01f);
    }
}
