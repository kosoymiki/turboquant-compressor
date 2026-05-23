/**
 * TurboQuant Core — LSQ Optimizer Implementation
 * Least-Squares Quantization Optimization
 */

#include "tq_lsq_optimizer.h"
#include "tq_core.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

// ── MSE helper ───────────────────────────────────────────────────────────────
static float mse(const float* a, const float* b, uint32_t count) {
    if (!a || !b || count == 0) return 0.0f;
    double total = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return (float)(total / count);
}

// ── Compute initial step size ─────────────────────────────────────────────────
void tq_lsq_compute_step_size(const float* values, uint32_t count, uint8_t bits, float* step_size_out) {
    if (!values || !step_size_out) return;
    int qmax = (1 << (bits - 1)) - 1;
    float abs_max = 0;
    for (uint32_t i = 0; i < count; i++) {
        float a = values[i];
        if (a < 0) a = -a;
        if (a > abs_max) abs_max = a;
    }
    *step_size_out = fmaxf(abs_max / (float)qmax, 1e-10f);
}

// ── Surrogate quantize symmetric ─────────────────────────────────────────────
float tq_lsq_surrogate_quantize_symmetric(
    const float* values, uint32_t count, float step_size,
    int qmin, int qmax, float* out
) {
    if (!values || !out) return 0.0f;
    float safe_step = fmaxf(step_size, 1e-10f);
    for (uint32_t i = 0; i < count; i++) {
        float scaled = values[i] / safe_step;
        float clamped = scaled;
        if (clamped < (float)qmin) clamped = (float)qmin;
        if (clamped > (float)qmax) clamped = (float)qmax;
        out[i] = clamped * safe_step;
    }
    return safe_step;
}

// ── Quantize symmetric ─────────────────────────────────────────────────────────
float tq_lsq_quantize_symmetric(
    const float* values, const float* teacher, uint32_t count,
    uint8_t bits, float step_size, float* out
) {
    if (!values || !out) return 0.0f;
    int qmax = (1 << (bits - 1)) - 1;
    int qmin = -qmax;
    float safe_step = fmaxf(step_size, 1e-10f);
    for (uint32_t i = 0; i < count; i++) {
        float q = values[i] / safe_step;
        int qi = (int)(q + 0.5f);
        if (qi < qmin) qi = qmin;
        if (qi > qmax) qi = qmax;
        out[i] = (float)qi * safe_step;
    }
    if (teacher) return mse(out, teacher, count);
    return 0.0f;
}

// ── Finite difference gradient ───────────────────────────────────────────────
float tq_lsq_finite_diff_gradient(
    const float* values, const float* teacher, uint32_t count,
    float step_size, int qmin, int qmax
) {
    if (!values || !teacher) return 0.0f;
    float eps = fmaxf(step_size * 0.001f, 1e-5f);
    float loss_plus = 0, loss_minus = 0;

    {
        float* qp = (float*)malloc(count * sizeof(float));
        tq_lsq_surrogate_quantize_symmetric(values, count, step_size + eps, qmin, qmax, qp);
        loss_plus = mse(qp, teacher, count);
        free(qp);
    }
    {
        float* qm = (float*)malloc(count * sizeof(float));
        tq_lsq_surrogate_quantize_symmetric(values, count, fmaxf(step_size - eps, 1e-8f), qmin, qmax, qm);
        loss_minus = mse(qm, teacher, count);
        free(qm);
    }

    return (loss_plus - loss_minus) / (2.0f * eps);
}

// ── Analytic gradient ────────────────────────────────────────────────────────
float tq_lsq_analytic_gradient(
    const float* values, const float* teacher, uint32_t count,
    float step_size, int qmin, int qmax
) {
    if (!values || !teacher) return 0.0f;
    float safe_step = fmaxf(step_size, 1e-10f);
    float gradient = 0;
    uint32_t clipped = 0;

    for (uint32_t i = 0; i < count; i++) {
        float scaled = values[i] / safe_step;
        float clamped = scaled;
        float dqds = 1.0f;
        if (clamped < (float)qmin) {
            clamped = (float)qmin;
            dqds = 0;
            clipped++;
        } else if (clamped > (float)qmax) {
            clamped = (float)qmax;
            dqds = 0;
            clipped++;
        }
        float dequantized = clamped * safe_step;
        gradient += 2.0f * (dequantized - teacher[i]) * dqds;
    }

    gradient /= (float)count;
    (void)clipped;  // tracked in tq_lsq_analytic_gradient_internal
    return gradient;
}

// Track clipped for result
static float tq_lsq_analytic_gradient_internal(
    const float* values, const float* teacher, uint32_t count,
    float step_size, int qmin, int qmax, uint32_t* clipped_out
) {
    if (!values || !teacher) return 0.0f;
    float safe_step = fmaxf(step_size, 1e-10f);
    float gradient = 0;
    uint32_t clipped = 0;

    for (uint32_t i = 0; i < count; i++) {
        float scaled = values[i] / safe_step;
        float clamped = scaled;
        float dqds = 1.0f;
        if (clamped < (float)qmin) {
            clamped = (float)qmin;
            dqds = 0;
            clipped++;
        } else if (clamped > (float)qmax) {
            clamped = (float)qmax;
            dqds = 0;
            clipped++;
        }
        float dequantized = clamped * safe_step;
        gradient += 2.0f * (dequantized - teacher[i]) * dqds;
    }

    gradient /= (float)count;
    if (clipped_out) *clipped_out = clipped;
    return gradient;
}

// ── Full optimization ──────────────────────────────────────────────────────────
int tq_lsq_optimize(
    const float* values, const float* teacher, uint32_t count,
    uint8_t bits, float learning_rate, tq_lsq_optimizer_result* result
) {
    if (!values || !teacher || !result) return TQ_ERR_NULL;
    memset(result, 0, sizeof(tq_lsq_optimizer_result));

    int qmax = (1 << (bits - 1)) - 1;
    int qmin = -qmax;

    // Initial step size
    tq_lsq_compute_step_size(values, count, bits, &result->initial_step_size);
    float step_size = result->initial_step_size;

    // Initial quantization
    float* initial_q = (float*)malloc(count * sizeof(float));
    tq_lsq_surrogate_quantize_symmetric(values, count, step_size, qmin, qmax, initial_q);
    result->initial_loss = mse(initial_q, teacher, count);

    // Compute gradients
    result->finite_diff_gradient = tq_lsq_finite_diff_gradient(values, teacher, count, step_size, qmin, qmax);
    uint32_t clipped_count = 0;
    result->analytic_gradient = tq_lsq_analytic_gradient_internal(values, teacher, count, step_size, qmin, qmax, &clipped_count);

    float grad_err_denom = fmaxf(fabsf(result->analytic_gradient), fmaxf(fabsf(result->finite_diff_gradient), 1e-8f));
    result->relative_gradient_error = fabsf(result->analytic_gradient - result->finite_diff_gradient) / grad_err_denom;

    result->gradient_scale = 1.0f / sqrtf((float)count * (float)qmax);
    float scaled_grad = result->analytic_gradient * result->gradient_scale;
    result->learning_rate = learning_rate > 0 ? learning_rate : step_size * 0.25f;
    result->updated_step_size = fmaxf(step_size - result->learning_rate * scaled_grad, 1e-8f);

    // Updated quantization
    float* updated_q = (float*)malloc(count * sizeof(float));
    tq_lsq_surrogate_quantize_symmetric(values, count, result->updated_step_size, qmin, qmax, updated_q);
    result->updated_loss = mse(updated_q, teacher, count);

    // Cosine similarity
    float dot = 0, norm_a = 0, norm_b = 0;
    for (uint32_t i = 0; i < count; i++) {
        dot += updated_q[i] * teacher[i];
        norm_a += updated_q[i] * updated_q[i];
        norm_b += teacher[i] * teacher[i];
    }
    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    result->cosine_similarity_after_update = denom > 0 ? dot / denom : 0;

    // Clipped fraction
    result->clipped_fraction = (float)clipped_count / (float)count;

    free(initial_q);
    free(updated_q);
    return TQ_OK;
}