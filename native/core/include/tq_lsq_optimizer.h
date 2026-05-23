/**
 * TurboQuant Core — LSQ Optimizer
 * Least-Squares Quantization Optimization
 * Ported from lsq_optimizer.ts
 */

#ifndef TQ_LSQ_OPTIMIZER_H
#define TQ_LSQ_OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Optimizer result ──────────────────────────────────────────────────────────
typedef struct {
    float initial_step_size;
    float updated_step_size;
    float initial_loss;
    float updated_loss;
    float finite_diff_gradient;
    float analytic_gradient;
    float relative_gradient_error;
    float cosine_similarity_after_update;
    float clipped_fraction;
    float learning_rate;
    float gradient_scale;
} tq_lsq_optimizer_result;

// ── Functions ─────────────────────────────────────────────────────────────────
void tq_lsq_compute_step_size(
    const float* values, uint32_t count, uint8_t bits, float* step_size_out
);

float tq_lsq_quantize_symmetric(
    const float* values, const float* teacher, uint32_t count,
    uint8_t bits, float step_size, float* out
);

float tq_lsq_surrogate_quantize_symmetric(
    const float* values, uint32_t count, float step_size,
    int qmin, int qmax, float* out
);

float tq_lsq_finite_diff_gradient(
    const float* values, const float* teacher, uint32_t count,
    float step_size, int qmin, int qmax
);

float tq_lsq_analytic_gradient(
    const float* values, const float* teacher, uint32_t count,
    float step_size, int qmin, int qmax
);

int tq_lsq_optimize(
    const float* values, const float* teacher, uint32_t count,
    uint8_t bits, float learning_rate, tq_lsq_optimizer_result* result
);

#ifdef __cplusplus
}
#endif

#endif // TQ_LSQ_OPTIMIZER_H