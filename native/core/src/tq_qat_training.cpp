/**
 * TurboQuant Core — QAT Training Loop Implementation
 * Full Quantization-Aware Training Pipeline
 */

#include "tq_qat_training.h"
#include "tq_calibration.h"
#include "tq_lsq_optimizer.h"
#include "tq_distillation.h"
#include "tq_core.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

// ── Full QAT pipeline ────────────────────────────────────────────────────────
int tq_qat_run(
    const float* activation_data, uint32_t activation_count,
    const float* weight_data, uint32_t weight_count,
    const float* kv_data, uint32_t kv_count,
    const float* teacher_data, const float* student_data, uint32_t data_count,
    const tq_qat_config* config,
    tq_qat_result* result
) {
    if (!result) return TQ_ERR_NULL;
    memset(result, 0, sizeof(tq_qat_result));

    tq_qat_config cfg;
    if (config) {
        memcpy(&cfg, config, sizeof(tq_qat_config));
    } else {
        memset(&cfg, 0, sizeof(tq_qat_config));
    }

    // Default config
    if (cfg.batch_size == 0) cfg.batch_size = 5;
    if (cfg.vocab_size == 0) cfg.vocab_size = 16;
    if (cfg.bits == 0) cfg.bits = 4;
    if (cfg.temperature == 0) cfg.temperature = 1.75f;
    if (cfg.alpha == 0) cfg.alpha = 0.7f;
    if (cfg.beta == 0) cfg.beta = 0.3f;
    if (cfg.learning_rate == 0) cfg.learning_rate = 0.85f;
    if (cfg.epochs == 0) cfg.epochs = 2;

    // ── Stage 1: Calibration ──────────────────────────────────────────────
    uint8_t calib_ready = 0;
    {
        uint8_t roles[3] = {TQ_CALIB_ROLE_ACTIVATION, TQ_CALIB_ROLE_WEIGHT, TQ_CALIB_ROLE_KV_VALUE};
        const float* surfaces[3];
        uint32_t counts[3];
        uint32_t surf_count = 0;

        if (activation_data && activation_count > 0) {
            surfaces[surf_count] = activation_data;
            counts[surf_count] = activation_count;
            roles[surf_count] = TQ_CALIB_ROLE_ACTIVATION;
            surf_count++;
        }
        if (weight_data && weight_count > 0) {
            surfaces[surf_count] = weight_data;
            counts[surf_count] = weight_count;
            roles[surf_count] = TQ_CALIB_ROLE_WEIGHT;
            surf_count++;
        }
        if (kv_data && kv_count > 0) {
            surfaces[surf_count] = kv_data;
            counts[surf_count] = kv_count;
            roles[surf_count] = TQ_CALIB_ROLE_KV_VALUE;
            surf_count++;
        }

        if (surf_count > 0) {
            tq_calib_runtime state;
            int r = tq_calib_build_runtime_state(surfaces, counts, surf_count, roles, &state);
            calib_ready = (r == TQ_OK && state.surface_count > 0);
        }
    }
    result->calibration_ready = calib_ready;

    // ── Stage 2: LSQ Optimization ────────────────────────────────────────
    float lsq_initial_loss = 0, lsq_updated_loss = 0;
    uint8_t lsq_ready = 0;
    {
        if (student_data && data_count > 0 && teacher_data) {
            tq_lsq_optimizer_result lsq_res;
            int r = tq_lsq_optimize(student_data, teacher_data, data_count, cfg.bits, cfg.learning_rate * 0.25f, &lsq_res);
            lsq_ready = (r == TQ_OK && lsq_res.updated_loss < lsq_res.initial_loss);
            lsq_initial_loss = lsq_res.initial_loss;
            lsq_updated_loss = lsq_res.updated_loss;
        }
    }
    result->lsq_ready = lsq_ready;
    result->initial_quant_loss = lsq_initial_loss;
    result->post_lsq_loss = lsq_updated_loss;

    // ── Stage 3: Distillation ───────────────────────────────────────────
    float dist_initial = 0, dist_updated = 0;
    uint8_t dist_ready = 0;
    {
        tq_distillation_result dist_res;
        int r = tq_distillation_run(
            cfg.batch_size, cfg.vocab_size,
            cfg.bits, cfg.temperature,
            cfg.alpha, cfg.beta, cfg.learning_rate,
            &dist_res
        );
        dist_ready = (r == TQ_OK && dist_res.updated_composite_loss < dist_res.initial_composite_loss);
        dist_initial = dist_res.initial_composite_loss;
        dist_updated = dist_res.updated_composite_loss;
    }
    result->distillation_ready = dist_ready;
    result->initial_distillation_loss = dist_initial;
    result->post_distillation_loss = dist_updated;

    // ── Aggregate results ─────────────────────────────────────────────────
    result->aggregate_initial_loss = lsq_initial_loss + dist_initial;
    result->aggregate_final_loss = lsq_updated_loss + dist_updated;
    result->aggregate_improvement = result->aggregate_initial_loss - result->aggregate_final_loss;

    // ── Loop readiness ─────────────────────────────────────────────────────
    result->loop_ready = calib_ready && lsq_ready && dist_ready &&
                         result->aggregate_final_loss < result->aggregate_initial_loss &&
                         result->aggregate_improvement > 0;

    result->epochs_completed = result->loop_ready ? cfg.epochs : 0;

    return TQ_OK;
}