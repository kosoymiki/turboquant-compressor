/**
 * TurboQuant Core — Precision Calibration
 * FP8/INT8 calibration for GPU training
 * Ported from precision_calibration.ts
 */

#ifndef TQ_CALIBRATION_H
#define TQ_CALIBRATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Calibration policy types ──────────────────────────────────────────────────
#define TQ_CALIB_AFFINE_PER_TENSOR    0
#define TQ_CALIB_SYMMETRIC_PER_TENSOR 1
#define TQ_CALIB_AFFINE_PER_GROUP     2
#define TQ_CALIB_GROUP_MINMAX         3

// ── Surface role types ──────────────────────────────────────────────────────
#define TQ_CALIB_ROLE_ACTIVATION  0
#define TQ_CALIB_ROLE_WEIGHT      1
#define TQ_CALIB_ROLE_KV_VALUE    2

// ── Observer statistics ───────────────────────────────────────────────────────
typedef struct {
    float min_val;
    float max_val;
    float abs_max;
    float mean;
    float variance;
    float p01;
    float p99;
} tq_calib_observer;

// ── Fake quant summary ───────────────────────────────────────────────────────
typedef struct {
    uint8_t policy;
    uint8_t bits;
    float mse;
    float max_abs_error;
    float cosine_similarity;
    float clipped_fraction;
} tq_calib_fake_quant;

// ── Step size summary ─────────────────────────────────────────────────────────
typedef struct {
    uint8_t policy;
    uint8_t learnable;
    float initial_step_size;
    float zero_point;
    int qmin;
    int qmax;
    float gradient_scale;
    uint32_t group_count;
    float min_group_step;
    float max_group_step;
} tq_calib_step_size;

// ── Surface summary ──────────────────────────────────────────────────────────
typedef struct {
    char name[64];
    uint8_t role;
    uint32_t sample_count;
    uint32_t element_count;
    tq_calib_observer observer;
    tq_calib_fake_quant fake_quant;
    tq_calib_step_size step_size;
} tq_calib_surface;

// ── Runtime state ────────────────────────────────────────────────────────────
typedef struct {
    uint32_t surface_count;
    tq_calib_surface surfaces[8];  // max 8 surfaces
    float activation_step_size_mean;
    float weight_step_size_mean;
    float kv_group_step_size_mean;
    float beta_codebook_min;
    float beta_codebook_max;
    float beta_mse_per_coord;
} tq_calib_runtime;

// ── Functions ─────────────────────────────────────────────────────────────────
void tq_calib_compute_observer(const float* values, uint32_t count, tq_calib_observer* out);
void tq_calib_fake_quant_affine(const float* values, uint32_t count, uint8_t bits, float* out);
void tq_calib_fake_quant_symmetric(const float* values, uint32_t count, uint8_t bits, float* out);
void tq_calib_fake_quant_group(const float* values, uint32_t count, uint32_t group_size, uint8_t bits, float* out);

int tq_calib_build_runtime_state(
    const float** surfaces_data,
    const uint32_t* surfaces_counts,
    uint32_t surface_count,
    uint8_t* roles,
    tq_calib_runtime* state_out
);

float tq_calib_compute_error_mse(const float* reference, const float* reconstructed, uint32_t count);
float tq_calib_compute_cosine_similarity(const float* a, const float* b, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif // TQ_CALIBRATION_H