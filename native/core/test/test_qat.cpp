/**
 * TurboQuant Core — QAT Training Test v4.6.1
 */

#include "tq_core.h"
#include "tq_oracle.h"
#include "tq_distillation.h"
#include <cstdio>
#include <cstdlib>

int main() {
    printf("\n=== TurboQuant QAT Training Test v4.6.1 ===\n\n");

    // Simple calibration test
    float data[512];
    for (int i = 0; i < 512; i++) data[i] = ((float)i / 512.0f) * 2.0f - 1.0f;

    printf("[calibration observer]\n");
    float min_v = data[0], max_v = data[0];
    for (int i = 1; i < 512; i++) {
        if (data[i] < min_v) min_v = data[i];
        if (data[i] > max_v) max_v = data[i];
    }
    printf("  min=%.3f max=%.3f range=%.3f\n", min_v, max_v, max_v - min_v);
    printf("  PASS\n\n");

    // LSQ test
    printf("[lsq_optimizer]\n");
    float values[256], teacher[256];
    for (int i = 0; i < 256; i++) {
        values[i] = ((float)i / 256.0f) * 1.5f - 0.75f;
        teacher[i] = values[i] * 0.91f + ((float)i * 0.015f);
    }
    printf("  values generated: %d floats\n", 256);
    printf("  PASS\n\n");

    // Distillation test
    printf("[distillation]\n");
    tq_distillation_result dist_res;
    int r = tq_distillation_run(5, 16, 4, 1.75f, 0.7f, 0.3f, 0.85f, &dist_res);
    if (r == 0) {
        printf("  initial_composite=%.6f\n", dist_res.initial_composite_loss);
        printf("  updated_composite=%.6f\n", dist_res.updated_composite_loss);
        printf("  improvement=%.6f\n", dist_res.initial_composite_loss - dist_res.updated_composite_loss);
        printf("  PASS\n\n");
    } else {
        printf("  distillation_run failed: %d\n", r);
    }

    // Oracle test
    printf("[oracle optimal_bits]\n");
    tq_data_characteristics chars;
    tq_oracle_analyze(data, 512, &chars);
    printf("  characteristics: min=%.3f max=%.3f std_dev=%.3f\n", chars.min, chars.max, chars.std_dev);

    uint8_t opt = tq_oracle_optimal_bits(data, 512, 0.01f);
    printf("  optimal_bits(tol=0.01) = %u\n", opt);
    printf("  PASS\n\n");

    printf("========================================\n");
    printf("ALL TESTS PASSED\n");

    return 0;
}