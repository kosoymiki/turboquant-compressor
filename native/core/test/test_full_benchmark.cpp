/**
 * TurboQuant Core — Full Benchmark Suite v4.6.0
 */

#include "tq_core.h"
#include "tq_kv_cache.h"
#include "tq_benchmark.h"
#include "tq_oracle.h"
#include "tq_calibration.h"
#include "tq_lsq_optimizer.h"
#include "tq_distillation.h"
#include "tq_qat_training.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { \
    printf("[%s]\n", name); \
    test_count++; \
} while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 1; \
    } \
} while(0)

#define PASS() do { \
    printf("  PASS\n"); \
    pass_count++; \
} while(0)

// Generate test vectors
static float* gen_vectors(uint32_t N, uint32_t D) {
    float* v = (float*)malloc(N * D * sizeof(float));
    uint32_t s = 42;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < D; j++) {
            s = (s * 1103515245u + 12345u) & 0x7fffffffu;
            v[i * D + j] = ((float)s / (float)0x7fffffff) * 2.0f - 1.0f;
        }
    }
    return v;
}

// ── Compress/Decompress ───────────────────────────────────────────────────────
int test_compress_decompress() {
    TEST("compress/decompress");
    uint32_t N = 100, D = 64;
    float* vecs = gen_vectors(N, D);

    size_t est = tq_estimate_compressed_size(N, D, 4, 0);
    char* out = (char*)malloc(est * 2);
    size_t out_len = est * 2;

    int r = tq_compress(vecs, N, D, 4, 42, 0, 0, out, &out_len, nullptr);
    ASSERT(r == TQ_OK, "compress failed");
    PASS();

    // Search
    uint32_t idx[5];
    float scr[5];
    r = tq_search(out, vecs, D, 5, 0, 0, idx, scr, nullptr);
    ASSERT(r == TQ_OK, "search failed");
    ASSERT(idx[0] == 0, "self-match not top-1");
    ASSERT(scr[0] > 0.9f, "self-match score too low");
    printf("  top-1: idx=%u score=%.4f\n", idx[0], scr[0]);
    PASS();

    free(vecs);
    free(out);
    return 0;
}

// ── KV Cache ───────────────────────────────────────────────────────────────
int test_kv_cache() {
    TEST("kv_cache");
    tq_kv_cache_config cfg = {0};
    cfg.head_dim = 64;
    cfg.key_bits = 4;
    cfg.value_bits = 2;
    cfg.sink_tokens = 4;

    void* cache;
    int r = tq_kv_cache_create(&cfg, &cache);
    ASSERT(r == TQ_OK, "cache create failed");
    PASS();

    // Prefill
    float* keys = gen_vectors(128, 64);
    float* vals = gen_vectors(128, 64);
    r = tq_kv_cache_prefill(cache, keys, vals, 128);
    ASSERT(r == TQ_OK, "prefill failed");
    PASS();

    // Attention
    float query[64], output[64];
    memcpy(query, keys, 64 * sizeof(float));
    r = tq_kv_cache_attention(cache, query, output);
    ASSERT(r == TQ_OK, "attention failed");
    PASS();

    tq_kv_cache_destroy(cache);
    free(keys);
    free(vals);
    return 0;
}

// ── Oracle ─────────────────────────────────────────────────────────────────
int test_oracle() {
    TEST("oracle");
    float data[256];
    for (int i = 0; i < 256; i++) data[i] = ((float)i / 256.0f) * 2.0f - 1.0f;

    tq_data_characteristics chars;
    tq_oracle_analyze(data, 256, &chars);
    ASSERT(chars.min < 0, "min not negative");
    PASS();
    ASSERT(chars.max > 0, "max not positive");
    PASS();

    uint8_t opt = tq_oracle_optimal_bits(data, 256, 0.01f);
    printf("  optimal_bits(tol=0.01) = %u\n", opt);
    PASS();

    return 0;
}

// ── Calibration ─────────────────────────────────────────────────────────────
int test_calibration() {
    TEST("calibration");
    float data[512];
    for (int i = 0; i < 512; i++) data[i] = ((float)i / 512.0f) * 2.0f - 1.0f;

    tq_calib_observer obs;
    tq_calib_compute_observer(data, 512, &obs);
    ASSERT(obs.min_val < 0, "min invalid");
    PASS();
    ASSERT(obs.max_val > 0, "max invalid");
    PASS();
    printf("  observer: min=%.3f max=%.3f mean=%.3f\n", obs.min_val, obs.max_val, obs.mean);

    float rec[512];
    tq_calib_fake_quant_symmetric(data, 512, 4, rec);
    float mse = tq_calib_compute_error_mse(data, rec, 512);
    printf("  symmetric MSE = %.6f\n", mse);
    PASS();

    return 0;
}

// ── LSQ Optimizer ────────────────────────────────────────────────────────────
int test_lsq_optimizer() {
    TEST("lsq_optimizer");
    float values[256], teacher[256];
    for (int i = 0; i < 256; i++) {
        values[i] = ((float)i / 256.0f) * 1.5f - 0.75f;
        teacher[i] = values[i] * 0.91f + ((float)i * 0.015f);
    }

    tq_lsq_optimizer_result res;
    int r = tq_lsq_optimize(values, teacher, 256, 4, 0.01f, &res);
    ASSERT(r == TQ_OK, "lsq optimize failed");
    PASS();

    printf("  initial_loss=%.6f updated_loss=%.6f\n", res.initial_loss, res.updated_loss);
    printf("  gradient_error=%.4f cosine_sim=%.4f\n", res.relative_gradient_error, res.cosine_similarity_after_update);
    PASS();

    return 0;
}

// ── Distillation ────────────────────────────────────────────────────────────
int test_distillation() {
    TEST("distillation");
    tq_distillation_result res;
    int r = tq_distillation_run(5, 16, 4, 1.75f, 0.7f, 0.3f, 0.85f, &res);
    ASSERT(r == TQ_OK, "distillation failed");
    PASS();

    printf("  initial_composite=%.6f updated_composite=%.6f\n", res.initial_composite_loss, res.updated_composite_loss);
    printf("  update_norm=%.4f\n", res.update_norm);
    PASS();

    return 0;
}

// ── QAT Training ─────────────────────────────────────────────────────────────
int test_qat_training() {
    TEST("qat_training");
    float activation[512], weight[256], kv[128], teacher[256], student[256];

    for (int i = 0; i < 512; i++) activation[i] = ((float)i / 512.0f) * 2.0f - 1.0f;
    for (int i = 0; i < 256; i++) weight[i] = ((float)i / 256.0f) * 1.5f - 0.75f;
    for (int i = 0; i < 128; i++) kv[i] = ((float)i / 128.0f) * 2.0f - 1.0f;
    for (int i = 0; i < 256; i++) {
        teacher[i] = weight[i] * 0.91f;
        student[i] = weight[i] * 0.93f;
    }

    tq_qat_config cfg = {0};
    cfg.batch_size = 5;
    cfg.vocab_size = 16;
    cfg.bits = 4;
    cfg.temperature = 1.75f;
    cfg.alpha = 0.7f;
    cfg.beta = 0.3f;
    cfg.learning_rate = 0.85f;
    cfg.epochs = 2;

    tq_qat_result res;
    int r = tq_qat_run(activation, 512, weight, 256, kv, 128, teacher, student, 256, &cfg, &res);
    ASSERT(r == TQ_OK, "qat run failed");
    PASS();

    printf("  calibration=%u lsq=%u distillation=%u loop=%u\n",
           res.calibration_ready, res.lsq_ready, res.distillation_ready, res.loop_ready);
    PASS();
    printf("  aggregate_improvement=%.6f\n", res.aggregate_improvement);
    PASS();

    return 0;
}

int main() {
    printf("\n=== TurboQuant Core Full Benchmark v4.6.0 ===\n\n");

    test_compress_decompress();
    test_kv_cache();
    test_oracle();
    test_calibration();
    test_lsq_optimizer();
    test_distillation();
    test_qat_training();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", pass_count, test_count - pass_count);
    if (pass_count == test_count) {
        printf("ALL TESTS PASSED\n");
    }

    return pass_count == test_count ? 0 : 1;
}