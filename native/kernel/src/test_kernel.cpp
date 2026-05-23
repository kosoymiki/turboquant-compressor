/**
 * TurboQuant Native Kernel — Test Suite
 */

#include "../include/tq_rotation_kernel.h"
#include <CL/cl.h>
#include "../include/tq_quantizer_kernel.h"
#include "../include/tq_qjl_kernel.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) \
    do { \
        fprintf(stderr, "[TEST] %s...", #name); \
        if (name()) { \
            fprintf(stderr, " PASS\n"); \
            g_passed++; \
        } else { \
            fprintf(stderr, " FAIL\n"); \
            g_failed++; \
        } \
    } while(0)

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "\n  ASSERT FAILED: %s\n", msg); \
            return 0; \
        } \
    } while(0)

int test_rotation_init_shutdown() {
    tq_rotation_kernel_t kernel;
    cl_int err = tq_rotation_init(&kernel, 1024, 42, TQ_ROTATION_MODE_FWHT_SIGN);
    if (err != CL_SUCCESS) return 0;
    ASSERT(kernel.is_initialized == 1, "kernel initialized");
    err = tq_rotation_shutdown(&kernel);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

int test_rotation_single_vector() {
    tq_rotation_kernel_t kernel;
    tq_rotation_init(&kernel, 1024, 42, TQ_ROTATION_MODE_FWHT_SIGN);

    float input[1024];
    float output[1024];
    for (int i = 0; i < 1024; i++) input[i] = (float)(i % 100) / 100.0f;

    cl_int err = tq_rotation_execute(&kernel, input, output, 1);
    if (err != CL_SUCCESS) { tq_rotation_shutdown(&kernel); return 0; }

    // Check norm preservation (within tolerance)
    float norm_in = 0, norm_out = 0;
    for (int i = 0; i < 1024; i++) {
        norm_in += input[i] * input[i];
        norm_out += output[i] * output[i];
    }
    norm_in = std::sqrt(norm_in);
    norm_out = std::sqrt(norm_out);

    tq_rotation_shutdown(&kernel);
    return std::abs(norm_in - norm_out) < 0.01f;
}

int test_rotation_cpp_batch() {
    tq::kernel::RotationKernel rot(512, 123);

    float vec[512];
    for (int i = 0; i < 512; i++) vec[i] = (float)(i % 50) / 50.0f;

    float out[512];
    rot.rotate(vec, out);

    float norm = 0;
    for (int i = 0; i < 512; i++) norm += out[i] * out[i];
    norm = std::sqrt(norm);

    return norm > 0.0f;
}

int test_quantizer_init() {
    tq_quantizer_kernel_t q;
    cl_int err = tq_quantizer_init(&q, 1024, 4);
    if (err != CL_SUCCESS) return 0;
    ASSERT(q.is_initialized == 1, "quantizer initialized");
    err = tq_quantizer_shutdown(&q);
    return err == CL_SUCCESS;
}

int test_quantizer_encode_decode() {
    tq_quantizer_kernel_t q;
    tq_quantizer_init(&q, 1024, 4);

    float values[1024];
    uint8_t packed[512];
    float decoded[1024];

    for (int i = 0; i < 1024; i++) values[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;

    cl_int err = tq_quantizer_encode(&q, values, packed, 1024);
    if (err != CL_SUCCESS) { tq_quantizer_shutdown(&q); return 0; }

    err = tq_quantizer_decode(&q, packed, decoded, 1024);
    if (err != CL_SUCCESS) { tq_quantizer_shutdown(&q); return 0; }

    float mse = tq_quantizer_compute_mse(&q, decoded, 1024);
    tq_quantizer_shutdown(&q);

    return mse < 1.0f;
}

int test_quantizer_cpp_beta() {
    tq::kernel::BetaCodebook cb(1024, 4);
    cb.compute();

    float samples[1024];
    for (int i = 0; i < 1024; i++) samples[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;

    float q = cb.quantize(samples[512]);
    uint32_t idx = cb.quantize_index(samples[512]);
    float dq = cb.dequantize(idx);

    return std::abs(q - dq) < 0.1f;
}

int test_qjl_init_shutdown() {
    tq_qjl_kernel_t qjl;
    cl_int err = tq_qjl_init(&qjl, 1024, 64, 42, 4, 1);
    if (err != CL_SUCCESS) return 0;
    ASSERT(qjl.is_initialized == 1, "qjl initialized");
    err = tq_qjl_shutdown(&qjl);
    return err == CL_SUCCESS;
}

int test_qjl_compress() {
    tq_qjl_kernel_t qjl;
    tq_qjl_init(&qjl, 1024, 64, 42, 4, 1);

    float residual[1024];
    uint8_t sketch[64];

    for (int i = 0; i < 1024; i++) residual[i] = (float)(i % 50) / 50.0f;

    cl_int err = tq_qjl_compress(&qjl, residual, sketch);
    tq_qjl_shutdown(&qjl);

    return err == CL_SUCCESS;
}

int test_qjl_cpp_projector() {
    tq::kernel::QJLProjector proj(512, 64, 123);

    float vec[512];
    float proj_out[64];
    for (int i = 0; i < 512; i++) vec[i] = (float)(i % 50) / 50.0f;

    proj.project(vec, proj_out);

    float dot = proj.compute_dot_product(vec, vec);
    return dot > 0.0f;
}

int test_qjl_cpp_sketch() {
    tq::kernel::QJLSketchCompressor comp(64, 4);

    float projected[64];
    for (int i = 0; i < 64; i++) projected[i] = (float)(i % 20) / 20.0f - 0.5f;

    uint8_t* sketch = comp.compress(projected);

    float* decompressed = comp.decompress(sketch);
    bool ok = decompressed != nullptr;

    delete[] sketch;
    delete[] decompressed;

    return ok;
}

int test_qjl_cpp_residual_estimator() {
    tq::kernel::QJLResidualEstimator est(512, 64, 4, 123);

    float res_a[512];
    float res_b[512];
    for (int i = 0; i < 512; i++) {
        res_a[i] = (float)(i % 50) / 50.0f;
        res_b[i] = (float)((i + 25) % 50) / 50.0f;
    }

    auto sketch_a = est.compress(res_a);
    auto sketch_b = est.compress(res_b);

    float est_dot = est.estimate_dot(sketch_a, sketch_b);
    est.free_sketch(&sketch_a);
    est.free_sketch(&sketch_b);

    return std::abs(est_dot) <= 1.0f;
}

int main() {
    fprintf(stderr, "\n=== TurboQuant Native Kernel Test Suite ===\n\n");

    TEST(test_rotation_init_shutdown);
    TEST(test_rotation_single_vector);
    TEST(test_rotation_cpp_batch);
    TEST(test_quantizer_init);
    TEST(test_quantizer_encode_decode);
    TEST(test_quantizer_cpp_beta);
    TEST(test_qjl_init_shutdown);
    TEST(test_qjl_compress);
    TEST(test_qjl_cpp_projector);
    TEST(test_qjl_cpp_sketch);
    TEST(test_qjl_cpp_residual_estimator);

    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "Passed: %d\n", g_passed);
    fprintf(stderr, "Failed: %d\n", g_failed);
    fprintf(stderr, "Total:  %d\n\n", g_passed + g_failed);

    return g_failed > 0 ? 1 : 0;
}