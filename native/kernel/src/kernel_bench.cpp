/**
 * TurboQuant Native Kernel — Benchmark Suite
 */

#include "../include/tq_rotation_kernel.h"
#include <CL/cl.h>
#include "../include/tq_quantizer_kernel.h"
#include "../include/tq_qjl_kernel.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

static uint64_t get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;
}

static void bench_rotation(uint32_t dims, uint32_t iterations) {
    fprintf(stderr, "\n[Bench] Rotation %ux%u\n", dims, iterations);

    tq_rotation_kernel_t kernel;
    tq_rotation_init(&kernel, dims, 42, TQ_ROTATION_MODE_FWHT_SIGN);

    float* input; posix_memalign((void**)&input, 32, dims * sizeof(float));
    float* output; posix_memalign((void**)&output, 32, dims * sizeof(float));

    for (uint32_t i = 0; i < dims; i++) input[i] = (float)(i % 100) / 100.0f;

    uint64_t start = get_time_us();
    for (uint32_t iter = 0; iter < iterations; iter++) {
        tq_rotation_execute(&kernel, input, output, 1);
    }
    uint64_t elapsed = get_time_us() - start;

    fprintf(stderr, "  Total:   %llu us\n", (unsigned long long)elapsed);
    fprintf(stderr, "  Per call: %.2f us\n", (double)elapsed / iterations);
    fprintf(stderr, "  Throughput: %.0f vecs/sec\n", iterations * 1000000.0 / elapsed);

    free(input);
    free(output);
    tq_rotation_shutdown(&kernel);
}

static void bench_quantizer(uint32_t dims, uint8_t bits, uint32_t iterations) {
    fprintf(stderr, "\n[Bench] Quantizer %ux%u-bit x%u\n", dims, bits, iterations);

    tq_quantizer_kernel_t q;
    tq_quantizer_init(&q, dims, bits);

    float* values; posix_memalign((void**)&values, 32, dims * sizeof(float));
    uint8_t* packed = (uint8_t*)malloc(((dims * bits) + 7) / 8);
    float* decoded; posix_memalign((void**)&decoded, 32, dims * sizeof(float));

    for (uint32_t i = 0; i < dims; i++) values[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;

    uint64_t start = get_time_us();
    for (uint32_t iter = 0; iter < iterations; iter++) {
        tq_quantizer_encode(&q, values, packed, dims);
    }
    uint64_t encode_time = get_time_us() - start;

    start = get_time_us();
    for (uint32_t iter = 0; iter < iterations; iter++) {
        tq_quantizer_decode(&q, packed, decoded, dims);
    }
    uint64_t decode_time = get_time_us() - start;

    fprintf(stderr, "  Encode total: %llu us (%.2f us/call)\n",
            (unsigned long long)encode_time, (double)encode_time / iterations);
    fprintf(stderr, "  Decode total: %llu us (%.2f us/call)\n",
            (unsigned long long)decode_time, (double)decode_time / iterations);

    free(values);
    free(packed);
    free(decoded);
    tq_quantizer_shutdown(&q);
}

static void bench_qjl(uint32_t orig_dims, uint32_t target_dims, uint32_t iterations) {
    fprintf(stderr, "\n[Bench] QJL %ux%u x%u\n", orig_dims, target_dims, iterations);

    tq_qjl_kernel_t qjl;
    tq_qjl_init(&qjl, orig_dims, target_dims, 42, 4, 1);

    float* residual; posix_memalign((void**)&residual, 32, orig_dims * sizeof(float));
    uint8_t* sketch = (uint8_t*)malloc(((target_dims * 4) + 7) / 8);

    for (uint32_t i = 0; i < orig_dims; i++) residual[i] = (float)(i % 50) / 50.0f;

    uint64_t start = get_time_us();
    for (uint32_t iter = 0; iter < iterations; iter++) {
        tq_qjl_compress(&qjl, residual, sketch);
    }
    uint64_t elapsed = get_time_us() - start;

    fprintf(stderr, "  Total:   %llu us\n", (unsigned long long)elapsed);
    fprintf(stderr, "  Per call: %.2f us\n", (double)elapsed / iterations);
    fprintf(stderr, "  Throughput: %.0f comp/sec\n", iterations * 1000000.0 / elapsed);

    free(residual);
    free(sketch);
    tq_qjl_shutdown(&qjl);
}

static void bench_cpp_rotation_batch() {
    fprintf(stderr, "\n[Bench] C++ Rotation batch (512 dims, 10000 vecs)\n");

    tq::kernel::RotationKernel rot(512, 123);

    float* input = new float[512 * 10000];
    float* output = new float[512 * 10000];

    for (size_t i = 0; i < 512 * 10000; i++) input[i] = (float)(i % 100) / 100.0f;

    uint64_t start = get_time_us();
    rot.rotate_batch(input, output, 10000);
    uint64_t elapsed = get_time_us() - start;

    fprintf(stderr, "  Total:   %llu us\n", (unsigned long long)elapsed);
    fprintf(stderr, "  Per vec: %.2f us\n", (double)elapsed / 10000);
    fprintf(stderr, "  Throughput: %.0f vecs/sec\n", 10000 * 1000000.0 / elapsed);

    delete[] input;
    delete[] output;
}

static void bench_cpp_beta_codebook() {
    fprintf(stderr, "\n[Bench] C++ Beta Codebook (1024 dims, 4-bit, 10000 quantize)\n");

    tq::kernel::BetaCodebook cb(1024, 4);
    cb.compute();

    float mse = 0.0f;
    float samples[1024];
    for (int i = 0; i < 1024; i++) samples[i] = (float)(i % 100) / 100.0f * 2.0f - 1.0f;

    uint64_t start = get_time_us();
    for (int iter = 0; iter < 10000; iter++) {
        for (int i = 0; i < 1024; i++) {
            cb.quantize_index(samples[i]);
        }
    }
    uint64_t elapsed = get_time_us() - start;

    fprintf(stderr, "  Total:   %llu us\n", (unsigned long long)elapsed);
    fprintf(stderr, "  Per quantize: %.2f us\n", (double)elapsed / 10240000);
    mse = cb.compute_mse(samples, 1024);
    fprintf(stderr, "  MSE: %.6f\n", mse);
}

static void bench_cpp_qjl_residual() {
    fprintf(stderr, "\n[Bench] C++ QJL Residual Estimator (512->64, 1000 dot estimates)\n");

    tq::kernel::QJLResidualEstimator est(512, 64, 4, 123);

    float res_a[512];
    float res_b[512];
    for (int i = 0; i < 512; i++) {
        res_a[i] = (float)(i % 50) / 50.0f;
        res_b[i] = (float)((i + 25) % 50) / 50.0f;
    }

    uint64_t start = get_time_us();
    for (int iter = 0; iter < 1000; iter++) {
        est.estimate_dot(res_a, res_b);
    }
    uint64_t elapsed = get_time_us() - start;

    fprintf(stderr, "  Total:   %llu us\n", (unsigned long long)elapsed);
    fprintf(stderr, "  Per estimate: %.2f us\n", (double)elapsed / 1000);
    fprintf(stderr, "  Throughput: %.0f est/sec\n", 1000 * 1000000.0 / elapsed);
}

int main(int argc, char** argv) {
    fprintf(stderr, "\n=== TurboQuant Native Kernel Benchmark ===\n");
    fprintf(stderr, "LLVM Clang %s\n\n", __VERSION__);

    // Rotation benchmarks
    bench_rotation(256, 1000);
    bench_rotation(512, 1000);
    bench_rotation(1024, 1000);

    // Quantizer benchmarks
    bench_quantizer(256, 4, 1000);
    bench_quantizer(512, 4, 1000);
    bench_quantizer(1024, 4, 1000);

    // QJL benchmarks
    bench_qjl(512, 64, 1000);
    bench_qjl(1024, 64, 1000);

    // C++ class benchmarks
    bench_cpp_rotation_batch();
    bench_cpp_beta_codebook();
    bench_cpp_qjl_residual();

    fprintf(stderr, "\n=== Benchmark Complete ===\n\n");
    return 0;
}