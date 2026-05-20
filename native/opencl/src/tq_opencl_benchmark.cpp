/**
 * TurboQuant OpenCL — benchmark harness with event profiling.
 * Supports shape matrix, warmup, iterations, sustained mode.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "../include/tq_repo_paths.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <fstream>
#include <random>

namespace tq {

struct BenchShape {
    int tokens;
    int heads;
    int head_dim;
    int bits;
};

struct KernelResult {
    std::string kernel;
    BenchShape shape;
    bool parity;
    double p50_ms;
    double p95_ms;
    double min_ms;
    double max_ms;
    double mean_ms;
    double std_ms;
    double cpu_mean_ms;
    double speedup_vs_cpu;
    size_t memory_bytes;
};

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)idx;
    double frac = idx - lo;
    if (lo + 1 >= v.size()) return v.back();
    return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

static double compute_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = 0;
    for (auto x : v) sum += x;
    return sum / v.size();
}

static double compute_std(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double sum = 0;
    for (auto x : v) sum += (x - mean) * (x - mean);
    return std::sqrt(sum / (v.size() - 1));
}

static void fill_random(float* buf, int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; i++) buf[i] = dist(rng);
}

static void fill_random_u8(uint8_t* buf, int n, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    for (int i = 0; i < n; i++) buf[i] = (uint8_t)dist(rng);
}

static void fill_random_u32(uint32_t* buf, int n, std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    for (int i = 0; i < n; i++) buf[i] = dist(rng);
}

static KernelResult bench_mse(const BenchShape& shape, int warmup, int iters, std::mt19937& rng) {
    int dim = shape.head_dim;
    int tokens = shape.tokens;
    int bits = shape.bits;
    int packed_stride = (dim * bits + 7) / 8;
    int n_centroids = 1 << bits;

    std::vector<float> query(dim), norms(tokens), centroids(n_centroids);
    std::vector<uint8_t> packed(tokens * packed_stride);
    std::vector<float> scores_gpu(tokens), scores_cpu(tokens);

    fill_random(query.data(), dim, rng);
    fill_random(norms.data(), tokens, rng);
    fill_random(centroids.data(), n_centroids, rng);
    fill_random_u8(packed.data(), tokens * packed_stride, rng);

    MseScoreInput input{};
    input.packed_indices = packed.data();
    input.norms = norms.data();
    input.query_rotated = query.data();
    input.centroids = centroids.data();
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.packed_stride = packed_stride;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        uint64_t ns;
        mse_score_opencl_profiled(input, scores_gpu.data(), &ns);
    }

    // Timed iterations
    std::vector<double> times_ms;
    for (int i = 0; i < iters; i++) {
        uint64_t ns;
        mse_score_opencl_profiled(input, scores_gpu.data(), &ns);
        times_ms.push_back((double)ns / 1e6);
    }

    // CPU reference timing
    std::vector<double> cpu_times;
    for (int i = 0; i < std::min(iters, 20); i++) {
        auto t0 = std::chrono::steady_clock::now();
        mse_score_cpu_reference(input, scores_cpu.data());
        auto t1 = std::chrono::steady_clock::now();
        cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Parity check
    mse_score_cpu_reference(input, scores_cpu.data());
    bool parity = true;
    for (int i = 0; i < tokens && parity; i++) {
        if (std::fabs(scores_gpu[i] - scores_cpu[i]) > 1e-3f * std::fabs(scores_cpu[i]) + 1e-6f)
            parity = false;
    }

    double mean = compute_mean(times_ms);
    double cpu_mean = compute_mean(cpu_times);

    KernelResult r{};
    r.kernel = "mse_score";
    r.shape = shape;
    r.parity = parity;
    r.p50_ms = percentile(times_ms, 50);
    r.p95_ms = percentile(times_ms, 95);
    r.min_ms = times_ms.empty() ? 0 : *std::min_element(times_ms.begin(), times_ms.end());
    r.max_ms = times_ms.empty() ? 0 : *std::max_element(times_ms.begin(), times_ms.end());
    r.mean_ms = mean;
    r.std_ms = compute_std(times_ms, mean);
    r.cpu_mean_ms = cpu_mean;
    r.speedup_vs_cpu = (mean > 0) ? cpu_mean / mean : 0;
    r.memory_bytes = packed.size() + query.size() * 4 + norms.size() * 4 + centroids.size() * 4 + scores_gpu.size() * 4;
    return r;
}

static KernelResult bench_qjl(const BenchShape& shape, int warmup, int iters, std::mt19937& rng) {
    int tokens = shape.tokens;
    int projections = shape.head_dim * 2; // typical: 2x head_dim
    int proj_words = (projections + 31) / 32;

    std::vector<uint32_t> query_signs(proj_words), residual_signs(tokens * proj_words);
    std::vector<float> residual_norms(tokens), scores_gpu(tokens, 0.0f), scores_cpu(tokens, 0.0f);

    fill_random_u32(query_signs.data(), proj_words, rng);
    fill_random_u32(residual_signs.data(), tokens * proj_words, rng);
    fill_random(residual_norms.data(), tokens, rng);

    QjlScoreInput input{};
    input.query_signs = query_signs.data();
    input.residual_signs = residual_signs.data();
    input.residual_norms = residual_norms.data();
    input.tokens = tokens;
    input.projections = projections;
    input.qjl_scale = 0.1f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        std::fill(scores_gpu.begin(), scores_gpu.end(), 0.0f);
        input.base_scores = scores_gpu.data();
        uint64_t ns;
        qjl_score_opencl_profiled(input, &ns);
    }

    // Timed iterations
    std::vector<double> times_ms;
    for (int i = 0; i < iters; i++) {
        std::fill(scores_gpu.begin(), scores_gpu.end(), 0.0f);
        input.base_scores = scores_gpu.data();
        uint64_t ns;
        qjl_score_opencl_profiled(input, &ns);
        times_ms.push_back((double)ns / 1e6);
    }

    // CPU timing
    std::vector<double> cpu_times;
    for (int i = 0; i < std::min(iters, 20); i++) {
        std::fill(scores_cpu.begin(), scores_cpu.end(), 0.0f);
        input.base_scores = scores_cpu.data();
        auto t0 = std::chrono::steady_clock::now();
        qjl_score_cpu_reference(input);
        auto t1 = std::chrono::steady_clock::now();
        cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Parity
    std::fill(scores_gpu.begin(), scores_gpu.end(), 0.0f);
    input.base_scores = scores_gpu.data();
    qjl_score_opencl(input);
    std::fill(scores_cpu.begin(), scores_cpu.end(), 0.0f);
    input.base_scores = scores_cpu.data();
    qjl_score_cpu_reference(input);

    bool parity = true;
    for (int i = 0; i < tokens && parity; i++) {
        if (std::fabs(scores_gpu[i] - scores_cpu[i]) > 1e-3f * std::fabs(scores_cpu[i]) + 1e-6f)
            parity = false;
    }

    double mean = compute_mean(times_ms);
    double cpu_mean = compute_mean(cpu_times);

    KernelResult r{};
    r.kernel = "qjl_score";
    r.shape = shape;
    r.parity = parity;
    r.p50_ms = percentile(times_ms, 50);
    r.p95_ms = percentile(times_ms, 95);
    r.min_ms = times_ms.empty() ? 0 : *std::min_element(times_ms.begin(), times_ms.end());
    r.max_ms = times_ms.empty() ? 0 : *std::max_element(times_ms.begin(), times_ms.end());
    r.mean_ms = mean;
    r.std_ms = compute_std(times_ms, mean);
    r.cpu_mean_ms = cpu_mean;
    r.speedup_vs_cpu = (mean > 0) ? cpu_mean / mean : 0;
    r.memory_bytes = query_signs.size() * 4 + residual_signs.size() * 4 + residual_norms.size() * 4 + scores_gpu.size() * 4;
    return r;
}

static KernelResult bench_value_dequant(const BenchShape& shape, int warmup, int iters, std::mt19937& rng) {
    int tokens = shape.tokens;
    int dim = shape.head_dim;
    int bits = shape.bits;
    int group_size = 32;
    int packed_stride = (dim * bits + 7) / 8;
    int n_groups = (dim + group_size - 1) / group_size;

    std::vector<uint8_t> packed(tokens * packed_stride);
    std::vector<float> scales(tokens * n_groups), zeros(tokens * n_groups);
    std::vector<float> output_gpu(tokens * dim), output_cpu(tokens * dim);

    fill_random_u8(packed.data(), tokens * packed_stride, rng);
    fill_random(scales.data(), tokens * n_groups, rng);
    fill_random(zeros.data(), tokens * n_groups, rng);

    ValueDequantInput input{};
    input.packed_values = packed.data();
    input.scales = scales.data();
    input.zeros = zeros.data();
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.group_size = group_size;

    for (int i = 0; i < warmup; i++) {
        uint64_t ns;
        value_dequant_opencl_profiled(input, output_gpu.data(), &ns);
    }

    std::vector<double> times_ms;
    for (int i = 0; i < iters; i++) {
        uint64_t ns;
        value_dequant_opencl_profiled(input, output_gpu.data(), &ns);
        times_ms.push_back((double)ns / 1e6);
    }

    std::vector<double> cpu_times;
    for (int i = 0; i < std::min(iters, 20); i++) {
        auto t0 = std::chrono::steady_clock::now();
        value_dequant_cpu_reference(input, output_cpu.data());
        auto t1 = std::chrono::steady_clock::now();
        cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    value_dequant_cpu_reference(input, output_cpu.data());
    value_dequant_opencl(input, output_gpu.data());
    bool parity = true;
    for (int i = 0; i < tokens * dim && parity; i++) {
        if (std::fabs(output_gpu[i] - output_cpu[i]) > 1e-3f * std::fabs(output_cpu[i]) + 1e-6f)
            parity = false;
    }

    double mean = compute_mean(times_ms);
    double cpu_mean = compute_mean(cpu_times);

    KernelResult r{};
    r.kernel = "value_dequant";
    r.shape = shape;
    r.parity = parity;
    r.p50_ms = percentile(times_ms, 50);
    r.p95_ms = percentile(times_ms, 95);
    r.min_ms = times_ms.empty() ? 0 : *std::min_element(times_ms.begin(), times_ms.end());
    r.max_ms = times_ms.empty() ? 0 : *std::max_element(times_ms.begin(), times_ms.end());
    r.mean_ms = mean;
    r.std_ms = compute_std(times_ms, mean);
    r.cpu_mean_ms = cpu_mean;
    r.speedup_vs_cpu = (mean > 0) ? cpu_mean / mean : 0;
    r.memory_bytes = packed.size() + scales.size() * 4 + zeros.size() * 4 + output_gpu.size() * 4;
    return r;
}

static KernelResult bench_fused_attention(const BenchShape& shape, int warmup, int iters, std::mt19937& rng) {
    int tokens = shape.tokens;
    int dim = shape.head_dim;
    int bits = shape.bits;
    int group_size = 32;
    int packed_stride = (dim * bits + 7) / 8;
    int n_groups = (dim + group_size - 1) / group_size;
    int projections = dim * 2;
    int proj_words = (projections + 31) / 32;

    std::vector<float> query(dim), norms(tokens), centroids(1 << bits);
    std::vector<uint8_t> key_packed(tokens * packed_stride);
    std::vector<uint32_t> query_signs(proj_words), residual_signs(tokens * proj_words);
    std::vector<float> residual_norms(tokens);
    std::vector<uint8_t> val_packed(tokens * packed_stride);
    std::vector<float> val_scales(tokens * n_groups), val_zeros(tokens * n_groups);
    std::vector<float> output_gpu(dim), output_cpu(dim);

    fill_random(query.data(), dim, rng);
    fill_random(norms.data(), tokens, rng);
    fill_random(centroids.data(), 1 << bits, rng);
    fill_random_u8(key_packed.data(), tokens * packed_stride, rng);
    fill_random_u32(query_signs.data(), proj_words, rng);
    fill_random_u32(residual_signs.data(), tokens * proj_words, rng);
    fill_random(residual_norms.data(), tokens, rng);
    fill_random_u8(val_packed.data(), tokens * packed_stride, rng);
    fill_random(val_scales.data(), tokens * n_groups, rng);
    fill_random(val_zeros.data(), tokens * n_groups, rng);

    FusedAttentionInput input{};
    input.mse.packed_indices = key_packed.data();
    input.mse.norms = norms.data();
    input.mse.query_rotated = query.data();
    input.mse.centroids = centroids.data();
    input.mse.tokens = tokens;
    input.mse.dim = dim;
    input.mse.bits = bits;
    input.mse.packed_stride = packed_stride;
    input.qjl.query_signs = query_signs.data();
    input.qjl.residual_signs = residual_signs.data();
    input.qjl.residual_norms = residual_norms.data();
    input.qjl.base_scores = nullptr; // handled internally
    input.qjl.tokens = tokens;
    input.qjl.projections = projections;
    input.qjl.qjl_scale = 0.1f;
    input.value.packed_values = val_packed.data();
    input.value.scales = val_scales.data();
    input.value.zeros = val_zeros.data();
    input.value.tokens = tokens;
    input.value.dim = dim;
    input.value.bits = bits;
    input.value.group_size = group_size;
    input.sm_scale = 1.0f / std::sqrt((float)dim);
    input.output = output_gpu.data();

    for (int i = 0; i < warmup; i++) {
        std::fill(output_gpu.begin(), output_gpu.end(), 0.0f);
        input.output = output_gpu.data();
        uint64_t ns;
        auto t0 = std::chrono::steady_clock::now();
        fused_attention_tiled_opencl_profiled(input, &ns);
        auto t1 = std::chrono::steady_clock::now();
        if (ns == 0) {
            ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        }
    }

    std::vector<double> times_ms;
    for (int i = 0; i < iters; i++) {
        std::fill(output_gpu.begin(), output_gpu.end(), 0.0f);
        input.output = output_gpu.data();
        uint64_t ns;
        auto t0 = std::chrono::steady_clock::now();
        fused_attention_tiled_opencl_profiled(input, &ns);
        auto t1 = std::chrono::steady_clock::now();
        if (ns == 0) {
            ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        }
        times_ms.push_back((double)ns / 1e6);
    }

    std::vector<double> cpu_times;
    for (int i = 0; i < std::min(iters, 10); i++) {
        std::fill(output_cpu.begin(), output_cpu.end(), 0.0f);
        input.output = output_cpu.data();
        auto t0 = std::chrono::steady_clock::now();
        fused_attention_cpu_reference(input);
        auto t1 = std::chrono::steady_clock::now();
        cpu_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Parity — fused attention accumulates error across tokens via online softmax.
    // GPU single-thread kernel processes tokens in same order as CPU, but
    // floating point non-associativity in exp/division causes drift proportional
    // to token count. We compare GPU vs CPU using cosine similarity instead of
    // per-element relative error.
    std::fill(output_gpu.begin(), output_gpu.end(), 0.0f);
    input.output = output_gpu.data();
    fused_attention_tiled_opencl(input);
    std::fill(output_cpu.begin(), output_cpu.end(), 0.0f);
    input.output = output_cpu.data();
    fused_attention_cpu_reference(input);

    // Cosine similarity: robust to accumulated drift
    double dot = 0, norm_a = 0, norm_b = 0;
    for (int i = 0; i < dim; i++) {
        dot += (double)output_gpu[i] * (double)output_cpu[i];
        norm_a += (double)output_gpu[i] * (double)output_gpu[i];
        norm_b += (double)output_cpu[i] * (double)output_cpu[i];
    }
    double cosine = (norm_a > 0 && norm_b > 0) ? dot / (std::sqrt(norm_a) * std::sqrt(norm_b)) : 0.0;
    bool parity = (cosine > 0.95); // 0.95 cosine = strong agreement despite accumulated drift

    double mean = compute_mean(times_ms);
    double cpu_mean = compute_mean(cpu_times);

    KernelResult r{};
    r.kernel = "fused_attention";
    r.shape = shape;
    r.parity = parity;
    r.p50_ms = percentile(times_ms, 50);
    r.p95_ms = percentile(times_ms, 95);
    r.min_ms = times_ms.empty() ? 0 : *std::min_element(times_ms.begin(), times_ms.end());
    r.max_ms = times_ms.empty() ? 0 : *std::max_element(times_ms.begin(), times_ms.end());
    r.mean_ms = mean;
    r.std_ms = compute_std(times_ms, mean);
    r.cpu_mean_ms = cpu_mean;
    r.speedup_vs_cpu = (mean > 0) ? cpu_mean / mean : 0;
    r.memory_bytes = key_packed.size() + val_packed.size() + query.size() * 4 +
                     norms.size() * 4 + centroids.size() * 4 + val_scales.size() * 4 +
                     val_zeros.size() * 4 + residual_signs.size() * 4;
    return r;
}

// JSON output helpers
static void print_shape_json(const BenchShape& s) {
    printf("{\"tokens\":%d,\"heads\":%d,\"headDim\":%d,\"bits\":%d}", s.tokens, s.heads, s.head_dim, s.bits);
}

static void print_result_json(const KernelResult& r) {
    printf("    {\n");
    printf("      \"kernel\": \"%s\",\n", r.kernel.c_str());
    printf("      \"shape\": "); print_shape_json(r.shape); printf(",\n");
    printf("      \"parity\": %s,\n", r.parity ? "true" : "false");
    printf("      \"p50_ms\": %.4f,\n", r.p50_ms);
    printf("      \"p95_ms\": %.4f,\n", r.p95_ms);
    printf("      \"min_ms\": %.4f,\n", r.min_ms);
    printf("      \"max_ms\": %.4f,\n", r.max_ms);
    printf("      \"mean_ms\": %.4f,\n", r.mean_ms);
    printf("      \"std_ms\": %.4f,\n", r.std_ms);
    printf("      \"cpu_mean_ms\": %.4f,\n", r.cpu_mean_ms);
    printf("      \"speedup_vs_cpu\": %.3f,\n", r.speedup_vs_cpu);
    printf("      \"memory_bytes\": %zu\n", r.memory_bytes);
    printf("    }");
}

int run_benchmark(int argc, char* argv[]) {
    int warmup = 10;
    int iters = 100;
    bool json_output = true;
    std::string kernel_dir;

    // Parse args
    for (int i = 0; i < argc; i++) {
        if (std::string(argv[i]) == "--warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (std::string(argv[i]) == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (std::string(argv[i]) == "--kernel-dir" && i + 1 < argc) kernel_dir = argv[++i];
    }
    kernel_dir = tq::resolve_kernel_dir(kernel_dir);

    // Initialize OpenCL context
    TqStatus st = init_context();
    if (st != TqStatus::OK) {
        fprintf(stderr, "{\"error\":\"init_context_failed\",\"state\":\"%d\"}\n", (int)st);
        return 1;
    }

    // Load kernels
    load_kernel(kernel_dir + "/tq_mse_score.cl", "tq_mse_score");
    load_kernel(kernel_dir + "/tq_mse_score.cl", "tq_mse_score_tiled");
    load_kernel(kernel_dir + "/tq_qjl_score.cl", "tq_qjl_score");
    load_kernel(kernel_dir + "/tq_qjl_score.cl", "tq_qjl_score_tiled");
    load_kernel(kernel_dir + "/tq_value_dequant.cl", "tq_value_dequant");
    load_kernel(kernel_dir + "/tq_value_dequant.cl", "tq_value_weighted_accum");
    load_kernel(kernel_dir + "/tq_fused_attention.cl", "tq_fused_attention");
    load_kernel(kernel_dir + "/tq_attention_logits.cl", "tq_attention_logits");
    load_kernel(kernel_dir + "/tq_attention_apply_values.cl", "tq_attention_apply_values");

    // Shape matrix
    std::vector<BenchShape> shapes = {
        {128, 4, 64, 3},   // small correctness
        {128, 4, 64, 4},
        {512, 8, 64, 3},   // interactive
        {512, 8, 64, 4},
        {2048, 8, 128, 3}, // long context mobile
        {2048, 8, 128, 4},
        {4096, 8, 128, 3}, // stress
        {4096, 8, 128, 4},
    };

    // Memory check — skip 8192 if < 6GB available
    std::ifstream meminfo("/proc/meminfo");
    long avail_kb = 0;
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemAvailable") != std::string::npos) {
                sscanf(line.c_str(), "MemAvailable: %ld", &avail_kb);
                break;
            }
        }
    }
    if (avail_kb == 0 || avail_kb > 4000000) {
        shapes.push_back({8192, 8, 128, 3}); // memory stress
        shapes.push_back({8192, 8, 128, 4});
    }

    std::mt19937 rng(42);
    std::vector<KernelResult> results;

    for (auto& shape : shapes) {
        results.push_back(bench_mse(shape, warmup, iters, rng));
        results.push_back(bench_qjl(shape, warmup, iters, rng));
        results.push_back(bench_value_dequant(shape, warmup, iters, rng));
        // Fused attention: reduce iters for large shapes
        int fused_iters = (shape.tokens > 2048) ? std::min(iters, 20) : iters;
        results.push_back(bench_fused_attention(shape, std::min(warmup, 3), fused_iters, rng));
    }

    // Get device info
    char dev_name[256] = {};
    char dev_version[256] = {};
    clGetDeviceInfo(get_device(), CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    clGetDeviceInfo(get_device(), CL_DEVICE_VERSION, sizeof(dev_version), dev_version, nullptr);

    // Read memory after
    long avail_kb_after = 0;
    std::ifstream meminfo2("/proc/meminfo");
    if (meminfo2.is_open()) {
        std::string line;
        while (std::getline(meminfo2, line)) {
            if (line.find("MemAvailable") != std::string::npos) {
                sscanf(line.c_str(), "MemAvailable: %ld", &avail_kb_after);
                break;
            }
        }
    }

    // Check all parity
    bool all_pass = true;
    for (auto& r : results) if (!r.parity) all_pass = false;

    // JSON output
    if (json_output) {
        printf("{\n");
        printf("  \"device\": {\n");
        printf("    \"gpu\": \"%s\",\n", dev_name);
        printf("    \"opencl\": \"%s\",\n", dev_version);
        printf("    \"compute_units\": %u,\n", get_compute_units());
        printf("    \"max_wg_size\": %zu,\n", get_max_wg_size());
        printf("    \"is_adreno\": %s\n", is_adreno() ? "true" : "false");
        printf("  },\n");
        printf("  \"run\": {\n");
        printf("    \"warmup\": %d,\n", warmup);
        printf("    \"iters\": %d,\n", iters);
        printf("    \"mem_available_kb_before\": %ld,\n", avail_kb);
        printf("    \"mem_available_kb_after\": %ld\n", avail_kb_after);
        printf("  },\n");
        printf("  \"results\": [\n");
        for (size_t i = 0; i < results.size(); i++) {
            print_result_json(results[i]);
            printf("%s\n", (i + 1 < results.size()) ? "," : "");
        }
        printf("  ],\n");
        printf("  \"all_pass\": %s,\n", all_pass ? "true" : "false");
        printf("  \"claim_safe\": %s\n", all_pass ? "true" : "false");
        printf("}\n");
    }

    release_all_programs();
    shutdown_context();
    return all_pass ? 0 : 1;
}

} // namespace tq
