/**
 * TurboQuant OpenCL CLI — sidecar entry point.
 * Commands: probe, benchmark, mse-score, qjl-score, value-dequant, fused-attention
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "../include/tq_repo_paths.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// Forward declaration from benchmark module
namespace tq { int run_benchmark(int argc, char* argv[]); }

namespace {
std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out << buf;
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

bool approx_equal(float a, float b, float rel_tol = 1e-3f, float abs_tol = 1e-6f) {
    return std::fabs(a - b) <= rel_tol * std::fabs(b) + abs_tol;
}

bool load_self_test_kernels(const std::string& kernel_dir) {
    const std::string build_opts = tq::get_default_build_opts();
    return
        tq::load_kernel(kernel_dir + "/tq_mse_score.cl", "tq_mse_score", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_mse_score.cl", "tq_mse_score_tiled", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_qjl_score.cl", "tq_qjl_score", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_qjl_score.cl", "tq_qjl_score_tiled", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_value_dequant.cl", "tq_value_dequant", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_value_dequant.cl", "tq_value_weighted_accum", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_attention_logits.cl", "tq_attention_logits", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_attention_apply_values.cl", "tq_attention_apply_values", build_opts) == tq::TqStatus::OK;
}

void print_json_result(const std::string& payload) {
    std::printf("%s\n", payload.c_str());
}

int print_self_test_error(const char* command, const std::string& reason) {
    print_json_result(
        std::string("{\"status\":\"error\",\"command\":\"") + command +
        "\",\"reason\":\"" + json_escape(reason) + "\"}"
    );
    return 1;
}

int run_mse_self_test() {
    const int tokens = 4, dim = 8, bits = 3;
    float centroids[8] = {-0.8f, -0.5f, -0.2f, 0.0f, 0.1f, 0.3f, 0.6f, 0.9f};
    float query[8] = {0.1f, 0.2f, 0.3f, 0.4f, -0.1f, -0.2f, -0.3f, -0.4f};
    float norms[4] = {1.0f, 1.5f, 0.8f, 2.0f};
    int packed_stride = (dim * bits + 7) / 8;
    std::vector<uint8_t> packed(tokens * packed_stride, 0);
    for (int j = 0; j < dim; j++) {
        int bit_pos = j * bits;
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        packed[1 * packed_stride + byte_idx] |= (uint8_t)(7 << bit_off);
        if (bit_off + bits > 8)
            packed[1 * packed_stride + byte_idx + 1] |= (uint8_t)(7 >> (8 - bit_off));
    }

    tq::MseScoreInput input{};
    input.packed_indices = packed.data();
    input.norms = norms;
    input.query_rotated = query;
    input.centroids = centroids;
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.packed_stride = packed_stride;

    float scores_gpu[4] = {};
    float scores_cpu[4] = {};
    auto status = tq::mse_score_opencl(input, scores_gpu);
    tq::mse_score_cpu_reference(input, scores_cpu);
    bool parity = true;
    for (int i = 0; i < tokens; i++) parity = parity && approx_equal(scores_gpu[i], scores_cpu[i]);

    std::ostringstream out;
    out << "{\"status\":\"" << (status == tq::TqStatus::OK && parity ? "ok" : "error")
        << "\",\"command\":\"mse-score\",\"parity\":" << (parity ? "true" : "false")
        << ",\"scores\":[";
    for (int i = 0; i < tokens; i++) {
        if (i) out << ",";
        out << scores_gpu[i];
    }
    out << "]}";
    print_json_result(out.str());
    return (status == tq::TqStatus::OK && parity) ? 0 : 1;
}

int run_qjl_self_test() {
    const int tokens = 4;
    const int projections = 64;
    const int proj_words = (projections + 31) / 32;
    uint32_t query_signs[proj_words] = {0xAAAAAAAAu, 0x13579BDFu};
    uint32_t residual_signs[tokens * proj_words] = {
        0xAAAAAAAAu, 0x13579BDFu,
        0x55555555u, 0x2468ACE0u,
        0xFFFFFFFFu, 0x00000000u,
        0x0F0F0F0Fu, 0xF0F0F0F0u,
    };
    float residual_norms[tokens] = {1.0f, 0.5f, 0.25f, 0.75f};
    float scores_gpu[tokens] = {0.1f, -0.2f, 0.3f, -0.4f};
    float scores_cpu[tokens] = {0.1f, -0.2f, 0.3f, -0.4f};

    tq::QjlScoreInput gpu_input{};
    gpu_input.query_signs = query_signs;
    gpu_input.residual_signs = residual_signs;
    gpu_input.residual_norms = residual_norms;
    gpu_input.base_scores = scores_gpu;
    gpu_input.tokens = tokens;
    gpu_input.projections = projections;
    gpu_input.qjl_scale = 0.125f;

    tq::QjlScoreInput cpu_input = gpu_input;
    cpu_input.base_scores = scores_cpu;

    auto status = tq::qjl_score_opencl(gpu_input);
    tq::qjl_score_cpu_reference(cpu_input);
    bool parity = true;
    for (int i = 0; i < tokens; i++) parity = parity && approx_equal(scores_gpu[i], scores_cpu[i]);

    std::ostringstream out;
    out << "{\"status\":\"" << (status == tq::TqStatus::OK && parity ? "ok" : "error")
        << "\",\"command\":\"qjl-score\",\"parity\":" << (parity ? "true" : "false")
        << ",\"scores\":[";
    for (int i = 0; i < tokens; i++) {
        if (i) out << ",";
        out << scores_gpu[i];
    }
    out << "]}";
    print_json_result(out.str());
    return (status == tq::TqStatus::OK && parity) ? 0 : 1;
}

int run_value_dequant_self_test() {
    const int tokens = 2, dim = 8, bits = 4, group_size = 4;
    const int packed_stride = (dim * bits + 7) / 8;
    const int n_groups = (dim + group_size - 1) / group_size;
    uint8_t packed[tokens * packed_stride] = {0x10, 0x32, 0x54, 0x76, 0x89, 0xAB, 0xCD, 0xEF};
    float scales[tokens * n_groups] = {0.25f, 0.50f, 0.75f, 1.00f};
    float zeros[tokens * n_groups] = {-1.0f, 0.5f, 1.0f, -0.5f};
    float output_gpu[tokens * dim] = {};
    float output_cpu[tokens * dim] = {};

    tq::ValueDequantInput input{};
    input.packed_values = packed;
    input.scales = scales;
    input.zeros = zeros;
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.group_size = group_size;

    auto status = tq::value_dequant_opencl(input, output_gpu);
    tq::value_dequant_cpu_reference(input, output_cpu);
    bool parity = true;
    for (int i = 0; i < tokens * dim; i++) parity = parity && approx_equal(output_gpu[i], output_cpu[i]);

    std::ostringstream out;
    out << "{\"status\":\"" << (status == tq::TqStatus::OK && parity ? "ok" : "error")
        << "\",\"command\":\"value-dequant\",\"parity\":" << (parity ? "true" : "false")
        << ",\"values\":[";
    for (int i = 0; i < tokens * dim; i++) {
        if (i) out << ",";
        out << output_gpu[i];
    }
    out << "]}";
    print_json_result(out.str());
    return (status == tq::TqStatus::OK && parity) ? 0 : 1;
}

int run_fused_attention_self_test() {
    const int tokens = 4, dim = 8, bits = 4, group_size = 4;
    const int packed_stride = (dim * bits + 7) / 8;
    const int n_groups = (dim + group_size - 1) / group_size;
    const int projections = dim * 2;
    const int proj_words = (projections + 31) / 32;

    float query[dim] = {0.25f, -0.15f, 0.10f, 0.05f, -0.20f, 0.30f, -0.35f, 0.40f};
    float norms[tokens] = {1.0f, 0.8f, 1.2f, 0.9f};
    float centroids[1 << bits];
    for (int i = 0; i < (1 << bits); i++) centroids[i] = -1.0f + 0.125f * i;
    uint8_t key_packed[tokens * packed_stride] = {0x10, 0x32, 0x54, 0x76, 0x89, 0xAB, 0xCD, 0xEF,
                                                  0x01, 0x23, 0x45, 0x67, 0x98, 0xBA, 0xDC, 0xFE};
    uint32_t query_signs[proj_words] = {0xAAAAAAAAu};
    uint32_t residual_signs[tokens * proj_words] = {0xAAAAAAAAu, 0x55555555u, 0xF0F0F0F0u, 0x0F0F0F0Fu};
    float residual_norms[tokens] = {1.0f, 0.5f, 0.25f, 0.75f};
    uint8_t value_packed[tokens * packed_stride] = {0xFE, 0xDC, 0xBA, 0x98, 0x67, 0x45, 0x23, 0x01,
                                                    0xEF, 0xCD, 0xAB, 0x89, 0x76, 0x54, 0x32, 0x10};
    float value_scales[tokens * n_groups] = {0.25f, 0.5f, 0.4f, 0.8f, 0.35f, 0.7f, 0.45f, 0.9f};
    float value_zeros[tokens * n_groups] = {0.0f, -0.1f, 0.2f, -0.2f, 0.1f, -0.3f, 0.3f, -0.4f};
    float output_gpu[dim] = {};
    float output_cpu[dim] = {};

    tq::FusedAttentionInput gpu_input{};
    gpu_input.mse.packed_indices = key_packed;
    gpu_input.mse.norms = norms;
    gpu_input.mse.query_rotated = query;
    gpu_input.mse.centroids = centroids;
    gpu_input.mse.tokens = tokens;
    gpu_input.mse.dim = dim;
    gpu_input.mse.bits = bits;
    gpu_input.mse.packed_stride = packed_stride;
    gpu_input.qjl.query_signs = query_signs;
    gpu_input.qjl.residual_signs = residual_signs;
    gpu_input.qjl.residual_norms = residual_norms;
    gpu_input.qjl.base_scores = nullptr;
    gpu_input.qjl.tokens = tokens;
    gpu_input.qjl.projections = projections;
    gpu_input.qjl.qjl_scale = 0.125f;
    gpu_input.value.packed_values = value_packed;
    gpu_input.value.scales = value_scales;
    gpu_input.value.zeros = value_zeros;
    gpu_input.value.tokens = tokens;
    gpu_input.value.dim = dim;
    gpu_input.value.bits = bits;
    gpu_input.value.group_size = group_size;
    gpu_input.sm_scale = 1.0f / std::sqrt((float)dim);
    gpu_input.output = output_gpu;

    tq::FusedAttentionInput cpu_input = gpu_input;
    cpu_input.output = output_cpu;

    auto status = tq::fused_attention_tiled_opencl(gpu_input);
    tq::fused_attention_cpu_reference(cpu_input);

    double dot = 0.0, norm_gpu = 0.0, norm_cpu = 0.0;
    for (int i = 0; i < dim; i++) {
        dot += (double)output_gpu[i] * (double)output_cpu[i];
        norm_gpu += (double)output_gpu[i] * (double)output_gpu[i];
        norm_cpu += (double)output_cpu[i] * (double)output_cpu[i];
    }
    double cosine = (norm_gpu > 0.0 && norm_cpu > 0.0) ? dot / (std::sqrt(norm_gpu) * std::sqrt(norm_cpu)) : 0.0;
    bool parity = cosine > 0.95;

    std::ostringstream out;
    out << "{\"status\":\"" << (status == tq::TqStatus::OK && parity ? "ok" : "error")
        << "\",\"command\":\"fused-attention\",\"parity\":" << (parity ? "true" : "false")
        << ",\"cosine\":" << cosine << ",\"values\":[";
    for (int i = 0; i < dim; i++) {
        if (i) out << ",";
        out << output_gpu[i];
    }
    out << "]}";
    print_json_result(out.str());
    return (status == tq::TqStatus::OK && parity) ? 0 : 1;
}
}

static void print_json_probe(const tq::ProbeResult& r) {
    printf("{\n");
    printf("  \"available\": %s,\n", r.available ? "true" : "false");
    printf("  \"platformCount\": %d,\n", r.platform_count);
    printf("  \"deviceCount\": %d,\n", r.device_count);
    printf("  \"devices\": [");
    for (size_t i = 0; i < r.devices.size(); i++) {
        auto& d = r.devices[i];
        if (i > 0) printf(",");
        printf("\n    {\"name\":\"%s\",\"vendor\":\"%s\",\"version\":\"%s\","
               "\"hasFp16\":%s,\"hasSubgroups\":%s,\"hasIlProgram\":%s,"
               "\"hasSvm\":%s,\"hasSvmCoarse\":%s,\"hasSvmFine\":%s,\"hasSvmAtomics\":%s,"
               "\"isAdreno\":%s,"
               "\"globalMemBytes\":%llu,\"computeUnits\":%u}",
               json_escape(d.name).c_str(), json_escape(d.vendor).c_str(), json_escape(d.version).c_str(),
               d.has_fp16 ? "true" : "false",
               d.has_subgroups ? "true" : "false",
               d.has_il_program ? "true" : "false",
               d.has_svm ? "true" : "false",
               d.has_svm_coarse ? "true" : "false",
               d.has_svm_fine ? "true" : "false",
               d.has_svm_atomics ? "true" : "false",
               d.is_adreno ? "true" : "false",
               (unsigned long long)d.global_mem_bytes, d.compute_units);
    }
    printf("\n  ],\n");
    printf("  \"recommendedBackend\": \"%s\",\n", r.recommended_backend.c_str());
    printf("  \"probeTimeMs\": %.2f,\n", r.probe_time_ms);
    printf("  \"warnings\": [");
    for (size_t i = 0; i < r.warnings.size(); i++) {
        if (i > 0) printf(",");
        printf("\"%s\"", json_escape(r.warnings[i]).c_str());
    }
    printf("]\n}\n");
}

static void usage() {
    fprintf(stderr, "Usage: tq_opencl_cli <command> [options]\n");
    fprintf(stderr, "Commands:\n");
    fprintf(stderr, "  probe                              Detect OpenCL platform/devices\n");
    fprintf(stderr, "  benchmark [--warmup N] [--iters N] [--autotune] Run profiled benchmark suite\n");
    fprintf(stderr, "  mse-score --self-test [--spirv|--source]       Compute MSE attention scores\n");
    fprintf(stderr, "  qjl-score --self-test [--spirv|--source]       Compute QJL correction scores\n");
    fprintf(stderr, "  value-dequant --self-test [--spirv|--source]   Dequantize values\n");
    fprintf(stderr, "  fused-attention --self-test [--spirv|--source] Full fused decode\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string cmd = argv[1];
    bool force_source = false;
    bool prefer_spirv = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--source") force_source = true;
        if (std::string(argv[i]) == "--spirv") prefer_spirv = true;
    }

    if (force_source) {
        setenv("TQ_OPENCL_FORCE_SOURCE", "1", 1);
    } else if (prefer_spirv) {
        unsetenv("TQ_OPENCL_FORCE_SOURCE");
    }

    if (cmd == "probe") {
        auto result = tq::probe_opencl();
        print_json_probe(result);
        return result.available ? 0 : 1;
    }

    if (cmd == "benchmark") {
        return tq::run_benchmark(argc - 2, argv + 2);
    }

    if (cmd == "mse-score") {
        if (argc >= 3 && std::string(argv[2]) == "--self-test") {
            std::string kernel_dir = tq::resolve_kernel_dir();
            if (kernel_dir.empty()) return print_self_test_error("mse-score", "kernel_dir_not_found");
            auto st = tq::init_context();
            if (st != tq::TqStatus::OK) return print_self_test_error("mse-score", "init_context_failed");
            if (!load_self_test_kernels(kernel_dir)) return print_self_test_error("mse-score", "kernel_load_failed");
            int rc = run_mse_self_test();
            tq::release_all_programs();
            tq::shutdown_context();
            return rc;
        }
        fprintf(stderr, "mse-score requires --self-test\n");
        return 2;
    }

    if (cmd == "qjl-score" || cmd == "value-dequant" || cmd == "fused-attention") {
        if (argc >= 3 && std::string(argv[2]) == "--self-test") {
            std::string kernel_dir = tq::resolve_kernel_dir();
            if (kernel_dir.empty()) return print_self_test_error(cmd.c_str(), "kernel_dir_not_found");
            auto st = tq::init_context();
            if (st != tq::TqStatus::OK) return print_self_test_error(cmd.c_str(), "init_context_failed");
            if (!load_self_test_kernels(kernel_dir)) return print_self_test_error(cmd.c_str(), "kernel_load_failed");
            int rc = 1;
            if (cmd == "qjl-score") rc = run_qjl_self_test();
            else if (cmd == "value-dequant") rc = run_value_dequant_self_test();
            else rc = run_fused_attention_self_test();
            tq::release_all_programs();
            tq::shutdown_context();
            return rc;
        }
        fprintf(stderr, "%s requires --self-test\n", cmd.c_str());
        return 2;
    }

    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    usage();
    return 1;
}
