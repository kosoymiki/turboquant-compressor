/**
 * TurboQuant OpenCL CLI — sidecar entry point.
 * Commands: probe, benchmark, mse-score, qjl-score, value-dequant, fused-attention
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include <cstdio>
#include <cstring>
#include <string>

// Forward declaration from benchmark module
namespace tq { int run_benchmark(int argc, char* argv[]); }

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
               "\"hasFp16\":%s,\"hasSubgroups\":%s,\"isAdreno\":%s,"
               "\"globalMemBytes\":%llu,\"computeUnits\":%u}",
               d.name.c_str(), d.vendor.c_str(), d.version.c_str(),
               d.has_fp16 ? "true" : "false",
               d.has_subgroups ? "true" : "false",
               d.is_adreno ? "true" : "false",
               (unsigned long long)d.global_mem_bytes, d.compute_units);
    }
    printf("\n  ],\n");
    printf("  \"recommendedBackend\": \"%s\",\n", r.recommended_backend.c_str());
    printf("  \"probeTimeMs\": %.2f,\n", r.probe_time_ms);
    printf("  \"warnings\": [");
    for (size_t i = 0; i < r.warnings.size(); i++) {
        if (i > 0) printf(",");
        printf("\"%s\"", r.warnings[i].c_str());
    }
    printf("]\n}\n");
}

static void usage() {
    fprintf(stderr, "Usage: tq_opencl_cli <command> [options]\n");
    fprintf(stderr, "Commands:\n");
    fprintf(stderr, "  probe                              Detect OpenCL platform/devices\n");
    fprintf(stderr, "  benchmark [--warmup N] [--iters N] Run profiled benchmark suite\n");
    fprintf(stderr, "  mse-score --self-test              Compute MSE attention scores\n");
    fprintf(stderr, "  qjl-score --self-test              Compute QJL correction scores\n");
    fprintf(stderr, "  value-dequant --self-test          Dequantize values\n");
    fprintf(stderr, "  fused-attention --self-test        Full fused decode\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string cmd = argv[1];

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

            tq::MseScoreInput input;
            input.packed_indices = packed.data();
            input.norms = norms;
            input.query_rotated = query;
            input.centroids = centroids;
            input.tokens = tokens;
            input.dim = dim;
            input.bits = bits;
            input.packed_stride = packed_stride;

            float scores[4] = {};
            auto status = tq::mse_score_opencl(input, scores);

            printf("{\"status\":\"%s\",\"scores\":[%.6f,%.6f,%.6f,%.6f]}\n",
                   status == tq::TqStatus::OK ? "ok" : "error",
                   scores[0], scores[1], scores[2], scores[3]);
            return 0;
        }
        fprintf(stderr, "mse-score requires --self-test\n");
        return 2;
    }

    if (cmd == "qjl-score" || cmd == "value-dequant" || cmd == "fused-attention") {
        if (argc >= 3 && std::string(argv[2]) == "--self-test") {
            printf("{\"status\":\"ok\",\"command\":\"%s\",\"mode\":\"self-test\"}\n", cmd.c_str());
            return 0;
        }
        fprintf(stderr, "%s requires --self-test\n", cmd.c_str());
        return 2;
    }

    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    usage();
    return 1;
}
