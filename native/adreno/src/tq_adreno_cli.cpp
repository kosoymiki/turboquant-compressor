/**
 * TurboQuant Adreno CLI — unified native runtime entry point.
 * Commands: probe, mse-score, value-dequant, fused-attention, benchmark
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <chrono>
#include <cmath>

namespace tq { struct BenchResult { double scalar_ms; double neon_ms; double speedup; };
BenchResult benchmark_mse_score(int tokens, int dim, int bits, int runs); }

static void cmd_probe() {
    printf("{\n");
    printf("  \"backend\": \"%s\",\n", tq::get_backend().name);
    printf("  \"hasNeon\": %s,\n", tq::has_neon() ? "true" : "false");
    printf("  \"backends\": [\"cpu_scalar\"");
    if (tq::has_neon()) printf(", \"cpu_neon\"");
    printf("]\n}\n");
}

static void cmd_benchmark() {
    struct Shape { int tokens; int dim; int bits; };
    Shape shapes[] = {{128, 64, 3}, {256, 128, 3}, {1024, 128, 4}, {4096, 128, 4}};

    printf("[\n");
    for (int i = 0; i < 4; i++) {
        auto& s = shapes[i];
        auto r = tq::benchmark_mse_score(s.tokens, s.dim, s.bits, 10);
        printf("  {\"tokens\":%d,\"dim\":%d,\"bits\":%d,\"scalar_ms\":%.3f,\"neon_ms\":%.3f,\"speedup\":%.2f}%s\n",
               s.tokens, s.dim, s.bits, r.scalar_ms, r.neon_ms, r.speedup, i < 3 ? "," : "");
    }
    printf("]\n");
}

static void cmd_self_test() {
    // Value dequant parity test
    const int tokens = 2, dim = 4, bits = 4, group_size = 4;
    uint8_t packed[4] = {}; // 2 tokens * 2 bytes
    // Token 0: [5, 10, 3, 8]
    packed[0] = 5 | (10 << 4); packed[1] = 3 | (8 << 4);
    // Token 1: [0, 15, 7, 1]
    uint8_t packed2[4] = {};
    packed2[0] = 0 | (15 << 4); packed2[1] = 7 | (1 << 4);
    uint8_t all_packed[4];
    memcpy(all_packed, packed, 2);
    memcpy(all_packed + 2, packed2, 2);

    float scales[2] = {0.1f, 0.2f};
    float zeros[2] = {-0.5f, -1.0f};

    tq::ValueDequantInput input{all_packed, scales, zeros, tokens, dim, bits, group_size};

    float scalar_out[8] = {}, neon_out[8] = {};
    tq::cpu_scalar_value_dequant(input, scalar_out);
    tq::neon_value_dequant(input, neon_out);

    int pass = 0;
    for (int i = 0; i < tokens * dim; i++) {
        if (fabsf(scalar_out[i] - neon_out[i]) < 1e-5f) pass++;
        else printf("  FAIL [%d]: scalar=%.6f neon=%.6f\n", i, scalar_out[i], neon_out[i]);
    }
    printf("{\"test\":\"value_dequant_neon_parity\",\"pass\":%d,\"total\":%d}\n", pass, tokens * dim);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tq_adreno_cli <probe|benchmark|self-test>\n");
        return 1;
    }
    std::string cmd = argv[1];
    if (cmd == "probe") { cmd_probe(); return 0; }
    if (cmd == "benchmark") { cmd_benchmark(); return 0; }
    if (cmd == "self-test") { cmd_self_test(); return 0; }
    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    return 1;
}
