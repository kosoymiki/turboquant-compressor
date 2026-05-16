/**
 * TurboQuant — benchmark utility for native backends.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"
#include <chrono>
#include <cstdio>

namespace tq {

struct BenchResult {
    double scalar_ms;
    double neon_ms;
    double speedup;
};

BenchResult benchmark_mse_score(int tokens, int dim, int bits, int runs) {
    int packed_stride = (dim * bits + 7) / 8;
    auto* packed = new uint8_t[tokens * packed_stride]();
    auto* norms = new float[tokens];
    auto* query = new float[dim];
    int n_centroids = 1 << bits;
    auto* centroids = new float[n_centroids];
    auto* scores = new float[tokens];

    for (int i = 0; i < tokens; i++) norms[i] = 1.0f;
    for (int i = 0; i < dim; i++) query[i] = 0.1f;
    for (int i = 0; i < n_centroids; i++) centroids[i] = -1.0f + 2.0f * i / (n_centroids - 1);

    MseInput input{packed, norms, query, centroids, tokens, dim, bits, packed_stride};

    // Scalar
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < runs; r++) cpu_scalar_mse_score(input, scores);
    auto t1 = std::chrono::steady_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / runs;

    // NEON
    auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < runs; r++) neon_mse_score(input, scores);
    auto t3 = std::chrono::steady_clock::now();
    double neon_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / runs;

    delete[] packed; delete[] norms; delete[] query; delete[] centroids; delete[] scores;
    return {scalar_ms, neon_ms, scalar_ms / neon_ms};
}

} // namespace tq
