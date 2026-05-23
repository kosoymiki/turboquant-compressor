/**
 * TurboQuant Core — Quality Benchmarks Implementation
 */

#include "tq_quality.h"
#include "tq_core.h"
#include "tq_beta_sphere.h"
#include "tq_base64.h"
#include "tq_kernel_inline.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>

float tq_quality_mse(
    const float* original_vectors,
    uint32_t N,
    uint32_t dims,
    uint8_t codebook_type
) {
    // Re-compress and measure distortion
    size_t est = tq_estimate_compressed_size(N, dims, 4, 0);
    std::string out(est * 2, '\0');
    size_t ol = est * 2;
    tq_compress_stats st;
    int r = tq_compress(original_vectors, N, dims, 4, 42, 0, codebook_type,
                        out.data(), &ol, &st);
    if (r != 0) return -1.0f;

    // Decode + re-quantize + measure MSE
    std::vector<uint8_t> bin = tq::core::base64::decode(std::string(out.data(), ol));
    if (bin.size() < 80) return -1.0f;

    uint32_t norms_off = 80, norms_len = N * 4;
    uint32_t quant_off = norms_off + norms_len;
    uint32_t padded = 0; std::memcpy(&padded, bin.data() + 12, 4);
    size_t pp = (size_t)padded * 4 / 8;

    float total_mse = 0.0f;
    uint32_t total_coords = 0;

    tq_beta_codebook bcb;
    const double* cents = nullptr;
    if (codebook_type == TQ_CODEBOOK_BETA) {
        bcb = tq_compute_lloyd_max_beta_codebook((int)dims, 4, 50, 1e-6);
        cents = bcb.centroids;
    }
    float levels_m1 = 15.0f;

    for (uint32_t vi = 0; vi < N; vi++) {
        // Get norms
        float norm; std::memcpy(&norm, bin.data() + norms_off + vi * 4, 4);

        // Unpack
        uint32_t idx[1024];
        tq_unpack_bits(bin.data() + quant_off + (size_t)vi * pp,
                       idx, dims, 4);

        // Reconstruct
        for (uint32_t i = 0; i < dims; i++) {
            float decoded;
            if (codebook_type == TQ_CODEBOOK_BETA && cents) {
                decoded = (float)cents[idx[i]];
            } else {
                decoded = ((float)idx[i] / levels_m1) * 2.0f - 1.0f;
            }
            decoded *= norm;
            float orig = original_vectors[(size_t)vi * dims + i];
            float diff = orig - decoded;
            total_mse += diff * diff;
        }
        total_coords += dims;
    }

    if (codebook_type == TQ_CODEBOOK_BETA) tq_free_beta_codebook(&bcb);
    return (total_coords > 0) ? (total_mse / total_coords) : -1.0f;
}

tq_quality_result tq_quality_benchmark(
    const float* vectors,
    uint32_t N,
    uint32_t dims,
    const char* db_base64,
    tq_quality_config cfg
) {
    tq_quality_result result;
    std::memset(&result, 0, sizeof(result));
    result.codebook_type = cfg.codebook_type;
    result.total_db_vectors = N;
    result.dimensions = dims;

    // ── MSE ─────────────────────────────────────────────────────────────
    result.mse_per_coord = tq_quality_mse(vectors, N, dims, cfg.codebook_type);

    // ── Search recall ────────────────────────────────────────────────────
    uint32_t k = cfg.top_k > 0 ? cfg.top_k : 5;
    uint32_t iters = cfg.iterations > 0 ? cfg.iterations : N;
    if (iters > N) iters = N;

    uint32_t* topk_idx = (uint32_t*)std::malloc(k * sizeof(uint32_t));
    float*    topk_scr = (float*)std::malloc(k * sizeof(float));
    if (!topk_idx || !topk_scr) {
        if (topk_idx) std::free(topk_idx);
        if (topk_scr) std::free(topk_scr);
        return result;
    }

    uint32_t recall_at_1 = 0, recall_at_5 = 0;
    uint64_t total_ns = 0;

    for (uint32_t iter = 0; iter < iters; iter++) {
        const float* query = vectors + (size_t)iter * dims;
        tq_search_stats ss;
        int r = tq_search(db_base64, query, dims, k,
                          0, cfg.use_qjl, topk_idx, topk_scr, &ss);
        if (r != 0) continue;

        total_ns += ss.search_ns;
        if (topk_idx[0] == iter) recall_at_1++;
        bool in_top5 = false;
        for (uint32_t j = 0; j < k && j < 5; j++) {
            if (topk_idx[j] == iter) { in_top5 = true; break; }
        }
        if (in_top5) recall_at_5++;
    }

    std::free(topk_idx); std::free(topk_scr);

    result.recall_at_1 = (float)recall_at_1 / (float)iters;
    result.recall_at_5 = (float)recall_at_5 / (float)iters;
    result.avg_search_ns = (iters > 0) ? (float)total_ns / (float)iters : 0.0f;
    result.total_queries = iters;

    return result;
}