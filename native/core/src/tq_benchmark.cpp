/**
 * TurboQuant Core — Benchmark Framework Implementation
 */

#include "tq_benchmark.h"
#include "tq_core.h"
#include "tq_beta_sphere.h"
#include "tq_kernel_inline.h"
#include "tq_base64.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>

// ── PRNG for test data generation ────────────────────────────────────────────
static uint32_t prng_next(uint32_t* state) {
    *state = (*state * 1103515245u + 12345u) & 0x7fffffffu;
    return *state;
}

static float prng_uniform(uint32_t* state) {
    return (float)prng_next(state) / (float)0x7fffffff;
}

// ── Generate test data ────────────────────────────────────────────────────────
static float* generate_test_data(uint32_t dims, uint32_t count, uint32_t seed) {
    float* vecs = (float*)malloc(count * dims * sizeof(float));
    uint32_t state = seed;
    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t j = 0; j < dims; j++) {
            vecs[i * dims + j] = prng_uniform(&state) * 2.0f - 1.0f;
        }
    }
    return vecs;
}

// ── Calculate norms ───────────────────────────────────────────────────────────
static float* compute_norms(const float* vecs, uint32_t count, uint32_t dims) {
    float* norms = (float*)malloc(count * sizeof(float));
    for (uint32_t i = 0; i < count; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < dims; j++) {
            float v = vecs[i * dims + j];
            sum += v * v;
        }
        norms[i] = sqrtf(sum);
    }
    return norms;
}

// ── MSE calculation ───────────────────────────────────────────────────────────
static float calculate_mse(
    const float* original, const float* reconstructed,
    uint32_t count, uint32_t dims
) {
    double total = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t j = 0; j < dims; j++) {
            float diff = original[i * dims + j] - reconstructed[i * dims + j];
            total += diff * diff;
        }
    }
    return (float)(total / (count * dims));
}

// ── Max error calculation ─────────────────────────────────────────────────────
static float calculate_max_error(
    const float* original, const float* reconstructed,
    uint32_t count, uint32_t dims
) {
    float max_err = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t j = 0; j < dims; j++) {
            float diff = fabsf(original[i * dims + j] - reconstructed[i * dims + j]);
            if (diff > max_err) max_err = diff;
        }
    }
    return max_err;
}

// ── Reconstruct vectors from compressed DB ────────────────────────────────────
static float* reconstruct_vectors(
    const char* base64, const float* norms,
    uint32_t count, uint32_t dims, uint8_t bits,
    uint8_t codebook_type
) {
    size_t base64_len = strlen(base64);
    std::vector<uint8_t> bin = tq::core::base64::decode(std::string(base64, base64_len));
    if (bin.size() < 80) return nullptr;

    uint32_t norms_off = 80;
    uint32_t quant_off = norms_off + count * 4;
    uint32_t padded = 0; memcpy(&padded, bin.data() + 12, 4);

    float* result = (float*)malloc(count * dims * sizeof(float));
    memset(result, 0, count * dims * sizeof(float));

    tq_beta_codebook bcb;
    const double* cents = nullptr;
    float levels_m1 = 15.0f;

    if (codebook_type == 2) { // TQ_CODEBOOK_BETA
        bcb = tq_compute_lloyd_max_beta_codebook((int)dims, bits, 50, 1e-6);
        cents = bcb.centroids;
    }

    for (uint32_t vi = 0; vi < count; vi++) {
        float norm; memcpy(&norm, bin.data() + norms_off + vi * 4, 4);
        uint32_t idx[1024];
        tq_unpack_bits(bin.data() + quant_off + (size_t)vi * padded, idx, dims, bits);

        for (uint32_t i = 0; i < dims; i++) {
            float decoded;
            if (codebook_type == 2 && cents) {
                decoded = (float)cents[idx[i]];
            } else {
                decoded = ((float)idx[i] / levels_m1) * 2.0f - 1.0f;
            }
            result[vi * dims + i] = decoded * norm;
        }
    }

    if (codebook_type == 2) tq_free_beta_codebook(&bcb);
    return result;
}

// ── Main benchmark function ────────────────────────────────────────────────────
int tq_benchmark_run_single(const tq_benchmark_config* cfg, tq_benchmark_result* result) {
    if (!cfg || !result) return TQ_ERR_NULL;

    memset(result, 0, sizeof(tq_benchmark_result));
    memcpy(&result->config, cfg, sizeof(tq_benchmark_config));

    uint32_t dims = cfg->dimensions;
    uint32_t count = cfg->vector_count;
    uint8_t bits = cfg->bits_per_value ? cfg->bits_per_value : 4;
    uint8_t codebook = cfg->codebook_type;
    uint32_t seed = 42;

    // Generate test data
    float* vecs = generate_test_data(dims, count, seed);
    float* norms = compute_norms(vecs, count, dims);

    // ── Encoding ─────────────────────────────────────────────────────────────
    size_t est = tq_estimate_compressed_size(count, dims, bits, cfg->use_qjl ? 64 : 0);
    std::vector<char> out(est * 2);
    size_t out_len = est * 2;

    clock_t encode_start = clock();
    int r = tq_compress(vecs, count, dims, bits, seed, cfg->use_qjl ? 64 : 0, codebook, out.data(), &out_len, nullptr);
    clock_t encode_end = clock();

    float encode_time = (float)(encode_end - encode_start) * 1000.0f / (float)CLOCKS_PER_SEC;

    if (r != TQ_OK) {
        free(vecs); free(norms);
        return r;
    }

    // ── Decoding ─────────────────────────────────────────────────────────────
    clock_t decode_start = clock();
    float* reconstructed = reconstruct_vectors(out.data(), norms, count, dims, bits, codebook);
    clock_t decode_end = clock();

    float decode_time = (float)(decode_end - decode_start) * 1000.0f / (float)CLOCKS_PER_SEC;

    // ── Metrics ──────────────────────────────────────────────────────────────
    float original_size = (float)count * dims * 4.0f;
    float compressed_size = (float)out_len;
    float compression_ratio = compressed_size > 0 ? original_size / compressed_size : 0.0f;

    float encode_throughput = encode_time > 0 ? (count / encode_time) * 1000.0f : 0.0f;
    float decode_throughput = decode_time > 0 ? (count / decode_time) * 1000.0f : 0.0f;
    float throughput_mb = encode_time > 0 ? (original_size / 1024.0f / 1024.0f) / (encode_time / 1000.0f) : 0.0f;

    float mse = reconstructed ? calculate_mse(vecs, reconstructed, count, dims) : -1.0f;
    float max_error = reconstructed ? calculate_max_error(vecs, reconstructed, count, dims) : -1.0f;

    result->metrics.encode_time_ms = encode_time;
    result->metrics.decode_time_ms = decode_time;
    result->metrics.compression_ratio = compression_ratio;
    result->metrics.encode_throughput = encode_throughput;
    result->metrics.decode_throughput = decode_throughput;
    result->metrics.throughput_mb_per_sec = throughput_mb;
    result->metrics.mse = mse;
    result->metrics.max_error = max_error;

    free(vecs); free(norms);
    if (reconstructed) free(reconstructed);
    return TQ_OK;
}

// ── Matrix runner ─────────────────────────────────────────────────────────────
int tq_benchmark_run_matrix(
    const tq_benchmark_config* configs,
    uint32_t config_count,
    tq_benchmark_result** results_out,
    uint32_t* result_count_out
) {
    if (!configs || !results_out || !result_count_out) return TQ_ERR_NULL;

    tq_benchmark_result* results = (tq_benchmark_result*)malloc(config_count * sizeof(tq_benchmark_result));
    if (!results) return TQ_ERR_ALLOC;

    for (uint32_t i = 0; i < config_count; i++) {
        tq_benchmark_run_single(&configs[i], &results[i]);
    }

    *results_out = results;
    *result_count_out = config_count;
    return TQ_OK;
}

// ── Summarize results ─────────────────────────────────────────────────────────
int tq_benchmark_summarize(
    const tq_benchmark_result* results,
    uint32_t result_count,
    tq_benchmark_summary* summary_out
) {
    if (!results || !summary_out) return TQ_ERR_NULL;
    if (result_count == 0) {
        memset(summary_out, 0, sizeof(tq_benchmark_summary));
        return TQ_OK;
    }

    memset(summary_out, 0, sizeof(tq_benchmark_summary));
    summary_out->total_tests = result_count;

    float sum_cr = 0.0f, sum_et = 0.0f, sum_dt = 0.0f;
    float best_cr = results[0].metrics.compression_ratio;
    float best_et = results[0].metrics.encode_throughput;
    float best_dt = results[0].metrics.decode_throughput;
    float worst_cr = results[0].metrics.compression_ratio;
    float worst_et = results[0].metrics.encode_throughput;
    float worst_dt = results[0].metrics.decode_throughput;

    for (uint32_t i = 0; i < result_count; i++) {
        float cr = results[i].metrics.compression_ratio;
        float et = results[i].metrics.encode_throughput;
        float dt = results[i].metrics.decode_throughput;

        sum_cr += cr; sum_et += et; sum_dt += dt;

        if (cr > best_cr) best_cr = cr;
        if (et > best_et) best_et = et;
        if (dt > best_dt) best_dt = dt;
        if (cr < worst_cr) worst_cr = cr;
        if (et < worst_et) worst_et = et;
        if (dt < worst_dt) worst_dt = dt;
    }

    summary_out->avg_compression_ratio = sum_cr / result_count;
    summary_out->avg_encode_throughput = sum_et / result_count;
    summary_out->avg_decode_throughput = sum_dt / result_count;
    summary_out->best_compression_ratio = best_cr;
    summary_out->best_encode_throughput = best_et;
    summary_out->best_decode_throughput = best_dt;
    summary_out->worst_compression_ratio = worst_cr;
    summary_out->worst_encode_throughput = worst_et;
    summary_out->worst_decode_throughput = worst_dt;

    return TQ_OK;
}

// ── Cleanup ──────────────────────────────────────────────────────────────────
void tq_benchmark_free_results(tq_benchmark_result* results, uint32_t count) {
    (void)count;
    free(results);
}