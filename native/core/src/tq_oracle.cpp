/**
 * TurboQuant Core — Quality Oracle Implementation
 */

#include "tq_oracle.h"
#include "tq_core.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

static float log2f_safe(float x) {
    return x > 0 ? log2f(x) : 0;
}

// ── Analyze data characteristics ─────────────────────────────────────────────
void tq_oracle_analyze(const float* data, uint32_t count, tq_data_characteristics* chars) {
    if (!data || !chars) return;
    memset(chars, 0, sizeof(tq_data_characteristics));
    if (count == 0) return;

    float min = data[0], max = data[0];
    double sum = 0, sum_sq = 0;

    for (uint32_t i = 0; i < count; i++) {
        float v = data[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        sum_sq += v * v;
    }

    double n = count;
    double mean = sum / n;
    double variance = (sum_sq / n) - (mean * mean);
    double std_dev = sqrt(variance > 0 ? variance : 0);

    // Skewness and kurtosis
    double skewness = 0, kurtosis = 0;
    if (std_dev > 1e-10) {
        for (uint32_t i = 0; i < count; i++) {
            float centered = (data[i] - (float)mean) / (float)std_dev;
            skewness += centered * centered * centered;
            kurtosis += centered * centered * centered * centered;
        }
        skewness /= n;
        kurtosis /= n;
    }

    // Entropy via histogram
    int histogram[1001] = {0};
    for (uint32_t i = 0; i < count; i++) {
        int idx = (int)roundf(data[i] * 500.0f + 500.0f);
        if (idx < 0) idx = 0;
        if (idx > 1000) idx = 1000;
        histogram[idx]++;
    }

    float entropy = 0.0f;
    for (int i = 0; i <= 1000; i++) {
        if (histogram[i] > 0) {
            float p = (float)histogram[i] / (float)count;
            entropy -= p * log2f_safe(p);
        }
    }

    chars->min = min;
    chars->max = max;
    chars->mean = (float)mean;
    chars->variance = (float)variance;
    chars->std_dev = (float)std_dev;
    chars->skewness = (float)skewness;
    chars->kurtosis = (float)kurtosis;
    chars->entropy = entropy;
    chars->dynamic_range = max - min;
}

// ── Estimate MSE ─────────────────────────────────────────────────────────────
float tq_oracle_estimate_mse(const tq_data_characteristics* chars, uint8_t bits) {
    if (!chars) return -1.0f;
    uint32_t levels = 1u << bits;
    float quantization_step = 2.0f / (float)(levels - 1);
    float theoretical_mse = (quantization_step * quantization_step) / 12.0f;

    float distribution_factor = 1.0f + fabsf(chars->skewness) * 0.1f;
    float variance_factor = fmaxf(1.0f, chars->variance * 2.0f);

    return theoretical_mse * distribution_factor * variance_factor;
}

// ── Estimate compression ratio ──────────────────────────────────────────────
float tq_oracle_estimate_compression_ratio(
    const tq_data_characteristics* chars,
    uint8_t bits,
    uint8_t original_bits
) {
    if (!chars) return 0.0f;
    float efficiency = fminf(1.0f, chars->entropy / 8.0f);
    return ((float)original_bits / (float)bits) * (0.9f + efficiency * 0.1f);
}

// ── Calculate confidence ────────────────────────────────────────────────────
float tq_oracle_calculate_confidence(const tq_data_characteristics* chars, uint8_t bits) {
    if (!chars) return 0.0f;
    float confidence = 1.0f;

    if (chars->dynamic_range > 2.0f) confidence -= 0.2f;
    if (chars->kurtosis > 3.0f) confidence -= 0.1f;
    if (fabsf(chars->skewness) > 1.0f) confidence -= 0.1f;

    float expected_entropy = log2f_safe((float)(1u << bits));
    float entropy_diff = fabsf(chars->entropy - expected_entropy);
    float entropy_match = 1.0f - entropy_diff / fmaxf(expected_entropy, 0.001f);
    confidence += entropy_match * 0.2f;

    return fmaxf(0.0f, fminf(1.0f, confidence));
}

// ── Full recommendation ──────────────────────────────────────────────────────
int tq_oracle_recommend(
    const float* data,
    uint32_t count,
    float target_mse,
    float target_compression,
    tq_oracle_recommendation* result
) {
    if (!data || !result) return TQ_ERR_NULL;

    tq_data_characteristics chars;
    tq_oracle_analyze(data, count, &chars);

    uint8_t candidates_bits[] = {2, 3, 4, 8};
    uint32_t n_candidates = 4;
    float best_confidence = -1.0f;
    uint8_t best_bits = 4;
    float best_mse = 1e9f;
    float best_cr = 0.0f;

    for (uint32_t i = 0; i < n_candidates; i++) {
        uint8_t bits = candidates_bits[i];
        float mse = tq_oracle_estimate_mse(&chars, bits);
        float cr = tq_oracle_estimate_compression_ratio(&chars, bits, 32);
        float conf = tq_oracle_calculate_confidence(&chars, bits);

        bool passes_mse = target_mse <= 0.0f || mse <= target_mse;
        bool passes_cr = target_compression <= 0.0f || cr >= target_compression;

        if (passes_mse && passes_cr && conf > best_confidence) {
            best_confidence = conf;
            best_bits = bits;
            best_mse = mse;
            best_cr = cr;
        }
    }

    // If no candidate passes constraints, use best available
    if (best_confidence < 0) {
        for (uint32_t i = 0; i < n_candidates; i++) {
            uint8_t bits = candidates_bits[i];
            float mse = tq_oracle_estimate_mse(&chars, bits);
            float cr = tq_oracle_estimate_compression_ratio(&chars, bits, 32);
            float conf = tq_oracle_calculate_confidence(&chars, bits);
            if (conf > best_confidence) {
                best_confidence = conf;
                best_bits = bits;
                best_mse = mse;
                best_cr = cr;
            }
        }
    }

    result->bits_per_value = best_bits;
    result->compression_ratio = best_cr;
    result->estimated_mse = best_mse;
    result->confidence = best_confidence > 0 ? best_confidence : 0.5f;

    return TQ_OK;
}

// ── Optimal bits ─────────────────────────────────────────────────────────────
uint8_t tq_oracle_optimal_bits(const float* data, uint32_t count, float tolerance) {
    if (!data || count == 0) return 8;
    tq_data_characteristics chars;
    tq_oracle_analyze(data, count, &chars);

    uint8_t candidates[] = {2, 3, 4, 8};
    for (uint32_t i = 0; i < 4; i++) {
        float mse = tq_oracle_estimate_mse(&chars, candidates[i]);
        if (mse <= tolerance) return candidates[i];
    }
    return 8;
}

// ── Compression ratio ────────────────────────────────────────────────────────
float tq_oracle_compression_ratio(const float* data, uint32_t count, uint8_t bits) {
    if (!data) return 0.0f;
    tq_data_characteristics chars;
    tq_oracle_analyze(data, count, &chars);
    return tq_oracle_estimate_compression_ratio(&chars, bits, 32);
}

// ── Predict error ───────────────────────────────────────────────────────────
float tq_oracle_predict_error(const float* data, uint32_t count, uint8_t bits) {
    if (!data) return -1.0f;
    tq_data_characteristics chars;
    tq_oracle_analyze(data, count, &chars);
    return tq_oracle_estimate_mse(&chars, bits);
}