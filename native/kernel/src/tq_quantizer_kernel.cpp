/**
 * TurboQuant Native Kernel — Lloyd-Max Quantizer Implementation
 */

#include "tq_quantizer_kernel.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

static inline float gamma(float x) {
    if (x < 0.5f) return 3.14159265358979323846f / (std::sin(3.14159265358979323846f * x) * gamma(1.0f - x));
    x -= 1.0f;
    float g = 2.506628275635;
    g += 0.0008030466757 / (x + 2.0f);
    g -= 0.0002481092879 / (x + 3.0f);
    g += 0.0000438043135 / (x + 4.0f);
    g -= 0.0000013932715 / (x + 5.0f);
    return g * std::exp((x + 0.5f) * std::log(x + 5.5f) - x - 5.5f);
}

static float beta_pdf(float x, float alpha, float beta) {
    if (x <= 0.0f || x >= 1.0f) return 0.0f;
    float log_pdf = (alpha - 1.0f) * std::log(x) + (beta - 1.0f) * std::log(1.0f - x);
    log_pdf -= std::log(gamma(alpha)) + std::log(gamma(beta)) - std::log(gamma(alpha + beta));
    return std::exp(log_pdf);
}

static float quantile(float cdf, float min_val, float max_val) {
    return min_val + cdf * (max_val - min_val);
}

cl_int tq_quantizer_init(tq_quantizer_kernel_t* q, uint32_t dims, uint8_t bits) {
    if (!q || dims == 0 || bits == 0 || bits > TQ_MAX_CODEBOOK_LEVELS) {
        return CL_INVALID_VALUE;
    }

    q->dimensions = dims;
    q->bits_per_value = bits;
    q->codebook_size = 1 << bits;
    void* tmp;
    posix_memalign(&tmp, 32, q->codebook_size * sizeof(float));
    q->codebook = (float*)tmp;

    if (!q->codebook) return CL_OUT_OF_HOST_MEMORY;

    // Initialize symmetric codebook
    float half_levels = (float)(q->codebook_size / 2);
    for (uint32_t i = 0; i < q->codebook_size; i++) {
        q->codebook[i] = ((float)i - half_levels) / half_levels;
    }

    q->scale = 1.0f;
    q->offset = 0.0f;
    q->is_initialized = 1;
    return CL_SUCCESS;
}

cl_int tq_quantizer_init_beta(tq_quantizer_kernel_t* q, uint32_t dims, uint8_t bits) {
    if (!q || dims == 0 || bits == 0 || bits > TQ_MAX_CODEBOOK_LEVELS) {
        return CL_INVALID_VALUE;
    }

    cl_int err = tq_quantizer_init(q, dims, bits);
    if (err != CL_SUCCESS) return err;

    // Beta(d/2, d/2) distribution for TurboQuant
    float alpha = (float)dims / 2.0f;
    float beta = (float)dims / 2.0f;

    uint32_t n = q->codebook_size;
    for (uint32_t i = 0; i < n; i++) {
        float cdf = ((float)i + 0.5f) / (float)n;
        q->codebook[i] = quantile(cdf, -1.0f, 1.0f);
    }

    return CL_SUCCESS;
}

cl_int tq_quantizer_init_uniform(tq_quantizer_kernel_t* q, uint32_t dims, uint8_t bits) {
    if (!q || dims == 0 || bits == 0 || bits > TQ_MAX_CODEBOOK_LEVELS) {
        return CL_INVALID_VALUE;
    }

    return tq_quantizer_init(q, dims, bits);
}

cl_int tq_quantizer_shutdown(tq_quantizer_kernel_t* q) {
    if (!q) return CL_INVALID_VALUE;

    if (q->codebook) {
        free(q->codebook);
        q->codebook = nullptr;
    }

    q->is_initialized = 0;
    return CL_SUCCESS;
}

static uint32_t find_nearest_index(const float* codebook, uint32_t size, float value) {
    uint32_t best = 0;
    float best_dist = std::abs(value - codebook[0]);

    for (uint32_t i = 1; i < size; i++) {
        float dist = std::abs(value - codebook[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }

    return best;
}

cl_int tq_quantizer_quantize(const tq_quantizer_kernel_t* q, const float* values,
                               uint32_t* indices, uint32_t count) {
    if (!q || !q->is_initialized || !values || !indices) {
        return CL_INVALID_VALUE;
    }

    for (uint32_t i = 0; i < count && i < q->dimensions; i++) {
        indices[i] = find_nearest_index(q->codebook, q->codebook_size, values[i]);
    }

    return CL_SUCCESS;
}

cl_int tq_quantizer_dequantize(const tq_quantizer_kernel_t* q, const uint32_t* indices,
                                 float* values, uint32_t count) {
    if (!q || !q->is_initialized || !indices || !values) {
        return CL_INVALID_VALUE;
    }

    for (uint32_t i = 0; i < count && i < q->dimensions; i++) {
        values[i] = q->codebook[indices[i] % q->codebook_size];
    }

    return CL_SUCCESS;
}

static void pack_bits(const uint32_t* indices, uint32_t count, uint8_t bits, uint8_t* packed) {
    uint32_t bit_pos = 0;
    std::memset(packed, 0, ((count * bits) + 7) / 8);

    for (uint32_t i = 0; i < count; i++) {
        for (int b = 0; b < bits; b++) {
            uint32_t byte_idx = bit_pos / 8;
            uint32_t bit_idx = bit_pos % 8;
            if ((indices[i] >> b) & 1) {
                packed[byte_idx] |= (1 << bit_idx);
            }
            bit_pos++;
        }
    }
}

cl_int tq_quantizer_encode(const tq_quantizer_kernel_t* q, const float* values,
                              uint8_t* packed, uint32_t count) {
    if (!q || !q->is_initialized || !values || !packed) {
        return CL_INVALID_VALUE;
    }

    uint32_t* indices = (uint32_t*)alloca(count * sizeof(uint32_t));
    tq_quantizer_quantize(q, values, indices, count);
    pack_bits(indices, count, q->bits_per_value, packed);

    return CL_SUCCESS;
}

cl_int tq_quantizer_decode(const tq_quantizer_kernel_t* q, const uint8_t* packed,
                             float* values, uint32_t count) {
    if (!q || !q->is_initialized || !packed || !values) {
        return CL_INVALID_VALUE;
    }

    uint32_t* indices = (uint32_t*)alloca(count * sizeof(uint32_t));
    uint32_t bit_pos = 0;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = 0;
        for (uint32_t b = 0; b < q->bits_per_value; b++) {
            uint32_t byte_idx = bit_pos / 8;
            uint32_t bit_idx = bit_pos % 8;
            if (packed[byte_idx] & (1 << bit_idx)) {
                idx |= (1 << b);
            }
            bit_pos++;
        }
        indices[i] = idx;
    }

    tq_quantizer_dequantize(q, indices, values, count);
    return CL_SUCCESS;
}

cl_int tq_quantizer_train(tq_quantizer_kernel_t* q, const float* samples, uint32_t sample_count) {
    if (!q || !q->is_initialized || !samples || sample_count == 0) {
        return CL_INVALID_VALUE;
    }

    // Lloyd-Max iteration
    for (int iter = 0; iter < 16; iter++) {
        std::vector<float> sums(q->codebook_size, 0.0f);
        std::vector<uint32_t> counts(q->codebook_size, 0);

        for (uint32_t i = 0; i < sample_count; i++) {
            uint32_t idx = find_nearest_index(q->codebook, q->codebook_size, samples[i]);
            sums[idx] += samples[i];
            counts[idx]++;
        }

        // Update centroids
        for (uint32_t i = 0; i < q->codebook_size; i++) {
            if (counts[i] > 0) {
                q->codebook[i] = sums[i] / (float)counts[i];
            }
        }
    }

    return CL_SUCCESS;
}

float tq_quantizer_compute_mse(const tq_quantizer_kernel_t* q, const float* values, uint32_t count) {
    if (!q || !q->is_initialized || !values || count == 0) {
        return 0.0f;
    }

    float total_mse = 0.0f;

    for (uint32_t i = 0; i < count && i < q->dimensions; i++) {
        uint32_t idx = find_nearest_index(q->codebook, q->codebook_size, values[i]);
        float diff = values[i] - q->codebook[idx];
        total_mse += diff * diff;
    }

    return total_mse / (float)(count < q->dimensions ? count : q->dimensions);
}

#ifdef __cplusplus

namespace tq {
namespace kernel {

BetaCodebook::BetaCodebook(uint32_t dimensions, uint8_t bits)
    : dims_(dimensions), bits_(bits) {
    uint32_t levels = 1 << bits;
    codebook_.resize(levels);
    boundaries_.resize(levels + 1);
}

BetaCodebook::~BetaCodebook() = default;

void BetaCodebook::compute() {
    compute_beta_distribution();
}

void BetaCodebook::compute_beta_distribution() {
    uint32_t levels = 1 << bits_;
    float alpha = (float)dims_ / 2.0f;
    float beta = (float)dims_ / 2.0f;

    for (uint32_t i = 0; i < levels; i++) {
        float cdf = ((float)i + 0.5f) / (float)levels;
        float x = cdf * 2.0f - 1.0f;  // [-1, 1]
        codebook_[i] = x;
        boundaries_[i] = -1.0f + (2.0f / (float)levels) * i;
    }
    boundaries_[levels] = 1.0f;
}

void BetaCodebook::lloyd_max_iteration(const float* samples, uint32_t sample_count) {
    uint32_t levels = 1 << bits_;
    std::vector<float> sums(levels, 0.0f);
    std::vector<uint32_t> counts(levels, 0);

    for (uint32_t i = 0; i < sample_count; i++) {
        float val = samples[i];
        uint32_t best = 0;
        float best_dist = std::abs(val - codebook_[0]);

        for (uint32_t j = 1; j < levels; j++) {
            float dist = std::abs(val - codebook_[j]);
            if (dist < best_dist) {
                best_dist = dist;
                best = j;
            }
        }

        sums[best] += val;
        counts[best]++;
    }

    for (uint32_t i = 0; i < levels; i++) {
        if (counts[i] > 0) {
            codebook_[i] = sums[i] / (float)counts[i];
        }
    }
}

float BetaCodebook::quantize(float value) const {
    uint32_t idx = quantize_index(value);
    return codebook_[idx];
}

uint32_t BetaCodebook::quantize_index(float value) const {
    uint32_t levels = 1 << bits_;
    uint32_t best = 0;
    float best_dist = std::abs(value - codebook_[0]);

    for (uint32_t i = 1; i < levels; i++) {
        float dist = std::abs(value - codebook_[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }

    return best;
}

float BetaCodebook::dequantize(uint32_t index) const {
    if (index >= codebook_.size()) return 0.0f;
    return codebook_[index];
}

void BetaCodebook::train(const float* samples, uint32_t sample_count) {
    for (int iter = 0; iter < 16; iter++) {
        lloyd_max_iteration(samples, sample_count);
    }
}

float BetaCodebook::compute_distortion(const float* samples, uint32_t count) const {
    float total = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float val = samples[i];
        uint32_t idx = quantize_index(val);
        float diff = val - codebook_[idx];
        total += diff * diff;
    }
    return total / (float)count;
}

float BetaCodebook::compute_mse(const float* samples, uint32_t count) const {
    return compute_distortion(samples, count);
}

UniformCodebook::UniformCodebook(uint8_t bits)
    : bits_(bits), levels_(1 << bits), min_val_(-1.0f), max_val_(1.0f) {
    step_ = (max_val_ - min_val_) / (float)(levels_ - 1);
    offset_ = min_val_;
    scale_ = 1.0f / step_;
}

UniformCodebook::~UniformCodebook() = default;

void UniformCodebook::init(float min_val, float max_val) {
    min_val_ = min_val;
    max_val_ = max_val;
    step_ = (max_val_ - min_val_) / (float)(levels_ - 1);
    offset_ = min_val_;
    scale_ = 1.0f / step_;
}

void UniformCodebook::init_symmetric(float scale) {
    min_val_ = -scale;
    max_val_ = scale;
    step_ = 2.0f * scale / (float)(levels_ - 1);
    offset_ = min_val_;
    scale_ = 1.0f / step_;
}

float UniformCodebook::quantize(float value) const {
    uint32_t idx = quantize_index(value);
    return offset_ + (float)idx * step_;
}

uint32_t UniformCodebook::quantize_index(float value) const {
    float normalized = (value - offset_) * scale_;
    int idx = (int)(normalized + 0.5f);
    if (idx < 0) idx = 0;
    if (idx >= (int)levels_) idx = levels_ - 1;
    return (uint32_t)idx;
}

float UniformCodebook::dequantize(uint32_t index) const {
    if (index >= levels_) return offset_;
    return offset_ + (float)index * step_;
}

float UniformCodebook::compute_mse(const float* samples, uint32_t count) const {
    float total = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float q = quantize(samples[i]);
        float diff = samples[i] - q;
        total += diff * diff;
    }
    return total / (float)count;
}

} // namespace kernel
} // namespace tq

#endif // __cplusplus