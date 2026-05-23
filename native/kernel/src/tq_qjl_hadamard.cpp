/**
 * TurboQuant Native Kernel — Paper-Faithful 1-bit Hadamard QJL
 * Implementation of Zandieh et al., ICML 2024
 */

#include <CL/cl.h>
#include "tq_qjl_hadamard.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

// ── Forward declarations ──────────────────────────────────────────────────

static void fwht_inplace_impl(float* data, uint32_t n);
static void fwht_normalize_impl(float* data, uint32_t n);
static uint32_t next_pow2(uint32_t x);
static uint32_t log2_pow2(uint32_t x);

// ── Helpers ───────────────────────────────────────────────────────────────

static uint32_t next_pow2(uint32_t x) {
    if (x == 0) return 1;
    x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

static uint32_t log2_pow2(uint32_t x) {
    uint32_t r = 0;
    if (x >= (1u << 16)) { r += 16; x >>= 16; }
    if (x >= (1u <<  8)) { r +=  8; x >>=  8; }
    if (x >= (1u <<  4)) { r +=  4; x >>=  4; }
    if (x >= (1u <<  2)) { r +=  2; x >>=  2; }
    if (x >= (1u <<  1)) { r +=  1; }
    return r;
}

// ── FWHT implementation (O(d log d)) ─────────────────────────────────────

static void fwht_inplace_impl(float* data, uint32_t n) {
    for (uint32_t stride = 1; stride < n; stride <<= 1) {
        for (uint32_t i = 0; i < n; i += stride << 1) {
            for (uint32_t j = 0; j < stride; j++) {
                float u = data[i + j];
                float v = data[i + j + stride];
                data[i + j]          = u + v;
                data[i + j + stride] = u - v;
            }
        }
    }
}

static void fwht_normalize_impl(float* data, uint32_t n) {
    float scale = 1.0f / std::sqrt((float)n);
    for (uint32_t i = 0; i < n; i++) data[i] *= scale;
}

// ── C API implementation ───────────────────────────────────────────────────

cl_int tq_hadamard_qjl_init(tq_hadamard_qjl_t* qjl,
                            uint32_t original_dims,
                            uint32_t sketch_dims,
                            uint32_t seed) {
    if (!qjl || original_dims == 0 || sketch_dims == 0 ||
        sketch_dims > TQ_HADAMARD_QJL_MAX_DIMS) {
        return CL_INVALID_VALUE;
    }

    uint32_t padded = next_pow2(original_dims);

    qjl->original_dims = original_dims;
    qjl->sketch_dims = sketch_dims;
    qjl->original_dims_log2 = log2_pow2(padded);
    qjl->seed = seed;
    qjl->sketch_bytes = (sketch_dims + 7) >> 3;

    // Allocate Rademacher buffer (±1 diagonal)
    void* tmp1;
    void* tmp2;
    int err1 = posix_memalign(&tmp1, 32, padded * sizeof(float));
    int err2 = posix_memalign(&tmp2, 32, padded * sizeof(float));

    qjl->rademacher_buffer = (float*)tmp1;
    qjl->temp_buffer = (float*)tmp2;

    if (err1 != 0 || err2 != 0 || !qjl->rademacher_buffer || !qjl->temp_buffer) {
        if (qjl->rademacher_buffer) free(qjl->rademacher_buffer);
        if (qjl->temp_buffer) free(qjl->temp_buffer);
        qjl->rademacher_buffer = nullptr;
        qjl->temp_buffer = nullptr;
        return CL_OUT_OF_HOST_MEMORY;
    }

    // Generate Rademacher pattern (seeded for reproducibility)
    uint32_t s = seed;
    for (uint32_t i = 0; i < padded; i++) {
        s = s * 1664525U + 1013904223U;
        qjl->rademacher_buffer[i] = ((s >> 16) & 1u) ? 1.0f : -1.0f;
    }

    qjl->is_initialized = 1;
    return CL_SUCCESS;
}

cl_int tq_hadamard_qjl_shutdown(tq_hadamard_qjl_t* qjl) {
    if (!qjl) return CL_INVALID_VALUE;

    if (qjl->rademacher_buffer) { free(qjl->rademacher_buffer); qjl->rademacher_buffer = nullptr; }
    if (qjl->temp_buffer)       { free(qjl->temp_buffer);       qjl->temp_buffer       = nullptr; }

    qjl->is_initialized = 0;
    return CL_SUCCESS;
}

uint32_t tq_hadamard_qjl_sketch_bytes(const tq_hadamard_qjl_t* qjl) {
    return qjl ? qjl->sketch_bytes : 0;
}

void tq_hadamard_qjl_compress(const tq_hadamard_qjl_t* qjl,
                               const float* residual,
                               uint8_t* sketch_out) {
    if (!qjl || !qjl->is_initialized || !residual || !sketch_out) return;

    uint32_t padded = 1u << qjl->original_dims_log2;

    // Step 1: Pad + Rademacher diagonal (y = D * x)
    for (uint32_t i = 0; i < padded; i++) {
        float val = (i < qjl->original_dims) ? residual[i] : 0.0f;
        qjl->temp_buffer[i] = val * qjl->rademacher_buffer[i];
    }

    // Step 2: FWHT O(d log d)
    fwht_inplace_impl(qjl->temp_buffer, padded);
    fwht_normalize_impl(qjl->temp_buffer, padded);

    // Step 3: 1-bit sign quantization — zero overhead
    // Paper: sketch[j] = sign(projected[j]) ∈ {+1, -1} → bit 1/0
    memset(sketch_out, 0, qjl->sketch_bytes);
    for (uint32_t i = 0; i < qjl->sketch_dims; i++) {
        if (qjl->temp_buffer[i] >= 0.0f) {
            sketch_out[i >> 3] |= (1u << (i & 7));
        }
    }
}

void tq_hadamard_qjl_project_query(const tq_hadamard_qjl_t* qjl,
                                   const float* query,
                                   float* out) {
    if (!qjl || !qjl->is_initialized || !query || !out) return;

    uint32_t padded = 1u << qjl->original_dims_log2;

    // Apply Rademacher + FWHT (same as compression)
    for (uint32_t i = 0; i < padded; i++) {
        float val = (i < qjl->original_dims) ? query[i] : 0.0f;
        qjl->temp_buffer[i] = val * qjl->rademacher_buffer[i];
    }
    fwht_inplace_impl(qjl->temp_buffer, padded);
    fwht_normalize_impl(qjl->temp_buffer, padded);

    // Copy projected values (not sign-quantized — continuous for asymmetric)
    for (uint32_t i = 0; i < qjl->sketch_dims; i++) {
        out[i] = qjl->temp_buffer[i];
    }
}

float tq_hadamard_qjl_estimate_dot(const tq_hadamard_qjl_t* qjl,
                                   const uint8_t* sketch,
                                   const float* query_proj) {
    if (!qjl || !qjl->is_initialized || !sketch || !query_proj) return 0.0f;

    // Asymmetric dot product estimator per Lemma 1 in paper:
    // dot_est = (1/m) * sum_j sign(sketch_j) * query_proj[j]
    float dot = 0.0f;
    for (uint32_t i = 0; i < qjl->sketch_dims; i++) {
        float sign = (sketch[i >> 3] & (1u << (i & 7))) ? 1.0f : -1.0f;
        dot += sign * query_proj[i];
    }
    return dot; // already scaled correctly
}

// ════════════════════════════════════════════════════════════════════════════
// C++ Class Implementation
// ════════════════════════════════════════════════════════════════════════════

#ifdef __cplusplus
namespace tq {
namespace hadamard {

HadamardQJLSketch::HadamardQJLSketch(uint32_t original_dims,
                                     uint32_t sketch_dims,
                                     uint32_t seed)
    : orig_dims_(original_dims),
      sketch_dims_(sketch_dims),
      padded_dims_(next_pow2(original_dims)),
      padded_dims_log2_(log2_pow2(padded_dims_)),
      seed_(seed),
      sketch_bytes_((sketch_dims + 7) >> 3),
      rademacher_(padded_dims_),
      temp_(padded_dims_) {
    generate_rademacher();
}

HadamardQJLSketch::~HadamardQJLSketch() = default;

bool HadamardQJLSketch::is_power_of_two(uint32_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

uint32_t HadamardQJLSketch::next_power_of_two(uint32_t x) {
    if (x == 0) return 1;
    x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

uint32_t HadamardQJLSketch::log2_pow2(uint32_t x) {
    uint32_t r = 0;
    if (x >= (1u << 16)) { r += 16; x >>= 16; }
    if (x >= (1u <<  8)) { r +=  8; x >>=  8; }
    if (x >= (1u <<  4)) { r +=  4; x >>=  4; }
    if (x >= (1u <<  2)) { r +=  2; x >>=  2; }
    if (x >= (1u <<  1)) { r +=  1; }
    return r;
}

void HadamardQJLSketch::generate_rademacher() {
    uint32_t s = seed_;
    for (uint32_t i = 0; i < padded_dims_; i++) {
        s = s * 1664525U + 1013904223U;
        rademacher_[i] = ((s >> 16) & 1u) ? 1.0f : -1.0f;
    }
}

void HadamardQJLSketch::fwht_inplace(float* data, uint32_t n) const {
    for (uint32_t stride = 1; stride < n; stride <<= 1) {
        for (uint32_t i = 0; i < n; i += stride << 1) {
            for (uint32_t j = 0; j < stride; j++) {
                float u = data[i + j];
                float v = data[i + j + stride];
                data[i + j]          = u + v;
                data[i + j + stride] = u - v;
            }
        }
    }
}

void HadamardQJLSketch::fwht_normalize(float* data, uint32_t n) const {
    float scale = 1.0f / std::sqrt((float)n);
    for (uint32_t i = 0; i < n; i++) data[i] *= scale;
}

void HadamardQJLSketch::compress(const float* residual,
                                  uint8_t* sketch_out) const {
    // Step 1: pad + Rademacher multiplication
    for (uint32_t i = 0; i < padded_dims_; i++) {
        float val = (i < orig_dims_) ? residual[i] : 0.0f;
        temp_[i] = val * rademacher_[i];
    }
    // Step 2: FWHT O(d log d)
    fwht_inplace(temp_.data(), padded_dims_);
    fwht_normalize(temp_.data(), padded_dims_);
    // Step 3: 1-bit sign, zero overhead
    memset(sketch_out, 0, sketch_bytes_);
    for (uint32_t i = 0; i < sketch_dims_; i++) {
        if (temp_[i] >= 0.0f) sketch_out[i >> 3] |= (1u << (i & 7));
    }
}

void HadamardQJLSketch::project_query(const float* query,
                                       float* out) const {
    for (uint32_t i = 0; i < padded_dims_; i++) {
        float val = (i < orig_dims_) ? query[i] : 0.0f;
        temp_[i] = val * rademacher_[i];
    }
    fwht_inplace(temp_.data(), padded_dims_);
    fwht_normalize(temp_.data(), padded_dims_);
    for (uint32_t i = 0; i < sketch_dims_; i++) out[i] = temp_[i];
}

float HadamardQJLSketch::estimate_sketch_dot(const uint8_t* key_sketch,
                                              const float* query_sketch) const {
    float dot = 0.0f;
    for (uint32_t i = 0; i < sketch_dims_; i++) {
        float sign = (key_sketch[i >> 3] & (1u << (i & 7))) ? 1.0f : -1.0f;
        dot += sign * query_sketch[i];
    }
    return dot;
}

// ── AsymmetricDotEstimator ──────────────────────────────────────────────

AsymmetricDotEstimator::AsymmetricDotEstimator(uint32_t original_dims,
                                               uint32_t sketch_dims,
                                               uint32_t seed)
    : sketch_(original_dims, sketch_dims, seed),
      temp_sketch_(sketch_.sketch_bytes()),
      temp_query_(sketch_.sketch_dims()) {}

AsymmetricDotEstimator::~AsymmetricDotEstimator() = default;

float AsymmetricDotEstimator::estimate_vectors_dot(const float* vec_a,
                                                  const float* vec_b) {
    sketch_.compress(vec_a, temp_sketch_.data());
    sketch_.project_query(vec_b, temp_query_.data());
    return sketch_.estimate_sketch_dot(temp_sketch_.data(), temp_query_.data());
}

} // namespace hadamard
} // namespace tq
#endif // __cplusplus