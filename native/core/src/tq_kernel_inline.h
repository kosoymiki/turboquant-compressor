/**
 * TurboQuant Core — Inline Kernel Implementations (pure C stdlib)
 * No external C++ headers needed. No <string>, <vector>, <functional>.
 * Matches the API contracts of tq_rotation_kernel.h / tq_quantizer_kernel.h / tq_qjl_kernel.h.
 */

#ifndef TQ_KERNEL_INLINE_H
#define TQ_KERNEL_INLINE_H

#include <cmath>
#include <cstring>
#include <cstdlib>

// ── Utility ──────────────────────────────────────────────────────────────
static inline uint32_t tq_next_pow2(uint32_t x) {
    if (x == 0) return 1;
    x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

// ── Rotation: sign pattern init ────────────────────────────────────────────
static inline void tq_init_sign_pattern(float* sign_pattern, uint32_t padded, uint32_t seed) {
    uint32_t r = seed;
    for (uint32_t i = 0; i < padded; i++) {
        r = r * 1664525U + 1013904223U;
        sign_pattern[i] = ((r >> 16) & 0x8000) ? 1.0f : -1.0f;
    }
}

// ── Rotation: FWHT ──────────────────────────────────────────────────────
static inline void tq_fwht_inplace(float* data, uint32_t n) {
    for (uint32_t stride = 1; stride < n; stride <<= 1) {
        for (uint32_t i = 0; i < n; i += stride << 1) {
            for (uint32_t j = 0; j < stride; j++) {
                float u = data[i + j];
                float v = data[i + j + stride];
                data[i + j] = u + v;
                data[i + j + stride] = u - v;
            }
        }
    }
}

static inline void tq_fwht_normalize(float* data, uint32_t n) {
    float s = 1.0f / sqrtf((float)n);
    for (uint32_t i = 0; i < n; i++) data[i] *= s;
}

// ── Quantizer: Beta Lloyd-Max codebook ───────────────────────────────────
static inline void tq_beta_codebook_16(float* centroids) {
    const float c[16] = {
        -0.9821f, -0.8999f, -0.8086f, -0.7105f,
        -0.6071f, -0.4991f, -0.3870f, -0.2709f,
         0.2709f,  0.3870f,  0.4991f,  0.6071f,
         0.7105f,  0.8086f,  0.8999f,  0.9821f
    };
    for (int i = 0; i < 16; i++) centroids[i] = c[i];
}

static inline uint32_t tq_nearest_centroid(float value, const float* centroids, int levels) {
    float best_dist = 1e9f;
    uint32_t best = 0;
    for (int i = 0; i < levels; i++) {
        float d = fabsf(value - centroids[i]);
        if (d < best_dist) { best_dist = d; best = (uint32_t)i; }
    }
    return best;
}

static inline void tq_quantize_beta(const float* normalized, uint32_t* indices,
                                 const float* centroids, uint32_t dims, int levels) {
    for (uint32_t i = 0; i < dims; i++) {
        indices[i] = tq_nearest_centroid(normalized[i], centroids, levels);
    }
}

static inline void tq_dequantize_beta(const uint32_t* indices, float* decoded,
                                     const float* centroids, uint32_t dims) {
    for (uint32_t i = 0; i < dims; i++) decoded[i] = centroids[indices[i]];
}

// ── Bit packing: compress path ──────────────────────────────────────────
static inline void tq_pack_bits(const uint32_t* indices, uint32_t count,
                               int bits_per_value, uint8_t* out) {
    uint32_t mask = (1u << bits_per_value) - 1u;
    size_t bit_pos = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t v = indices[i] & mask;
        size_t byte_pos = bit_pos >> 3;
        int bit_off = (int)(bit_pos & 7);
        out[byte_pos] |= (uint8_t)(v << bit_off);
        if (bit_off + bits_per_value > 8)
            out[byte_pos + 1] |= (uint8_t)(v >> (8 - bit_off));
        bit_pos += bits_per_value;
    }
}

// ── Bit unpacking: search path ──────────────────────────────────────────
static inline void tq_unpack_bits(const uint8_t* bytes, uint32_t* indices,
                                 uint32_t count, int bits_per_value) {
    uint32_t mask = (1u << bits_per_value) - 1u;
    size_t bit_pos = 0;
    for (uint32_t i = 0; i < count; i++) {
        size_t byte_pos = bit_pos >> 3;
        int bit_off = (int)(bit_pos & 7);
        uint32_t v = (bytes[byte_pos] >> bit_off) & mask;
        if (bit_off + bits_per_value > 8)
            v |= ((uint32_t)bytes[byte_pos + 1] << (8 - bit_off)) & mask;
        indices[i] = v;
        bit_pos += bits_per_value;
    }
}

// ── QJL sign sketch ──────────────────────────────────────────────────
static inline void tq_qjl_sign_sketch(const float* residual, uint32_t residual_dims,
                                      uint8_t* sketch_out) {
    uint32_t padded = tq_next_pow2(residual_dims);
    float* tmp = (float*)malloc(padded * sizeof(float));
    memcpy(tmp, residual, residual_dims * sizeof(float));
    memset((uint8_t*)tmp + residual_dims * sizeof(float), 0,
           (padded - residual_dims) * sizeof(float));
    tq_fwht_inplace(tmp, padded);
    memset(sketch_out, 0, (padded + 7) >> 3);
    for (uint32_t i = 0; i < padded; i++) {
        if (tmp[i] >= 0.0f) sketch_out[i >> 3] |= (1u << (i & 7));
    }
    free(tmp);
}

#endif // TQ_KERNEL_INLINE_H
