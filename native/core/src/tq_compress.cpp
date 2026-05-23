/**
 * TurboQuant Core — End-to-End Compression Pipeline
 *
 * v4.5.4: TS format.ts v3 binary compatibility
 *   Byte 28: codebook type (bits 0-1), reserved (bits 2-7)
 *   Bytes 32-55: TS V3 explicit offset fields (headerLength, normsOffset, quantizedOffset, etc.)
 *   CRC at byte 64: payload = norms + quantized (not qjl), per TS decodeV3 validation
 *   QJL NOT in CRC, per TS encodeV3: crc32(normsOffset .. qjlOffset)
 */

#include "tq_core.h"
#include "tq_kernel_inline.h"
#include "tq_base64.h"
#include "tq_beta_sphere.h"

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>

// ── Helpers ────────────────────────────────────────────────────────────────
static uint64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// ── Size estimation ──────────────────────────────────────────────────────
size_t tq_estimate_compressed_size(uint32_t N, uint32_t dims, uint8_t bits, uint8_t qjl_dims) {
    uint32_t pad = 1;
    while (pad < dims) pad <<= 1;
    size_t header  = 80;
    size_t norms   = (size_t)N * 4;
    size_t packed  = (size_t)N * pad * bits / 8;
    size_t qjl     = (qjl_dims > 0) ? (((size_t)N * qjl_dims + 7) / 8) : 0;
    return header + norms + packed + qjl + 64;
}

// ── Beta Lloyd-Max encode ────────────────────────────────────────────────
static float encode_beta_lm(const float* norm, uint32_t D,
                           const double* cents, int n_levels,
                           uint32_t* indices) {
    float mse = 0.0f;
    for (uint32_t i = 0; i < D; i++) {
        float v = norm[i];
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        int lo = 0, hi = n_levels - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (cents[mid] < v) lo = mid + 1; else hi = mid;
        }
        int best = lo;
        float best_dist = 1e9f;
        for (int j = std::max(0, lo - 1); j <= std::min(n_levels - 1, lo + 1); j++) {
            float d = v - (float)cents[j];
            if (d < 0) d = -d;
            if (d < best_dist) { best_dist = d; best = j; }
        }
        indices[i] = (uint32_t)best;
        float decoded = (float)cents[best];
        float diff = v - decoded;
        mse += diff * diff;
    }
    return mse;
}

// ── QJL residual sketch: residual = normalized - decoded ───────────────────
static void qjl_residual_sketch(const float* normed, const float* decoded,
                                 uint32_t D, uint32_t pad,
                                 float* tmp, uint8_t* sketch_out, uint8_t qjl_dims) {
    for (uint32_t i = 0; i < D; i++) tmp[i] = normed[i] - decoded[i];
    for (uint32_t i = D; i < pad; i++) tmp[i] = 0.0f;
    tq_fwht_inplace(tmp, pad);
    tq_fwht_normalize(tmp, pad);
    std::memset(sketch_out, 0, ((size_t)qjl_dims + 7) / 8);
    for (uint32_t i = 0; i < qjl_dims; i++)
        if (tmp[i] >= 0.0f) sketch_out[i >> 3] |= (1u << (i & 7));
}

// ── Beta codebook dequantize ─────────────────────────────────────────────
static void dequantize_beta(const uint32_t* idx, uint32_t D,
                           const double* cents, float* out) {
    for (uint32_t i = 0; i < D; i++) out[i] = (float)cents[idx[i]];
}

extern "C" {

int tq_compress(const float* vecs, uint32_t N, uint32_t D, uint8_t bits,
                uint32_t seed, uint8_t qjl_dims, uint8_t codebook_type,
                char* out_b64, size_t* out_len,
                tq_compress_stats* stats) {
    if (!vecs || !out_b64 || !out_len) return TQ_ERR_NULL;
    if (D == 0 || N == 0) return TQ_ERR_DIM;
    if (bits < 1 || bits > 8) return TQ_ERR_INTERNAL;
    if (codebook_type > 1) return TQ_ERR_INTERNAL;

    uint64_t t0 = now_ns();

    // ── Padding ──────────────────────────────────────────────────────
    uint32_t pad = 1;
    while (pad < D) pad <<= 1;
    uint8_t levels = (uint8_t)(1u << bits);   // 16 for 4-bit
    float levels_m1 = (float)(levels - 1);

    size_t packed_per = (size_t)pad * bits / 8;
    size_t qjl_per_vec = (qjl_dims > 0) ? ((qjl_dims + 7) / 8) : 0;

    uint32_t HEADER_SIZE = 80;
    uint32_t norms_off   = HEADER_SIZE;
    uint32_t norms_len  = N * 4;
    uint32_t quant_off  = norms_off + norms_len;
    uint32_t quant_len = (uint32_t)((uint64_t)N * packed_per);
    uint32_t qjl_off = quant_off + quant_len;
    uint32_t qjl_len = qjl_dims > 0 ? (uint32_t)(((uint64_t)N * qjl_dims + 7) / 8) : 0;
    size_t bin_size = (size_t)qjl_off + qjl_len;

    uint8_t* bin = (uint8_t*)std::malloc(bin_size);
    if (!bin) return TQ_ERR_ALLOC;
    std::memset(bin, 0, bin_size);

    // ── Write TS-compatible V3 header ────────────────────────────────────
    bin[0] = 'T'; bin[1] = 'Q'; bin[2] = 'M'; bin[3] = 'C';
    uint32_t ver3 = 3;
    std::memcpy(bin + 4,  &ver3,    4);   // version
    std::memcpy(bin + 8,  &D,       4);   // dimensions
    std::memcpy(bin + 12, &pad,     4);   // paddedDimensions
    std::memcpy(bin + 16, &N,       4);   // vectorCount
    bin[20] = bits;                       // bitsPerValue
    std::memcpy(bin + 24, &seed,    4);   // rotationSeed
    bin[28] = codebook_type;              // flags = codebook_type (TS: encodeCodebookType)
    bin[29] = 0;                          // byte 29: reserved (TS writes 0)
    std::memcpy(bin + 32, &HEADER_SIZE, 4);  // headerLength = 80
    std::memset(bin + 36, 0, 4);           // bytes 36-39: reserved
    std::memcpy(bin + 40, &norms_off,  4);   // normsOffset
    std::memcpy(bin + 44, &norms_len,  4);   // normsLength
    std::memcpy(bin + 48, &quant_off, 4);   // quantizedOffset
    std::memcpy(bin + 52, &quant_len, 4);   // quantizedLength
    std::memcpy(bin + 56, &qjl_off,   4);   // qjlOffset
    std::memcpy(bin + 60, &qjl_len,   4);   // qjlLength
    // bytes 64-67: CRC written after loop
    // bytes 68-79: reserved

    // ── Beta codebook precompute ──────────────────────────────────────────
    tq_beta_codebook bcb;
    const double* cents = nullptr;
    int n_levels = levels;
    if (codebook_type == TQ_CODEBOOK_BETA) {
        bcb = tq_compute_lloyd_max_beta_codebook((int)D, (int)bits, 200, 1e-12);
        cents = bcb.centroids;
        n_levels = bcb.n_clusters;
    }

    // ── Working buffers ────────────────────────────────────────────────
    uint32_t* idx  = (uint32_t*)std::calloc(pad, 4);
    float* rot   = (float*)std::malloc(pad * 4);
    float* sign  = (float*)std::calloc(pad, 4);
    float* nvec  = (float*)std::malloc(pad * 4);   // normalized rotated
    float* decv  = (float*)malloc(pad * 4);   // dequantized

    if (!idx || !rot || !sign || !nvec || !decv) {
        if (codebook_type == TQ_CODEBOOK_BETA) tq_free_beta_codebook(&bcb);
        std::free(bin); std::free(idx); std::free(rot);
        std::free(sign); std::free(nvec); std::free(decv);
        return TQ_ERR_ALLOC;
    }

    tq_init_sign_pattern(sign, pad, seed);
    float* norms_arr = (float*)(bin + norms_off);
    uint8_t* quant_arr = bin + quant_off;
    uint8_t* qjl_arr  = qjl_dims > 0 ? (bin + qjl_off) : nullptr;

    // ── Per-vector pipeline ────────────────────────────────────────────
    for (uint32_t vi = 0; vi < N; vi++) {
        const float* in = vecs + (size_t)vi * D;

        // 1) Copy + sign pattern
        std::memset(rot, 0, pad * 4);
        for (uint32_t i = 0; i < D; i++) rot[i] = in[i] * sign[i];
        for (uint32_t i = D; i < pad; i++) rot[i] = 0.0f;

        // 2) FWHT + normalize
        tq_fwht_inplace(rot, pad);
        tq_fwht_normalize(rot, pad);

        // 3) Norm + normalize
        float nsq = 0.0f;
        for (uint32_t i = 0; i < D; i++) nsq += rot[i] * rot[i];
        float norm = (nsq > 1e-9f) ? std::sqrt(nsq) : 1.0f;
        norms_arr[vi] = norm;
        float inv = 1.0f / norm;
        for (uint32_t i = 0; i < D; i++) nvec[i] = rot[i] * inv;

        // 4) Quantize
        if (codebook_type == TQ_CODEBOOK_BETA && cents) {
            encode_beta_lm(nvec, D, cents, n_levels, idx);
        } else {
            for (uint32_t i = 0; i < D; i++) {
                float v = nvec[i];
                if (v >  1.0f) v =  1.0f;
                if (v < -1.0f) v = -1.0f;
                float nv = (v + 1.0f) * levels_m1 * 0.5f;
                idx[i] = (uint32_t)(nv + 0.5f);
            }
        }

        // 5) Bit-pack
        tq_pack_bits(idx, D, (int)bits, quant_arr + (size_t)vi * packed_per);

        // 6) QJL residual sketch: residual = normalized - decoded
        if (qjl_dims > 0 && qjl_arr) {
            if (codebook_type == TQ_CODEBOOK_BETA && cents) {
                dequantize_beta(idx, D, cents, decv);
            } else {
                for (uint32_t i = 0; i < D; i++)
                    decv[i] = ((float)idx[i] / levels_m1) * 2.0f - 1.0f;
            }
            qjl_residual_sketch(nvec, decv, D, pad, rot,
                               qjl_arr + (size_t)vi * qjl_per_vec, qjl_dims);
        }
    }

    // ── CRC32 of payload (norms + quantized, NOT qjl) ────────────────────────
    uint32_t crc = tq::crc32((const uint8_t*)(bin + norms_off), quant_off + quant_len - norms_off);
    std::memcpy(bin + 64, &crc, 4);

    // ── Encode → base64 ────────────────────────────────────────────────
    std::string b64 = tq::core::base64::encode(bin, bin_size);
    size_t b64_len = b64.size();
    if (b64_len + 1 > *out_len) {
        if (codebook_type == TQ_CODEBOOK_BETA) tq_free_beta_codebook(&bcb);
        std::free(bin); std::free(idx); std::free(rot);
        std::free(sign); std::free(nvec); std::free(decv);
        return TQ_ERR_ALLOC;
    }
    std::memcpy(out_b64, b64.c_str(), b64_len);
    out_b64[b64_len] = '\0';
    *out_len = b64_len;

    // ── Stats ───────────────────────────────────────────────────────
    if (stats) {
        std::memset(stats, 0, sizeof(*stats));
        stats->magic = TQ_CORE_MAGIC;
        stats->version = 3;
        stats->original_bytes = N * D * 4;
        stats->compressed_bytes = (uint32_t)bin_size;
        stats->compression_ratio = (float)(N * D * 4) / (float)bin_size;
        stats->vector_count = N;
        stats->dimensions = D;
        stats->padded_dimensions = pad;
        stats->bits_per_value = bits;
        stats->codebook_type = codebook_type;
        stats->qjl_dims = qjl_dims;
        stats->qjl_bytes = qjl_len;
        stats->has_qjl = qjl_dims > 0 ? 1 : 0;
        uint64_t t1 = now_ns();
        stats->compress_ns = (uint32_t)(t1 - t0);
    }

    if (codebook_type == TQ_CODEBOOK_BETA) tq_free_beta_codebook(&bcb);
    std::free(bin); std::free(idx); std::free(rot);
    std::free(sign); std::free(nvec); std::free(decv);
    return TQ_OK;
}

const char* tq_core_version(void) { return TQ_CORE_VERSION; }

} // extern "C"
