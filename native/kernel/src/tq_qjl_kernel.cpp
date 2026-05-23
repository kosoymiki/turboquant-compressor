/**
 * TurboQuant Native Kernel -- QJL Residual Correction
 * Johnson-Lindenstrauss projection for unbiased dot-product estimation.
 * Based on Zandieh et al., ICML 2024.
 */
#include <CL/cl.h>

#include "tq_qjl_kernel.h"
#include <cstring>
#include <cmath>
#include <algorithm>

static uint32_t next_pow2(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

cl_int tq_qjl_init(tq_qjl_kernel_t* qjl, uint32_t orig_dims, uint32_t target_dims,
                   uint32_t seed, uint8_t bits, uint8_t use_hadamard) {
    if (!qjl || orig_dims == 0 || target_dims == 0 || target_dims > TQ_QJL_MAX_TARGET_DIMS) {
        return CL_INVALID_VALUE;
    }

    qjl->original_dims = orig_dims;
    qjl->target_dims = target_dims;
    qjl->seed = seed;
    qjl->bits_per_value = bits;
    qjl->use_hadamard = use_hadamard;

    uint32_t padded = use_hadamard ? next_pow2(orig_dims) : orig_dims;

    void* tmpbuf1;
    void* tmpbuf2;
    int err1 = posix_memalign(&tmpbuf1, 32, target_dims * padded * sizeof(float));
    int err2 = posix_memalign(&tmpbuf2, 32, target_dims * sizeof(float));
    qjl->projection_matrix = (float*)tmpbuf1;
    qjl->sketch_buffer = (float*)tmpbuf2;

    if (err1 != 0 || err2 != 0 || !qjl->projection_matrix || !qjl->sketch_buffer) {
        if (qjl->projection_matrix) free(qjl->projection_matrix);
        if (qjl->sketch_buffer) free(qjl->sketch_buffer);
        qjl->projection_matrix = nullptr;
        qjl->sketch_buffer = nullptr;
        return CL_OUT_OF_HOST_MEMORY;
    }

    uint32_t s = seed;
    for (uint32_t i = 0; i < target_dims * padded; i++) {
        s = s * 1664525U + 1013904223U;
        float u1 = (float)(s >> 16) / 65536.0f;
        s = s * 1664525U + 1013904223U;
        float u2 = (float)(s >> 16) / 65536.0f;
        float gaussian = std::sqrt(-2.0f * std::log(u1 + 1e-6f)) * std::cos(2.0f * 3.14159265359f * u2);
        qjl->projection_matrix[i] = gaussian / std::sqrt((float)target_dims);
    }

    qjl->is_initialized = 1;
    return CL_SUCCESS;
}

cl_int tq_qjl_shutdown(tq_qjl_kernel_t* qjl) {
    if (!qjl) return CL_INVALID_VALUE;
    if (qjl->projection_matrix) { free(qjl->projection_matrix); qjl->projection_matrix = nullptr; }
    if (qjl->sketch_buffer) { free(qjl->sketch_buffer); qjl->sketch_buffer = nullptr; }
    qjl->is_initialized = 0;
    return CL_SUCCESS;
}

cl_int tq_qjl_project(const tq_qjl_kernel_t* qjl, const float* vector, float* projected) {
    if (!qjl || !qjl->is_initialized || !vector || !projected) return CL_INVALID_VALUE;
    for (uint32_t j = 0; j < qjl->target_dims; j++) {
        float dot = 0.0f;
        for (uint32_t i = 0; i < qjl->original_dims; i++) {
            dot += qjl->projection_matrix[j * qjl->original_dims + i] * vector[i];
        }
        qjl->sketch_buffer[j] = dot;
    }
    std::memcpy(projected, qjl->sketch_buffer, qjl->target_dims * sizeof(float));
    return CL_SUCCESS;
}

cl_int tq_qjl_compress(const tq_qjl_kernel_t* qjl, const float* residual, uint8_t* sketch) {
    if (!qjl || !qjl->is_initialized || !residual || !sketch) return CL_INVALID_VALUE;
    float projected[128];
    tq_qjl_project(qjl, residual, projected);
    uint32_t levels = 1 << qjl->bits_per_value;
    float step = 2.0f / (float)levels;
    uint32_t bit_pos = 0;
    std::memset(sketch, 0, ((qjl->target_dims * qjl->bits_per_value) + 7) / 8);
    for (uint32_t i = 0; i < qjl->target_dims; i++) {
        float normalized = (projected[i] + 1.0f) / step;
        uint32_t idx = (uint32_t)(normalized + 0.5f);
        if ((int)idx < 0) idx = 0;
        if (idx >= levels) idx = levels - 1;
        for (uint32_t b = 0; b < qjl->bits_per_value; b++) {
            uint32_t byte_idx = bit_pos / 8;
            uint32_t bit_idx = bit_pos % 8;
            if ((idx >> b) & 1) sketch[byte_idx] |= (1 << bit_idx);
            bit_pos++;
        }
    }
    return CL_SUCCESS;
}

cl_int tq_qjl_estimate_dot(const tq_qjl_kernel_t* qjl, const uint8_t* sketch_a, const uint8_t* sketch_b,
                           float* estimate, const float* /* scale_a */, const float* /* scale_b */) {
    if (!qjl || !qjl->is_initialized || !sketch_a || !sketch_b || !estimate) return CL_INVALID_VALUE;
    float vals_a[128], vals_b[128];
    uint32_t bit_pos = 0;
    for (uint32_t i = 0; i < qjl->target_dims; i++) {
        uint32_t idx_a = 0, idx_b = 0;
        for (uint32_t b = 0; b < qjl->bits_per_value; b++) {
            uint32_t byte_idx = bit_pos / 8;
            uint32_t bit_idx = bit_pos % 8;
            if (sketch_a[byte_idx] & (1 << bit_idx)) idx_a |= (1 << b);
            if (sketch_b[byte_idx] & (1 << bit_idx)) idx_b |= (1 << b);
            bit_pos++;
        }
        uint32_t levels = 1 << qjl->bits_per_value;
        float step = 2.0f / (float)levels;
        vals_a[i] = -1.0f + ((float)idx_a + 0.5f) * step;
        vals_b[i] = -1.0f + ((float)idx_b + 0.5f) * step;
    }
    float dot = 0.0f;
    for (uint32_t i = 0; i < qjl->target_dims; i++) dot += vals_a[i] * vals_b[i];
    *estimate = dot / (float)qjl->target_dims;
    return CL_SUCCESS;
}

float tq_qjl_approximation_error(const tq_qjl_kernel_t* qjl, const float* original, const float* approximated) {
    if (!qjl || !qjl->is_initialized || !original || !approximated) return 0.0f;
    float total_mse = 0.0f;
    for (uint32_t i = 0; i < qjl->original_dims; i++) {
        float diff = original[i] - approximated[i];
        total_mse += diff * diff;
    }
    return std::sqrt(total_mse / (float)qjl->original_dims);
}

#ifdef __cplusplus
namespace tq {
namespace kernel {

QJLProjector::QJLProjector(uint32_t orig_dims, uint32_t target_dims, uint32_t seed)
    : orig_dims_(orig_dims), target_dims_(target_dims), seed_(seed), use_hadamard_(true) {
    projection_matrix_.resize(target_dims * orig_dims);
    uint32_t s = seed;
    for (size_t i = 0; i < projection_matrix_.size(); i++) {
        s = s * 1664525U + 1013904223U;
        float u1 = (float)(s >> 16) / 65536.0f;
        s = s * 1664525U + 1013904223U;
        float u2 = (float)(s >> 16) / 65536.0f;
        float gaussian = std::sqrt(-2.0f * std::log(u1 + 1e-6f)) * std::cos(2.0f * 3.14159265359f * u2);
        projection_matrix_[i] = gaussian / std::sqrt((float)target_dims);
    }
}

QJLProjector::~QJLProjector() = default;
QJLProjector::QJLProjector(QJLProjector&&) noexcept = default;
QJLProjector& QJLProjector::operator=(QJLProjector&&) noexcept = default;

bool QJLProjector::is_power_of_two(uint32_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

uint32_t QJLProjector::next_power_of_two(uint32_t x) {
    if (x == 0) return 1;
    x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

void QJLProjector::apply_projection(const float* input, float* output) {
    for (uint32_t j = 0; j < target_dims_; j++) {
        float dot = 0.0f;
        for (uint32_t i = 0; i < orig_dims_; i++) {
            dot += projection_matrix_[j * orig_dims_ + i] * input[i];
        }
        output[j] = dot;
    }
}

void QJLProjector::project(const float* input, float* output) { apply_projection(input, output); }

void QJLProjector::inverse_project(const float* projected, float* output) {
    for (uint32_t i = 0; i < orig_dims_; i++) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < target_dims_; j++) {
            dot += projection_matrix_[j * orig_dims_ + i] * projected[j];
        }
        output[i] = dot * (float)orig_dims_ / (float)target_dims_;
    }
}

float QJLProjector::compute_dot_product(const float* vec_a, const float* vec_b) const {
    float dot = 0.0f;
    for (uint32_t i = 0; i < orig_dims_; i++) dot += vec_a[i] * vec_b[i];
    return dot;
}

QJLSketchCompressor::QJLSketchCompressor(uint32_t target_dims, uint8_t bits)
    : target_dims_(target_dims), bits_(bits) {
    sketch_bytes_ = ((target_dims * bits) + 7) / 8;
}

QJLSketchCompressor::~QJLSketchCompressor() = default;

void QJLSketchCompressor::compress(const float* projected, uint8_t* sketch) {
    uint32_t bit_pos = 0;
    std::memset(sketch, 0, sketch_bytes_);
    for (uint32_t i = 0; i < target_dims_; i++) {
        uint32_t levels = 1 << bits_;
        float step = 2.0f / (float)levels;
        float normalized = (projected[i] + 1.0f) / step;
        int idx = (int)(normalized + 0.5f);
        if (idx < 0) idx = 0;
        if (idx >= (int)levels) idx = levels - 1;
        for (uint32_t b = 0; b < bits_; b++) {
            uint32_t byte_idx = bit_pos / 8;
            uint32_t bit_idx = bit_pos % 8;
            if ((idx >> b) & 1) sketch[byte_idx] |= (1 << bit_idx);
            bit_pos++;
        }
    }
}

uint8_t* QJLSketchCompressor::compress(const float* projected) {
    uint8_t* sketch = new uint8_t[sketch_bytes_];
    compress(projected, sketch);
    return sketch;
}

void QJLSketchCompressor::decompress(const uint8_t* sketch, float* projected) {
    uint32_t bit_pos = 0;
    for (uint32_t i = 0; i < target_dims_; i++) {
        uint32_t idx = 0;
        for (uint32_t b = 0; b < bits_; b++) {
            uint32_t byte_idx = bit_pos / 8;
            uint32_t bit_idx = bit_pos % 8;
            if (sketch[byte_idx] & (1 << bit_idx)) idx |= (1 << b);
            bit_pos++;
        }
        uint32_t levels = 1 << bits_;
        float step = 2.0f / (float)levels;
        projected[i] = -1.0f + ((float)idx + 0.5f) * step;
    }
}

float* QJLSketchCompressor::decompress(const uint8_t* sketch) {
    float* projected = new float[target_dims_];
    decompress(sketch, projected);
    return projected;
}

QJLResidualEstimator::QJLResidualEstimator(uint32_t orig_dims, uint32_t target_dims,
                                           uint8_t sketch_bits, uint32_t seed)
    : projector_(orig_dims, target_dims, seed),
      compressor_(target_dims, sketch_bits) {
}

QJLResidualEstimator::~QJLResidualEstimator() = default;

QJLResidualEstimator::SketchResult QJLResidualEstimator::compress(const float* residual) {
    SketchResult result;
    result.sketch = compressor_.compress(residual);
    result.sketch_bytes = compressor_.sketch_bytes();
    float norm = 0.0f;
    for (uint32_t i = 0; i < original_dims(); i++) norm += residual[i] * residual[i];
    result.scale = std::sqrt(norm);
    last_scale_a_ = result.scale;
    return result;
}

void QJLResidualEstimator::free_sketch(SketchResult* result) {
    delete[] result->sketch;
    result->sketch = nullptr;
}

float QJLResidualEstimator::estimate_dot(const float* residual_a, const float* residual_b) {
    SketchResult a = compress(residual_a);
    SketchResult b;
    b.sketch = compressor_.compress(residual_b);
    b.sketch_bytes = compressor_.sketch_bytes();

    float* da = compressor_.decompress(a.sketch);
    float* db = compressor_.decompress(b.sketch);

    float dot = 0.0f;
    for (uint32_t i = 0; i < target_dims(); i++) dot += da[i] * db[i];

    delete[] da;
    delete[] db;
    free_sketch(&a);
    free_sketch(&b);

    return dot / (float)target_dims();
}

float QJLResidualEstimator::estimate_dot(const SketchResult& sketch_a, const SketchResult& sketch_b) {
    float* da = compressor_.decompress(sketch_a.sketch);
    float* db = compressor_.decompress(sketch_b.sketch);
    float dot = 0.0f;
    for (uint32_t i = 0; i < target_dims(); i++) dot += da[i] * db[i];
    delete[] da;
    delete[] db;
    return dot / (float)target_dims();
}

QJLDotProductEstimator::QJLDotProductEstimator(uint32_t dims, uint32_t sketch_dims, uint32_t seed)
    : estimator_(dims, sketch_dims, 4, seed), last_error_(0.0f), call_count_(0) {
}

QJLDotProductEstimator::~QJLDotProductEstimator() = default;

float QJLDotProductEstimator::estimate(const float* vec_a, const float* vec_b, bool use_sketch) {
    call_count_++;
    if (!use_sketch) {
        last_error_ = 0.0f;
        return exact_dot(vec_a, vec_b);
    }
    float est = estimator_.estimate_dot(vec_a, vec_b);
    float exact = exact_dot(vec_a, vec_b);

    last_error_ = std::abs(est - exact);
    return est;
}

float QJLDotProductEstimator::exact_dot(const float* vec_a, const float* vec_b) {
    float dot = 0.0f;
    for (uint32_t i = 0; i < estimator_.original_dims(); i++) dot += vec_a[i] * vec_b[i];
    return dot;
}

void QJLDotProductEstimator::reset_stats() {
    last_error_ = 0.0f;
    call_count_ = 0;
}

} // namespace kernel
} // namespace tq
#endif // __cplusplus
