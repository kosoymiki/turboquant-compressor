/**
 * TurboQuant Native Kernel — FWHT Rotation Engine
 * Low-level Hadamard rotation implementation.
 */

#include "tq_rotation_kernel.h"
#include <cstring>
#include <cstdlib>
#include <algorithm>

static inline int is_pow2(uint32_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

static inline uint32_t next_pow2(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

cl_int tq_rotation_init(tq_rotation_kernel_t* kernel, uint32_t dimensions, uint32_t seed, tq_rotation_mode_t /* mode */) {
    if (!kernel || dimensions == 0 || dimensions > TQ_ROTATION_MAX_DIMS) {
        return CL_INVALID_VALUE;
    }

    kernel->dimensions = dimensions;
    kernel->padded_dimensions = is_pow2(dimensions) ? dimensions : next_pow2(dimensions);
    kernel->seed = seed;

    void* tmp1;
    void* tmp2;
    posix_memalign(&tmp1, TQ_ROTATION_ALIGN, kernel->padded_dimensions * sizeof(float));
    posix_memalign(&tmp2, TQ_ROTATION_ALIGN, kernel->padded_dimensions * sizeof(float));
    kernel->sign_pattern = (float*)tmp1;
    kernel->temp_buffer = (float*)tmp2;

    if (!kernel->sign_pattern || !kernel->temp_buffer) {
        if (kernel->sign_pattern) free(kernel->sign_pattern);
        if (kernel->temp_buffer) free(kernel->temp_buffer);
        kernel->sign_pattern = nullptr;
        kernel->temp_buffer = nullptr;
        return CL_OUT_OF_HOST_MEMORY;
    }

    // Generate sign pattern
    for (uint32_t i = 0; i < kernel->padded_dimensions; i++) {
        uint32_t r = ((seed * 1664525U + 1013904223U) >> 16) & 0xFFFF;
        seed = (seed * 1664525U + 1013904223U);
        kernel->sign_pattern[i] = (r < 32768) ? -1.0f : 1.0f;
    }

    kernel->is_initialized = 1;
    return CL_SUCCESS;
}

cl_int tq_rotation_shutdown(tq_rotation_kernel_t* kernel) {
    if (!kernel) return CL_INVALID_VALUE;

    if (kernel->sign_pattern) {
        free(kernel->sign_pattern);
        kernel->sign_pattern = nullptr;
    }
    if (kernel->temp_buffer) {
        free(kernel->temp_buffer);
        kernel->temp_buffer = nullptr;
    }

    kernel->is_initialized = 0;
    return CL_SUCCESS;
}

static void fwht_inplace_impl(float* data, uint32_t n) {
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

static void fwht_normalize(float* data, uint32_t n) {
    float scale = 1.0f / std::sqrt((float)n);
    for (uint32_t i = 0; i < n; i++) {
        data[i] *= scale;
    }
}

cl_int tq_rotation_execute(tq_rotation_kernel_t* kernel, const float* input, float* output, uint32_t count) {
    if (!kernel || !kernel->is_initialized || !input || !output) {
        return CL_INVALID_VALUE;
    }

    uint32_t padded = kernel->padded_dimensions;

    if (kernel->mode == TQ_ROTATION_MODE_NONE) {
        std::memcpy(output, input, std::min(count, kernel->dimensions) * sizeof(float));
        return CL_SUCCESS;
    }

    // Apply sign flip + FWHT
    for (uint32_t idx = 0; idx < count && idx < kernel->dimensions; idx++) {
        // Load into temp with padding
        for (uint32_t i = 0; i < padded; i++) {
            kernel->temp_buffer[i] = (i < kernel->dimensions) ? input[idx * kernel->dimensions + i] : 0.0f;
        }

        // Apply sign flip
        if (kernel->mode == TQ_ROTATION_MODE_FWHT_SIGN) {
            for (uint32_t i = 0; i < padded; i++) {
                kernel->temp_buffer[i] *= kernel->sign_pattern[i];
            }
        }

        // FWHT
        fwht_inplace_impl(kernel->temp_buffer, padded);
        fwht_normalize(kernel->temp_buffer, padded);

        // Copy back
        for (uint32_t i = 0; i < kernel->dimensions; i++) {
            output[idx * kernel->dimensions + i] = kernel->temp_buffer[i];
        }
    }

    return CL_SUCCESS;
}

cl_int tq_rotation_execute_inplace(tq_rotation_kernel_t* kernel, float* data, uint32_t count) {
    if (!kernel || !kernel->is_initialized || !data) {
        return CL_INVALID_VALUE;
    }

    uint32_t padded = kernel->padded_dimensions;

    if (kernel->mode == TQ_ROTATION_MODE_NONE) {
        return CL_SUCCESS;
    }

    for (uint32_t idx = 0; idx < count && idx < kernel->dimensions; idx++) {
        float* vec = data + idx * kernel->dimensions;

        // Apply sign flip in-place
        if (kernel->mode == TQ_ROTATION_MODE_FWHT_SIGN) {
            for (uint32_t i = 0; i < padded; i++) {
                if (i < kernel->dimensions) {
                    vec[i] *= kernel->sign_pattern[i];
                }
            }
        }

        // Copy to temp
        for (uint32_t i = 0; i < padded; i++) {
            kernel->temp_buffer[i] = (i < kernel->dimensions) ? vec[i] : 0.0f;
        }

        // FWHT
        fwht_inplace_impl(kernel->temp_buffer, padded);
        fwht_normalize(kernel->temp_buffer, padded);

        // Copy back
        for (uint32_t i = 0; i < kernel->dimensions; i++) {
            vec[i] = kernel->temp_buffer[i];
        }
    }

    return CL_SUCCESS;
}

cl_int tq_rotation_inverse(tq_rotation_kernel_t* kernel, const float* rotated, float* original) {
    if (!kernel || !kernel->is_initialized || !rotated || !original) {
        return CL_INVALID_VALUE;
    }

    // FWHT is self-inverse (up to normalization)
    return tq_rotation_execute(kernel, rotated, original, 1);
}

#ifdef __cplusplus

namespace tq {
namespace kernel {

RotationKernel::RotationKernel(uint32_t dimensions, uint32_t seed, tq_rotation_mode_t mode)
    : dims_(dimensions), seed_(seed), mode_(mode) {

    padded_dims_ = is_power_of_two(dimensions) ? dimensions : next_power_of_two(dimensions);
    sign_pattern_.resize(padded_dims_);
    temp_buffer_.resize(padded_dims_);

    compute_sign_pattern();
}

RotationKernel::~RotationKernel() = default;

RotationKernel::RotationKernel(RotationKernel&&) noexcept = default;
RotationKernel& RotationKernel::operator=(RotationKernel&&) noexcept = default;

bool RotationKernel::is_power_of_two(uint32_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

uint32_t RotationKernel::next_power_of_two(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

void RotationKernel::compute_sign_pattern() {
    uint32_t s = seed_;
    for (uint32_t i = 0; i < padded_dims_; i++) {
        s = s * 1664525U + 1013904223U;
        uint32_t r = (s >> 16) & 0xFFFF;
        sign_pattern_[i] = (r < 32768) ? -1.0f : 1.0f;
    }
}

void RotationKernel::fwht_inplace(float* data, uint32_t n) {
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

void RotationKernel::apply_sign_flip(float* data) {
    for (uint32_t i = 0; i < padded_dims_; i++) {
        if (i < dims_) {
            data[i] *= sign_pattern_[i];
        }
    }
}

void RotationKernel::rotate(const float* input, float* output) {
    if (mode_ == TQ_ROTATION_MODE_NONE) {
        std::memcpy(output, input, dims_ * sizeof(float));
        return;
    }

    // Load with padding
    for (uint32_t i = 0; i < padded_dims_; i++) {
        temp_buffer_[i] = (i < dims_) ? input[i] : 0.0f;
    }

    // Sign flip
    if (mode_ == TQ_ROTATION_MODE_FWHT_SIGN) {
        apply_sign_flip(temp_buffer_.data());
    }

    // FWHT + normalize
    fwht_inplace(temp_buffer_.data(), padded_dims_);
    float scale = 1.0f / std::sqrt((float)padded_dims_);
    for (uint32_t i = 0; i < padded_dims_; i++) {
        temp_buffer_[i] *= scale;
    }

    // Copy back
    for (uint32_t i = 0; i < dims_; i++) {
        output[i] = temp_buffer_[i];
    }
}

void RotationKernel::rotate_inplace(float* data) {
    if (mode_ == TQ_ROTATION_MODE_NONE) {
        return;
    }

    apply_sign_flip(data);

    // Pad remaining
    for (uint32_t i = dims_; i < padded_dims_; i++) {
        temp_buffer_[i] = 0.0f;
    }
    for (uint32_t i = 0; i < dims_; i++) {
        temp_buffer_[i] = data[i];
    }

    fwht_inplace(temp_buffer_.data(), padded_dims_);

    float scale = 1.0f / std::sqrt((float)padded_dims_);
    for (uint32_t i = 0; i < padded_dims_; i++) {
        temp_buffer_[i] *= scale;
    }

    for (uint32_t i = 0; i < dims_; i++) {
        data[i] = temp_buffer_[i];
    }
}

void RotationKernel::inverse_rotate(const float* rotated, float* original) {
    rotate(rotated, original);
}

void RotationKernel::rotate_batch(const float* vectors, float* outputs, uint32_t batch_size) {
    if (mode_ == TQ_ROTATION_MODE_NONE) {
        std::memcpy(outputs, vectors, batch_size * dims_ * sizeof(float));
        return;
    }

    for (uint32_t b = 0; b < batch_size; b++) {
        rotate(vectors + b * dims_, outputs + b * dims_);
    }
}

} // namespace kernel
} // namespace tq

#endif // __cplusplus