/**
 * TurboQuant Native Kernel — FWHT Rotation Engine
 *
 * Low-level Hadamard rotation using Fast Walsh-Hadamard Transform.
 * Replaces TypeScript rotation.ts for precise memory register work.
 */

#include <CL/cl.h>
#ifndef TQ_ROTATION_KERNEL_H
#define TQ_ROTATION_KERNEL_H

#include <cstdint>
#include <cstddef>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ_ROTATION_MAX_DIMS 8192
#define TQ_ROTATION_ALIGN 32

typedef enum {
    TQ_ROTATION_MODE_FWHT_SIGN = 0,
    TQ_ROTATION_MODE_NONE = 1,
    TQ_ROTATION_MODE_GAUSSIAN = 2
} tq_rotation_mode_t;

typedef struct {
    uint32_t dimensions;
    uint32_t padded_dimensions;
    uint32_t seed;
    float* sign_pattern;
    float* temp_buffer;
    uint8_t is_initialized;
    tq_rotation_mode_t mode;
} tq_rotation_kernel_t;

cl_int tq_rotation_init(tq_rotation_kernel_t* kernel, uint32_t dimensions, uint32_t seed, tq_rotation_mode_t mode);
cl_int tq_rotation_shutdown(tq_rotation_kernel_t* kernel);
cl_int tq_rotation_execute(tq_rotation_kernel_t* kernel, const float* input, float* output, uint32_t count);
cl_int tq_rotation_execute_inplace(tq_rotation_kernel_t* kernel, float* data, uint32_t count);
cl_int tq_rotation_inverse(tq_rotation_kernel_t* kernel, const float* rotated, float* original);

float* tq_rotation_rotate_vector(tq_rotation_kernel_t* kernel, const float* vector);
float* tq_rotation_rotate_batch(tq_rotation_kernel_t* kernel, const float* vectors, uint32_t batch_size);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <memory>
#include <vector>
#include <functional>

namespace tq {
namespace kernel {

class RotationKernel {
public:
    explicit RotationKernel(uint32_t dimensions, uint32_t seed = 0,
                            tq_rotation_mode_t mode = TQ_ROTATION_MODE_FWHT_SIGN);
    ~RotationKernel();

    RotationKernel(const RotationKernel&) = delete;
    RotationKernel& operator=(const RotationKernel&) = delete;
    RotationKernel(RotationKernel&&) noexcept;
    RotationKernel& operator=(RotationKernel&&) noexcept;

    void rotate(const float* input, float* output);
    void rotate_inplace(float* data);
    void inverse_rotate(const float* rotated, float* original);

    void rotate_batch(const float* vectors, float* outputs, uint32_t batch_size);

    uint32_t dimensions() const { return dims_; }
    uint32_t padded_dimensions() const { return padded_dims_; }

    static bool is_power_of_two(uint32_t x);
    static uint32_t next_power_of_two(uint32_t x);

private:
    uint32_t dims_;
    uint32_t padded_dims_;
    uint32_t seed_;
    tq_rotation_mode_t mode_;

    std::vector<float> sign_pattern_;
    std::vector<float> temp_buffer_;

    void compute_sign_pattern();
    void fwht_inplace(float* data, uint32_t n);
    void fwht_batched(float* data, uint32_t n);
    void apply_sign_flip(float* data);
    void apply_sign_flip_batched(float* data);
};

class RotationBatchProcessor {
public:
    explicit RotationBatchProcessor(uint32_t dimensions, uint32_t seed = 0);
    ~RotationBatchProcessor();

    void process(const float* input_batch, float* output_batch, uint32_t batch_size,
                 std::function<void(uint32_t, float*)> callback = nullptr);

    uint32_t dimensions() const { return dims_; }
    uint32_t batch_size() const { return batch_size_; }

private:
    uint32_t dims_;
    uint32_t padded_dims_;
    uint32_t seed_;
    uint32_t batch_size_;

    std::unique_ptr<float[]> sign_pattern_;
    std::unique_ptr<float[]> temp_buffer_;
};

} // namespace kernel
} // namespace tq

#endif // __cplusplus

#endif // TQ_ROTATION_KERNEL_H