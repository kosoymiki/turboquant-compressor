/**
 * TurboQuant Native Kernel — QJL Residual Correction
 *
 * Johnson-Lindenstrauss projection for unbiased dot-product estimation.
 * Based on Zandieh et al., ICML 2024.
 * Replaces TypeScript qjl.ts for precise register work.
 */

#ifndef TQ_QJL_KERNEL_H
#define TQ_QJL_KERNEL_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ_QJL_MAX_TARGET_DIMS 128

typedef struct {
    uint32_t original_dims;
    uint32_t target_dims;
    uint32_t seed;
    float* projection_matrix;
    float* sketch_buffer;
    uint8_t bits_per_value;
    uint8_t use_hadamard;
    uint8_t is_initialized;
} tq_qjl_kernel_t;

cl_int tq_qjl_init(tq_qjl_kernel_t* qjl, uint32_t original_dims, uint32_t target_dims,
                   uint32_t seed, uint8_t bits, uint8_t use_hadamard);
cl_int tq_qjl_shutdown(tq_qjl_kernel_t* qjl);

cl_int tq_qjl_compress(const tq_qjl_kernel_t* qjl, const float* residual, uint8_t* sketch);
cl_int tq_qjl_estimate_dot(const tq_qjl_kernel_t* qjl, const uint8_t* sketch_a, const uint8_t* sketch_b,
                           float* estimate, const float* scale_a, const float* scale_b);

cl_int tq_qjl_project(const tq_qjl_kernel_t* qjl, const float* vector, float* projected);
cl_int tq_qjl_inverse_project(const tq_qjl_kernel_t* qjl, const float* projected, float* original);

float tq_qjl_approximation_error(const tq_qjl_kernel_t* qjl, const float* original, const float* approximated);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <memory>
#include <vector>

namespace tq {
namespace kernel {

class QJLProjector {
public:
    QJLProjector(uint32_t original_dims, uint32_t target_dims, uint32_t seed = 0);
    ~QJLProjector();

    QJLProjector(const QJLProjector&) = delete;
    QJLProjector& operator=(const QJLProjector&) = delete;
    QJLProjector(QJLProjector&&) noexcept;
    QJLProjector& operator=(QJLProjector&&) noexcept;

    void project(const float* input, float* output);
    void inverse_project(const float* projected, float* output);

    float compute_dot_product(const float* vec_a, const float* vec_b) const;

    uint32_t original_dims() const { return orig_dims_; }
    uint32_t target_dims() const { return target_dims_; }

private:
    uint32_t orig_dims_;
    uint32_t target_dims_;
    uint32_t seed_;
    bool use_hadamard_;

    std::vector<float> projection_matrix_;
    std::vector<float> hadamard_signs_;

    void generate_gaussian_matrix();
    void generate_hadamard_signs();
    void apply_hadamard(float* data, uint32_t n);
    void apply_projection(const float* input, float* output);

    static bool is_power_of_two(uint32_t x);
    static uint32_t next_power_of_two(uint32_t x);
};

class QJLSketchCompressor {
public:
    QJLSketchCompressor(uint32_t target_dims, uint8_t bits = 4);
    ~QJLSketchCompressor();

    void compress(const float* projected, uint8_t* sketch);
    void decompress(const uint8_t* sketch, float* projected);

    uint8_t* compress(const float* projected);
    float* decompress(const uint8_t* sketch);

    uint32_t sketch_bytes() const { return sketch_bytes_; }
    uint32_t target_dims() const { return target_dims_; }
    uint8_t bits() const { return bits_; }

private:
    uint32_t target_dims_;
    uint8_t bits_;
    uint32_t sketch_bytes_;

    float quantize_value(float value) const;
    float dequantize_value(float quantized) const;
};

class QJLResidualEstimator {
public:
    QJLResidualEstimator(uint32_t original_dims, uint32_t target_dims,
                         uint8_t sketch_bits = 4, uint32_t seed = 0);
    ~QJLResidualEstimator();

    struct SketchResult {
        uint8_t* sketch;
        uint32_t sketch_bytes;
        float scale;
    };

    SketchResult compress(const float* residual);
    void free_sketch(SketchResult* result);

    float estimate_dot(const float* residual_a, const float* residual_b);
    float estimate_dot(const SketchResult& sketch_a, const SketchResult& sketch_b);

    uint32_t original_dims() const { return projector_.original_dims(); }
    uint32_t target_dims() const { return projector_.target_dims(); }

private:
    QJLProjector projector_;
    QJLSketchCompressor compressor_;

    float last_scale_a_;
};

class QJLDotProductEstimator {
public:
    QJLDotProductEstimator(uint32_t dims, uint32_t sketch_dims, uint32_t seed = 0);
    ~QJLDotProductEstimator();

    float estimate(const float* vec_a, const float* vec_b, bool use_sketch = false);
    float exact_dot(const float* vec_a, const float* vec_b);

    float approximation_error() const { return last_error_; }
    uint32_t calls() const { return call_count_; }

    void reset_stats();

private:
    QJLResidualEstimator estimator_;
    float last_error_;
    uint32_t call_count_;
};

} // namespace kernel
} // namespace tq

#endif // __cplusplus

#endif // TQ_QJL_KERNEL_H