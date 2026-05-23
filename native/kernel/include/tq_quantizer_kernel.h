/**
 * TurboQuant Native Kernel — Lloyd-Max Quantizer
 *
 * Scalar quantization with Beta-distributed codebook.
 * Replaces TypeScript quantizer.ts for precise register work.
 */

#include <CL/cl.h>
#ifndef TQ_QUANTIZER_KERNEL_H
#define TQ_QUANTIZER_KERNEL_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ_MAX_CODEBOOK_LEVELS 16

typedef struct {
    uint32_t dimensions;
    uint8_t bits_per_value;
    float* codebook;
    uint32_t codebook_size;
    float scale;
    float offset;
    uint8_t is_initialized;
} tq_quantizer_kernel_t;

cl_int tq_quantizer_init(tq_quantizer_kernel_t* quantizer, uint32_t dimensions, uint8_t bits);
cl_int tq_quantizer_init_beta(tq_quantizer_kernel_t* quantizer, uint32_t dimensions, uint8_t bits);
cl_int tq_quantizer_init_uniform(tq_quantizer_kernel_t* quantizer, uint32_t dimensions, uint8_t bits);
cl_int tq_quantizer_shutdown(tq_quantizer_kernel_t* quantizer);

cl_int tq_quantizer_quantize(const tq_quantizer_kernel_t* quantizer, const float* values, uint32_t* indices, uint32_t count);
cl_int tq_quantizer_dequantize(const tq_quantizer_kernel_t* quantizer, const uint32_t* indices, float* values, uint32_t count);
cl_int tq_quantizer_encode(const tq_quantizer_kernel_t* quantizer, const float* values, uint8_t* packed, uint32_t count);
cl_int tq_quantizer_decode(const tq_quantizer_kernel_t* quantizer, const uint8_t* packed, float* values, uint32_t count);

cl_int tq_quantizer_train(tq_quantizer_kernel_t* quantizer, const float* samples, uint32_t sample_count);
float tq_quantizer_compute_mse(const tq_quantizer_kernel_t* quantizer, const float* values, uint32_t count);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <memory>
#include <vector>

namespace tq {
namespace kernel {

class BetaCodebook {
public:
    BetaCodebook(uint32_t dimensions, uint8_t bits);
    ~BetaCodebook();

    void compute();
    void train(const float* samples, uint32_t sample_count);

    float quantize(float value) const;
    uint32_t quantize_index(float value) const;
    float dequantize(uint32_t index) const;

    const float* codebook() const { return codebook_.data(); }
    uint32_t codebook_size() const { return static_cast<uint32_t>(codebook_.size()); }
    uint32_t dimensions() const { return dims_; }
    uint8_t bits() const { return bits_; }

    float compute_mse(const float* samples, uint32_t count) const;

private:
    uint32_t dims_;
    uint8_t bits_;
    std::vector<float> codebook_;
    std::vector<float> boundaries_;

    void compute_beta_distribution();
    void lloyd_max_iteration(const float* samples, uint32_t sample_count);
    float compute_distortion(const float* samples, uint32_t count) const;
};

class UniformCodebook {
public:
    UniformCodebook(uint8_t bits);
    ~UniformCodebook();

    void init(float min_val, float max_val);
    void init_symmetric(float scale);

    float quantize(float value) const;
    uint32_t quantize_index(float value) const;
    float dequantize(uint32_t index) const;

    uint8_t bits() const { return bits_; }
    float min_val() const { return min_val_; }
    float max_val() const { return max_val_; }
    float step() const { return step_; }

    float compute_mse(const float* samples, uint32_t count) const;

private:
    uint8_t bits_;
    uint32_t levels_;
    float min_val_;
    float max_val_;
    float step_;
    float offset_;
    float scale_;
};

class PackedVector {
public:
    PackedVector() : data_(nullptr), packed_bytes_(0), value_count_(0), bits_(0) {}
    ~PackedVector() { delete[] data_; }

    PackedVector(const PackedVector&) = delete;
    PackedVector& operator=(const PackedVector&) = delete;
    PackedVector(PackedVector&&) noexcept;
    PackedVector& operator=(PackedVector&&) noexcept;

    void init(uint32_t count, uint8_t bits) {
        delete[] data_;
        value_count_ = count;
        bits_ = bits;
        packed_bytes_ = ((count * bits) + 7) / 8;
        data_ = new uint8_t[packed_bytes_]();
    }

    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t packed_bytes() const { return packed_bytes_; }
    uint32_t value_count() const { return value_count_; }
    uint8_t bits() const { return bits_; }

private:
    uint8_t* data_;
    size_t packed_bytes_;
    uint32_t value_count_;
    uint8_t bits_;
};

class VectorEncoder {
public:
    explicit VectorEncoder(uint32_t dimensions, uint8_t bits);
    ~VectorEncoder();

    void encode(const float* values, PackedVector* output);
    void decode(const PackedVector* input, float* values);

    void set_codebook(const float* codebook, uint32_t size);

    uint32_t dimensions() const { return dims_; }
    uint8_t bits() const { return bits_; }

private:
    uint32_t dims_;
    uint8_t bits_;
    std::unique_ptr<float[]> codebook_;
    uint32_t codebook_size_;

    uint32_t find_nearest(const float* values, uint32_t count);
};

} // namespace kernel
} // namespace tq

#endif // __cplusplus

#endif // TQ_QUANTIZER_KERNEL_H