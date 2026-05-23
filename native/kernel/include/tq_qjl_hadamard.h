/**
 * TurboQuant Native Kernel — Paper-Faithful 1-bit Hadamard QJL
 *
 * Zandieh et al., ICML 2024 — "Efficient and Accurate Inner Products for LLMs"
 *
 * Key properties:
 *   - Subsampled Hadamard: O(d log d) projection (not O(d^2))
 *   - 1-bit sign quantization: zero per-block overhead
 *   - Asymmetric estimator: QJL on keys, Gaussian JL on queries → unbiased
 *
 * Paper: arXiv:2406.03482
 */

#ifndef TQ_QJL_HADAMARD_H
#define TQ_QJL_HADAMARD_H

#include <stdint.h>
#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ_HADAMARD_QJL_MAX_DIMS 2048

typedef struct {
    uint32_t original_dims;
    uint32_t sketch_dims;
    uint32_t original_dims_log2;
    uint32_t seed;
    float* rademacher_buffer;   // ±1 diagonal for Rademacher transform
    float* temp_buffer;          // scratch for FWHT
    uint32_t sketch_bytes;
    uint8_t is_initialized;
} tq_hadamard_qjl_t;

// ── Lifecycle ────────────────────────────────────────────────────────────────

/**
 * Initialize paper-faithful Hadamard QJL sketch.
 *
 * @param qjl              Sketch structure (must be allocated by caller)
 * @param original_dims    Original vector dimension
 * @param sketch_dims      Target sketch dimension (m in paper)
 *                         Rule: m = O(1/epsilon^2) for JL guarantee
 *                         For epsilon=0.05: m >= 400
 *                         For epsilon=0.03: m >= 1100
 * @param seed            RNG seed for reproducible Rademacher pattern
 * @return CL_SUCCESS on success
 */
cl_int tq_hadamard_qjl_init(tq_hadamard_qjl_t* qjl,
                            uint32_t original_dims,
                            uint32_t sketch_dims,
                            uint32_t seed);

/**
 * Shutdown and free resources.
 */
cl_int tq_hadamard_qjl_shutdown(tq_hadamard_qjl_t* qjl);

/**
 * Query sketch memory requirements.
 */
uint32_t tq_hadamard_qjl_sketch_bytes(const tq_hadamard_qjl_t* qjl);

// ── Sketch operations ────────────────────────────────────────────────────

/**
 * Compress residual vector to 1-bit Hadamard sketch.
 *
 * Key compression: 1-bit sign only, no scale, no zero.
 * This is the fundamental property enabling zero overhead.
 *
 * @param qjl          Initialized sketch structure
 * @param residual     Residual vector [original_dims]
 * @param sketch_out   Output sketch [sketch_bytes]
 */
void tq_hadamard_qjl_compress(const tq_hadamard_qjl_t* qjl,
                               const float* residual,
                               uint8_t* sketch_out);

/**
 * Project query to sketch space using standard Gaussian JL.
 *
 * Asymmetric: queries use continuous (not sign-quantized) projection.
 * This is what makes the estimator unbiased per Lemma 1 in the paper.
 *
 * @param qjl    Initialized sketch structure
 * @param query  Query vector [original_dims]
 * @param out    Projected vector [sketch_dims]
 */
void tq_hadamard_qjl_project_query(const tq_hadamard_qjl_t* qjl,
                                     const float* query,
                                     float* out);

/**
 * Estimate dot product between key sketch and query projection.
 *
 * Asymmetric dot product estimator from paper:
 *   dot_est = (1/m) * sum_j sign(sketch[j]) * query_proj[j]
 *
 * This is unbiased: E[dot_est] = dot_original
 *
 * @param qjl         Initialized sketch structure
 * @param sketch      Key sketch [sketch_bytes]
 * @param query_proj  Query projection [sketch_dims]
 * @return Estimated dot product
 */
float tq_hadamard_qjl_estimate_dot(const tq_hadamard_qjl_t* qjl,
                                    const uint8_t* sketch,
                                    const float* query_proj);

#ifdef __cplusplus
}
#endif

// ── C++ Classes ────────────────────────────────────────────────────────────
#ifdef __cplusplus

#include <vector>
#include <cstring>
#include <cmath>

namespace tq {
namespace hadamard {

/**
 * HadamardQJLSketch — Paper-faithful 1-bit Hadamard QJL implementation
 *
 * Based on Zandieh et al., ICML 2024, Algorithm 1 & 2.
 */
class HadamardQJLSketch {
public:
    /**
     * Constructor
     * @param original_dims  Vector dimension (d)
     * @param sketch_dims    Target dimension (m), typically O(1/epsilon^2)
     * @param seed           RNG seed for Rademacher pattern
     */
    HadamardQJLSketch(uint32_t original_dims,
                       uint32_t sketch_dims,
                       uint32_t seed = 42);

    ~HadamardQJLSketch();

    // Non-copyable, non-movable for simplicity
    HadamardQJLSketch(const HadamardQJLSketch&) = delete;
    HadamardQJLSketch& operator=(const HadamardQJLSketch&) = delete;

    // ── Accessors ───────────────────────────────────────────────────────────
    uint32_t original_dims() const { return orig_dims_; }
    uint32_t sketch_dims() const { return sketch_dims_; }
    uint32_t sketch_bytes() const { return sketch_bytes_; }

    // ── Sketch operations ───────────────────────────────────────────────────

    /**
     * Compress residual to 1-bit Hadamard sketch.
     *
     * Steps (per paper Algorithm 1):
     *   1. y = D * x  (Rademacher diagonal multiplication)
     *   2. z = H * y  (FWHT, O(d log d))
     *   3. sketch = sign(z[:m])  (1-bit, zero overhead)
     *
     * @param residual     Input residual [original_dims]
     * @param sketch_out   Output sketch [sketch_bytes]
     */
    void compress(const float* residual, uint8_t* sketch_out) const;

    /**
     * Project query using standard Gaussian JL approximation.
     *
     * Asymmetric: uses continuous (not sign-quantized) Hadamard.
     * This gives unbiased dot product estimator.
     *
     * @param query  Input query [original_dims]
     * @param out    Projected vector [sketch_dims]
     */
    void project_query(const float* query, float* out) const;

    /**
     * Estimate dot product via asymmetric estimator.
     *
     * Formula: dot_est = (1/m) * sum_j sign(sketch_j) * query_j
     *
     * @param key_sketch   Key sketch [sketch_bytes]
     * @param query_sketch Query projection [sketch_dims]
     * @return Estimated dot product (scaled)
     */
    float estimate_sketch_dot(const uint8_t* key_sketch,
                       const float* query_sketch) const;

private:
    uint32_t orig_dims_;
    uint32_t sketch_dims_;
    uint32_t padded_dims_;
    uint32_t padded_dims_log2_;
    uint32_t seed_;
    uint32_t sketch_bytes_;

    std::vector<float> rademacher_;   // ±1 random diagonal
    mutable std::vector<float> temp_; // scratch for FWHT

    // ── Helpers ────────────────────────────────────────────────────────────
    static bool is_power_of_two(uint32_t x);
    static uint32_t next_power_of_two(uint32_t x);
    static uint32_t log2_pow2(uint32_t x);
    void generate_rademacher();
    void fwht_inplace(float* data, uint32_t n) const;
    void fwht_normalize(float* data, uint32_t n) const;
};

/**
 * AsymmetricDotEstimator — End-to-end dot product estimation
 *
 * Encapsulates key compression + query projection + dot estimation.
 * Convenience class for integration.
 */
class AsymmetricDotEstimator {
public:
    AsymmetricDotEstimator(uint32_t original_dims,
                           uint32_t sketch_dims,
                           uint32_t seed = 42);

    ~AsymmetricDotEstimator();

    AsymmetricDotEstimator(const AsymmetricDotEstimator&) = delete;
    AsymmetricDotEstimator& operator=(const AsymmetricDotEstimator&) = delete;

    // ── Accessors ───────────────────────────────────────────────────────────
    uint32_t sketch_bytes() const { return sketch_.sketch_bytes(); }
    uint32_t sketch_dims() const { return sketch_.sketch_dims(); }

    // ── Operations ──────────────────────────────────────────────────────────

    /**
     * Compress key residual to sketch.
     */
    void compress_key(const float* residual, uint8_t* sketch_out) {
        sketch_.compress(residual, sketch_out);
    }

    /**
     * Project query to sketch space.
     */
    void project_query(const float* query, float* out) {
        sketch_.project_query(query, out);
    }

    /**
     * Estimate dot product between two vectors.
     *
     * Convenience: compress A, project B, estimate.
     */
    float estimate_vectors_dot(const float* vec_a, const float* vec_b);

private:
    HadamardQJLSketch sketch_;
    std::vector<uint8_t> temp_sketch_;
    std::vector<float> temp_query_;
};

} // namespace hadamard
} // namespace tq

#endif // __cplusplus

#endif // TQ_QJL_HADAMARD_H