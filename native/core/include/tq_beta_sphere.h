/**
 * TurboQuant Beta Sphere — Paper-faithful Beta Math Library
 *
 * Derived from src/core/beta_sphere.ts (1:1 port).
 * No external dependencies beyond <cmath>, <cstdint>.
 */

#ifndef TQ_BETA_SPHERE_H
#define TQ_BETA_SPHERE_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Algorithm classification ────────────────────────────────────────────────
#define TQ_ALGORITHM_UNIFORM          0
#define TQ_ALGORITHM_LLOYD_MAX         1
#define TQ_ALGORITHM_LLOYD_MAX_BETA    2  // paper-faithful TurboQuant

#define TQ_SOURCE_PRELOADED  0
#define TQ_SOURCE_COMPUTED   1

// ── Codebook structure ───────────────────────────────────────────────────────
typedef struct {
    double*  centroids;      // [n_clusters]
    double*  boundaries;     // [n_clusters + 1]
    int      n_clusters;     // 2^bits
    int      dimension;
    int      bits;
    int      algorithm;      // TQ_ALGORITHM_* constant
    int      source;         // TQ_SOURCE_* constant
    double   mse_per_coord;
    double   mse_total;
    double   centroid_norm;  // RMS L2 norm of centroids (for search score normalization)
} tq_beta_codebook;

// ── Public API ───────────────────────────────────────────────────────────────
double tq_log_gamma(double x);
double tq_log_factorial(int n);
double tq_sphere_coordinate_beta_pdf(double x, int dimension);
double tq_conditional_mean_beta_sphere(double lo, double hi, int dimension);

// Allocate + compute, caller owns the pointer and must call free_beta_codebook()
tq_beta_codebook tq_compute_lloyd_max_beta_codebook(
    int dimension, int bits, int max_iter, double tol
);
void tq_free_beta_codebook(tq_beta_codebook* cb);

#ifdef __cplusplus
}

// ── C++ wrapper class ────────────────────────────────────────────────────────
namespace tq {
namespace beta {

class TurboQuantBetaCodebook {
public:
    TurboQuantBetaCodebook(int dimension, int bits);
    ~TurboQuantBetaCodebook();

    const tq_beta_codebook* compute(int max_iter = 200, double tol = 1e-12);
    int  quantize_index(double value) const;
    double quantize(double value) const;
    double dequantize(int index) const;

    const tq_beta_codebook* get_codebook() const { return codebook_; }
    bool is_ready() const { return codebook_ != nullptr; }

private:
    int              dimension_;
    int              bits_;
    tq_beta_codebook* codebook_;
};

} // namespace beta
} // namespace tq

#endif // __cplusplus
#endif // TQ_BETA_SPHERE_H
