/**
 * TurboQuant Beta Sphere — Paper-faithful Beta Math Library
 *
 * After random Hadamard rotation, each coordinate of a unit-norm vector follows:
 *   f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
 * which is a scaled Beta distribution on [-1, 1].
 *
 * This module provides:
 *   - log_gamma:   Lanczos approximation
 *   - sphere_coordinate_beta_pdf: Beta PDF for rotated unit vectors
 *   - integrate_adaptive_simpson: adaptive numerical integration
 *   - conditional_mean_beta_sphere: E[X | lo < X < hi] under Beta PDF
 *   - compute_lloyd_max_beta_codebook: optimal 2^bits-level Lloyd-Max codebook
 *
 * Derived from src/core/beta_sphere.ts (1:1 port, no approximation).
 */

#include "tq_beta_sphere.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace tq {
namespace beta {

// ── Log-Gamma (Lanczos approximation) ──────────────────────────────────────
static double log_gamma_lanczos(double x) {
    if (x <= 0.0) return 0.0; // undefined, but guard
    const int g = 7;
    static const double c[] = {
        0.99999999999980993,
        676.5203681218851,
       -1259.1392167224028,
        771.32342877765313,
      -176.61502916214059,
        12.507343278686905,
       -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };
    if (x < 0.5) {
        return std::log(M_PI / (std::sin(M_PI * x) * std::exp(log_gamma_lanczos(1.0 - x))));
    }
    x -= 1.0;
    double a = c[0];
    for (int i = 1; i < g + 2; i++) a += c[i] / (x + i);
    double t = x + g + 0.5;
    return 0.5 * std::log(2.0 * M_PI) + (x + 0.5) * std::log(t) - t + std::log(a);
}

double log_gamma(double x) {
    if (x <= 0.0) return 0.0;
    return log_gamma_lanczos(x);
}

double log_factorial(int n) {
    if (n <= 1) return 0.0;
    static double cache[171] = {0}; // 170! is near double overflow
    static bool init = false;
    if (!init) {
        cache[0] = 0.0;
        cache[1] = 0.0;
        for (int i = 2; i <= 170; i++) cache[i] = cache[i-1] + std::log((double)i);
        init = true;
    }
    if (n >= 0 && n <= 170) return cache[n];
    return log_gamma_lanczos((double)n + 1.0);
}

// ── Beta sphere coordinate PDF ──────────────────────────────────────────────
double sphere_coordinate_beta_pdf(double x, int dimension) {
    if (dimension < 3) return 0.0;
    if (x < -1.0 || x > 1.0) return 0.0;

    double log_const = log_gamma_lanczos((double)dimension / 2.0)
                     - 0.5 * std::log(M_PI)
                     - log_gamma_lanczos(((double)dimension - 1.0) / 2.0);
    double exponent = ((double)dimension - 3.0) / 2.0;
    double xx = x * x;
    double one_minus_xx = (1.0 - xx > 1e-15) ? 1.0 - xx : 1e-15;
    double log_val = log_const + exponent * std::log(one_minus_xx);
    return std::exp(log_val);
}

// ── Adaptive Simpson integration ───────────────────────────────────────────
static double simpson(double (*f)(double, void*), void* ctx, double a, double b) {
    double c = (a + b) * 0.5;
    return (b - a) * (f(a, ctx) + 4.0 * f(c, ctx) + f(b, ctx)) / 6.0;
}

// f(x) = x * sphere_coordinate_beta_pdf(x, dim) for conditional mean
static double integrand_mean(double x, void* ctx) {
    int dim = *(int*)ctx;
    return x * sphere_coordinate_beta_pdf(x, dim);
}

// f(x) = sphere_coordinate_beta_pdf(x, dim) for normalization
static double integrand_pdf(double x, void* ctx) {
    int dim = *(int*)ctx;
    return sphere_coordinate_beta_pdf(x, dim);
}

static double adaptive_simpson_recursive(
    double (*f)(double, void*),
    void* ctx,
    double a, double b,
    double whole,
    int depth,
    double eps,
    int max_depth
) {
    double c = (a + b) * 0.5;
    double left  = simpson(f, ctx, a, c);
    double right = simpson(f, ctx, c, b);
    double delta = left + right - whole;

    if (depth >= max_depth || std::abs(delta) < 15.0 * eps) {
        return left + right + delta / 15.0;
    }
    return adaptive_simpson_recursive(f, ctx, a, c, left, depth + 1, eps, max_depth)
         + adaptive_simpson_recursive(f, ctx, c, b, right, depth + 1, eps, max_depth);
}

static double adaptive_simpson(
    double (*f)(double, void*),
    void* ctx,
    double lo, double hi,
    double eps,
    int max_depth
) {
    double whole = simpson(f, ctx, lo, hi);
    return adaptive_simpson_recursive(f, ctx, lo, hi, whole, 0, eps, max_depth);
}

double conditional_mean_beta_sphere(double lo, double hi, int dimension) {
    int dim = dimension;
    double eps = 1e-10;
    double num = adaptive_simpson(integrand_mean, &dim, lo, hi, eps, 50);
    double den = adaptive_simpson(integrand_pdf,  &dim, lo, hi, eps, 50);
    if (den < 1e-30) return (lo + hi) * 0.5;
    return num / den;
}

// ── Lloyd-Max Beta codebook computation ─────────────────────────────────────
static double integrand_mse(double x, void* ctx) {
    // ctx points to MSEIntegrand { dim, c }
    struct MSEIntegrand { int dim; double c; };
    MSEIntegrand* p = (MSEIntegrand*)ctx;
    double dx = x - p->c;
    return dx * dx * sphere_coordinate_beta_pdf(x, p->dim);
}

static double x_grid[10000];

static void init_x_grid(void) {
    static bool done = false;
    if (done) return;
    for (int i = 0; i < 10000; i++) x_grid[i] = -1.0 + 2.0 * i / 9999.0;
    done = true;
}

static void compute_pdf_and_cdf(int dimension, double* pdf_vals, double* cdf_vals) {
    init_x_grid();
    double cumsum = 0.0;
    for (int i = 0; i < 10000; i++) {
        pdf_vals[i] = sphere_coordinate_beta_pdf(x_grid[i], dimension);
        cumsum += pdf_vals[i];
        cdf_vals[i] = cumsum;
    }
    double total = cdf_vals[9999];
    if (total > 0) {
        for (int i = 0; i < 10000; i++) cdf_vals[i] /= total;
    }
}

tq_beta_codebook compute_lloyd_max_beta_codebook(int dimension, int bits, int max_iter, double tol) {
    tq_beta_codebook result;
    memset(&result, 0, sizeof(result));
    result.dimension = dimension;
    result.bits = bits;
    result.algorithm = TQ_ALGORITHM_LLOYD_MAX_BETA;

    int n_clusters = 1 << bits; // 2^bits
    result.centroids = (double*)malloc(sizeof(double) * n_clusters);
    result.boundaries = (double*)malloc(sizeof(double) * (n_clusters + 1));
    result.n_clusters = n_clusters;

    if (!result.centroids || !result.boundaries) {
        free(result.centroids);
        free(result.boundaries);
        memset(&result, 0, sizeof(result));
        return result;
    }

    // Compute PDF + CDF grid for initialization
    double pdf_vals[10000];
    double cdf_vals[10000];
    compute_pdf_and_cdf(dimension, pdf_vals, cdf_vals);

    // Initialize centroids at quantile midpoints
    for (int i = 0; i < n_clusters; i++) {
        double q_lo = (double)i / (double)n_clusters;
        double q_hi = (double)(i + 1) / (double)n_clusters;
        double q_mid = (q_lo + q_hi) * 0.5;
        // Find x corresponding to q_mid in CDF
        int idx = 0;
        for (int j = 0; j < 10000; j++) {
            if (cdf_vals[j] >= q_mid) { idx = j; break; }
            idx = j;
        }
        result.centroids[i] = x_grid[idx < 10000 ? idx : 9999];
    }

    // Lloyd-Max iterations
    double prev_cost = 1e100;
    double final_cost = 0.0;

    struct MSEIntegrand { int dim; double c; };
    struct MSEIntegrand mi;
    mi.dim = dimension;
    mi.c = 0.0;

    for (int iteration = 0; iteration < max_iter; iteration++) {
        // Update boundaries from centroids
        result.boundaries[0] = -1.0;
        result.boundaries[n_clusters] = 1.0;
        for (int i = 0; i < n_clusters - 1; i++) {
            result.boundaries[i + 1] = (result.centroids[i] + result.centroids[i + 1]) * 0.5;
        }

        // Update centroids: conditional mean in each cell
        double new_centroids[64]; // max 64 clusters (6-bit = 64 levels)
        for (int i = 0; i < n_clusters; i++) {
            double lo = result.boundaries[i];
            double hi = result.boundaries[i + 1];
            new_centroids[i] = conditional_mean_beta_sphere(lo, hi, dimension);
        }

        // Compute cost (MSE per coordinate)
        mi.c = 0.0;
        double next_cost = 0.0;
        for (int i = 0; i < n_clusters; i++) {
            mi.c = new_centroids[i];
            double cell_mse = adaptive_simpson(integrand_mse, &mi,
                result.boundaries[i], result.boundaries[i + 1], 1e-10, 200);
            next_cost += cell_mse;
        }

        // Commit new centroids
        for (int i = 0; i < n_clusters; i++) result.centroids[i] = new_centroids[i];

        final_cost = next_cost;
        if (std::abs(prev_cost - next_cost) < tol) break;
        prev_cost = next_cost;
    }

    // Final boundaries from converged centroids
    result.boundaries[0] = -1.0;
    result.boundaries[n_clusters] = 1.0;
    for (int i = 0; i < n_clusters - 1; i++) {
        result.boundaries[i + 1] = (result.centroids[i] + result.centroids[i + 1]) * 0.5;
    }

    // Compute RMS L2 norm of centroids (for search score normalization)
    double cn2 = 0.0;
    for (int i = 0; i < n_clusters; i++) cn2 += result.centroids[i] * result.centroids[i];
    result.centroid_norm = std::sqrt(cn2 / (double)n_clusters);

    result.mse_per_coord = final_cost;
    result.mse_total = final_cost * dimension;
    result.source = TQ_SOURCE_COMPUTED;
    return result;
}

void free_beta_codebook(tq_beta_codebook* cb) {
    if (!cb) return;
    free(cb->centroids);
    free(cb->boundaries);
    memset(cb, 0, sizeof(*cb));
}

// ── TurboQuantBetaCodebook class ────────────────────────────────────────────
TurboQuantBetaCodebook::TurboQuantBetaCodebook(int dimension, int bits)
    : dimension_(dimension), bits_(bits), codebook_{nullptr} {}

TurboQuantBetaCodebook::~TurboQuantBetaCodebook() {
    if (codebook_) {
        free_beta_codebook(codebook_);
        free(codebook_);
        codebook_ = nullptr;
    }
}

const tq_beta_codebook* TurboQuantBetaCodebook::compute(int max_iter, double tol) {
    if (codebook_) {
        free_beta_codebook(codebook_);
    } else {
        codebook_ = (tq_beta_codebook*)calloc(1, sizeof(tq_beta_codebook));
    }
    *codebook_ = compute_lloyd_max_beta_codebook(dimension_, bits_, max_iter, tol);
    return codebook_;
}

int TurboQuantBetaCodebook::quantize_index(double value) const {
    if (!codebook_) return 0;
    if (value <= codebook_->boundaries[0]) return 0;
    if (value >= codebook_->boundaries[codebook_->n_clusters]) return codebook_->n_clusters - 1;
    for (int i = 0; i < codebook_->n_clusters; i++) {
        if (value >= codebook_->boundaries[i] && value < codebook_->boundaries[i + 1])
            return i;
    }
    return codebook_->n_clusters - 1;
}

double TurboQuantBetaCodebook::quantize(double value) const {
    return dequantize(quantize_index(value));
}

double TurboQuantBetaCodebook::dequantize(int index) const {
    if (!codebook_) return 0.0;
    if (index < 0) index = 0;
    if (index >= codebook_->n_clusters) index = codebook_->n_clusters - 1;
    return codebook_->centroids[index];
}

} // namespace beta
} // namespace tq

// ── C ABI wrappers ───────────────────────────────────────────────────────────
extern "C" {

double tq_log_gamma(double x) { return tq::beta::log_gamma(x); }

double tq_log_factorial(int n) { return tq::beta::log_factorial(n); }

double tq_sphere_coordinate_beta_pdf(double x, int dim) {
    return tq::beta::sphere_coordinate_beta_pdf(x, dim);
}

double tq_conditional_mean_beta_sphere(double lo, double hi, int dim) {
    return tq::beta::conditional_mean_beta_sphere(lo, hi, dim);
}

tq_beta_codebook tq_compute_lloyd_max_beta_codebook(
    int dim, int bits, int max_iter, double tol
) {
    return tq::beta::compute_lloyd_max_beta_codebook(dim, bits, max_iter, tol);
}

void tq_free_beta_codebook(tq_beta_codebook* cb) {
    tq::beta::free_beta_codebook(cb);
}

} // extern "C"
