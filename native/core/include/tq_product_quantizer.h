/**
 * TurboQuant Kernel — Product Quantization with k-Means Training
 * Based on Jegou et al., "Product Quantization for Nearest Neighbor Search" (TPAMI 2011)
 */

#ifndef TQ_PRODUCT_QUANTIZER_H
#define TQ_PRODUCT_QUANTIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ_PQ_MAGIC 0x54500150

typedef struct {
    uint32_t dimensions;
    uint8_t  num_subspaces;
    uint8_t  bits_per_subspace;
    uint8_t  k_means_iterations;
    uint8_t  use_opq;
    uint32_t rotation_seed;
} tq_pq_config;

typedef struct {
    uint32_t dimensions;
    uint8_t  num_subspaces;
    uint8_t  bits_per_subspace;
    uint8_t  k_log2;
    uint32_t k;
    uint32_t dim_per_sub;
    uint8_t  use_opq;
    uint32_t rotation_seed;
    float* centroids;     // [num_subspaces][k][dim_per_sub]
    float* rotation;     // [dims][dims] if OPQ
    uint8_t trained;
} tq_pq_codebook;

typedef struct {
    uint32_t num_vectors;
    uint8_t  num_subspaces;
    uint32_t codes_per_vec;
    uint8_t* codes;      // [num_vectors][codes_per_vec]
    float* norms;        // [num_vectors]
    uint8_t initialized;
} tq_pq_database;

typedef struct {
    tq_pq_codebook codebook;
    tq_pq_database database;
} tq_pq_model;

tq_pq_config tq_pq_default_config(uint32_t dimensions);
tq_pq_model* tq_pq_train(const tq_pq_config* cfg, const float* train_x, uint32_t num_train, uint32_t seed);
void tq_pq_encode(const tq_pq_model* model, const float* x, uint8_t* code);
void tq_pq_encode_batch(const tq_pq_model* model, const float* X, uint32_t num_vec, uint8_t* codes_out);
void tq_pq_decode(const tq_pq_model* model, const uint8_t* code, float* x_out);
void tq_pq_search(const tq_pq_model* model, const float* query, uint32_t top_k, uint32_t* idx_out, float* score_out);
int tq_pq_serialize(const tq_pq_model* model, char** out_base64, size_t* out_len);
int tq_pq_deserialize(const char* base64, size_t len, tq_pq_model** model_out);
void tq_pq_model_free(tq_pq_model* model);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <vector>
#include <cstring>

namespace tq {
namespace pq {

class ProductQuantizer {
public:
    ProductQuantizer(uint32_t dims, uint8_t m, uint8_t k_bits);
    ~ProductQuantizer();
    void train(const float* X, uint32_t N, uint32_t seed = 42, uint32_t max_iter = 50, bool use_opq = false);
    bool is_trained() const { return trained_; }
    void encode(const float* x, uint8_t* code) const;
    void encode_batch(const float* X, uint32_t N, uint8_t* codes_out) const;
    void build_lut(const float* query, float* lut_out) const;
    float estimate_distance(const uint8_t* code, const float* lut) const;
    void search(const uint8_t* codes, const float* norms, uint32_t N, const float* query, uint32_t top_k, uint32_t* idx_out, float* score_out) const;
    uint32_t dimensions() const { return dims_; }
    uint8_t num_subspaces() const { return m_; }
    uint32_t k() const { return k_; }
    uint32_t codes_per_vector() const { return cperv_; }

private:
    uint32_t dims_, k_, dim_sub_, cperv_;
    uint8_t m_, k_bits_;
    std::vector<float> cents_;
    std::vector<float> opq_;
    bool trained_;
    mutable std::vector<float> lut_;
    mutable std::vector<float> tmp_x_;
    mutable std::vector<float> tmp_rx_;
    mutable std::vector<float> tmp_subs_;

    void apply_opq(const float* x, float* rx) const;
    void split(const float* x, float* subs) const;
    float kmeans(const float* pts, uint32_t n, uint32_t dim, uint32_t k, uint32_t seed, float* out) const;
};

ProductQuantizer::ProductQuantizer(uint32_t dims, uint8_t m, uint8_t k_bits)
    : dims_(dims), m_(m), k_(1 << k_bits), k_bits_(k_bits),
      dim_sub_(dims / m), cperv_((m * k_bits + 7) >> 3), trained_(false) {}

ProductQuantizer::~ProductQuantizer() = default;

bool is_pow2(uint32_t x) { return x && !(x & (x - 1)); }
uint32_t next_pow2(uint32_t x) { if (!x) return 1; x--; x |= x>>1; x |= x>>2; x |= x>>4; x |= x>>8; x |= x>>16; return x + 1; }

float l2sq(const float* a, const float* b, uint32_t d) {
    float s = 0; for (uint32_t i = 0; i < d; i++) { float t = a[i] - b[i]; s += t * t; } return s;
}

void ProductQuantizer::apply_opq(const float* x, float* rx) const {
    if (opq_.empty()) { memcpy(rx, x, dims_ * sizeof(float)); return; }
    for (uint32_t i = 0; i < dims_; i++) {
        float sum = 0;
        for (uint32_t j = 0; j < dims_; j++) sum += opq_[i * dims_ + j] * x[j];
        rx[i] = sum;
    }
}

void ProductQuantizer::split(const float* x, float* subs) const {
    for (uint8_t s = 0; s < m_; s++) {
        memcpy(subs + s * dim_sub_, x + s * dim_sub_, dim_sub_ * sizeof(float));
    }
}

float ProductQuantizer::kmeans(const float* pts, uint32_t n, uint32_t dim, uint32_t k, uint32_t seed, float* out) const {
    std::vector<float> C(k * dim);
    std::vector<float> As(n);
    uint32_t s = seed;

    // k-means++ init
    for (uint32_t ck = 0; ck < k; ck++) {
        if (ck == 0) {
            uint32_t idx = ((s = s * 1664525U + 1013904223U) >> 16) % n;
            memcpy(C.data() + ck * dim, pts + idx * dim, dim * sizeof(float));
        } else {
            float max_d = 0;
            uint32_t best_idx = 0;
            for (uint32_t i = 0; i < n; i++) {
                float d = l2sq(pts + i * dim, C.data(), dim);
                if (d > max_d) { max_d = d; best_idx = i; }
            }
            memcpy(C.data() + ck * dim, pts + best_idx * dim, dim * sizeof(float));
        }
    }

    // Iterate
    for (uint32_t iter = 0; iter < 50; iter++) {
        for (uint32_t i = 0; i < n; i++) {
            float best_d = 1e9f;
            int best_c = 0;
            for (uint32_t ck = 0; ck < k; ck++) {
                float d = l2sq(pts + i * dim, C.data() + ck * dim, dim);
                if (d < best_d) { best_d = d; best_c = ck; }
            }
            As[i] = (float)best_c;
        }

        memset(out, 0, k * dim * sizeof(float));
        std::vector<uint32_t> cnt(k, 0);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ck = (uint32_t)As[i];
            cnt[ck]++;
            for (uint32_t d = 0; d < dim; d++) out[ck * dim + d] += pts[i * dim + d];
        }
        for (uint32_t ck = 0; ck < k; ck++) {
            if (cnt[ck] > 0) {
                for (uint32_t d = 0; d < dim; d++) out[ck * dim + d] /= (float)cnt[ck];
            } else {
                memcpy(out + ck * dim, C.data() + ck * dim, dim * sizeof(float));
            }
        }
        memcpy(C.data(), out, k * dim * sizeof(float));
    }

    // Return final centroids
    memcpy(out, C.data(), k * dim * sizeof(float));
    return 0.0f;
}

void ProductQuantizer::train(const float* X, uint32_t N, uint32_t seed, uint32_t max_iter, bool use_opq) {
    (void)max_iter;
    cents_.resize(m_ * k_ * dim_sub_);
    tmp_x_.resize(dims_);
    tmp_rx_.resize(dims_);
    tmp_subs_.resize(m_ * dim_sub_);
    lut_.resize(m_ * k_);

    // Apply rotation and split into subspaces
    for (uint32_t n = 0; n < N; n++) {
        apply_opq(X + n * dims_, tmp_rx_.data());
        split(tmp_rx_.data(), tmp_subs_.data());

        for (uint8_t s = 0; s < m_; s++) {
            kmeans(tmp_subs_.data() + s * dim_sub_, N, dim_sub_, k_, seed + s, cents_.data() + s * k_ * dim_sub_);
        }
    }

    trained_ = true;
}

void ProductQuantizer::encode(const float* x, uint8_t* code) const {
    apply_opq(x, tmp_rx_.data());
    split(tmp_rx_.data(), tmp_subs_.data());

    memset(code, 0, cperv_);
    for (uint8_t s = 0; s < m_; s++) {
        float best_d = 1e9f;
        uint32_t best_c = 0;
        const float* sub = tmp_subs_.data() + s * dim_sub_;
        const float* cb = cents_.data() + s * k_ * dim_sub_;

        for (uint32_t ck = 0; ck < k_; ck++) {
            float d = l2sq(sub, cb + ck * dim_sub_, dim_sub_);
            if (d < best_d) { best_d = d; best_c = ck; }
        }

        uint32_t bit_pos = s * k_bits_;
        uint32_t byte_idx = bit_pos >> 3;
        uint32_t bit_off = bit_pos & 7;
        code[byte_idx] |= (uint8_t)(best_c << bit_off);
        if (bit_off + k_bits_ > 8) {
            code[byte_idx + 1] |= (uint8_t)(best_c >> (8 - bit_off));
        }
    }
}

void ProductQuantizer::encode_batch(const float* X, uint32_t N, uint8_t* codes_out) const {
    for (uint32_t n = 0; n < N; n++) encode(X + n * dims_, codes_out + n * cperv_);
}

void ProductQuantizer::build_lut(const float* query, float* lut_out) const {
    apply_opq(query, tmp_rx_.data());
    split(tmp_rx_.data(), tmp_subs_.data());

    for (uint8_t s = 0; s < m_; s++) {
        const float* sub = tmp_subs_.data() + s * dim_sub_;
        const float* cb = cents_.data() + s * k_ * dim_sub_;
        float* l = lut_out + s * k_;

        for (uint32_t ck = 0; ck < k_; ck++) {
            l[ck] = 0;
            for (uint32_t d = 0; d < dim_sub_; d++) l[ck] += sub[d] * cb[ck * dim_sub_ + d];
        }
    }
}

float ProductQuantizer::estimate_distance(const uint8_t* code, const float* lut) const {
    float dist = 0;
    for (uint8_t s = 0; s < m_; s++) {
        uint32_t bit_pos = s * k_bits_;
        uint32_t byte_idx = bit_pos >> 3;
        uint32_t bit_off = bit_pos & 7;
        uint32_t ck = (code[byte_idx] >> bit_off) & (k_ - 1U);
        if (bit_off + k_bits_ > 8) {
            ck |= ((code[byte_idx + 1] << (8 - bit_off)) & (k_ - 1U));
        }
        dist += lut[s * k_ + ck];
    }
    return -dist; // negative for max-heap
}

void ProductQuantizer::search(const uint8_t* codes, const float* norms, uint32_t N, const float* query, uint32_t top_k, uint32_t* idx_out, float* score_out) const {
    build_lut(query, lut_.data());

    struct TK { uint32_t i; float s; };
    std::vector<TK> heap;
    heap.reserve(top_k);

    for (uint32_t n = 0; n < N; n++) {
        float s = estimate_distance(codes + n * cperv_, lut_.data()) * norms[n];
        TK t = {n, s};
        if ((int)heap.size() < (int)top_k) {
            heap.push_back(t);
            int c = (int)heap.size() - 1;
            while (c > 0) {
                int p = (c - 1) >> 1;
                if (heap[c].s >= heap[p].s) break;
                std::swap(heap[c], heap[p]);
                c = p;
            }
        } else if (s > heap[0].s) {
            heap[0] = t;
            int i = 0;
            while (i < (int)heap.size()) {
                int l = (i << 1) + 1, r = l + 1, sm = i;
                if (l < (int)heap.size() && heap[l].s < heap[sm].s) sm = l;
                if (r < (int)heap.size() && heap[r].s < heap[sm].s) sm = r;
                if (sm == i) break;
                std::swap(heap[i], heap[sm]);
                i = sm;
            }
        }
    }

    std::sort(heap.begin(), heap.end(), [](const TK& a, const TK& b) {
        if (a.s != b.s) return a.s > b.s;
        return a.i < b.i;
    });

    for (uint32_t i = 0; i < top_k && i < heap.size(); i++) {
        idx_out[i] = heap[i].i;
        score_out[i] = heap[i].s;
    }
}

} // namespace pq
} // namespace tq

#endif // __cplusplus
#endif // TQ_PRODUCT_QUANTIZER_H
