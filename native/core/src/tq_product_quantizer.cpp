/**
 * TurboQuant Core — Product Quantization Implementation
 * Jegou et al., "Product Quantization for Nearest Neighbor Search" (TPAMI 2011)
 */

#include "tq_product_quantizer.h"
#include "tq_base64.h"
#include "tq_core.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

// ── C API stubs (C++ implementation in header) ───────────────────────────────

tq_pq_config tq_pq_default_config(uint32_t dimensions) {
    tq_pq_config cfg = {
        .dimensions = dimensions,
        .num_subspaces = 8,
        .bits_per_subspace = 8,
        .k_means_iterations = 50,
        .use_opq = 0,
        .rotation_seed = 42
    };
    return cfg;
}

tq_pq_model* tq_pq_train(const tq_pq_config* cfg, const float* train_x, uint32_t num_train, uint32_t seed) {
    if (!cfg || !train_x) return nullptr;
    tq_pq_model* m = (tq_pq_model*)calloc(1, sizeof(tq_pq_model));
    if (!m) return nullptr;

    m->codebook.dimensions = cfg->dimensions;
    m->codebook.num_subspaces = cfg->num_subspaces;
    m->codebook.bits_per_subspace = cfg->bits_per_subspace;
    m->codebook.k_log2 = cfg->bits_per_subspace;
    m->codebook.k = 1 << cfg->bits_per_subspace;
    m->codebook.dim_per_sub = cfg->dimensions / cfg->num_subspaces;
    m->codebook.use_opq = cfg->use_opq;
    m->codebook.rotation_seed = cfg->rotation_seed;
    m->database.num_vectors = num_train;
    m->database.num_subspaces = cfg->num_subspaces;
    m->database.codes_per_vec = (cfg->num_subspaces * cfg->bits_per_subspace + 7) / 8;
    m->database.codes = (uint8_t*)malloc(num_train * m->database.codes_per_vec);
    m->database.norms = (float*)malloc(num_train * 4);

    // Copy centroids from C++ ProductQuantizer
    tq::pq::ProductQuantizer pq(cfg->dimensions, cfg->num_subspaces, cfg->bits_per_subspace);
    pq.train(train_x, num_train, seed, cfg->k_means_iterations, cfg->use_opq);

    size_t cents_size = cfg->num_subspaces * pq.k() * pq.dimensions() / cfg->num_subspaces * sizeof(float);
    m->codebook.centroids = (float*)malloc(cents_size);
    // Note: For full integration, we'd need to copy the centroids from pq.cents_
    // For now, mark as trained for search to use
    m->codebook.trained = 1;
    m->database.initialized = 1;

    return m;
}

void tq_pq_encode(const tq_pq_model* model, const float* x, uint8_t* code) {
    (void)model; (void)x; (void)code;
    // TODO: Wire C++ ProductQuantizer.encode()
}

void tq_pq_encode_batch(const tq_pq_model* model, const float* X, uint32_t num_vec, uint8_t* codes_out) {
    (void)model; (void)X; (void)num_vec; (void)codes_out;
}

void tq_pq_decode(const tq_pq_model* model, const uint8_t* code, float* x_out) {
    (void)model; (void)code; (void)x_out;
}

void tq_pq_search(const tq_pq_model* model, const float* query, uint32_t top_k, uint32_t* idx_out, float* score_out) {
    (void)model; (void)query; (void)top_k; (void)idx_out; (void)score_out;
}

int tq_pq_serialize(const tq_pq_model* model, char** out_base64, size_t* out_len) {
    if (!model || !out_base64 || !out_len) return TQ_ERR_NULL;
    (void)model; (void)out_base64; (void)out_len;
    return TQ_ERR_INTERNAL;
}

int tq_pq_deserialize(const char* base64, size_t len, tq_pq_model** model_out) {
    if (!base64 || !model_out) return TQ_ERR_NULL;
    (void)base64; (void)len; (void)model_out;
    return TQ_ERR_INTERNAL;
}

void tq_pq_model_free(tq_pq_model* model) {
    if (!model) return;
    if (model->codebook.centroids) free(model->codebook.centroids);
    if (model->codebook.rotation) free(model->codebook.rotation);
    if (model->database.codes) free(model->database.codes);
    if (model->database.norms) free(model->database.norms);
    free(model);
}
