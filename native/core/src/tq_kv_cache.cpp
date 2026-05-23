/**
 * TurboQuant Core — KV Cache Implementation
 * v4.6.1: P0 Hadamard QJL, P1 2-bit Beta, P2 ProductQuantizer
 */

#include "tq_kv_cache.h"
#include "tq_core.h"
#include "tq_beta_sphere.h"
#include "tq_kernel_inline.h"
#include "tq_base64.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace tq { namespace kv {

struct Q8Vector { int8_t* quantized; float scale; };

struct Cache {
    uint32_t head_dim, key_bits, value_bits, value_group_size, buffer_size;
    float sparse_threshold;
    uint8_t use_q8_query, outlier_aware_rotation;
    uint32_t sink_tokens, rotation_seed;
    float* sign_pattern;
    uint32_t pad_dim;
    double* centroids;
    int n_centroids;
    float* centroid_values;
    uint16_t* channel_order;
    uint32_t channel_order_len;
    tq_key_quantized key_quantized;
    tq_value_quantized value_quantized;
    float* key_buffer;
    float* value_buffer;
    uint32_t buffer_token_count, seq_len;
    uint64_t sparse_skipped, total_scores;
};

static void quantize_q8(const float* vec, uint32_t d, Q8Vector* out) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < d; i++) { float a = vec[i]; if (a < 0) a = -a; if (a > amax) amax = a; }
    if (amax < 1e-30f) { out->quantized = (int8_t*)calloc(d, 1); out->scale = 1e-10f; return; }
    out->quantized = (int8_t*)malloc(d);
    out->scale = amax / 127.0f;
    float inv = 127.0f / amax;
    for (uint32_t i = 0; i < d; i++) out->quantized[i] = (int8_t)::roundf(vec[i] * inv);
}

static uint16_t* compute_channel_order(const float* keys, uint32_t nt, uint32_t d) {
    float* v = (float*)calloc(d, 4);
    for (uint32_t i = 0; i < d; i++) {
        float s = 0, s2 = 0;
        for (uint32_t t = 0; t < nt; t++) { float x = keys[t * d + i]; s += x; s2 += x * x; }
        v[i] = s2 / nt - (s / nt) * (s / nt);
    }
    uint16_t* o = (uint16_t*)malloc(d * 2);
    for (uint32_t i = 0; i < d; i++) o[i] = i;
    for (uint32_t i = 0; i < d - 1; i++)
        for (uint32_t j = i + 1; j < d; j++)
            if (v[j] > v[i]) { float tv = v[i]; v[i] = v[j]; v[j] = tv;
                uint16_t ti = o[i]; o[i] = o[j]; o[j] = ti; }
    free(v); return o;
}

static int quantize_keys(Cache* c, const float* keys, uint32_t nt, tq_key_quantized* kq) {
    uint32_t d = c->head_dim;
    uint32_t bpt = (d * c->key_bits + 7) / 8;
    kq->indices = (uint8_t*)malloc(nt * bpt);
    kq->norms = (float*)malloc(nt * 4);
    kq->num_tokens = nt; kq->head_dim = d; kq->bits = c->key_bits;
    if (!kq->indices || !kq->norms) { free(kq->indices); free(kq->norms); return TQ_ERR_ALLOC; }
    memset(kq->indices, 0, nt * bpt);
    float* tmp = (float*)malloc(c->pad_dim * 4);
    float* rot = (float*)malloc(c->pad_dim * 4);
    if (c->outlier_aware_rotation && !c->channel_order && nt >= 4) {
        c->channel_order = compute_channel_order(keys, nt, d); c->channel_order_len = d; }
    for (uint32_t t = 0; t < nt; t++) {
        memset(tmp, 0, c->pad_dim * 4);
        for (uint32_t i = 0; i < d; i++) tmp[i] = keys[t * d + i] * c->sign_pattern[i];
        tq_fwht_inplace(tmp, c->pad_dim); tq_fwht_normalize(tmp, c->pad_dim);
        float nsq = 0.0f; for (uint32_t i = 0; i < d; i++) nsq += tmp[i] * tmp[i];
        float norm = (nsq > 1e-10f) ? sqrtf(nsq) : 1.0f;
        kq->norms[t] = norm;
        float inv = 1.0f / norm; for (uint32_t i = 0; i < d; i++) tmp[i] *= inv;
        if (c->channel_order) { for (uint32_t i = 0; i < d; i++) rot[i] = tmp[c->channel_order[i]];
            for (uint32_t i = d; i < c->pad_dim; i++) rot[i] = 0.0f; }
        else { memcpy(rot, tmp, d * 4); }
        tq_fwht_inplace(rot, c->pad_dim); tq_fwht_normalize(rot, c->pad_dim);
        for (uint32_t i = 0; i < d; i++) {
            float v = rot[i]; if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
            int lo = 0, hi = c->n_centroids - 1;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (c->centroids[mid] < v) lo = mid + 1; else hi = mid; }
            int best = lo; float bd = 1e9f;
            for (int j = std::max(0, lo - 1); j <= std::min(c->n_centroids - 1, lo + 1); j++) {
                float d2 = v - (float)c->centroids[j]; if (d2 < 0) d2 = -d2; if (d2 < bd) { bd = d2; best = j; } }
            uint64_t bo = (uint64_t)t * bpt * 8 + i * c->key_bits;
            uint32_t bp = (uint32_t)(bo >> 3), bi = (uint32_t)(bo & 7);
            uint8_t m = (1 << c->key_bits) - 1;
            kq->indices[bp] |= (uint8_t)((best & m) << bi);
            if (bi + c->key_bits > 8) kq->indices[bp + 1] |= (uint8_t)((best & m) >> (8 - bi)); } }
    free(tmp); free(rot); return TQ_OK;
}

} } // namespace kv, tq

using namespace tq::kv;

int tq_kv_cache_create(tq_kv_cache_config* cfg, void** out) {
    if (!cfg || !out) return TQ_ERR_NULL;
    if (cfg->head_dim < 1 || cfg->head_dim > TQ_KV_MAX_HEAD_DIM) return TQ_ERR_DIM;
    Cache* c = (Cache*)calloc(1, sizeof(Cache)); if (!c) return TQ_ERR_ALLOC;
    c->head_dim = cfg->head_dim;
    c->key_bits = cfg->key_bits ? cfg->key_bits : 4;
    c->value_bits = cfg->value_bits ? cfg->value_bits : 2;
    c->value_group_size = cfg->value_group_size ? cfg->value_group_size : TQ_KV_BLOCK_SIZE;
    c->buffer_size = cfg->buffer_size ? cfg->buffer_size : TQ_KV_BLOCK_SIZE;
    c->sparse_threshold = cfg->sparse_threshold > 0 ? cfg->sparse_threshold : TQ_KV_SPARSE_DEFAULT;
    c->use_q8_query = cfg->use_q8_query != 0;
    c->outlier_aware_rotation = cfg->outlier_aware_rotation != 0;
    c->sink_tokens = cfg->sink_tokens ? cfg->sink_tokens : TQ_KV_SINK_TOKENS;
    c->rotation_seed = cfg->rotation_seed ? cfg->rotation_seed : 42;
    c->pad_dim = 1; while (c->pad_dim < c->head_dim) c->pad_dim <<= 1;
    c->sign_pattern = (float*)calloc(c->pad_dim, 4);
    tq_init_sign_pattern(c->sign_pattern, c->pad_dim, c->rotation_seed);
    c->n_centroids = 1 << c->key_bits;
    tq_beta_codebook bcb = tq_compute_lloyd_max_beta_codebook((int)c->head_dim, (int)c->key_bits, 200, 1e-12);
    c->centroids = bcb.centroids;
    c->centroid_values = (float*)malloc((uint32_t)c->n_centroids * 4);
    for (int i = 0; i < c->n_centroids; i++) c->centroid_values[i] = (float)c->centroids[i];
    *out = c; return TQ_OK;
}

void tq_kv_cache_destroy(void* cache) {
    if (!cache) return; Cache* c = (Cache*)cache;
    free(c->sign_pattern); free(c->centroid_values); free(c->channel_order);
    free(c->key_quantized.indices); free(c->key_quantized.norms);
    free(c->value_quantized.data); free(c->value_quantized.scales); free(c->value_quantized.zeros);
    free(c->key_buffer); free(c->value_buffer); free(c);
}

int tq_kv_cache_prefill(void* cache, const float* keys, const float* values, uint32_t nt) {
    if (!cache || !keys) return TQ_ERR_NULL; Cache* c = (Cache*)cache; uint32_t d = c->head_dim;
    c->seq_len = nt;
    if (nt <= c->buffer_size) {
        c->key_buffer = (float*)malloc(nt * d * 4); c->value_buffer = (float*)malloc(nt * d * 4);
        memcpy(c->key_buffer, keys, nt * d * 4); memcpy(c->value_buffer, values, nt * d * 4);
        c->buffer_token_count = nt; return TQ_OK; }
    uint32_t nq = nt - c->buffer_size;
    int r = quantize_keys(c, keys, nq, &c->key_quantized); if (r != TQ_OK) return r;
    c->key_buffer = (float*)malloc(c->buffer_size * d * 4);
    c->value_buffer = (float*)malloc(c->buffer_size * d * 4);
    memcpy(c->key_buffer, keys + nq * d, c->buffer_size * d * 4);
    memcpy(c->value_buffer, values + nq * d, c->buffer_size * d * 4);
    c->buffer_token_count = c->buffer_size; return TQ_OK;
}

int tq_kv_cache_append(void* cache, const float* key, const float* value) {
    if (!cache || !key) return TQ_ERR_NULL; Cache* c = (Cache*)cache; uint32_t d = c->head_dim;
    c->seq_len++;
    if (!c->key_buffer) {
        c->key_buffer = (float*)malloc(d * 4); c->value_buffer = (float*)malloc(d * 4);
        memcpy(c->key_buffer, key, d * 4); memcpy(c->value_buffer, value, d * 4);
        c->buffer_token_count = 1; return TQ_OK; }
    float* nk = (float*)malloc((c->buffer_token_count + 1) * d * 4);
    float* nv = (float*)malloc((c->buffer_token_count + 1) * d * 4);
    memcpy(nk, c->key_buffer, c->buffer_token_count * d * 4);
    memcpy(nv, c->value_buffer, c->buffer_token_count * d * 4);
    memcpy(nk + c->buffer_token_count * d, key, d * 4);
    memcpy(nv + c->buffer_token_count * d, value, d * 4);
    free(c->key_buffer); free(c->value_buffer);
    c->key_buffer = nk; c->value_buffer = nv; c->buffer_token_count++; return TQ_OK;
}

int tq_kv_cache_attention(void* cache, const float* query, float* output) {
    if (!cache || !query || !output) return TQ_ERR_NULL;
    Cache* c = (Cache*)cache; uint32_t d = c->head_dim; float ss = 1.0f / sqrtf((float)d);
    float* rq = (float*)malloc(c->pad_dim * 4); memset(rq, 0, c->pad_dim * 4);
    for (uint32_t i = 0; i < d; i++) rq[i] = query[i] * c->sign_pattern[i];
    tq_fwht_inplace(rq, c->pad_dim); tq_fwht_normalize(rq, c->pad_dim);
    Q8Vector q8; if (c->use_q8_query) quantize_q8(rq, d, &q8);
    memset(output, 0, d * 4); float li = 0.0f, mi = -1e30f;
    uint32_t q_tokens = c->key_quantized.num_tokens;
    uint32_t b_tokens = c->buffer_token_count;
    uint32_t bpt = (d * c->key_bits + 7) / 8; uint8_t km = (1 << c->key_bits) - 1;

    // Handle unquantized buffer tokens (when no quantized keys exist)
    if (q_tokens == 0 && b_tokens > 0) {
        for (uint32_t t = 0; t < b_tokens; t++) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < d; i++) {
                dot += rq[i] * c->key_buffer[t * d + i];
            }
            c->total_scores++;
            float score = dot * ss;
            float mn = std::max(score, mi);
            float a = expf(mi - mn);
            float p = expf(score - mn);
            for (uint32_t j = 0; j < d; j++) {
                output[j] = output[j] * a + p * c->key_buffer[t * d + j];
            }
            li = li * a + p;
            mi = mn;
        }
        if (li > 1e-10f) {
            float inv = 1.0f / li;
            for (uint32_t i = 0; i < d; i++) output[i] *= inv;
        }
        free(rq);
        return TQ_OK;
    }

    // Process quantized tokens
    for (uint32_t t = 0; t < q_tokens; t++) {
        float dot = 0.0f;
        for (uint32_t i = 0; i < d; i++) {
            uint64_t bo = (uint64_t)t * bpt * 8 + i * c->key_bits;
            uint32_t bp = (uint32_t)(bo >> 3), bi = (uint32_t)(bo & 7);
            uint32_t idx = (c->key_quantized.indices[bp] >> bi) & km;
            if (bi + c->key_bits > 8) idx |= (c->key_quantized.indices[bp + 1] >> (8 - bi)) & km;
            if (c->use_q8_query) dot += (float)q8.quantized[i] * q8.scale * c->centroid_values[idx];
            else dot += rq[i] * c->centroid_values[idx]; }
        c->total_scores++;
        float score = dot * c->key_quantized.norms[t] * ss;
        float mn = std::max(score, mi); float a = expf(mi - mn); float p = expf(score - mn);
        for (uint32_t j = 0; j < d; j++) output[j] = output[j] * a + p * rq[j];
        li = li * a + p;
        mi = mn;
    }
    if (li > 1e-10f) {
        float inv = 1.0f / li;
        for (uint32_t i = 0; i < d; i++) output[i] *= inv;
    }
    free(rq);
    if (c->use_q8_query) free(q8.quantized);
    return TQ_OK;
}

int tq_kv_cache_dequantize_keys(void* cache, float* out) {
    if (!cache || !out) return TQ_ERR_NULL; Cache* c = (Cache*)cache;
    uint32_t d = c->head_dim, nt = c->key_quantized.num_tokens;
    uint32_t bpt = (d * c->key_bits + 7) / 8; uint8_t km = (1 << c->key_bits) - 1;
    for (uint32_t t = 0; t < nt; t++) {
        for (uint32_t i = 0; i < d; i++) {
            uint64_t bo = (uint64_t)t * bpt * 8 + i * c->key_bits;
            uint32_t bp = (uint32_t)(bo >> 3), bi = (uint32_t)(bo & 7);
            uint32_t idx = (c->key_quantized.indices[bp] >> bi) & km;
            if (bi + c->key_bits > 8) idx |= (c->key_quantized.indices[bp + 1] >> (8 - bi)) & km;
            out[t * d + i] = c->centroid_values[idx] * c->key_quantized.norms[t]; } }
    return TQ_OK;
}

void tq_kv_cache_get_stats(void* cache, tq_kv_stats* s) {
    if (!cache || !s) return; Cache* c = (Cache*)cache;
    s->seq_len = c->seq_len; s->sparse_skipped = c->sparse_skipped;
    s->total_scores = c->total_scores;
    s->sparse_skip_ratio = c->total_scores > 0 ? (float)c->sparse_skipped / c->total_scores : 0.0f;
}

void tq_kv_cache_get_memory(void* cache, tq_kv_memory* m) {
    if (!cache || !m) return; Cache* c = (Cache*)cache;
    m->quantized_keys_bytes = c->key_quantized.indices ? c->key_quantized.num_tokens * (c->head_dim * c->key_bits + 7) / 8 : 0;
    m->quantized_values_bytes = c->value_quantized.data ? c->value_quantized.num_tokens * c->head_dim * c->value_bits / 8 : 0;
    m->buffer_bytes = c->key_buffer ? c->buffer_token_count * c->head_dim * 4 * 2 : 0;
    m->total_bytes = m->quantized_keys_bytes + m->quantized_values_bytes + m->buffer_bytes;
}

uint32_t tq_kv_cache_get_seq_len(void* cache) {
    return cache ? ((Cache*)cache)->seq_len : 0;
}

int tq_kv_cache_serialize(void* cache, char* out, size_t* len) {
    (void)cache; (void)out; (void)len;
    return TQ_ERR_INTERNAL;
}

int tq_kv_cache_deserialize(void** cache_out, const char* base64, size_t base64_len) {
    (void)cache_out; (void)base64; (void)base64_len;
    return TQ_ERR_INTERNAL;
}
