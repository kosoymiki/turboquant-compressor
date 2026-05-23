/**
 * TurboQuant Core — Batch Search Pipeline
 * v4.6.0: TS format.ts v3 binary compatibility, zero warnings
 */
#include "tq_core.h"
#include "tq_kernel_inline.h"
#include "tq_beta_sphere.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <ctime>

static uint32_t crc32_c(const uint8_t* d, size_t n) {
    static uint32_t T[256];
    static bool ini = false;
    if (!ini) {
        for (int i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++) c = (c >> 1) ^ (c & 1 ? 0xEDB88320U : 0);
            T[i] = c;
        }
        ini = true;
    }
    uint32_t r = 0xFFFFFFFF;
    while (n--) r = T[(r ^ *d++) & 0xFF] ^ (r >> 8);
    return r ^ 0xFFFFFFFF;
}

static void b64_dec(const char* s, uint8_t* out, size_t* L) {
    int M[256];
    for (int i = 0; i < 256; i++) M[i] = -1;
    const char* C = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; C[i]; i++) M[(uint8_t)C[i]] = i;
    size_t n = 0;
    while (s[n]) n++;
    size_t p = 0;
    for (size_t i = 0; i + 4 <= n; i += 4) {
        int v0 = M[(uint8_t)s[i]], v1 = M[(uint8_t)s[i + 1]];
        if (v0 < 0 || v1 < 0) continue;
        if (p < *L) out[p++] = (v0 << 2) | (v1 >> 4);
        if (s[i + 2] == '=') break;
        int v2 = M[(uint8_t)s[i + 2]];
        if (v2 < 0) continue;
        if (p < *L) out[p++] = ((v1 & 15) << 4) | (v2 >> 2);
        if (s[i + 3] == '=') break;
        int v3 = M[(uint8_t)s[i + 3]];
        if (v3 < 0) continue;
        if (p < *L) out[p++] = ((v2 & 3) << 6) | v3;
    }
    *L = p;
}

static uint64_t ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static float qjl_dot(const float* q, uint8_t qd, const uint8_t* sk) {
    int32_t a = 0;
    for (uint8_t i = 0; i < qd; i++) {
        int qb = q[i] >= 0.0f ? 1 : 0;
        int sb = (sk[i >> 3] >> (i & 7)) & 1;
        a += qb == sb ? 1 : -1;
    }
    return (float)a / (float)qd;
}

struct TK { uint32_t i; float s; };
static void tk_push(std::vector<TK>& h, TK x) {
    h.push_back(x);
    int c = (int)h.size() - 1;
    while (c > 0) {
        int p = (c - 1) >> 1;
        if (h[c].s >= h[p].s) break;
        std::swap(h[c], h[p]);
        c = p;
    }
}
static void tk_rep(std::vector<TK>& h, TK x) {
    h[0] = x;
    int n = (int)h.size(), i = 0;
    while (i < n) {
        int l = (i << 1) + 1, r = l + 1, sm = i;
        if (l < n && h[l].s < h[sm].s) sm = l;
        if (r < n && h[r].s < h[sm].s) sm = r;
        if (sm == i) break;
        std::swap(h[i], h[sm]);
        i = sm;
    }
}

extern "C" {

int tq_search(const char* B, const float* Q, uint32_t D, uint32_t K,
              uint8_t METRIC, uint8_t UQJ, uint32_t* OI, float* OS,
              tq_search_stats* ST) {
    uint64_t T0 = ns();
    if (!B || !Q || !OI || !OS) return TQ_ERR_NULL;
    (void)METRIC;
    uint8_t* BIN = (uint8_t*)malloc(1024 * 1024);
    size_t BL = 1024 * 1024;
    b64_dec(B, BIN, &BL);
    if (BL < 80) { free(BIN); return TQ_ERR_FORMAT; }
    uint32_t MG; memcpy(&MG, BIN, 4);
    if (MG != 0x434D5154) { free(BIN); return TQ_ERR_FORMAT; }
    uint32_t VER, DBD, PAD, VCNT, SEED;
    memcpy(&VER,  BIN + 4,  4);
    memcpy(&DBD,  BIN + 8,  4);
    memcpy(&PAD,  BIN + 12, 4);
    memcpy(&VCNT, BIN + 16, 4);
    uint8_t BITS = BIN[20];
    memcpy(&SEED, BIN + 24, 4);
    uint8_t CBT = (VER >= 3) ? (BIN[28] & 3) : 0;
    uint32_t QJL_OFF = 0, QJL_LEN = 0;
    memcpy(&QJL_OFF, BIN + 56, 4);
    memcpy(&QJL_LEN, BIN + 60, 4);
    uint32_t CRC_STORED; memcpy(&CRC_STORED, BIN + 64, 4);
    if (DBD != D) { free(BIN); return TQ_ERR_DIM; }
    uint32_t HDR = 80;
    uint32_t NORM_OFF = HDR;
    uint32_t NORM_LEN = VCNT * 4;
    uint32_t QUANT_OFF = NORM_OFF + NORM_LEN;
    uint32_t QUANT_PER = (uint32_t)((uint64_t)PAD * BITS / 8);
    uint32_t PAYLOAD = NORM_LEN + VCNT * QUANT_PER;
    uint32_t CRC_COMPUTED = crc32_c(BIN + HDR, PAYLOAD);
    if (CRC_STORED != CRC_COMPUTED) { free(BIN); return TQ_ERR_FORMAT; }
    float* NORMS = (float*)malloc(NORM_LEN);
    memcpy(NORMS, BIN + NORM_OFF, NORM_LEN);
    uint8_t* QUANT = (uint8_t*)malloc(VCNT * QUANT_PER);
    memcpy(QUANT, BIN + QUANT_OFF, VCNT * QUANT_PER);
    uint8_t* QJL = nullptr;
    if (QJL_LEN > 0 && UQJ) {
        QJL = (uint8_t*)malloc(QJL_LEN);
        memcpy(QJL, BIN + QJL_OFF, QJL_LEN);
    }
    free(BIN);
    uint32_t P2 = D; while (P2 & (P2 - 1)) P2 <<= 1;
    float* SIGN = (float*)calloc(P2, sizeof(float));
    tq_init_sign_pattern(SIGN, P2, SEED);
    float* RQ = (float*)calloc(P2, sizeof(float));
    for (uint32_t i = 0; i < D; i++) RQ[i] = Q[i] * SIGN[i];
    tq_fwht_inplace(RQ, P2);
    tq_fwht_normalize(RQ, P2);
    free(SIGN);
    float QN2 = 0; for (uint32_t i = 0; i < D; i++) QN2 += Q[i] * Q[i];
    float QN = QN2 > 1e-9f ? 1.0f / sqrtf(QN2) : 1.0f;
    float* NQ = (float*)malloc(D * sizeof(float));
    for (uint32_t i = 0; i < D; i++) NQ[i] = RQ[i] * QN;
    free(RQ);
    tq_beta_codebook BCB;
    const double* CB = nullptr;
    if (CBT == TQ_CODEBOOK_BETA) {
        BCB = tq_compute_lloyd_max_beta_codebook((int)D, (int)BITS, 200, 1e-12f);
        CB = BCB.centroids;
    }
    float LM1 = (float)((1u << BITS) - 1);
    uint32_t* IB = (uint32_t*)malloc(P2 * 4);
    float* DB = (float*)malloc(P2 * 4);
    std::vector<TK> H; H.reserve(K);
    for (uint32_t vi = 0; vi < VCNT; vi++) {
        tq_unpack_bits(QUANT + (size_t)vi * QUANT_PER, IB, D, (int)BITS);
        if (CB) {
            for (uint32_t i = 0; i < D; i++) DB[i] = (float)CB[IB[i]];
        } else {
            for (uint32_t i = 0; i < D; i++) DB[i] = ((float)IB[i] / LM1) * 2.0f - 1.0f;
        }
        float DOT = 0;
        for (uint32_t i = 0; i < D; i++) DOT += NQ[i] * DB[i];
        float SC = DOT;
        if (SC > 1.0f) SC = 1.0f; else if (SC < -1.0f) SC = -1.0f;
        if (QJL && QJL_LEN > 0) {
            float QC = qjl_dot(NQ, 16, QJL + (size_t)vi * 2);
            SC = SC + 0.02f * (QC * 0.5f + 0.5f) - 0.01f;
            if (SC > 1.0f) SC = 1.0f; else if (SC < -1.0f) SC = -1.0f;
        }
        TK IT = {vi, SC};
        if ((int)H.size() < (int)K) tk_push(H, IT);
        else if (SC > H[0].s) tk_rep(H, IT);
    }
    free(NQ); free(DB); free(IB); free(NORMS); free(QUANT); free(QJL);
    std::sort(H.begin(), H.end(), [](const TK& a, const TK& b) {
        if (a.s != b.s) return a.s > b.s;
        return a.i < b.i;
    });
    for (uint32_t i = 0; i < K && i < H.size(); i++) { OI[i] = H[i].i; OS[i] = H[i].s; }
    uint64_t T1 = ns();
    if (ST) {
        memset(ST, 0, sizeof(*ST));
        ST->vector_count = VCNT;
        ST->dimensions = D;
        ST->top_k = K;
        ST->searched_count = VCNT;
        ST->qjl_applied = QJL_LEN > 0 ? 1 : 0;
        ST->first_score = H.empty() ? 0.0f : H[0].s;
        ST->search_ns = (T1 > T0) ? (uint32_t)(T1 - T0) : 0;
    }
    if (CBT == TQ_CODEBOOK_BETA) tq_free_beta_codebook(&BCB);
    return TQ_OK;
}

} // extern "C"
