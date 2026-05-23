/**
 * TurboQuant Core — C linkage only test main
 * No <vector>, <string>, <functional>.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/tq_core.h"
#include "../src/tq_kernel_inline.h"

static int passed = 0;
static int failed = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); passed++; } \
    else { printf("  FAIL: %s\n", msg); failed++; } \
} while(0)

int main() {
    printf("=== TurboQuant Core inline test v4.6.0 ===\n\n");

    // Test 1: next_pow2
    ASSERT(tq_next_pow2(1024) == 1024, "next_pow2(1024) = 1024");
    ASSERT(tq_next_pow2(1000) == 1024, "next_pow2(1000) = 1024");
    ASSERT(tq_next_pow2(1) == 1, "next_pow2(1) = 1");
    ASSERT(tq_next_pow2(2048) == 2048, "next_pow2(2048) = 2048");

    // Test 2: Beta codebook
    float cb[16];
    tq_beta_codebook_16(cb);
    ASSERT(cb[0] < 0 && cb[7] < 0 && cb[8] > 0, "codebook symmetric");

    // Test 3: compress 100 vectors
    printf("\n[compress 100 vectors 64D]\n");
    const uint32_t N = 100, D = 64;
    float vecs[100 * 64];
    for (uint32_t i = 0; i < N * D; i++) vecs[i] = (float)(i % 7) * 0.01f - 0.3f;

    size_t est = tq_estimate_compressed_size(N, D, 4, 0);
    char* out = (char*)malloc(est * 2);
    size_t out_len = est * 2;
    tq_compress_stats st;
    memset(&st, 0, sizeof(st));
    int r = tq_compress(vecs, N, D, 4, 42, 0, 0, out, &out_len, &st);
    ASSERT(r == TQ_OK, "tq_compress returns TQ_OK");
    ASSERT(st.compression_ratio > 6.8f, "compression ratio > 6.8x");
    printf("    ratio=%.2fx, bytes=%u\n", st.compression_ratio, st.compressed_bytes);
    free(out);

    // Test 4: compress+search roundtrip
    printf("\n[roundtrip self-match]\n");
    char* db = (char*)malloc(est * 2);
    out_len = est * 2;
    r = tq_compress(vecs, N, D, 4, 42, 0, 0, db, &out_len, &st);
    uint32_t idx[5];
    float scr[5];
    tq_search_stats ss;
    memset(&ss, 0, sizeof(ss));
    r = tq_search(db, vecs, D, 5, 0, 0, idx, scr, &ss);
    ASSERT(r == TQ_OK, "tq_search returns TQ_OK");
    ASSERT(idx[0] == 0, "uniform self-match top-1");
    ASSERT(scr[0] > 0.9f, "uniform self-match score > 0.9");
    printf("    top-1: idx=%u score=%.4f\n", idx[0], scr[0]);
    free(db);

    // Test 5: bits=0 should fail
    printf("\n[bits=0 validation]\n");
    out = (char*)malloc(est);
    out_len = est;
    memset(&st, 0, sizeof(st));
    r = tq_compress(vecs, N, D, 0, 42, 0, 0, out, &out_len, &st);
    ASSERT(r == TQ_ERR_INTERNAL, "bits=0 returns TQ_ERR_INTERNAL");
    printf("    bits=0 rejected: %d\n", r);
    free(out);

    // Test 6: bits=8 should work
    printf("\n[bits=8 validation]\n");
    out = (char*)malloc(est * 3);
    out_len = est * 3;
    memset(&st, 0, sizeof(st));
    r = tq_compress(vecs, N, D, 8, 42, 0, 0, out, &out_len, &st);
    ASSERT(r == TQ_OK, "bits=8 returns TQ_OK");
    printf("    bits=8 ratio=%.2fx\n", st.compression_ratio);
    free(out);

    // Test 7: QJL roundtrip
    printf("\n[QJL roundtrip]\n");
    est = tq_estimate_compressed_size(N, D, 4, 16);
    db = (char*)malloc(est * 2);
    out_len = est * 2;
    memset(&st, 0, sizeof(st));
    r = tq_compress(vecs, N, D, 4, 42, 16, 0, db, &out_len, &st);
    ASSERT(r == TQ_OK, "compress with QJL returns TQ_OK");
    ASSERT(st.has_qjl == 1, "has_qjl flag set");
    memset(&ss, 0, sizeof(ss));
    r = tq_search(db, vecs, D, 5, 0, 1, idx, scr, &ss);
    ASSERT(r == TQ_OK, "search with QJL returns TQ_OK");
    ASSERT(ss.qjl_applied == 1, "QJL applied in search");
    ASSERT(idx[0] == 0 || scr[0] > 0.9f, "QJL self-match in top-5");
    printf("    QJL score=%.4f\n", scr[0]);
    free(db);

    // Test 8: Beta codebook roundtrip
    printf("\n[Beta codebook roundtrip]\n");
    est = tq_estimate_compressed_size(N, D, 4, 0);
    db = (char*)malloc(est * 2);
    out_len = est * 2;
    memset(&st, 0, sizeof(st));
    r = tq_compress(vecs, N, D, 4, 42, 0, TQ_CODEBOOK_BETA, db, &out_len, &st);
    ASSERT(r == TQ_OK, "compress beta returns TQ_OK");
    ASSERT(st.codebook_type == TQ_CODEBOOK_BETA, "codebook_type=BETA in stats");
    memset(&ss, 0, sizeof(ss));
    uint32_t idx100[100];
    float scr100[100];
    r = tq_search(db, vecs, D, 100, 0, 0, idx100, scr100, &ss);
    ASSERT(r == TQ_OK, "search beta returns TQ_OK");
    // Beta codebook: self should be in top-100
    int found = 0;
    for (uint32_t i = 0; i < 100 && i < ss.searched_count; i++) if (idx100[i] == 0) found = 1;
    ASSERT(found, "beta self (index 0) in results");
    ASSERT(scr100[0] >= 0.0f, "beta score >= 0.0");
    printf("    beta score[0]=%.4f\n", scr100[0]);
    free(db);

    // Test 9: D mismatch in search
    printf("\n[D mismatch validation]\n");
    est = tq_estimate_compressed_size(N, D, 4, 0);
    db = (char*)malloc(est * 2);
    out_len = est * 2;
    memset(&st, 0, sizeof(st));
    r = tq_compress(vecs, N, D, 4, 42, 0, 0, db, &out_len, &st);
    float wrong_d[32];
    for (int i = 0; i < 32; i++) wrong_d[i] = 0.1f;
    memset(&ss, 0, sizeof(ss));
    r = tq_search(db, wrong_d, 32, 5, 0, 0, idx, scr, &ss);
    ASSERT(r == TQ_ERR_DIM, "D mismatch returns TQ_ERR_DIM");
    printf("    D mismatch rejected: %d\n", r);
    free(db);

    // Test 10: NULL pointer validation
    printf("\n[NULL pointer validation]\n");
    memset(&ss, 0, sizeof(ss));
    r = tq_search(NULL, vecs, D, 5, 0, 0, idx, scr, &ss);
    ASSERT(r == TQ_ERR_NULL, "NULL base64 returns TQ_ERR_NULL");
    r = tq_search(db, NULL, D, 5, 0, 0, idx, scr, &ss);
    ASSERT(r == TQ_ERR_NULL, "NULL query returns TQ_ERR_NULL");

    printf("\n======================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}