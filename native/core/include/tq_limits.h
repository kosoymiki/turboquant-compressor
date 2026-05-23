/**
 * TurboQuant Core — Compile-time and Runtime Limits
 * Ported from TypeScript src/core/limits.ts
 */

#ifndef TQ_LIMITS_H
#define TQ_LIMITS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Compile-time limits ─────────────────────────────────────────────────────
#define TQ_MAX_DIMENSIONS       131072
#define TQ_MAX_VECTORS         10000000
#define TQ_MAX_BITS             8
#define TQ_MAX_QJL_DIMS         255
#define TQ_MAX_CODEBOOK_CLUSTERS 256  // 8-bit quantization

#define TQ_MIN_DIMENSIONS       1
#define TQ_MIN_VECTOR_COUNT     1
#define TQ_MIN_BITS             1

// ── Runtime validation ───────────────────────────────────────────────────────
int tq_validate_params(uint32_t N, uint32_t dims, uint8_t bits, uint8_t qjl_dims);
const char* tq_limit_error(int err_code);

// Error codes: positive = invalid, 0 = OK
#define TQ_LIMIT_OK             0
#define TQ_LIMIT_Edims          1
#define TQ_LIMIT_Evectors        2
#define TQ_LIMIT_Ebits          3
#define TQ_LIMIT_Eqjl            4

#ifdef __cplusplus
}
#endif

#endif // TQ_LIMITS_H