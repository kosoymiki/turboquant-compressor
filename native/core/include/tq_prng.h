/**
 * TurboQuant Core — Mulberry32 PRNG
 * Ported from TypeScript src/core/prng.ts
 * Seedable, fast, good distribution for 32-bit integers and floats.
 */

#ifndef TQ_PRNG_H
#define TQ_PRNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t state;
} tq_prng_t;

// Initialize PRNG with seed
void tq_prng_init(tq_prng_t* prng, uint32_t seed);

// Advance state (called implicitly by each random_* function)
uint32_t tq_prng_next(tq_prng_t* prng);

// Uniform float32 in [0, 1)
float tq_prng_random_float(tq_prng_t* prng);

// Uniform float64 in [0, 1)
double tq_prng_random_double(tq_prng_t* prng);

// Uniform uint32 in [0, max]
uint32_t tq_prng_random_uint(tq_prng_t* prng, uint32_t max);

// Gaussian (Box-Muller), mean=0, std=1
float tq_prng_gaussian(tq_prng_t* prng);

#ifdef __cplusplus
}
#endif

#endif // TQ_PRNG_H