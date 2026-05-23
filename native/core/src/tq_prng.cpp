/**
 * TurboQuant Core — Mulberry32 PRNG Implementation
 */

#include "tq_prng.h"
#include <math.h>

void tq_prng_init(tq_prng_t* prng, uint32_t seed) {
    prng->state = seed;
}

uint32_t tq_prng_next(tq_prng_t* prng) {
    uint32_t x = prng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    prng->state = x;
    return x;
}

float tq_prng_random_float(tq_prng_t* prng) {
    return (float)(tq_prng_next(prng) >> 8) / 16777216.0f; // divide by 2^24
}

double tq_prng_random_double(tq_prng_t* prng) {
    return (double)(tq_prng_next(prng) >> 8) / 16777216.0;
}

uint32_t tq_prng_random_uint(tq_prng_t* prng, uint32_t max) {
    if (max == 0) return 0;
    return tq_prng_next(prng) % (max + 1);
}

float tq_prng_gaussian(tq_prng_t* prng) {
    // Box-Muller transform
    float u1 = tq_prng_random_float(prng);
    float u2 = tq_prng_random_float(prng);
    if (u1 <= 0.0f) u1 = 1e-10f;
    float mag = sqrtf(-2.0f * logf(u1));
    return mag * cosf(2.0f * 3.14159265359f * u2);
}