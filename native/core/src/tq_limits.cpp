/**
 * TurboQuant Core — Limits Implementation
 */

#include "tq_limits.h"
#include <stddef.h>

int tq_validate_params(uint32_t N, uint32_t dims, uint8_t bits, uint8_t qjl_dims) {
    if (dims < TQ_MIN_DIMENSIONS || dims > TQ_MAX_DIMENSIONS) return TQ_LIMIT_Edims;
    if (N < TQ_MIN_VECTOR_COUNT || N > TQ_MAX_VECTORS) return TQ_LIMIT_Evectors;
    if (bits < TQ_MIN_BITS || bits > TQ_MAX_BITS) return TQ_LIMIT_Ebits;
    if (qjl_dims > TQ_MAX_QJL_DIMS) return TQ_LIMIT_Eqjl;
    return TQ_LIMIT_OK;
}

const char* tq_limit_error(int err_code) {
    switch (err_code) {
        case TQ_LIMIT_OK:      return "OK";
        case TQ_LIMIT_Edims:   return "dimensions out of range [1, 131072]";
        case TQ_LIMIT_Evectors: return "vector count out of range [1, 10000000]";
        case TQ_LIMIT_Ebits:   return "bits out of range [1, 8]";
        case TQ_LIMIT_Eqjl:    return "QJL dimensions out of range [0, 256]";
        default:               return "unknown error";
    }
}