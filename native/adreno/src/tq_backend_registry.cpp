/**
 * TurboQuant — backend registry. Selects best available backend.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_adreno.h"

namespace tq {

static Backend cpu_scalar_backend = {
    BACKEND_CPU_SCALAR, "cpu_scalar",
    cpu_scalar_mse_score, cpu_scalar_value_dequant, cpu_scalar_fused_attention
};

static Backend neon_backend = {
    BACKEND_CPU_NEON, "cpu_neon",
    neon_mse_score, neon_value_dequant, neon_fused_attention
};

bool has_neon() {
#ifdef TQ_HAS_NEON
    return true;
#else
    return false;
#endif
}

const Backend& get_backend() {
    if (has_neon()) return neon_backend;
    return cpu_scalar_backend;
}

const Backend& get_backend_by_kind(BackendKind kind) {
    switch (kind) {
        case BACKEND_CPU_NEON: return neon_backend;
        default: return cpu_scalar_backend;
    }
}

} // namespace tq
