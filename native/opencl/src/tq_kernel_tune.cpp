/**
 * TurboQuant Per-Kernel Tuning — implementation.
 *
 * Maps kernel names to optimal dispatch parameters per GPU generation.
 * Covers all TQ compute kernels: attention, dequant, MSE, QJL.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_kernel_tune.h"
#include <cstring>

namespace tq {

// --- Kernel tuning database ---

static KernelTuneParams tune_fused_attention(const GpuProfile& p) {
    KernelTuneParams t;
    t.kernel_name = "tq_fused_attention";
    t.wg_size_x = p.attention_wg_size;
    t.preferred_multiple = (uint32_t)p.compute_wave;
    t.subgroup_use = SubgroupUse::REDUCE;
    t.items_per_thread = (p.gen == GpuGen::ADRENO_8XX) ? 4 : 2;
    t.vec_width = p.supports_fp16_fma ? 8 : 4;  // half8 or float4
    t.use_gmem_tiling = (p.tile_strategy == TileStrategy::GMEM_TILED);
    t.prefetch_global = (p.gen >= GpuGen::ADRENO_7XX);

    // Tiling for attention: M=tokens, N=dim, K=head_dim
    if (p.gen == GpuGen::ADRENO_8XX) {
        t.tile_m = 8;
        t.tile_n = 16;
        t.tile_k = 8;
        t.input_layout = MemoryLayout::TILED_16x4;
    } else if (p.gen == GpuGen::ADRENO_7XX) {
        t.tile_m = 4;
        t.tile_n = 16;
        t.tile_k = 4;
        t.input_layout = MemoryLayout::TILED_8x8;
    } else {
        t.tile_m = 4;
        t.tile_n = 8;
        t.tile_k = 4;
        t.input_layout = MemoryLayout::TILED_4x4;
    }

    // The shipped kernel declares static local arrays, so host-side dynamic __local is unused.
    t.local_mem_bytes = 0;
    return t;
}

static KernelTuneParams tune_value_dequant(const GpuProfile& p) {
    KernelTuneParams t;
    t.kernel_name = "tq_value_dequant";
    t.wg_size_x = p.dequant_wg_size;
    t.preferred_multiple = (uint32_t)p.compute_wave;
    t.subgroup_use = SubgroupUse::NONE;  // Embarrassingly parallel
    t.items_per_thread = (p.gen == GpuGen::ADRENO_8XX) ? 8 : 4;
    t.vec_width = 4;  // Always float4 for dequant output
    t.input_layout = MemoryLayout::LINEAR;
    t.output_layout = MemoryLayout::LINEAR;
    t.prefetch_global = true;  // Sequential access pattern
    t.local_mem_bytes = 0;     // No shared memory needed
    return t;
}

static KernelTuneParams tune_mse_score(const GpuProfile& p) {
    KernelTuneParams t;
    t.kernel_name = "tq_mse_score";
    t.wg_size_x = p.mse_wg_size;
    t.preferred_multiple = (uint32_t)p.compute_wave;
    t.subgroup_use = SubgroupUse::REDUCE;  // Reduction across dim
    t.items_per_thread = 4;
    t.vec_width = p.supports_fp16_fma ? 8 : 4;
    t.input_layout = MemoryLayout::INTERLEAVED;  // Coalesced centroid access
    t.use_gmem_tiling = false;  // Scoring is bandwidth-bound, not compute-bound
    t.prefetch_global = (p.gen >= GpuGen::ADRENO_7XX);

    // Local memory for partial reductions
    t.local_mem_bytes = t.wg_size_x * sizeof(float);
    return t;
}

static KernelTuneParams tune_qjl_score(const GpuProfile& p) {
    KernelTuneParams t;
    t.kernel_name = "tq_qjl_score";
    t.wg_size_x = p.qjl_wg_size;
    t.preferred_multiple = (uint32_t)p.compute_wave;
    t.subgroup_use = SubgroupUse::REDUCE;  // popcount reduction
    t.items_per_thread = 8;  // Many uint32 words per thread
    t.vec_width = 4;  // uint4 for sign bits
    t.input_layout = MemoryLayout::LINEAR;
    t.use_gmem_tiling = false;
    t.prefetch_global = true;

    // Local memory for partial popcount sums
    t.local_mem_bytes = t.wg_size_x * sizeof(uint32_t);
    return t;
}

static KernelTuneParams tune_attention_logits(const GpuProfile& p) {
    KernelTuneParams t;
    t.kernel_name = "tq_attention_logits";
    t.wg_size_x = p.attention_wg_size;
    t.preferred_multiple = (uint32_t)p.compute_wave;
    t.subgroup_use = SubgroupUse::REDUCE;
    t.items_per_thread = 2;
    t.vec_width = p.supports_fp16_fma ? 8 : 4;
    t.input_layout = MemoryLayout::TILED_16x4;  // Wide Q rows
    t.use_gmem_tiling = (p.gmem_size_kb >= 1024);
    t.prefetch_global = true;
    // The shipped kernel has no dynamic __local argument.
    t.local_mem_bytes = 0;
    return t;
}

static KernelTuneParams tune_attention_apply_values(const GpuProfile& p) {
    KernelTuneParams t;
    t.kernel_name = "tq_attention_apply_values";
    t.wg_size_x = p.attention_wg_size;
    t.preferred_multiple = (uint32_t)p.compute_wave;
    t.items_per_thread = 1;
    t.vec_width = 4;
    t.input_layout = MemoryLayout::LINEAR;
    t.output_layout = MemoryLayout::LINEAR;
    t.local_mem_bytes = 0;
    return t;
}

static KernelTuneParams tune_generic(const GpuProfile& p, const char* name) {
    KernelTuneParams t;
    t.kernel_name = name;
    t.wg_size_x = p.preferred_wg_size;
    t.preferred_multiple = (uint32_t)p.compute_wave;
    t.vec_width = 4;
    return t;
}

// --- Public API ---

KernelTuneParams get_kernel_tune_for(const char* kernel_name, const GpuProfile& profile) {
    if (!kernel_name) return tune_generic(profile, "unknown");

    if (strstr(kernel_name, "fused_attention"))
        return tune_fused_attention(profile);
    if (strstr(kernel_name, "value_dequant"))
        return tune_value_dequant(profile);
    if (strstr(kernel_name, "mse_score"))
        return tune_mse_score(profile);
    if (strstr(kernel_name, "qjl_score"))
        return tune_qjl_score(profile);
    if (strstr(kernel_name, "attention_logits"))
        return tune_attention_logits(profile);
    if (strstr(kernel_name, "attention_apply_values"))
        return tune_attention_apply_values(profile);

    return tune_generic(profile, kernel_name);
}

KernelTuneParams get_kernel_tune(const char* kernel_name) {
    return get_kernel_tune_for(kernel_name, active_profile());
}

void get_local_work_size(const KernelTuneParams& t, size_t out[3]) {
    out[0] = t.wg_size_x;
    out[1] = t.wg_size_y;
    out[2] = t.wg_size_z;
}

} // namespace tq
