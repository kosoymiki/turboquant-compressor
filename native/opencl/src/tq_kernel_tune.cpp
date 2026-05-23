/**
 * TurboQuant Per-Kernel Tuning — implementation.
 *
 * Maps kernel names to optimal dispatch parameters per GPU generation.
 * Covers all TQ compute kernels: attention, dequant, MSE, QJL.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_kernel_tune.h"
#include "../include/tq_opencl.h"
#include "../include/tq_repo_paths.h"
#include <fstream>
#include <regex>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace tq {

namespace {

std::unordered_map<std::string, KernelTuneParams> g_overrides;
bool g_overrides_loaded = false;

uint32_t clamp_down_to_multiple(uint32_t value, uint32_t multiple) {
    if (multiple == 0 || value == 0) return value;
    uint32_t clamped = (value / multiple) * multiple;
    return clamped == 0 ? multiple : clamped;
}

KernelTuneParams apply_runtime_kernel_metadata(const char* kernel_name, const KernelTuneParams& base) {
    if (!kernel_name || !is_initialized()) return base;
    KernelMetadata meta = get_kernel_metadata(kernel_name);
    if (meta.work_group_size == 0 && meta.preferred_work_group_size_multiple == 0) return base;

    KernelTuneParams tuned = base;
    uint32_t preferred_multiple = static_cast<uint32_t>(meta.preferred_work_group_size_multiple);
    uint32_t max_kernel_wg = static_cast<uint32_t>(meta.work_group_size);

    if (preferred_multiple > 0) {
        tuned.preferred_multiple = preferred_multiple;
        tuned.wg_size_x = clamp_down_to_multiple(tuned.wg_size_x, preferred_multiple);
        if (tuned.wg_size_x == 0) tuned.wg_size_x = preferred_multiple;
    }
    if (max_kernel_wg > 0) {
        tuned.wg_size_x = std::min(tuned.wg_size_x, max_kernel_wg);
        if (preferred_multiple > 0) {
            tuned.wg_size_x = clamp_down_to_multiple(tuned.wg_size_x, preferred_multiple);
        }
        if (tuned.wg_size_x == 0) tuned.wg_size_x = std::max<uint32_t>(1u, std::min<uint32_t>(max_kernel_wg, preferred_multiple ? preferred_multiple : 1u));
    }
    if (meta.local_mem_size > 0 && tuned.local_mem_bytes > meta.local_mem_size) {
        tuned.local_mem_bytes = static_cast<uint32_t>(std::min<cl_ulong>(meta.local_mem_size, tuned.local_mem_bytes));
    }
    return tuned;
}

std::string load_cache_path() {
    std::string forensics_dir = resolve_forensics_dir();
    return forensics_dir.empty() ? std::string("forensics/autotune-cache.json")
                                 : join_repo_path(forensics_dir, "autotune-cache.json");
}

void ensure_overrides_loaded() {
    if (g_overrides_loaded) return;
    g_overrides_loaded = true;

    std::ifstream in(load_cache_path());
    if (!in.is_open()) return;

    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::regex entry_re("\"([^\"]+)\"\\s*:\\s*\\{([^\\}]*)\\}");
    std::regex uint_field_re("\"([^\"]+)\"\\s*:\\s*([0-9]+)");
    auto begin = std::sregex_iterator(json.begin(), json.end(), entry_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        const std::string kernel_name = (*it)[1].str();
        const std::string body = (*it)[2].str();
        KernelTuneParams t = {};
        t.kernel_name = kernel_name;
        auto field_begin = std::sregex_iterator(body.begin(), body.end(), uint_field_re);
        auto field_end = std::sregex_iterator();
        for (auto field_it = field_begin; field_it != field_end; ++field_it) {
            const std::string key = (*field_it)[1].str();
            const uint32_t value = static_cast<uint32_t>(std::stoul((*field_it)[2].str()));
            if (key == "wg_size") t.wg_size_x = value;
            else if (key == "items_per_thread") t.items_per_thread = static_cast<int>(value);
            else if (key == "preferred_multiple") t.preferred_multiple = value;
            else if (key == "local_mem_bytes") t.local_mem_bytes = value;
        }
        if (!t.kernel_name.empty() && t.wg_size_x > 0) {
            g_overrides[t.kernel_name] = t;
        }
    }
}

} // namespace

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
    t.items_per_thread = (p.gen == GpuGen::ADRENO_8XX) ? 4 : (p.gen == GpuGen::ADRENO_7XX ? 2 : 1);
    t.vec_width = 4;
    t.tile_k = (p.gmem_size_kb >= 1024) ? 16 : 8;
    t.input_layout = MemoryLayout::LINEAR;
    t.output_layout = MemoryLayout::LINEAR;
    t.use_gmem_tiling = (p.tile_strategy != TileStrategy::SYSMEM && p.gmem_size_kb >= 768);
    t.prefetch_global = (p.gen >= GpuGen::ADRENO_7XX);
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
    ensure_overrides_loaded();
    auto override_it = g_overrides.find(kernel_name);
    if (override_it != g_overrides.end()) {
        KernelTuneParams tuned = override_it->second;
        tuned.kernel_name = kernel_name;
        tuned.preferred_multiple = static_cast<uint32_t>(profile.compute_wave);
        return apply_runtime_kernel_metadata(kernel_name, tuned);
    }

    if (strstr(kernel_name, "fused_attention"))
        return apply_runtime_kernel_metadata(kernel_name, tune_fused_attention(profile));
    if (strstr(kernel_name, "value_dequant"))
        return apply_runtime_kernel_metadata(kernel_name, tune_value_dequant(profile));
    if (strstr(kernel_name, "mse_score"))
        return apply_runtime_kernel_metadata(kernel_name, tune_mse_score(profile));
    if (strstr(kernel_name, "qjl_score"))
        return apply_runtime_kernel_metadata(kernel_name, tune_qjl_score(profile));
    if (strstr(kernel_name, "attention_logits"))
        return apply_runtime_kernel_metadata(kernel_name, tune_attention_logits(profile));
    if (strstr(kernel_name, "attention_apply_values"))
        return apply_runtime_kernel_metadata(kernel_name, tune_attention_apply_values(profile));

    return apply_runtime_kernel_metadata(kernel_name, tune_generic(profile, kernel_name));
}

KernelTuneParams get_kernel_tune(const char* kernel_name) {
    return get_kernel_tune_for(kernel_name, active_profile());
}

void get_local_work_size(const KernelTuneParams& t, size_t out[3]) {
    out[0] = t.wg_size_x;
    out[1] = t.wg_size_y;
    out[2] = t.wg_size_z;
}

void set_kernel_tune_override(const char* kernel_name, const KernelTuneParams& params) {
    if (!kernel_name) return;
    g_overrides_loaded = true;
    g_overrides[kernel_name] = params;
}

void clear_kernel_tune_overrides() {
    g_overrides.clear();
    g_overrides_loaded = true;
}

std::string autotune_cache_path() {
    return load_cache_path();
}

} // namespace tq
