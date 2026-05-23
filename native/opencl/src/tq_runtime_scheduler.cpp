/**
 * TurboQuant OpenCL — runtime scheduler/allocator state.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_runtime_scheduler.h"
#include "../include/tq_kernel_cache.h"
#include "../include/tq_kernel_tune.h"
#include "../include/tq_repo_paths.h"

#include <CL/cl.h>
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tq {

namespace {

struct RuntimeSchedulerState {
    std::mutex mu;
    std::unordered_set<std::string> compile_keys_seen;
    std::unordered_map<std::string, uint32_t> kernel_ready_hits;
    std::unordered_map<std::string, KernelMetadata> kernel_metadata;
    std::vector<RuntimeCompileCandidate> compile_candidates;
    uint32_t execution_dispatch_count = 0;
    uint32_t execution_profiled_dispatch_count = 0;
    uint64_t last_dispatch_kernel_ns = 0;
    size_t last_dispatch_global_size = 0;
    size_t last_dispatch_local_size = 0;
    std::string last_dispatch_kernel_name;
    uint32_t compile_intents = 0;
    uint32_t deduped_compile_intents = 0;
    uint32_t program_reuse_hits = 0;
    uint32_t binary_cache_hits = 0;
    uint32_t precompiled_binary_hits = 0;
    uint32_t source_builds = 0;
    uint32_t il_builds = 0;
    uint32_t binary_builds = 0;
    uint32_t async_build_callbacks = 0;
    uint32_t kernel_ready_count = 0;
};

RuntimeSchedulerState g_state;

uint64_t directory_bytes(const std::string& path) {
    std::error_code ec;
    if (path.empty() || !std::filesystem::exists(path, ec)) return 0;
    uint64_t total = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path, ec)) {
        if (ec) break;
        if (entry.is_regular_file(ec)) {
            total += static_cast<uint64_t>(entry.file_size(ec));
        }
    }
    return total;
}

std::vector<RuntimeResidencyCandidate> top_file_candidates(
    const std::string& path, const char* tier, bool persistent, uint32_t priority, size_t limit) {
    struct FileHit {
        std::string key;
        uint64_t bytes = 0;
    };

    std::error_code ec;
    std::vector<FileHit> hits;
    if (path.empty() || !std::filesystem::exists(path, ec)) return {};
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const uint64_t bytes = static_cast<uint64_t>(entry.file_size(ec));
        if (bytes == 0) continue;
        hits.push_back({entry.path().filename().string(), bytes});
    }
    std::sort(hits.begin(), hits.end(), [](const FileHit& a, const FileHit& b) {
        if (a.bytes != b.bytes) return a.bytes > b.bytes;
        return a.key < b.key;
    });
    if (hits.size() > limit) hits.resize(limit);

    std::vector<RuntimeResidencyCandidate> out;
    out.reserve(hits.size());
    for (const auto& hit : hits) {
        RuntimeResidencyCandidate candidate;
        candidate.tier = tier ? tier : "unknown";
        candidate.key = hit.key;
        candidate.bytes = hit.bytes;
        candidate.persistent = persistent;
        candidate.priority = priority;
        out.push_back(std::move(candidate));
    }
    return out;
}

std::vector<RuntimeResidencyCandidate> top_autotune_candidates(const std::string& path, size_t limit) {
    std::ifstream in(path);
    if (!in.is_open()) return {};
    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const uint64_t total_bytes = static_cast<uint64_t>(json.size());
    if (total_bytes == 0) return {};

    uint32_t entry_count = 0;
    bool in_string = false;
    for (size_t i = 0; i < json.size(); ++i) {
        if (json[i] == '"' && (i == 0 || json[i - 1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string && json[i] == '{') {
            ++entry_count;
        }
    }
    if (entry_count > 0) {
        --entry_count;
    }
    if (entry_count == 0) return {};

    std::vector<RuntimeResidencyCandidate> out;
    const size_t count = std::min<size_t>(limit, entry_count);
    const uint64_t per_entry_bytes = total_bytes / static_cast<uint64_t>(entry_count);
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        RuntimeResidencyCandidate candidate;
        candidate.tier = "autotune_cache";
        candidate.key = "autotune-entry-" + std::to_string(i + 1);
        candidate.bytes = per_entry_bytes;
        candidate.persistent = true;
        candidate.priority = 2;
        out.push_back(std::move(candidate));
    }
    return out;
}

uint32_t count_autotune_entries() {
    std::ifstream in(autotune_cache_path());
    if (!in.is_open()) return 0;
    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    uint32_t count = 0;
    bool in_string = false;
    for (size_t i = 0; i < json.size(); ++i) {
        if (json[i] == '"' && (i == 0 || json[i - 1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string && json[i] == '{') {
            ++count;
        }
    }
    return count > 0 ? count - 1 : 0;
}

uint64_t query_global_mem_bytes() {
    if (!is_initialized() || !get_device()) return 0;
    cl_ulong bytes = 0;
    if (clGetDeviceInfo(get_device(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(bytes), &bytes, nullptr) != CL_SUCCESS) {
        return 0;
    }
    return static_cast<uint64_t>(bytes);
}

std::string artifact_path() {
    const std::string forensics_dir = resolve_forensics_dir();
    return forensics_dir.empty()
        ? std::string("forensics/runtime-scheduler-state.json")
        : join_repo_path(forensics_dir, "runtime-scheduler-state.json");
}

uint64_t recommended_budget_bytes(uint64_t global_mem_bytes) {
    if (global_mem_bytes == 0) return 0;
    const uint64_t quarter = global_mem_bytes / 4;
    const uint64_t floor = 128ull * 1024ull * 1024ull;
    return std::max(floor, quarter);
}

std::string resident_pressure_band(uint32_t permille) {
    if (permille >= 900) return "critical";
    if (permille >= 700) return "high";
    if (permille >= 400) return "moderate";
    return "low";
}

std::string queue_pressure_band(uint32_t permille) {
    if (permille >= 900) return "critical";
    if (permille >= 700) return "high";
    if (permille >= 400) return "moderate";
    return "low";
}

std::string queue_age_band(uint32_t ticks) {
    if (ticks >= 12) return "stale";
    if (ticks >= 6) return "warm";
    if (ticks >= 2) return "active";
    return "fresh";
}

std::string classify_latency_class(
    const std::string& kernel_name,
    const KernelMetadata& meta,
    uint64_t kernel_ns,
    std::string* out_reason) {
    if (kernel_ns > 4'000'000ull || meta.local_mem_size >= 8192 || meta.private_mem_size >= 256) {
        if (out_reason) *out_reason = "runtime_latency_or_memory_pressure";
        return "throughput";
    }
    if (kernel_name.find("value_dequant") != std::string::npos ||
        kernel_ns <= 1'500'000ull ||
        meta.suggested_local_work_size <= 128) {
        if (out_reason) *out_reason = "bounded_kernel_fast_path";
        return "latency_sensitive";
    }
    if (out_reason) *out_reason = "balanced_kernel_profile";
    return "balanced";
}

std::string kernel_class_for_name(const std::string& kernel_name) {
    if (kernel_name.find("value_dequant") != std::string::npos) return "value_dequant";
    if (kernel_name.find("mse_score") != std::string::npos) return "mse_score";
    if (kernel_name.find("qjl_score") != std::string::npos) return "qjl_score";
    if (kernel_name.find("attention_logits") != std::string::npos) return "attention_logits";
    if (kernel_name.find("attention_apply_values") != std::string::npos) return "attention_apply_values";
    return "generic_kernel";
}

std::string urgency_for_latency_class(const std::string& latency_class) {
    if (latency_class == "latency_sensitive") return "interactive";
    if (latency_class == "throughput") return "background";
    return "balanced";
}

uint32_t base_budget_permille_for_class(const std::string& kernel_class, const std::string& latency_class) {
    if (kernel_class == "value_dequant" || latency_class == "latency_sensitive") return 350;
    if (kernel_class == "mse_score") return 160;
    if (kernel_class == "qjl_score") return 170;
    if (kernel_class == "attention_logits") return 150;
    if (kernel_class == "attention_apply_values") return 120;
    if (latency_class == "throughput") return 110;
    return 130;
}

uint32_t queue_age_boost_permille(uint32_t ticks, const std::string& urgency) {
    uint32_t boost = 0;
    if (ticks >= 12) boost = 120;
    else if (ticks >= 6) boost = 70;
    else if (ticks >= 2) boost = 30;
    if (urgency == "interactive") boost /= 2;
    return boost;
}

uint32_t starvation_score(
    uint32_t queue_age_ticks,
    const std::string& urgency,
    const std::string& admission,
    bool live_dispatch_anchor) {
    uint32_t score = queue_age_ticks * 40u;
    if (urgency == "background") score += 80;
    else if (urgency == "balanced") score += 40;
    if (admission.find("defer") != std::string::npos) score += 120;
    if (!live_dispatch_anchor) score += 30;
    return std::min<uint32_t>(1000u, score);
}

std::string pressure_action_for_policy(
    const RuntimeKernelClassPolicy& policy,
    uint32_t queue_pressure_permille,
    bool queue_guard_triggered) {
    if (policy.live_dispatch_anchor && policy.latency_class == "latency_sensitive") {
        return queue_guard_triggered ? "protect_live_anchor" : "prefer_live_anchor";
    }
    if (queue_pressure_permille >= 700 && policy.urgency == "background") return "throttle_background";
    if (queue_pressure_permille >= 400 && policy.starvation_boosted) return "age_boost_recovery";
    return "steady_dispatch";
}

std::string pressure_reason_for_policy(
    const RuntimeKernelClassPolicy& policy,
    uint32_t queue_pressure_permille,
    bool queue_guard_triggered) {
    if (policy.live_dispatch_anchor && policy.latency_class == "latency_sensitive") {
        return queue_guard_triggered ? "queue_guard_preserves_interactive_lane" : "interactive_lane_anchor";
    }
    if (queue_pressure_permille >= 700 && policy.urgency == "background") return "queue_pressure_demotes_background_lane";
    if (queue_pressure_permille >= 400 && policy.starvation_boosted) return "aging_boost_offsets_queue_pressure";
    return "pressure_stable";
}

uint32_t resident_pressure_permille(uint64_t resident_bytes, uint64_t budget_bytes) {
    if (budget_bytes == 0) return 0;
    if (resident_bytes >= budget_bytes) return 1000;
    return static_cast<uint32_t>((resident_bytes * 1000ull) / budget_bytes);
}

uint64_t budget_slice(uint64_t budget_bytes, uint32_t permille) {
    return (budget_bytes * static_cast<uint64_t>(permille)) / 1000ull;
}

std::string preferred_build_path(
    uint32_t precompiled_binary_hits,
    uint32_t binary_cache_hits,
    uint32_t il_builds,
    uint32_t source_builds,
    uint32_t binary_builds) {
    if (precompiled_binary_hits > 0) return "precompiled_binary";
    if (binary_cache_hits > 0 || binary_builds > 0) return "binary_cache";
    if (il_builds > 0) return "spirv_il";
    if (source_builds > 0) return "source_build";
    return "unresolved";
}

uint32_t slot_utilization_permille(uint32_t used_slots, uint32_t total_slots) {
    if (total_slots == 0) return 0;
    return std::min<uint32_t>(
        1000u,
        static_cast<uint32_t>((static_cast<uint64_t>(used_slots) * 1000ull) / static_cast<uint64_t>(total_slots)));
}

} // namespace

void runtime_scheduler_reset() {
    std::lock_guard<std::mutex> lock(g_state.mu);
    g_state.compile_keys_seen.clear();
    g_state.kernel_ready_hits.clear();
    g_state.kernel_metadata.clear();
    g_state.compile_candidates.clear();
    g_state.execution_dispatch_count = 0;
    g_state.execution_profiled_dispatch_count = 0;
    g_state.last_dispatch_kernel_ns = 0;
    g_state.last_dispatch_global_size = 0;
    g_state.last_dispatch_local_size = 0;
    g_state.last_dispatch_kernel_name.clear();
    g_state.compile_intents = 0;
    g_state.deduped_compile_intents = 0;
    g_state.program_reuse_hits = 0;
    g_state.binary_cache_hits = 0;
    g_state.precompiled_binary_hits = 0;
    g_state.source_builds = 0;
    g_state.il_builds = 0;
    g_state.binary_builds = 0;
    g_state.async_build_callbacks = 0;
    g_state.kernel_ready_count = 0;
}

void runtime_scheduler_note_compile_intent(const std::string&, const std::string& compile_key) {
    std::lock_guard<std::mutex> lock(g_state.mu);
    ++g_state.compile_intents;
    const bool inserted = g_state.compile_keys_seen.insert(compile_key).second;
    if (!inserted) {
        ++g_state.deduped_compile_intents;
    }
}

void runtime_scheduler_note_program_reuse(const std::string&, const std::string&) {
    std::lock_guard<std::mutex> lock(g_state.mu);
    ++g_state.program_reuse_hits;
}

void runtime_scheduler_note_cache_hit(const char* kind, const std::string&, const std::string&, size_t) {
    std::lock_guard<std::mutex> lock(g_state.mu);
    if (kind && std::string(kind) == "precompiled") {
        ++g_state.precompiled_binary_hits;
    } else {
        ++g_state.binary_cache_hits;
    }
}

void runtime_scheduler_note_compile_candidate(
    const std::string& kernel_name,
    const std::string& compile_key,
    bool has_precompiled,
    bool has_binary_cache,
    bool has_spirv_il,
    bool has_source,
    bool source_only_forced) {
    RuntimeCompileCandidate candidate;
    candidate.kernel_name = kernel_name;
    candidate.compile_key = compile_key;
    candidate.has_precompiled = has_precompiled;
    candidate.has_binary_cache = has_binary_cache;
    candidate.has_spirv_il = has_spirv_il;
    candidate.has_source = has_source;
    candidate.speculative = has_precompiled || has_binary_cache || has_spirv_il;

    if (source_only_forced) {
        candidate.preferred_path = "source_build";
        candidate.admission = "admit_source_only";
        candidate.reason = "source_only_forced";
        candidate.admitted = has_source;
    } else if (has_precompiled) {
        candidate.preferred_path = "precompiled_binary";
        candidate.admission = "admit_precompiled";
        candidate.reason = "persistent_binary_ready";
        candidate.admitted = true;
    } else if (has_binary_cache) {
        candidate.preferred_path = "binary_cache";
        candidate.admission = "admit_binary_cache";
        candidate.reason = "runtime_binary_cache_ready";
        candidate.admitted = true;
    } else if (has_spirv_il) {
        candidate.preferred_path = "spirv_il";
        candidate.admission = "admit_spirv_il";
        candidate.reason = "device_il_program_available";
        candidate.admitted = true;
    } else if (has_source) {
        candidate.preferred_path = "source_build";
        candidate.admission = "admit_source_fallback";
        candidate.reason = "fallback_required";
        candidate.admitted = true;
    } else {
        candidate.preferred_path = "unresolved";
        candidate.admission = "defer_missing_artifact";
        candidate.reason = "no_viable_input";
        candidate.admitted = false;
    }

    std::lock_guard<std::mutex> lock(g_state.mu);
    candidate.queue_ordinal = static_cast<uint32_t>(g_state.compile_candidates.size() + 1);
    g_state.compile_candidates.push_back(std::move(candidate));
}

void runtime_scheduler_note_build_stage(const char* stage, bool ok, bool async_callback_observed) {
    std::lock_guard<std::mutex> lock(g_state.mu);
    if (!ok) return;
    const std::string stage_name = stage ? stage : "";
    if (stage_name == "source") ++g_state.source_builds;
    else if (stage_name == "IL") ++g_state.il_builds;
    else if (stage_name == "binary") ++g_state.binary_builds;
    if (async_callback_observed) ++g_state.async_build_callbacks;
}

void runtime_scheduler_note_kernel_ready(const std::string& kernel_name, const KernelMetadata& meta) {
    std::lock_guard<std::mutex> lock(g_state.mu);
    ++g_state.kernel_ready_count;
    ++g_state.kernel_ready_hits[kernel_name];
    g_state.kernel_metadata[kernel_name] = meta;
}

void runtime_scheduler_note_execution_dispatch(
    const std::string& kernel_name,
    size_t global_size,
    size_t local_size,
    bool profiled,
    uint64_t kernel_ns) {
    std::lock_guard<std::mutex> lock(g_state.mu);
    ++g_state.execution_dispatch_count;
    if (profiled) {
        ++g_state.execution_profiled_dispatch_count;
    }
    g_state.last_dispatch_kernel_ns = kernel_ns;
    g_state.last_dispatch_global_size = global_size;
    g_state.last_dispatch_local_size = local_size;
    g_state.last_dispatch_kernel_name = kernel_name;
}

RuntimeSchedulerSummary runtime_scheduler_get_summary() {
    RuntimeSchedulerSummary summary;
    summary.available = true;
    {
        std::lock_guard<std::mutex> lock(g_state.mu);
        summary.compile_intents = g_state.compile_intents;
        summary.deduped_compile_intents = g_state.deduped_compile_intents;
        summary.program_reuse_hits = g_state.program_reuse_hits;
        summary.binary_cache_hits = g_state.binary_cache_hits;
        summary.precompiled_binary_hits = g_state.precompiled_binary_hits;
        summary.source_builds = g_state.source_builds;
        summary.il_builds = g_state.il_builds;
        summary.binary_builds = g_state.binary_builds;
        summary.async_build_callbacks = g_state.async_build_callbacks;
        summary.kernel_ready_count = g_state.kernel_ready_count;
        summary.compile_candidates = g_state.compile_candidates;
        summary.execution_dispatch_count = g_state.execution_dispatch_count;
        summary.execution_profiled_dispatch_count = g_state.execution_profiled_dispatch_count;
        summary.last_dispatch_kernel_ns = g_state.last_dispatch_kernel_ns;
        summary.last_dispatch_global_size = g_state.last_dispatch_global_size;
        summary.last_dispatch_local_size = g_state.last_dispatch_local_size;
        summary.last_dispatch_kernel_name = g_state.last_dispatch_kernel_name;
        const auto meta_it = g_state.kernel_metadata.find(summary.last_dispatch_kernel_name);
        const KernelMetadata dispatch_meta = meta_it == g_state.kernel_metadata.end() ? KernelMetadata{} : meta_it->second;
        summary.last_dispatch_latency_class = classify_latency_class(
            summary.last_dispatch_kernel_name,
            dispatch_meta,
            summary.last_dispatch_kernel_ns,
            &summary.last_dispatch_latency_reason);
    }

    summary.global_mem_bytes = query_global_mem_bytes();
    summary.recommended_budget_bytes = recommended_budget_bytes(summary.global_mem_bytes);
    KernelBinaryCache local_cache;
    summary.kernel_cache_bytes = directory_bytes(local_cache.cache_dir());

    uint64_t precompiled_total = 0;
    for (const auto& root : runtime_pack_roots()) {
        precompiled_total += directory_bytes(join_repo_path(root, "layer1-compute/precompiled-kernels"));
    }
    const std::string forensics_dir = resolve_forensics_dir();
    if (!forensics_dir.empty()) {
        precompiled_total += directory_bytes(join_repo_path(forensics_dir, "precompiled-kernel-library"));
    }
    summary.precompiled_library_bytes = precompiled_total;
    summary.resident_bytes = summary.kernel_cache_bytes + summary.precompiled_library_bytes;
    summary.resident_pressure_permille = resident_pressure_permille(
        summary.resident_bytes, summary.recommended_budget_bytes);
    summary.resident_pressure_band = resident_pressure_band(summary.resident_pressure_permille);
    summary.eviction_recommended =
        summary.recommended_budget_bytes > 0 && summary.resident_bytes > summary.recommended_budget_bytes;
    summary.autotune_entries = count_autotune_entries();
    summary.preferred_build_path = preferred_build_path(
        summary.precompiled_binary_hits,
        summary.binary_cache_hits,
        summary.il_builds,
        summary.source_builds,
        summary.binary_builds);
    summary.compile_candidate_count = static_cast<uint32_t>(summary.compile_candidates.size());
    summary.compile_queue_depth = summary.compile_candidate_count;
    for (const auto& candidate : summary.compile_candidates) {
        if (candidate.admitted) {
            ++summary.admitted_compile_count;
        } else {
            ++summary.deferred_compile_count;
        }
        if (candidate.speculative) {
            ++summary.speculative_candidate_count;
        }
    }
    std::unordered_map<std::string, uint32_t> kernel_ordinals;
    for (const auto& candidate : summary.compile_candidates) {
        const auto [it, inserted] = kernel_ordinals.emplace(candidate.kernel_name, candidate.queue_ordinal);
        if (!inserted) {
            it->second = std::min<uint32_t>(it->second, candidate.queue_ordinal);
        }
    }
    const uint32_t queue_multiplier = get_gpu_profile().dual_wave_dispatch ? 2u : 1u;
    summary.execution_queue_slots = std::max<uint32_t>(1u, get_compute_units() * queue_multiplier);
    summary.dispatch_budget_permille = 700;
    summary.compile_budget_permille = 1000 - summary.dispatch_budget_permille;
    summary.execution_budget_bytes = budget_slice(summary.recommended_budget_bytes, summary.dispatch_budget_permille);
    const uint32_t execution_pressure_permille =
        resident_pressure_permille(summary.resident_bytes, summary.execution_budget_bytes);
    summary.dispatch_wave_slots =
        summary.last_dispatch_local_size == 0
            ? 0
            : static_cast<uint32_t>(
                  (summary.last_dispatch_global_size + summary.last_dispatch_local_size - 1) /
                  summary.last_dispatch_local_size);
    summary.active_dispatch_slots = summary.execution_dispatch_count == 0
        ? 0
        : std::min<uint32_t>(
              std::max<uint32_t>(1u, summary.dispatch_wave_slots),
              summary.execution_queue_slots);
    summary.queue_pressure_permille = slot_utilization_permille(
        summary.active_dispatch_slots, summary.execution_queue_slots);
    summary.execution_budget_slots = std::max<uint32_t>(
        1u,
        static_cast<uint32_t>(
            (static_cast<uint64_t>(summary.execution_queue_slots) * summary.dispatch_budget_permille) / 1000ull));
    summary.compile_budget_slots = summary.execution_queue_slots > summary.execution_budget_slots
        ? summary.execution_queue_slots - summary.execution_budget_slots
        : 1u;
    summary.dispatch_utilization_permille = slot_utilization_permille(
        summary.active_dispatch_slots, summary.execution_budget_slots);
    summary.compile_utilization_permille = slot_utilization_permille(
        summary.compile_queue_depth, summary.compile_budget_slots);
    summary.compile_execute_balance_permille =
        std::min<uint32_t>(
            1000u,
            static_cast<uint32_t>(
                (static_cast<uint64_t>(summary.dispatch_utilization_permille) * 1000ull) /
                std::max<uint32_t>(1u, summary.dispatch_utilization_permille + summary.compile_utilization_permille)));
    summary.execution_pressure_band = resident_pressure_band(execution_pressure_permille);
    summary.queue_pressure_band = queue_pressure_band(summary.queue_pressure_permille);
    summary.compile_queue_age_ticks =
        summary.compile_queue_depth * (summary.execution_dispatch_count > 0 ? 2u : 1u);
    summary.dispatch_queue_age_ticks =
        summary.execution_dispatch_count == 0
            ? 0
            : std::max<uint32_t>(
                  1u,
                  summary.dispatch_wave_slots +
                      (summary.last_dispatch_kernel_ns >= 1'500'000ull ? 1u : 0u));
    summary.compile_queue_age_band = queue_age_band(summary.compile_queue_age_ticks);
    summary.dispatch_queue_age_band = queue_age_band(summary.dispatch_queue_age_ticks);
    summary.dispatch_urgency = urgency_for_latency_class(summary.last_dispatch_latency_class);
    summary.queue_guard_triggered =
        summary.queue_pressure_permille >= 850 || summary.dispatch_utilization_permille >= 850;
    if (!summary.queue_guard_triggered) {
        summary.queue_guard_reason = "guard_not_triggered";
    } else if (summary.queue_pressure_permille >= 850) {
        summary.queue_guard_reason = "queue_pressure_critical";
    } else if (summary.dispatch_utilization_permille >= 850) {
        summary.queue_guard_reason = "dispatch_budget_saturated";
    } else {
        summary.queue_guard_reason = "dispatch_budget_guarded";
    }
    summary.compile_deferred_for_execution =
        ((summary.execution_dispatch_count > 0 && execution_pressure_permille >= 700) ||
         summary.queue_pressure_permille >= 700 ||
         summary.dispatch_utilization_permille >= 700)
            ? summary.compile_queue_depth
            : 0;
    summary.multi_queue_mode = summary.execution_queue_slots > 1 ? "multi_queue_budgeted" : "single_queue_budgeted";
    if (summary.execution_dispatch_count > 0 && summary.queue_guard_triggered) {
        summary.execution_admission = "dispatch_admitted_queue_guarded";
        summary.execution_admission_reason = "queue_guard_threshold_crossed";
    } else if (summary.execution_dispatch_count > 0) {
        summary.execution_admission = "dispatch_admitted";
        summary.execution_admission_reason = "dispatch_budget_available";
    } else if (summary.compile_queue_depth > 0) {
        summary.execution_admission = "ready_no_dispatch";
        summary.execution_admission_reason = "compile_side_ready_without_live_dispatch";
    } else {
        summary.execution_admission = "idle";
        summary.execution_admission_reason = "no_live_work";
    }
    summary.arbitration_mode =
        summary.compile_deferred_for_execution > 0 ? "execute_priority" :
        (summary.execution_dispatch_count > 0 && summary.compile_queue_depth > 0 ? "balanced_compile_execute" :
        (summary.compile_queue_depth > 0 ? "compile_priority" : "idle"));
    if (summary.compile_deferred_for_execution > 0) {
        summary.arbitration_reason =
            summary.queue_pressure_permille >= 700 ? "queue_pressure_high" : "execution_budget_pressure";
    } else if (summary.execution_dispatch_count > 0 && summary.compile_queue_depth > 0) {
        summary.arbitration_reason = "dispatch_and_compile_capacity_available";
    } else if (summary.compile_queue_depth > 0) {
        summary.arbitration_reason = "compile_backlog_without_live_dispatch";
    } else {
        summary.arbitration_reason = "idle";
    }
    if (summary.execution_dispatch_count == 0) {
        summary.latency_admission = "idle";
        summary.latency_admission_reason = "no_live_dispatch";
    } else if (summary.last_dispatch_latency_class == "latency_sensitive" &&
               summary.dispatch_queue_age_ticks <= 2 &&
               summary.queue_pressure_permille < 700) {
        summary.latency_admission = "admit_latency_sensitive";
        summary.latency_admission_reason = "fresh_low_pressure_dispatch";
    } else if (summary.last_dispatch_latency_class == "throughput" &&
               summary.compile_queue_age_ticks >= summary.dispatch_queue_age_ticks) {
        summary.latency_admission = "admit_throughput_guarded";
        summary.latency_admission_reason = "compile_queue_aging_dominates";
    } else {
        summary.latency_admission = "admit_balanced";
        summary.latency_admission_reason = "balanced_latency_queue_state";
    }
    summary.kernel_cache_budget_bytes = budget_slice(summary.recommended_budget_bytes, 550);
    summary.precompiled_budget_bytes = budget_slice(summary.recommended_budget_bytes, 400);
    summary.autotune_budget_bytes = budget_slice(summary.recommended_budget_bytes, 50);
    summary.persistent_residency_ready =
        summary.precompiled_binary_hits > 0 && summary.precompiled_library_bytes > 0 && summary.autotune_entries > 0;
    summary.compaction_recommended =
        summary.resident_pressure_permille >= 700 ||
        (summary.kernel_cache_bytes > summary.kernel_cache_budget_bytes && summary.kernel_cache_budget_bytes > 0);

    std::vector<RuntimeResidencyCandidate> candidates;
    const std::string kernel_cache_dir = local_cache.cache_dir();
    auto kernel_cache_candidates = top_file_candidates(kernel_cache_dir, "kernel_cache", false, 1, 3);
    auto precompiled_candidates = top_file_candidates(
        forensics_dir.empty() ? std::string() : join_repo_path(forensics_dir, "precompiled-kernel-library"),
        "precompiled_library",
        true,
        3,
        2);
    auto autotune_candidates = top_autotune_candidates(autotune_cache_path(), 2);
    candidates.insert(candidates.end(), kernel_cache_candidates.begin(), kernel_cache_candidates.end());
    candidates.insert(candidates.end(), autotune_candidates.begin(), autotune_candidates.end());
    candidates.insert(candidates.end(), precompiled_candidates.begin(), precompiled_candidates.end());
    std::sort(candidates.begin(), candidates.end(), [](const RuntimeResidencyCandidate& a, const RuntimeResidencyCandidate& b) {
        if (a.priority != b.priority) return a.priority < b.priority;
        if (a.bytes != b.bytes) return a.bytes > b.bytes;
        return a.key < b.key;
    });
    summary.eviction_candidate_count = static_cast<uint32_t>(candidates.size());
    summary.eviction_candidates = std::move(candidates);
    {
        std::lock_guard<std::mutex> lock(g_state.mu);
        for (const auto& entry : g_state.kernel_metadata) {
            RuntimeKernelClassPolicy policy;
            policy.kernel_name = entry.first;
            policy.kernel_class = kernel_class_for_name(entry.first);
            policy.queue_ordinal = kernel_ordinals.count(entry.first) ? kernel_ordinals[entry.first] : 0;
            const bool live_anchor = entry.first == summary.last_dispatch_kernel_name && summary.execution_dispatch_count > 0;
            policy.live_dispatch_anchor = live_anchor;
            uint64_t kernel_ns = live_anchor ? summary.last_dispatch_kernel_ns : 0;
            policy.latency_class = classify_latency_class(
                entry.first, entry.second, kernel_ns, &policy.latency_reason);
            policy.urgency = urgency_for_latency_class(policy.latency_class);
            policy.queue_age_ticks =
                policy.queue_ordinal == 0
                    ? 1u
                    : policy.queue_ordinal + (policy.latency_class == "throughput" ? 2u : 1u);
            if (live_anchor) {
                policy.queue_age_ticks = summary.dispatch_queue_age_ticks;
            } else if (summary.compile_queue_depth > 0) {
                policy.queue_age_ticks += summary.compile_queue_depth;
            }
            policy.queue_age_band = queue_age_band(policy.queue_age_ticks);
            if (live_anchor && policy.latency_class == "latency_sensitive") {
                policy.admission = "dispatch_anchor_admitted";
                policy.admission_reason = "live_latency_sensitive_dispatch";
            } else if (policy.latency_class == "throughput" && summary.compile_queue_age_ticks >= 6) {
                policy.admission = "defer_throughput_bias";
                policy.admission_reason = "compile_queue_aging_guard";
            } else if (policy.latency_class == "balanced") {
                policy.admission = "admit_balanced_class";
                policy.admission_reason = "balanced_class_capacity";
            } else {
                policy.admission = "admit_kernel_class";
                policy.admission_reason = "kernel_class_ready";
            }
            policy.base_budget_permille = base_budget_permille_for_class(policy.kernel_class, policy.latency_class);
            policy.age_boost_permille = queue_age_boost_permille(policy.queue_age_ticks, policy.urgency);
            policy.effective_budget_permille = std::min<uint32_t>(1000u, policy.base_budget_permille + policy.age_boost_permille);
            policy.starvation_score = starvation_score(
                policy.queue_age_ticks, policy.urgency, policy.admission, policy.live_dispatch_anchor);
            policy.dispatch_lane = policy.kernel_class;
            summary.kernel_class_policies.push_back(std::move(policy));
        }
    }
    std::sort(
        summary.kernel_class_policies.begin(),
        summary.kernel_class_policies.end(),
        [](const RuntimeKernelClassPolicy& a, const RuntimeKernelClassPolicy& b) {
            if (a.live_dispatch_anchor != b.live_dispatch_anchor) return a.live_dispatch_anchor > b.live_dispatch_anchor;
            if (a.starvation_score != b.starvation_score) return a.starvation_score > b.starvation_score;
            if (a.queue_ordinal != b.queue_ordinal) return a.queue_ordinal < b.queue_ordinal;
            return a.kernel_name < b.kernel_name;
        });
    summary.kernel_class_policy_count = static_cast<uint32_t>(summary.kernel_class_policies.size());
    RuntimeKernelClassPolicy* starvation_target = nullptr;
    RuntimeKernelClassPolicy* throttle_target = nullptr;
    uint32_t fairness_debt = 0;
    for (auto& policy : summary.kernel_class_policies) {
        const bool is_starvation_candidate =
            !policy.live_dispatch_anchor &&
            policy.starvation_score >= 360 &&
            policy.queue_age_ticks >= summary.dispatch_queue_age_ticks;
        policy.starvation_boosted = is_starvation_candidate;
        if (policy.starvation_boosted) {
            policy.effective_budget_permille = std::min<uint32_t>(1000u, policy.effective_budget_permille + 90u);
            policy.redistribution_action = "boost_starved_lane";
            policy.redistribution_reason = "queue_aging_exceeded_live_lane";
            ++summary.starvation_prevented_count;
            if (!starvation_target || policy.starvation_score > starvation_target->starvation_score) {
                starvation_target = &policy;
            }
        } else if (policy.live_dispatch_anchor) {
            policy.redistribution_action = "protect_anchor_lane";
            policy.redistribution_reason = "live_dispatch_latency_anchor";
        } else if (policy.urgency == "background" && summary.queue_pressure_permille >= 400) {
            policy.redistribution_action = "throttle_background_lane";
            policy.redistribution_reason = "queue_pressure_requires_budget_return";
        } else {
            policy.redistribution_action = "hold_class_budget";
            policy.redistribution_reason = "class_budget_stable";
        }

        policy.throttle_recommended =
            !policy.live_dispatch_anchor &&
            policy.urgency == "background" &&
            (summary.queue_guard_triggered || summary.queue_pressure_permille >= 400);
        if (policy.throttle_recommended) {
            policy.effective_budget_permille =
                policy.effective_budget_permille > 60u ? policy.effective_budget_permille - 60u : 40u;
            if (!throttle_target || policy.queue_age_ticks > throttle_target->queue_age_ticks) {
                throttle_target = &policy;
            }
        }
        policy.pressure_action =
            pressure_action_for_policy(policy, summary.queue_pressure_permille, summary.queue_guard_triggered);
        policy.pressure_reason =
            pressure_reason_for_policy(policy, summary.queue_pressure_permille, summary.queue_guard_triggered);
        fairness_debt = std::max<uint32_t>(
            fairness_debt,
            policy.starvation_boosted
                ? 0u
                : (policy.starvation_score > policy.base_budget_permille
                    ? policy.starvation_score - policy.base_budget_permille
                    : 0u));
    }
    summary.fairness_debt_permille = std::min<uint32_t>(1000u, fairness_debt);
    summary.class_aware_dispatch_mode =
        summary.kernel_class_policy_count > 0 ? "class_aware_budget_redistribution" : "class_matrix_unavailable";
    if (starvation_target) {
        summary.redistribution_mode = "aging_boost_redistribution";
        summary.redistribution_reason = "starved_kernel_class_promoted";
        summary.starvation_prevention_mode = "aging_guard_enabled";
        summary.starvation_prevention_reason = "class_queue_age_crossed_threshold";
        summary.starvation_target_kernel = starvation_target->kernel_name;
    } else if (summary.last_dispatch_latency_class == "latency_sensitive") {
        summary.redistribution_mode = "anchor_priority_redistribution";
        summary.redistribution_reason = "interactive_lane_protected";
        summary.starvation_prevention_mode = "monitor_only";
        summary.starvation_prevention_reason = "no_starved_lane_detected";
        summary.starvation_target_kernel = "";
    } else {
        summary.redistribution_mode = "balanced_class_redistribution";
        summary.redistribution_reason = "class_matrix_balanced";
        summary.starvation_prevention_mode = "monitor_only";
        summary.starvation_prevention_reason = "aging_below_threshold";
        summary.starvation_target_kernel = "";
    }
    if (throttle_target) {
        summary.throttle_target_kernel = throttle_target->kernel_name;
        summary.pressure_recovery_action = "throttle_background_lane";
        summary.pressure_recovery_reason = "queue_pressure_budget_recovery";
    } else if (summary.queue_guard_triggered) {
        summary.throttle_target_kernel = "";
        summary.pressure_recovery_action = "guard_live_dispatch";
        summary.pressure_recovery_reason = "queue_guard_triggered_without_safe_throttle_target";
    } else {
        summary.throttle_target_kernel = "";
        summary.pressure_recovery_action = "steady_state";
        summary.pressure_recovery_reason = "queue_pressure_low";
    }
    summary.artifact_path = artifact_path();
    return summary;
}

bool runtime_scheduler_write_artifact(std::string* out_path) {
    const RuntimeSchedulerSummary summary = runtime_scheduler_get_summary();
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(summary.artifact_path).parent_path(), ec);
    std::ofstream out(summary.artifact_path, std::ios::trunc);
    if (!out.is_open()) return false;

    out << "{\n"
        << "  \"available\": true,\n"
        << "  \"compileIntents\": " << summary.compile_intents << ",\n"
        << "  \"dedupedCompileIntents\": " << summary.deduped_compile_intents << ",\n"
        << "  \"programReuseHits\": " << summary.program_reuse_hits << ",\n"
        << "  \"binaryCacheHits\": " << summary.binary_cache_hits << ",\n"
        << "  \"precompiledBinaryHits\": " << summary.precompiled_binary_hits << ",\n"
        << "  \"sourceBuilds\": " << summary.source_builds << ",\n"
        << "  \"ilBuilds\": " << summary.il_builds << ",\n"
        << "  \"binaryBuilds\": " << summary.binary_builds << ",\n"
        << "  \"asyncBuildCallbacks\": " << summary.async_build_callbacks << ",\n"
        << "  \"kernelReadyCount\": " << summary.kernel_ready_count << ",\n"
        << "  \"autotuneEntries\": " << summary.autotune_entries << ",\n"
        << "  \"memoryPlan\": {\n"
        << "    \"globalMemBytes\": " << summary.global_mem_bytes << ",\n"
        << "    \"recommendedResidentBudgetBytes\": " << summary.recommended_budget_bytes << ",\n"
        << "    \"kernelCacheBudgetBytes\": " << summary.kernel_cache_budget_bytes << ",\n"
        << "    \"precompiledBudgetBytes\": " << summary.precompiled_budget_bytes << ",\n"
        << "    \"autotuneBudgetBytes\": " << summary.autotune_budget_bytes << ",\n"
        << "    \"kernelCacheBytes\": " << summary.kernel_cache_bytes << ",\n"
        << "    \"precompiledLibraryBytes\": " << summary.precompiled_library_bytes << ",\n"
        << "    \"residentBytes\": " << summary.resident_bytes << ",\n"
        << "    \"residentPressurePermille\": " << summary.resident_pressure_permille << ",\n"
        << "    \"residentPressureBand\": \"" << summary.resident_pressure_band << "\",\n"
        << "    \"evictionRecommended\": " << (summary.eviction_recommended ? "true" : "false") << ",\n"
        << "    \"compactionRecommended\": " << (summary.compaction_recommended ? "true" : "false") << ",\n"
        << "    \"persistentResidencyReady\": " << (summary.persistent_residency_ready ? "true" : "false") << "\n"
        << "  },\n"
        << "  \"planner\": {\n"
        << "    \"preferredBuildPath\": \"" << summary.preferred_build_path << "\",\n"
        << "    \"compileQueueDepth\": " << summary.compile_queue_depth << ",\n"
        << "    \"admittedCompileCount\": " << summary.admitted_compile_count << ",\n"
        << "    \"deferredCompileCount\": " << summary.deferred_compile_count << ",\n"
        << "    \"speculativeCandidateCount\": " << summary.speculative_candidate_count << ",\n"
        << "    \"executionQueueSlots\": " << summary.execution_queue_slots << ",\n"
        << "    \"executionBudgetSlots\": " << summary.execution_budget_slots << ",\n"
        << "    \"compileBudgetSlots\": " << summary.compile_budget_slots << ",\n"
        << "    \"compileDeferredForExecution\": " << summary.compile_deferred_for_execution << ",\n"
        << "    \"dispatchBudgetPermille\": " << summary.dispatch_budget_permille << ",\n"
        << "    \"compileBudgetPermille\": " << summary.compile_budget_permille << ",\n"
        << "    \"executionAdmission\": \"" << summary.execution_admission << "\",\n"
        << "    \"executionAdmissionReason\": \"" << summary.execution_admission_reason << "\",\n"
        << "    \"arbitrationMode\": \"" << summary.arbitration_mode << "\",\n"
        << "    \"arbitrationReason\": \"" << summary.arbitration_reason << "\",\n"
        << "    \"multiQueueMode\": \"" << summary.multi_queue_mode << "\",\n"
        << "    \"queueGuardTriggered\": " << (summary.queue_guard_triggered ? "true" : "false") << ",\n"
        << "    \"queueGuardReason\": \"" << summary.queue_guard_reason << "\",\n"
        << "    \"compileExecuteBalancePermille\": " << summary.compile_execute_balance_permille << ",\n"
        << "    \"compileQueueAgeTicks\": " << summary.compile_queue_age_ticks << ",\n"
        << "    \"dispatchQueueAgeTicks\": " << summary.dispatch_queue_age_ticks << ",\n"
        << "    \"starvationPreventedCount\": " << summary.starvation_prevented_count << ",\n"
        << "    \"fairnessDebtPermille\": " << summary.fairness_debt_permille << ",\n"
        << "    \"compileQueueAgeBand\": \"" << summary.compile_queue_age_band << "\",\n"
        << "    \"dispatchQueueAgeBand\": \"" << summary.dispatch_queue_age_band << "\",\n"
        << "    \"latencyAdmission\": \"" << summary.latency_admission << "\",\n"
        << "    \"latencyAdmissionReason\": \"" << summary.latency_admission_reason << "\",\n"
        << "    \"dispatchUrgency\": \"" << summary.dispatch_urgency << "\",\n"
        << "    \"classAwareDispatchMode\": \"" << summary.class_aware_dispatch_mode << "\",\n"
        << "    \"redistributionMode\": \"" << summary.redistribution_mode << "\",\n"
        << "    \"redistributionReason\": \"" << summary.redistribution_reason << "\",\n"
        << "    \"starvationPreventionMode\": \"" << summary.starvation_prevention_mode << "\",\n"
        << "    \"starvationPreventionReason\": \"" << summary.starvation_prevention_reason << "\",\n"
        << "    \"starvationTargetKernel\": \"" << summary.starvation_target_kernel << "\",\n"
        << "    \"throttleTargetKernel\": \"" << summary.throttle_target_kernel << "\",\n"
        << "    \"pressureRecoveryAction\": \"" << summary.pressure_recovery_action << "\",\n"
        << "    \"pressureRecoveryReason\": \"" << summary.pressure_recovery_reason << "\",\n"
        << "    \"kernelClassPolicyCount\": " << summary.kernel_class_policy_count << ",\n"
        << "    \"evictionCandidateCount\": " << summary.eviction_candidate_count << ",\n"
        << "    \"persistentResidencyReady\": " << (summary.persistent_residency_ready ? "true" : "false") << ",\n"
        << "    \"compactionRecommended\": " << (summary.compaction_recommended ? "true" : "false") << "\n"
        << "  },\n"
        << "  \"execution\": {\n"
        << "    \"dispatchCount\": " << summary.execution_dispatch_count << ",\n"
        << "    \"profiledDispatchCount\": " << summary.execution_profiled_dispatch_count << ",\n"
        << "    \"dispatchWaveSlots\": " << summary.dispatch_wave_slots << ",\n"
        << "    \"activeDispatchSlots\": " << summary.active_dispatch_slots << ",\n"
        << "    \"queuePressurePermille\": " << summary.queue_pressure_permille << ",\n"
        << "    \"queuePressureBand\": \"" << summary.queue_pressure_band << "\",\n"
        << "    \"dispatchUtilizationPermille\": " << summary.dispatch_utilization_permille << ",\n"
        << "    \"compileUtilizationPermille\": " << summary.compile_utilization_permille << ",\n"
        << "    \"lastDispatchKernelName\": \"" << summary.last_dispatch_kernel_name << "\",\n"
        << "    \"lastDispatchLatencyClass\": \"" << summary.last_dispatch_latency_class << "\",\n"
        << "    \"lastDispatchLatencyReason\": \"" << summary.last_dispatch_latency_reason << "\",\n"
        << "    \"lastDispatchKernelNs\": " << summary.last_dispatch_kernel_ns << ",\n"
        << "    \"lastDispatchGlobalSize\": " << summary.last_dispatch_global_size << ",\n"
        << "    \"lastDispatchLocalSize\": " << summary.last_dispatch_local_size << ",\n"
        << "    \"executionBudgetBytes\": " << summary.execution_budget_bytes << ",\n"
        << "    \"executionPressureBand\": \"" << summary.execution_pressure_band << "\"\n"
        << "  },\n"
        << "  \"compileCandidates\": [\n";
    for (size_t i = 0; i < summary.compile_candidates.size(); ++i) {
        const auto& candidate = summary.compile_candidates[i];
        out << "    {\"kernelName\":\"" << candidate.kernel_name
            << "\",\"compileKey\":\"" << candidate.compile_key
            << "\",\"preferredPath\":\"" << candidate.preferred_path
            << "\",\"admission\":\"" << candidate.admission
            << "\",\"reason\":\"" << candidate.reason
            << "\",\"queueOrdinal\":" << candidate.queue_ordinal
            << ",\"speculative\":" << (candidate.speculative ? "true" : "false")
            << ",\"admitted\":" << (candidate.admitted ? "true" : "false")
            << ",\"hasPrecompiled\":" << (candidate.has_precompiled ? "true" : "false")
            << ",\"hasBinaryCache\":" << (candidate.has_binary_cache ? "true" : "false")
            << ",\"hasSpirvIl\":" << (candidate.has_spirv_il ? "true" : "false")
            << ",\"hasSource\":" << (candidate.has_source ? "true" : "false") << "}";
        if (i + 1 < summary.compile_candidates.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n"
        << "  \"kernelClassPolicies\": [\n";
    for (size_t i = 0; i < summary.kernel_class_policies.size(); ++i) {
        const auto& policy = summary.kernel_class_policies[i];
        out << "    {\"kernelName\":\"" << policy.kernel_name
            << "\",\"kernelClass\":\"" << policy.kernel_class
            << "\",\"latencyClass\":\"" << policy.latency_class
            << "\",\"latencyReason\":\"" << policy.latency_reason
            << "\",\"admission\":\"" << policy.admission
            << "\",\"admissionReason\":\"" << policy.admission_reason
            << "\",\"queueAgeBand\":\"" << policy.queue_age_band
            << "\",\"urgency\":\"" << policy.urgency
            << "\",\"dispatchLane\":\"" << policy.dispatch_lane
            << "\",\"redistributionAction\":\"" << policy.redistribution_action
            << "\",\"redistributionReason\":\"" << policy.redistribution_reason
            << "\",\"pressureAction\":\"" << policy.pressure_action
            << "\",\"pressureReason\":\"" << policy.pressure_reason
            << "\",\"queueAgeTicks\":" << policy.queue_age_ticks
            << ",\"queueOrdinal\":" << policy.queue_ordinal
            << ",\"baseBudgetPermille\":" << policy.base_budget_permille
            << ",\"ageBoostPermille\":" << policy.age_boost_permille
            << ",\"effectiveBudgetPermille\":" << policy.effective_budget_permille
            << ",\"starvationScore\":" << policy.starvation_score
            << ",\"starvationBoosted\":" << (policy.starvation_boosted ? "true" : "false")
            << ",\"throttleRecommended\":" << (policy.throttle_recommended ? "true" : "false")
            << ",\"liveDispatchAnchor\":" << (policy.live_dispatch_anchor ? "true" : "false") << "}";
        if (i + 1 < summary.kernel_class_policies.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n"
        << "  \"evictionCandidates\": [\n";
    for (size_t i = 0; i < summary.eviction_candidates.size(); ++i) {
        const auto& candidate = summary.eviction_candidates[i];
        out << "    {\"tier\":\"" << candidate.tier
            << "\",\"key\":\"" << candidate.key
            << "\",\"bytes\":" << candidate.bytes
            << ",\"persistent\":" << (candidate.persistent ? "true" : "false")
            << ",\"priority\":" << candidate.priority << "}";
        if (i + 1 < summary.eviction_candidates.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n"
        << "}\n";

    if (out_path) *out_path = summary.artifact_path;
    return true;
}

} // namespace tq
