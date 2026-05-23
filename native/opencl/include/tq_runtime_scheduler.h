/**
 * TurboQuant OpenCL — runtime scheduler/allocator state.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_opencl.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tq {

struct RuntimeResidencyCandidate {
    std::string tier;
    std::string key;
    uint64_t bytes = 0;
    bool persistent = false;
    uint32_t priority = 0;
};

struct RuntimeCompileCandidate {
    std::string kernel_name;
    std::string compile_key;
    std::string preferred_path;
    std::string admission;
    std::string reason;
    uint32_t queue_ordinal = 0;
    bool speculative = false;
    bool admitted = false;
    bool has_precompiled = false;
    bool has_binary_cache = false;
    bool has_spirv_il = false;
    bool has_source = false;
};

struct RuntimeKernelClassPolicy {
    std::string kernel_name;
    std::string kernel_class;
    std::string latency_class;
    std::string latency_reason;
    std::string admission;
    std::string admission_reason;
    std::string queue_age_band;
    std::string urgency;
    std::string dispatch_lane;
    std::string redistribution_action;
    std::string redistribution_reason;
    std::string pressure_action;
    std::string pressure_reason;
    uint32_t queue_age_ticks = 0;
    uint32_t queue_ordinal = 0;
    uint32_t base_budget_permille = 0;
    uint32_t age_boost_permille = 0;
    uint32_t effective_budget_permille = 0;
    uint32_t starvation_score = 0;
    bool starvation_boosted = false;
    bool throttle_recommended = false;
    bool live_dispatch_anchor = false;
};

struct RuntimeSchedulerSummary {
    bool available = false;
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
    uint64_t global_mem_bytes = 0;
    uint64_t recommended_budget_bytes = 0;
    uint64_t kernel_cache_bytes = 0;
    uint64_t precompiled_library_bytes = 0;
    uint64_t resident_bytes = 0;
    uint32_t resident_pressure_permille = 0;
    std::string resident_pressure_band;
    bool eviction_recommended = false;
    uint32_t autotune_entries = 0;
    std::string preferred_build_path;
    uint32_t compile_queue_depth = 0;
    uint32_t admitted_compile_count = 0;
    uint32_t deferred_compile_count = 0;
    uint32_t speculative_candidate_count = 0;
    uint32_t execution_queue_slots = 0;
    uint32_t execution_dispatch_count = 0;
    uint32_t execution_profiled_dispatch_count = 0;
    uint32_t active_dispatch_slots = 0;
    uint32_t queue_pressure_permille = 0;
    uint32_t dispatch_utilization_permille = 0;
    uint32_t compile_utilization_permille = 0;
    uint32_t compile_deferred_for_execution = 0;
    uint32_t compile_budget_slots = 0;
    uint32_t execution_budget_slots = 0;
    uint32_t dispatch_budget_permille = 0;
    uint32_t compile_budget_permille = 0;
    uint64_t last_dispatch_kernel_ns = 0;
    size_t last_dispatch_global_size = 0;
    size_t last_dispatch_local_size = 0;
    std::string last_dispatch_kernel_name;
    std::string last_dispatch_latency_class;
    std::string last_dispatch_latency_reason;
    uint64_t execution_budget_bytes = 0;
    std::string execution_pressure_band;
    std::string queue_pressure_band;
    std::string execution_admission;
    std::string execution_admission_reason;
    std::string arbitration_mode;
    std::string arbitration_reason;
    std::string multi_queue_mode;
    bool queue_guard_triggered = false;
    std::string queue_guard_reason;
    uint32_t dispatch_wave_slots = 0;
    uint32_t compile_execute_balance_permille = 0;
    uint32_t compile_queue_age_ticks = 0;
    uint32_t dispatch_queue_age_ticks = 0;
    uint32_t starvation_prevented_count = 0;
    uint32_t fairness_debt_permille = 0;
    std::string compile_queue_age_band;
    std::string dispatch_queue_age_band;
    std::string latency_admission;
    std::string latency_admission_reason;
    std::string dispatch_urgency;
    std::string class_aware_dispatch_mode;
    std::string redistribution_mode;
    std::string redistribution_reason;
    std::string starvation_prevention_mode;
    std::string starvation_prevention_reason;
    std::string starvation_target_kernel;
    std::string throttle_target_kernel;
    std::string pressure_recovery_action;
    std::string pressure_recovery_reason;
    uint64_t kernel_cache_budget_bytes = 0;
    uint64_t precompiled_budget_bytes = 0;
    uint64_t autotune_budget_bytes = 0;
    bool persistent_residency_ready = false;
    bool compaction_recommended = false;
    uint32_t eviction_candidate_count = 0;
    uint32_t compile_candidate_count = 0;
    uint32_t kernel_class_policy_count = 0;
    std::vector<RuntimeResidencyCandidate> eviction_candidates;
    std::vector<RuntimeCompileCandidate> compile_candidates;
    std::vector<RuntimeKernelClassPolicy> kernel_class_policies;
    std::string artifact_path;
};

void runtime_scheduler_reset();
void runtime_scheduler_note_compile_intent(const std::string& kernel_name, const std::string& compile_key);
void runtime_scheduler_note_program_reuse(const std::string& kernel_name, const std::string& program_key);
void runtime_scheduler_note_cache_hit(const char* kind, const std::string& kernel_name, const std::string& key, size_t bytes);
void runtime_scheduler_note_compile_candidate(
    const std::string& kernel_name,
    const std::string& compile_key,
    bool has_precompiled,
    bool has_binary_cache,
    bool has_spirv_il,
    bool has_source,
    bool source_only_forced);
void runtime_scheduler_note_build_stage(const char* stage, bool ok, bool async_callback_observed);
void runtime_scheduler_note_kernel_ready(const std::string& kernel_name, const KernelMetadata& meta);
void runtime_scheduler_note_execution_dispatch(
    const std::string& kernel_name,
    size_t global_size,
    size_t local_size,
    bool profiled,
    uint64_t kernel_ns);
RuntimeSchedulerSummary runtime_scheduler_get_summary();
bool runtime_scheduler_write_artifact(std::string* out_path = nullptr);

} // namespace tq
