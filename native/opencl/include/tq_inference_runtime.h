/**
 * TurboQuant OpenCL — model inference runtime state.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tq {

struct InferenceRequestPlan {
    std::string request_id;
    uint32_t prompt_tokens = 0;
    uint32_t decode_tokens = 0;
    uint32_t prefill_chunk_size = 0;
    uint32_t prefill_chunks = 0;
    uint32_t prefill_chunks_executed = 0;
    uint32_t decode_steps_executed = 0;
    uint32_t peak_context_tokens = 0;
    uint32_t continuous_batch_slots_seen = 0;
    bool entered_mixed_wave = false;
    bool prefill_complete = false;
    bool decode_complete = false;
    std::string prefill_backend;
    std::string decode_backend;
};

struct InferenceWaveRecord {
    uint32_t wave_id = 0;
    std::string phase;
    uint32_t prefill_request_count = 0;
    uint32_t decode_request_count = 0;
    uint32_t batched_request_count = 0;
    uint32_t batched_token_count = 0;
    uint32_t prefill_chunk_tokens = 0;
    uint32_t decode_step_tokens = 0;
    uint32_t decode_queue_depth = 0;
    uint32_t prefill_queue_depth = 0;
    bool continuous_batching = false;
    bool decode_prioritized = false;
    bool mixed_with_running_decode = false;
    uint64_t fused_attention_ns = 0;
    uint64_t qjl_ns = 0;
    uint64_t value_dequant_ns = 0;
    std::string anchor_kernel;
    std::vector<std::string> request_ids;
};

struct InferenceRuntimeSummary {
    bool available = false;
    bool prefill_decode_split_ready = false;
    bool chunked_prefill_ready = false;
    bool continuous_batching_ready = false;
    bool mixed_prefill_decode_ready = false;
    uint32_t request_count = 0;
    uint32_t chunk_size_tokens = 0;
    uint32_t max_batch_tokens = 0;
    uint32_t total_prefill_tokens = 0;
    uint32_t total_decode_tokens = 0;
    uint32_t total_prefill_chunks = 0;
    uint32_t prefill_wave_count = 0;
    uint32_t decode_wave_count = 0;
    uint32_t mixed_wave_count = 0;
    uint32_t scheduler_turns = 0;
    uint32_t max_continuous_batch_size = 0;
    uint32_t max_decode_batch_size = 0;
    uint32_t max_prefill_batch_size = 0;
    uint32_t max_prefill_chunk_tokens = 0;
    uint32_t avg_prefill_chunk_tokens = 0;
    uint32_t decode_queue_peak = 0;
    uint32_t prefill_queue_peak = 0;
    uint32_t decode_priority_turns = 0;
    uint64_t total_fused_attention_ns = 0;
    uint64_t total_qjl_ns = 0;
    uint64_t total_value_dequant_ns = 0;
    std::string prefill_backend;
    std::string decode_backend;
    std::string batching_policy;
    std::string chunking_policy;
    std::string artifact_path;
    std::vector<InferenceRequestPlan> requests;
    std::vector<InferenceWaveRecord> waves;
};

struct PagedKvRequestSummary {
    std::string request_id;
    uint32_t prompt_tokens = 0;
    uint32_t decode_tokens = 0;
    uint32_t logical_block_count = 0;
    uint32_t unique_physical_block_count = 0;
    uint32_t shared_physical_block_count = 0;
    uint32_t shared_prefix_blocks = 0;
    uint32_t decode_block_appends = 0;
    uint32_t peak_context_tokens = 0;
    std::vector<uint32_t> physical_block_ids;
};

struct PagedKvBlockRecord {
    uint32_t physical_block_id = 0;
    uint32_t token_count = 0;
    uint32_t logical_ref_count = 0;
    uint32_t ref_count = 0;
    uint32_t last_access_tick = 0;
    bool shared_prefix = false;
    bool live = false;
    bool pinned = false;
    bool evictable = false;
    bool evicted = false;
    std::string cache_key;
    std::vector<std::string> owner_request_ids;
};

struct PagedKvPrefixCacheSummary {
    bool available = false;
    bool paged_kv_allocator_ready = false;
    bool block_table_ready = false;
    bool prefix_cache_sharing_ready = false;
    bool cache_eviction_ready = false;
    uint32_t request_count = 0;
    uint32_t block_size_tokens = 0;
    uint32_t physical_block_count = 0;
    uint32_t logical_block_ref_count = 0;
    uint32_t shared_block_count = 0;
    uint32_t unique_block_count = 0;
    uint32_t shared_prefix_group_count = 0;
    uint32_t peak_live_block_count = 0;
    uint32_t resident_block_count = 0;
    uint32_t evictable_block_count = 0;
    uint32_t evicted_block_count = 0;
    uint32_t max_block_refcount = 0;
    uint64_t total_fused_attention_ns = 0;
    uint64_t total_qjl_ns = 0;
    uint64_t total_value_dequant_ns = 0;
    std::string prefill_backend;
    std::string decode_backend;
    std::string sharing_policy;
    std::string eviction_policy;
    std::string eviction_decision;
    std::string artifact_path;
    std::vector<PagedKvRequestSummary> requests;
    std::vector<PagedKvBlockRecord> blocks;
};

void inference_runtime_reset();
bool inference_runtime_run_smoke();
InferenceRuntimeSummary inference_runtime_get_summary();
bool inference_runtime_write_artifact(std::string* out_path = nullptr);
void paged_kv_prefix_cache_reset();
bool paged_kv_prefix_cache_run_smoke();
PagedKvPrefixCacheSummary paged_kv_prefix_cache_get_summary();
bool paged_kv_prefix_cache_write_artifact(std::string* out_path = nullptr);

} // namespace tq
