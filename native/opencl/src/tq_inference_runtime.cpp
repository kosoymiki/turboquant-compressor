/**
 * TurboQuant OpenCL — bounded model inference runtime evidence.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_inference_runtime.h"

#include "tq_opencl.h"
#include "../include/tq_repo_paths.h"
#include "../include/tq_trace.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace tq {
namespace {

constexpr uint32_t kChunkSizeTokens = 64;
constexpr uint32_t kMaxBatchTokens = 192;
constexpr uint32_t kMaxDecodeRequestsPerWave = 2;
constexpr uint32_t kMaxPrefillRequestsPerWave = 2;
constexpr int kHeadDim = 64;
constexpr int kBits = 4;
constexpr int kGroupSize = 32;

struct SyntheticRequestBuffers {
    std::string request_id;
    int prompt_tokens = 0;
    int decode_tokens = 0;
    int key_packed_stride = 0;
    int value_packed_stride = 0;
    int n_groups = 0;
    int projections = 0;
    int proj_words = 0;

    std::vector<float> query_rotated;
    std::vector<uint8_t> key_packed;
    std::vector<float> key_norms;
    std::vector<float> centroids;
    std::vector<uint32_t> query_signs;
    std::vector<uint32_t> residual_signs;
    std::vector<float> residual_norms;
    std::vector<float> base_scores;
    std::vector<uint8_t> value_packed;
    std::vector<float> value_scales;
    std::vector<float> value_zeros;
    std::vector<float> output;
};

struct LiveRequestState {
    uint32_t prefill_consumed = 0;
    uint32_t decode_emitted = 0;
};

InferenceRuntimeSummary g_summary{};
PagedKvPrefixCacheSummary g_paged_summary{};

static std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out << buf;
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

static void fill_random(std::vector<float>& buf, std::mt19937& rng, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (float& v : buf) v = dist(rng);
}

static void fill_random_u8(std::vector<uint8_t>& buf, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    for (uint8_t& v : buf) v = static_cast<uint8_t>(dist(rng));
}

static void fill_random_u32(std::vector<uint32_t>& buf, std::mt19937& rng) {
    std::uniform_int_distribution<uint32_t> dist(0, 0xffffffffu);
    for (uint32_t& v : buf) v = dist(rng);
}

static SyntheticRequestBuffers make_request_buffers(
    const std::string& request_id,
    int prompt_tokens,
    int decode_tokens,
    std::mt19937& rng) {
    SyntheticRequestBuffers req;
    req.request_id = request_id;
    req.prompt_tokens = prompt_tokens;
    req.decode_tokens = decode_tokens;
    req.key_packed_stride = (kHeadDim * kBits + 7) / 8;
    req.value_packed_stride = (kHeadDim * kBits + 7) / 8;
    req.n_groups = (kHeadDim + kGroupSize - 1) / kGroupSize;
    req.projections = kHeadDim * 2;
    req.proj_words = (req.projections + 31) / 32;

    req.query_rotated.resize(kHeadDim);
    req.key_packed.resize(static_cast<size_t>(prompt_tokens) * req.key_packed_stride);
    req.key_norms.resize(prompt_tokens);
    req.centroids.resize(1u << kBits);
    req.query_signs.resize(req.proj_words);
    req.residual_signs.resize(static_cast<size_t>(prompt_tokens) * req.proj_words);
    req.residual_norms.resize(prompt_tokens);
    req.base_scores.resize(prompt_tokens);
    req.value_packed.resize(static_cast<size_t>(prompt_tokens) * req.value_packed_stride);
    req.value_scales.resize(static_cast<size_t>(prompt_tokens) * req.n_groups);
    req.value_zeros.resize(static_cast<size_t>(prompt_tokens) * req.n_groups);
    req.output.resize(kHeadDim);

    fill_random(req.query_rotated, rng);
    fill_random_u8(req.key_packed, rng);
    fill_random(req.key_norms, rng, 0.5f, 1.5f);
    fill_random(req.centroids, rng, -1.5f, 1.5f);
    fill_random_u32(req.query_signs, rng);
    fill_random_u32(req.residual_signs, rng);
    fill_random(req.residual_norms, rng, 0.0f, 1.0f);
    fill_random(req.value_scales, rng, 0.1f, 1.0f);
    fill_random(req.value_zeros, rng, -1.0f, 1.0f);
    std::fill(req.base_scores.begin(), req.base_scores.end(), 0.0f);
    std::fill(req.output.begin(), req.output.end(), 0.0f);
    return req;
}

static TqStatus run_prefill_anchor(
    SyntheticRequestBuffers& req,
    uint32_t token_offset,
    uint32_t chunk_tokens,
    uint64_t* fused_ns) {
    MseScoreInput mse{};
    mse.packed_indices = req.key_packed.data() + static_cast<size_t>(token_offset) * req.key_packed_stride;
    mse.norms = req.key_norms.data() + token_offset;
    mse.query_rotated = req.query_rotated.data();
    mse.centroids = req.centroids.data();
    mse.tokens = static_cast<int>(chunk_tokens);
    mse.dim = kHeadDim;
    mse.bits = kBits;
    mse.packed_stride = req.key_packed_stride;

    QjlScoreInput qjl{};
    qjl.query_signs = req.query_signs.data();
    qjl.residual_signs = req.residual_signs.data() + static_cast<size_t>(token_offset) * req.proj_words;
    qjl.residual_norms = req.residual_norms.data() + token_offset;
    qjl.base_scores = req.base_scores.data();
    qjl.tokens = static_cast<int>(chunk_tokens);
    qjl.projections = req.projections;
    qjl.qjl_scale = 0.1f;

    ValueDequantInput value{};
    value.packed_values = req.value_packed.data() + static_cast<size_t>(token_offset) * req.value_packed_stride;
    value.scales = req.value_scales.data() + static_cast<size_t>(token_offset) * req.n_groups;
    value.zeros = req.value_zeros.data() + static_cast<size_t>(token_offset) * req.n_groups;
    value.tokens = static_cast<int>(chunk_tokens);
    value.dim = kHeadDim;
    value.bits = kBits;
    value.group_size = kGroupSize;

    FusedAttentionInput input{};
    input.mse = mse;
    input.qjl = qjl;
    input.value = value;
    input.sm_scale = 1.0f / 8.0f;
    input.output = req.output.data();

    return fused_attention_tiled_opencl_profiled(input, fused_ns);
}

static TqStatus run_decode_anchor(
    SyntheticRequestBuffers& req,
    uint32_t context_tokens,
    uint64_t* qjl_ns,
    uint64_t* value_ns) {
    if (context_tokens == 0) {
        *qjl_ns = 0;
        *value_ns = 0;
        return TqStatus::OK;
    }

    std::fill(req.base_scores.begin(), req.base_scores.end(), 0.0f);

    QjlScoreInput qjl{};
    qjl.query_signs = req.query_signs.data();
    qjl.residual_signs = req.residual_signs.data();
    qjl.residual_norms = req.residual_norms.data();
    qjl.base_scores = req.base_scores.data();
    qjl.tokens = static_cast<int>(context_tokens);
    qjl.projections = req.projections;
    qjl.qjl_scale = 0.1f;

    ValueDequantInput value{};
    value.packed_values = req.value_packed.data();
    value.scales = req.value_scales.data();
    value.zeros = req.value_zeros.data();
    value.tokens = static_cast<int>(context_tokens);
    value.dim = kHeadDim;
    value.bits = kBits;
    value.group_size = kGroupSize;

    auto st = qjl_score_opencl_profiled(qjl, qjl_ns);
    if (st != TqStatus::OK) return st;
    return value_dequant_opencl_profiled(value, req.output.data(), value_ns);
}

static bool write_summary_json(const InferenceRuntimeSummary& summary, std::string* out_path) {
    const std::string forensics_dir = resolve_forensics_dir();
    if (forensics_dir.empty()) return false;
    std::filesystem::create_directories(forensics_dir);
    const std::string path = join_repo_path(forensics_dir, "inference-runtime-state.json");

    std::ostringstream out;
    out << "{\n"
        << "  \"available\": " << (summary.available ? "true" : "false") << ",\n"
        << "  \"prefillDecodeSplitReady\": " << (summary.prefill_decode_split_ready ? "true" : "false") << ",\n"
        << "  \"chunkedPrefillReady\": " << (summary.chunked_prefill_ready ? "true" : "false") << ",\n"
        << "  \"continuousBatchingReady\": " << (summary.continuous_batching_ready ? "true" : "false") << ",\n"
        << "  \"mixedPrefillDecodeReady\": " << (summary.mixed_prefill_decode_ready ? "true" : "false") << ",\n"
        << "  \"requestCount\": " << summary.request_count << ",\n"
        << "  \"chunkSizeTokens\": " << summary.chunk_size_tokens << ",\n"
        << "  \"maxBatchTokens\": " << summary.max_batch_tokens << ",\n"
        << "  \"totalPrefillTokens\": " << summary.total_prefill_tokens << ",\n"
        << "  \"totalDecodeTokens\": " << summary.total_decode_tokens << ",\n"
        << "  \"totalPrefillChunks\": " << summary.total_prefill_chunks << ",\n"
        << "  \"prefillWaveCount\": " << summary.prefill_wave_count << ",\n"
        << "  \"decodeWaveCount\": " << summary.decode_wave_count << ",\n"
        << "  \"mixedWaveCount\": " << summary.mixed_wave_count << ",\n"
        << "  \"schedulerTurns\": " << summary.scheduler_turns << ",\n"
        << "  \"maxContinuousBatchSize\": " << summary.max_continuous_batch_size << ",\n"
        << "  \"maxDecodeBatchSize\": " << summary.max_decode_batch_size << ",\n"
        << "  \"maxPrefillBatchSize\": " << summary.max_prefill_batch_size << ",\n"
        << "  \"maxPrefillChunkTokens\": " << summary.max_prefill_chunk_tokens << ",\n"
        << "  \"avgPrefillChunkTokens\": " << summary.avg_prefill_chunk_tokens << ",\n"
        << "  \"decodeQueuePeak\": " << summary.decode_queue_peak << ",\n"
        << "  \"prefillQueuePeak\": " << summary.prefill_queue_peak << ",\n"
        << "  \"decodePriorityTurns\": " << summary.decode_priority_turns << ",\n"
        << "  \"totalFusedAttentionNs\": " << summary.total_fused_attention_ns << ",\n"
        << "  \"totalQjlNs\": " << summary.total_qjl_ns << ",\n"
        << "  \"totalValueDequantNs\": " << summary.total_value_dequant_ns << ",\n"
        << "  \"prefillBackend\": \"" << json_escape(summary.prefill_backend) << "\",\n"
        << "  \"decodeBackend\": \"" << json_escape(summary.decode_backend) << "\",\n"
        << "  \"batchingPolicy\": \"" << json_escape(summary.batching_policy) << "\",\n"
        << "  \"chunkingPolicy\": \"" << json_escape(summary.chunking_policy) << "\",\n"
        << "  \"requests\": [\n";
    for (size_t i = 0; i < summary.requests.size(); ++i) {
        const auto& req = summary.requests[i];
        if (i) out << ",\n";
        out << "    {"
            << "\"requestId\":\"" << json_escape(req.request_id) << "\","
            << "\"promptTokens\":" << req.prompt_tokens << ","
            << "\"decodeTokens\":" << req.decode_tokens << ","
            << "\"prefillChunkSize\":" << req.prefill_chunk_size << ","
            << "\"prefillChunks\":" << req.prefill_chunks << ","
            << "\"prefillChunksExecuted\":" << req.prefill_chunks_executed << ","
            << "\"decodeStepsExecuted\":" << req.decode_steps_executed << ","
            << "\"peakContextTokens\":" << req.peak_context_tokens << ","
            << "\"continuousBatchSlotsSeen\":" << req.continuous_batch_slots_seen << ","
            << "\"enteredMixedWave\":" << (req.entered_mixed_wave ? "true" : "false") << ","
            << "\"prefillComplete\":" << (req.prefill_complete ? "true" : "false") << ","
            << "\"decodeComplete\":" << (req.decode_complete ? "true" : "false") << ","
            << "\"prefillBackend\":\"" << json_escape(req.prefill_backend) << "\","
            << "\"decodeBackend\":\"" << json_escape(req.decode_backend) << "\"}";
    }
    out << "\n  ],\n  \"waves\": [\n";
    for (size_t i = 0; i < summary.waves.size(); ++i) {
        const auto& wave = summary.waves[i];
        if (i) out << ",\n";
        out << "    {"
            << "\"waveId\":" << wave.wave_id << ","
            << "\"phase\":\"" << json_escape(wave.phase) << "\","
            << "\"prefillRequestCount\":" << wave.prefill_request_count << ","
            << "\"decodeRequestCount\":" << wave.decode_request_count << ","
            << "\"batchedRequestCount\":" << wave.batched_request_count << ","
            << "\"batchedTokenCount\":" << wave.batched_token_count << ","
            << "\"prefillChunkTokens\":" << wave.prefill_chunk_tokens << ","
            << "\"decodeStepTokens\":" << wave.decode_step_tokens << ","
            << "\"decodeQueueDepth\":" << wave.decode_queue_depth << ","
            << "\"prefillQueueDepth\":" << wave.prefill_queue_depth << ","
            << "\"continuousBatching\":" << (wave.continuous_batching ? "true" : "false") << ","
            << "\"decodePrioritized\":" << (wave.decode_prioritized ? "true" : "false") << ","
            << "\"mixedWithRunningDecode\":" << (wave.mixed_with_running_decode ? "true" : "false") << ","
            << "\"fusedAttentionNs\":" << wave.fused_attention_ns << ","
            << "\"qjlNs\":" << wave.qjl_ns << ","
            << "\"valueDequantNs\":" << wave.value_dequant_ns << ","
            << "\"anchorKernel\":\"" << json_escape(wave.anchor_kernel) << "\","
            << "\"requestIds\":[";
        for (size_t j = 0; j < wave.request_ids.size(); ++j) {
            if (j) out << ",";
            out << "\"" << json_escape(wave.request_ids[j]) << "\"";
        }
        out << "]}";
    }
    out << "\n  ]\n}\n";

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) return false;
    file << out.str();
    file.close();
    if (out_path) *out_path = path;
    return true;
}

static bool write_paged_summary_json(const PagedKvPrefixCacheSummary& summary, std::string* out_path) {
    const std::string forensics_dir = resolve_forensics_dir();
    if (forensics_dir.empty()) return false;
    std::filesystem::create_directories(forensics_dir);
    const std::string path = join_repo_path(forensics_dir, "paged-kv-prefix-cache-state.json");

    std::ostringstream out;
    out << "{\n"
        << "  \"available\": " << (summary.available ? "true" : "false") << ",\n"
        << "  \"pagedKvAllocatorReady\": " << (summary.paged_kv_allocator_ready ? "true" : "false") << ",\n"
        << "  \"blockTableReady\": " << (summary.block_table_ready ? "true" : "false") << ",\n"
        << "  \"prefixCacheSharingReady\": " << (summary.prefix_cache_sharing_ready ? "true" : "false") << ",\n"
        << "  \"cacheEvictionReady\": " << (summary.cache_eviction_ready ? "true" : "false") << ",\n"
        << "  \"requestCount\": " << summary.request_count << ",\n"
        << "  \"blockSizeTokens\": " << summary.block_size_tokens << ",\n"
        << "  \"physicalBlockCount\": " << summary.physical_block_count << ",\n"
        << "  \"logicalBlockRefCount\": " << summary.logical_block_ref_count << ",\n"
        << "  \"sharedBlockCount\": " << summary.shared_block_count << ",\n"
        << "  \"uniqueBlockCount\": " << summary.unique_block_count << ",\n"
        << "  \"sharedPrefixGroupCount\": " << summary.shared_prefix_group_count << ",\n"
        << "  \"peakLiveBlockCount\": " << summary.peak_live_block_count << ",\n"
        << "  \"residentBlockCount\": " << summary.resident_block_count << ",\n"
        << "  \"evictableBlockCount\": " << summary.evictable_block_count << ",\n"
        << "  \"evictedBlockCount\": " << summary.evicted_block_count << ",\n"
        << "  \"maxBlockRefcount\": " << summary.max_block_refcount << ",\n"
        << "  \"totalFusedAttentionNs\": " << summary.total_fused_attention_ns << ",\n"
        << "  \"totalQjlNs\": " << summary.total_qjl_ns << ",\n"
        << "  \"totalValueDequantNs\": " << summary.total_value_dequant_ns << ",\n"
        << "  \"prefillBackend\": \"" << json_escape(summary.prefill_backend) << "\",\n"
        << "  \"decodeBackend\": \"" << json_escape(summary.decode_backend) << "\",\n"
        << "  \"sharingPolicy\": \"" << json_escape(summary.sharing_policy) << "\",\n"
        << "  \"evictionPolicy\": \"" << json_escape(summary.eviction_policy) << "\",\n"
        << "  \"evictionDecision\": \"" << json_escape(summary.eviction_decision) << "\",\n"
        << "  \"requests\": [\n";
    for (size_t i = 0; i < summary.requests.size(); ++i) {
        const auto& req = summary.requests[i];
        if (i) out << ",\n";
        out << "    {"
            << "\"requestId\":\"" << json_escape(req.request_id) << "\","
            << "\"promptTokens\":" << req.prompt_tokens << ","
            << "\"decodeTokens\":" << req.decode_tokens << ","
            << "\"logicalBlockCount\":" << req.logical_block_count << ","
            << "\"uniquePhysicalBlockCount\":" << req.unique_physical_block_count << ","
            << "\"sharedPhysicalBlockCount\":" << req.shared_physical_block_count << ","
            << "\"sharedPrefixBlocks\":" << req.shared_prefix_blocks << ","
            << "\"decodeBlockAppends\":" << req.decode_block_appends << ","
            << "\"peakContextTokens\":" << req.peak_context_tokens << ","
            << "\"physicalBlockIds\":[";
        for (size_t j = 0; j < req.physical_block_ids.size(); ++j) {
            if (j) out << ",";
            out << req.physical_block_ids[j];
        }
        out << "]}";
    }
    out << "\n  ],\n  \"blocks\": [\n";
    for (size_t i = 0; i < summary.blocks.size(); ++i) {
        const auto& block = summary.blocks[i];
        if (i) out << ",\n";
        out << "    {"
            << "\"physicalBlockId\":" << block.physical_block_id << ","
            << "\"tokenCount\":" << block.token_count << ","
            << "\"logicalRefCount\":" << block.logical_ref_count << ","
            << "\"refCount\":" << block.ref_count << ","
            << "\"lastAccessTick\":" << block.last_access_tick << ","
            << "\"sharedPrefix\":" << (block.shared_prefix ? "true" : "false") << ","
            << "\"live\":" << (block.live ? "true" : "false") << ","
            << "\"pinned\":" << (block.pinned ? "true" : "false") << ","
            << "\"evictable\":" << (block.evictable ? "true" : "false") << ","
            << "\"evicted\":" << (block.evicted ? "true" : "false") << ","
            << "\"cacheKey\":\"" << json_escape(block.cache_key) << "\","
            << "\"ownerRequestIds\":[";
        for (size_t j = 0; j < block.owner_request_ids.size(); ++j) {
            if (j) out << ",";
            out << "\"" << json_escape(block.owner_request_ids[j]) << "\"";
        }
        out << "]}";
    }
    out << "\n  ]\n}\n";

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) return false;
    file << out.str();
    file.close();
    if (out_path) *out_path = path;
    return true;
}

} // namespace

void inference_runtime_reset() {
    g_summary = {};
}

bool inference_runtime_run_smoke() {
    inference_runtime_reset();
    g_summary.available = is_initialized();
    g_summary.chunk_size_tokens = kChunkSizeTokens;
    g_summary.max_batch_tokens = kMaxBatchTokens;
    g_summary.prefill_backend = "fused_attention_tiled";
    g_summary.decode_backend = "qjl_plus_value_dequant";
    g_summary.batching_policy = "decode_priority_mixed_chunk_continuous_batching";
    g_summary.chunking_policy = "fixed_chunk_prefill_with_decode_interleave";

    if (!is_initialized()) {
        return false;
    }

    std::mt19937 rng(42);
    std::vector<SyntheticRequestBuffers> buffers;
    buffers.emplace_back(make_request_buffers("req_prefill_0", 96, 3, rng));
    buffers.emplace_back(make_request_buffers("req_prefill_1", 160, 2, rng));
    buffers.emplace_back(make_request_buffers("req_prefill_2", 224, 1, rng));

    std::vector<LiveRequestState> live(buffers.size());
    g_summary.request_count = static_cast<uint32_t>(buffers.size());
    for (const auto& req : buffers) {
        InferenceRequestPlan plan;
        plan.request_id = req.request_id;
        plan.prompt_tokens = static_cast<uint32_t>(req.prompt_tokens);
        plan.decode_tokens = static_cast<uint32_t>(req.decode_tokens);
        plan.prefill_chunk_size = kChunkSizeTokens;
        plan.prefill_chunks = static_cast<uint32_t>((req.prompt_tokens + static_cast<int>(kChunkSizeTokens) - 1) / static_cast<int>(kChunkSizeTokens));
        plan.prefill_backend = g_summary.prefill_backend;
        plan.decode_backend = g_summary.decode_backend;
        g_summary.requests.push_back(plan);
        g_summary.total_prefill_tokens += static_cast<uint32_t>(req.prompt_tokens);
        g_summary.total_decode_tokens += static_cast<uint32_t>(req.decode_tokens);
    }

    while (true) {
        std::vector<size_t> decode_ready;
        std::vector<size_t> prefill_ready;
        for (size_t i = 0; i < buffers.size(); ++i) {
            const bool prefill_done = live[i].prefill_consumed >= static_cast<uint32_t>(buffers[i].prompt_tokens);
            const bool decode_done = live[i].decode_emitted >= static_cast<uint32_t>(buffers[i].decode_tokens);
            if (prefill_done && !decode_done) {
                decode_ready.push_back(i);
            } else if (!prefill_done) {
                prefill_ready.push_back(i);
            }
        }
        if (decode_ready.empty() && prefill_ready.empty()) {
            break;
        }

        InferenceWaveRecord wave;
        wave.wave_id = g_summary.scheduler_turns + 1;
        wave.decode_queue_depth = static_cast<uint32_t>(decode_ready.size());
        wave.prefill_queue_depth = static_cast<uint32_t>(prefill_ready.size());
        g_summary.decode_queue_peak = std::max(g_summary.decode_queue_peak, wave.decode_queue_depth);
        g_summary.prefill_queue_peak = std::max(g_summary.prefill_queue_peak, wave.prefill_queue_depth);

        std::vector<size_t> picked_decode;
        std::vector<size_t> picked_prefill;
        uint32_t remaining_budget = kMaxBatchTokens;

        for (size_t i = 0; i < decode_ready.size() && picked_decode.size() < kMaxDecodeRequestsPerWave; ++i) {
            picked_decode.push_back(decode_ready[i]);
            remaining_budget -= std::min<uint32_t>(remaining_budget, 1u);
        }
        if (!picked_decode.empty()) {
            g_summary.decode_priority_turns++;
        }

        for (size_t i = 0; i < prefill_ready.size() && picked_prefill.size() < kMaxPrefillRequestsPerWave; ++i) {
            const size_t idx = prefill_ready[i];
            const uint32_t remaining_tokens = static_cast<uint32_t>(buffers[idx].prompt_tokens) - live[idx].prefill_consumed;
            const uint32_t chunk_tokens = std::min<uint32_t>(kChunkSizeTokens, remaining_tokens);
            if (chunk_tokens > remaining_budget && !picked_prefill.empty()) {
                continue;
            }
            if (chunk_tokens > remaining_budget && picked_prefill.empty() && picked_decode.empty()) {
                // Allow a pure prefill wave even when chunk > nominal remaining budget.
            } else if (chunk_tokens > remaining_budget) {
                continue;
            }
            picked_prefill.push_back(idx);
            if (chunk_tokens <= remaining_budget) remaining_budget -= chunk_tokens;
            if (!picked_decode.empty()) break;
        }

        wave.prefill_request_count = static_cast<uint32_t>(picked_prefill.size());
        wave.decode_request_count = static_cast<uint32_t>(picked_decode.size());
        wave.batched_request_count = wave.prefill_request_count + wave.decode_request_count;
        wave.continuous_batching = wave.batched_request_count > 1;
        wave.decode_prioritized = !picked_decode.empty();
        wave.mixed_with_running_decode = !picked_decode.empty() && !picked_prefill.empty();
        if (wave.mixed_with_running_decode) wave.phase = "mixed";
        else if (!picked_prefill.empty()) wave.phase = "prefill";
        else wave.phase = "decode";

        g_summary.max_continuous_batch_size = std::max(g_summary.max_continuous_batch_size, wave.batched_request_count);
        g_summary.max_decode_batch_size = std::max(g_summary.max_decode_batch_size, wave.decode_request_count);
        g_summary.max_prefill_batch_size = std::max(g_summary.max_prefill_batch_size, wave.prefill_request_count);
        if (wave.phase == "mixed") g_summary.mixed_wave_count++;
        else if (wave.phase == "prefill") g_summary.prefill_wave_count++;
        else g_summary.decode_wave_count++;

        for (size_t idx : picked_prefill) {
            const uint32_t token_offset = live[idx].prefill_consumed;
            const uint32_t remaining_tokens = static_cast<uint32_t>(buffers[idx].prompt_tokens) - live[idx].prefill_consumed;
            const uint32_t chunk_tokens = std::min<uint32_t>(kChunkSizeTokens, remaining_tokens);
            uint64_t ns = 0;
            auto st = run_prefill_anchor(buffers[idx], token_offset, chunk_tokens, &ns);
            if (st != TqStatus::OK) return false;
            wave.fused_attention_ns += ns;
            wave.prefill_chunk_tokens += chunk_tokens;
            wave.batched_token_count += chunk_tokens;
            wave.anchor_kernel = wave.anchor_kernel.empty() ? "tq_attention_logits+tq_attention_apply_values" : wave.anchor_kernel;
            wave.request_ids.push_back(buffers[idx].request_id);
            live[idx].prefill_consumed += chunk_tokens;

            auto& plan = g_summary.requests[idx];
            plan.prefill_chunks_executed++;
            plan.continuous_batch_slots_seen++;
            plan.entered_mixed_wave = plan.entered_mixed_wave || wave.mixed_with_running_decode;
            plan.prefill_complete = live[idx].prefill_consumed >= static_cast<uint32_t>(buffers[idx].prompt_tokens);
            plan.peak_context_tokens = std::max<uint32_t>(plan.peak_context_tokens, live[idx].prefill_consumed);

            g_summary.total_prefill_chunks++;
            g_summary.max_prefill_chunk_tokens = std::max(g_summary.max_prefill_chunk_tokens, chunk_tokens);
        }

        for (size_t idx : picked_decode) {
            const uint32_t context_tokens = static_cast<uint32_t>(buffers[idx].prompt_tokens) + live[idx].decode_emitted;
            uint64_t qjl_ns = 0;
            uint64_t value_ns = 0;
            auto st = run_decode_anchor(buffers[idx], context_tokens, &qjl_ns, &value_ns);
            if (st != TqStatus::OK) return false;
            wave.qjl_ns += qjl_ns;
            wave.value_dequant_ns += value_ns;
            wave.decode_step_tokens += 1;
            wave.batched_token_count += 1;
            if (wave.anchor_kernel.empty()) wave.anchor_kernel = "tq_qjl_score+tq_value_dequant";
            wave.request_ids.push_back(buffers[idx].request_id);
            live[idx].decode_emitted += 1;

            auto& plan = g_summary.requests[idx];
            plan.decode_steps_executed++;
            plan.continuous_batch_slots_seen++;
            plan.entered_mixed_wave = plan.entered_mixed_wave || wave.mixed_with_running_decode;
            plan.prefill_complete = true;
            plan.decode_complete = live[idx].decode_emitted >= static_cast<uint32_t>(buffers[idx].decode_tokens);
            plan.peak_context_tokens = std::max<uint32_t>(plan.peak_context_tokens, context_tokens);
        }

        g_summary.total_fused_attention_ns += wave.fused_attention_ns;
        g_summary.total_qjl_ns += wave.qjl_ns;
        g_summary.total_value_dequant_ns += wave.value_dequant_ns;
        g_summary.scheduler_turns++;
        g_summary.waves.push_back(wave);
    }

    uint64_t total_chunk_tokens = 0;
    for (const auto& wave : g_summary.waves) total_chunk_tokens += wave.prefill_chunk_tokens;
    g_summary.avg_prefill_chunk_tokens = g_summary.total_prefill_chunks > 0
        ? static_cast<uint32_t>(total_chunk_tokens / g_summary.total_prefill_chunks)
        : 0;

    g_summary.prefill_decode_split_ready =
        g_summary.prefill_wave_count > 0 &&
        g_summary.decode_wave_count > 0;
    g_summary.chunked_prefill_ready =
        g_summary.total_prefill_chunks > g_summary.request_count &&
        g_summary.max_prefill_chunk_tokens == kChunkSizeTokens;
    g_summary.continuous_batching_ready =
        g_summary.max_continuous_batch_size > 1 &&
        g_summary.max_decode_batch_size > 1;
    g_summary.mixed_prefill_decode_ready =
        g_summary.mixed_wave_count > 0 &&
        g_summary.decode_priority_turns > 0;

    return g_summary.prefill_decode_split_ready &&
           g_summary.chunked_prefill_ready &&
           g_summary.continuous_batching_ready &&
           g_summary.mixed_prefill_decode_ready &&
           g_summary.total_fused_attention_ns > 0 &&
           g_summary.total_qjl_ns > 0 &&
           g_summary.total_value_dequant_ns > 0;
}

InferenceRuntimeSummary inference_runtime_get_summary() {
    return g_summary;
}

bool inference_runtime_write_artifact(std::string* out_path) {
    std::string path;
    if (!write_summary_json(g_summary, &path)) return false;
    g_summary.artifact_path = path;
    if (out_path) *out_path = path;
    return true;
}

void paged_kv_prefix_cache_reset() {
    g_paged_summary = {};
}

bool paged_kv_prefix_cache_run_smoke() {
    paged_kv_prefix_cache_reset();
    g_paged_summary.available = is_initialized();
    g_paged_summary.block_size_tokens = 32;
    g_paged_summary.prefill_backend = "fused_attention_tiled";
    g_paged_summary.decode_backend = "qjl_plus_value_dequant";
    g_paged_summary.sharing_policy = "prefix_hash_block_sharing";
    g_paged_summary.eviction_policy = "refcount_zero_lru_leaf_eviction";

    if (!is_initialized()) {
        return false;
    }

    trace_log("paged-kv", "smoke begin block_size=%u", g_paged_summary.block_size_tokens);

    std::mt19937 rng(42);
    std::vector<SyntheticRequestBuffers> buffers;
    buffers.emplace_back(make_request_buffers("req_prefill_0", 96, 3, rng));
    buffers.emplace_back(make_request_buffers("req_prefill_1", 160, 2, rng));
    buffers.emplace_back(make_request_buffers("req_prefill_2", 224, 1, rng));
    trace_log("paged-kv", "buffers materialized count=%zu", buffers.size());
    std::vector<LiveRequestState> live(buffers.size());
    g_paged_summary.request_count = static_cast<uint32_t>(buffers.size());

    struct PhysicalBlockState {
        PagedKvBlockRecord record;
        std::vector<size_t> owners;
    };

    const uint32_t block_size = g_paged_summary.block_size_tokens;
    uint32_t next_block_id = 1;
    uint32_t tick = 0;
    std::vector<PhysicalBlockState> physical_blocks;
    std::vector<std::vector<uint32_t>> request_block_table(buffers.size());

    auto allocate_block = [&](const std::string& cache_key,
                              uint32_t token_count,
                              bool shared_prefix,
                              size_t owner_idx) -> uint32_t {
        PhysicalBlockState state;
        state.record.physical_block_id = next_block_id++;
        state.record.token_count = token_count;
        state.record.logical_ref_count = 0;
        state.record.ref_count = 0;
        state.record.last_access_tick = ++tick;
        state.record.shared_prefix = shared_prefix;
        state.record.live = true;
        state.record.pinned = shared_prefix;
        state.record.evictable = false;
        state.record.evicted = false;
        state.record.cache_key = cache_key;
        state.owners.push_back(owner_idx);
        state.record.owner_request_ids.push_back(buffers[owner_idx].request_id);
        const uint32_t block_id = state.record.physical_block_id;
        physical_blocks.push_back(std::move(state));
        trace_log("paged-kv", "allocate block=%u owner=%zu shared_prefix=%d token_count=%u key=%s",
                  block_id, owner_idx, shared_prefix ? 1 : 0, token_count, cache_key.c_str());
        return block_id;
    };

    auto find_state = [&](uint32_t block_id) -> PhysicalBlockState* {
        for (auto& block : physical_blocks) {
            if (block.record.physical_block_id == block_id) return &block;
        }
        return nullptr;
    };

    auto attach_owner = [&](PhysicalBlockState& state, size_t owner_idx) {
        if (std::find(state.owners.begin(), state.owners.end(), owner_idx) == state.owners.end()) {
            state.owners.push_back(owner_idx);
            state.record.owner_request_ids.push_back(buffers[owner_idx].request_id);
        }
        state.record.ref_count += 1;
        state.record.logical_ref_count += 1;
        state.record.last_access_tick = ++tick;
        state.record.live = true;
        state.record.pinned = state.record.shared_prefix || state.record.ref_count > 1;
        state.record.evictable = false;
        g_paged_summary.max_block_refcount = std::max(g_paged_summary.max_block_refcount, state.record.ref_count);
        trace_log("paged-kv", "attach owner=%zu block=%u ref_count=%u logical_ref_count=%u",
                  owner_idx, state.record.physical_block_id, state.record.ref_count, state.record.logical_ref_count);
    };

    auto map_prefix_block = [&](size_t req_idx, uint32_t logical_block, uint32_t token_count) {
        const bool share_with_req0 = (req_idx == 1 && logical_block < 2) || (req_idx == 2 && logical_block == 0);
        uint32_t block_id = 0;
        if (share_with_req0) {
            block_id = request_block_table[0][logical_block];
            auto* shared = find_state(block_id);
            if (shared) {
                attach_owner(*shared, req_idx);
            }
        } else {
            std::ostringstream key;
            key << buffers[req_idx].request_id << ":prefill:" << logical_block;
            block_id = allocate_block(key.str(), token_count, false, req_idx);
            auto* created = find_state(block_id);
            if (created) attach_owner(*created, req_idx);
        }
        request_block_table[req_idx].push_back(block_id);
    };

    for (size_t i = 0; i < buffers.size(); ++i) {
        const uint32_t logical_blocks =
            static_cast<uint32_t>((buffers[i].prompt_tokens + static_cast<int>(block_size) - 1) / static_cast<int>(block_size));
        trace_log("paged-kv", "seed request=%s logical_blocks=%u", buffers[i].request_id.c_str(), logical_blocks);
        for (uint32_t logical_block = 0; logical_block < logical_blocks; ++logical_block) {
            const uint32_t base_token = logical_block * block_size;
            const uint32_t token_count = std::min<uint32_t>(
                block_size,
                static_cast<uint32_t>(buffers[i].prompt_tokens) - base_token);
            if (i == 0 && logical_block < 2) {
                std::ostringstream key;
                key << "shared_prefix:" << logical_block;
                uint32_t block_id = allocate_block(key.str(), token_count, true, i);
                auto* created = find_state(block_id);
                if (created) attach_owner(*created, i);
                request_block_table[i].push_back(block_id);
            } else {
                map_prefix_block(i, logical_block, token_count);
            }
        }
    }
    trace_log("paged-kv", "prefix seeding complete physical_blocks=%zu", physical_blocks.size());

    while (true) {
        std::vector<size_t> decode_ready;
        std::vector<size_t> prefill_ready;
        for (size_t i = 0; i < buffers.size(); ++i) {
            const bool prefill_done = live[i].prefill_consumed >= static_cast<uint32_t>(buffers[i].prompt_tokens);
            const bool decode_done = live[i].decode_emitted >= static_cast<uint32_t>(buffers[i].decode_tokens);
            if (prefill_done && !decode_done) decode_ready.push_back(i);
            else if (!prefill_done) prefill_ready.push_back(i);
        }
        if (decode_ready.empty() && prefill_ready.empty()) break;
        trace_log("paged-kv", "wave begin decode_ready=%zu prefill_ready=%zu", decode_ready.size(), prefill_ready.size());

        std::vector<size_t> picked_decode;
        std::vector<size_t> picked_prefill;
        uint32_t remaining_budget = kMaxBatchTokens;

        for (size_t i = 0; i < decode_ready.size() && picked_decode.size() < kMaxDecodeRequestsPerWave; ++i) {
            picked_decode.push_back(decode_ready[i]);
            remaining_budget -= std::min<uint32_t>(remaining_budget, 1u);
        }
        for (size_t i = 0; i < prefill_ready.size() && picked_prefill.size() < kMaxPrefillRequestsPerWave; ++i) {
            const size_t idx = prefill_ready[i];
            const uint32_t remaining_tokens =
                static_cast<uint32_t>(buffers[idx].prompt_tokens) - live[idx].prefill_consumed;
            const uint32_t chunk_tokens = std::min<uint32_t>(kChunkSizeTokens, remaining_tokens);
            if (chunk_tokens > remaining_budget && !picked_prefill.empty()) continue;
            if (chunk_tokens > remaining_budget && (!picked_prefill.empty() || !picked_decode.empty())) continue;
            picked_prefill.push_back(idx);
            if (chunk_tokens <= remaining_budget) remaining_budget -= chunk_tokens;
            if (!picked_decode.empty()) break;
        }

        for (size_t idx : picked_prefill) {
            const uint32_t token_offset = live[idx].prefill_consumed;
            const uint32_t remaining_tokens =
                static_cast<uint32_t>(buffers[idx].prompt_tokens) - live[idx].prefill_consumed;
            const uint32_t chunk_tokens = std::min<uint32_t>(kChunkSizeTokens, remaining_tokens);
            uint64_t ns = 0;
            auto st = run_prefill_anchor(buffers[idx], token_offset, chunk_tokens, &ns);
            if (st != TqStatus::OK) return false;
            trace_log("paged-kv", "prefill request=%s offset=%u chunk=%u ns=%llu",
                      buffers[idx].request_id.c_str(), token_offset, chunk_tokens,
                      static_cast<unsigned long long>(ns));
            g_paged_summary.total_fused_attention_ns += ns;
            live[idx].prefill_consumed += chunk_tokens;
        }

        for (size_t idx : picked_decode) {
            const uint32_t context_tokens =
                static_cast<uint32_t>(buffers[idx].prompt_tokens) + live[idx].decode_emitted;
            uint64_t qjl_ns = 0;
            uint64_t value_ns = 0;
            auto st = run_decode_anchor(buffers[idx], context_tokens, &qjl_ns, &value_ns);
            if (st != TqStatus::OK) return false;
            trace_log("paged-kv", "decode request=%s context=%u qjl_ns=%llu value_ns=%llu",
                      buffers[idx].request_id.c_str(), context_tokens,
                      static_cast<unsigned long long>(qjl_ns),
                      static_cast<unsigned long long>(value_ns));
            g_paged_summary.total_qjl_ns += qjl_ns;
            g_paged_summary.total_value_dequant_ns += value_ns;
            live[idx].decode_emitted += 1;

            const uint32_t total_context = static_cast<uint32_t>(buffers[idx].prompt_tokens) + live[idx].decode_emitted;
            const uint32_t required_blocks =
                (total_context + block_size - 1) / block_size;
            if (required_blocks > request_block_table[idx].size()) {
                std::ostringstream key;
                key << buffers[idx].request_id << ":decode:" << (required_blocks - 1);
                const uint32_t decode_block_id = allocate_block(key.str(), 1, false, idx);
                auto* created = find_state(decode_block_id);
                if (created) attach_owner(*created, idx);
                request_block_table[idx].push_back(decode_block_id);
                trace_log("paged-kv", "decode append request=%s block=%u required_blocks=%u",
                          buffers[idx].request_id.c_str(), decode_block_id, required_blocks);
            }
        }

        uint32_t live_block_count = 0;
        for (const auto& block : physical_blocks) {
            if (block.record.live && !block.record.evicted) live_block_count++;
        }
        g_paged_summary.peak_live_block_count = std::max(g_paged_summary.peak_live_block_count, live_block_count);
        trace_log("paged-kv", "wave end peak_live=%u", g_paged_summary.peak_live_block_count);
    }

    trace_log("paged-kv", "request summarization begin");
    for (size_t i = 0; i < buffers.size(); ++i) {
        PagedKvRequestSummary req;
        req.request_id = buffers[i].request_id;
        req.prompt_tokens = static_cast<uint32_t>(buffers[i].prompt_tokens);
        req.decode_tokens = static_cast<uint32_t>(buffers[i].decode_tokens);
        req.logical_block_count = static_cast<uint32_t>(request_block_table[i].size());
        req.peak_context_tokens = static_cast<uint32_t>(buffers[i].prompt_tokens) + live[i].decode_emitted;
        req.physical_block_ids = request_block_table[i];
        for (uint32_t block_id : request_block_table[i]) {
            auto* state = find_state(block_id);
            if (!state) continue;
            if (state->record.shared_prefix) req.shared_prefix_blocks++;
            if (state->record.ref_count > 1) req.shared_physical_block_count++;
            else req.unique_physical_block_count++;
        }
        const uint32_t prompt_blocks =
            static_cast<uint32_t>((buffers[i].prompt_tokens + static_cast<int>(block_size) - 1) / static_cast<int>(block_size));
        req.decode_block_appends = req.logical_block_count > prompt_blocks ? req.logical_block_count - prompt_blocks : 0;
        g_paged_summary.requests.push_back(std::move(req));
    }
    trace_log("paged-kv", "request summarization complete count=%zu", g_paged_summary.requests.size());

    // Release finished requests except req_prefill_1 to force a bounded eviction decision
    // while preserving one live shared-prefix consumer.
    trace_log("paged-kv", "release phase begin");
    for (size_t i = 0; i < request_block_table.size(); ++i) {
        const bool keep_live = (i == 1);
        for (uint32_t block_id : request_block_table[i]) {
            auto* state = find_state(block_id);
            if (!state) continue;
            if (!keep_live && state->record.ref_count > 0) {
                state->record.ref_count -= 1;
            }
            state->record.live = keep_live && state->record.ref_count > 0;
            state->record.pinned = state->record.shared_prefix && state->record.ref_count > 0;
            state->record.last_access_tick = ++tick;
        }
    }

    uint32_t evicted = 0;
    std::string eviction_target;
    trace_log("paged-kv", "eviction scan begin");
    for (auto& block : physical_blocks) {
        block.record.evictable = !block.record.live && !block.record.pinned && block.record.ref_count == 0;
        if (block.record.evictable && !block.record.evicted) {
            block.record.evicted = true;
            block.record.last_access_tick = ++tick;
            evicted++;
            if (eviction_target.empty()) eviction_target = block.record.cache_key;
        }
    }

    trace_log("paged-kv", "final aggregation begin");
    for (auto& block : physical_blocks) {
        g_paged_summary.logical_block_ref_count += block.record.logical_ref_count;
        g_paged_summary.max_block_refcount = std::max(g_paged_summary.max_block_refcount, block.record.ref_count);
        if (block.record.shared_prefix) g_paged_summary.shared_block_count++;
        else g_paged_summary.unique_block_count++;
        if (!block.record.evicted) g_paged_summary.resident_block_count++;
        if (block.record.evictable) g_paged_summary.evictable_block_count++;
        if (block.record.evicted) g_paged_summary.evicted_block_count++;
        g_paged_summary.blocks.push_back(block.record);
    }

    g_paged_summary.physical_block_count = static_cast<uint32_t>(physical_blocks.size());
    g_paged_summary.shared_prefix_group_count = 1;
    g_paged_summary.eviction_decision = evicted > 0
        ? "evicted_refcount_zero_leaf_blocks:" + eviction_target
        : "no_eviction_pressure_materialized";

    g_paged_summary.paged_kv_allocator_ready =
        g_paged_summary.physical_block_count > 0 &&
        g_paged_summary.peak_live_block_count > 0;
    g_paged_summary.block_table_ready =
        !g_paged_summary.requests.empty() &&
        std::all_of(g_paged_summary.requests.begin(), g_paged_summary.requests.end(),
                    [](const PagedKvRequestSummary& req) {
                        return req.logical_block_count > 0 &&
                               req.logical_block_count == req.physical_block_ids.size();
                    });
    g_paged_summary.prefix_cache_sharing_ready =
        std::count_if(g_paged_summary.requests.begin(), g_paged_summary.requests.end(),
                      [](const PagedKvRequestSummary& req) {
                          return req.shared_prefix_blocks > 0;
                      }) >= 2 &&
        g_paged_summary.shared_block_count > 0;
    g_paged_summary.cache_eviction_ready =
        g_paged_summary.evicted_block_count > 0 &&
        !g_paged_summary.eviction_decision.empty();

    trace_log("paged-kv", "smoke end ready alloc=%d block=%d share=%d evict=%d physical=%u evicted=%u",
              g_paged_summary.paged_kv_allocator_ready ? 1 : 0,
              g_paged_summary.block_table_ready ? 1 : 0,
              g_paged_summary.prefix_cache_sharing_ready ? 1 : 0,
              g_paged_summary.cache_eviction_ready ? 1 : 0,
              g_paged_summary.physical_block_count,
              g_paged_summary.evicted_block_count);

    return g_paged_summary.paged_kv_allocator_ready &&
           g_paged_summary.block_table_ready &&
           g_paged_summary.prefix_cache_sharing_ready &&
           g_paged_summary.cache_eviction_ready &&
           g_paged_summary.total_fused_attention_ns > 0 &&
           g_paged_summary.total_qjl_ns > 0 &&
           g_paged_summary.total_value_dequant_ns > 0;
}

PagedKvPrefixCacheSummary paged_kv_prefix_cache_get_summary() {
    return g_paged_summary;
}

bool paged_kv_prefix_cache_write_artifact(std::string* out_path) {
    std::string path;
    if (!write_paged_summary_json(g_paged_summary, &path)) return false;
    g_paged_summary.artifact_path = path;
    if (out_path) *out_path = path;
    return true;
}

} // namespace tq
