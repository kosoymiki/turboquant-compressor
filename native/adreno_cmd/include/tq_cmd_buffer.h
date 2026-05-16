/**
 * TurboQuant Command Buffer — unified compute dispatch protocol.
 * Vortek-inspired: one command schema for OpenCL/NEON/Vulkan/Mesa/Android service.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_quirks.h"
#include <cstdint>
#include <string>
#include <vector>

namespace tq {

enum class CmdOp : uint8_t {
    MSE_SCORE = 1,
    QJL_SCORE = 2,
    VALUE_DEQUANT = 3,
    FUSED_ATTENTION = 4,
    BENCHMARK = 5,
    PROBE = 6,
};

struct CmdShape {
    int tokens = 0;
    int heads = 0;
    int head_dim = 0;
    int bits = 0;
    int group_size = 32;
    int projections = 0;
};

struct CmdRequirements {
    bool allow_fallback = true;
    bool require_gpu = false;
    bool require_parity = true;
    float max_latency_ms = 0.0f; // 0 = no limit
};

struct Command {
    CmdOp op;
    BackendRoute backend = BackendRoute::AUTO;
    CmdShape shape;
    CmdRequirements requirements;
};

struct CmdResult {
    CmdOp op;
    BackendRoute executed_backend = BackendRoute::AUTO;
    bool success = false;
    bool parity_verified = false;
    double latency_ms = 0.0;
    std::string error;
};

struct CommandBuffer {
    int version = 1;
    std::vector<Command> commands;
};

struct CommandBufferResult {
    int version = 1;
    std::vector<CmdResult> results;
    bool all_success = false;
};

// Serialize/deserialize
std::string cmd_buffer_to_json(const CommandBuffer& buf);
CommandBuffer cmd_buffer_from_json(const std::string& json);
std::string cmd_result_to_json(const CommandBufferResult& result);

// Execute command buffer using quirks-based routing
CommandBufferResult execute_cmd_buffer(const CommandBuffer& buf, const QuirksProfile& profile);

} // namespace tq
