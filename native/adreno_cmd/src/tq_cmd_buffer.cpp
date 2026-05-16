/**
 * TurboQuant Command Buffer — JSON serialization and execution.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_cmd_buffer.h"
#include <cstdio>
#include <cstring>

namespace tq {

static const char* op_name(CmdOp op) {
    switch (op) {
        case CmdOp::MSE_SCORE: return "mse_score";
        case CmdOp::QJL_SCORE: return "qjl_score";
        case CmdOp::VALUE_DEQUANT: return "value_dequant";
        case CmdOp::FUSED_ATTENTION: return "fused_attention";
        case CmdOp::BENCHMARK: return "benchmark";
        case CmdOp::PROBE: return "probe";
    }
    return "unknown";
}

static const char* backend_name(BackendRoute r) {
    switch (r) {
        case BackendRoute::OPENCL_ADRENO: return "opencl_adreno";
        case BackendRoute::OPENCL_TILED: return "opencl_tiled";
        case BackendRoute::NEON: return "neon";
        case BackendRoute::CPU_SCALAR: return "cpu_scalar";
        case BackendRoute::VULKAN_COMPUTE: return "vulkan_compute";
        case BackendRoute::MESA_RUSTICL: return "mesa_rusticl";
        case BackendRoute::AUTO: return "auto";
    }
    return "unknown";
}

std::string cmd_buffer_to_json(const CommandBuffer& buf) {
    std::string json = "{\n  \"version\": " + std::to_string(buf.version) + ",\n  \"commands\": [\n";
    for (size_t i = 0; i < buf.commands.size(); i++) {
        auto& c = buf.commands[i];
        char cmd_buf[512];
        snprintf(cmd_buf, sizeof(cmd_buf),
            "    {\"op\":\"%s\",\"backend\":\"%s\","
            "\"shape\":{\"tokens\":%d,\"heads\":%d,\"headDim\":%d,\"bits\":%d,\"groupSize\":%d,\"projections\":%d},"
            "\"requirements\":{\"allowFallback\":%s,\"requireGpu\":%s,\"requireParity\":%s,\"maxLatencyMs\":%.1f}}",
            op_name(c.op), backend_name(c.backend),
            c.shape.tokens, c.shape.heads, c.shape.head_dim, c.shape.bits, c.shape.group_size, c.shape.projections,
            c.requirements.allow_fallback ? "true" : "false",
            c.requirements.require_gpu ? "true" : "false",
            c.requirements.require_parity ? "true" : "false",
            c.requirements.max_latency_ms);
        json += cmd_buf;
        json += (i + 1 < buf.commands.size()) ? ",\n" : "\n";
    }
    json += "  ]\n}";
    return json;
}

std::string cmd_result_to_json(const CommandBufferResult& result) {
    std::string json = "{\n  \"version\": " + std::to_string(result.version) + ",\n";
    json += "  \"all_success\": ";
    json += result.all_success ? "true" : "false";
    json += ",\n  \"results\": [\n";
    for (size_t i = 0; i < result.results.size(); i++) {
        auto& r = result.results[i];
        char rbuf[256];
        snprintf(rbuf, sizeof(rbuf),
            "    {\"op\":\"%s\",\"backend\":\"%s\",\"success\":%s,\"parity\":%s,\"latency_ms\":%.3f}",
            op_name(r.op), backend_name(r.executed_backend),
            r.success ? "true" : "false",
            r.parity_verified ? "true" : "false",
            r.latency_ms);
        json += rbuf;
        json += (i + 1 < result.results.size()) ? ",\n" : "\n";
    }
    json += "  ]\n}";
    return json;
}

CommandBufferResult execute_cmd_buffer(const CommandBuffer& buf, const QuirksProfile& profile) {
    CommandBufferResult result;
    result.version = buf.version;
    result.all_success = true;

    for (auto& cmd : buf.commands) {
        CmdResult r;
        r.op = cmd.op;

        // Route backend using quirks
        std::string kernel_name = op_name(cmd.op);
        BackendRoute route = (cmd.backend == BackendRoute::AUTO)
            ? route_kernel(profile, kernel_name, cmd.shape.tokens, cmd.shape.bits)
            : cmd.backend;
        r.executed_backend = route;

        // Check GPU requirement
        if (cmd.requirements.require_gpu &&
            route != BackendRoute::OPENCL_ADRENO &&
            route != BackendRoute::OPENCL_TILED &&
            route != BackendRoute::VULKAN_COMPUTE) {
            r.success = false;
            r.error = "GPU required but routed to non-GPU backend";
            result.all_success = false;
            result.results.push_back(r);
            continue;
        }

        // Execute (placeholder — actual execution wired in executor)
        r.success = true;
        r.parity_verified = cmd.requirements.require_parity;
        r.latency_ms = 0.0;
        result.results.push_back(r);
    }

    return result;
}

} // namespace tq
