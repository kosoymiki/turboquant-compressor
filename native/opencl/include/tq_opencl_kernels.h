/**
 * TurboQuant OpenCL — kernel interfaces.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_opencl_types.h"
#include <cstdint>

namespace tq {

// Standard dispatch (falls back to CPU if context not initialized)
TqStatus mse_score_opencl(const MseScoreInput& input, float* scores);
TqStatus qjl_score_opencl(const QjlScoreInput& input);
TqStatus value_dequant_opencl(const ValueDequantInput& input, float* output);
TqStatus fused_attention_opencl(const FusedAttentionInput& input);

// Profiled dispatch (returns kernel execution time in nanoseconds via event profiling)
TqStatus mse_score_opencl_profiled(const MseScoreInput& input, float* scores, uint64_t* kernel_ns);
TqStatus qjl_score_opencl_profiled(const QjlScoreInput& input, uint64_t* kernel_ns);
TqStatus value_dequant_opencl_profiled(const ValueDequantInput& input, float* output, uint64_t* kernel_ns);
TqStatus fused_attention_opencl_profiled(const FusedAttentionInput& input, uint64_t* kernel_ns);

// Tiled fused attention (two-pass: logits parallel over tokens, values parallel over dim)
TqStatus fused_attention_tiled_opencl(const FusedAttentionInput& input);
TqStatus fused_attention_tiled_opencl_profiled(const FusedAttentionInput& input, uint64_t* kernel_ns);

} // namespace tq
