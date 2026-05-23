/**
 * TurboQuant OpenCL — MSE score kernel host code.
 * Provides both OpenCL dispatch and CPU reference for parity testing.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "tq_buffer.h"
#include "tq_kernel_tune.h"
#include "../include/tq_trace.h"
#include <CL/cl.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

namespace tq {

namespace {
const char* select_mse_kernel_name(int dim, size_t local_size) {
    return (dim >= (int)local_size * 2) ? "tq_mse_score_tiled" : "tq_mse_score";
}

size_t choose_local_size_from_runtime_query(cl_kernel kernel, size_t fallback_local_size, size_t global_size) {
    size_t suggested_local_size = 0;
    if (global_size > 0 &&
        query_kernel_suggested_local_work_size(kernel, 1, nullptr, &global_size, &suggested_local_size) &&
        suggested_local_size > 0) {
        return suggested_local_size;
    }
    return fallback_local_size;
}
}

// --- CPU reference implementation (always available) ---

static inline uint32_t extract_bits_cpu(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) {
        word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    }
    return (word >> bit_off) & ((1u << bits) - 1u);
}

void mse_score_cpu_reference(const MseScoreInput& input, float* scores) {
    for (int n = 0; n < input.tokens; n++) {
        const uint8_t* packed = input.packed_indices + n * input.packed_stride;
        float dot = 0.0f;
        for (int j = 0; j < input.dim; j++) {
            uint32_t idx = extract_bits_cpu(packed, j, input.bits);
            dot += input.query_rotated[j] * input.centroids[idx];
        }
        scores[n] = dot * input.norms[n];
    }
}

// --- OpenCL dispatch ---

TqStatus mse_score_opencl(const MseScoreInput& input, float* scores) {
    KernelTuneParams tune = get_kernel_tune("tq_mse_score");
    size_t local_work[3];
    get_local_work_size(tune, local_work);
    const char* kernel_name = select_mse_kernel_name(input.dim, local_work[0]);
    if (!is_initialized()) {
        mse_score_cpu_reference(input, scores);
        return TqStatus::OK;
    }

    cl_kernel k = get_kernel(kernel_name);
    if (!k) {
        mse_score_cpu_reference(input, scores);
        return TqStatus::OK;
    }

    cl_command_queue queue = get_queue();

    try {
        TqBuffer<float> d_query((size_t)input.dim, CL_MEM_READ_ONLY);
        TqBuffer<uint8_t> d_packed((size_t)input.tokens * input.packed_stride, CL_MEM_READ_ONLY);
        TqBuffer<float> d_norms((size_t)input.tokens, CL_MEM_READ_ONLY);
        TqBuffer<float> d_centroids((size_t)(1 << input.bits), CL_MEM_READ_ONLY);
        TqBuffer<float> d_scores((size_t)input.tokens, CL_MEM_WRITE_ONLY);
        trace_log("mse-profiled", "shape tokens=%d dim=%d bits=%d local_hint=%zu kernel=%s svm_scores=%d",
                  input.tokens, input.dim, input.bits, local_work[0], kernel_name, d_scores.uses_svm() ? 1 : 0);

        d_query.write_to_device(queue, input.query_rotated);
        d_packed.write_to_device(queue, input.packed_indices);
        d_norms.write_to_device(queue, input.norms);
        d_centroids.write_to_device(queue, input.centroids);

        if (d_query.bind_kernel_arg(k, 0) != TqStatus::OK ||
            d_packed.bind_kernel_arg(k, 1) != TqStatus::OK ||
            d_norms.bind_kernel_arg(k, 2) != TqStatus::OK ||
            d_centroids.bind_kernel_arg(k, 3) != TqStatus::OK ||
            d_scores.bind_kernel_arg(k, 4) != TqStatus::OK) {
            mse_score_cpu_reference(input, scores);
            return TqStatus::OK;
        }
        clSetKernelArg(k, 5, sizeof(int), &input.tokens);
        clSetKernelArg(k, 6, sizeof(int), &input.dim);
        clSetKernelArg(k, 7, sizeof(int), &input.bits);
        clSetKernelArg(k, 8, sizeof(int), &input.packed_stride);

        size_t local_size = local_work[0];
        if (local_size == 0) local_size = 64;
        size_t global_size = (kernel_name == std::string("tq_mse_score_tiled"))
            ? ((size_t)input.tokens * local_size)
            : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
        local_size = choose_local_size_from_runtime_query(k, local_size, global_size);
        if (local_size == 0) local_size = 64;
        global_size = (kernel_name == std::string("tq_mse_score_tiled"))
            ? ((size_t)input.tokens * local_size)
            : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
        if (kernel_name == std::string("tq_mse_score_tiled")) {
            clSetKernelArg(k, 9, local_size * sizeof(cl_float), nullptr);
        }
        cl_event event;
        cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
        if (err != CL_SUCCESS) {
            mse_score_cpu_reference(input, scores);
            return TqStatus::OK;
        }

        clWaitForEvents(1, &event);
        d_scores.read_from_device(queue, scores);

        clReleaseEvent(event);
        return TqStatus::OK;
    } catch (...) {
        mse_score_cpu_reference(input, scores);
        return TqStatus::OK;
    }
}

// --- Profiled dispatch (returns kernel time in nanoseconds) ---

TqStatus mse_score_opencl_profiled(const MseScoreInput& input, float* scores, uint64_t* kernel_ns) {
    KernelTuneParams tune = get_kernel_tune("tq_mse_score");
    size_t local_work[3];
    get_local_work_size(tune, local_work);
    const char* kernel_name = select_mse_kernel_name(input.dim, local_work[0]);
    if (!is_initialized() || !get_kernel(kernel_name)) {
        mse_score_cpu_reference(input, scores);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    cl_kernel k = get_kernel(kernel_name);
    cl_command_queue queue = get_queue();

    try {
        TqBuffer<float> d_query((size_t)input.dim, CL_MEM_READ_ONLY);
        TqBuffer<uint8_t> d_packed((size_t)input.tokens * input.packed_stride, CL_MEM_READ_ONLY);
        TqBuffer<float> d_norms((size_t)input.tokens, CL_MEM_READ_ONLY);
        TqBuffer<float> d_centroids((size_t)(1 << input.bits), CL_MEM_READ_ONLY);
        TqBuffer<float> d_scores((size_t)input.tokens, CL_MEM_WRITE_ONLY);

        d_query.write_to_device(queue, input.query_rotated);
        d_packed.write_to_device(queue, input.packed_indices);
        d_norms.write_to_device(queue, input.norms);
        d_centroids.write_to_device(queue, input.centroids);

        if (d_query.bind_kernel_arg(k, 0) != TqStatus::OK ||
            d_packed.bind_kernel_arg(k, 1) != TqStatus::OK ||
            d_norms.bind_kernel_arg(k, 2) != TqStatus::OK ||
            d_centroids.bind_kernel_arg(k, 3) != TqStatus::OK ||
            d_scores.bind_kernel_arg(k, 4) != TqStatus::OK) {
            mse_score_cpu_reference(input, scores);
            *kernel_ns = 0;
            return TqStatus::OK;
        }
        clSetKernelArg(k, 5, sizeof(int), &input.tokens);
        clSetKernelArg(k, 6, sizeof(int), &input.dim);
        clSetKernelArg(k, 7, sizeof(int), &input.bits);
        clSetKernelArg(k, 8, sizeof(int), &input.packed_stride);

        size_t local_size = local_work[0];
        if (local_size == 0) local_size = 64;
        size_t global_size = (kernel_name == std::string("tq_mse_score_tiled"))
            ? ((size_t)input.tokens * local_size)
            : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
        local_size = choose_local_size_from_runtime_query(k, local_size, global_size);
        if (local_size == 0) local_size = 64;
        global_size = (kernel_name == std::string("tq_mse_score_tiled"))
            ? ((size_t)input.tokens * local_size)
            : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
        if (kernel_name == std::string("tq_mse_score_tiled")) {
            clSetKernelArg(k, 9, local_size * sizeof(cl_float), nullptr);
        }
        cl_event event;
        cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
        if (err != CL_SUCCESS) {
            mse_score_cpu_reference(input, scores);
            *kernel_ns = 0;
            return TqStatus::OK;
        }

        auto wall_start = std::chrono::steady_clock::now();
        clWaitForEvents(1, &event);
        auto wall_end = std::chrono::steady_clock::now();

        if (profiling_enabled()) {
            cl_ulong t_queue = 0, t_start = 0, t_end = 0;
            if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(t_queue), &t_queue, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr) == CL_SUCCESS) {
                *kernel_ns = (uint64_t)(t_end - t_start);
                trace_log("mse-profiled", "event_ns queued=%llu start=%llu end=%llu dur=%llu wall=%llu",
                          static_cast<unsigned long long>(t_queue),
                          static_cast<unsigned long long>(t_start),
                          static_cast<unsigned long long>(t_end),
                          static_cast<unsigned long long>(t_end - t_start),
                          static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count()));
            } else {
                *kernel_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count();
            }
        } else {
            *kernel_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count();
        }

        d_scores.read_from_device(queue, scores);

        clReleaseEvent(event);
        return TqStatus::OK;
    } catch (...) {
        mse_score_cpu_reference(input, scores);
        *kernel_ns = 0;
        return TqStatus::OK;
    }
}

} // namespace tq
