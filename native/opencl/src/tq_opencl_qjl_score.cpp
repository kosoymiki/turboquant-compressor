/**
 * TurboQuant OpenCL — QJL score kernel host code.
 * CPU reference + real OpenCL dispatch with profiling.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "tq_buffer.h"
#include "tq_kernel_tune.h"
#include <CL/cl.h>
#include <chrono>
#include <cstdint>

namespace tq {

namespace {
const char* select_qjl_kernel_name(int proj_words, size_t local_size) {
    return (proj_words >= (int)local_size * 2) ? "tq_qjl_score_tiled" : "tq_qjl_score";
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

static inline uint32_t popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return ((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24;
}

void qjl_score_cpu_reference(const QjlScoreInput& input) {
    int proj_words = (input.projections + 31) / 32;
    for (int n = 0; n < input.tokens; n++) {
        const uint32_t* my_signs = input.residual_signs + n * proj_words;
        uint32_t hamming = 0;
        for (int w = 0; w < proj_words; w++) {
            hamming += popcount32(input.query_signs[w] ^ my_signs[w]);
        }
        float sign_dot = (float)input.projections - 2.0f * (float)hamming;
        input.base_scores[n] += input.qjl_scale * input.residual_norms[n] * sign_dot;
    }
}

TqStatus qjl_score_opencl(const QjlScoreInput& input) {
    KernelTuneParams tune = get_kernel_tune("tq_qjl_score");
    size_t local_work[3];
    get_local_work_size(tune, local_work);
    int proj_words = (input.projections + 31) / 32;
    const char* kernel_name = select_qjl_kernel_name(proj_words, local_work[0]);
    if (!is_initialized() || !get_kernel(kernel_name)) {
        qjl_score_cpu_reference(input);
        return TqStatus::OK;
    }

    cl_kernel k = get_kernel(kernel_name);
    cl_command_queue queue = get_queue();
    const int tokens = input.tokens;
    const int projections = input.projections;
    const float qjl_scale = input.qjl_scale;

    try {
        TqBuffer<uint32_t> d_qsigns((size_t)proj_words, CL_MEM_READ_ONLY);
        TqBuffer<uint32_t> d_rsigns((size_t)input.tokens * proj_words, CL_MEM_READ_ONLY);
        TqBuffer<float> d_rnorms((size_t)input.tokens, CL_MEM_READ_ONLY);
        TqBuffer<float> d_scores((size_t)input.tokens, CL_MEM_READ_WRITE);

        d_qsigns.write_to_device(queue, input.query_signs);
        d_rsigns.write_to_device(queue, input.residual_signs);
        d_rnorms.write_to_device(queue, input.residual_norms);
        d_scores.write_to_device(queue, input.base_scores);

        if (d_qsigns.bind_kernel_arg(k, 0) != TqStatus::OK ||
            d_rsigns.bind_kernel_arg(k, 1) != TqStatus::OK ||
            d_rnorms.bind_kernel_arg(k, 2) != TqStatus::OK ||
            d_scores.bind_kernel_arg(k, 3) != TqStatus::OK) {
            qjl_score_cpu_reference(input);
            return TqStatus::OK;
        }
        clSetKernelArg(k, 4, sizeof(int), &tokens);
        clSetKernelArg(k, 5, sizeof(int), &projections);
        clSetKernelArg(k, 6, sizeof(float), &qjl_scale);
        clSetKernelArg(k, 7, sizeof(int), &proj_words);

        size_t local_size = local_work[0];
        if (local_size == 0) local_size = 64;
        size_t global_size = (kernel_name == std::string("tq_qjl_score_tiled"))
            ? ((size_t)input.tokens * local_size)
            : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
        local_size = choose_local_size_from_runtime_query(k, local_size, global_size);
        if (local_size == 0) local_size = 64;
        global_size = (kernel_name == std::string("tq_qjl_score_tiled"))
            ? ((size_t)input.tokens * local_size)
            : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
        cl_event event;
        if (kernel_name == std::string("tq_qjl_score_tiled")) {
            clSetKernelArg(k, 8, local_size * sizeof(cl_uint), nullptr);
        }
        cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
        if (err != CL_SUCCESS) {
            qjl_score_cpu_reference(input);
            return TqStatus::OK;
        }

        clWaitForEvents(1, &event);
        d_scores.read_from_device(queue, input.base_scores);

        clReleaseEvent(event);
        return TqStatus::OK;
    } catch (...) {
        qjl_score_cpu_reference(input);
        return TqStatus::OK;
    }
}

TqStatus qjl_score_opencl_profiled(const QjlScoreInput& input, uint64_t* kernel_ns) {
    KernelTuneParams tune = get_kernel_tune("tq_qjl_score");
    size_t local_work[3];
    get_local_work_size(tune, local_work);
    int proj_words = (input.projections + 31) / 32;
    const char* kernel_name = select_qjl_kernel_name(proj_words, local_work[0]);
    if (!is_initialized() || !get_kernel(kernel_name)) {
        qjl_score_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    cl_kernel k = get_kernel(kernel_name);
    cl_command_queue queue = get_queue();
    const int tokens = input.tokens;
    const int projections = input.projections;
    const float qjl_scale = input.qjl_scale;

    try {
        TqBuffer<uint32_t> d_qsigns((size_t)proj_words, CL_MEM_READ_ONLY);
        TqBuffer<uint32_t> d_rsigns((size_t)input.tokens * proj_words, CL_MEM_READ_ONLY);
        TqBuffer<float> d_rnorms((size_t)input.tokens, CL_MEM_READ_ONLY);
        TqBuffer<float> d_scores((size_t)input.tokens, CL_MEM_READ_WRITE);

        d_qsigns.write_to_device(queue, input.query_signs);
        d_rsigns.write_to_device(queue, input.residual_signs);
        d_rnorms.write_to_device(queue, input.residual_norms);
        d_scores.write_to_device(queue, input.base_scores);

        if (d_qsigns.bind_kernel_arg(k, 0) != TqStatus::OK ||
            d_rsigns.bind_kernel_arg(k, 1) != TqStatus::OK ||
            d_rnorms.bind_kernel_arg(k, 2) != TqStatus::OK ||
            d_scores.bind_kernel_arg(k, 3) != TqStatus::OK) {
            qjl_score_cpu_reference(input);
            *kernel_ns = 0;
            return TqStatus::OK;
        }
        clSetKernelArg(k, 4, sizeof(int), &tokens);
        clSetKernelArg(k, 5, sizeof(int), &projections);
        clSetKernelArg(k, 6, sizeof(float), &qjl_scale);
        clSetKernelArg(k, 7, sizeof(int), &proj_words);

        size_t local_size = local_work[0];
        if (local_size == 0) local_size = 64;
        size_t global_size = (kernel_name == std::string("tq_qjl_score_tiled"))
            ? ((size_t)input.tokens * local_size)
            : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
        local_size = choose_local_size_from_runtime_query(k, local_size, global_size);
        if (local_size == 0) local_size = 64;
        global_size = (kernel_name == std::string("tq_qjl_score_tiled"))
            ? ((size_t)input.tokens * local_size)
            : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
        cl_event event;
        if (kernel_name == std::string("tq_qjl_score_tiled")) {
            clSetKernelArg(k, 8, local_size * sizeof(cl_uint), nullptr);
        }
        cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
        if (err != CL_SUCCESS) {
            qjl_score_cpu_reference(input);
            *kernel_ns = 0;
            return TqStatus::OK;
        }

        auto wall_start = std::chrono::steady_clock::now();
        clWaitForEvents(1, &event);
        auto wall_end = std::chrono::steady_clock::now();

        if (profiling_enabled()) {
            cl_ulong t_start = 0, t_end = 0;
            if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr) == CL_SUCCESS) {
                *kernel_ns = (uint64_t)(t_end - t_start);
            } else {
                *kernel_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count();
            }
        } else {
            *kernel_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count();
        }

        d_scores.read_from_device(queue, input.base_scores);

        clReleaseEvent(event);
        return TqStatus::OK;
    } catch (...) {
        qjl_score_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }
}

} // namespace tq
