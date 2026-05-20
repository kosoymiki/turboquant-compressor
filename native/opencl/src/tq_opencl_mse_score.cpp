/**
 * TurboQuant OpenCL — MSE score kernel host code.
 * Provides both OpenCL dispatch and CPU reference for parity testing.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "tq_kernel_tune.h"
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

    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    size_t packed_size = (size_t)input.tokens * input.packed_stride;
    size_t norms_size = (size_t)input.tokens * sizeof(float);
    size_t query_size = (size_t)input.dim * sizeof(float);
    size_t centroids_size = (size_t)(1 << input.bits) * sizeof(float);
    size_t scores_size = (size_t)input.tokens * sizeof(float);

    cl_mem d_query = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, query_size, (void*)input.query_rotated, &err);
    cl_mem d_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, packed_size, (void*)input.packed_indices, &err);
    cl_mem d_norms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, norms_size, (void*)input.norms, &err);
    cl_mem d_centroids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, centroids_size, (void*)input.centroids, &err);
    cl_mem d_scores = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, scores_size, nullptr, &err);

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_query);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_packed);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_norms);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_centroids);
    clSetKernelArg(k, 4, sizeof(cl_mem), &d_scores);
    clSetKernelArg(k, 5, sizeof(int), &input.tokens);
    clSetKernelArg(k, 6, sizeof(int), &input.dim);
    clSetKernelArg(k, 7, sizeof(int), &input.bits);
    clSetKernelArg(k, 8, sizeof(int), &input.packed_stride);

    size_t local_size = local_work[0];
    if (local_size == 0) local_size = 64;
    size_t global_size = (kernel_name == std::string("tq_mse_score_tiled"))
        ? ((size_t)input.tokens * local_size)
        : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
    if (kernel_name == std::string("tq_mse_score_tiled")) {
        clSetKernelArg(k, 9, local_size * sizeof(cl_float), nullptr);
    }
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_query); clReleaseMemObject(d_packed);
        clReleaseMemObject(d_norms); clReleaseMemObject(d_centroids);
        clReleaseMemObject(d_scores);
        mse_score_cpu_reference(input, scores);
        return TqStatus::OK;
    }

    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, scores_size, scores, 0, nullptr, nullptr);

    clReleaseEvent(event);
    clReleaseMemObject(d_query); clReleaseMemObject(d_packed);
    clReleaseMemObject(d_norms); clReleaseMemObject(d_centroids);
    clReleaseMemObject(d_scores);

    return TqStatus::OK;
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
    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    size_t packed_size = (size_t)input.tokens * input.packed_stride;
    size_t norms_size = (size_t)input.tokens * sizeof(float);
    size_t query_size = (size_t)input.dim * sizeof(float);
    size_t centroids_size = (size_t)(1 << input.bits) * sizeof(float);
    size_t scores_size = (size_t)input.tokens * sizeof(float);

    cl_mem d_query = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, query_size, (void*)input.query_rotated, &err);
    cl_mem d_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, packed_size, (void*)input.packed_indices, &err);
    cl_mem d_norms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, norms_size, (void*)input.norms, &err);
    cl_mem d_centroids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, centroids_size, (void*)input.centroids, &err);
    cl_mem d_scores = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, scores_size, nullptr, &err);

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_query);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_packed);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_norms);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_centroids);
    clSetKernelArg(k, 4, sizeof(cl_mem), &d_scores);
    clSetKernelArg(k, 5, sizeof(int), &input.tokens);
    clSetKernelArg(k, 6, sizeof(int), &input.dim);
    clSetKernelArg(k, 7, sizeof(int), &input.bits);
    clSetKernelArg(k, 8, sizeof(int), &input.packed_stride);

    size_t local_size = local_work[0];
    if (local_size == 0) local_size = 64;
    size_t global_size = (kernel_name == std::string("tq_mse_score_tiled"))
        ? ((size_t)input.tokens * local_size)
        : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
    if (kernel_name == std::string("tq_mse_score_tiled")) {
        clSetKernelArg(k, 9, local_size * sizeof(cl_float), nullptr);
    }
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_query); clReleaseMemObject(d_packed);
        clReleaseMemObject(d_norms); clReleaseMemObject(d_centroids);
        clReleaseMemObject(d_scores);
        mse_score_cpu_reference(input, scores);
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

    clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, scores_size, scores, 0, nullptr, nullptr);

    clReleaseEvent(event);
    clReleaseMemObject(d_query); clReleaseMemObject(d_packed);
    clReleaseMemObject(d_norms); clReleaseMemObject(d_centroids);
    clReleaseMemObject(d_scores);

    return TqStatus::OK;
}

} // namespace tq
