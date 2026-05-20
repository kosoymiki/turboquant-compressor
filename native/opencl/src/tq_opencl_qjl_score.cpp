/**
 * TurboQuant OpenCL — QJL score kernel host code.
 * CPU reference + real OpenCL dispatch with profiling.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "tq_kernel_tune.h"
#include <CL/cl.h>
#include <chrono>
#include <cstdint>

namespace tq {

namespace {
const char* select_qjl_kernel_name(int proj_words, size_t local_size) {
    return (proj_words >= (int)local_size * 2) ? "tq_qjl_score_tiled" : "tq_qjl_score";
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
    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    size_t qsigns_size = (size_t)proj_words * sizeof(uint32_t);
    size_t rsigns_size = (size_t)input.tokens * proj_words * sizeof(uint32_t);
    size_t rnorms_size = (size_t)input.tokens * sizeof(float);
    size_t scores_size = (size_t)input.tokens * sizeof(float);

    cl_mem d_qsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qsigns_size, (void*)input.query_signs, &err);
    cl_mem d_rsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rsigns_size, (void*)input.residual_signs, &err);
    cl_mem d_rnorms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rnorms_size, (void*)input.residual_norms, &err);
    cl_mem d_scores = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, scores_size, (void*)input.base_scores, &err);

    int tokens = input.tokens;
    int projections = input.projections;
    float qjl_scale = input.qjl_scale;

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_qsigns);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_rsigns);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_rnorms);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_scores);
    clSetKernelArg(k, 4, sizeof(int), &tokens);
    clSetKernelArg(k, 5, sizeof(int), &projections);
    clSetKernelArg(k, 6, sizeof(float), &qjl_scale);
    clSetKernelArg(k, 7, sizeof(int), &proj_words);

    size_t local_size = local_work[0];
    if (local_size == 0) local_size = 64;
    size_t global_size = (kernel_name == std::string("tq_qjl_score_tiled"))
        ? ((size_t)input.tokens * local_size)
        : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
    cl_event event;
    if (kernel_name == std::string("tq_qjl_score_tiled")) {
        clSetKernelArg(k, 8, local_size * sizeof(cl_uint), nullptr);
    }
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
        clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);
        qjl_score_cpu_reference(input);
        return TqStatus::OK;
    }

    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, scores_size, input.base_scores, 0, nullptr, nullptr);

    clReleaseEvent(event);
    clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
    clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);

    return TqStatus::OK;
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
    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    size_t qsigns_size = (size_t)proj_words * sizeof(uint32_t);
    size_t rsigns_size = (size_t)input.tokens * proj_words * sizeof(uint32_t);
    size_t rnorms_size = (size_t)input.tokens * sizeof(float);
    size_t scores_size = (size_t)input.tokens * sizeof(float);

    cl_mem d_qsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, qsigns_size, (void*)input.query_signs, &err);
    cl_mem d_rsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rsigns_size, (void*)input.residual_signs, &err);
    cl_mem d_rnorms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rnorms_size, (void*)input.residual_norms, &err);
    cl_mem d_scores = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, scores_size, (void*)input.base_scores, &err);

    int tokens = input.tokens;
    int projections = input.projections;
    float qjl_scale = input.qjl_scale;

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_qsigns);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_rsigns);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_rnorms);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_scores);
    clSetKernelArg(k, 4, sizeof(int), &tokens);
    clSetKernelArg(k, 5, sizeof(int), &projections);
    clSetKernelArg(k, 6, sizeof(float), &qjl_scale);
    clSetKernelArg(k, 7, sizeof(int), &proj_words);

    size_t local_size = local_work[0];
    if (local_size == 0) local_size = 64;
    size_t global_size = (kernel_name == std::string("tq_qjl_score_tiled"))
        ? ((size_t)input.tokens * local_size)
        : (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
    cl_event event;
    if (kernel_name == std::string("tq_qjl_score_tiled")) {
        clSetKernelArg(k, 8, local_size * sizeof(cl_uint), nullptr);
    }
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
        clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);
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

    clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, scores_size, input.base_scores, 0, nullptr, nullptr);

    clReleaseEvent(event);
    clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
    clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);

    return TqStatus::OK;
}

} // namespace tq
