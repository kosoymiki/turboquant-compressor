/**
 * TurboQuant OpenCL — tiled fused attention (two-pass).
 * Pass 1: tq_attention_logits — parallel over tokens (MSE + QJL scores)
 * Pass 2: tq_attention_apply_values — parallel over dim (softmax + value accum)
 * No full V materialization. Event profiling on both passes.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include <CL/cl.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace tq {

TqStatus fused_attention_tiled_opencl(const FusedAttentionInput& input) {
    if (!is_initialized()) {
        fused_attention_cpu_reference(input);
        return TqStatus::OK;
    }

    cl_kernel k_logits = get_kernel("tq_attention_logits");
    cl_kernel k_values = get_kernel("tq_attention_apply_values");
    if (!k_logits || !k_values) {
        fused_attention_cpu_reference(input);
        return TqStatus::OK;
    }

    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int key_packed_stride = input.mse.packed_stride;
    int val_packed_stride = (input.value.dim * input.value.bits + 7) / 8;
    int n_groups = (input.value.dim + input.value.group_size - 1) / input.value.group_size;
    int proj_words = (input.qjl.projections + 31) / 32;

    // Buffers for pass 1 (logits)
    cl_mem d_query = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        dim * sizeof(float), (void*)input.mse.query_rotated, &err);
    cl_mem d_key_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * key_packed_stride, (void*)input.mse.packed_indices, &err);
    cl_mem d_key_norms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens * sizeof(float), (void*)input.mse.norms, &err);
    cl_mem d_centroids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (1 << input.mse.bits) * sizeof(float), (void*)input.mse.centroids, &err);
    cl_mem d_qsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        proj_words * sizeof(uint32_t), (void*)input.qjl.query_signs, &err);
    cl_mem d_rsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * proj_words * sizeof(uint32_t), (void*)input.qjl.residual_signs, &err);
    cl_mem d_rnorms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens * sizeof(float), (void*)input.qjl.residual_norms, &err);

    // Intermediate scores buffer (device only)
    cl_mem d_scores = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        tokens * sizeof(float), nullptr, &err);

    // Buffers for pass 2 (values)
    cl_mem d_val_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * val_packed_stride, (void*)input.value.packed_values, &err);
    cl_mem d_val_scales = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * n_groups * sizeof(float), (void*)input.value.scales, &err);
    cl_mem d_val_zeros = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * n_groups * sizeof(float), (void*)input.value.zeros, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        dim * sizeof(float), nullptr, &err);

    // Pass 1: logits kernel args
    int key_bits = input.mse.bits;
    int projections = input.qjl.projections;
    float qjl_scale = input.qjl.qjl_scale;
    float sm_scale = input.sm_scale;

    int arg = 0;
    clSetKernelArg(k_logits, arg++, sizeof(cl_mem), &d_query);
    clSetKernelArg(k_logits, arg++, sizeof(cl_mem), &d_key_packed);
    clSetKernelArg(k_logits, arg++, sizeof(cl_mem), &d_key_norms);
    clSetKernelArg(k_logits, arg++, sizeof(cl_mem), &d_centroids);
    clSetKernelArg(k_logits, arg++, sizeof(cl_mem), &d_qsigns);
    clSetKernelArg(k_logits, arg++, sizeof(cl_mem), &d_rsigns);
    clSetKernelArg(k_logits, arg++, sizeof(cl_mem), &d_rnorms);
    clSetKernelArg(k_logits, arg++, sizeof(cl_mem), &d_scores);
    clSetKernelArg(k_logits, arg++, sizeof(int), &tokens);
    clSetKernelArg(k_logits, arg++, sizeof(int), &dim);
    clSetKernelArg(k_logits, arg++, sizeof(int), &key_bits);
    clSetKernelArg(k_logits, arg++, sizeof(int), &key_packed_stride);
    clSetKernelArg(k_logits, arg++, sizeof(int), &projections);
    clSetKernelArg(k_logits, arg++, sizeof(int), &proj_words);
    clSetKernelArg(k_logits, arg++, sizeof(float), &qjl_scale);
    clSetKernelArg(k_logits, arg++, sizeof(float), &sm_scale);

    // Launch pass 1: one work-item per token
    size_t global_tokens = (size_t)tokens;
    err = clEnqueueNDRangeKernel(queue, k_logits, 1, nullptr, &global_tokens, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Cleanup and fallback
        clReleaseMemObject(d_query); clReleaseMemObject(d_key_packed);
        clReleaseMemObject(d_key_norms); clReleaseMemObject(d_centroids);
        clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
        clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);
        clReleaseMemObject(d_val_packed); clReleaseMemObject(d_val_scales);
        clReleaseMemObject(d_val_zeros); clReleaseMemObject(d_output);
        fused_attention_cpu_reference(input);
        return TqStatus::OK;
    }

    // Pass 2: apply values kernel args
    int val_bits = input.value.bits;
    int group_size = input.value.group_size;

    arg = 0;
    clSetKernelArg(k_values, arg++, sizeof(cl_mem), &d_scores);
    clSetKernelArg(k_values, arg++, sizeof(cl_mem), &d_val_packed);
    clSetKernelArg(k_values, arg++, sizeof(cl_mem), &d_val_scales);
    clSetKernelArg(k_values, arg++, sizeof(cl_mem), &d_val_zeros);
    clSetKernelArg(k_values, arg++, sizeof(cl_mem), &d_output);
    clSetKernelArg(k_values, arg++, sizeof(int), &tokens);
    clSetKernelArg(k_values, arg++, sizeof(int), &dim);
    clSetKernelArg(k_values, arg++, sizeof(int), &val_bits);
    clSetKernelArg(k_values, arg++, sizeof(int), &val_packed_stride);
    clSetKernelArg(k_values, arg++, sizeof(int), &group_size);
    clSetKernelArg(k_values, arg++, sizeof(int), &n_groups);

    // Launch pass 2: one work-item per dim
    size_t global_dim = (size_t)dim;
    err = clEnqueueNDRangeKernel(queue, k_values, 1, nullptr, &global_dim, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_query); clReleaseMemObject(d_key_packed);
        clReleaseMemObject(d_key_norms); clReleaseMemObject(d_centroids);
        clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
        clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);
        clReleaseMemObject(d_val_packed); clReleaseMemObject(d_val_scales);
        clReleaseMemObject(d_val_zeros); clReleaseMemObject(d_output);
        fused_attention_cpu_reference(input);
        return TqStatus::OK;
    }

    // Read output
    clFinish(queue);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, dim * sizeof(float), input.output, 0, nullptr, nullptr);

    // Cleanup
    clReleaseMemObject(d_query); clReleaseMemObject(d_key_packed);
    clReleaseMemObject(d_key_norms); clReleaseMemObject(d_centroids);
    clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
    clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);
    clReleaseMemObject(d_val_packed); clReleaseMemObject(d_val_scales);
    clReleaseMemObject(d_val_zeros); clReleaseMemObject(d_output);

    return TqStatus::OK;
}

TqStatus fused_attention_tiled_opencl_profiled(const FusedAttentionInput& input, uint64_t* kernel_ns) {
    if (!is_initialized() || !get_kernel("tq_attention_logits") || !get_kernel("tq_attention_apply_values")) {
        fused_attention_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    cl_kernel k_logits = get_kernel("tq_attention_logits");
    cl_kernel k_values = get_kernel("tq_attention_apply_values");
    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int key_packed_stride = input.mse.packed_stride;
    int val_packed_stride = (input.value.dim * input.value.bits + 7) / 8;
    int n_groups = (input.value.dim + input.value.group_size - 1) / input.value.group_size;
    int proj_words = (input.qjl.projections + 31) / 32;

    cl_mem d_query = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        dim * sizeof(float), (void*)input.mse.query_rotated, &err);
    cl_mem d_key_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * key_packed_stride, (void*)input.mse.packed_indices, &err);
    cl_mem d_key_norms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens * sizeof(float), (void*)input.mse.norms, &err);
    cl_mem d_centroids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (1 << input.mse.bits) * sizeof(float), (void*)input.mse.centroids, &err);
    cl_mem d_qsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        proj_words * sizeof(uint32_t), (void*)input.qjl.query_signs, &err);
    cl_mem d_rsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * proj_words * sizeof(uint32_t), (void*)input.qjl.residual_signs, &err);
    cl_mem d_rnorms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens * sizeof(float), (void*)input.qjl.residual_norms, &err);
    cl_mem d_scores = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        tokens * sizeof(float), nullptr, &err);
    cl_mem d_val_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * val_packed_stride, (void*)input.value.packed_values, &err);
    cl_mem d_val_scales = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * n_groups * sizeof(float), (void*)input.value.scales, &err);
    cl_mem d_val_zeros = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * n_groups * sizeof(float), (void*)input.value.zeros, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
        dim * sizeof(float), nullptr, &err);

    int key_bits = input.mse.bits;
    int projections = input.qjl.projections;
    float qjl_scale = input.qjl.qjl_scale;
    float sm_scale = input.sm_scale;
    int val_bits = input.value.bits;
    int group_size = input.value.group_size;

    int a = 0;
    clSetKernelArg(k_logits, a++, sizeof(cl_mem), &d_query);
    clSetKernelArg(k_logits, a++, sizeof(cl_mem), &d_key_packed);
    clSetKernelArg(k_logits, a++, sizeof(cl_mem), &d_key_norms);
    clSetKernelArg(k_logits, a++, sizeof(cl_mem), &d_centroids);
    clSetKernelArg(k_logits, a++, sizeof(cl_mem), &d_qsigns);
    clSetKernelArg(k_logits, a++, sizeof(cl_mem), &d_rsigns);
    clSetKernelArg(k_logits, a++, sizeof(cl_mem), &d_rnorms);
    clSetKernelArg(k_logits, a++, sizeof(cl_mem), &d_scores);
    clSetKernelArg(k_logits, a++, sizeof(int), &tokens);
    clSetKernelArg(k_logits, a++, sizeof(int), &dim);
    clSetKernelArg(k_logits, a++, sizeof(int), &key_bits);
    clSetKernelArg(k_logits, a++, sizeof(int), &key_packed_stride);
    clSetKernelArg(k_logits, a++, sizeof(int), &projections);
    clSetKernelArg(k_logits, a++, sizeof(int), &proj_words);
    clSetKernelArg(k_logits, a++, sizeof(float), &qjl_scale);
    clSetKernelArg(k_logits, a++, sizeof(float), &sm_scale);

    size_t global_tokens = (size_t)tokens;
    cl_event ev1;
    err = clEnqueueNDRangeKernel(queue, k_logits, 1, nullptr, &global_tokens, nullptr, 0, nullptr, &ev1);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_query); clReleaseMemObject(d_key_packed);
        clReleaseMemObject(d_key_norms); clReleaseMemObject(d_centroids);
        clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
        clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);
        clReleaseMemObject(d_val_packed); clReleaseMemObject(d_val_scales);
        clReleaseMemObject(d_val_zeros); clReleaseMemObject(d_output);
        fused_attention_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    a = 0;
    clSetKernelArg(k_values, a++, sizeof(cl_mem), &d_scores);
    clSetKernelArg(k_values, a++, sizeof(cl_mem), &d_val_packed);
    clSetKernelArg(k_values, a++, sizeof(cl_mem), &d_val_scales);
    clSetKernelArg(k_values, a++, sizeof(cl_mem), &d_val_zeros);
    clSetKernelArg(k_values, a++, sizeof(cl_mem), &d_output);
    clSetKernelArg(k_values, a++, sizeof(int), &tokens);
    clSetKernelArg(k_values, a++, sizeof(int), &dim);
    clSetKernelArg(k_values, a++, sizeof(int), &val_bits);
    clSetKernelArg(k_values, a++, sizeof(int), &val_packed_stride);
    clSetKernelArg(k_values, a++, sizeof(int), &group_size);
    clSetKernelArg(k_values, a++, sizeof(int), &n_groups);

    size_t global_dim = (size_t)dim;
    cl_event ev2;
    err = clEnqueueNDRangeKernel(queue, k_values, 1, nullptr, &global_dim, nullptr, 1, &ev1, &ev2);
    if (err != CL_SUCCESS) {
        clWaitForEvents(1, &ev1);
        clReleaseEvent(ev1);
        clReleaseMemObject(d_query); clReleaseMemObject(d_key_packed);
        clReleaseMemObject(d_key_norms); clReleaseMemObject(d_centroids);
        clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
        clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);
        clReleaseMemObject(d_val_packed); clReleaseMemObject(d_val_scales);
        clReleaseMemObject(d_val_zeros); clReleaseMemObject(d_output);
        fused_attention_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    clWaitForEvents(1, &ev2);

    // Total time = logits start to values end
    cl_ulong t1_start = 0, t2_end = 0;
    clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_START, sizeof(t1_start), &t1_start, nullptr);
    clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_END, sizeof(t2_end), &t2_end, nullptr);
    *kernel_ns = (uint64_t)(t2_end - t1_start);

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, dim * sizeof(float), input.output, 0, nullptr, nullptr);

    clReleaseEvent(ev1); clReleaseEvent(ev2);
    clReleaseMemObject(d_query); clReleaseMemObject(d_key_packed);
    clReleaseMemObject(d_key_norms); clReleaseMemObject(d_centroids);
    clReleaseMemObject(d_qsigns); clReleaseMemObject(d_rsigns);
    clReleaseMemObject(d_rnorms); clReleaseMemObject(d_scores);
    clReleaseMemObject(d_val_packed); clReleaseMemObject(d_val_scales);
    clReleaseMemObject(d_val_zeros); clReleaseMemObject(d_output);

    return TqStatus::OK;
}

} // namespace tq
