/**
 * TurboQuant OpenCL — fused attention host code.
 * CPU reference: online softmax over compressed KV, no full dequant buffer.
 * Real OpenCL dispatch with profiling.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include <CL/cl.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace tq {

static inline uint32_t fused_extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    return (word >> bit_off) & ((1u << bits) - 1u);
}

static inline uint32_t fused_popcount32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return ((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24;
}

void fused_attention_cpu_reference(const FusedAttentionInput& input) {
    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int val_packed_stride = (input.value.dim * input.value.bits + 7) / 8;
    int n_groups = (input.value.dim + input.value.group_size - 1) / input.value.group_size;

    std::vector<float> scores(tokens);
    mse_score_cpu_reference(input.mse, scores.data());

    QjlScoreInput qjl_in = input.qjl;
    qjl_in.base_scores = scores.data();
    qjl_score_cpu_reference(qjl_in);

    for (int n = 0; n < tokens; n++) scores[n] *= input.sm_scale;

    float m_i = -1e30f;
    float l_i = 0.0f;
    std::vector<float> acc(dim, 0.0f);

    for (int n = 0; n < tokens; n++) {
        float s = scores[n];
        float m_new = std::max(m_i, s);
        float alpha = std::exp(m_i - m_new);
        float p = std::exp(s - m_new);
        l_i = l_i * alpha + p;

        for (int j = 0; j < dim; j++) acc[j] *= alpha;

        const uint8_t* v_packed = input.value.packed_values + n * val_packed_stride;
        for (int j = 0; j < dim; j++) {
            uint32_t raw = fused_extract_bits(v_packed, j, input.value.bits);
            int g = j / input.value.group_size;
            float val = (float)raw * input.value.scales[n * n_groups + g]
                       + input.value.zeros[n * n_groups + g];
            acc[j] += p * val;
        }

        m_i = m_new;
    }

    if (l_i > 0.0f) {
        for (int j = 0; j < dim; j++) input.output[j] = acc[j] / l_i;
    } else {
        std::memset(input.output, 0, dim * sizeof(float));
    }
}

TqStatus fused_attention_opencl(const FusedAttentionInput& input) {
    if (!is_initialized() || !get_kernel("tq_fused_attention")) {
        fused_attention_cpu_reference(input);
        return TqStatus::OK;
    }

    cl_kernel k = get_kernel("tq_fused_attention");
    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int key_packed_stride = input.mse.packed_stride;
    int val_packed_stride = (input.value.dim * input.value.bits + 7) / 8;
    int n_groups = (input.value.dim + input.value.group_size - 1) / input.value.group_size;
    int proj_words = (input.qjl.projections + 31) / 32;

    // Allocate buffers
    cl_mem d_query = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        dim * sizeof(float), (void*)input.mse.query_rotated, &err);
    cl_mem d_qsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        proj_words * sizeof(uint32_t), (void*)input.qjl.query_signs, &err);
    cl_mem d_key_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * key_packed_stride, (void*)input.mse.packed_indices, &err);
    cl_mem d_key_norms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens * sizeof(float), (void*)input.mse.norms, &err);
    cl_mem d_rsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * proj_words * sizeof(uint32_t), (void*)input.qjl.residual_signs, &err);
    cl_mem d_rnorms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens * sizeof(float), (void*)input.qjl.residual_norms, &err);
    cl_mem d_centroids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (1 << input.mse.bits) * sizeof(float), (void*)input.mse.centroids, &err);
    cl_mem d_val_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * val_packed_stride, (void*)input.value.packed_values, &err);
    cl_mem d_val_scales = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * n_groups * sizeof(float), (void*)input.value.scales, &err);
    cl_mem d_val_zeros = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * n_groups * sizeof(float), (void*)input.value.zeros, &err);

    // Output buffer — zero-initialized
    std::vector<float> zeros(dim, 0.0f);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        dim * sizeof(float), zeros.data(), &err);

    int key_bits = input.mse.bits;
    int val_bits = input.value.bits;
    int group_size = input.value.group_size;
    int projections = input.qjl.projections;
    float qjl_scale = input.qjl.qjl_scale;
    float sm_scale = input.sm_scale;

    int arg = 0;
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_query);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_qsigns);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_key_packed);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_key_norms);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_rsigns);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_rnorms);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_centroids);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_val_packed);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_val_scales);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_val_zeros);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_output);
    clSetKernelArg(k, arg++, sizeof(int), &tokens);
    clSetKernelArg(k, arg++, sizeof(int), &dim);
    clSetKernelArg(k, arg++, sizeof(int), &key_bits);
    clSetKernelArg(k, arg++, sizeof(int), &key_packed_stride);
    clSetKernelArg(k, arg++, sizeof(int), &val_bits);
    clSetKernelArg(k, arg++, sizeof(int), &val_packed_stride);
    clSetKernelArg(k, arg++, sizeof(int), &group_size);
    clSetKernelArg(k, arg++, sizeof(int), &n_groups);
    clSetKernelArg(k, arg++, sizeof(int), &projections);
    clSetKernelArg(k, arg++, sizeof(int), &proj_words);
    clSetKernelArg(k, arg++, sizeof(float), &qjl_scale);
    clSetKernelArg(k, arg++, sizeof(float), &sm_scale);

    // Fused kernel runs as single work-group
    size_t local_size = 1; // Reference: single thread
    size_t global_size = 1;
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_query); clReleaseMemObject(d_qsigns);
        clReleaseMemObject(d_key_packed); clReleaseMemObject(d_key_norms);
        clReleaseMemObject(d_rsigns); clReleaseMemObject(d_rnorms);
        clReleaseMemObject(d_centroids); clReleaseMemObject(d_val_packed);
        clReleaseMemObject(d_val_scales); clReleaseMemObject(d_val_zeros);
        clReleaseMemObject(d_output);
        fused_attention_cpu_reference(input);
        return TqStatus::OK;
    }

    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, dim * sizeof(float), input.output, 0, nullptr, nullptr);

    clReleaseEvent(event);
    clReleaseMemObject(d_query); clReleaseMemObject(d_qsigns);
    clReleaseMemObject(d_key_packed); clReleaseMemObject(d_key_norms);
    clReleaseMemObject(d_rsigns); clReleaseMemObject(d_rnorms);
    clReleaseMemObject(d_centroids); clReleaseMemObject(d_val_packed);
    clReleaseMemObject(d_val_scales); clReleaseMemObject(d_val_zeros);
    clReleaseMemObject(d_output);

    return TqStatus::OK;
}

TqStatus fused_attention_opencl_profiled(const FusedAttentionInput& input, uint64_t* kernel_ns) {
    if (!is_initialized() || !get_kernel("tq_fused_attention")) {
        fused_attention_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    // Same as fused_attention_opencl but captures profiling event
    cl_kernel k = get_kernel("tq_fused_attention");
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
    cl_mem d_qsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        proj_words * sizeof(uint32_t), (void*)input.qjl.query_signs, &err);
    cl_mem d_key_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * key_packed_stride, (void*)input.mse.packed_indices, &err);
    cl_mem d_key_norms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens * sizeof(float), (void*)input.mse.norms, &err);
    cl_mem d_rsigns = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * proj_words * sizeof(uint32_t), (void*)input.qjl.residual_signs, &err);
    cl_mem d_rnorms = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens * sizeof(float), (void*)input.qjl.residual_norms, &err);
    cl_mem d_centroids = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (1 << input.mse.bits) * sizeof(float), (void*)input.mse.centroids, &err);
    cl_mem d_val_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * val_packed_stride, (void*)input.value.packed_values, &err);
    cl_mem d_val_scales = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * n_groups * sizeof(float), (void*)input.value.scales, &err);
    cl_mem d_val_zeros = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        (size_t)tokens * n_groups * sizeof(float), (void*)input.value.zeros, &err);

    std::vector<float> zeros_vec(dim, 0.0f);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        dim * sizeof(float), zeros_vec.data(), &err);

    int key_bits = input.mse.bits;
    int val_bits = input.value.bits;
    int group_size = input.value.group_size;
    int projections = input.qjl.projections;
    float qjl_scale = input.qjl.qjl_scale;
    float sm_scale = input.sm_scale;

    int arg = 0;
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_query);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_qsigns);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_key_packed);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_key_norms);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_rsigns);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_rnorms);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_centroids);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_val_packed);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_val_scales);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_val_zeros);
    clSetKernelArg(k, arg++, sizeof(cl_mem), &d_output);
    clSetKernelArg(k, arg++, sizeof(int), &tokens);
    clSetKernelArg(k, arg++, sizeof(int), &dim);
    clSetKernelArg(k, arg++, sizeof(int), &key_bits);
    clSetKernelArg(k, arg++, sizeof(int), &key_packed_stride);
    clSetKernelArg(k, arg++, sizeof(int), &val_bits);
    clSetKernelArg(k, arg++, sizeof(int), &val_packed_stride);
    clSetKernelArg(k, arg++, sizeof(int), &group_size);
    clSetKernelArg(k, arg++, sizeof(int), &n_groups);
    clSetKernelArg(k, arg++, sizeof(int), &projections);
    clSetKernelArg(k, arg++, sizeof(int), &proj_words);
    clSetKernelArg(k, arg++, sizeof(float), &qjl_scale);
    clSetKernelArg(k, arg++, sizeof(float), &sm_scale);

    size_t local_size = 1;
    size_t global_size = 1;
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_query); clReleaseMemObject(d_qsigns);
        clReleaseMemObject(d_key_packed); clReleaseMemObject(d_key_norms);
        clReleaseMemObject(d_rsigns); clReleaseMemObject(d_rnorms);
        clReleaseMemObject(d_centroids); clReleaseMemObject(d_val_packed);
        clReleaseMemObject(d_val_scales); clReleaseMemObject(d_val_zeros);
        clReleaseMemObject(d_output);
        fused_attention_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    clWaitForEvents(1, &event);

    cl_ulong t_start = 0, t_end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, nullptr);
    *kernel_ns = (uint64_t)(t_end - t_start);

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, dim * sizeof(float), input.output, 0, nullptr, nullptr);

    clReleaseEvent(event);
    clReleaseMemObject(d_query); clReleaseMemObject(d_qsigns);
    clReleaseMemObject(d_key_packed); clReleaseMemObject(d_key_norms);
    clReleaseMemObject(d_rsigns); clReleaseMemObject(d_rnorms);
    clReleaseMemObject(d_centroids); clReleaseMemObject(d_val_packed);
    clReleaseMemObject(d_val_scales); clReleaseMemObject(d_val_zeros);
    clReleaseMemObject(d_output);

    return TqStatus::OK;
}

} // namespace tq
