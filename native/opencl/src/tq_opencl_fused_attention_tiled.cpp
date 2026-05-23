/**
 * TurboQuant OpenCL — tiled fused attention (two-pass).
 * Pass 1: tq_attention_logits — parallel over tokens (MSE + QJL scores)
 * Pass 2: tq_attention_apply_values — parallel over dim (softmax + value accum)
 * No full V materialization. Event profiling on both passes.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "tq_buffer.h"
#include "../include/tq_kernel_tune.h"
#include "../include/tq_trace.h"
#include <CL/cl.h>
#include <chrono>
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

    cl_command_queue queue = get_queue();

    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int key_packed_stride = input.mse.packed_stride;
    int val_packed_stride = (input.value.dim * input.value.bits + 7) / 8;
    int n_groups = (input.value.dim + input.value.group_size - 1) / input.value.group_size;
    int proj_words = (input.qjl.projections + 31) / 32;

    try {
        TqBuffer<float> d_query((size_t)dim, CL_MEM_READ_ONLY);
        TqBuffer<uint8_t> d_key_packed((size_t)tokens * key_packed_stride, CL_MEM_READ_ONLY);
        TqBuffer<float> d_key_norms((size_t)tokens, CL_MEM_READ_ONLY);
        TqBuffer<float> d_centroids((size_t)(1 << input.mse.bits), CL_MEM_READ_ONLY);
        TqBuffer<uint32_t> d_qsigns((size_t)proj_words, CL_MEM_READ_ONLY);
        TqBuffer<uint32_t> d_rsigns((size_t)tokens * proj_words, CL_MEM_READ_ONLY);
        TqBuffer<float> d_rnorms((size_t)tokens, CL_MEM_READ_ONLY);
        TqBuffer<float> d_scores((size_t)tokens, CL_MEM_READ_WRITE);
        TqBuffer<uint8_t> d_val_packed((size_t)tokens * val_packed_stride, CL_MEM_READ_ONLY);
        TqBuffer<float> d_val_scales((size_t)tokens * n_groups, CL_MEM_READ_ONLY);
        TqBuffer<float> d_val_zeros((size_t)tokens * n_groups, CL_MEM_READ_ONLY);
        TqBuffer<float> d_output((size_t)dim, CL_MEM_WRITE_ONLY);

        d_query.write_to_device(queue, input.mse.query_rotated);
        d_key_packed.write_to_device(queue, input.mse.packed_indices);
        d_key_norms.write_to_device(queue, input.mse.norms);
        d_centroids.write_to_device(queue, input.mse.centroids);
        d_qsigns.write_to_device(queue, input.qjl.query_signs);
        d_rsigns.write_to_device(queue, input.qjl.residual_signs);
        d_rnorms.write_to_device(queue, input.qjl.residual_norms);
        d_val_packed.write_to_device(queue, input.value.packed_values);
        d_val_scales.write_to_device(queue, input.value.scales);
        d_val_zeros.write_to_device(queue, input.value.zeros);

        // Pass 1: logits kernel args
        int key_bits = input.mse.bits;
        int projections = input.qjl.projections;
        float qjl_scale = input.qjl.qjl_scale;
        float sm_scale = input.sm_scale;

        int arg = 0;
        if (d_query.bind_kernel_arg(k_logits, arg++) != TqStatus::OK ||
            d_key_packed.bind_kernel_arg(k_logits, arg++) != TqStatus::OK ||
            d_key_norms.bind_kernel_arg(k_logits, arg++) != TqStatus::OK ||
            d_centroids.bind_kernel_arg(k_logits, arg++) != TqStatus::OK ||
            d_qsigns.bind_kernel_arg(k_logits, arg++) != TqStatus::OK ||
            d_rsigns.bind_kernel_arg(k_logits, arg++) != TqStatus::OK ||
            d_rnorms.bind_kernel_arg(k_logits, arg++) != TqStatus::OK ||
            d_scores.bind_kernel_arg(k_logits, arg++) != TqStatus::OK) {
            fused_attention_cpu_reference(input);
            return TqStatus::OK;
        }
        clSetKernelArg(k_logits, arg++, sizeof(int), &tokens);
        clSetKernelArg(k_logits, arg++, sizeof(int), &dim);
        clSetKernelArg(k_logits, arg++, sizeof(int), &key_bits);
        clSetKernelArg(k_logits, arg++, sizeof(int), &key_packed_stride);
        clSetKernelArg(k_logits, arg++, sizeof(int), &projections);
        clSetKernelArg(k_logits, arg++, sizeof(int), &proj_words);
        clSetKernelArg(k_logits, arg++, sizeof(float), &qjl_scale);
        clSetKernelArg(k_logits, arg++, sizeof(float), &sm_scale);

        KernelTuneParams tune_logits = get_kernel_tune("tq_attention_logits");
        size_t local_logits = tune_logits.wg_size_x;
        size_t global_tokens = ((size_t)tokens + local_logits - 1) / local_logits * local_logits;

        if (tune_logits.local_mem_bytes > 0) {
            clSetKernelArg(k_logits, arg++, tune_logits.local_mem_bytes, nullptr);
        }

        cl_int err = clEnqueueNDRangeKernel(queue, k_logits, 1, nullptr, &global_tokens, &local_logits, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            fused_attention_cpu_reference(input);
            return TqStatus::OK;
        }

        int val_bits = input.value.bits;
        int group_size = input.value.group_size;
        arg = 0;
        if (d_scores.bind_kernel_arg(k_values, arg++) != TqStatus::OK ||
            d_val_packed.bind_kernel_arg(k_values, arg++) != TqStatus::OK ||
            d_val_scales.bind_kernel_arg(k_values, arg++) != TqStatus::OK ||
            d_val_zeros.bind_kernel_arg(k_values, arg++) != TqStatus::OK ||
            d_output.bind_kernel_arg(k_values, arg++) != TqStatus::OK) {
            fused_attention_cpu_reference(input);
            return TqStatus::OK;
        }
        clSetKernelArg(k_values, arg++, sizeof(int), &tokens);
        clSetKernelArg(k_values, arg++, sizeof(int), &dim);
        clSetKernelArg(k_values, arg++, sizeof(int), &val_bits);
        clSetKernelArg(k_values, arg++, sizeof(int), &val_packed_stride);
        clSetKernelArg(k_values, arg++, sizeof(int), &group_size);
        clSetKernelArg(k_values, arg++, sizeof(int), &n_groups);

        KernelTuneParams tune_values = get_kernel_tune("tq_attention_apply_values");
        size_t local_values = tune_values.wg_size_x;
        const size_t items_per_thread = tune_values.items_per_thread > 0 ? (size_t)tune_values.items_per_thread : 1u;
        size_t work_items = ((size_t)dim + items_per_thread - 1) / items_per_thread;
        size_t global_dim = ((size_t)work_items + local_values - 1) / local_values * local_values;

        if (tune_values.local_mem_bytes > 0) {
            clSetKernelArg(k_values, arg++, tune_values.local_mem_bytes, nullptr);
        }

        err = clEnqueueNDRangeKernel(queue, k_values, 1, nullptr, &global_dim, &local_values, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            fused_attention_cpu_reference(input);
            return TqStatus::OK;
        }

        clFinish(queue);
        d_output.read_from_device(queue, input.output);

        return TqStatus::OK;
    } catch (...) {
        fused_attention_cpu_reference(input);
        return TqStatus::OK;
    }
}

TqStatus fused_attention_tiled_opencl_profiled(const FusedAttentionInput& input, uint64_t* kernel_ns) {
    if (!is_initialized() || !get_kernel("tq_attention_logits") || !get_kernel("tq_attention_apply_values")) {
        fused_attention_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    cl_kernel k_logits = get_kernel("tq_attention_logits");
    cl_kernel k_values = get_kernel("tq_attention_apply_values");
    cl_command_queue queue = get_queue();

    int dim = input.mse.dim;
    int tokens = input.mse.tokens;
    int key_packed_stride = input.mse.packed_stride;
    int val_packed_stride = (input.value.dim * input.value.bits + 7) / 8;
    int n_groups = (input.value.dim + input.value.group_size - 1) / input.value.group_size;
    int proj_words = (input.qjl.projections + 31) / 32;

    try {
        TqBuffer<float> d_query((size_t)dim, CL_MEM_READ_ONLY);
        TqBuffer<uint8_t> d_key_packed((size_t)tokens * key_packed_stride, CL_MEM_READ_ONLY);
        TqBuffer<float> d_key_norms((size_t)tokens, CL_MEM_READ_ONLY);
        TqBuffer<float> d_centroids((size_t)(1 << input.mse.bits), CL_MEM_READ_ONLY);
        TqBuffer<uint32_t> d_qsigns((size_t)proj_words, CL_MEM_READ_ONLY);
        TqBuffer<uint32_t> d_rsigns((size_t)tokens * proj_words, CL_MEM_READ_ONLY);
        TqBuffer<float> d_rnorms((size_t)tokens, CL_MEM_READ_ONLY);
        TqBuffer<float> d_scores((size_t)tokens, CL_MEM_READ_WRITE);
        TqBuffer<uint8_t> d_val_packed((size_t)tokens * val_packed_stride, CL_MEM_READ_ONLY);
        TqBuffer<float> d_val_scales((size_t)tokens * n_groups, CL_MEM_READ_ONLY);
        TqBuffer<float> d_val_zeros((size_t)tokens * n_groups, CL_MEM_READ_ONLY);
        TqBuffer<float> d_output((size_t)dim, CL_MEM_WRITE_ONLY);

        d_query.write_to_device(queue, input.mse.query_rotated);
        d_key_packed.write_to_device(queue, input.mse.packed_indices);
        d_key_norms.write_to_device(queue, input.mse.norms);
        d_centroids.write_to_device(queue, input.mse.centroids);
        d_qsigns.write_to_device(queue, input.qjl.query_signs);
        d_rsigns.write_to_device(queue, input.qjl.residual_signs);
        d_rnorms.write_to_device(queue, input.qjl.residual_norms);
        d_val_packed.write_to_device(queue, input.value.packed_values);
        d_val_scales.write_to_device(queue, input.value.scales);
        d_val_zeros.write_to_device(queue, input.value.zeros);

        int key_bits = input.mse.bits;
        int projections = input.qjl.projections;
        float qjl_scale = input.qjl.qjl_scale;
        float sm_scale = input.sm_scale;
        int val_bits = input.value.bits;
        int group_size = input.value.group_size;

        KernelTuneParams tune_logits = get_kernel_tune("tq_attention_logits");
        size_t local_logits = tune_logits.wg_size_x;
        KernelTuneParams tune_values = get_kernel_tune("tq_attention_apply_values");
        size_t local_values = tune_values.wg_size_x;
        const size_t items_per_thread = tune_values.items_per_thread > 0 ? (size_t)tune_values.items_per_thread : 1u;
        trace_log(
            "fused-profiled",
            "shape tokens=%d dim=%d key_bits=%d val_bits=%d local_logits=%zu local_values=%zu svm_scores=%d svm_output=%d",
            tokens,
            dim,
            key_bits,
            val_bits,
            local_logits,
            local_values,
            d_scores.uses_svm() ? 1 : 0,
            d_output.uses_svm() ? 1 : 0);

        int a = 0;
        if (d_query.bind_kernel_arg(k_logits, a++) != TqStatus::OK ||
            d_key_packed.bind_kernel_arg(k_logits, a++) != TqStatus::OK ||
            d_key_norms.bind_kernel_arg(k_logits, a++) != TqStatus::OK ||
            d_centroids.bind_kernel_arg(k_logits, a++) != TqStatus::OK ||
            d_qsigns.bind_kernel_arg(k_logits, a++) != TqStatus::OK ||
            d_rsigns.bind_kernel_arg(k_logits, a++) != TqStatus::OK ||
            d_rnorms.bind_kernel_arg(k_logits, a++) != TqStatus::OK ||
            d_scores.bind_kernel_arg(k_logits, a++) != TqStatus::OK) {
            fused_attention_cpu_reference(input);
            *kernel_ns = 0;
            return TqStatus::OK;
        }
        clSetKernelArg(k_logits, a++, sizeof(int), &tokens);
        clSetKernelArg(k_logits, a++, sizeof(int), &dim);
        clSetKernelArg(k_logits, a++, sizeof(int), &key_bits);
        clSetKernelArg(k_logits, a++, sizeof(int), &key_packed_stride);
        clSetKernelArg(k_logits, a++, sizeof(int), &projections);
        clSetKernelArg(k_logits, a++, sizeof(int), &proj_words);
        clSetKernelArg(k_logits, a++, sizeof(float), &qjl_scale);
        clSetKernelArg(k_logits, a++, sizeof(float), &sm_scale);
        if (tune_logits.local_mem_bytes > 0) {
            clSetKernelArg(k_logits, a++, tune_logits.local_mem_bytes, nullptr);
        }

        size_t global_tokens = ((size_t)tokens + local_logits - 1) / local_logits * local_logits;
        cl_event ev1;
        cl_int err = clEnqueueNDRangeKernel(queue, k_logits, 1, nullptr, &global_tokens, &local_logits, 0, nullptr, &ev1);
        if (err != CL_SUCCESS) {
            fused_attention_cpu_reference(input);
            *kernel_ns = 0;
            return TqStatus::OK;
        }

        a = 0;
        if (d_scores.bind_kernel_arg(k_values, a++) != TqStatus::OK ||
            d_val_packed.bind_kernel_arg(k_values, a++) != TqStatus::OK ||
            d_val_scales.bind_kernel_arg(k_values, a++) != TqStatus::OK ||
            d_val_zeros.bind_kernel_arg(k_values, a++) != TqStatus::OK ||
            d_output.bind_kernel_arg(k_values, a++) != TqStatus::OK) {
            clWaitForEvents(1, &ev1);
            clReleaseEvent(ev1);
            fused_attention_cpu_reference(input);
            *kernel_ns = 0;
            return TqStatus::OK;
        }
        clSetKernelArg(k_values, a++, sizeof(int), &tokens);
        clSetKernelArg(k_values, a++, sizeof(int), &dim);
        clSetKernelArg(k_values, a++, sizeof(int), &val_bits);
        clSetKernelArg(k_values, a++, sizeof(int), &val_packed_stride);
        clSetKernelArg(k_values, a++, sizeof(int), &group_size);
        clSetKernelArg(k_values, a++, sizeof(int), &n_groups);
        if (tune_values.local_mem_bytes > 0) {
            clSetKernelArg(k_values, a++, tune_values.local_mem_bytes, nullptr);
        }

        size_t work_items = ((size_t)dim + items_per_thread - 1) / items_per_thread;
        size_t global_dim = ((size_t)work_items + local_values - 1) / local_values * local_values;
        cl_event ev2;
        err = clEnqueueNDRangeKernel(queue, k_values, 1, nullptr, &global_dim, &local_values, 1, &ev1, &ev2);
        if (err != CL_SUCCESS) {
            clWaitForEvents(1, &ev1);
            clReleaseEvent(ev1);
            fused_attention_cpu_reference(input);
            *kernel_ns = 0;
            return TqStatus::OK;
        }

        auto wall_start = std::chrono::steady_clock::now();
        clWaitForEvents(1, &ev2);
        auto wall_end = std::chrono::steady_clock::now();

        if (profiling_enabled()) {
            cl_ulong t1_queue = 0, t1_start = 0, t1_end = 0, t2_queue = 0, t2_start = 0, t2_end = 0;
            if (clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_QUEUED, sizeof(t1_queue), &t1_queue, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_START, sizeof(t1_start), &t1_start, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev1, CL_PROFILING_COMMAND_END, sizeof(t1_end), &t1_end, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_QUEUED, sizeof(t2_queue), &t2_queue, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_START, sizeof(t2_start), &t2_start, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev2, CL_PROFILING_COMMAND_END, sizeof(t2_end), &t2_end, nullptr) == CL_SUCCESS) {
                *kernel_ns = (uint64_t)(t2_end - t1_start);
                trace_log(
                    "fused-profiled",
                    "event_ns logits(queue=%llu,start=%llu,end=%llu,dur=%llu) values(queue=%llu,start=%llu,end=%llu,dur=%llu) span=%llu wall=%llu",
                    static_cast<unsigned long long>(t1_queue),
                    static_cast<unsigned long long>(t1_start),
                    static_cast<unsigned long long>(t1_end),
                    static_cast<unsigned long long>(t1_end - t1_start),
                    static_cast<unsigned long long>(t2_queue),
                    static_cast<unsigned long long>(t2_start),
                    static_cast<unsigned long long>(t2_end),
                    static_cast<unsigned long long>(t2_end - t2_start),
                    static_cast<unsigned long long>(*kernel_ns),
                    static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count()));
            } else {
                *kernel_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count();
            }
        } else {
            *kernel_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start).count();
        }

        d_output.read_from_device(queue, input.output);

        clReleaseEvent(ev1);
        clReleaseEvent(ev2);
        return TqStatus::OK;
    } catch (...) {
        fused_attention_cpu_reference(input);
        *kernel_ns = 0;
        return TqStatus::OK;
    }
}

} // namespace tq
