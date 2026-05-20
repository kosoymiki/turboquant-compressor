/**
 * TurboQuant OpenCL — value dequantization host code.
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
const char* select_value_kernel_name(const ValueDequantInput&) {
    return "tq_value_dequant";
}
}

static inline uint32_t val_extract_bits(const uint8_t* packed, int coord, int bits) {
    uint32_t bit_pos = (uint32_t)coord * (uint32_t)bits;
    uint32_t byte_idx = bit_pos >> 3;
    uint32_t bit_off = bit_pos & 7u;
    uint32_t word = packed[byte_idx];
    if (bit_off + (uint32_t)bits > 8u) {
        word |= ((uint32_t)packed[byte_idx + 1]) << 8;
    }
    return (word >> bit_off) & ((1u << bits) - 1u);
}

void value_dequant_cpu_reference(const ValueDequantInput& input, float* output) {
    int n_groups = (input.dim + input.group_size - 1) / input.group_size;
    int packed_stride = (input.dim * input.bits + 7) / 8;
    for (int n = 0; n < input.tokens; n++) {
        const uint8_t* packed = input.packed_values + n * packed_stride;
        for (int j = 0; j < input.dim; j++) {
            uint32_t raw = val_extract_bits(packed, j, input.bits);
            int g = j / input.group_size;
            output[n * input.dim + j] = (float)raw * input.scales[n * n_groups + g]
                                       + input.zeros[n * n_groups + g];
        }
    }
}

TqStatus value_dequant_opencl(const ValueDequantInput& input, float* output) {
    KernelTuneParams tune = get_kernel_tune("tq_value_dequant");
    size_t local_work[3];
    get_local_work_size(tune, local_work);
    const char* kernel_name = select_value_kernel_name(input);
    if (!is_initialized() || !get_kernel(kernel_name)) {
        value_dequant_cpu_reference(input, output);
        return TqStatus::OK;
    }

    cl_kernel k = get_kernel(kernel_name);
    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    int packed_stride = (input.dim * input.bits + 7) / 8;
    int n_groups = (input.dim + input.group_size - 1) / input.group_size;

    size_t packed_size = (size_t)input.tokens * packed_stride;
    size_t scales_size = (size_t)input.tokens * n_groups * sizeof(float);
    size_t zeros_size = scales_size;
    size_t output_size = (size_t)input.tokens * input.dim * sizeof(float);

    cl_mem d_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, packed_size, (void*)input.packed_values, &err);
    cl_mem d_scales = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales_size, (void*)input.scales, &err);
    cl_mem d_zeros = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, zeros_size, (void*)input.zeros, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, output_size, nullptr, &err);

    int tokens = input.tokens;
    int dim = input.dim;
    int bits = input.bits;
    int group_size = input.group_size;

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_packed);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_scales);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_zeros);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_output);
    clSetKernelArg(k, 4, sizeof(int), &tokens);
    clSetKernelArg(k, 5, sizeof(int), &dim);
    clSetKernelArg(k, 6, sizeof(int), &bits);
    clSetKernelArg(k, 7, sizeof(int), &group_size);
    clSetKernelArg(k, 8, sizeof(int), &packed_stride);
    clSetKernelArg(k, 9, sizeof(int), &n_groups);

    size_t local_size = local_work[0];
    if (local_size == 0) local_size = 64;
    size_t global_size = (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_packed); clReleaseMemObject(d_scales);
        clReleaseMemObject(d_zeros); clReleaseMemObject(d_output);
        value_dequant_cpu_reference(input, output);
        return TqStatus::OK;
    }

    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size, output, 0, nullptr, nullptr);

    clReleaseEvent(event);
    clReleaseMemObject(d_packed); clReleaseMemObject(d_scales);
    clReleaseMemObject(d_zeros); clReleaseMemObject(d_output);

    return TqStatus::OK;
}

TqStatus value_dequant_opencl_profiled(const ValueDequantInput& input, float* output, uint64_t* kernel_ns) {
    KernelTuneParams tune = get_kernel_tune("tq_value_dequant");
    size_t local_work[3];
    get_local_work_size(tune, local_work);
    const char* kernel_name = select_value_kernel_name(input);
    if (!is_initialized() || !get_kernel(kernel_name)) {
        value_dequant_cpu_reference(input, output);
        *kernel_ns = 0;
        return TqStatus::OK;
    }

    cl_kernel k = get_kernel(kernel_name);
    cl_int err;
    cl_context ctx = get_context();
    cl_command_queue queue = get_queue();

    int packed_stride = (input.dim * input.bits + 7) / 8;
    int n_groups = (input.dim + input.group_size - 1) / input.group_size;

    size_t packed_size = (size_t)input.tokens * packed_stride;
    size_t scales_size = (size_t)input.tokens * n_groups * sizeof(float);
    size_t zeros_size = scales_size;
    size_t output_size = (size_t)input.tokens * input.dim * sizeof(float);

    cl_mem d_packed = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, packed_size, (void*)input.packed_values, &err);
    cl_mem d_scales = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales_size, (void*)input.scales, &err);
    cl_mem d_zeros = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, zeros_size, (void*)input.zeros, &err);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, output_size, nullptr, &err);

    int tokens = input.tokens;
    int dim = input.dim;
    int bits = input.bits;
    int group_size = input.group_size;

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_packed);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_scales);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_zeros);
    clSetKernelArg(k, 3, sizeof(cl_mem), &d_output);
    clSetKernelArg(k, 4, sizeof(int), &tokens);
    clSetKernelArg(k, 5, sizeof(int), &dim);
    clSetKernelArg(k, 6, sizeof(int), &bits);
    clSetKernelArg(k, 7, sizeof(int), &group_size);
    clSetKernelArg(k, 8, sizeof(int), &packed_stride);
    clSetKernelArg(k, 9, sizeof(int), &n_groups);

    size_t local_size = local_work[0];
    if (local_size == 0) local_size = 64;
    size_t global_size = (((size_t)input.tokens + local_size - 1) / local_size) * local_size;
    cl_event event;
    err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &global_size, &local_size, 0, nullptr, &event);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(d_packed); clReleaseMemObject(d_scales);
        clReleaseMemObject(d_zeros); clReleaseMemObject(d_output);
        value_dequant_cpu_reference(input, output);
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

    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size, output, 0, nullptr, nullptr);

    clReleaseEvent(event);
    clReleaseMemObject(d_packed); clReleaseMemObject(d_scales);
    clReleaseMemObject(d_zeros); clReleaseMemObject(d_output);

    return TqStatus::OK;
}

} // namespace tq
