/**
 * TurboQuant OpenCL — memory management with real cl_mem buffers.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl_memory.h"
#include <CL/cl.h>

namespace tq {

// Forward declarations from context
extern cl_context get_context();
extern cl_command_queue get_queue();
extern bool is_initialized();

TqStatus alloc_buffer(GpuBuffer& buf, size_t bytes) {
    if (!is_initialized()) return TqStatus::ERR_NO_PLATFORM;
    cl_int err;
    cl_mem mem = clCreateBuffer(get_context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS) return TqStatus::ERR_OUT_OF_MEMORY;
    buf.handle = (void*)mem;
    buf.size_bytes = bytes;
    return TqStatus::OK;
}

TqStatus alloc_buffer_readonly(GpuBuffer& buf, size_t bytes) {
    if (!is_initialized()) return TqStatus::ERR_NO_PLATFORM;
    cl_int err;
    cl_mem mem = clCreateBuffer(get_context(), CL_MEM_READ_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS) return TqStatus::ERR_OUT_OF_MEMORY;
    buf.handle = (void*)mem;
    buf.size_bytes = bytes;
    return TqStatus::OK;
}

TqStatus free_buffer(GpuBuffer& buf) {
    if (buf.handle) {
        clReleaseMemObject((cl_mem)buf.handle);
        buf.handle = nullptr;
        buf.size_bytes = 0;
    }
    return TqStatus::OK;
}

TqStatus upload(GpuBuffer& buf, const void* host_ptr, size_t bytes) {
    if (!is_initialized()) return TqStatus::ERR_NO_PLATFORM;
    cl_int err = clEnqueueWriteBuffer(get_queue(), (cl_mem)buf.handle, CL_TRUE, 0, bytes, host_ptr, 0, nullptr, nullptr);
    return (err == CL_SUCCESS) ? TqStatus::OK : TqStatus::ERR_KERNEL_LAUNCH;
}

TqStatus download(const GpuBuffer& buf, void* host_ptr, size_t bytes) {
    if (!is_initialized()) return TqStatus::ERR_NO_PLATFORM;
    cl_int err = clEnqueueReadBuffer(get_queue(), (cl_mem)buf.handle, CL_TRUE, 0, bytes, host_ptr, 0, nullptr, nullptr);
    return (err == CL_SUCCESS) ? TqStatus::OK : TqStatus::ERR_KERNEL_LAUNCH;
}

} // namespace tq
