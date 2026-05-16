/**
 * TurboQuant OpenCL — memory management interface.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_opencl_types.h"

namespace tq {

struct GpuBuffer {
    void* handle = nullptr;
    size_t size_bytes = 0;
};

TqStatus alloc_buffer(GpuBuffer& buf, size_t bytes);
TqStatus alloc_buffer_readonly(GpuBuffer& buf, size_t bytes);
TqStatus free_buffer(GpuBuffer& buf);
TqStatus upload(GpuBuffer& buf, const void* host_ptr, size_t bytes);
TqStatus download(const GpuBuffer& buf, void* host_ptr, size_t bytes);

} // namespace tq
