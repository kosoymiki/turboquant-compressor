/**
 * TurboQuant OpenCL — buffer abstraction with optional SVM.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "tq_opencl_errors.h"
#include <CL/cl.h>
#include <cstddef>

namespace tq {

template <typename T>
class TqBuffer {
public:
    explicit TqBuffer(size_t count, cl_mem_flags flags = CL_MEM_READ_WRITE);
    ~TqBuffer();

    T* host_ptr();
    const T* host_ptr() const;

    void write_to_device(cl_command_queue queue, const T* host_data);
    void read_from_device(cl_command_queue queue, T* host_data) const;
    TqStatus bind_kernel_arg(cl_kernel kernel, cl_uint arg_index) const;

    cl_mem device_mem() const { return cl_buffer_; }
    void* svm_ptr() const { return svm_ptr_; }
    bool uses_svm() const { return use_svm_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

private:
    void* svm_ptr_ = nullptr;
    cl_mem cl_buffer_ = nullptr;
    size_t count_ = 0;
    bool use_svm_ = false;
    cl_mem_flags flags_ = CL_MEM_READ_WRITE;
};

} // namespace tq
