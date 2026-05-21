/**
 * TurboQuant OpenCL — buffer abstraction with optional SVM.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_buffer.h"
#include "../include/tq_opencl.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace tq {

namespace {

#if defined(CL_VERSION_2_0)
void svm_copy_host_to_device(cl_command_queue queue, void* svm_ptr, const void* host_data, size_t bytes) {
    if (!queue || !svm_ptr || !host_data || bytes == 0) return;
    if (clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, svm_ptr, bytes, 0, nullptr, nullptr) != CL_SUCCESS) return;
    std::memcpy(svm_ptr, host_data, bytes);
    (void)clEnqueueSVMUnmap(queue, svm_ptr, 0, nullptr, nullptr);
    (void)clFinish(queue);
}

void svm_copy_device_to_host(cl_command_queue queue, const void* svm_ptr, void* host_data, size_t bytes) {
    if (!queue || !svm_ptr || !host_data || bytes == 0) return;
    if (clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, const_cast<void*>(svm_ptr), bytes, 0, nullptr, nullptr) != CL_SUCCESS) return;
    std::memcpy(host_data, svm_ptr, bytes);
    (void)clEnqueueSVMUnmap(queue, const_cast<void*>(svm_ptr), 0, nullptr, nullptr);
    (void)clFinish(queue);
}
#endif

} // namespace

template <typename T>
TqBuffer<T>::TqBuffer(size_t count, cl_mem_flags flags)
    : count_(count) {
    if (!is_initialized() || count_ == 0) return;

#if defined(CL_VERSION_2_0)
    if (device_has_svm_coarse() && bytes() >= (1u << 20)) {
        svm_ptr_ = clSVMAlloc(get_context(), CL_MEM_READ_WRITE, bytes(), 0);
        if (svm_ptr_) {
            use_svm_ = true;
            return;
        }
    }
#endif

    cl_int err = CL_SUCCESS;
    cl_buffer_ = clCreateBuffer(get_context(), flags, bytes(), nullptr, &err);
    if (err != CL_SUCCESS) {
        cl_buffer_ = nullptr;
        throw std::runtime_error("TqBuffer: clCreateBuffer failed");
    }
}

template <typename T>
TqBuffer<T>::~TqBuffer() {
    if (svm_ptr_) {
#if defined(CL_VERSION_2_0)
        clSVMFree(get_context(), svm_ptr_);
#endif
    }
    if (cl_buffer_) clReleaseMemObject(cl_buffer_);
}

template <typename T>
T* TqBuffer<T>::host_ptr() {
    return use_svm_ ? static_cast<T*>(svm_ptr_) : nullptr;
}

template <typename T>
const T* TqBuffer<T>::host_ptr() const {
    return use_svm_ ? static_cast<const T*>(svm_ptr_) : nullptr;
}

template <typename T>
void TqBuffer<T>::write_to_device(cl_command_queue queue, const T* host_data) {
    if (use_svm_) {
#if defined(CL_VERSION_2_0)
        if (host_data && host_data != svm_ptr_) svm_copy_host_to_device(queue, svm_ptr_, host_data, bytes());
#endif
        return;
    }
    if (cl_buffer_ && host_data) {
        clEnqueueWriteBuffer(queue, cl_buffer_, CL_TRUE, 0, bytes(), host_data, 0, nullptr, nullptr);
    }
}

template <typename T>
void TqBuffer<T>::read_from_device(cl_command_queue queue, T* host_data) const {
    if (use_svm_) {
#if defined(CL_VERSION_2_0)
        if (host_data && host_data != svm_ptr_) svm_copy_device_to_host(queue, svm_ptr_, host_data, bytes());
#endif
        return;
    }
    if (cl_buffer_ && host_data) {
        clEnqueueReadBuffer(queue, cl_buffer_, CL_TRUE, 0, bytes(), host_data, 0, nullptr, nullptr);
    }
}

template <typename T>
TqStatus TqBuffer<T>::bind_kernel_arg(cl_kernel kernel, cl_uint arg_index) const {
    if (!kernel) return TqStatus::ERR_INVALID_ARG;
    cl_int err = CL_SUCCESS;
    if (use_svm_) {
#if defined(CL_VERSION_2_0)
        err = clSetKernelArgSVMPointer(kernel, arg_index, svm_ptr_);
#else
        err = CL_INVALID_OPERATION;
#endif
    } else {
        err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &cl_buffer_);
    }
    return err == CL_SUCCESS ? TqStatus::OK : TqStatus::ERR_INVALID_ARG;
}

template class TqBuffer<float>;
template class TqBuffer<uint8_t>;
template class TqBuffer<uint32_t>;

} // namespace tq
