/**
 * TurboQuant OpenCL — buffer abstraction with optional SVM.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_buffer.h"
#include "../include/tq_opencl.h"
#include "../include/tq_trace.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tq {

namespace {

struct BufferPoolKey {
    size_t bytes = 0;
    cl_mem_flags flags = 0;

    bool operator==(const BufferPoolKey& other) const {
        return bytes == other.bytes && flags == other.flags;
    }
};

struct BufferPoolKeyHash {
    size_t operator()(const BufferPoolKey& key) const {
        return std::hash<size_t>{}(key.bytes) ^ (std::hash<cl_mem_flags>{}(key.flags) << 1);
    }
};

std::mutex g_pool_mutex;
std::unordered_map<BufferPoolKey, std::vector<cl_mem>, BufferPoolKeyHash> g_cl_buffer_pool;
#if defined(CL_VERSION_2_0)
std::unordered_map<size_t, std::vector<void*>> g_svm_pool;
#endif
constexpr size_t kPoolDepthLimit = 8;

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
    : count_(count),
      flags_(flags) {
    if (!is_initialized() || count_ == 0) return;

#if defined(CL_VERSION_2_0)
    if (device_has_svm_coarse() && bytes() >= (1u << 20)) {
        bool reused_svm = false;
        {
            std::lock_guard<std::mutex> lock(g_pool_mutex);
            auto it = g_svm_pool.find(bytes());
            if (it != g_svm_pool.end() && !it->second.empty()) {
                svm_ptr_ = it->second.back();
                it->second.pop_back();
                reused_svm = true;
            }
        }
        if (!svm_ptr_) {
            svm_ptr_ = clSVMAlloc(get_context(), CL_MEM_READ_WRITE, bytes(), 0);
        }
        if (svm_ptr_) {
            use_svm_ = true;
            trace_log("buffer", "alloc svm bytes=%zu reused=%d", bytes(), reused_svm ? 1 : 0);
            return;
        }
    }
#endif

    bool reused_clmem = false;
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        BufferPoolKey key{bytes(), flags_};
        auto it = g_cl_buffer_pool.find(key);
        if (it != g_cl_buffer_pool.end() && !it->second.empty()) {
            cl_buffer_ = it->second.back();
            it->second.pop_back();
            reused_clmem = true;
        }
    }
    cl_int err = CL_SUCCESS;
    if (!cl_buffer_) {
        cl_buffer_ = clCreateBuffer(get_context(), flags_, bytes(), nullptr, &err);
    }
    if (err != CL_SUCCESS) {
        cl_buffer_ = nullptr;
        throw std::runtime_error("TqBuffer: clCreateBuffer failed");
    }
    trace_log("buffer", "alloc clmem bytes=%zu flags=%llu reused=%d", bytes(),
              static_cast<unsigned long long>(flags_), reused_clmem ? 1 : 0);
}

template <typename T>
TqBuffer<T>::~TqBuffer() {
    if (svm_ptr_) {
#if defined(CL_VERSION_2_0)
        bool pooled = false;
        {
            std::lock_guard<std::mutex> lock(g_pool_mutex);
            auto& bucket = g_svm_pool[bytes()];
            if (bucket.size() < kPoolDepthLimit) {
                bucket.push_back(svm_ptr_);
                pooled = true;
            }
        }
        if (!pooled) clSVMFree(get_context(), svm_ptr_);
#endif
    }
    if (cl_buffer_) {
        bool pooled = false;
        {
            std::lock_guard<std::mutex> lock(g_pool_mutex);
            auto& bucket = g_cl_buffer_pool[BufferPoolKey{bytes(), flags_}];
            if (bucket.size() < kPoolDepthLimit) {
                bucket.push_back(cl_buffer_);
                pooled = true;
            }
        }
        if (!pooled) clReleaseMemObject(cl_buffer_);
    }
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
