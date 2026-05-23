/**
 * TurboQuant OpenCL CLI — sidecar entry point.
 * Commands: probe, frontier-smoke, system-svm-smoke, async-build-smoke, benchmark,
 *           mse-score, qjl-score, value-dequant, fused-attention
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl.h"
#include "../include/tq_repo_paths.h"
#include "../include/tq_inference_runtime.h"
#include "../include/tq_runtime_scheduler.h"
#include <CL/cl_ext.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>

// Forward declaration from benchmark module
namespace tq { int run_benchmark(int argc, char* argv[]); }

namespace {
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

void* reserve_replayable_page_slot(size_t page_bytes, size_t slot_index) {
    constexpr uintptr_t kReplayTop = 0x4000000000ull;
    constexpr size_t kProbePages = 0x1000;

    for (size_t probe = slot_index; probe < kProbePages; ++probe) {
        uintptr_t candidate = kReplayTop - ((probe + 1) * page_bytes);
        void* mapped = mmap(reinterpret_cast<void*>(candidate), page_bytes,
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                            -1, 0);
        if (mapped == MAP_FAILED) {
            if (errno == EEXIST || errno == EBUSY || errno == EINVAL) {
                continue;
            }
            continue;
        }
        if (reinterpret_cast<uintptr_t>(mapped) == candidate) {
            return mapped;
        }
        munmap(mapped, page_bytes);
    }

    return MAP_FAILED;
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out << buf;
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    return out.str();
}

bool approx_equal(float a, float b, float rel_tol = 1e-3f, float abs_tol = 1e-6f) {
    return std::fabs(a - b) <= rel_tol * std::fabs(b) + abs_tol;
}

bool load_self_test_kernels(const std::string& kernel_dir) {
    const std::string build_opts = tq::get_default_build_opts();
    return
        tq::load_kernel(kernel_dir + "/tq_mse_score.cl", "tq_mse_score", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_mse_score.cl", "tq_mse_score_tiled", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_qjl_score.cl", "tq_qjl_score", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_qjl_score.cl", "tq_qjl_score_tiled", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_value_dequant.cl", "tq_value_dequant", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_value_dequant.cl", "tq_value_weighted_accum", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_attention_logits.cl", "tq_attention_logits", build_opts) == tq::TqStatus::OK &&
        tq::load_kernel(kernel_dir + "/tq_attention_apply_values.cl", "tq_attention_apply_values", build_opts) == tq::TqStatus::OK;
}

void print_json_result(const std::string& payload) {
    std::printf("%s\n", payload.c_str());
}

template <typename T>
T load_platform_extension_fn(const char* name) {
    cl_platform_id platform = tq::get_platform();
    if (!platform) return nullptr;
    return reinterpret_cast<T>(clGetExtensionFunctionAddressForPlatform(platform, name));
}

bool can_attempt_external_semaphore_lane() {
#if defined(CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR)
    if (!tq::is_initialized() || !tq::get_device()) {
        return false;
    }
    size_t bytes = 0;
    if (clGetDeviceInfo(
            tq::get_device(),
            CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
            0,
            nullptr,
            &bytes) == CL_SUCCESS &&
        bytes >= sizeof(cl_external_semaphore_handle_type_khr)) {
        return true;
    }
#endif
    return tq::device_has_external_semaphore();
}

int print_self_test_error(const char* command, const std::string& reason) {
    print_json_result(
        std::string("{\"status\":\"error\",\"command\":\"") + command +
        "\",\"reason\":\"" + json_escape(reason) + "\"}"
    );
    return 1;
}

struct AsyncBuildSmokeResult {
    bool available = false;
    bool build_ok = false;
    bool callback_observed = false;
    int build_err = 0;
    std::string build_log;
};

AsyncBuildSmokeResult run_async_build_smoke() {
    AsyncBuildSmokeResult result;
    result.available = tq::device_has_async_program_compilation();
    if (!tq::is_initialized() || !tq::get_context() || !tq::get_device()) {
        return result;
    }

    unsetenv("TQ_OPENCL_LAST_ASYNC_BUILD_CALLBACK");

    static const char* kSource = R"CLC(
        __kernel void tq_async_build_smoke(__global int* out) {
            const size_t gid = get_global_id(0);
            out[gid] = (int)gid;
        }
    )CLC";

    tq::TqStatus build_status = tq::TqStatus::ERR_BUILD_FAILED;
    cl_program program = nullptr;
    build_status = tq::build_program(kSource, tq::get_default_build_opts(), &program);
    result.build_err = static_cast<int>(build_status);
    result.build_ok = (build_status == tq::TqStatus::OK && program != nullptr);
    const char* callback_seen = std::getenv("TQ_OPENCL_LAST_ASYNC_BUILD_CALLBACK");
    result.callback_observed = callback_seen && std::strcmp(callback_seen, "1") == 0;

    if (program) {
        size_t log_size = 0;
        if (clGetProgramBuildInfo(
                program,
                tq::get_device(),
                CL_PROGRAM_BUILD_LOG,
                0,
                nullptr,
                &log_size) == CL_SUCCESS &&
            log_size > 1) {
            std::vector<char> log(log_size);
            if (clGetProgramBuildInfo(
                    program,
                    tq::get_device(),
                    CL_PROGRAM_BUILD_LOG,
                    log_size,
                    log.data(),
                    nullptr) == CL_SUCCESS) {
                result.build_log.assign(log.data());
            }
        }
        clReleaseProgram(program);
    }

    return result;
}

struct SystemSvmSmokeResult {
    bool raw_pointer_ok = false;
    bool atomic_ok = false;
    int build_err = 0;
    int kernel_err = 0;
    int exec_info_system_err = 0;
    int exec_info_ptrs_err = 0;
    int set_arg_err = 0;
    int enqueue_err = 0;
    int atomic_exec_info_system_err = 0;
    int atomic_exec_info_ptrs_err = 0;
    int atomic_set_arg_err = 0;
    int atomic_enqueue_err = 0;
    std::string build_log;
};

SystemSvmSmokeResult run_system_svm_smoke() {
    SystemSvmSmokeResult result;
    if (!tq::is_initialized() || !tq::get_context() || !tq::get_device() || !tq::get_queue()) {
        return result;
    }

    static const char* kSource = R"CLC(
        __kernel void tq_system_svm_write(__global int* p) {
            p[0] = 123;
            p[1] = p[1] + 1;
        }

        __kernel void tq_system_svm_atomic(volatile __global int* p) {
            atom_inc(p);
        }
    )CLC";

    cl_int err = CL_SUCCESS;
    cl_device_id device = tq::get_device();
    cl_program program = clCreateProgramWithSource(
        tq::get_context(), 1, &kSource, nullptr, &err);
    result.build_err = err;
    if (!program || err != CL_SUCCESS) {
        return result;
    }

    const char* build_opts = "-cl-std=CL3.0";
    err = clBuildProgram(program, 1, &device, build_opts, nullptr, nullptr);
    result.build_err = err;
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        if (clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                nullptr,
                &log_size) == CL_SUCCESS &&
            log_size > 1) {
            std::vector<char> build_log(log_size);
            if (clGetProgramBuildInfo(
                    program,
                    device,
                    CL_PROGRAM_BUILD_LOG,
                    log_size,
                    build_log.data(),
                    nullptr) == CL_SUCCESS) {
                result.build_log.assign(build_log.data());
            }
        }
        clReleaseProgram(program);
        return result;
    }

    cl_kernel write_kernel = clCreateKernel(program, "tq_system_svm_write", &err);
    result.kernel_err = err;
    cl_kernel atomic_kernel = clCreateKernel(program, "tq_system_svm_atomic", &err);
    if (result.kernel_err == CL_SUCCESS) {
        result.kernel_err = err;
    }
    if (!write_kernel || !atomic_kernel || err != CL_SUCCESS) {
        if (write_kernel) clReleaseKernel(write_kernel);
        if (atomic_kernel) clReleaseKernel(atomic_kernel);
        clReleaseProgram(program);
        return result;
    }

    const size_t page_bytes = 4096;
    int* raw_ptr = static_cast<int*>(reserve_replayable_page_slot(page_bytes, 0));
    int* atomic_ptr = static_cast<int*>(reserve_replayable_page_slot(page_bytes, 1));
    if (raw_ptr == MAP_FAILED || atomic_ptr == MAP_FAILED) {
        if (raw_ptr != MAP_FAILED) munmap(raw_ptr, page_bytes);
        if (atomic_ptr != MAP_FAILED) munmap(atomic_ptr, page_bytes);
        clReleaseKernel(write_kernel);
        clReleaseKernel(atomic_kernel);
        clReleaseProgram(program);
        return result;
    }

    raw_ptr[0] = 7;
    raw_ptr[1] = 41;
    atomic_ptr[0] = 0;

    const cl_bool enable_system = CL_TRUE;
    const void* raw_exec_ptr = raw_ptr;
    const void* atomic_exec_ptr = atomic_ptr;
    const size_t gws = 1;

    result.exec_info_system_err = clSetKernelExecInfo(
        write_kernel,
        CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM,
        sizeof(enable_system),
        &enable_system);
    result.exec_info_ptrs_err = clSetKernelExecInfo(
        write_kernel,
        CL_KERNEL_EXEC_INFO_SVM_PTRS,
        sizeof(raw_exec_ptr),
        &raw_exec_ptr);
    result.set_arg_err = clSetKernelArgSVMPointer(write_kernel, 0, raw_ptr);
    if (result.exec_info_system_err == CL_SUCCESS &&
        result.exec_info_ptrs_err == CL_SUCCESS &&
        result.set_arg_err == CL_SUCCESS) {
        result.enqueue_err = clEnqueueNDRangeKernel(
            tq::get_queue(), write_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (result.enqueue_err == CL_SUCCESS && clFinish(tq::get_queue()) == CL_SUCCESS) {
            result.raw_pointer_ok = raw_ptr[0] == 123 && raw_ptr[1] == 42;
        }
    }

    result.atomic_exec_info_system_err = clSetKernelExecInfo(
        atomic_kernel,
        CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM,
        sizeof(enable_system),
        &enable_system);
    result.atomic_exec_info_ptrs_err = clSetKernelExecInfo(
        atomic_kernel,
        CL_KERNEL_EXEC_INFO_SVM_PTRS,
        sizeof(atomic_exec_ptr),
        &atomic_exec_ptr);
    result.atomic_set_arg_err = clSetKernelArgSVMPointer(atomic_kernel, 0, atomic_ptr);
    if (result.atomic_exec_info_system_err == CL_SUCCESS &&
        result.atomic_exec_info_ptrs_err == CL_SUCCESS &&
        result.atomic_set_arg_err == CL_SUCCESS) {
        result.atomic_enqueue_err = clEnqueueNDRangeKernel(
            tq::get_queue(), atomic_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
        if (result.atomic_enqueue_err == CL_SUCCESS && clFinish(tq::get_queue()) == CL_SUCCESS) {
            result.atomic_ok = atomic_ptr[0] == 1;
        }
    }

    munmap(raw_ptr, page_bytes);
    munmap(atomic_ptr, page_bytes);
    clReleaseKernel(write_kernel);
    clReleaseKernel(atomic_kernel);
    clReleaseProgram(program);
    return result;
}

int run_frontier_smoke() {
    const tq::TqStatus init_status = tq::init_context();
    const bool init_ok = init_status == tq::TqStatus::OK;

    bool ext_mem_dmabuf_alloc = false;
    bool ext_mem_import_ok = false;
    bool ext_mem_acquire_ok = false;
    bool ext_mem_release_ok = false;
    int ext_mem_dmabuf_errno = 0;
    int ext_mem_import_err = 0;
    int ext_mem_acquire_err = 0;
    int ext_mem_release_err = 0;
    std::string ext_mem_dmabuf_stage = "not_attempted";
    std::string ext_mem_import_path = "not_attempted";

    bool ext_sema_create_ok = false;
    bool ext_sema_export_ok = false;
    bool ext_sema_signal_ok = false;
    bool ext_sema_import_ok = false;
    bool ext_sema_wait_ok = false;
    int ext_sema_create_err = 0;
    int ext_sema_export_err = 0;
    int ext_sema_signal_err = 0;
    int ext_sema_import_err = 0;
    int ext_sema_wait_err = 0;
    int ext_sema_fd = -1;
    SystemSvmSmokeResult system_svm_smoke = {};

    tq::InteropBuffer interop_buf = {};
    cl_mem imported_mem = nullptr;

    if (init_ok && tq::device_has_external_memory()) {
        interop_buf = tq::alloc_interop_buffer(4096, tq::InteropBufferType::HOST_VISIBLE);
        ext_mem_dmabuf_alloc = interop_buf.valid && interop_buf.dmabuf_fd >= 0;
        ext_mem_dmabuf_errno = tq::last_interop_dmabuf_errno();
        ext_mem_dmabuf_stage = tq::last_interop_dmabuf_stage();
        if (ext_mem_dmabuf_alloc) {
            imported_mem = static_cast<cl_mem>(tq::import_dmabuf_to_cl(interop_buf.dmabuf_fd, interop_buf.size));
            ext_mem_import_ok = imported_mem != nullptr;
            ext_mem_import_path = tq::last_interop_cl_import_path();
            if (!ext_mem_import_ok) {
                ext_mem_import_err = tq::last_interop_cl_import_error();
            } else {
                auto fn_acquire =
                    load_platform_extension_fn<clEnqueueAcquireExternalMemObjectsKHR_fn>(
                        "clEnqueueAcquireExternalMemObjectsKHR");
                auto fn_release =
                    load_platform_extension_fn<clEnqueueReleaseExternalMemObjectsKHR_fn>(
                        "clEnqueueReleaseExternalMemObjectsKHR");
                if (fn_acquire && fn_release) {
                    ext_mem_acquire_err =
                        fn_acquire(tq::get_queue(), 1, &imported_mem, 0, nullptr, nullptr);
                    ext_mem_acquire_ok = ext_mem_acquire_err == CL_SUCCESS;
                    ext_mem_release_err =
                        fn_release(tq::get_queue(), 1, &imported_mem, 0, nullptr, nullptr);
                    ext_mem_release_ok = ext_mem_release_err == CL_SUCCESS;
                    (void)clFinish(tq::get_queue());
                }
            }
        }
    }

    if (init_ok && can_attempt_external_semaphore_lane()) {
        auto fn_create =
            load_platform_extension_fn<clCreateSemaphoreWithPropertiesKHR_fn>(
                "clCreateSemaphoreWithPropertiesKHR");
        auto fn_get_handle =
            load_platform_extension_fn<clGetSemaphoreHandleForTypeKHR_fn>(
                "clGetSemaphoreHandleForTypeKHR");
        auto fn_signal =
            load_platform_extension_fn<clEnqueueSignalSemaphoresKHR_fn>(
                "clEnqueueSignalSemaphoresKHR");
        auto fn_wait =
            load_platform_extension_fn<clEnqueueWaitSemaphoresKHR_fn>(
                "clEnqueueWaitSemaphoresKHR");
        auto fn_release_sema =
            load_platform_extension_fn<clReleaseSemaphoreKHR_fn>(
                "clReleaseSemaphoreKHR");
        if (fn_create && fn_get_handle && fn_signal && fn_wait && fn_release_sema) {
            cl_semaphore_properties_khr props[] = {
                CL_SEMAPHORE_TYPE_KHR,
                CL_SEMAPHORE_TYPE_BINARY_KHR,
                CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
                CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
                CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR,
                0,
            };
            cl_semaphore_khr semaphore = fn_create(tq::get_context(), props, &ext_sema_create_err);
            ext_sema_create_ok = semaphore != nullptr && ext_sema_create_err == CL_SUCCESS;
            if (ext_sema_create_ok) {
                ext_sema_signal_err = fn_signal(
                    tq::get_queue(),
                    1,
                    &semaphore,
                    nullptr,
                    0,
                    nullptr,
                    nullptr);
                ext_sema_signal_ok = ext_sema_signal_err == CL_SUCCESS && clFinish(tq::get_queue()) == CL_SUCCESS;
                if (ext_sema_signal_ok) {
                    ext_sema_export_err = fn_get_handle(
                        semaphore,
                        tq::get_device(),
                        CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
                        sizeof(ext_sema_fd),
                        &ext_sema_fd,
                        nullptr);
                    ext_sema_export_ok = ext_sema_export_err == CL_SUCCESS && ext_sema_fd >= 0;
                    if (ext_sema_export_ok) {
                        cl_semaphore_properties_khr import_props[] = {
                            CL_SEMAPHORE_TYPE_KHR,
                            CL_SEMAPHORE_TYPE_BINARY_KHR,
                            CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
                            static_cast<cl_semaphore_properties_khr>(ext_sema_fd),
                            0,
                        };
                        cl_semaphore_khr imported =
                            fn_create(tq::get_context(), import_props, &ext_sema_import_err);
                        ext_sema_import_ok = imported != nullptr && ext_sema_import_err == CL_SUCCESS;
                        if (ext_sema_import_ok) {
                            ext_sema_fd = -1; // ownership transferred on successful import
                            ext_sema_wait_err = fn_wait(
                                tq::get_queue(),
                                1,
                                &imported,
                                nullptr,
                                0,
                                nullptr,
                                nullptr);
                            ext_sema_wait_ok =
                                ext_sema_wait_err == CL_SUCCESS && clFinish(tq::get_queue()) == CL_SUCCESS;
                            (void)fn_release_sema(imported);
                        }
                    }
                }
            }
            if (semaphore) {
                (void)fn_release_sema(semaphore);
            }
        }
    }

    if (ext_sema_fd >= 0) {
        close(ext_sema_fd);
    }
    if (imported_mem) {
        clReleaseMemObject(imported_mem);
    }
    if (interop_buf.valid) {
        tq::free_interop_buffer(interop_buf);
    }

    if (init_ok && tq::device_has_svm_fine()) {
        system_svm_smoke = run_system_svm_smoke();
    }

    std::ostringstream out;
    out << "{"
        << "\"status\":\"" << (init_ok ? "ok" : "error") << "\","
        << "\"command\":\"frontier-smoke\","
        << "\"initContext\":" << (init_ok ? "true" : "false") << ","
        << "\"initStatus\":" << static_cast<int>(init_status) << ","
        << "\"capabilities\":{"
        << "\"initializeMemory\":" << (tq::device_has_initialize_memory() ? "true" : "false") << ","
        << "\"externalMemory\":" << (tq::device_has_external_memory() ? "true" : "false") << ","
        << "\"externalMemoryAhb\":" << (tq::device_has_external_memory_ahb() ? "true" : "false") << ","
        << "\"externalSemaphore\":" << (tq::device_has_external_semaphore() ? "true" : "false") << ","
        << "\"commandBuffer\":" << (tq::device_has_command_buffer() ? "true" : "false") << ","
        << "\"svm\":" << (tq::device_has_svm() ? "true" : "false") << ","
        << "\"svmCoarse\":" << (tq::device_has_svm_coarse() ? "true" : "false") << ","
        << "\"svmFine\":" << (tq::device_has_svm_fine() ? "true" : "false") << ","
        << "\"svmAtomics\":" << (tq::device_has_svm_atomics() ? "true" : "false")
        << "},"
        << "\"externalMemorySmoke\":{"
        << "\"dmabufAllocated\":" << (ext_mem_dmabuf_alloc ? "true" : "false") << ","
        << "\"dmabufErrno\":" << ext_mem_dmabuf_errno << ","
        << "\"dmabufStage\":\"" << json_escape(ext_mem_dmabuf_stage) << "\","
        << "\"importPath\":\"" << json_escape(ext_mem_import_path) << "\","
        << "\"imported\":" << (ext_mem_import_ok ? "true" : "false") << ","
        << "\"acquireOk\":" << (ext_mem_acquire_ok ? "true" : "false") << ","
        << "\"releaseOk\":" << (ext_mem_release_ok ? "true" : "false") << ","
        << "\"importErr\":" << ext_mem_import_err << ","
        << "\"acquireErr\":" << ext_mem_acquire_err << ","
        << "\"releaseErr\":" << ext_mem_release_err
        << "},"
        << "\"externalSemaphoreSmoke\":{"
        << "\"createdExportable\":" << (ext_sema_create_ok ? "true" : "false") << ","
        << "\"signaled\":" << (ext_sema_signal_ok ? "true" : "false") << ","
        << "\"exportedSyncFd\":" << (ext_sema_export_ok ? "true" : "false") << ","
        << "\"importedSyncFd\":" << (ext_sema_import_ok ? "true" : "false") << ","
        << "\"waitedImported\":" << (ext_sema_wait_ok ? "true" : "false") << ","
        << "\"createErr\":" << ext_sema_create_err << ","
        << "\"signalErr\":" << ext_sema_signal_err << ","
        << "\"exportErr\":" << ext_sema_export_err << ","
        << "\"importErr\":" << ext_sema_import_err << ","
        << "\"waitErr\":" << ext_sema_wait_err
        << "},"
        << "\"systemSvmSmoke\":{"
        << "\"rawPointerOk\":" << (system_svm_smoke.raw_pointer_ok ? "true" : "false") << ","
        << "\"atomicOk\":" << (system_svm_smoke.atomic_ok ? "true" : "false") << ","
        << "\"buildErr\":" << system_svm_smoke.build_err << ","
        << "\"kernelErr\":" << system_svm_smoke.kernel_err << ","
        << "\"execInfoSystemErr\":" << system_svm_smoke.exec_info_system_err << ","
        << "\"execInfoPtrsErr\":" << system_svm_smoke.exec_info_ptrs_err << ","
        << "\"setArgErr\":" << system_svm_smoke.set_arg_err << ","
        << "\"enqueueErr\":" << system_svm_smoke.enqueue_err << ","
        << "\"atomicExecInfoSystemErr\":" << system_svm_smoke.atomic_exec_info_system_err << ","
        << "\"atomicExecInfoPtrsErr\":" << system_svm_smoke.atomic_exec_info_ptrs_err << ","
        << "\"atomicSetArgErr\":" << system_svm_smoke.atomic_set_arg_err << ","
        << "\"atomicEnqueueErr\":" << system_svm_smoke.atomic_enqueue_err << ","
        << "\"buildLog\":\"" << json_escape(system_svm_smoke.build_log) << "\""
        << "}"
        << "}";
    print_json_result(out.str());
    tq::shutdown_context();
    return init_ok ? 0 : 1;
}

int run_system_svm_smoke_command() {
    const tq::TqStatus init_status = tq::init_context();
    const bool init_ok = init_status == tq::TqStatus::OK;
    SystemSvmSmokeResult system_svm_smoke = {};

    if (init_ok && tq::device_has_svm_fine()) {
        system_svm_smoke = run_system_svm_smoke();
    }

    std::ostringstream out;
    out << "{"
        << "\"status\":\"" << (init_ok ? "ok" : "error") << "\","
        << "\"command\":\"system-svm-smoke\","
        << "\"initContext\":" << (init_ok ? "true" : "false") << ","
        << "\"initStatus\":" << static_cast<int>(init_status) << ","
        << "\"capabilities\":{"
        << "\"svm\":" << (tq::device_has_svm() ? "true" : "false") << ","
        << "\"svmCoarse\":" << (tq::device_has_svm_coarse() ? "true" : "false") << ","
        << "\"svmFine\":" << (tq::device_has_svm_fine() ? "true" : "false") << ","
        << "\"svmAtomics\":" << (tq::device_has_svm_atomics() ? "true" : "false")
        << "},"
        << "\"systemSvmSmoke\":{"
        << "\"rawPointerOk\":" << (system_svm_smoke.raw_pointer_ok ? "true" : "false") << ","
        << "\"atomicOk\":" << (system_svm_smoke.atomic_ok ? "true" : "false") << ","
        << "\"buildErr\":" << system_svm_smoke.build_err << ","
        << "\"kernelErr\":" << system_svm_smoke.kernel_err << ","
        << "\"execInfoSystemErr\":" << system_svm_smoke.exec_info_system_err << ","
        << "\"execInfoPtrsErr\":" << system_svm_smoke.exec_info_ptrs_err << ","
        << "\"setArgErr\":" << system_svm_smoke.set_arg_err << ","
        << "\"enqueueErr\":" << system_svm_smoke.enqueue_err << ","
        << "\"atomicExecInfoSystemErr\":" << system_svm_smoke.atomic_exec_info_system_err << ","
        << "\"atomicExecInfoPtrsErr\":" << system_svm_smoke.atomic_exec_info_ptrs_err << ","
        << "\"atomicSetArgErr\":" << system_svm_smoke.atomic_set_arg_err << ","
        << "\"atomicEnqueueErr\":" << system_svm_smoke.atomic_enqueue_err << ","
        << "\"buildLog\":\"" << json_escape(system_svm_smoke.build_log) << "\""
        << "}"
        << "}";
    print_json_result(out.str());
    tq::shutdown_context();
    return init_ok ? 0 : 1;
}

int run_async_build_smoke_command() {
    const tq::TqStatus init_status = tq::init_context();
    const bool init_ok = init_status == tq::TqStatus::OK;
    AsyncBuildSmokeResult smoke = {};
    if (init_ok) {
        smoke = run_async_build_smoke();
    }

    std::ostringstream out;
    out << "{"
        << "\"status\":\"" << ((init_ok && smoke.build_ok) ? "ok" : "error") << "\","
        << "\"command\":\"async-build-smoke\","
        << "\"initContext\":" << (init_ok ? "true" : "false") << ","
        << "\"initStatus\":" << static_cast<int>(init_status) << ","
        << "\"available\":" << (smoke.available ? "true" : "false") << ","
        << "\"buildOk\":" << (smoke.build_ok ? "true" : "false") << ","
        << "\"callbackObserved\":" << (smoke.callback_observed ? "true" : "false") << ","
        << "\"buildErr\":" << smoke.build_err << ","
        << "\"buildLog\":\"" << json_escape(smoke.build_log) << "\""
        << "}";
    print_json_result(out.str());
    tq::shutdown_context();
    return (init_ok && smoke.build_ok) ? 0 : 1;
}

int run_mse_self_test() {
    const int tokens = 4, dim = 8, bits = 3;
    float centroids[8] = {-0.8f, -0.5f, -0.2f, 0.0f, 0.1f, 0.3f, 0.6f, 0.9f};
    float query[8] = {0.1f, 0.2f, 0.3f, 0.4f, -0.1f, -0.2f, -0.3f, -0.4f};
    float norms[4] = {1.0f, 1.5f, 0.8f, 2.0f};
    int packed_stride = (dim * bits + 7) / 8;
    std::vector<uint8_t> packed(tokens * packed_stride, 0);
    for (int j = 0; j < dim; j++) {
        int bit_pos = j * bits;
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        packed[1 * packed_stride + byte_idx] |= (uint8_t)(7 << bit_off);
        if (bit_off + bits > 8)
            packed[1 * packed_stride + byte_idx + 1] |= (uint8_t)(7 >> (8 - bit_off));
    }

    tq::MseScoreInput input{};
    input.packed_indices = packed.data();
    input.norms = norms;
    input.query_rotated = query;
    input.centroids = centroids;
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.packed_stride = packed_stride;

    float scores_gpu[4] = {};
    float scores_cpu[4] = {};
    auto status = tq::mse_score_opencl(input, scores_gpu);
    tq::mse_score_cpu_reference(input, scores_cpu);
    bool parity = true;
    for (int i = 0; i < tokens; i++) parity = parity && approx_equal(scores_gpu[i], scores_cpu[i]);

    std::ostringstream out;
    out << "{\"status\":\"" << (status == tq::TqStatus::OK && parity ? "ok" : "error")
        << "\",\"command\":\"mse-score\",\"parity\":" << (parity ? "true" : "false")
        << ",\"scores\":[";
    for (int i = 0; i < tokens; i++) {
        if (i) out << ",";
        out << scores_gpu[i];
    }
    out << "]}";
    print_json_result(out.str());
    return (status == tq::TqStatus::OK && parity) ? 0 : 1;
}

int run_qjl_self_test() {
    const int tokens = 4;
    const int projections = 64;
    const int proj_words = (projections + 31) / 32;
    uint32_t query_signs[proj_words] = {0xAAAAAAAAu, 0x13579BDFu};
    uint32_t residual_signs[tokens * proj_words] = {
        0xAAAAAAAAu, 0x13579BDFu,
        0x55555555u, 0x2468ACE0u,
        0xFFFFFFFFu, 0x00000000u,
        0x0F0F0F0Fu, 0xF0F0F0F0u,
    };
    float residual_norms[tokens] = {1.0f, 0.5f, 0.25f, 0.75f};
    float scores_gpu[tokens] = {0.1f, -0.2f, 0.3f, -0.4f};
    float scores_cpu[tokens] = {0.1f, -0.2f, 0.3f, -0.4f};

    tq::QjlScoreInput gpu_input{};
    gpu_input.query_signs = query_signs;
    gpu_input.residual_signs = residual_signs;
    gpu_input.residual_norms = residual_norms;
    gpu_input.base_scores = scores_gpu;
    gpu_input.tokens = tokens;
    gpu_input.projections = projections;
    gpu_input.qjl_scale = 0.125f;

    tq::QjlScoreInput cpu_input = gpu_input;
    cpu_input.base_scores = scores_cpu;

    auto status = tq::qjl_score_opencl(gpu_input);
    tq::qjl_score_cpu_reference(cpu_input);
    bool parity = true;
    for (int i = 0; i < tokens; i++) parity = parity && approx_equal(scores_gpu[i], scores_cpu[i]);

    std::ostringstream out;
    out << "{\"status\":\"" << (status == tq::TqStatus::OK && parity ? "ok" : "error")
        << "\",\"command\":\"qjl-score\",\"parity\":" << (parity ? "true" : "false")
        << ",\"scores\":[";
    for (int i = 0; i < tokens; i++) {
        if (i) out << ",";
        out << scores_gpu[i];
    }
    out << "]}";
    print_json_result(out.str());
    return (status == tq::TqStatus::OK && parity) ? 0 : 1;
}

int run_value_dequant_self_test() {
    const int tokens = 2, dim = 8, bits = 4, group_size = 4;
    const int packed_stride = (dim * bits + 7) / 8;
    const int n_groups = (dim + group_size - 1) / group_size;
    uint8_t packed[tokens * packed_stride] = {0x10, 0x32, 0x54, 0x76, 0x89, 0xAB, 0xCD, 0xEF};
    float scales[tokens * n_groups] = {0.25f, 0.50f, 0.75f, 1.00f};
    float zeros[tokens * n_groups] = {-1.0f, 0.5f, 1.0f, -0.5f};
    float output_gpu[tokens * dim] = {};
    float output_cpu[tokens * dim] = {};

    tq::ValueDequantInput input{};
    input.packed_values = packed;
    input.scales = scales;
    input.zeros = zeros;
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.group_size = group_size;

    auto status = tq::value_dequant_opencl(input, output_gpu);
    tq::value_dequant_cpu_reference(input, output_cpu);
    bool parity = true;
    for (int i = 0; i < tokens * dim; i++) parity = parity && approx_equal(output_gpu[i], output_cpu[i]);

    std::ostringstream out;
    out << "{\"status\":\"" << (status == tq::TqStatus::OK && parity ? "ok" : "error")
        << "\",\"command\":\"value-dequant\",\"parity\":" << (parity ? "true" : "false")
        << ",\"values\":[";
    for (int i = 0; i < tokens * dim; i++) {
        if (i) out << ",";
        out << output_gpu[i];
    }
    out << "]}";
    print_json_result(out.str());
    return (status == tq::TqStatus::OK && parity) ? 0 : 1;
}

int run_fused_attention_self_test() {
    const int tokens = 4, dim = 8, bits = 4, group_size = 4;
    const int packed_stride = (dim * bits + 7) / 8;
    const int n_groups = (dim + group_size - 1) / group_size;
    const int projections = dim * 2;
    const int proj_words = (projections + 31) / 32;

    float query[dim] = {0.25f, -0.15f, 0.10f, 0.05f, -0.20f, 0.30f, -0.35f, 0.40f};
    float norms[tokens] = {1.0f, 0.8f, 1.2f, 0.9f};
    float centroids[1 << bits];
    for (int i = 0; i < (1 << bits); i++) centroids[i] = -1.0f + 0.125f * i;
    uint8_t key_packed[tokens * packed_stride] = {0x10, 0x32, 0x54, 0x76, 0x89, 0xAB, 0xCD, 0xEF,
                                                  0x01, 0x23, 0x45, 0x67, 0x98, 0xBA, 0xDC, 0xFE};
    uint32_t query_signs[proj_words] = {0xAAAAAAAAu};
    uint32_t residual_signs[tokens * proj_words] = {0xAAAAAAAAu, 0x55555555u, 0xF0F0F0F0u, 0x0F0F0F0Fu};
    float residual_norms[tokens] = {1.0f, 0.5f, 0.25f, 0.75f};
    uint8_t value_packed[tokens * packed_stride] = {0xFE, 0xDC, 0xBA, 0x98, 0x67, 0x45, 0x23, 0x01,
                                                    0xEF, 0xCD, 0xAB, 0x89, 0x76, 0x54, 0x32, 0x10};
    float value_scales[tokens * n_groups] = {0.25f, 0.5f, 0.4f, 0.8f, 0.35f, 0.7f, 0.45f, 0.9f};
    float value_zeros[tokens * n_groups] = {0.0f, -0.1f, 0.2f, -0.2f, 0.1f, -0.3f, 0.3f, -0.4f};
    float output_gpu[dim] = {};
    float output_cpu[dim] = {};

    tq::FusedAttentionInput gpu_input{};
    gpu_input.mse.packed_indices = key_packed;
    gpu_input.mse.norms = norms;
    gpu_input.mse.query_rotated = query;
    gpu_input.mse.centroids = centroids;
    gpu_input.mse.tokens = tokens;
    gpu_input.mse.dim = dim;
    gpu_input.mse.bits = bits;
    gpu_input.mse.packed_stride = packed_stride;
    gpu_input.qjl.query_signs = query_signs;
    gpu_input.qjl.residual_signs = residual_signs;
    gpu_input.qjl.residual_norms = residual_norms;
    gpu_input.qjl.base_scores = nullptr;
    gpu_input.qjl.tokens = tokens;
    gpu_input.qjl.projections = projections;
    gpu_input.qjl.qjl_scale = 0.125f;
    gpu_input.value.packed_values = value_packed;
    gpu_input.value.scales = value_scales;
    gpu_input.value.zeros = value_zeros;
    gpu_input.value.tokens = tokens;
    gpu_input.value.dim = dim;
    gpu_input.value.bits = bits;
    gpu_input.value.group_size = group_size;
    gpu_input.sm_scale = 1.0f / std::sqrt((float)dim);
    gpu_input.output = output_gpu;

    tq::FusedAttentionInput cpu_input = gpu_input;
    cpu_input.output = output_cpu;

    auto status = tq::fused_attention_tiled_opencl(gpu_input);
    tq::fused_attention_cpu_reference(cpu_input);

    double dot = 0.0, norm_gpu = 0.0, norm_cpu = 0.0;
    for (int i = 0; i < dim; i++) {
        dot += (double)output_gpu[i] * (double)output_cpu[i];
        norm_gpu += (double)output_gpu[i] * (double)output_gpu[i];
        norm_cpu += (double)output_cpu[i] * (double)output_cpu[i];
    }
    double cosine = (norm_gpu > 0.0 && norm_cpu > 0.0) ? dot / (std::sqrt(norm_gpu) * std::sqrt(norm_cpu)) : 0.0;
    bool parity = cosine > 0.95;

    std::ostringstream out;
    out << "{\"status\":\"" << (status == tq::TqStatus::OK && parity ? "ok" : "error")
        << "\",\"command\":\"fused-attention\",\"parity\":" << (parity ? "true" : "false")
        << ",\"cosine\":" << cosine << ",\"values\":[";
    for (int i = 0; i < dim; i++) {
        if (i) out << ",";
        out << output_gpu[i];
    }
    out << "]}";
    print_json_result(out.str());
    return (status == tq::TqStatus::OK && parity) ? 0 : 1;
}

int run_runtime_scheduler_smoke() {
    std::string kernel_dir = tq::resolve_kernel_dir();
    if (kernel_dir.empty()) return print_self_test_error("runtime-scheduler-smoke", "kernel_dir_not_found");

    tq::runtime_scheduler_reset();
    auto st = tq::init_context();
    if (st != tq::TqStatus::OK) return print_self_test_error("runtime-scheduler-smoke", "init_context_failed");
    if (!load_self_test_kernels(kernel_dir)) {
        tq::shutdown_context();
        return print_self_test_error("runtime-scheduler-smoke", "kernel_load_failed");
    }

    const int tokens = 2, dim = 8, bits = 4, group_size = 4;
    const int packed_stride = (dim * bits + 7) / 8;
    const int n_groups = (dim + group_size - 1) / group_size;
    uint8_t packed[tokens * packed_stride] = {0x10, 0x32, 0x54, 0x76, 0x89, 0xAB, 0xCD, 0xEF};
    float scales[tokens * n_groups] = {0.25f, 0.50f, 0.75f, 1.00f};
    float zeros[tokens * n_groups] = {-1.0f, 0.5f, 1.0f, -0.5f};
    float output_gpu[tokens * dim] = {};
    uint64_t dispatch_ns = 0;
    tq::ValueDequantInput input{};
    input.packed_values = packed;
    input.scales = scales;
    input.zeros = zeros;
    input.tokens = tokens;
    input.dim = dim;
    input.bits = bits;
    input.group_size = group_size;
    auto dispatch_status = tq::value_dequant_opencl_profiled(input, output_gpu, &dispatch_ns);

    std::string artifact_path;
    const bool artifact_ok = tq::runtime_scheduler_write_artifact(&artifact_path);
    const tq::RuntimeSchedulerSummary summary = tq::runtime_scheduler_get_summary();

    std::ostringstream out;
    out << "{"
        << "\"status\":\"" << (artifact_ok ? "ok" : "error") << "\","
        << "\"command\":\"runtime-scheduler-smoke\","
        << "\"artifactOk\":" << (artifact_ok ? "true" : "false") << ","
        << "\"artifactPath\":\"" << json_escape(artifact_path) << "\","
        << "\"compileIntents\":" << summary.compile_intents << ","
        << "\"dedupedCompileIntents\":" << summary.deduped_compile_intents << ","
        << "\"programReuseHits\":" << summary.program_reuse_hits << ","
        << "\"binaryCacheHits\":" << summary.binary_cache_hits << ","
        << "\"precompiledBinaryHits\":" << summary.precompiled_binary_hits << ","
        << "\"sourceBuilds\":" << summary.source_builds << ","
        << "\"ilBuilds\":" << summary.il_builds << ","
        << "\"binaryBuilds\":" << summary.binary_builds << ","
        << "\"asyncBuildCallbacks\":" << summary.async_build_callbacks << ","
        << "\"kernelReadyCount\":" << summary.kernel_ready_count << ","
        << "\"autotuneEntries\":" << summary.autotune_entries << ","
        << "\"planner\":{"
        << "\"preferredBuildPath\":\"" << json_escape(summary.preferred_build_path) << "\","
        << "\"compileQueueDepth\":" << summary.compile_queue_depth << ","
        << "\"admittedCompileCount\":" << summary.admitted_compile_count << ","
        << "\"deferredCompileCount\":" << summary.deferred_compile_count << ","
        << "\"speculativeCandidateCount\":" << summary.speculative_candidate_count << ","
        << "\"executionQueueSlots\":" << summary.execution_queue_slots << ","
        << "\"executionBudgetSlots\":" << summary.execution_budget_slots << ","
        << "\"compileBudgetSlots\":" << summary.compile_budget_slots << ","
        << "\"compileDeferredForExecution\":" << summary.compile_deferred_for_execution << ","
        << "\"dispatchBudgetPermille\":" << summary.dispatch_budget_permille << ","
        << "\"compileBudgetPermille\":" << summary.compile_budget_permille << ","
        << "\"executionAdmission\":\"" << json_escape(summary.execution_admission) << "\","
        << "\"executionAdmissionReason\":\"" << json_escape(summary.execution_admission_reason) << "\","
        << "\"arbitrationMode\":\"" << json_escape(summary.arbitration_mode) << "\","
        << "\"arbitrationReason\":\"" << json_escape(summary.arbitration_reason) << "\","
        << "\"multiQueueMode\":\"" << json_escape(summary.multi_queue_mode) << "\","
        << "\"queueGuardTriggered\":" << (summary.queue_guard_triggered ? "true" : "false") << ","
        << "\"queueGuardReason\":\"" << json_escape(summary.queue_guard_reason) << "\","
        << "\"compileExecuteBalancePermille\":" << summary.compile_execute_balance_permille << ","
        << "\"compileQueueAgeTicks\":" << summary.compile_queue_age_ticks << ","
        << "\"dispatchQueueAgeTicks\":" << summary.dispatch_queue_age_ticks << ","
        << "\"starvationPreventedCount\":" << summary.starvation_prevented_count << ","
        << "\"fairnessDebtPermille\":" << summary.fairness_debt_permille << ","
        << "\"compileQueueAgeBand\":\"" << json_escape(summary.compile_queue_age_band) << "\","
        << "\"dispatchQueueAgeBand\":\"" << json_escape(summary.dispatch_queue_age_band) << "\","
        << "\"latencyAdmission\":\"" << json_escape(summary.latency_admission) << "\","
        << "\"latencyAdmissionReason\":\"" << json_escape(summary.latency_admission_reason) << "\","
        << "\"dispatchUrgency\":\"" << json_escape(summary.dispatch_urgency) << "\","
        << "\"classAwareDispatchMode\":\"" << json_escape(summary.class_aware_dispatch_mode) << "\","
        << "\"redistributionMode\":\"" << json_escape(summary.redistribution_mode) << "\","
        << "\"redistributionReason\":\"" << json_escape(summary.redistribution_reason) << "\","
        << "\"starvationPreventionMode\":\"" << json_escape(summary.starvation_prevention_mode) << "\","
        << "\"starvationPreventionReason\":\"" << json_escape(summary.starvation_prevention_reason) << "\","
        << "\"starvationTargetKernel\":\"" << json_escape(summary.starvation_target_kernel) << "\","
        << "\"throttleTargetKernel\":\"" << json_escape(summary.throttle_target_kernel) << "\","
        << "\"pressureRecoveryAction\":\"" << json_escape(summary.pressure_recovery_action) << "\","
        << "\"pressureRecoveryReason\":\"" << json_escape(summary.pressure_recovery_reason) << "\","
        << "\"kernelClassPolicyCount\":" << summary.kernel_class_policy_count
        << "},"
        << "\"execution\":{"
        << "\"dispatchStatus\":\"" << (dispatch_status == tq::TqStatus::OK ? "ok" : "error") << "\","
        << "\"dispatchCount\":" << summary.execution_dispatch_count << ","
        << "\"profiledDispatchCount\":" << summary.execution_profiled_dispatch_count << ","
        << "\"dispatchWaveSlots\":" << summary.dispatch_wave_slots << ","
        << "\"activeDispatchSlots\":" << summary.active_dispatch_slots << ","
        << "\"queuePressurePermille\":" << summary.queue_pressure_permille << ","
        << "\"queuePressureBand\":\"" << json_escape(summary.queue_pressure_band) << "\","
        << "\"dispatchUtilizationPermille\":" << summary.dispatch_utilization_permille << ","
        << "\"compileUtilizationPermille\":" << summary.compile_utilization_permille << ","
        << "\"lastDispatchKernelName\":\"" << json_escape(summary.last_dispatch_kernel_name) << "\","
        << "\"lastDispatchLatencyClass\":\"" << json_escape(summary.last_dispatch_latency_class) << "\","
        << "\"lastDispatchLatencyReason\":\"" << json_escape(summary.last_dispatch_latency_reason) << "\","
        << "\"lastDispatchKernelNs\":" << summary.last_dispatch_kernel_ns << ","
        << "\"lastDispatchGlobalSize\":" << summary.last_dispatch_global_size << ","
        << "\"lastDispatchLocalSize\":" << summary.last_dispatch_local_size << ","
        << "\"executionBudgetBytes\":" << summary.execution_budget_bytes << ","
        << "\"executionPressureBand\":\"" << json_escape(summary.execution_pressure_band) << "\""
        << "},"
        << "\"memoryPlan\":{"
        << "\"globalMemBytes\":" << summary.global_mem_bytes << ","
        << "\"recommendedResidentBudgetBytes\":" << summary.recommended_budget_bytes << ","
        << "\"kernelCacheBytes\":" << summary.kernel_cache_bytes << ","
        << "\"precompiledLibraryBytes\":" << summary.precompiled_library_bytes << ","
        << "\"residentPressureBand\":\"" << json_escape(summary.resident_pressure_band) << "\","
        << "\"evictionRecommended\":" << (summary.eviction_recommended ? "true" : "false") << ","
        << "\"compactionRecommended\":" << (summary.compaction_recommended ? "true" : "false") << ","
        << "\"persistentResidencyReady\":" << (summary.persistent_residency_ready ? "true" : "false")
        << "},"
        << "\"kernelClassPolicies\":[";
    for (size_t i = 0; i < summary.kernel_class_policies.size(); ++i) {
        const auto& policy = summary.kernel_class_policies[i];
        if (i > 0) out << ",";
        out << "{"
            << "\"kernelName\":\"" << json_escape(policy.kernel_name) << "\","
            << "\"kernelClass\":\"" << json_escape(policy.kernel_class) << "\","
            << "\"latencyClass\":\"" << json_escape(policy.latency_class) << "\","
            << "\"latencyReason\":\"" << json_escape(policy.latency_reason) << "\","
            << "\"admission\":\"" << json_escape(policy.admission) << "\","
            << "\"admissionReason\":\"" << json_escape(policy.admission_reason) << "\","
            << "\"queueAgeBand\":\"" << json_escape(policy.queue_age_band) << "\","
            << "\"urgency\":\"" << json_escape(policy.urgency) << "\","
            << "\"dispatchLane\":\"" << json_escape(policy.dispatch_lane) << "\","
            << "\"redistributionAction\":\"" << json_escape(policy.redistribution_action) << "\","
            << "\"redistributionReason\":\"" << json_escape(policy.redistribution_reason) << "\","
            << "\"pressureAction\":\"" << json_escape(policy.pressure_action) << "\","
            << "\"pressureReason\":\"" << json_escape(policy.pressure_reason) << "\","
            << "\"queueAgeTicks\":" << policy.queue_age_ticks << ","
            << "\"queueOrdinal\":" << policy.queue_ordinal << ","
            << "\"baseBudgetPermille\":" << policy.base_budget_permille << ","
            << "\"ageBoostPermille\":" << policy.age_boost_permille << ","
            << "\"effectiveBudgetPermille\":" << policy.effective_budget_permille << ","
            << "\"starvationScore\":" << policy.starvation_score << ","
            << "\"starvationBoosted\":" << (policy.starvation_boosted ? "true" : "false") << ","
            << "\"throttleRecommended\":" << (policy.throttle_recommended ? "true" : "false") << ","
            << "\"liveDispatchAnchor\":" << (policy.live_dispatch_anchor ? "true" : "false")
            << "}";
    }
    out << "]}";

    tq::release_all_programs();
    tq::shutdown_context();
    print_json_result(out.str());
    return artifact_ok ? 0 : 1;
}

int run_inference_runtime_smoke() {
    std::string kernel_dir = tq::resolve_kernel_dir();
    if (kernel_dir.empty()) return print_self_test_error("inference-runtime-smoke", "kernel_dir_not_found");

    tq::inference_runtime_reset();
    auto st = tq::init_context();
    if (st != tq::TqStatus::OK) return print_self_test_error("inference-runtime-smoke", "init_context_failed");
    if (!load_self_test_kernels(kernel_dir)) {
        tq::shutdown_context();
        return print_self_test_error("inference-runtime-smoke", "kernel_load_failed");
    }

    const bool smoke_ok = tq::inference_runtime_run_smoke();
    std::string artifact_path;
    const bool artifact_ok = tq::inference_runtime_write_artifact(&artifact_path);
    const tq::InferenceRuntimeSummary summary = tq::inference_runtime_get_summary();

    std::ostringstream out;
    out << "{"
        << "\"status\":\"" << (smoke_ok && artifact_ok ? "ok" : "error") << "\","
        << "\"command\":\"inference-runtime-smoke\","
        << "\"available\":" << (summary.available ? "true" : "false") << ","
        << "\"artifactOk\":" << (artifact_ok ? "true" : "false") << ","
        << "\"artifactPath\":\"" << json_escape(artifact_path) << "\","
        << "\"prefillDecodeSplitReady\":" << (summary.prefill_decode_split_ready ? "true" : "false") << ","
        << "\"chunkedPrefillReady\":" << (summary.chunked_prefill_ready ? "true" : "false") << ","
        << "\"continuousBatchingReady\":" << (summary.continuous_batching_ready ? "true" : "false") << ","
        << "\"mixedPrefillDecodeReady\":" << (summary.mixed_prefill_decode_ready ? "true" : "false") << ","
        << "\"requestCount\":" << summary.request_count << ","
        << "\"chunkSizeTokens\":" << summary.chunk_size_tokens << ","
        << "\"maxBatchTokens\":" << summary.max_batch_tokens << ","
        << "\"totalPrefillTokens\":" << summary.total_prefill_tokens << ","
        << "\"totalDecodeTokens\":" << summary.total_decode_tokens << ","
        << "\"totalPrefillChunks\":" << summary.total_prefill_chunks << ","
        << "\"prefillWaveCount\":" << summary.prefill_wave_count << ","
        << "\"decodeWaveCount\":" << summary.decode_wave_count << ","
        << "\"mixedWaveCount\":" << summary.mixed_wave_count << ","
        << "\"schedulerTurns\":" << summary.scheduler_turns << ","
        << "\"maxContinuousBatchSize\":" << summary.max_continuous_batch_size << ","
        << "\"maxDecodeBatchSize\":" << summary.max_decode_batch_size << ","
        << "\"maxPrefillBatchSize\":" << summary.max_prefill_batch_size << ","
        << "\"maxPrefillChunkTokens\":" << summary.max_prefill_chunk_tokens << ","
        << "\"avgPrefillChunkTokens\":" << summary.avg_prefill_chunk_tokens << ","
        << "\"decodeQueuePeak\":" << summary.decode_queue_peak << ","
        << "\"prefillQueuePeak\":" << summary.prefill_queue_peak << ","
        << "\"decodePriorityTurns\":" << summary.decode_priority_turns << ","
        << "\"totalFusedAttentionNs\":" << summary.total_fused_attention_ns << ","
        << "\"totalQjlNs\":" << summary.total_qjl_ns << ","
        << "\"totalValueDequantNs\":" << summary.total_value_dequant_ns << ","
        << "\"prefillBackend\":\"" << json_escape(summary.prefill_backend) << "\","
        << "\"decodeBackend\":\"" << json_escape(summary.decode_backend) << "\","
        << "\"batchingPolicy\":\"" << json_escape(summary.batching_policy) << "\","
        << "\"chunkingPolicy\":\"" << json_escape(summary.chunking_policy) << "\","
        << "\"requests\":[";
    for (size_t i = 0; i < summary.requests.size(); ++i) {
        const auto& req = summary.requests[i];
        if (i > 0) out << ",";
        out << "{"
            << "\"requestId\":\"" << json_escape(req.request_id) << "\","
            << "\"promptTokens\":" << req.prompt_tokens << ","
            << "\"decodeTokens\":" << req.decode_tokens << ","
            << "\"prefillChunkSize\":" << req.prefill_chunk_size << ","
            << "\"prefillChunks\":" << req.prefill_chunks << ","
            << "\"prefillChunksExecuted\":" << req.prefill_chunks_executed << ","
            << "\"decodeStepsExecuted\":" << req.decode_steps_executed << ","
            << "\"peakContextTokens\":" << req.peak_context_tokens << ","
            << "\"continuousBatchSlotsSeen\":" << req.continuous_batch_slots_seen << ","
            << "\"enteredMixedWave\":" << (req.entered_mixed_wave ? "true" : "false") << ","
            << "\"prefillComplete\":" << (req.prefill_complete ? "true" : "false") << ","
            << "\"decodeComplete\":" << (req.decode_complete ? "true" : "false") << ","
            << "\"prefillBackend\":\"" << json_escape(req.prefill_backend) << "\","
            << "\"decodeBackend\":\"" << json_escape(req.decode_backend) << "\"}";
    }
    out << "],\"waves\":[";
    for (size_t i = 0; i < summary.waves.size(); ++i) {
        const auto& wave = summary.waves[i];
        if (i > 0) out << ",";
        out << "{"
            << "\"waveId\":" << wave.wave_id << ","
            << "\"phase\":\"" << json_escape(wave.phase) << "\","
            << "\"prefillRequestCount\":" << wave.prefill_request_count << ","
            << "\"decodeRequestCount\":" << wave.decode_request_count << ","
            << "\"batchedRequestCount\":" << wave.batched_request_count << ","
            << "\"batchedTokenCount\":" << wave.batched_token_count << ","
            << "\"prefillChunkTokens\":" << wave.prefill_chunk_tokens << ","
            << "\"decodeStepTokens\":" << wave.decode_step_tokens << ","
            << "\"decodeQueueDepth\":" << wave.decode_queue_depth << ","
            << "\"prefillQueueDepth\":" << wave.prefill_queue_depth << ","
            << "\"continuousBatching\":" << (wave.continuous_batching ? "true" : "false") << ","
            << "\"decodePrioritized\":" << (wave.decode_prioritized ? "true" : "false") << ","
            << "\"mixedWithRunningDecode\":" << (wave.mixed_with_running_decode ? "true" : "false") << ","
            << "\"fusedAttentionNs\":" << wave.fused_attention_ns << ","
            << "\"qjlNs\":" << wave.qjl_ns << ","
            << "\"valueDequantNs\":" << wave.value_dequant_ns << ","
            << "\"anchorKernel\":\"" << json_escape(wave.anchor_kernel) << "\","
            << "\"requestIds\":[";
        for (size_t j = 0; j < wave.request_ids.size(); ++j) {
            if (j > 0) out << ",";
            out << "\"" << json_escape(wave.request_ids[j]) << "\"";
        }
        out << "]}";
    }
    out << "]}";
    print_json_result(out.str());
    std::fflush(stdout);
    std::fflush(stderr);
    // This smoke lane is intentionally process-scoped: Rusticl/Mesa teardown
    // can assert after mixed multi-wave profiling even when the evidence
    // artifact and live kernel execution are already complete.
    std::_Exit((smoke_ok && artifact_ok) ? 0 : 1);
}

int run_paged_kv_prefix_cache_smoke() {
    std::string kernel_dir = tq::resolve_kernel_dir();
    if (kernel_dir.empty()) return print_self_test_error("paged-kv-prefix-cache-smoke", "kernel_dir_not_found");

    tq::paged_kv_prefix_cache_reset();
    auto st = tq::init_context();
    if (st != tq::TqStatus::OK) return print_self_test_error("paged-kv-prefix-cache-smoke", "init_context_failed");
    if (!load_self_test_kernels(kernel_dir)) {
        tq::shutdown_context();
        return print_self_test_error("paged-kv-prefix-cache-smoke", "kernel_load_failed");
    }

    const bool smoke_ok = tq::paged_kv_prefix_cache_run_smoke();
    std::string artifact_path;
    const bool artifact_ok = tq::paged_kv_prefix_cache_write_artifact(&artifact_path);
    const tq::PagedKvPrefixCacheSummary summary = tq::paged_kv_prefix_cache_get_summary();

    std::ostringstream out;
    out << "{"
        << "\"status\":\"" << (smoke_ok && artifact_ok ? "ok" : "error") << "\","
        << "\"command\":\"paged-kv-prefix-cache-smoke\","
        << "\"available\":" << (summary.available ? "true" : "false") << ","
        << "\"artifactOk\":" << (artifact_ok ? "true" : "false") << ","
        << "\"artifactPath\":\"" << json_escape(artifact_path) << "\","
        << "\"pagedKvAllocatorReady\":" << (summary.paged_kv_allocator_ready ? "true" : "false") << ","
        << "\"blockTableReady\":" << (summary.block_table_ready ? "true" : "false") << ","
        << "\"prefixCacheSharingReady\":" << (summary.prefix_cache_sharing_ready ? "true" : "false") << ","
        << "\"cacheEvictionReady\":" << (summary.cache_eviction_ready ? "true" : "false") << ","
        << "\"requestCount\":" << summary.request_count << ","
        << "\"blockSizeTokens\":" << summary.block_size_tokens << ","
        << "\"physicalBlockCount\":" << summary.physical_block_count << ","
        << "\"logicalBlockRefCount\":" << summary.logical_block_ref_count << ","
        << "\"sharedBlockCount\":" << summary.shared_block_count << ","
        << "\"uniqueBlockCount\":" << summary.unique_block_count << ","
        << "\"sharedPrefixGroupCount\":" << summary.shared_prefix_group_count << ","
        << "\"peakLiveBlockCount\":" << summary.peak_live_block_count << ","
        << "\"residentBlockCount\":" << summary.resident_block_count << ","
        << "\"evictableBlockCount\":" << summary.evictable_block_count << ","
        << "\"evictedBlockCount\":" << summary.evicted_block_count << ","
        << "\"maxBlockRefcount\":" << summary.max_block_refcount << ","
        << "\"totalFusedAttentionNs\":" << summary.total_fused_attention_ns << ","
        << "\"totalQjlNs\":" << summary.total_qjl_ns << ","
        << "\"totalValueDequantNs\":" << summary.total_value_dequant_ns << ","
        << "\"prefillBackend\":\"" << json_escape(summary.prefill_backend) << "\","
        << "\"decodeBackend\":\"" << json_escape(summary.decode_backend) << "\","
        << "\"sharingPolicy\":\"" << json_escape(summary.sharing_policy) << "\","
        << "\"evictionPolicy\":\"" << json_escape(summary.eviction_policy) << "\","
        << "\"evictionDecision\":\"" << json_escape(summary.eviction_decision) << "\","
        << "\"requests\":[";
    for (size_t i = 0; i < summary.requests.size(); ++i) {
        const auto& req = summary.requests[i];
        if (i > 0) out << ",";
        out << "{"
            << "\"requestId\":\"" << json_escape(req.request_id) << "\","
            << "\"promptTokens\":" << req.prompt_tokens << ","
            << "\"decodeTokens\":" << req.decode_tokens << ","
            << "\"logicalBlockCount\":" << req.logical_block_count << ","
            << "\"uniquePhysicalBlockCount\":" << req.unique_physical_block_count << ","
            << "\"sharedPhysicalBlockCount\":" << req.shared_physical_block_count << ","
            << "\"sharedPrefixBlocks\":" << req.shared_prefix_blocks << ","
            << "\"decodeBlockAppends\":" << req.decode_block_appends << ","
            << "\"peakContextTokens\":" << req.peak_context_tokens << ","
            << "\"physicalBlockIds\":[";
        for (size_t j = 0; j < req.physical_block_ids.size(); ++j) {
            if (j > 0) out << ",";
            out << req.physical_block_ids[j];
        }
        out << "]}";
    }
    out << "],\"blocks\":[";
    for (size_t i = 0; i < summary.blocks.size(); ++i) {
        const auto& block = summary.blocks[i];
        if (i > 0) out << ",";
        out << "{"
            << "\"physicalBlockId\":" << block.physical_block_id << ","
            << "\"tokenCount\":" << block.token_count << ","
            << "\"logicalRefCount\":" << block.logical_ref_count << ","
            << "\"refCount\":" << block.ref_count << ","
            << "\"lastAccessTick\":" << block.last_access_tick << ","
            << "\"sharedPrefix\":" << (block.shared_prefix ? "true" : "false") << ","
            << "\"live\":" << (block.live ? "true" : "false") << ","
            << "\"pinned\":" << (block.pinned ? "true" : "false") << ","
            << "\"evictable\":" << (block.evictable ? "true" : "false") << ","
            << "\"evicted\":" << (block.evicted ? "true" : "false") << ","
            << "\"cacheKey\":\"" << json_escape(block.cache_key) << "\","
            << "\"ownerRequestIds\":[";
        for (size_t j = 0; j < block.owner_request_ids.size(); ++j) {
            if (j > 0) out << ",";
            out << "\"" << json_escape(block.owner_request_ids[j]) << "\"";
        }
        out << "]}";
    }
    out << "]}";
    print_json_result(out.str());
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit((smoke_ok && artifact_ok) ? 0 : 1);
}
}

static void print_json_probe(const tq::ProbeResult& r) {
    printf("{\n");
    printf("  \"available\": %s,\n", r.available ? "true" : "false");
    printf("  \"platformCount\": %d,\n", r.platform_count);
    printf("  \"deviceCount\": %d,\n", r.device_count);
    printf("  \"devices\": [");
    for (size_t i = 0; i < r.devices.size(); i++) {
        auto& d = r.devices[i];
        if (i > 0) printf(",");
        printf("\n    {\"name\":\"%s\",\"vendor\":\"%s\",\"version\":\"%s\","
               "\"hasFp16\":%s,\"hasSubgroups\":%s,\"hasSubgroupShuffle\":%s,"
               "\"hasSubgroupShuffleRelative\":%s,\"hasSubgroupBallot\":%s,"
               "\"hasSubgroupClusteredReduce\":%s,\"hasSubgroupNonUniformArithmetic\":%s,"
               "\"hasSubgroupForwardProgress\":%s,"
               "\"maxSubgroups\":%u,\"hasIlProgram\":%s,\"hasSuggestedLocalWorkSize\":%s,"
               "\"hasCreateCommandQueue\":%s,\"hasInitializeMemory\":%s,\"hasDeviceUuid\":%s,"
               "\"deviceUuid\":\"%s\",\"hasPriorityHints\":%s,\"hasThrottleHints\":%s,"
               "\"hasCommandBuffer\":%s,\"hasCommandBufferMutableDispatch\":%s,"
               "\"hasExternalMemory\":%s,\"hasExternalMemoryAhb\":%s,\"hasExternalSemaphore\":%s,"
               "\"hasIntegerDotProduct\":%s,\"integerDotProductCapabilities\":%u,"
               "\"commandBufferCapabilities\":%llu,"
               "\"hasSvm\":%s,\"hasSvmCoarse\":%s,\"hasSvmFine\":%s,\"hasSvmAtomics\":%s,"
               "\"isAdreno\":%s,"
               "\"globalMemBytes\":%llu,\"computeUnits\":%u}",
               json_escape(d.name).c_str(), json_escape(d.vendor).c_str(), json_escape(d.version).c_str(),
               d.has_fp16 ? "true" : "false",
               d.has_subgroups ? "true" : "false",
               d.has_subgroup_shuffle ? "true" : "false",
               d.has_subgroup_shuffle_relative ? "true" : "false",
               d.has_subgroup_ballot ? "true" : "false",
               d.has_subgroup_clustered_reduce ? "true" : "false",
               d.has_subgroup_non_uniform_arithmetic ? "true" : "false",
               d.has_subgroup_forward_progress ? "true" : "false",
               d.max_subgroups,
               d.has_il_program ? "true" : "false",
               d.has_suggested_local_work_size ? "true" : "false",
               d.has_create_command_queue ? "true" : "false",
               d.has_initialize_memory ? "true" : "false",
               d.has_device_uuid ? "true" : "false",
               json_escape(d.device_uuid).c_str(),
               d.has_priority_hints ? "true" : "false",
               d.has_throttle_hints ? "true" : "false",
               d.has_command_buffer ? "true" : "false",
               d.has_command_buffer_mutable_dispatch ? "true" : "false",
               d.has_external_memory ? "true" : "false",
               d.has_external_memory_ahb ? "true" : "false",
               d.has_external_semaphore ? "true" : "false",
               d.has_integer_dot_product ? "true" : "false",
               d.integer_dot_product_capabilities,
               (unsigned long long)d.command_buffer_capabilities,
               d.has_svm ? "true" : "false",
               d.has_svm_coarse ? "true" : "false",
               d.has_svm_fine ? "true" : "false",
               d.has_svm_atomics ? "true" : "false",
               d.is_adreno ? "true" : "false",
               (unsigned long long)d.global_mem_bytes, d.compute_units);
    }
    printf("\n  ],\n");
    printf("  \"recommendedBackend\": \"%s\",\n", r.recommended_backend.c_str());
    printf("  \"probeTimeMs\": %.2f,\n", r.probe_time_ms);
    printf("  \"warnings\": [");
    for (size_t i = 0; i < r.warnings.size(); i++) {
        if (i > 0) printf(",");
        printf("\"%s\"", json_escape(r.warnings[i]).c_str());
    }
    printf("]\n}\n");
}

static void usage() {
    fprintf(stderr, "Usage: tq_opencl_cli <command> [options]\n");
    fprintf(stderr, "Commands:\n");
    fprintf(stderr, "  probe                              Detect OpenCL platform/devices\n");
    fprintf(stderr, "  frontier-smoke                     Exercise frontier runtime surfaces\n");
    fprintf(stderr, "  async-build-smoke                  Exercise async program compilation lane\n");
    fprintf(stderr, "  runtime-scheduler-smoke            Materialize scheduler/allocator state artifact\n");
    fprintf(stderr, "  inference-runtime-smoke            Materialize inference-runtime state artifact\n");
    fprintf(stderr, "  paged-kv-prefix-cache-smoke        Materialize paged-KV/prefix-cache state artifact\n");
    fprintf(stderr, "  benchmark [--warmup N] [--iters N] [--autotune] Run profiled benchmark suite\n");
    fprintf(stderr, "  mse-score --self-test [--spirv|--source]       Compute MSE attention scores\n");
    fprintf(stderr, "  qjl-score --self-test [--spirv|--source]       Compute QJL correction scores\n");
    fprintf(stderr, "  value-dequant --self-test [--spirv|--source]   Dequantize values\n");
    fprintf(stderr, "  fused-attention --self-test [--spirv|--source] Full fused decode\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string cmd = argv[1];
    bool force_source = false;
    bool prefer_spirv = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--source") force_source = true;
        if (std::string(argv[i]) == "--spirv") prefer_spirv = true;
    }

    if (force_source) {
        setenv("TQ_OPENCL_FORCE_SOURCE", "1", 1);
    } else if (prefer_spirv) {
        unsetenv("TQ_OPENCL_FORCE_SOURCE");
    }

    if (cmd == "probe") {
        auto result = tq::probe_opencl();
        print_json_probe(result);
        return result.available ? 0 : 1;
    }

    if (cmd == "frontier-smoke") {
        return run_frontier_smoke();
    }

    if (cmd == "system-svm-smoke") {
        return run_system_svm_smoke_command();
    }

    if (cmd == "async-build-smoke") {
        return run_async_build_smoke_command();
    }

    if (cmd == "runtime-scheduler-smoke") {
        return run_runtime_scheduler_smoke();
    }

    if (cmd == "inference-runtime-smoke") {
        return run_inference_runtime_smoke();
    }

    if (cmd == "paged-kv-prefix-cache-smoke") {
        return run_paged_kv_prefix_cache_smoke();
    }

    if (cmd == "benchmark") {
        return tq::run_benchmark(argc - 2, argv + 2);
    }

    if (cmd == "mse-score") {
        if (argc >= 3 && std::string(argv[2]) == "--self-test") {
            std::string kernel_dir = tq::resolve_kernel_dir();
            if (kernel_dir.empty()) return print_self_test_error("mse-score", "kernel_dir_not_found");
            auto st = tq::init_context();
            if (st != tq::TqStatus::OK) return print_self_test_error("mse-score", "init_context_failed");
            if (!load_self_test_kernels(kernel_dir)) return print_self_test_error("mse-score", "kernel_load_failed");
            int rc = run_mse_self_test();
            tq::release_all_programs();
            tq::shutdown_context();
            return rc;
        }
        fprintf(stderr, "mse-score requires --self-test\n");
        return 2;
    }

    if (cmd == "qjl-score" || cmd == "value-dequant" || cmd == "fused-attention") {
        if (argc >= 3 && std::string(argv[2]) == "--self-test") {
            std::string kernel_dir = tq::resolve_kernel_dir();
            if (kernel_dir.empty()) return print_self_test_error(cmd.c_str(), "kernel_dir_not_found");
            auto st = tq::init_context();
            if (st != tq::TqStatus::OK) return print_self_test_error(cmd.c_str(), "init_context_failed");
            if (!load_self_test_kernels(kernel_dir)) return print_self_test_error(cmd.c_str(), "kernel_load_failed");
            int rc = 1;
            if (cmd == "qjl-score") rc = run_qjl_self_test();
            else if (cmd == "value-dequant") rc = run_value_dequant_self_test();
            else rc = run_fused_attention_self_test();
            tq::release_all_programs();
            tq::shutdown_context();
            return rc;
        }
        fprintf(stderr, "%s requires --self-test\n", cmd.c_str());
        return 2;
    }

    fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
    usage();
    return 1;
}
