/**
 * TurboQuant OpenCL — context management with profiling queue.
 * Enhanced with GPU auto-profile integration.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl_types.h"
#include "../include/tq_gpu_profile.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstring>

namespace tq {

namespace {

cl_command_queue create_legacy_command_queue(
    cl_context ctx,
    cl_device_id dev,
    cl_command_queue_properties props,
    cl_int* err) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, props, err);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return queue;
}

} // namespace

struct OpenClContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    bool initialized = false;
    bool is_adreno = false;
    uint32_t compute_units = 0;
    size_t max_wg_size = 0;
    bool profiling_enabled = false;
    bool has_fp16 = false;
    bool has_subgroups = false;
    bool has_il_program = false;
    bool has_svm = false;
    bool has_svm_coarse = false;
    bool has_svm_fine = false;
    bool has_svm_atomics = false;
    GpuProfile gpu_profile = {};
};

static OpenClContext g_ctx;

cl_platform_id get_platform() { return g_ctx.platform; }
cl_device_id get_device() { return g_ctx.device; }
cl_context get_context() { return g_ctx.context; }
cl_command_queue get_queue() { return g_ctx.queue; }
bool is_initialized() { return g_ctx.initialized; }
bool is_adreno() { return g_ctx.is_adreno; }
uint32_t get_compute_units() { return g_ctx.compute_units; }
size_t get_max_wg_size() { return g_ctx.max_wg_size; }
bool profiling_enabled() { return g_ctx.profiling_enabled; }
const GpuProfile& get_gpu_profile() { return g_ctx.gpu_profile; }
bool device_has_fp16() { return g_ctx.has_fp16; }
bool device_has_subgroups() { return g_ctx.has_subgroups; }
bool device_has_il_program() { return g_ctx.has_il_program; }
bool device_has_svm() { return g_ctx.has_svm; }
bool device_has_svm_coarse() { return g_ctx.has_svm_coarse; }
bool device_has_svm_fine() { return g_ctx.has_svm_fine; }
bool device_has_svm_atomics() { return g_ctx.has_svm_atomics; }

std::string get_default_build_opts() {
    std::string opts;
    if (g_ctx.has_fp16) {
        opts += " -DTQ_HAS_FP16=1";
    }
    if (g_ctx.has_subgroups) {
        opts += " -DUSE_SUBGROUPS=1";
    }
    return opts;
}

TqStatus init_context() {
    if (g_ctx.initialized) return TqStatus::OK;

    cl_uint num_platforms = 0;
    cl_int err;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::fprintf(stderr, "[tq-opencl] clGetPlatformIDs failed err=%d num_platforms=%u\n", err, num_platforms);
        return TqStatus::ERR_NO_PLATFORM;
    }

    cl_platform_id plat;
    err = clGetPlatformIDs(1, &plat, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[tq-opencl] clGetPlatformIDs(fetch) failed err=%d\n", err);
        return TqStatus::ERR_NO_PLATFORM;
    }

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        std::fprintf(stderr, "[tq-opencl] clGetDeviceIDs(count) failed err=%d num_devices=%u\n", err, num_devices);
        return TqStatus::ERR_NO_DEVICE;
    }

    cl_device_id dev;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[tq-opencl] clGetDeviceIDs(fetch) failed err=%d\n", err);
        return TqStatus::ERR_NO_DEVICE;
    }

    cl_context ctx;
    ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !ctx) {
        std::fprintf(stderr, "[tq-opencl] clCreateContext failed err=%d\n", err);
        return TqStatus::ERR_NO_PLATFORM;
    }

    // Create command queue with profiling enabled (OpenCL 2.0+ API)
    cl_queue_properties qprops[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, qprops, &err);
    bool queue_profiling_enabled = true;
    if (err != CL_SUCCESS) {
#if defined(CL_TARGET_OPENCL_VERSION) && CL_TARGET_OPENCL_VERSION >= 120
        err = CL_SUCCESS;
        queue = create_legacy_command_queue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
#else
        queue = nullptr;
#endif
        if (err != CL_SUCCESS || !queue) {
#if defined(CL_TARGET_OPENCL_VERSION) && CL_TARGET_OPENCL_VERSION >= 120
            err = CL_SUCCESS;
            queue = create_legacy_command_queue(ctx, dev, 0, &err);
#endif
            queue_profiling_enabled = false;
            if (err != CL_SUCCESS || !queue) {
                clReleaseContext(ctx);
                return TqStatus::ERR_NO_PLATFORM;
            }
        }
    }

    // Query device info
    char name_buf[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr);

    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);

    size_t wg = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);

    size_t ext_size = 0;
    std::string exts;
    if (clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size) == CL_SUCCESS && ext_size > 1) {
        exts.resize(ext_size);
        if (clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_size, exts.data(), nullptr) != CL_SUCCESS) {
            exts.clear();
        }
    }

    size_t il_size = 0;
    std::string il_versions;
#if defined(CL_DEVICE_IL_VERSION)
    if (clGetDeviceInfo(dev, CL_DEVICE_IL_VERSION, 0, nullptr, &il_size) == CL_SUCCESS && il_size > 1) {
        il_versions.resize(il_size);
        if (clGetDeviceInfo(dev, CL_DEVICE_IL_VERSION, il_size, il_versions.data(), nullptr) != CL_SUCCESS) {
            il_versions.clear();
        }
    }
#endif

    cl_device_svm_capabilities svm_caps = 0;
#if defined(CL_DEVICE_SVM_CAPABILITIES)
    (void)clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
#endif

    g_ctx.platform = plat;
    g_ctx.device = dev;
    g_ctx.context = ctx;
    g_ctx.queue = queue;
    g_ctx.initialized = true;
    g_ctx.is_adreno = (strstr(name_buf, "Adreno") != nullptr || strstr(name_buf, "QUALCOMM") != nullptr);
    g_ctx.compute_units = cu;
    g_ctx.max_wg_size = wg;
    g_ctx.profiling_enabled = queue_profiling_enabled;
    g_ctx.has_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
    g_ctx.has_subgroups = exts.find("cl_khr_subgroups") != std::string::npos;
    g_ctx.has_il_program =
        (!il_versions.empty() && il_versions.find("SPIR-V") != std::string::npos) ||
        exts.find("cl_khr_il_program") != std::string::npos;
    g_ctx.has_svm_coarse = (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) != 0;
    g_ctx.has_svm_fine = (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
    g_ctx.has_svm_atomics =
        (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) != 0 ||
        (svm_caps & CL_DEVICE_SVM_ATOMICS) != 0;
    g_ctx.has_svm = g_ctx.has_svm_coarse || g_ctx.has_svm_fine;

    // Detect GPU profile for per-kernel tuning
    g_ctx.gpu_profile = detect_gpu_profile();
    // Override max_wg_size from profile if profile detected a known GPU
    if (g_ctx.gpu_profile.gen != GpuGen::UNKNOWN && g_ctx.gpu_profile.preferred_wg_size > 0) {
        g_ctx.max_wg_size = (g_ctx.max_wg_size > 0)
            ? g_ctx.max_wg_size  // Keep hardware limit
            : g_ctx.gpu_profile.preferred_wg_size;
    }

    return TqStatus::OK;
}

void shutdown_context() {
    if (!g_ctx.initialized) return;
    clReleaseCommandQueue(g_ctx.queue);
    clReleaseContext(g_ctx.context);
    g_ctx = OpenClContext{};
}

} // namespace tq
