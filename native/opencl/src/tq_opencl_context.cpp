/**
 * TurboQuant OpenCL — context management with profiling queue.
 * Enhanced with GPU auto-profile integration.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl_types.h"
#include "../include/tq_gpu_profile.h"
#include "../include/tq_trace.h"
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <string>

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

bool query_subgroup_support(cl_device_id dev, const std::string& exts) {
    if (exts.find("cl_khr_subgroups") != std::string::npos) {
        return true;
    }
#if defined(CL_DEVICE_MAX_NUM_SUB_GROUPS)
    cl_uint max_subgroups = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_MAX_NUM_SUB_GROUPS, sizeof(max_subgroups), &max_subgroups, nullptr) ==
            CL_SUCCESS &&
        max_subgroups > 0) {
        return true;
    }
#endif
    return false;
}

bool query_external_memory_caps(cl_device_id dev) {
#if defined(CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR)
    size_t bytes = 0;
    return clGetDeviceInfo(dev, CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR, 0, nullptr, &bytes) == CL_SUCCESS &&
           bytes >= sizeof(cl_external_memory_handle_type_khr);
#else
    (void)dev;
    return false;
#endif
}

std::string hex_encode_uuid(const unsigned char* data, size_t size) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(size * 2);
    for (size_t i = 0; i < size; ++i) {
        out.push_back(hex[(data[i] >> 4) & 0xF]);
        out.push_back(hex[data[i] & 0xF]);
    }
    return out;
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
    bool has_subgroup_shuffle = false;
    bool has_subgroup_shuffle_relative = false;
    bool has_subgroup_ballot = false;
    bool has_subgroup_clustered_reduce = false;
    bool has_subgroup_non_uniform_arithmetic = false;
    bool has_il_program = false;
    bool has_async_program_compilation = false;
    bool has_integer_dot_product = false;
    bool has_suggested_local_work_size = false;
    bool has_create_command_queue = false;
    bool has_initialize_memory = false;
    bool has_device_uuid = false;
    bool has_priority_hints = false;
    bool has_throttle_hints = false;
    bool has_command_buffer = false;
    bool has_command_buffer_mutable_dispatch = false;
    bool has_external_memory = false;
    bool has_external_memory_ahb = false;
    bool has_external_semaphore = false;
    std::string device_uuid;
    uint64_t command_buffer_capabilities = 0;
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
bool device_has_async_program_compilation() { return g_ctx.has_async_program_compilation; }
bool device_has_suggested_local_work_size() { return g_ctx.has_suggested_local_work_size; }
bool device_has_create_command_queue() { return g_ctx.has_create_command_queue; }
bool device_has_initialize_memory() { return g_ctx.has_initialize_memory; }
bool device_has_device_uuid() { return g_ctx.has_device_uuid; }
bool device_has_priority_hints() { return g_ctx.has_priority_hints; }
bool device_has_throttle_hints() { return g_ctx.has_throttle_hints; }
bool device_has_command_buffer() { return g_ctx.has_command_buffer; }
bool device_has_command_buffer_mutable_dispatch() { return g_ctx.has_command_buffer_mutable_dispatch; }
bool device_has_external_memory() { return g_ctx.has_external_memory; }
bool device_has_external_memory_ahb() { return g_ctx.has_external_memory_ahb; }
bool device_has_external_semaphore() { return g_ctx.has_external_semaphore; }
std::string get_device_uuid() { return g_ctx.device_uuid; }
uint64_t get_command_buffer_capabilities() { return g_ctx.command_buffer_capabilities; }
bool device_has_svm() { return g_ctx.has_svm; }
bool device_has_svm_coarse() { return g_ctx.has_svm_coarse; }
bool device_has_svm_fine() { return g_ctx.has_svm_fine; }
bool device_has_svm_atomics() { return g_ctx.has_svm_atomics; }

bool query_kernel_suggested_local_work_size(
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_offset,
    const size_t* global_work_size,
    size_t* suggested_local_work_size) {
    if (!g_ctx.initialized || !g_ctx.platform || !g_ctx.queue || !kernel || !global_work_size || !suggested_local_work_size) {
        return false;
    }
    if (!g_ctx.has_suggested_local_work_size) {
        return false;
    }

    auto fn = reinterpret_cast<clGetKernelSuggestedLocalWorkSizeKHR_fn>(
        clGetExtensionFunctionAddressForPlatform(g_ctx.platform, "clGetKernelSuggestedLocalWorkSizeKHR"));
    if (!fn) {
        return false;
    }

    cl_int err = fn(g_ctx.queue, kernel, work_dim, global_work_offset, global_work_size, suggested_local_work_size);
    return err == CL_SUCCESS;
}

std::string get_default_build_opts() {
    std::string opts = get_build_opts(g_ctx.gpu_profile);
    if (g_ctx.has_subgroups) {
        opts += " -DUSE_SUBGROUPS=1";
    }
    if (g_ctx.has_integer_dot_product) {
        opts += " -DTQ_HAS_INTEGER_DOT_PRODUCT=1";
    }
    return opts;
}

TqStatus init_context() {
    if (g_ctx.initialized) return TqStatus::OK;
    trace_log("context", "init_context begin");

    cl_uint num_platforms = 0;
    cl_int err;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::fprintf(stderr, "[tq-opencl] clGetPlatformIDs failed err=%d num_platforms=%u\n", err, num_platforms);
        return TqStatus::ERR_NO_PLATFORM;
    }
    trace_log("context", "platform_count=%u", num_platforms);

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
    trace_log("context", "gpu_device_count=%u", num_devices);

    cl_device_id dev;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "[tq-opencl] clGetDeviceIDs(fetch) failed err=%d\n", err);
        return TqStatus::ERR_NO_DEVICE;
    }

    size_t ext_size = 0;
    std::string exts;
    if (clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size) == CL_SUCCESS && ext_size > 1) {
        exts.resize(ext_size);
        if (clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_size, exts.data(), nullptr) != CL_SUCCESS) {
            exts.clear();
        }
    }

    cl_context ctx;
    cl_context_properties ctx_props[5] = {0, 0, 0, 0, 0};
    const bool has_initialize_memory = exts.find("cl_khr_initialize_memory") != std::string::npos;
    const bool has_priority_hints = exts.find("cl_khr_priority_hints") != std::string::npos;
    const bool has_throttle_hints = exts.find("cl_khr_throttle_hints") != std::string::npos;
    const bool has_create_command_queue = exts.find("cl_khr_create_command_queue") != std::string::npos;
    trace_log(
        "context",
        "extension_gates init_mem=%d priority=%d throttle=%d create_queue=%d cmd_buf=%d ext_mem=%d ext_mem_ahb=%d ext_sema=%d",
        has_initialize_memory ? 1 : 0,
        has_priority_hints ? 1 : 0,
        has_throttle_hints ? 1 : 0,
        has_create_command_queue ? 1 : 0,
        exts.find("cl_khr_command_buffer") != std::string::npos ? 1 : 0,
        (exts.find("cl_khr_external_memory") != std::string::npos ||
         exts.find("cl_khr_external_memory_dma_buf") != std::string::npos) ? 1 : 0,
        exts.find("cl_khr_external_memory_android_hardware_buffer") != std::string::npos ? 1 : 0,
        exts.find("cl_khr_external_semaphore") != std::string::npos ? 1 : 0);
    size_t ctx_prop_count = 0;
#if defined(CL_CONTEXT_MEMORY_INITIALIZE_KHR)
    if (has_initialize_memory) {
        ctx_props[ctx_prop_count++] = CL_CONTEXT_MEMORY_INITIALIZE_KHR;
        ctx_props[ctx_prop_count++] =
            CL_CONTEXT_MEMORY_INITIALIZE_LOCAL_KHR | CL_CONTEXT_MEMORY_INITIALIZE_PRIVATE_KHR;
        trace_log("context", "context_memory_initialize requested flags=0x%x",
            CL_CONTEXT_MEMORY_INITIALIZE_LOCAL_KHR | CL_CONTEXT_MEMORY_INITIALIZE_PRIVATE_KHR);
    }
#endif
    ctx = clCreateContext(ctx_prop_count ? ctx_props : nullptr, 1, &dev, nullptr, nullptr, &err);
    if ((err != CL_SUCCESS || !ctx) && ctx_prop_count > 0) {
        trace_log("context", "context_create_with_initialize failed err=%d, retrying without extension", err);
        err = CL_SUCCESS;
        ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    }
    if (err != CL_SUCCESS || !ctx) {
        std::fprintf(stderr, "[tq-opencl] clCreateContext failed err=%d\n", err);
        return TqStatus::ERR_NO_PLATFORM;
    }

    // Create command queue with profiling enabled (OpenCL 2.0+ API)
    cl_queue_properties qprops[9] = {0};
    size_t qprop_count = 0;
    qprops[qprop_count++] = CL_QUEUE_PROPERTIES;
    qprops[qprop_count++] = CL_QUEUE_PROFILING_ENABLE;
#if defined(CL_QUEUE_PRIORITY_KHR)
    if (has_priority_hints) {
        qprops[qprop_count++] = CL_QUEUE_PRIORITY_KHR;
        qprops[qprop_count++] = CL_QUEUE_PRIORITY_MED_KHR;
    }
#endif
#if defined(CL_QUEUE_THROTTLE_KHR)
    if (has_throttle_hints) {
        qprops[qprop_count++] = CL_QUEUE_THROTTLE_KHR;
        qprops[qprop_count++] = CL_QUEUE_THROTTLE_MED_KHR;
    }
#endif
    qprops[qprop_count++] = 0;
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, qprops, &err);
    bool queue_profiling_enabled = true;
    trace_log("context", "queue_create_with_properties err=%d", err);
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
            trace_log("context", "fallback_queue_no_profiling err=%d", err);
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
    g_ctx.has_subgroups = query_subgroup_support(dev, exts);
    g_ctx.has_subgroup_shuffle = exts.find("cl_khr_subgroup_shuffle") != std::string::npos;
    g_ctx.has_subgroup_shuffle_relative = exts.find("cl_khr_subgroup_shuffle_relative") != std::string::npos;
    g_ctx.has_subgroup_ballot = exts.find("cl_khr_subgroup_ballot") != std::string::npos;
    g_ctx.has_subgroup_clustered_reduce = exts.find("cl_khr_subgroup_clustered_reduce") != std::string::npos;
    g_ctx.has_subgroup_non_uniform_arithmetic =
        exts.find("cl_khr_subgroup_non_uniform_arithmetic") != std::string::npos;
    g_ctx.has_il_program =
        (!il_versions.empty() && il_versions.find("SPIR-V") != std::string::npos) ||
        exts.find("cl_khr_il_program") != std::string::npos;
    g_ctx.has_async_program_compilation =
        exts.find("cl_khr_async_program_compilation") != std::string::npos;
    g_ctx.has_integer_dot_product = exts.find("cl_khr_integer_dot_product") != std::string::npos;
    g_ctx.has_suggested_local_work_size = exts.find("cl_khr_suggested_local_work_size") != std::string::npos;
    g_ctx.has_create_command_queue = has_create_command_queue;
    g_ctx.has_initialize_memory = has_initialize_memory;
    g_ctx.has_device_uuid = exts.find("cl_khr_device_uuid") != std::string::npos;
    g_ctx.has_priority_hints = has_priority_hints;
    g_ctx.has_throttle_hints = has_throttle_hints;
    g_ctx.has_command_buffer = exts.find("cl_khr_command_buffer") != std::string::npos;
    g_ctx.has_command_buffer_mutable_dispatch =
        exts.find("cl_khr_command_buffer_mutable_dispatch") != std::string::npos;
    g_ctx.has_external_memory =
        exts.find("cl_khr_external_memory") != std::string::npos ||
        exts.find("cl_khr_external_memory_dma_buf") != std::string::npos ||
        query_external_memory_caps(dev);
    g_ctx.has_external_memory_ahb =
        exts.find("cl_khr_external_memory_android_hardware_buffer") != std::string::npos;
    g_ctx.has_external_semaphore =
        exts.find("cl_khr_external_semaphore") != std::string::npos;
#if defined(CL_DEVICE_UUID_KHR) && defined(CL_UUID_SIZE_KHR)
    if (g_ctx.has_device_uuid) {
        unsigned char uuid[CL_UUID_SIZE_KHR] = {};
        if (clGetDeviceInfo(dev, CL_DEVICE_UUID_KHR, sizeof(uuid), uuid, nullptr) == CL_SUCCESS) {
            g_ctx.device_uuid = hex_encode_uuid(uuid, sizeof(uuid));
        }
    }
#endif
#if defined(CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR)
    if (g_ctx.has_command_buffer) {
        cl_device_command_buffer_capabilities_khr caps = 0;
        const cl_int err =
            clGetDeviceInfo(dev, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR, sizeof(caps), &caps, nullptr);
        if (err == CL_SUCCESS) {
            g_ctx.command_buffer_capabilities = static_cast<uint64_t>(caps);
        } else {
            trace_log(
                "context",
                "command_buffer_caps_query err=%d has_command_buffer=%d",
                static_cast<int>(err),
                g_ctx.has_command_buffer ? 1 : 0);
        }
    }
#endif
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
    trace_log(
        "context",
        "device=%s compute_units=%u max_wg=%zu profiling=%d fp16=%d subgroups=%d il=%d async_build=%d suggested_lws=%d create_queue=%d init_mem=%d uuid=%d cmd_buf=%d ext_mem=%d ext_sema=%d svm=%d coarse=%d fine=%d atomics=%d",
        name_buf,
        g_ctx.compute_units,
        g_ctx.max_wg_size,
        g_ctx.profiling_enabled ? 1 : 0,
        g_ctx.has_fp16 ? 1 : 0,
        g_ctx.has_subgroups ? 1 : 0,
        g_ctx.has_il_program ? 1 : 0,
        g_ctx.has_async_program_compilation ? 1 : 0,
        g_ctx.has_suggested_local_work_size ? 1 : 0,
        g_ctx.has_create_command_queue ? 1 : 0,
        g_ctx.has_initialize_memory ? 1 : 0,
        g_ctx.has_device_uuid ? 1 : 0,
        g_ctx.has_command_buffer ? 1 : 0,
        g_ctx.has_external_memory ? 1 : 0,
        g_ctx.has_external_semaphore ? 1 : 0,
        g_ctx.has_svm ? 1 : 0,
        g_ctx.has_svm_coarse ? 1 : 0,
        g_ctx.has_svm_fine ? 1 : 0,
        g_ctx.has_svm_atomics ? 1 : 0);

    return TqStatus::OK;
}

void shutdown_context() {
    if (!g_ctx.initialized) return;
    clReleaseCommandQueue(g_ctx.queue);
    clReleaseContext(g_ctx.context);
    g_ctx = OpenClContext{};
}

} // namespace tq
