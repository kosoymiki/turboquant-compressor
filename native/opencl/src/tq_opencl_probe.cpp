/**
 * TurboQuant OpenCL — platform/device probe.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl_probe.h"
#include "../include/tq_repo_paths.h"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sstream>

static std::string cl_loaded_path;

namespace {
bool custom_stack_only() {
    const char* env = getenv("TQ_OPENCL_CUSTOM_STACK_ONLY");
    return !env || std::strcmp(env, "0") != 0;
}

bool should_silence_opencl_stderr() {
    const char* env = std::getenv("TQ_OPENCL_SILENCE_STDERR");
    return env && std::strcmp(env, "1") == 0;
}

struct StderrSilencer {
    int saved_fd = -1;
    int null_fd = -1;
    bool active = false;

    StderrSilencer() {
        if (!should_silence_opencl_stderr()) return;
        saved_fd = dup(STDERR_FILENO);
        if (saved_fd < 0) return;
        null_fd = open("/dev/null", O_WRONLY | O_CLOEXEC);
        if (null_fd < 0) return;
        if (dup2(null_fd, STDERR_FILENO) < 0) return;
        active = true;
    }

    ~StderrSilencer() {
        if (active) dup2(saved_fd, STDERR_FILENO);
        if (saved_fd >= 0) close(saved_fd);
        if (null_fd >= 0) close(null_fd);
    }
};
}

#ifdef TQ_OPENCL_DYNAMIC_LOAD
#include <dlfcn.h>
static void* cl_lib = nullptr;

static bool try_load_opencl() {
    std::vector<std::string> paths;
    for (const auto& root : tq::runtime_pack_roots()) {
        paths.emplace_back(tq::join_repo_path(root, "layer1-compute/libRusticlOpenCL.so.1"));
        paths.emplace_back(tq::join_repo_path(root, "layer1-compute/libRusticlOpenCL.so"));
        paths.emplace_back(tq::join_repo_path(root, "libRusticlOpenCL.so.1"));
    }
    const char* vendor_paths[] = {
        "/system/vendor/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL_adreno.so",
        nullptr
    };
    for (const auto& path : paths) {
        cl_lib = dlopen(path.c_str(), RTLD_LAZY);
        if (cl_lib) {
            cl_loaded_path = path;
            return true;
        }
    }
    if (!custom_stack_only()) {
        for (int i = 0; vendor_paths[i]; i++) {
            cl_lib = dlopen(vendor_paths[i], RTLD_LAZY);
            if (cl_lib) {
                cl_loaded_path = vendor_paths[i];
                return true;
            }
        }
    }
    return false;
}
#else
#include <CL/cl.h>
#endif

namespace tq {

namespace {

void populate_extension_flags(DeviceInfo& info, const std::string& exts) {
    info.has_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
    info.has_il_program = exts.find("cl_khr_il_program") != std::string::npos;
    info.has_integer_dot_product = exts.find("cl_khr_integer_dot_product") != std::string::npos;
    info.has_suggested_local_work_size = exts.find("cl_khr_suggested_local_work_size") != std::string::npos;
    info.has_create_command_queue = exts.find("cl_khr_create_command_queue") != std::string::npos;
    info.has_initialize_memory = exts.find("cl_khr_initialize_memory") != std::string::npos;
    info.has_device_uuid = exts.find("cl_khr_device_uuid") != std::string::npos;
    info.has_priority_hints = exts.find("cl_khr_priority_hints") != std::string::npos;
    info.has_throttle_hints = exts.find("cl_khr_throttle_hints") != std::string::npos;
    info.has_command_buffer = exts.find("cl_khr_command_buffer") != std::string::npos;
    info.has_command_buffer_mutable_dispatch =
        exts.find("cl_khr_command_buffer_mutable_dispatch") != std::string::npos;
    info.has_external_memory =
        exts.find("cl_khr_external_memory") != std::string::npos ||
        exts.find("cl_khr_external_memory_dma_buf") != std::string::npos;
    info.has_external_memory_ahb =
        exts.find("cl_khr_external_memory_android_hardware_buffer") != std::string::npos;
    info.has_external_semaphore = exts.find("cl_khr_external_semaphore") != std::string::npos;
    info.has_subgroup_ballot = exts.find("cl_khr_subgroup_ballot") != std::string::npos;
    info.has_subgroup_clustered_reduce = exts.find("cl_khr_subgroup_clustered_reduce") != std::string::npos;
    info.has_subgroup_non_uniform_arithmetic =
        exts.find("cl_khr_subgroup_non_uniform_arithmetic") != std::string::npos;
    info.has_subgroup_shuffle = exts.find("cl_khr_subgroup_shuffle") != std::string::npos;
    info.has_subgroup_shuffle_relative = exts.find("cl_khr_subgroup_shuffle_relative") != std::string::npos;
    info.has_async_program_compilation =
        exts.find("cl_khr_async_program_compilation") != std::string::npos;
}

void query_uuid_if_available(cl_device_id dev, DeviceInfo& info) {
#if defined(CL_DEVICE_UUID_KHR) && defined(CL_UUID_SIZE_KHR)
    if (!info.has_device_uuid) return;
    unsigned char uuid[CL_UUID_SIZE_KHR] = {};
    if (clGetDeviceInfo(dev, CL_DEVICE_UUID_KHR, sizeof(uuid), uuid, nullptr) == CL_SUCCESS) {
        info.device_uuid = hex_encode_bytes(uuid, sizeof(uuid));
        info.has_device_uuid = !info.device_uuid.empty();
    }
#else
    (void)dev;
    (void)info;
#endif
}

void query_command_buffer_caps_if_available(cl_device_id dev, DeviceInfo& info) {
#if defined(CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR)
    if (!info.has_command_buffer) return;
    cl_device_command_buffer_capabilities_khr caps = 0;
    const cl_int err =
        clGetDeviceInfo(dev, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR, sizeof(caps), &caps, nullptr);
    if (err == CL_SUCCESS) {
        info.command_buffer_capabilities = static_cast<uint64_t>(caps);
    } else {
        std::fprintf(stderr,
                     "[tq-probe] command_buffer_caps_query err=%d has_command_buffer=%d\n",
                     static_cast<int>(err),
                     info.has_command_buffer ? 1 : 0);
    }
#else
    (void)dev;
    (void)info;
#endif
}

void query_integer_dot_product_caps_if_available(cl_device_id dev, DeviceInfo& info) {
#if defined(CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR)
    cl_device_integer_dot_product_capabilities_khr caps = 0;
    const cl_int err = clGetDeviceInfo(
        dev,
        CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
        sizeof(caps),
        &caps,
        nullptr);
    if (err == CL_SUCCESS) {
        info.integer_dot_product_capabilities = static_cast<uint32_t>(caps);
        if (caps != 0) info.has_integer_dot_product = true;
    }
#else
    (void)dev;
    (void)info;
#endif
}

void query_external_handle_caps(cl_device_id dev, DeviceInfo& info) {
#if defined(CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR)
    size_t bytes = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR, 0, nullptr, &bytes) == CL_SUCCESS &&
        bytes >= sizeof(cl_external_memory_handle_type_khr)) {
        info.has_external_memory = true;
    }
#else
    (void)dev;
#endif

#if defined(CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR)
    size_t bytes = 0;
    if (clGetDeviceInfo(dev, CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR, 0, nullptr, &bytes) == CL_SUCCESS &&
        bytes >= sizeof(cl_external_semaphore_handle_type_khr)) {
        info.has_external_semaphore = true;
    }
#endif
#if !defined(CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR) && !defined(CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR)
    (void)dev;
    (void)info;
#endif
}

} // namespace

static bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool path_list_contains(const char* path_list, const char* needle) {
    if (!path_list || !needle || !*needle) return false;
    return std::strstr(path_list, needle) != nullptr;
}

static bool has_rusticl_icd_contract() {
    const char* driver_root = std::getenv("TQ_DRIVER_ROOT");
    const char* ocl_icd_vendors = std::getenv("OCL_ICD_VENDORS");
    const char* ld_library_path = std::getenv("LD_LIBRARY_PATH");

    if (driver_root && *driver_root) {
        const std::string root(driver_root);
        if (file_exists(root + "/layer1-compute/rusticl.icd") &&
            file_exists(root + "/layer1-compute/libRusticlOpenCL.so.1") &&
            path_list_contains(ocl_icd_vendors, "/layer1-compute") &&
            path_list_contains(ld_library_path, "/layer1-compute")) {
            return true;
        }
    }

    if (ocl_icd_vendors && *ocl_icd_vendors) {
        const std::string vendors(ocl_icd_vendors);
        if ((file_exists(vendors + "/rusticl.icd") ||
             vendors.find("rusticl.icd") != std::string::npos) &&
            path_list_contains(ld_library_path, "libRusticlOpenCL")) {
            return true;
        }
    }

    return false;
}

static bool is_custom_driver_path(const std::string& path) {
    if (path.find("libRusticlOpenCL.so") != std::string::npos) return true;
    const char* driver_root = std::getenv("TQ_DRIVER_ROOT");
    if (driver_root && *driver_root) {
        const std::string root(driver_root);
        if (path.rfind(root, 0) == 0) return true;
    }
    return false;
}

static bool detect_subgroup_support(cl_device_id dev, const std::string& exts, uint32_t* max_subgroups_out,
                                    bool* forward_progress_out) {
    bool has_subgroups = exts.find("cl_khr_subgroups") != std::string::npos;
    cl_uint max_subgroups = 0;
    cl_bool forward_progress = CL_FALSE;
#if defined(CL_DEVICE_MAX_NUM_SUB_GROUPS)
    (void)clGetDeviceInfo(dev, CL_DEVICE_MAX_NUM_SUB_GROUPS, sizeof(max_subgroups), &max_subgroups, nullptr);
    if (max_subgroups > 0) has_subgroups = true;
#endif
#if defined(CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS)
    (void)clGetDeviceInfo(dev,
                          CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS,
                          sizeof(forward_progress),
                          &forward_progress,
                          nullptr);
#endif
    if (max_subgroups_out) *max_subgroups_out = max_subgroups;
    if (forward_progress_out) *forward_progress_out = (forward_progress == CL_TRUE);
    return has_subgroups;
}

ProbeResult probe_opencl() {
    auto start = std::chrono::steady_clock::now();
    ProbeResult result;

#ifdef TQ_OPENCL_DYNAMIC_LOAD
    if (!try_load_opencl()) {
        result.available = false;
        result.recommended_backend = "unavailable";
        result.warnings.push_back("No OpenCL library found via dlopen");
        auto end = std::chrono::steady_clock::now();
        result.probe_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return result;
    }
    // Dynamic function pointers
    using PFN_clGetPlatformIDs = int(*)(unsigned int, void**, unsigned int*);
    using PFN_clGetPlatformInfo = int(*)(void*, unsigned int, size_t, void*, size_t*);
    using PFN_clGetDeviceIDs = int(*)(void*, unsigned long long, unsigned int, void**, unsigned int*);
    using PFN_clGetDeviceInfo = int(*)(void*, unsigned int, size_t, void*, size_t*);

    auto fn_getPlatformIDs = (PFN_clGetPlatformIDs)dlsym(cl_lib, "clGetPlatformIDs");
    auto fn_getPlatformInfo = (PFN_clGetPlatformInfo)dlsym(cl_lib, "clGetPlatformInfo");
    auto fn_getDeviceIDs = (PFN_clGetDeviceIDs)dlsym(cl_lib, "clGetDeviceIDs");
    auto fn_getDeviceInfo = (PFN_clGetDeviceInfo)dlsym(cl_lib, "clGetDeviceInfo");

    if (!fn_getPlatformIDs || !fn_getDeviceIDs || !fn_getDeviceInfo) {
        result.available = false;
        result.recommended_backend = "unavailable";
        result.warnings.push_back("OpenCL symbols not found in library");
        auto end = std::chrono::steady_clock::now();
        result.probe_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return result;
    }

    // Use dynamic pointers
    unsigned int num_platforms = 0;
    {
        StderrSilencer silencer;
        fn_getPlatformIDs(0, nullptr, &num_platforms);
    }
#else
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
#endif

    if (num_platforms == 0) {
        result.available = false;
        result.recommended_backend = "unavailable";
        result.warnings.push_back("No OpenCL platforms found");
        auto end = std::chrono::steady_clock::now();
        result.probe_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return result;
    }

    result.available = true;
    result.platform_count = (int)num_platforms;

#ifdef TQ_OPENCL_DYNAMIC_LOAD
    std::vector<void*> platforms(num_platforms);
    {
        StderrSilencer silencer;
        fn_getPlatformIDs(num_platforms, platforms.data(), nullptr);
    }

    for (auto& plat : platforms) {
        unsigned int num_devices = 0;
        {
            StderrSilencer silencer;
            fn_getDeviceIDs(plat, 4ULL, 0, nullptr, &num_devices);
        }
        if (num_devices == 0) continue;

        std::vector<void*> devices(num_devices);
        {
            StderrSilencer silencer;
            fn_getDeviceIDs(plat, 4ULL, num_devices, devices.data(), nullptr);
        }

        for (auto& dev : devices) {
            DeviceInfo info;
            char buf[256];

            fn_getDeviceInfo(dev, 0x102B, sizeof(buf), buf, nullptr);
            info.name = buf;
            fn_getDeviceInfo(dev, 0x102C, sizeof(buf), buf, nullptr);
            info.vendor = buf;
            fn_getDeviceInfo(dev, 0x102F, sizeof(buf), buf, nullptr);
            info.version = buf;

            unsigned long long mem = 0;
            fn_getDeviceInfo(dev, 0x101F, sizeof(mem), &mem, nullptr);
            info.global_mem_bytes = mem;

            size_t wg = 0;
            fn_getDeviceInfo(dev, 0x1004, sizeof(wg), &wg, nullptr);
            info.max_work_group_size = (uint32_t)wg;

            unsigned int cu = 0;
            fn_getDeviceInfo(dev, 0x1002, sizeof(cu), &cu, nullptr);
            info.compute_units = cu;

            char ext_buf[4096] = {};
            fn_getDeviceInfo(dev, 0x1030, sizeof(ext_buf), ext_buf, nullptr);
            std::string exts(ext_buf);
            populate_extension_flags(info, exts);
            info.has_subgroups =
                detect_subgroup_support((cl_device_id)dev, exts, &info.max_subgroups, &info.has_subgroup_forward_progress);
            query_uuid_if_available((cl_device_id)dev, info);
            query_external_handle_caps((cl_device_id)dev, info);
            query_command_buffer_caps_if_available((cl_device_id)dev, info);
            query_integer_dot_product_caps_if_available((cl_device_id)dev, info);
            info.is_adreno = (info.name.find("Adreno") != std::string::npos ||
                              info.vendor.find("QUALCOMM") != std::string::npos ||
                              info.vendor.find("Qualcomm") != std::string::npos ||
                              info.name.find("FD") != std::string::npos);

            result.devices.push_back(info);
            result.device_count++;
        }
    }
#else
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (auto& plat : platforms) {
        cl_uint num_devices = 0;
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);

        for (auto& dev : devices) {
            DeviceInfo info;
            char buf[256];

            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
            info.name = buf;

            clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(buf), buf, nullptr);
            info.vendor = buf;

            clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(buf), buf, nullptr);
            info.version = buf;

            cl_ulong mem = 0;
            clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
            info.global_mem_bytes = mem;

            size_t wg = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);
            info.max_work_group_size = (uint32_t)wg;

            cl_uint cu = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
            info.compute_units = cu;

            // Check extensions
            char ext_buf[4096] = {};
            clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(ext_buf), ext_buf, nullptr);
            std::string exts(ext_buf);
            populate_extension_flags(info, exts);
            info.has_subgroups =
                detect_subgroup_support(dev, exts, &info.max_subgroups, &info.has_subgroup_forward_progress);
#if defined(CL_DEVICE_IL_VERSION)
            size_t il_size = 0;
            if (clGetDeviceInfo(dev, CL_DEVICE_IL_VERSION, 0, nullptr, &il_size) == CL_SUCCESS && il_size > 1) {
                std::string il_versions(il_size, '\0');
                if (clGetDeviceInfo(dev, CL_DEVICE_IL_VERSION, il_size, il_versions.data(), nullptr) == CL_SUCCESS) {
                    info.has_il_program = info.has_il_program || il_versions.find("SPIR-V") != std::string::npos;
                }
            }
#endif
            query_uuid_if_available(dev, info);
            query_external_handle_caps(dev, info);
            query_command_buffer_caps_if_available(dev, info);
            query_integer_dot_product_caps_if_available(dev, info);
#if defined(CL_DEVICE_SVM_CAPABILITIES)
            cl_device_svm_capabilities svm_caps = 0;
            if (clGetDeviceInfo(dev, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr) == CL_SUCCESS) {
                info.has_svm_coarse = (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) != 0;
                info.has_svm_fine = (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) != 0;
                info.has_svm_atomics =
                    (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) != 0 ||
                    (svm_caps & CL_DEVICE_SVM_ATOMICS) != 0;
                info.has_svm = info.has_svm_coarse || info.has_svm_fine;
            }
#endif
            info.is_adreno = (info.name.find("Adreno") != std::string::npos ||
                              info.vendor.find("QUALCOMM") != std::string::npos ||
                              info.vendor.find("Qualcomm") != std::string::npos);

            result.devices.push_back(info);
            result.device_count++;
        }
    }
#endif

    bool adreno = false;
    for (auto& d : result.devices) {
        if (d.is_adreno) { adreno = true; break; }
    }
    const bool custom_driver_loaded = is_custom_driver_path(cl_loaded_path);
    const bool rusticl_icd_contract = has_rusticl_icd_contract();
    const bool vendor_driver_loaded = !cl_loaded_path.empty() && !custom_driver_loaded;
    if (custom_driver_loaded || rusticl_icd_contract) {
        result.recommended_backend = "mesa_rusticl_kgsl";
    } else if (vendor_driver_loaded) {
        result.recommended_backend = "opencl_generic";
        result.warnings.push_back("Vendor/system OpenCL loaded; diagnostic evidence only, not production route");
    } else if (adreno) {
        result.recommended_backend = "opencl_generic";
        result.warnings.push_back("Adreno detected without custom Rusticl path proof; keep as diagnostic-only");
    } else {
        result.recommended_backend = "opencl_generic";
    }
    if (custom_stack_only() && !cl_loaded_path.empty() &&
        cl_loaded_path.find("RusticlOpenCL") == std::string::npos) {
        result.warnings.push_back("Custom-stack policy active but loaded OpenCL library is not Rusticl");
    }

    auto end = std::chrono::steady_clock::now();
    result.probe_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

} // namespace tq
