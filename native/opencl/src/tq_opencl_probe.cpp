/**
 * TurboQuant OpenCL — platform/device probe.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl_probe.h"
#include <chrono>
#include <cstring>

#ifdef TQ_OPENCL_DYNAMIC_LOAD
#include <dlfcn.h>
static void* cl_lib = nullptr;
static bool try_load_opencl() {
    const char* paths[] = {
        "/system/vendor/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL_adreno.so",
        nullptr
    };
    for (int i = 0; paths[i]; i++) {
        cl_lib = dlopen(paths[i], RTLD_LAZY);
        if (cl_lib) return true;
    }
    return false;
}
#else
#include <CL/cl.h>
#endif

namespace tq {

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
    fn_getPlatformIDs(0, nullptr, &num_platforms);
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

#ifndef TQ_OPENCL_DYNAMIC_LOAD
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
            info.has_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
            info.has_subgroups = exts.find("cl_khr_subgroups") != std::string::npos;
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
    result.recommended_backend = adreno ? "opencl_adreno" : "opencl_generic";

    auto end = std::chrono::steady_clock::now();
    result.probe_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

} // namespace tq
