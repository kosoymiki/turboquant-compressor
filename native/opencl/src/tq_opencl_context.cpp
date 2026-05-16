/**
 * TurboQuant OpenCL — context management with profiling queue.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "tq_opencl_types.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstring>

namespace tq {

struct OpenClContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    bool initialized = false;
    bool is_adreno = false;
    uint32_t compute_units = 0;
    size_t max_wg_size = 0;
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

TqStatus init_context() {
    if (g_ctx.initialized) return TqStatus::OK;

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) return TqStatus::ERR_NO_PLATFORM;

    cl_platform_id plat;
    clGetPlatformIDs(1, &plat, nullptr);

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) return TqStatus::ERR_NO_DEVICE;

    cl_device_id dev;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return TqStatus::ERR_NO_PLATFORM;

    // Create command queue with profiling enabled
    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, props, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(ctx);
        return TqStatus::ERR_NO_PLATFORM;
    }

    // Query device info
    char name_buf[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr);

    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);

    size_t wg = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr);

    g_ctx.platform = plat;
    g_ctx.device = dev;
    g_ctx.context = ctx;
    g_ctx.queue = queue;
    g_ctx.initialized = true;
    g_ctx.is_adreno = (strstr(name_buf, "Adreno") != nullptr || strstr(name_buf, "QUALCOMM") != nullptr);
    g_ctx.compute_units = cu;
    g_ctx.max_wg_size = wg;

    return TqStatus::OK;
}

void shutdown_context() {
    if (!g_ctx.initialized) return;
    clReleaseCommandQueue(g_ctx.queue);
    clReleaseContext(g_ctx.context);
    g_ctx = OpenClContext{};
}

} // namespace tq
