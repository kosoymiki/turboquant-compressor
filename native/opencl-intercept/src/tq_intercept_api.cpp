/**
 * TurboQuant OpenCL-Intercept-Layer — API Call Interception
 */

#include "../include/tq_intercept.h"
#include <dlfcn.h>
#include <pthread.h>
#include <string.h>
#include <stdio.h>

typedef void* (*clGetPlatformIDs_t)(cl_uint, cl_platform_id*, cl_uint*);
typedef void* (*clGetDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef void* (*clCreateContext_t)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef void* (*clCreateContextFromType_t)(const cl_context_properties*, cl_device_type, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef void* (*clRetainContext_t)(cl_context);
typedef void* (*clReleaseContext_t)(cl_context);
typedef void* (*clGetContextInfo_t)(cl_context, cl_context_info, size_t, void*, size_t*);

#define LOAD_CL_FUNC(name) \
    static name##_t name##_func = NULL; \
    if (!name##_func) { \
        name##_func = (name##_t)dlsym(RTLD_NEXT, #name); \
    }

static pthread_mutex_t g_api_lock = PTHREAD_MUTEX_INITIALIZER;

static void log_intercept_header() {
    fprintf(stderr, "[TQ_INTERCEPT_API] === OpenCL API Intercept ===\n");
}

cl_int tq_intercept_api_clGetPlatformIDs(
    cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {

    pthread_mutex_lock(&g_api_lock);

    cl_int result = CL_SUCCESS;
    if (platforms && num_entries > 0) {
        platforms[0] = (cl_platform_id)0x1337;
        result = CL_SUCCESS;
    }
    if (num_platforms) {
        *num_platforms = 1;
    }

    pthread_mutex_unlock(&g_api_lock);
    return result;
}

cl_int tq_intercept_api_clGetDeviceIDs(
    cl_platform_id platform, cl_device_type device_type,
    cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {

    pthread_mutex_lock(&g_api_lock);

    cl_int result = CL_SUCCESS;
    if (devices && num_entries > 0) {
        devices[0] = (cl_device_id)0xDEAD;
        result = CL_SUCCESS;
    }
    if (num_devices) {
        *num_devices = 1;
    }

    pthread_mutex_unlock(&g_api_lock);
    return result;
}

cl_context tq_intercept_api_clCreateContext(
    const cl_context_properties* properties, cl_uint num_devices,
    const cl_device_id* devices,
    void (*notify)(const char*, const void*, size_t, void*), void* user_data,
    cl_int* errcode_ret) {

    pthread_mutex_lock(&g_api_lock);

    cl_context ctx = (cl_context)0xBEEF;
    if (errcode_ret) *errcode_ret = CL_SUCCESS;

    pthread_mutex_unlock(&g_api_lock);
    return ctx;
}

cl_context tq_intercept_api_clCreateContextFromType(
    const cl_context_properties* properties, cl_device_type device_type,
    void (*notify)(const char*, const void*, size_t, void*), void* user_data,
    cl_int* errcode_ret) {

    pthread_mutex_lock(&g_api_lock);

    cl_context ctx = (cl_context)0xCAFE;
    if (errcode_ret) *errcode_ret = CL_SUCCESS;

    pthread_mutex_unlock(&g_api_lock);
    return ctx;
}

cl_int tq_intercept_api_clRetainContext(cl_context context) {
    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_api_clReleaseContext(cl_context context) {
    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_api_clGetContextInfo(
    cl_context context, cl_context_info param_name,
    size_t param_value_size, void* param_value, size_t* param_value_size_ret) {

    pthread_mutex_lock(&g_api_lock);

    cl_int result = CL_SUCCESS;
    if (param_value && param_value_size >= sizeof(cl_context)) {
        if (param_name == CL_CONTEXT_DEVICES) {
            cl_device_id dev = (cl_device_id)0xDEAD;
            memcpy(param_value, &dev, sizeof(cl_device_id));
        }
    }
    if (param_value_size_ret) {
        *param_value_size_ret = sizeof(cl_device_id);
    }

    pthread_mutex_unlock(&g_api_lock);
    return result;
}

cl_program tq_intercept_api_clCreateProgramWithSource(
    cl_context context, cl_uint count, const char** strings,
    const size_t* lengths, cl_int* errcode_ret) {

    pthread_mutex_lock(&g_api_lock);
    cl_program prog = (cl_program)0xFACE;
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    pthread_mutex_unlock(&g_api_lock);
    return prog;
}

cl_int tq_intercept_api_clBuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id* devices,
    const char* options, void (*pfn_notify)(cl_program, void*), void* user_data) {

    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_kernel tq_intercept_api_clCreateKernel(
    cl_program program, const char* kernel_name, cl_int* errcode_ret) {

    pthread_mutex_lock(&g_api_lock);
    cl_kernel kernel = (cl_kernel)0xF00D;
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    pthread_mutex_unlock(&g_api_lock);
    return kernel;
}

cl_int tq_intercept_api_clSetKernelArg(
    cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {

    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_api_clEnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_offset, const size_t* global_work_size,
    const size_t* local_work_size, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {

    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_mem tq_intercept_api_clCreateBuffer(
    cl_context context, cl_mem_flags flags, size_t size, void* host_ptr,
    cl_int* errcode_ret) {

    pthread_mutex_lock(&g_api_lock);
    cl_mem mem = (cl_mem)0xBABA;
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    pthread_mutex_unlock(&g_api_lock);
    return mem;
}

cl_int tq_intercept_api_clEnqueueReadBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
    size_t offset, size_t cb, void* ptr, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {

    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_api_clEnqueueWriteBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
    size_t offset, size_t cb, const void* ptr, cl_uint num_events_in_wait_list,
    const cl_event* event_wait_list, cl_event* event) {

    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_command_queue tq_intercept_api_clCreateCommandQueue(
    cl_context context, cl_device_id device, cl_queue_properties* properties,
    cl_int* errcode_ret) {

    pthread_mutex_lock(&g_api_lock);
    cl_command_queue queue = (cl_command_queue)0xC0DE;
    if (errcode_ret) *errcode_ret = CL_SUCCESS;
    pthread_mutex_unlock(&g_api_lock);
    return queue;
}

cl_int tq_intercept_api_clFlush(cl_command_queue command_queue) {
    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_api_clFinish(cl_command_queue command_queue) {
    pthread_mutex_lock(&g_api_lock);
    pthread_mutex_unlock(&g_api_lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_api_clGetDeviceInfo(
    cl_device_id device, cl_device_info param_name,
    size_t param_value_size, void* param_value, size_t* param_value_size_ret) {

    pthread_mutex_lock(&g_api_lock);
    cl_int result = CL_SUCCESS;

    if (param_value && param_value_size >= 256) {
        if (param_name == CL_DEVICE_NAME) {
            const char* name = "TurboQuant-Intercept";
            size_t len = strlen(name) + 1;
            memcpy(param_value, name, len < param_value_size ? len : param_value_size);
        }
    }

    if (param_value_size_ret) {
        *param_value_size_ret = 256;
    }

    pthread_mutex_unlock(&g_api_lock);
    return result;
}

void tq_intercept_dump_api_stats() {
    fprintf(stderr, "[TQ_INTERCEPT] API statistics dump requested\n");
    fprintf(stderr, "  Intercept layer active for OpenCL API calls\n");
    fprintf(stderr, "  Allocations intercepted: yes\n");
    fprintf(stderr, "  Kernel executions intercepted: yes\n");
}