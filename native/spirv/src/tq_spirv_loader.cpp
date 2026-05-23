/**
 * TurboQuant SPIR-V Loader — Implementation
 */

#include "tq_spirv_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <sys/stat.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "tq_spirv"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) fprintf(stderr, "[TQ_SPIRV] " __VA_ARGS__)
#define LOGE(...) fprintf(stderr, "[TQ_SPIRV ERROR] " __VA_ARGS__)
#endif

cl_int tq_spirv_loader_init(tq_spirv_loader_t* loader, cl_context ctx, cl_device_id dev) {
    if (!loader) return CL_INVALID_VALUE;

    memset(loader, 0, sizeof(tq_spirv_loader_t));
    loader->context = ctx;
    loader->device = dev;
    loader->is_loaded = 0;

    if (ctx) clRetainContext(ctx);
    if (dev) clRetainDevice(dev);

    return CL_SUCCESS;
}

static size_t file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return st.st_size;
}

cl_int tq_spirv_loader_load_file(tq_spirv_loader_t* loader, const char* spv_path) {
    if (!loader || !spv_path) return CL_INVALID_VALUE;

    size_t fsize = file_size(spv_path);
    if (fsize == 0) {
        snprintf(loader->last_error, sizeof(loader->last_error), "Empty or missing file: %s", spv_path);
        return CL_INVALID_VALUE;
    }

    FILE* f = fopen(spv_path, "rb");
    if (!f) {
        snprintf(loader->last_error, sizeof(loader->last_error), "Cannot open file: %s", spv_path);
        return CL_INVALID_VALUE;
    }

    uint32_t* data = (uint32_t*)malloc(fsize);
    if (!data) {
        fclose(f);
        snprintf(loader->last_error, sizeof(loader->last_error), "Out of memory");
        return CL_OUT_OF_HOST_MEMORY;
    }

    size_t read = fread(data, 1, fsize, f);
    fclose(f);

    if (read != fsize) {
        free(data);
        snprintf(loader->last_error, sizeof(loader->last_error), "Read error");
        return CL_INVALID_VALUE;
    }

    cl_int err = tq_spirv_loader_load_binary(loader, data, fsize / 4);
    free(data);

    return err;
}

cl_int tq_spirv_loader_load_binary(tq_spirv_loader_t* loader, const uint32_t* spv_data, size_t spv_words) {
    if (!loader || !spv_data || spv_words == 0) return CL_INVALID_VALUE;

    if (loader->program) {
        clReleaseProgram(loader->program);
        loader->program = NULL;
    }

    for (uint32_t i = 0; i < loader->kernel_count; i++) {
        if (loader->kernels && loader->kernels[i]) {
            clReleaseKernel(loader->kernels[i]);
        }
    }
    free(loader->kernels);
    free(loader->kernel_names);

    loader->program = clCreateProgramWithIL(
        loader->context,
        (void*)spv_data,
        spv_words * sizeof(uint32_t),
        &loader->is_loaded
    );

    if (!loader->program || loader->is_loaded != CL_SUCCESS) {
        snprintf(loader->last_error, sizeof(loader->last_error),
                 "clCreateProgramWithIL failed: %d", (int)loader->is_loaded);
        loader->is_loaded = 0;
        return CL_BUILD_ERROR;
    }

    cl_int build_err = clBuildProgram(loader->program, 1, &loader->device, "-cl-opt-disable", NULL, NULL);
    if (build_err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(loader->program, loader->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (log_size > 0) {
            char* log = (char*)malloc(log_size + 1);
            if (log) {
                clGetProgramBuildInfo(loader->program, loader->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                log[log_size] = '\0';
                snprintf(loader->last_error, sizeof(loader->last_error), "Build error: %s", log);
                free(log);
            }
        }
        return build_err;
    }

    cl_uint num_kernels = 0;
    cl_int err = clGetProgramInfo(loader->program, CL_PROGRAM_NUM_KERNELS, sizeof(num_kernels), &num_kernels, NULL);
    if (err != CL_SUCCESS || num_kernels == 0) {
        snprintf(loader->last_error, sizeof(loader->last_error), "No kernels in SPIR-V module");
        return CL_INVALID_VALUE;
    }

    size_t name_size = 0;
    clGetProgramInfo(loader->program, CL_PROGRAM_KERNEL_NAMES, 0, NULL, &name_size);
    if (name_size == 0) {
        snprintf(loader->last_error, sizeof(loader->last_error), "Cannot get kernel names");
        return CL_INVALID_VALUE;
    }

    char* all_names = (char*)malloc(name_size + 1);
    clGetProgramInfo(loader->program, CL_PROGRAM_KERNEL_NAMES, name_size, all_names, NULL);
    all_names[name_size] = '\0';

    loader->kernel_count = num_kernels;
    loader->kernels = (cl_kernel*)malloc(num_kernels * sizeof(cl_kernel));
    loader->kernel_names = (char**)malloc(num_kernels * sizeof(char*));

    char* saveptr;
    char* name = strtok_r(all_names, ";", &saveptr);
    for (cl_uint i = 0; i < num_kernels && name; i++) {
        loader->kernel_names[i] = strdup(name);
        loader->kernels[i] = clCreateKernel(loader->program, name, NULL);
        name = strtok_r(NULL, ";", &saveptr);
    }

    free(all_names);
    loader->is_loaded = 1;

    LOGI("Loaded %u kernels from SPIR-V module\n", num_kernels);
    return CL_SUCCESS;
}

cl_kernel tq_spirv_loader_get_kernel(tq_spirv_loader_t* loader, const char* kernel_name) {
    if (!loader || !kernel_name || !loader->is_loaded) return NULL;

    for (uint32_t i = 0; i < loader->kernel_count; i++) {
        if (loader->kernel_names[i] && strcmp(loader->kernel_names[i], kernel_name) == 0) {
            return loader->kernels[i];
        }
    }
    return NULL;
}

const char** tq_spirv_loader_get_kernel_names(tq_spirv_loader_t* loader, uint32_t* count) {
    if (!loader || !count) return NULL;
    *count = loader->kernel_count;
    return (const char**)loader->kernel_names;
}

cl_int tq_spirv_loader_print_info(tq_spirv_loader_t* loader) {
    if (!loader) return CL_INVALID_VALUE;

    fprintf(stderr, "=== SPIR-V Loader Info ===\n");
    fprintf(stderr, "  Loaded: %s\n", loader->is_loaded ? "yes" : "no");
    fprintf(stderr, "  Kernel count: %u\n", loader->kernel_count);
    fprintf(stderr, "  Kernels:\n");
    for (uint32_t i = 0; i < loader->kernel_count; i++) {
        fprintf(stderr, "    [%u] %s\n", i, loader->kernel_names[i] ? loader->kernel_names[i] : "unknown");
    }

    if (loader->program) {
        cl_uint num_devices = 0;
        clGetProgramInfo(loader->program, CL_PROGRAM_NUM_DEVICES, sizeof(num_devices), &num_devices, NULL);
        fprintf(stderr, "  Devices: %u\n", num_devices);
    }

    return CL_SUCCESS;
}

cl_int tq_spirv_launch(
    tq_spirv_loader_t* loader,
    const char* kernel_name,
    cl_command_queue queue,
    size_t global_size,
    size_t local_size,
    uint32_t arg_count,
    ...
) {
    if (!loader || !kernel_name || !queue) return CL_INVALID_VALUE;

    cl_kernel kernel = tq_spirv_loader_get_kernel(loader, kernel_name);
    if (!kernel) {
        snprintf(loader->last_error, sizeof(loader->last_error),
                 "Kernel not found: %s", kernel_name);
        return CL_INVALID_KERNEL;
    }

    va_list args;
    va_start(args, arg_count);
    for (uint32_t i = 0; i < arg_count; i++) {
        void* ptr = va_arg(args, void*);
        clSetKernelArg(kernel, i, sizeof(void*), &ptr);
    }
    va_end(args);

    size_t gws[] = {global_size, 1, 1};
    size_t lws[] = {local_size > 0 ? local_size : 1, 1, 1};

    return clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, lws, 0, NULL, NULL);
}

cl_int tq_spirv_loader_shutdown(tq_spirv_loader_t* loader) {
    if (!loader) return CL_INVALID_VALUE;

    for (uint32_t i = 0; i < loader->kernel_count; i++) {
        if (loader->kernels && loader->kernels[i]) {
            clReleaseKernel(loader->kernels[i]);
        }
        if (loader->kernel_names && loader->kernel_names[i]) {
            free(loader->kernel_names[i]);
        }
    }
    free(loader->kernels);
    free(loader->kernel_names);

    if (loader->program) {
        clReleaseProgram(loader->program);
    }

    if (loader->context) clReleaseContext(loader->context);
    if (loader->device) clReleaseDevice(loader->device);

    memset(loader, 0, sizeof(tq_spirv_loader_t));
    return CL_SUCCESS;
}

const char* tq_spirv_loader_error(tq_spirv_loader_t* loader) {
    if (!loader) return "NULL loader";
    return loader->last_error;
}

#ifdef __cplusplus

namespace tq {
namespace spirv {

Loader::Loader(cl_context ctx, cl_device_id dev) : last_error_("") {
    tq_spirv_loader_init(&loader_, ctx, dev);
}

Loader::~Loader() {
    tq_spirv_loader_shutdown(&loader_);
}

bool Loader::load_file(const std::string& path) {
    cl_int err = tq_spirv_loader_load_file(&loader_, path.c_str());
    if (err != CL_SUCCESS) {
        last_error_ = tq_spirv_loader_error(&loader_);
        return false;
    }
    return true;
}

bool Loader::load_binary(const uint32_t* data, size_t size) {
    cl_int err = tq_spirv_loader_load_binary(&loader_, data, size);
    if (err != CL_SUCCESS) {
        last_error_ = tq_spirv_loader_error(&loader_);
        return false;
    }
    return true;
}

cl_kernel Loader::kernel(const std::string& name) const {
    return tq_spirv_loader_get_kernel(&loader_, name.c_str());
}

std::vector<std::string> Loader::kernel_names() const {
    std::vector<std::string> names;
    uint32_t count = 0;
    const char** names_ptr = tq_spirv_loader_get_kernel_names(&loader_, &count);
    for (uint32_t i = 0; i < count; i++) {
        if (names_ptr && names_ptr[i]) {
            names.push_back(names_ptr[i]);
        }
    }
    return names;
}

bool Loader::print_info() const {
    return tq_spirv_loader_print_info(&loader_) == CL_SUCCESS;
}

void Loader::launch(const std::string& name, cl_command_queue queue,
                      size_t global_size, size_t local_size,
                      std::function<void(cl_kernel)> set_args) {
    cl_kernel k = kernel(name);
    if (!k) {
        last_error_ = "Kernel not found: " + name;
        return;
    }

    if (set_args) set_args(k);

    size_t gws[] = {global_size, 1, 1};
    size_t lws[] = {local_size > 0 ? local_size : 1, 1, 1};
    clEnqueueNDRangeKernel(queue, k, 1, NULL, gws, lws, 0, NULL, NULL);
}

KernelLauncher::KernelLauncher(cl_command_queue queue)
    : queue_(queue), kernel_(NULL) {}

KernelLauncher::~KernelLauncher() {}

KernelLauncher& KernelLauncher::with_kernel(cl_kernel kernel) {
    kernel_ = kernel;
    return *this;
}

KernelLauncher& KernelLauncher::with_global(size_t gx, size_t gy, size_t gz) {
    global_[0] = gx; global_[1] = gy; global_[2] = gz;
    return *this;
}

KernelLauncher& KernelLauncher::with_local(size_t lx, size_t ly, size_t lz) {
    local_[0] = lx; local_[1] = ly; local_[2] = lz;
    return *this;
}

KernelLauncher& KernelLauncher::arg(int idx, cl_mem val) {
    if (kernel_) clSetKernelArg(kernel_, idx, sizeof(cl_mem), &val);
    return *this;
}

KernelLauncher& KernelLauncher::arg(int idx, cl_uint val) {
    if (kernel_) clSetKernelArg(kernel_, idx, sizeof(cl_uint), &val);
    return *this;
}

KernelLauncher& KernelLauncher::arg(int idx, float val) {
    if (kernel_) clSetKernelArg(kernel_, idx, sizeof(float), &val);
    return *this;
}

KernelLauncher& KernelLauncher::arg(int idx, int val) {
    if (kernel_) clSetKernelArg(kernel_, idx, sizeof(int), &val);
    return *this;
}

KernelLauncher& KernelLauncher::arg(int idx, size_t val) {
    if (kernel_) clSetKernelArg(kernel_, idx, sizeof(size_t), &val);
    return *this;
}

KernelLauncher& KernelLauncher::arg_ptr(int idx, void* ptr, size_t size) {
    if (kernel_) clSetKernelArg(kernel_, idx, size, ptr);
    return *this;
}

cl_int KernelLauncher::execute() {
    if (!kernel_ || !queue_) return CL_INVALID_VALUE;
    return clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, global_, local_, 0, NULL, NULL);
}

} // namespace spirv
} // namespace tq

#endif // __cplusplus