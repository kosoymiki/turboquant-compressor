/**
 * TurboQuant SPIR-V Loader — Load and execute compiled SPIR-V kernels via OpenCL
 *
 * Loads pre-compiled SPIR-V binaries and executes them on Adreno GPU
 * via Mesa Rusticl / Turnip Vulkan compute stack.
 */

#ifndef TQ_SPIR_V_LOADER_H
#define TQ_SPIR_V_LOADER_H

#include <CL/cl.h>
#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    cl_context context;
    cl_device_id device;
    cl_program program;
    cl_kernel* kernels;
    char** kernel_names;
    uint32_t kernel_count;
    uint8_t is_loaded;
    char last_error[512];
} tq_spirv_loader_t;

typedef struct {
    cl_kernel kernel;
    const char* name;
    uint32_t arg_count;
    size_t local_size;
    size_t global_size;
} tq_spirv_launch_config_t;

/**
 * Initialize SPIR-V loader
 */
cl_int tq_spirv_loader_init(tq_spirv_loader_t* loader, cl_context ctx, cl_device_id dev);

/**
 * Load SPIR-V binary from file
 */
cl_int tq_spirv_loader_load_file(tq_spirv_loader_t* loader, const char* spv_path);

/**
 * Load SPIR-V binary from memory
 */
cl_int tq_spirv_loader_load_binary(tq_spirv_loader_t* loader, const uint32_t* spv_data, size_t spv_size);

/**
 * Get kernel by name
 */
cl_kernel tq_spirv_loader_get_kernel(tq_spirv_loader_t* loader, const char* kernel_name);

/**
 * Get kernel names
 */
const char** tq_spirv_loader_get_kernel_names(tq_spirv_loader_t* loader, uint32_t* count);

/**
 * Print kernel info
 */
cl_int tq_spirv_loader_print_info(tq_spirv_loader_t* loader);

/**
 * Launch kernel with config
 */
cl_int tq_spirv_launch(
    tq_spirv_loader_t* loader,
    const char* kernel_name,
    cl_command_queue queue,
    size_t global_size,
    size_t local_size,
    uint32_t arg_count,
    ...
);

/**
 * Shutdown loader
 */
cl_int tq_spirv_loader_shutdown(tq_spirv_loader_t* loader);

/**
 * Get last error string
 */
const char* tq_spirv_loader_error(tq_spirv_loader_t* loader);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace tq {
namespace spirv {

class Loader {
public:
    explicit Loader(cl_context ctx, cl_device_id dev);
    ~Loader();

    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;

    bool load_file(const std::string& path);
    bool load_binary(const uint32_t* data, size_t size);

    cl_kernel kernel(const std::string& name) const;
    std::vector<std::string> kernel_names() const;

    bool print_info() const;

    void launch(const std::string& name, cl_command_queue queue,
               size_t global_size, size_t local_size,
               std::function<void(cl_kernel)> set_args);

    const std::string& last_error() const { return last_error_; }

private:
    tq_spirv_loader_t loader_;
    std::string last_error_;
    std::vector<std::string> kernel_names_;
};

class KernelLauncher {
public:
    explicit KernelLauncher(cl_command_queue queue);
    ~KernelLauncher();

    KernelLauncher& with_kernel(cl_kernel kernel);
    KernelLauncher& with_global(size_t gx, size_t gy = 1, size_t gz = 1);
    KernelLauncher& with_local(size_t lx, size_t ly = 1, size_t lz = 1);

    KernelLauncher& arg(int idx, cl_mem val);
    KernelLauncher& arg(int idx, cl_uint val);
    KernelLauncher& arg(int idx, float val);
    KernelLauncher& arg(int idx, int val);
    KernelLauncher& arg(int idx, size_t val);
    KernelLauncher& arg_ptr(int idx, void* ptr, size_t size);

    cl_int execute();

private:
    cl_command_queue queue_;
    cl_kernel kernel_;
    size_t global_[3] = {1, 1, 1};
    size_t local_[3] = {1, 1, 1};
    std::vector<std::pair<void*, size_t>> arg_buffers_;
    std::vector<cl_mem> retained_mems_;
};

class KernelInfo {
public:
    KernelInfo() : work_group_size_(0), private_mem_size_(0) {}

    const char* name() const { return name_.c_str(); }
    cl_uint work_group_size() const { return work_group_size_; }
    cl_ulong private_mem_size() const { return private_mem_size_; }
    cl_uint compile_work_group_size(uint32_t dim) const {
        return compile_work_group_size_[dim];
    }

private:
    std::string name_;
    cl_uint work_group_size_;
    cl_ulong private_mem_size_;
    cl_uint compile_work_group_size_[3];
};

} // namespace spirv
} // namespace tq

#endif // __cplusplus

#endif // TQ_SPIR_V_LOADER_H