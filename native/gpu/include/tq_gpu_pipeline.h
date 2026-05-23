/**
 * TurboQuant GPU Execution Pipeline — SPIR-V Integration
 *
 * Full pipeline: SPIR-V → clCreateProgramWithIL → clBuildProgram →
 * clCreateKernel → clEnqueueNDRangeKernel → Mesa Rusticl / Turnip Vulkan → Adreno GPU
 */

#ifndef TQ_GPU_PIPELINE_H
#define TQ_GPU_PIPELINE_H

#include <CL/cl.h>
#include <cstdint>
#include <cstddef>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TQ_GPU_MODE_SPIRV = 0,
    TQ_GPU_MODE_OPENCL_C,
    TQ_GPU_MODE_HYBRID
} tq_gpu_mode_t;

typedef enum {
    TQ_KERNEL_MSE_SCORE = 0,
    TQ_KERNEL_QJL_SCORE,
    TQ_KERNEL_VALUE_DEQUANT,
    TQ_KERNEL_ATTENTION_LOGITS,
    TQ_KERNEL_ATTENTION_APPLY_VALUES,
    TQ_KERNEL_COUNT
} tq_kernel_id_t;

typedef struct {
    cl_mem mem;
    size_t size;
    void* host_ptr;
    cl_mem_flags flags;
    uint8_t is_mapped;
} tq_gpu_buffer_t;

typedef struct {
    tq_kernel_id_t kernel_id;
    size_t global_size[3];
    size_t local_size[3];
    uint32_t arg_count;
    void* args[32];
    size_t arg_sizes[32];
} tq_kernel_launch_t;

typedef struct {
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program spirv_programs[TQ_KERNEL_COUNT];
    cl_kernel spirv_kernels[TQ_KERNEL_COUNT];
    cl_program opencl_programs[TQ_KERNEL_COUNT];
    cl_kernel opencl_kernels[TQ_KERNEL_COUNT];
    tq_gpu_mode_t active_mode;
    uint8_t spirv_loaded;
    uint8_t opencl_loaded;
    uint8_t is_initialized;
    char error_buffer[1024];
    uint64_t total_kernel_executions;
    uint64_t total_kernel_ns;
} tq_gpu_pipeline_t;

cl_int tq_gpu_pipeline_init(tq_gpu_pipeline_t* p, cl_context ctx, cl_device_id dev, tq_gpu_mode_t mode);
cl_int tq_gpu_pipeline_load_spirv(tq_gpu_pipeline_t* p, tq_kernel_id_t kid, const char* spv_path);
cl_int tq_gpu_pipeline_load_spirv_binary(tq_gpu_pipeline_t* p, tq_kernel_id_t kid, const uint32_t* d, size_t words);
cl_int tq_gpu_pipeline_build_opencl(tq_gpu_pipeline_t* p, tq_kernel_id_t kid, const char* source);
cl_int tq_gpu_buffer_alloc(tq_gpu_pipeline_t* p, tq_gpu_buffer_t* b, size_t sz, cl_mem_flags flags, void* host);
cl_int tq_gpu_buffer_free(tq_gpu_pipeline_t* p, tq_gpu_buffer_t* b);
cl_int tq_gpu_buffer_map(tq_gpu_pipeline_t* p, tq_gpu_buffer_t* b, cl_map_flags flags);
cl_int tq_gpu_buffer_unmap(tq_gpu_pipeline_t* p, tq_gpu_buffer_t* b);
cl_int tq_gpu_kernel_launch(tq_gpu_pipeline_t* p, const tq_kernel_launch_t* cfg);

cl_int tq_gpu_launch_mse_score(tq_gpu_pipeline_t* p, cl_mem packed, cl_mem query, cl_mem norms,
                                  cl_mem centroids, cl_mem scores, size_t count, size_t dims, uint32_t bits);
cl_int tq_gpu_launch_qjl_score(tq_gpu_pipeline_t* p, cl_mem sketches, cl_mem query, cl_mem scores,
                                  size_t batch, size_t qdim, size_t sdim);
cl_int tq_gpu_launch_value_dequant(tq_gpu_pipeline_t* p, cl_mem packed, cl_mem codebook, cl_mem out,
                                     size_t count, size_t dims, uint32_t bits);
cl_int tq_gpu_launch_attention_logits(tq_gpu_pipeline_t* p, cl_mem Q, cl_mem K, cl_mem out,
                                        size_t batch, size_t heads, size_t seq, size_t hdim);
cl_int tq_gpu_launch_attention_apply_values(tq_gpu_pipeline_t* p, cl_mem W, cl_mem V, cl_mem out,
                                               size_t batch, size_t heads, size_t seq, size_t hdim);
cl_int tq_gpu_pipeline_sync(tq_gpu_pipeline_t* p);
cl_int tq_gpu_pipeline_get_stats(const tq_gpu_pipeline_t* p, uint64_t* execs, uint64_t* ns, double* avg);
cl_int tq_gpu_pipeline_shutdown(tq_gpu_pipeline_t* p);
const char* tq_gpu_pipeline_error(tq_gpu_pipeline_t* p);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <memory>
#include <vector>
#include <functional>

namespace tq {
namespace gpu {

class Pipeline {
public:
    explicit Pipeline(cl_context ctx, cl_device_id dev, tq_gpu_mode_t mode = TQ_GPU_MODE_SPIRV);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    bool load_spirv(tq_kernel_id_t kid, const std::string& path);
    bool build_opencl(tq_kernel_id_t kid, const std::string& source);

    struct Buffer {
        cl_mem mem = nullptr;
        size_t size = 0;
        bool mapped = false;
    };

    Buffer allocate(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
    bool write(const Buffer& buf, const void* data, size_t offset = 0);
    bool read(const Buffer& buf, void* data, size_t offset = 0);
    bool free(const Buffer& buf);

    struct KernelLaunch {
        tq_kernel_id_t kernel;
        std::vector<size_t> global;
        std::vector<size_t> local;
        std::vector<std::pair<void*, size_t>> args;

        KernelLaunch& arg_ptr(void* ptr, size_t size) { args.emplace_back(ptr, size); return *this; }
        KernelLaunch& arg_mem(cl_mem m) { return arg_ptr(&m, sizeof(cl_mem)); }
        KernelLaunch& arg_uint(uint32_t v) { return arg_ptr(&v, sizeof(uint32_t)); }
        KernelLaunch& gsize(size_t x, size_t y = 1, size_t z = 1) { global = {x, y, z}; return *this; }
        KernelLaunch& lsize(size_t x, size_t y = 1, size_t z = 1) { local = {x, y, z}; return *this; }
    };

    bool launch(const KernelLaunch& k);
    bool sync();
    tq_gpu_mode_t active_mode() const { return pipeline_.active_mode; }
    const std::string& last_error() const { return last_error_; }

    bool mse_score(cl_mem packed, cl_mem query, cl_mem norms, cl_mem centroids, cl_mem scores,
                   size_t vectors, size_t dims, uint32_t bits);
    bool qjl_score(cl_mem sketches, cl_mem query, cl_mem scores,
                   size_t batch, size_t qdim, size_t sdim);
    bool value_dequant(cl_mem packed, cl_mem codebook, cl_mem output,
                       size_t vectors, size_t dims, uint32_t bits);
    bool attention_logits(cl_mem Q, cl_mem K, cl_mem out,
                          size_t batch, size_t heads, size_t seq, size_t hdim);
    bool attention_apply(cl_mem W, cl_mem V, cl_mem out,
                          size_t batch, size_t heads, size_t seq, size_t hdim);

    struct Stats { uint64_t executions = 0, total_ns = 0; double avg_ns = 0; };
    Stats stats() const;

private:
    tq_gpu_pipeline_t pipeline_;
    std::string last_error_;
    std::vector<cl_mem> allocated_buffers_;
};

class FusedAttentionPipeline {
public:
    FusedAttentionPipeline(cl_context ctx, cl_device_id dev);
    ~FusedAttentionPipeline();
    bool init(const std::string& spirv_dir);
    bool forward(size_t batch, size_t heads, size_t seq, size_t hdim,
                 cl_mem Q, cl_mem K, cl_mem V, cl_mem output);

private:
    Pipeline pipeline_;
};

} // namespace gpu
} // namespace tq

#endif // __cplusplus

#endif // TQ_GPU_PIPELINE_H