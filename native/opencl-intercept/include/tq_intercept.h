/**
 * TurboQuant OpenCL-Intercept-Layer — Native C++ Library
 *
 * Intercepts OpenCL API calls for:
 * - Memory register tracing and analysis
 * - Kernel execution profiling
 * - Call graph tracking
 * - Performance bottleneck identification
 *
 * Prepares infrastructure for full TurboQuant C++ migration.
 * Enables precise memory register work without TypeScript limitations.
 */

#ifndef TQ_INTERCEPT_H
#define TQ_INTERCEPT_H

#include <CL/cl.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ_INTERCEPT_VERSION_MAJOR 1
#define TQ_INTERCEPT_VERSION_MINOR 0
#define TQ_INTERCEPT_VERSION_PATCH 0

typedef uint32_t tq_intercept_flags_t;
#define TQ_INTERCEPT_FLAG_TRACE_CALLS      (1 << 0)
#define TQ_INTERCEPT_FLAG_TRACE_ARGS       (1 << 1)
#define TQ_INTERCEPT_FLAG_TRACE_RETURNS    (1 << 2)
#define TQ_INTERCEPT_FLAG_LOG_MEMORY       (1 << 3)
#define TQ_INTERCEPT_FLAG_LOG_KERNELS      (1 << 4)
#define TQ_INTERCEPT_FLAG_PROFILE_PERF     (1 << 5)
#define TQ_INTERCEPT_FLAG_CHAIN_LAYERS     (1 << 6)
#define TQ_INTERCEPT_FLAG_THREAD_SAFE     (1 << 7)
#define TQ_INTERCEPT_FLAG_BUFFER_RECYCLING (1 << 8)

typedef enum tq_mem_op_type {
    TQ_MEM_OP_BUFFER_CREATE,
    TQ_MEM_OP_BUFFER_READ,
    TQ_MEM_OP_BUFFER_WRITE,
    TQ_MEM_OP_BUFFER_COPY,
    TQ_MEM_OP_BUFFER_MAP,
    TQ_MEM_OP_BUFFER_UNMAP,
    TQ_MEM_OP_IMAGE_CREATE,
    TQ_MEM_OP_IMAGE_READ,
    TQ_MEM_OP_IMAGE_WRITE,
    TQ_MEM_OP_SVM_ALLOC,
    TQ_MEM_OP_SVM_FREE,
    TQ_MEM_OP_UNKNOWN
} tq_mem_op_type_t;

typedef struct tq_kernel_exec_info {
    cl_kernel kernel;
    cl_program program;
    const char* kernel_name;
    uint64_t enqueue_time_ns;
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    cl_uint work_dim;
    size_t global_work_size[3];
    size_t local_work_size[3];
    cl_event event;
    cl_int status;
} tq_kernel_exec_info_t;

typedef struct tq_mem_op_record {
    uint64_t timestamp_ns;
    tq_mem_op_type_t op_type;
    cl_mem memory_object;
    size_t size_bytes;
    void* host_ptr;
    cl_bool blocking;
    size_t offset;
    size_t cb;
    cl_event event;
    cl_int status;
} tq_mem_op_record_t;

typedef struct tq_intercept_context tq_intercept_context_t;

typedef struct tq_intercept_layer {
    const char* name;
    cl_int (*init)(struct tq_intercept_layer* layer, tq_intercept_context_t* ctx);
    cl_int (*shutdown)(struct tq_intercept_layer* layer);
    cl_int (*on_call)(struct tq_intercept_layer* layer, const char* func_name, void* args, void* ret);
    cl_int (*on_mem_op)(struct tq_intercept_layer* layer, const tq_mem_op_record_t* op);
    cl_int (*on_kernel_exec)(struct tq_intercept_layer* layer, const tq_kernel_exec_info_t* info);
    void* user_data;
    struct tq_intercept_layer* next;
} tq_intercept_layer_t;

typedef struct tq_intercept_config {
    tq_intercept_flags_t flags;
    uint32_t log_buffer_size;
    uint32_t max_mem_ops;
    uint32_t max_kernel_execs;
    uint32_t thread_count;
    const char* log_file_path;
    tq_intercept_layer_t* layer_chain;
    void* user_data;
} tq_intercept_config_t;

typedef struct tq_perf_metrics {
    uint64_t total_api_calls;
    uint64_t total_mem_ops;
    uint64_t total_kernel_execs;
    uint64_t total_api_time_ns;
    uint64_t total_mem_time_ns;
    uint64_t total_kernel_time_ns;
    double avg_api_call_ns;
    double avg_mem_op_ns;
    double avg_kernel_exec_ns;
    uint64_t max_api_call_ns;
    uint64_t max_mem_op_ns;
    uint64_t max_kernel_exec_ns;
} tq_perf_metrics_t;

typedef struct tq_mem_alloc_info {
    cl_mem memory;
    size_t size;
    cl_mem_object_type type;
    cl_mem_flags flags;
    void* host_ptr;
    cl_context context;
    uint64_t alloc_time_ns;
    uint64_t last_access_ns;
    uint32_t access_count;
} tq_mem_alloc_info_t;

typedef void (*tq_context_create_callback_t)(
    cl_context context, cl_device_id device,
    const tq_intercept_config_t* config, void* user_data
);

typedef cl_device_id (*tq_device_select_callback_t)(
    cl_platform_id platform, cl_device_type device_type, void* user_data
);

cl_int tq_intercept_init(tq_intercept_config_t* config);
cl_int tq_intercept_shutdown(void);
tq_intercept_context_t* tq_intercept_get_context(cl_device_id device);
void tq_intercept_set_context_create_callback(tq_context_create_callback_t cb, void* user_data);
void tq_intercept_set_device_select_callback(tq_device_select_callback_t cb, void* user_data);
cl_int tq_intercept_register_layer(tq_intercept_layer_t* layer);
cl_int tq_intercept_unregister_layer(const char* layer_name);
cl_int tq_intercept_get_metrics(tq_perf_metrics_t* metrics);
cl_int tq_intercept_reset_metrics(void);
cl_int tq_intercept_get_mem_allocations(tq_mem_alloc_info_t* allocs, uint32_t* count, uint32_t max_count);
cl_int tq_intercept_flush_logs(void);
const char* tq_intercept_version_string(void);
cl_bool tq_intercept_is_active(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace tq {
namespace intercept {

class Context {
public:
    explicit Context(cl_device_id device);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;

    cl_device_id device() const { return device_; }
    bool is_active() const { return active_; }
    std::vector<tq_mem_alloc_info_t> get_allocations();
    tq_perf_metrics_t get_metrics();
    void flush_logs();

private:
    cl_device_id device_;
    bool active_;
    tq_intercept_context_t* ctx_;
};

class ConfigBuilder {
public:
    ConfigBuilder& set_flags(tq_intercept_flags_t flags);
    ConfigBuilder& enable_trace_calls();
    ConfigBuilder& enable_trace_args();
    ConfigBuilder& enable_trace_returns();
    ConfigBuilder& enable_log_memory();
    ConfigBuilder& enable_log_kernels();
    ConfigBuilder& enable_profile_perf();
    ConfigBuilder& enable_layer_chain();
    ConfigBuilder& enable_thread_safety();
    ConfigBuilder& enable_buffer_recycling();
    ConfigBuilder& set_log_file(const std::string& path);
    ConfigBuilder& set_log_buffer_size(uint32_t size);
    ConfigBuilder& set_max_mem_ops(uint32_t max);
    ConfigBuilder& set_max_kernel_execs(uint32_t max);
    ConfigBuilder& set_thread_count(uint32_t count);
    ConfigBuilder& add_layer(tq_intercept_layer_t* layer);
    tq_intercept_config_t build();

private:
    tq_intercept_config_t config_;
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual cl_int init(tq_intercept_context_t* ctx) = 0;
    virtual cl_int shutdown() = 0;
    virtual cl_int on_call(const char* func_name, void* args, void* ret) = 0;
    virtual cl_int on_mem_op(const tq_mem_op_record_t* op) = 0;
    virtual cl_int on_kernel_exec(const tq_kernel_exec_info_t* info) = 0;
    const char* name() const { return name_.c_str(); }

protected:
    explicit Layer(const char* name) : name_(name) {}

private:
    std::string name_;
};

class Profiler {
public:
    static Profiler& instance();
    void record_api_call(const char* func_name, uint64_t duration_ns);
    void record_mem_op(tq_mem_op_type_t op_type, size_t size, uint64_t duration_ns);
    void record_kernel_exec(const char* kernel_name, uint64_t duration_ns);
    tq_perf_metrics_t get_metrics();
    void reset();
    void print_report();

private:
    Profiler() = default;
    ~Profiler() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MemoryTracker {
public:
    static MemoryTracker& instance();
    void record_alloc(cl_mem mem, size_t size, cl_mem_object_type type,
                      cl_mem_flags flags, void* host_ptr);
    void record_free(cl_mem mem);
    void record_access(cl_mem mem);
    void record_mem_op(const tq_mem_op_record_t* op);
    std::vector<tq_mem_alloc_info_t> get_allocations();
    size_t total_allocated_bytes();
    size_t peak_allocated_bytes();
    void reset();

private:
    MemoryTracker() = default;
    ~MemoryTracker() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace intercept
} // namespace tq

#endif // __cplusplus

#endif // TQ_INTERCEPT_H