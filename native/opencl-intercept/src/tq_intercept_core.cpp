/**
 * TurboQuant OpenCL-Intercept-Layer — Core Implementation
 *
 * Core intercept logic and API call interception.
 */

#include "tq_intercept.h"
#include <CL/cl.h>
#include <dlfcn.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "tq_intercept"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) fprintf(stdout, "[TQ_INTERCEPT] " __VA_ARGS__)
#define LOGE(...) fprintf(stderr, "[TQ_INTERCEPT ERROR] " __VA_ARGS__)
#endif

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define MAX_LAYERS 16
#define MAX_CONTEXTS 8
#define DEFAULT_LOG_BUFFER_SIZE (1024 * 1024)
#define DEFAULT_MAX_MEM_OPS 1024
#define DEFAULT_MAX_KERNEL_EXECS 256

typedef struct {
    cl_context context;
    cl_device_id device;
    void* device_data;
    uint64_t created_at_ns;
} tq_device_context_t;

typedef struct {
    tq_intercept_layer_t* layer;
    void* layer_data;
} tq_layer_entry_t;

typedef struct {
    pthread_mutex_t lock;
    tq_intercept_config_t config;
    tq_perf_metrics_t metrics;
    FILE* log_file;
    char* log_buffer;
    size_t log_buffer_size;
    size_t log_buffer_used;
    tq_device_context_t device_contexts[MAX_CONTEXTS];
    int device_context_count;
    tq_layer_entry_t layers[MAX_LAYERS];
    int layer_count;
    tq_context_create_callback_t context_create_cb;
    void* context_create_cb_data;
    tq_device_select_callback_t device_select_cb;
    void* device_select_cb_data;
    volatile int active;
} tq_global_state_t;

static tq_global_state_t g_state = {0};
static int g_initialized = 0;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

static void* g_real_opencl = NULL;
static pthread_mutex_t g_call_mutex = PTHREAD_MUTEX_INITIALIZER;

static void ensure_log_file() {
    if (!g_state.log_file && g_state.config.log_file_path) {
        g_state.log_file = fopen(g_state.config.log_file_path, "a");
        if (!g_state.log_file) {
            LOGE("Failed to open log file: %s\n", g_state.config.log_file_path);
        }
    }
}

static void flush_log_buffer() {
    if (g_state.log_file && g_state.log_buffer_used > 0) {
        fwrite(g_state.log_buffer, 1, g_state.log_buffer_used, g_state.log_file);
        fflush(g_state.log_file);
        g_state.log_buffer_used = 0;
    }
}

static void log_to_buffer(const char* msg) {
    size_t len = strlen(msg);
    if (g_state.log_buffer_used + len >= g_state.log_buffer_size) {
        flush_log_buffer();
    }
    if (g_state.log_buffer) {
        memcpy(g_state.log_buffer + g_state.log_buffer_used, msg, len);
        g_state.log_buffer_used += len;
    }
}

static void log_api_call(const char* func_name, void* args, void* ret, uint64_t duration_ns) {
    if (!(g_state.config.flags & TQ_INTERCEPT_FLAG_TRACE_CALLS)) return;

    char buf[512];
    int offset = 0;
    offset += snprintf(buf + offset, sizeof(buf) - offset, "[%llu] API: %s (took %llu ns)",
                      (unsigned long long)get_time_ns(), func_name,
                      (unsigned long long)duration_ns);

    if ((g_state.config.flags & TQ_INTERCEPT_FLAG_TRACE_ARGS) && args) {
        offset += snprintf(buf + offset, sizeof(buf) - offset, " [ARGS]");
    }

    if (g_state.config.flags & TQ_INTERCEPT_FLAG_TRACE_RETURNS && ret) {
        offset += snprintf(buf + offset, sizeof(buf) - offset, " -> ret");
    }

    offset += snprintf(buf + offset, sizeof(buf) - offset, "\n");
    log_to_buffer(buf);
    ensure_log_file();
}

cl_int tq_intercept_init(tq_intercept_config_t* config) {
    pthread_mutex_lock(&g_init_mutex);

    if (g_initialized) {
        pthread_mutex_unlock(&g_init_mutex);
        return CL_SUCCESS;
    }

    if (!config) {
        LOGE("Null config passed to tq_intercept_init\n");
        pthread_mutex_unlock(&g_init_mutex);
        return CL_INVALID_VALUE;
    }

    memset(&g_state, 0, sizeof(g_state));
    pthread_mutex_init(&g_state.lock, NULL);

    g_state.config = *config;
    g_state.active = 1;

    if (g_state.config.log_buffer_size == 0) {
        g_state.config.log_buffer_size = DEFAULT_LOG_BUFFER_SIZE;
    }
    if (g_state.config.max_mem_ops == 0) {
        g_state.config.max_mem_ops = DEFAULT_MAX_MEM_OPS;
    }
    if (g_state.config.max_kernel_execs == 0) {
        g_state.config.max_kernel_execs = DEFAULT_MAX_KERNEL_EXECS;
    }

    g_state.log_buffer = (char*)malloc(g_state.config.log_buffer_size);
    if (!g_state.log_buffer) {
        LOGE("Failed to allocate log buffer\n");
        pthread_mutex_unlock(&g_init_mutex);
        return CL_OUT_OF_HOST_MEMORY;
    }

    if (config->log_file_path) {
        g_state.log_file = fopen(config->log_file_path, "a");
        if (!g_state.log_file) {
            LOGE("Failed to open log file: %s\n", config->log_file_path);
        }
    }

    g_real_opencl = dlopen("libOpenCL.so", RTLD_NOW | RTLD_NOLOAD);
    if (!g_real_opencl) {
        g_real_opencl = dlopen("/system/vendor/lib64/libOpenCL.so", RTLD_NOW);
    }
    if (!g_real_opencl) {
        g_real_opencl = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    }
    if (!g_real_opencl) {
        LOGI("OpenCL library not preloaded — using intercept mode only\n");
    }

    LOGI("TurboQuant OpenCL-Intercept-Layer v%d.%d.%d initialized\n",
         TQ_INTERCEPT_VERSION_MAJOR, TQ_INTERCEPT_VERSION_MINOR,
         TQ_INTERCEPT_VERSION_PATCH);
    LOGI("Flags: 0x%x, log_buffer: %u bytes, max_mem_ops: %u, max_kernels: %u\n",
         config->flags, config->log_buffer_size,
         config->max_mem_ops, config->max_kernel_execs);

    g_initialized = 1;
    pthread_mutex_unlock(&g_init_mutex);
    return CL_SUCCESS;
}

cl_int tq_intercept_shutdown(void) {
    if (!g_initialized) return CL_SUCCESS;

    pthread_mutex_lock(&g_init_mutex);

    flush_log_buffer();

    if (g_state.log_file) {
        fclose(g_state.log_file);
        g_state.log_file = NULL;
    }

    free(g_state.log_buffer);
    g_state.log_buffer = NULL;

    for (int i = 0; i < g_state.layer_count; i++) {
        if (g_state.layers[i].layer && g_state.layers[i].layer->shutdown) {
            g_state.layers[i].layer->shutdown(g_state.layers[i].layer);
        }
    }

    if (g_real_opencl) {
        dlclose(g_real_opencl);
        g_real_opencl = NULL;
    }

    pthread_mutex_destroy(&g_state.lock);
    memset(&g_state, 0, sizeof(g_state));
    g_initialized = 0;

    LOGI("TurboQuant OpenCL-Intercept-Layer shutdown complete\n");

    pthread_mutex_unlock(&g_init_mutex);
    return CL_SUCCESS;
}

tq_intercept_context_t* tq_intercept_get_context(cl_device_id device) {
    if (!g_initialized) return NULL;

    pthread_mutex_lock(&g_state.lock);

    for (int i = 0; i < g_state.device_context_count; i++) {
        if (g_state.device_contexts[i].device == device) {
            pthread_mutex_unlock(&g_state.lock);
            return (tq_intercept_context_t*)&g_state.device_contexts[i];
        }
    }

    if (g_state.device_context_count < MAX_CONTEXTS) {
        int idx = g_state.device_context_count++;
        g_state.device_contexts[idx].device = device;
        g_state.device_contexts[idx].created_at_ns = get_time_ns();
        pthread_mutex_unlock(&g_state.lock);
        return (tq_intercept_context_t*)&g_state.device_contexts[idx];
    }

    pthread_mutex_unlock(&g_state.lock);
    return NULL;
}

void tq_intercept_set_context_create_callback(
    tq_context_create_callback_t callback, void* user_data) {
    g_state.context_create_cb = callback;
    g_state.context_create_cb_data = user_data;
}

void tq_intercept_set_device_select_callback(
    tq_device_select_callback_t callback, void* user_data) {
    g_state.device_select_cb = callback;
    g_state.device_select_cb_data = user_data;
}

cl_int tq_intercept_register_layer(tq_intercept_layer_t* layer) {
    if (!layer || !layer->name) {
        return CL_INVALID_VALUE;
    }

    pthread_mutex_lock(&g_state.lock);

    if (g_state.layer_count >= MAX_LAYERS) {
        pthread_mutex_unlock(&g_state.lock);
        return CL_OUT_OF_RESOURCES;
    }

    g_state.layers[g_state.layer_count].layer = layer;
    g_state.layers[g_state.layer_count].layer_data = NULL;
    g_state.layer_count++;

    LOGI("Registered intercept layer: %s\n", layer->name);

    pthread_mutex_unlock(&g_state.lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_unregister_layer(const char* layer_name) {
    if (!layer_name) return CL_INVALID_VALUE;

    pthread_mutex_lock(&g_state.lock);

    for (int i = 0; i < g_state.layer_count; i++) {
        if (strcmp(g_state.layers[i].layer->name, layer_name) == 0) {
            if (g_state.layers[i].layer->shutdown) {
                g_state.layers[i].layer->shutdown(g_state.layers[i].layer);
            }

            for (int j = i; j < g_state.layer_count - 1; j++) {
                g_state.layers[j] = g_state.layers[j + 1];
            }
            g_state.layer_count--;

            LOGI("Unregistered intercept layer: %s\n", layer_name);
            pthread_mutex_unlock(&g_state.lock);
            return CL_SUCCESS;
        }
    }

    pthread_mutex_unlock(&g_state.lock);
    return CL_INVALID_VALUE;
}

cl_int tq_intercept_get_metrics(tq_perf_metrics_t* metrics) {
    if (!metrics) return CL_INVALID_VALUE;

    pthread_mutex_lock(&g_state.lock);
    *metrics = g_state.metrics;

    if (g_state.metrics.total_api_calls > 0) {
        metrics->avg_api_call_ns = (double)g_state.metrics.total_api_time_ns /
                                   g_state.metrics.total_api_calls;
    }
    if (g_state.metrics.total_mem_ops > 0) {
        metrics->avg_mem_op_ns = (double)g_state.metrics.total_mem_time_ns /
                                 g_state.metrics.total_mem_ops;
    }
    if (g_state.metrics.total_kernel_execs > 0) {
        metrics->avg_kernel_exec_ns = (double)g_state.metrics.total_kernel_time_ns /
                                       g_state.metrics.total_kernel_execs;
    }

    pthread_mutex_unlock(&g_state.lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_reset_metrics(void) {
    pthread_mutex_lock(&g_state.lock);
    memset(&g_state.metrics, 0, sizeof(tq_perf_metrics_t));
    pthread_mutex_unlock(&g_state.lock);
    return CL_SUCCESS;
}

cl_int tq_intercept_get_mem_allocations(
    tq_mem_alloc_info_t* allocations, uint32_t* count, uint32_t max_count) {
    if (!count) return CL_INVALID_VALUE;
    *count = 0;

    if (!allocations || max_count == 0) {
        return CL_SUCCESS;
    }

    return CL_SUCCESS;
}

cl_int tq_intercept_flush_logs(void) {
    if (!g_initialized) return CL_INVALID_OPERATION;
    flush_log_buffer();
    return CL_SUCCESS;
}

const char* tq_intercept_version_string(void) {
    static char version[64];
    snprintf(version, sizeof(version), "%d.%d.%d",
             TQ_INTERCEPT_VERSION_MAJOR, TQ_INTERCEPT_VERSION_MINOR,
             TQ_INTERCEPT_VERSION_PATCH);
    return version;
}

cl_bool tq_intercept_is_active(void) {
    return g_initialized ? CL_TRUE : CL_FALSE;
}

#ifdef __cplusplus
namespace tq {
namespace intercept {

Context::Context(cl_device_id device) : device_(device), active_(false), ctx_(nullptr) {
    ctx_ = tq_intercept_get_context(device);
    active_ = (ctx_ != nullptr);
}

Context::~Context() {
}

Context::Context(Context&& other) noexcept
    : device_(other.device_), active_(other.active_), ctx_(other.ctx_) {
    other.active_ = false;
    other.ctx_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
    if (this != &other) {
        device_ = other.device_;
        active_ = other.active_;
        ctx_ = other.ctx_;
        other.active_ = false;
        other.ctx_ = nullptr;
    }
    return *this;
}

std::vector<tq_mem_alloc_info_t> Context::get_allocations() {
    std::vector<tq_mem_alloc_info_t> result;
    uint32_t count = 0;
    tq_mem_alloc_info_t allocs[64];
    if (tq_intercept_get_mem_allocations(allocs, &count, 64) == CL_SUCCESS) {
        for (uint32_t i = 0; i < count; i++) {
            result.push_back(allocs[i]);
        }
    }
    return result;
}

tq_perf_metrics_t Context::get_metrics() {
    tq_perf_metrics_t m;
    memset(&m, 0, sizeof(m));
    tq_intercept_get_metrics(&m);
    return m;
}

void Context::flush_logs() {
    tq_intercept_flush_logs();
}

ConfigBuilder& ConfigBuilder::set_flags(tq_intercept_flags_t flags) {
    config_.flags = flags;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_trace_calls() {
    config_.flags |= TQ_INTERCEPT_FLAG_TRACE_CALLS;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_trace_args() {
    config_.flags |= TQ_INTERCEPT_FLAG_TRACE_ARGS;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_trace_returns() {
    config_.flags |= TQ_INTERCEPT_FLAG_TRACE_RETURNS;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_log_memory() {
    config_.flags |= TQ_INTERCEPT_FLAG_LOG_MEMORY;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_log_kernels() {
    config_.flags |= TQ_INTERCEPT_FLAG_LOG_KERNELS;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_profile_perf() {
    config_.flags |= TQ_INTERCEPT_FLAG_PROFILE_PERF;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_layer_chain() {
    config_.flags |= TQ_INTERCEPT_FLAG_CHAIN_LAYERS;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_thread_safety() {
    config_.flags |= TQ_INTERCEPT_FLAG_THREAD_SAFE;
    return *this;
}

ConfigBuilder& ConfigBuilder::enable_buffer_recycling() {
    config_.flags |= TQ_INTERCEPT_FLAG_BUFFER_RECYCLING;
    return *this;
}

ConfigBuilder& ConfigBuilder::set_log_file(const std::string& path) {
    config_.log_file_path = path.c_str();
    return *this;
}

ConfigBuilder& ConfigBuilder::set_log_buffer_size(uint32_t size) {
    config_.log_buffer_size = size;
    return *this;
}

ConfigBuilder& ConfigBuilder::set_max_mem_ops(uint32_t max) {
    config_.max_mem_ops = max;
    return *this;
}

ConfigBuilder& ConfigBuilder::set_max_kernel_execs(uint32_t max) {
    config_.max_kernel_execs = max;
    return *this;
}

ConfigBuilder& ConfigBuilder::set_thread_count(uint32_t count) {
    config_.thread_count = count;
    return *this;
}

ConfigBuilder& ConfigBuilder::add_layer(tq_intercept_layer_t* layer) {
    config_.layer_chain = layer;
    return *this;
}

tq_intercept_config_t ConfigBuilder::build() {
    return config_;
}

} // namespace intercept
} // namespace tq
#endif // __cplusplus