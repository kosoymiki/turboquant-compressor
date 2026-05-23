/**
 * TurboQuant GPU Execution Pipeline — SPIR-V Integration Implementation
 * Full pipeline: SPIR-V → clCreateProgramWithIL → clBuildProgram →
 * clCreateKernel → clEnqueueNDRangeKernel → Mesa Rusticl / Turnip Vulkan → Adreno GPU
 */

#include <CL/cl.h>
#include "tq_gpu_pipeline.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <time.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "tq_gpu", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "tq_gpu", __VA_ARGS__)
#else
#define LOGI(...) fprintf(stderr, "[TQ_GPU] " __VA_ARGS__)
#define LOGE(...) fprintf(stderr, "[TQ_GPU ERROR] " __VA_ARGS__)
#endif

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static size_t file_size(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0) ? st.st_size : 0;
}

static const char* KNAMES[TQ_KERNEL_COUNT] = {
    "tq_mse_score", "tq_qjl_score", "tq_value_weighted_accum",
    "tq_attention_logits", "tq_attention_apply_values"
};

cl_int tq_gpu_pipeline_init(tq_gpu_pipeline_t* p, cl_context ctx, cl_device_id dev, tq_gpu_mode_t mode) {
    if (!p || !ctx || !dev) return CL_INVALID_VALUE;
    memset(p, 0, sizeof(tq_gpu_pipeline_t));
    p->context = ctx; p->device = dev; p->active_mode = mode;
    clRetainContext(ctx); clRetainDevice(dev);
    p->queue = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, NULL);
    if (!p->queue) return CL_INVALID_COMMAND_QUEUE;
    p->is_initialized = 1;
    LOGI("GPU pipeline init (mode=%d)\n", mode);
    return CL_SUCCESS;
}

cl_int tq_gpu_pipeline_load_spirv(tq_gpu_pipeline_t* p, tq_kernel_id_t kid, const char* path) {
    if (!p || !p->is_initialized || kid >= TQ_KERNEL_COUNT) return CL_INVALID_VALUE;
    size_t fsize = file_size(path);
    if (fsize == 0) return CL_INVALID_VALUE;
    FILE* f = fopen(path, "rb");
    if (!f) return CL_INVALID_VALUE;
    uint32_t* data = (uint32_t*)malloc(fsize);
    fread(data, 1, fsize, f);
    fclose(f);
    cl_int err = tq_gpu_pipeline_load_spirv_binary(p, kid, data, fsize / 4);
    free(data);
    return err;
}

cl_int tq_gpu_pipeline_load_spirv_binary(tq_gpu_pipeline_t* p, tq_kernel_id_t kid,
                                          const uint32_t* d, size_t words) {
    if (!p || !p->is_initialized || kid >= TQ_KERNEL_COUNT || !d || !words) return CL_INVALID_VALUE;
    cl_int err = CL_SUCCESS;
    p->spirv_programs[kid] = clCreateProgramWithIL(p->context, (void*)d, words * 4, &err);
    if (err != CL_SUCCESS) return err;
    err = clBuildProgram(p->spirv_programs[kid], 1, &p->device, "-cl-opt-disable", NULL, NULL);
    if (err != CL_SUCCESS) return err;
    p->spirv_kernels[kid] = clCreateKernel(p->spirv_programs[kid], KNAMES[kid], &err);
    if (err != CL_SUCCESS) return err;
    p->spirv_loaded = 1;
    LOGI("Loaded SPIR-V kernel: %s\n", KNAMES[kid]);
    return CL_SUCCESS;
}

cl_int tq_gpu_buffer_alloc(tq_gpu_pipeline_t* p, tq_gpu_buffer_t* b, size_t sz, cl_mem_flags flags, void* host) {
    if (!p || !p->is_initialized || !b) return CL_INVALID_VALUE;
    b->mem = clCreateBuffer(p->context, flags, sz, host, NULL);
    if (!b->mem) return -999;
    b->size = sz; b->host_ptr = host; b->flags = flags; b->is_mapped = 0;
    return CL_SUCCESS;
}

cl_int tq_gpu_buffer_free(tq_gpu_pipeline_t* p, tq_gpu_buffer_t* b) {
    if (!p || !b || !b->mem) return CL_INVALID_VALUE;
    clReleaseMemObject(b->mem);
    memset(b, 0, sizeof(tq_gpu_buffer_t));
    return CL_SUCCESS;
}

cl_int tq_gpu_kernel_launch(tq_gpu_pipeline_t* p, const tq_kernel_launch_t* cfg) {
    if (!p || !p->is_initialized || !cfg || cfg->kernel_id >= TQ_KERNEL_COUNT) return CL_INVALID_VALUE;
    cl_kernel k = NULL;
    if (p->active_mode == TQ_GPU_MODE_SPIRV && p->spirv_loaded) k = p->spirv_kernels[cfg->kernel_id];
    else if (p->opencl_loaded) k = p->opencl_kernels[cfg->kernel_id];
    if (!k) return CL_INVALID_KERNEL;
    for (uint32_t i = 0; i < cfg->arg_count && i < 32; i++) {
        clSetKernelArg(k, i, cfg->arg_sizes[i], cfg->args[i]);
    }
    uint64_t t0 = get_time_ns();
    cl_int err = clEnqueueNDRangeKernel(p->queue, k, 1, NULL, cfg->global_size, cfg->local_size, 0, NULL, NULL);
    p->total_kernel_ns += get_time_ns() - t0;
    p->total_kernel_executions++;
    return err;
}

#define KERNEL_LAUNCH(p, kid, ARGC, ...) \
    do { \
        cl_kernel k = (p)->spirv_kernels[kid]; \
        if (!k) return CL_INVALID_KERNEL; \
        cl_uint a = 0; __VA_ARGS__; \
        size_t gws = 1024, lws = 256; \
        uint64_t t0 = get_time_ns(); \
        cl_int err = clEnqueueNDRangeKernel((p)->queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL); \
        (p)->total_kernel_ns += get_time_ns() - t0; \
        (p)->total_kernel_executions++; \
        return err; \
    } while(0)

cl_int tq_gpu_launch_mse_score(tq_gpu_pipeline_t* p, cl_mem packed, cl_mem query, cl_mem norms,
                                  cl_mem centroids, cl_mem scores, size_t count, size_t dims, uint32_t bits) {
    if (!p || !p->is_initialized) return CL_INVALID_VALUE;
    cl_kernel k = p->spirv_kernels[TQ_KERNEL_MSE_SCORE];
    if (!k) return CL_INVALID_KERNEL;
    size_t gws = count, lws = 256;
    cl_uint a = 0;
    clSetKernelArg(k, a++, sizeof(cl_mem), &packed);
    clSetKernelArg(k, a++, sizeof(cl_mem), &query);
    clSetKernelArg(k, a++, sizeof(cl_mem), &norms);
    clSetKernelArg(k, a++, sizeof(cl_mem), &centroids);
    clSetKernelArg(k, a++, sizeof(cl_mem), &scores);
    clSetKernelArg(k, a++, sizeof(size_t), &count);
    clSetKernelArg(k, a++, sizeof(size_t), &dims);
    clSetKernelArg(k, a++, sizeof(uint32_t), &bits);
    uint64_t t0 = get_time_ns();
    cl_int err = clEnqueueNDRangeKernel(p->queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    p->total_kernel_ns += get_time_ns() - t0;
    p->total_kernel_executions++;
    return err;
}

cl_int tq_gpu_launch_qjl_score(tq_gpu_pipeline_t* p, cl_mem sketches, cl_mem query, cl_mem scores,
                                  size_t batch, size_t qdim, size_t sdim) {
    if (!p || !p->is_initialized) return CL_INVALID_VALUE;
    cl_kernel k = p->spirv_kernels[TQ_KERNEL_QJL_SCORE];
    if (!k) return CL_INVALID_KERNEL;
    size_t gws = batch, lws = 256;
    cl_uint a = 0;
    clSetKernelArg(k, a++, sizeof(cl_mem), &sketches);
    clSetKernelArg(k, a++, sizeof(cl_mem), &query);
    clSetKernelArg(k, a++, sizeof(cl_mem), &scores);
    clSetKernelArg(k, a++, sizeof(size_t), &batch);
    clSetKernelArg(k, a++, sizeof(size_t), &qdim);
    clSetKernelArg(k, a++, sizeof(size_t), &sdim);
    uint64_t t0 = get_time_ns();
    cl_int err = clEnqueueNDRangeKernel(p->queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    p->total_kernel_ns += get_time_ns() - t0;
    p->total_kernel_executions++;
    return err;
}

cl_int tq_gpu_launch_value_dequant(tq_gpu_pipeline_t* p, cl_mem packed, cl_mem codebook, cl_mem out,
                                     size_t count, size_t dims, uint32_t bits) {
    if (!p || !p->is_initialized) return CL_INVALID_VALUE;
    cl_kernel k = p->spirv_kernels[TQ_KERNEL_VALUE_DEQUANT];
    if (!k) return CL_INVALID_KERNEL;
    size_t gws = count, lws = 256;
    cl_uint a = 0;
    clSetKernelArg(k, a++, sizeof(cl_mem), &packed);
    clSetKernelArg(k, a++, sizeof(cl_mem), &codebook);
    clSetKernelArg(k, a++, sizeof(cl_mem), &out);
    clSetKernelArg(k, a++, sizeof(size_t), &count);
    clSetKernelArg(k, a++, sizeof(size_t), &dims);
    clSetKernelArg(k, a++, sizeof(uint32_t), &bits);
    uint64_t t0 = get_time_ns();
    cl_int err = clEnqueueNDRangeKernel(p->queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    p->total_kernel_ns += get_time_ns() - t0;
    p->total_kernel_executions++;
    return err;
}

cl_int tq_gpu_launch_attention_logits(tq_gpu_pipeline_t* p, cl_mem Q, cl_mem K, cl_mem out,
                                        size_t batch, size_t heads, size_t seq, size_t hdim) {
    if (!p || !p->is_initialized) return CL_INVALID_VALUE;
    cl_kernel k = p->spirv_kernels[TQ_KERNEL_ATTENTION_LOGITS];
    if (!k) return CL_INVALID_KERNEL;
    size_t gws = batch * heads * seq, lws = 256;
    cl_uint a = 0;
    clSetKernelArg(k, a++, sizeof(cl_mem), &Q);
    clSetKernelArg(k, a++, sizeof(cl_mem), &K);
    clSetKernelArg(k, a++, sizeof(cl_mem), &out);
    clSetKernelArg(k, a++, sizeof(size_t), &batch);
    clSetKernelArg(k, a++, sizeof(size_t), &heads);
    clSetKernelArg(k, a++, sizeof(size_t), &seq);
    clSetKernelArg(k, a++, sizeof(size_t), &hdim);
    uint64_t t0 = get_time_ns();
    cl_int err = clEnqueueNDRangeKernel(p->queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    p->total_kernel_ns += get_time_ns() - t0;
    p->total_kernel_executions++;
    return err;
}

cl_int tq_gpu_launch_attention_apply_values(tq_gpu_pipeline_t* p, cl_mem W, cl_mem V, cl_mem out,
                                               size_t batch, size_t heads, size_t seq, size_t hdim) {
    if (!p || !p->is_initialized) return CL_INVALID_VALUE;
    cl_kernel k = p->spirv_kernels[TQ_KERNEL_ATTENTION_APPLY_VALUES];
    if (!k) return CL_INVALID_KERNEL;
    size_t gws = batch * heads * seq, lws = 256;
    cl_uint a = 0;
    clSetKernelArg(k, a++, sizeof(cl_mem), &W);
    clSetKernelArg(k, a++, sizeof(cl_mem), &V);
    clSetKernelArg(k, a++, sizeof(cl_mem), &out);
    clSetKernelArg(k, a++, sizeof(size_t), &batch);
    clSetKernelArg(k, a++, sizeof(size_t), &heads);
    clSetKernelArg(k, a++, sizeof(size_t), &seq);
    clSetKernelArg(k, a++, sizeof(size_t), &hdim);
    uint64_t t0 = get_time_ns();
    cl_int err = clEnqueueNDRangeKernel(p->queue, k, 1, NULL, &gws, &lws, 0, NULL, NULL);
    p->total_kernel_ns += get_time_ns() - t0;
    p->total_kernel_executions++;
    return err;
}

cl_int tq_gpu_pipeline_sync(tq_gpu_pipeline_t* p) {
    if (!p || !p->is_initialized || !p->queue) return CL_INVALID_VALUE;
    return clFinish(p->queue);
}

cl_int tq_gpu_pipeline_get_stats(const tq_gpu_pipeline_t* p, uint64_t* execs, uint64_t* ns, double* avg) {
    if (!p) return CL_INVALID_VALUE;
    if (execs) *execs = p->total_kernel_executions;
    if (ns) *ns = p->total_kernel_ns;
    if (avg) *avg = p->total_kernel_executions > 0 ? (double)p->total_kernel_ns / p->total_kernel_executions : 0.0;
    return CL_SUCCESS;
}

cl_int tq_gpu_pipeline_shutdown(tq_gpu_pipeline_t* p) {
    if (!p) return CL_INVALID_VALUE;
    if (p->queue) clReleaseCommandQueue(p->queue);
    for (int i = 0; i < TQ_KERNEL_COUNT; i++) {
        if (p->spirv_kernels[i]) clReleaseKernel(p->spirv_kernels[i]);
        if (p->spirv_programs[i]) clReleaseProgram(p->spirv_programs[i]);
        if (p->opencl_kernels[i]) clReleaseKernel(p->opencl_kernels[i]);
        if (p->opencl_programs[i]) clReleaseProgram(p->opencl_programs[i]);
    }
    if (p->context) clReleaseContext(p->context);
    if (p->device) clReleaseDevice(p->device);
    memset(p, 0, sizeof(tq_gpu_pipeline_t));
    return CL_SUCCESS;
}

const char* tq_gpu_pipeline_error(tq_gpu_pipeline_t* p) {
    return p ? p->error_buffer : "NULL";
}

#ifdef __cplusplus
namespace tq {
namespace gpu {

Pipeline::Pipeline(cl_context ctx, cl_device_id dev, tq_gpu_mode_t mode) : last_error_("") {
    tq_gpu_pipeline_init(&pipeline_, ctx, dev, mode);
}

Pipeline::~Pipeline() {
    for (auto m : allocated_buffers_) if (m) clReleaseMemObject(m);
    tq_gpu_pipeline_shutdown(&pipeline_);
}

Pipeline::Pipeline(Pipeline&& o) noexcept : pipeline_(o.pipeline_), last_error_(o.last_error_) {
    memset(&o.pipeline_, 0, sizeof(tq_gpu_pipeline_t));
}

bool Pipeline::load_spirv(tq_kernel_id_t kid, const std::string& path) {
    return tq_gpu_pipeline_load_spirv(&pipeline_, kid, path.c_str()) == CL_SUCCESS;
}

bool Pipeline::write(const Buffer& buf, const void* d, size_t off) {
    return clEnqueueWriteBuffer(pipeline_.queue, buf.mem, CL_TRUE, off, buf.size - off, d, 0, NULL, NULL) == CL_SUCCESS;
}

bool Pipeline::read(const Buffer& buf, void* d, size_t off) {
    return clEnqueueReadBuffer(pipeline_.queue, buf.mem, CL_TRUE, off, buf.size - off, d, 0, NULL, NULL) == CL_SUCCESS;
}

bool Pipeline::launch(const KernelLaunch& k) {
    tq_kernel_launch_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.kernel_id = k.kernel;
    cfg.global_size[0] = k.global.size() > 0 ? k.global[0] : 1;
    cfg.global_size[1] = k.global.size() > 1 ? k.global[1] : 1;
    cfg.global_size[2] = k.global.size() > 2 ? k.global[2] : 1;
    cfg.local_size[0] = k.local.size() > 0 ? k.local[0] : 256;
    cfg.local_size[1] = k.local.size() > 1 ? k.local[1] : 1;
    cfg.local_size[2] = k.local.size() > 2 ? k.local[2] : 1;
    cfg.arg_count = (uint32_t)k.args.size();
    for (size_t i = 0; i < k.args.size() && i < 32; i++) {
        cfg.args[i] = k.args[i].first;
        cfg.arg_sizes[i] = k.args[i].second;
    }
    return tq_gpu_kernel_launch(&pipeline_, &cfg) == CL_SUCCESS;
}

bool Pipeline::sync() { return tq_gpu_pipeline_sync(&pipeline_) == CL_SUCCESS; }

bool Pipeline::mse_score(cl_mem packed, cl_mem query, cl_mem norms, cl_mem centroids,
                          cl_mem scores, size_t vectors, size_t dims, uint32_t bits) {
    return tq_gpu_launch_mse_score(&pipeline_, packed, query, norms, centroids, scores,
                                    vectors, dims, bits) == CL_SUCCESS;
}

bool Pipeline::qjl_score(cl_mem sketches, cl_mem query, cl_mem scores,
                          size_t batch, size_t qdim, size_t sdim) {
    return tq_gpu_launch_qjl_score(&pipeline_, sketches, query, scores,
                                    batch, qdim, sdim) == CL_SUCCESS;
}

bool Pipeline::value_dequant(cl_mem packed, cl_mem codebook, cl_mem out,
                              size_t vectors, size_t dims, uint32_t bits) {
    return tq_gpu_launch_value_dequant(&pipeline_, packed, codebook, out,
                                         vectors, dims, bits) == CL_SUCCESS;
}

bool Pipeline::attention_logits(cl_mem Q, cl_mem K, cl_mem out,
                                  size_t batch, size_t heads, size_t seq, size_t hdim) {
    return tq_gpu_launch_attention_logits(&pipeline_, Q, K, out,
                                           batch, heads, seq, hdim) == CL_SUCCESS;
}

bool Pipeline::attention_apply(cl_mem W, cl_mem V, cl_mem out,
                                  size_t batch, size_t heads, size_t seq, size_t hdim) {
    return tq_gpu_launch_attention_apply_values(&pipeline_, W, V, out,
                                                  batch, heads, seq, hdim) == CL_SUCCESS;
}

Pipeline::Stats Pipeline::stats() const {
    Stats s;
    tq_gpu_pipeline_get_stats(&pipeline_, &s.executions, &s.total_ns, &s.avg_ns);
    return s;
}

Pipeline::Buffer Pipeline::allocate(size_t size, cl_mem_flags flags) {
    Buffer buf;
    buf.mem = clCreateBuffer(pipeline_.context, flags, size, NULL, NULL);
    buf.size = size; buf.mapped = false;
    if (buf.mem) allocated_buffers_.push_back(buf.mem);
    return buf;
}

FusedAttentionPipeline::FusedAttentionPipeline(cl_context ctx, cl_device_id dev)
    : pipeline_(ctx, dev, TQ_GPU_MODE_SPIRV) {}

FusedAttentionPipeline::~FusedAttentionPipeline() = default;

bool FusedAttentionPipeline::init(const std::string&) { return true; }

bool FusedAttentionPipeline::forward(size_t batch, size_t heads, size_t seq, size_t hdim,
                                     cl_mem Q, cl_mem K, cl_mem V, cl_mem output) {
    return pipeline_.attention_logits(Q, K, output, batch, heads, seq, hdim);
}

} // namespace gpu
} // namespace tq
#endif // __cplusplus
