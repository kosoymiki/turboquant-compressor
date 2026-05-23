/**
 * TurboQuant Inference Runtime — GPU Pipeline Integration
 *
 * Bridges tq_inference_runtime with tq_gpu_pipeline for full model execution
 * on Adreno GPU via Mesa Rusticl / Turnip Vulkan.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef TQ_INFERENCE_RUNTIME_GPU_H
#define TQ_INFERENCE_RUNTIME_GPU_H

#include "tq_inference_runtime.h"
#include "tq_gpu_pipeline.h"

namespace tq {

class InferenceRuntimeGpu {
public:
    InferenceRuntimeGpu();
    ~InferenceRuntimeGpu();

    bool init(cl_context ctx, cl_device_id dev);
    bool load_spirv_kernels(const std::string& spirv_dir);
    bool shutdown();

    // Execution paths
    bool execute_prefill(const InferenceRequestPlan& plan,
                         const std::vector<float>& query_rotated,
                         const std::vector<uint8_t>& key_packed,
                         const std::vector<float>& key_norms,
                         std::vector<float>* scores);

    bool execute_decode(const InferenceRequestPlan& plan,
                         const std::vector<float>& residual_signs,
                         const std::vector<uint8_t>& value_packed,
                         const std::vector<float>& value_scales,
                         const std::vector<float>& value_zeros,
                         std::vector<float>* output);

    bool execute_attention(const std::vector<float>& Q,
                           const std::vector<float>& K,
                           const std::vector<float>& V,
                           std::vector<float>* output,
                           size_t batch, size_t heads,
                           size_t seq, size_t head_dim);

    // Profiling
    InferenceRuntimeSummary summary() const;
    void reset_summary();
    void flush_kernel_stats();

private:
    gpu::Pipeline* gpu_pipeline_ = nullptr;
    bool is_initialized_ = false;
    std::string last_error_;
    InferenceRuntimeSummary summary_;

    struct KernelHandles {
        cl_mem query_buf = nullptr;
        cl_mem key_buf = nullptr;
        cl_mem value_buf = nullptr;
        cl_mem score_buf = nullptr;
        cl_mem attention_buf = nullptr;
    } buffers_;

    bool allocate_buffers(size_t max_tokens);
    bool free_buffers();
    bool copy_to_device(const void* host, cl_mem dev, size_t size);
    bool copy_from_device(cl_mem dev, void* host, size_t size);
};

} // namespace tq

#endif // TQ_INFERENCE_RUNTIME_GPU_H