# TurboQuant GPU Execution Pipeline

**SPIR-V → clCreateProgramWithIL → clBuildProgram → clCreateKernel → clEnqueueNDRangeKernel → Mesa Rusticl / Turnip Vulkan → Adreno GPU**

| | |
|---|---|
| LLVM | 21.1.8 |
| SPIR-V | 1.2 |
| Status | Production-ready |

---

## Pipeline Overview

```
OpenCL C (.cl)
    ↓ clang -emit-llvm -target spir64
LLVM IR (.bc)
    ↓ llvm-spirv
SPIR-V 1.2 (.spv)
    ↓ clCreateProgramWithIL (OpenCL 2.1+)
Program
    ↓ clBuildProgram
Kernel
    ↓ clCreateKernel
Kernel Function
    ↓ clEnqueueNDRangeKernel
Mesa Rusticl / Turnip Vulkan → Adreno GPU (KGSL)
```

---

## Architecture

```
native/gpu/
├── include/
│   └── tq_gpu_pipeline.h     — Full pipeline API
├── src/
│   ├── tq_gpu_pipeline.cpp    — Implementation
│   └── test_gpu_pipeline.cpp — Tests
├── CMakeLists.txt
└── libtq_gpu_pipeline.a       — Static library (229KB)
```

---

## Kernels Executed

| Kernel | Description | Args |
|---------|-------------|------|
| `tq_mse_score` | MSE dot-product from packed indices | packed, query, norms, centroids, scores |
| `tq_qjl_score` | QJL-corrected dot products | sketches, query, scores |
| `tq_value_weighted_accum` | Value dequantization | packed, codebook, output |
| `tq_attention_logits` | Attention Q×K^T scores | Q, K, output |
| `tq_attention_apply_values` | Attention V×softmax weights | weights, V, output |

---

## C API

```c
#include "tq_gpu_pipeline.h"

tq_gpu_pipeline_t pipeline;
tq_gpu_pipeline_init(&pipeline, ctx, dev, TQ_GPU_MODE_SPIRV);
tq_gpu_pipeline_load_spirv(&pipeline, TQ_KERNEL_MSE_SCORE, "tq_mse_score.spv");

// Allocate buffers
tq_gpu_buffer_t scores;
tq_gpu_buffer_alloc(&pipeline, &scores, vector_count * sizeof(float), CL_MEM_WRITE_ONLY, NULL);

// Launch kernel
tq_gpu_launch_mse_score(&pipeline, packed, query, norms, centroids, scores.mem,
                          vector_count, dimensions, bits_per_value);

// Sync and shutdown
tq_gpu_pipeline_sync(&pipeline);
tq_gpu_pipeline_shutdown(&pipeline);
```

---

## C++ API

```cpp
#include "tq_gpu_pipeline.h"

tq::gpu::Pipeline pipeline(ctx, dev, TQ_GPU_MODE_SPIRV);
pipeline.load_spirv(TQ_KERNEL_MSE_SCORE, "tq_mse_score.spv");

auto scores = pipeline.allocate(vector_count * sizeof(float));
pipeline.mse_score(packed, query, norms, centroids, scores.mem,
                    vector_count, dimensions, bits);

pipeline.sync();
auto stats = pipeline.stats();
```

---

## Next: Inference Runtime Integration

```
tq_gpu_pipeline + tq_inference_runtime → tq_runtime_scheduler → Adreno GPU
     ↓
  SPIR-V path + OpenCL C fallback
```

See: `native/opencl/src/tq_inference_runtime.cpp`