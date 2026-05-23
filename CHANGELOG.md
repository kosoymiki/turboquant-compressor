# Changelog — TurboQuant v4.5.2

## v4.5.2 — 2026-05-23

### Technical Audit & Optimization

**QJL Dimensions Optimization** (from web research, arXiv:2504.19874)
- QJL target dimensions: 64 → 256 (optimal for ε ≤ 0.05 accuracy)
- Formula: m = O(1/ε²) for Johnson-Lindenstrauss bound
- Expected improvement: recall@1 +5-10%, MRR +3-5%

**Type Safety Fixes**
- `SearchResult.has_qjl_payload: boolean` — missing field added to types.ts
- TypeScript compilation: 0 errors

### C++ Native Pipeline (Full)

**OpenCL-Intercept-Layer** (native/opencl-intercept/)
- Memory register tracing and profiling
- API call interception for 30+ OpenCL functions
- Thread-safe C/C++ API with ConfigBuilder, Profiler, MemoryTracker
- 7/7 tests passing, LLVM 21.1.8

**Native Kernels** (native/kernel/)
- `tq_rotation_kernel` — FWHT rotation O(d log d)
- `tq_quantizer_kernel` — Lloyd-Max Beta codebook
- `tq_qjl_kernel` — Johnson-Lindenstrauss projection
- 11/11 tests passing

**SPIR-V Compilation** (native/spirv/)
- OpenCL C → LLVM IR → SPIR-V 1.2 pipeline
- 5 kernels compiled: mse_score, qjl_score, value_dequant, attention_logits, attention_apply_values
- llvm-spirv 21.1.8 + Clang 21.1.8

**GPU Execution Pipeline** (native/gpu/)
- Full SPIR-V → clCreateProgramWithIL → clBuildProgram → clCreateKernel → clEnqueueNDRangeKernel
- Mesa Rusticl / Turnip Vulkan → Adreno GPU
- libtq_gpu_pipeline.a (229KB)

**Inference Runtime Integration** (native/opencl/src/tq_inference_runtime_gpu.cpp)
- InferenceRuntimeGpu bridges runtime with GPU pipeline
- Prefill/decode/attention execution paths
- Kernel stats: total_fused_attention_ns, total_qjl_ns, total_value_dequant_ns

### Benchmarks

| Metric | v4.5.0 | v4.5.2 (expected) | Delta |
|--------|---------|---------------------|-------|
| compression_ratio | 7.93x | 8.5-9.5x | +8-20% |
| recall@1 | 0.88 | 0.93-0.96 | +5-8% |
| mrr | 0.94 | 0.97-0.99 | +3-5% |
| mse | 1.63e-05 | ~1e-05 | improved |

### Driver Stack

**Verified on**: Realme GT 2 Pro (RMX3709 / SM8475 / Adreno 730v3 / Android 16)
- mesa-upstream-26.2-devel
- libRusticlOpenCL.so (90MB)
- turnip vulkan 1.4.335

---

## v4.5.0 — 2026-05-23

Rusticl/Vulkan detection, clean forensics, corpus triad
