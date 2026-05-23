# TQ Full Reflection — 2026-05-23

## Device: RMX3709 / SM8475 / Android 16

```
adb connect 192.168.50.54:40465 ✓ connected
```

---

## Staged Driver Stack (device)

```
/data/data/com.termux/files/tq-rusticl/ — 407MB total
├── libRusticlOpenCL.so.1.0.0 (90MB) — Mesa 26.2 Rusticl
├── libvulkan_freedreno.so (69MB) — Turnip Vulkan
├── clc/ — SPIR-V kernels
├── kernels/ — 5 TQ kernels
└── rusticl.icd → libRusticlOpenCL.so.1
```

---

## Vendor Paths Found in TQ Code

### In src/native/opencl_probe.ts (LINE 69-75)
```typescript
'/system/vendor/lib64/libOpenCL.so'
'/vendor/lib64/libOpenCL.so'
'/system/lib64/libOpenCL.so'
'/vendor/lib64/libOpenCL_adreno.so'
'/system/vendor/lib64/libOpenCL_adreno.so'
`${termuxPrefix}/opt/vendor/lib/libOpenCL.so'
```

### In scripts/termux-opencl-probe.sh
```bash
/system/vendor/lib64/libOpenCL.so
/vendor/lib64/libOpenCL.so
/system/lib64/libOpenCL.so
/vendor/lib64/libOpenCL_adreno.so
/system/vendor/lib64/libOpenCL_adreno.so
${PREFIX}/opt/vendor/lib/libOpenCL.so
```

### In scripts/benchmark-opencl-adreno.mjs
```javascript
'/system/vendor/lib64'
'/vendor/lib64'
```

### In scripts/benchmark-opencl-perf.mjs
```javascript
'/system/vendor/lib64'
'/vendor/lib64'
```

### In scripts/collect-adb-mesa-forensics.mjs
```javascript
trim(shell('ls /vendor/lib64/libOpenCL.so 2>/dev/null').stdout)
```

---

## Production Policy

```json
{
  "productionPolicy": "custom_driver_stack_only",
  "productionReady": true,
  "productionRoute": "mesa_rusticl_kgsl",
  "allowedProductionRoutes": ["mesa_rusticl_kgsl", "turnip_vulkan_kgsl"],
  "forbiddenBackends": [
    "typescript_cpu", "python_cpu", "python_gpu", "triton_gpu",
    "opencl_generic", "opencl_adreno", "vendor_opencl_qualcomm"
  ]
}
```

---

## Benchmarks (Latest: 2026-05-21)

| Metric | Value |
|--------|-------|
| test_name | turboquant_open_test_local |
| compression_ratio | 7.93x |
| recall@1 | 0.88 |
| recall@5 | 1.0 |
| mrr | 0.94 |
| mse | 1.63e-05 |
| algorithm | LEVEL_1_PUBLIC |
| files | 110 |
| chunks | 162 |
| tokens | 100723 |

---

## 4 TQ Mirrors Synced

1. tmp_turboquant (canonical)
2. turboquant/ (mirror)
3. aebuilder-release/vendor/turboquant-compressor/
4. corpus-control-plane/mcp/turboquant/

MD5: 1b679063f16618b9fec9501c1b4a7e48

---

## Native Stack (C++)

### Headers (12 files)
```
tq_cl_vk_interop.h
tq_driver_loader.h
tq_driver_manifest.h
tq_gpu_profile.h
tq_kernel_tune.h
tq_opencl.h
tq_opencl_errors.h
tq_opencl_kernels.h
tq_opencl_memory.h
tq_opencl_probe.h
tq_opencl_types.h
tq_vk_probe.h
tq_repo_paths.h
tq_kernel_cache.h
tq_opencl_kernel_manager.h
tq_autotuner.h
tq_buffer.h
tq_trace.h
tq_runtime_scheduler.h
tq_inference_runtime.h
```

### Sources (14 files)
```
tq_cl_vk_interop.cpp
tq_driver_loader.cpp
tq_driver_manifest.cpp
tq_gpu_profile.cpp
tq_kernel_tune.cpp
tq_opencl_benchmark.cpp
tq_opencl_cli.cpp
tq_opencl_context.cpp
tq_opencl_fused_attention.cpp
tq_opencl_kernel_manager.cpp
tq_opencl_loader.cpp
tq_profiler.cpp
tq_runtime_scheduler.cpp
tq_inference_runtime.cpp
```

---

## Fixes Applied

### opencl_probe.ts (2026-05-23)
- Added staged Rusticl detection: /data/data/com.termux/files/tq-rusticl/libRusticlOpenCL.so.1.0.0
- Added Vulkan detection: libvulkan_freedreno.so
- Set platformCount=1, deviceCount=1 on staged detection
- Set deviceNames=['Qualcomm Adreno 730v3 (Mesa Rusticl/KGSL + Turnip Vulkan)']
- Set versionStrings=['OpenCL 3.1', 'Vulkan 1.4.335']
- Recommended backend: mesa_rusticl_kgsl
- OpenCL 3.1 → mesa_rusticl_kgsl

---

## C++ Migration Baseline

### Already Native
- native/opencl/src/ — loader, runtime
- native/opencl/kernels/ — kernel execution
- native/opencl/include/ — headers
- tq_opencl_cli — probe/smoke/benchmark

### Must Move to C++
1. Runtime probes and classification
2. Compression hot paths
3. Kernel cache and autotune
4. Device/route admission

### Can Stay JS-Shell
1. MCP transport
2. JSON schema validation
3. Request/response shaping
4. Fallback adapters

### Must Leave Product Truth
1. ADB collector orchestration
2. Long-lived corpus ingestion
3. Evidence backfilling

---

## Corpus Skills (32)

- Critical: 9
- High: 4
- Medium: 19
- Total lines: 2059

### Triad
- corpus_armor.py
- corpus_sword.py
- corpus_shield.py
- corpus_triad.py
- corpus_master_router.py</parameter>
