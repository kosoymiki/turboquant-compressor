# TurboQuant C++ Migration Roadmap v4.5.4

## Executive Summary

Complete migration of TurboQuant hot-path from TypeScript to C++ for:
- Maximum performance (Android/Termux deployment)
- Custom driver stack integration (mesa/rusticl/turnip/kgsl)
- Production-grade vector compression

## Current Status

### COMPLETE ✅
| Module | C++ File | Lines | Status |
|--------|----------|-------|---------|
| format.ts | tq_compress.cpp, tq_search.cpp | 472 | ✅ |
| pack.ts | tq_kernel_inline.h | inline | ✅ |
| hadamard.ts | tq_kernel_inline.h | inline | ✅ |
| beta_sphere.ts | tq_beta_sphere.cpp | 400+ | ✅ |
| codebook.ts | tq_beta_sphere.cpp | merged | ✅ |
| quantizer.ts | tq_quantizer_kernel.cpp | 300+ | ✅ |
| qjl.ts | tq_qjl_kernel.cpp | 300+ | ✅ |
| rotation.ts | tq_rotation_kernel.cpp | 250+ | ✅ |
| crc32.ts | tq_format.cpp | inline | ✅ |
| prng.ts | tq_prng.cpp | 41 | ✅ |
| quality.ts | tq_quality.cpp | 142 | ✅ |
| limits.ts | tq_limits.cpp | 25 | ✅ |
| benchmark.ts | tq_benchmark.cpp | 250+ | ✅ |
| oracle.ts | tq_oracle.cpp | 180 | ✅ |
| context_pack_v2.ts | tq_context_pack.cpp | 200+ | ✅ |

### REMAINING WORK

## Phase 1: Core Hot Path — COMPLETE ✅
All Phase 1 modules migrated to C++:
- tq_kv_cache.cpp (KV cache compression)
- tq_benchmark.cpp (performance benchmarks)
- tq_oracle.cpp (quality oracle)
- tq_context_pack.cpp (context compression)

## Phase 2: Context & Storage — COMPLETE ✅

Context pack compression implemented in tq_context_pack.cpp

## Phase 3: Hybrid GPU Training (DEFERRED)

Training-time modules kept in TypeScript:
- precision_calibration.ts (FP8/INT8 calibration)
- lsq_optimizer.ts (least-squares optimization)
- distillation_for_quantization.ts (knowledge distillation)

## Phase 4: Legacy Cleanup — COMPLETE ✅

Deleted legacy files (superseded by C++):
- rabitq_baseline.ts
- value_quant.ts
- hybrid_score.ts

## Phase 5: Driver Stack Integration

### 5.1 Custom Mesa/Rusticl Integration

```cpp
// native/adreno/ integration
#include "tq_adreno_driver.h"

struct tq_driver_config {
    const char* driver_path;
    uint32_t device_id;
    uint8_t use_rusticl;
    uint8_t use_turnip;
};

int tq_driver_init(const tq_driver_config* cfg);
int tq_driver_probe(tq_driver_info* info);
```

**Roadmap**:
- [ ] Define driver API in `native/adreno/tq_driver.h`
- [ ] Implement mesa/rusticl probe in `native/adreno/tq_driver.cpp`
- [ ] Add turnip Vulkan fallback
- [ ] Integrate with kernel cache

### 5.2 OpenCL Driver Probe

```cpp
// native/opencl/ integration
int tq_opencl_init(void);
int tq_opencl_get_devices(tq_device_info* devices, uint32_t* count);
cl_context tq_opencl_create_context(int device_index);
```

## Build System

### CMakeLists.txt Update

```cmake
cmake_minimum_required(VERSION 3.18)
project(TurboQuantCore VERSION 4.5.4 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3")

# Core library
add_library(tq_core STATIC
    src/tq_compress.cpp
    src/tq_search.cpp
    src/tq_beta_sphere.cpp
    src/tq_kv_cache.cpp      # NEW
    src/tq_benchmark.cpp       # NEW
    src/tq_oracle.cpp         # NEW
    src/tq_context_pack.cpp   # NEW
    src/tq_format.cpp
    src/tq_base64.cpp
    src/tq_prng.cpp
    src/tq_limits.cpp
    src/tq_quality.cpp
)

target_include_directories(tq_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/native/kernel/include
)

# Kernel library (depends on core)
add_library(tq_kernel STATIC
    ../kernel/src/tq_rotation_kernel.cpp
    ../kernel/src/tq_quantizer_kernel.cpp
    ../kernel/src/tq_qjl_kernel.cpp
)

# CLI tool
add_executable(tq_cli src/tq_cli.cpp)
target_link_libraries(tq_cli tq_core tq_kernel m pthread)

# Tests
add_executable(tq_core_test src/main_test.cpp)
target_link_libraries(tq_core_test tq_core tq_kernel m pthread)
```

## Testing Infrastructure

### 1. Unit Tests (Google Test)

```cpp
// native/core/test/test_compress.cpp
#include <gtest/gtest.h>

TEST(CompressTest, BasicRoundtrip) {
    const uint32_t N = 100, D = 64;
    float vecs[N * D];
    // ... populate

    size_t est = tq_estimate_compressed_size(N, D, 4, 0);
    std::vector<char> out(est * 2);
    size_t out_len = est * 2;

    int r = tq_compress(vecs, N, D, 4, 42, 0, 0, out.data(), &out_len, nullptr);
    EXPECT_EQ(r, TQ_OK);

    uint32_t idx[5];
    float scr[5];
    r = tq_search(out.data(), vecs, D, 5, 0, 0, idx, scr, nullptr);
    EXPECT_EQ(r, TQ_OK);
    EXPECT_EQ(idx[0], 0);  // self-match
    EXPECT_GT(scr[0], 0.9f);
}
```

### 2. Integration Tests

```cpp
// native/core/test/test_integration.cpp
// Full pipeline: compress -> decode -> search -> verify recall
```

### 3. Performance Tests

```cpp
// native/core/test/test_performance.cpp
// Latency benchmarks for encode/decode/search
```

## Rollout Timeline

| Phase | Module | Effort | Priority |
|-------|--------|--------|----------|
| 1.1 | compressed_kv_cache.cpp | 2 days | CRITICAL |
| 1.2 | benchmark.cpp | 1 day | HIGH |
| 1.3 | oracle.cpp | 1 day | HIGH |
| 2.1 | context_pack.cpp | 2 days | MEDIUM |
| 5.1 | driver integration | 3 days | MEDIUM |
| 4.1-4.4 | Legacy cleanup | 0.5 day | LOW |
| 3.1-3.3 | ML Training (defer) | N/A | DEFERRED |

## Success Metrics

1. **Zero TypeScript on hot path**: All compression/search in C++
2. **Benchmark parity**: C++ vs TS results match within 0.1%
3. **Build warnings**: 0 across all modules
4. **Test coverage**: 100% for C++ API surface
5. **Driver stack**: mesa_rusticl_kgsl probe passes

## Notes

- Python scripts in bin/ are orchestration layer, acceptable
- TypeScript MCP tools call C++ library via FFI
- WASM backend can call C++ directly for browser deployment