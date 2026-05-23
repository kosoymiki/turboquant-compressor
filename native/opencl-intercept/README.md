# TurboQuant OpenCL-Intercept-Layer

**Native C++ library for OpenCL API interception and profiling.**

| | |
|---|---|
| Version | 1.0.0 |
| Language | C++17 / C11 |
| Compiler | LLVM Clang 21.1.8 |
| Status | Foundation for C++ migration |

---

## Purpose

Intercepts OpenCL API calls for:

- **Memory register tracing** — track all buffer/image allocations, reads, writes
- **Kernel execution profiling** — monitor enqueue/start/complete times
- **Call graph tracking** — API call flow analysis
- **Performance bottleneck identification** — find slow operations

Prepares infrastructure for full TurboQuant C++ migration, enabling precise memory register work without TypeScript limitations.

---

## Architecture

```
native/opencl-intercept/
├── include/
│   └── tq_intercept.h          — Public C/C++ API
├── src/
│   ├── tq_intercept_core.cpp   — Core intercept engine
│   ├── tq_intercept_api.cpp    — OpenCL API hook
│   ├── tq_intercept_memory.cpp — Memory tracking
│   ├── tq_intercept_kernel.cpp — Kernel profiling
│   ├── tq_intercept_profiler.cpp — Performance metrics
│   └── intercept_cli.cpp       — CLI tool
├── test/
│   └── test_intercept.cpp      — Test suite
└── CMakeLists.txt
```

---

## Building

```bash
cd native/opencl-intercept
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
make -j$(nproc)
```

Output:
- `libtq_intercept.so` — shared library
- `intercept_cli` — CLI tool
- `test_intercept` — test suite

---

## API

### C API

```c
// Initialize intercept layer
tq_intercept_config_t config = {0};
config.flags = TQ_INTERCEPT_FLAG_THREAD_SAFE | TQ_INTERCEPT_FLAG_PROFILE_PERF;
tq_intercept_init(&config);

// Get metrics
tq_perf_metrics_t metrics;
tq_intercept_get_metrics(&metrics);

// Shutdown
tq_intercept_shutdown();
```

### C++ API

```cpp
#include "tq_intercept.h"

tq::intercept::ConfigBuilder builder;
builder.enable_profile_perf()
       .enable_log_memory()
       .enable_trace_calls();

auto config = builder.build();
tq_intercept_init(&config);
```

---

## Configuration Flags

| Flag | Purpose |
|------|---------|
| `TRACE_CALLS` | Log all OpenCL API calls |
| `TRACE_ARGS` | Log API call arguments |
| `TRACE_RETURNS` | Log API return values |
| `LOG_MEMORY` | Track memory allocations |
| `LOG_KERNELS` | Track kernel executions |
| `PROFILE_PERF` | Collect performance metrics |
| `CHAIN_LAYERS` | Enable layer chaining |
| `THREAD_SAFE` | Thread-safe operation |
| `BUFFER_RECYCLING` | Reuse buffer pools |

---

## CLI Tool

```bash
# Version
./intercept_cli --version

# Initialize with tracing
./intercept_cli --init --trace --perf --log /tmp/tq_intercept.log

# Show metrics
./intercept_cli --metrics

# Reset metrics
./intercept_cli --reset

# Flush logs
./intercept_cli --flush

# Dump tracking
./intercept_cli --dump-mem
./intercept_cli --dump-kernel
```

---

## Tests

```bash
LD_LIBRARY_PATH=. ./test_intercept

=== TurboQuant OpenCL-Intercept-Layer Test Suite ===

[TEST] test_version_string...  (version: 1.0.0) PASS
[TEST] test_init_shutdown... PASS
[TEST] test_metrics... PASS
[TEST] test_context_tracking... PASS
[TEST] test_memory_alloc_tracking... PASS
[TEST] test_layer_chain... PASS
[TEST] test_flush_logs... PASS

=== Results ===
Passed: 7
Failed: 0
Total:  7
```

---

## C++ Migration Path

This library provides the foundation for the full TurboQuant C++ migration:

1. **Phase 1** (current): OpenCL-Intercept-Layer core
2. **Phase 2**: Native kernel implementations
3. **Phase 3**: Memory register precision
4. **Phase 4**: Full vector compression pipeline

Enables:
- Precise memory register work
- No TypeScript limitations
- GPU memory optimization
- Custom kernel tuning