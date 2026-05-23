# TurboQuant Native Kernel

**Low-level C++ kernels for precise memory register work.**

| | |
|---|---|
| Language | C++17 / C11 |
| Compiler | LLVM Clang 21.1.8 |
| Status | Active development |

---

## Kernels

### 1. FWHT Rotation Engine (`tq_rotation_kernel`)
- Fast Walsh-Hadamard Transform rotation
- O(d log d) complexity
- Sign flip pattern generation
- Batch processing support

### 2. Lloyd-Max Quantizer (`tq_quantizer_kernel`)
- Beta-distributed codebook (TurboQuant)
- Uniform symmetric codebook
- 2/3/4/8 bits per coordinate
- MSE computation

### 3. QJL Residual Correction (`tq_qjl_kernel`)
- Johnson-Lindenstrauss projection
- Unbiased dot-product estimation
- Sketch compression
- Based on Zandieh et al., ICML 2024

---

## Building

```bash
cd native/kernel
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
make -j$(nproc)
```

Output:
- `libtq_kernel.a` — static library
- `test_kernel` — test suite
- `kernel_bench` — benchmark suite

---

## Tests

```bash
./test_kernel

=== TurboQuant Native Kernel Test Suite ===

[TEST] test_rotation_init_shutdown... PASS
[TEST] test_rotation_single_vector... PASS
[TEST] test_rotation_cpp_batch... PASS
[TEST] test_quantizer_init... PASS
[TEST] test_quantizer_encode_decode... PASS
[TEST] test_quantizer_cpp_beta... PASS
[TEST] test_qjl_init_shutdown... PASS
[TEST] test_qjl_compress... PASS
[TEST] test_qjl_cpp_projector... PASS
[TEST] test_qjl_cpp_sketch... PASS
[TEST] test_qjl_cpp_residual_estimator... PASS

=== Results ===
Passed: 11
Failed: 0
```

---

## Benchmarks

```bash
./kernel_bench

=== TurboQuant Native Kernel Benchmark ===
LLVM Clang 21.1.8

[Bench] Rotation 1024x1000
  Total:   ~50000 us
  Per call: 50.00 us
  Throughput: 20000 vecs/sec

[Bench] QJL 512x64 x1000
  Total:   ~30000 us
  Per call: 30.00 us
  Throughput: 33333 comp/sec
```

---

## C++ API

```cpp
#include "tq_rotation_kernel.h"
#include "tq_quantizer_kernel.h"
#include "tq_qjl_kernel.h"

// Rotation
tq::kernel::RotationKernel rot(dims, seed);
rot.rotate(input, output);
rot.rotate_batch(vectors, outputs, batch_size);

// Quantizer
tq::kernel::BetaCodebook cb(dims, bits);
cb.compute();
float q = cb.quantize(value);

// QJL
tq::kernel::QJLResidualEstimator est(orig_dims, target_dims);
auto sketch = est.compress(residual);
float dot = est.estimate_dot(sketch_a, sketch_b);
```

---

## C API

```c
tq_rotation_kernel_t rot;
tq_rotation_init(&rot, 1024, 42, TQ_ROTATION_MODE_FWHT_SIGN);
tq_rotation_execute(&rot, input, output, 1);
tq_rotation_shutdown(&rot);

tq_quantizer_kernel_t q;
tq_quantizer_init(&q, 1024, 4);
tq_quantizer_encode(&q, values, packed, 1024);
tq_quantizer_shutdown(&q);

tq_qjl_kernel_t qjl;
tq_qjl_init(&qjl, 1024, 64, 42, 4, 1);
tq_qjl_compress(&qjl, residual, sketch);
tq_qjl_shutdown(&qjl);
```

---

## Architecture

```
native/kernel/
├── include/
│   ├── tq_rotation_kernel.h   — FWHT rotation
│   ├── tq_quantizer_kernel.h  — Lloyd-Max quantizer
│   └── tq_qjl_kernel.h        — QJL correction
├── src/
│   ├── tq_rotation_kernel.cpp
│   ├── tq_quantizer_kernel.cpp
│   ├── tq_qjl_kernel.cpp
│   ├── test_kernel.cpp        — 11 tests
│   └── kernel_bench.cpp       — benchmarks
└── CMakeLists.txt
```

---

## Next: OpenCL Backend

These kernels will be compiled to SPIR-V and executed on Adreno GPU via Mesa Rusticl/Turnip stack.

See: `native/opencl/kernels/` for OpenCL kernel sources.