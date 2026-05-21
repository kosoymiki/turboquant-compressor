# OpenCL Donor-to-File Audit

This audit records which external OpenCL donor patterns were accepted into `native/opencl` and `driver-pack`, where they landed, and what was explicitly rejected.

## Accepted Patterns

`Khronos OpenCL Registry`
- `clCreateProgramWithIL`, `clSetKernelArgSVMPointer`, `clEnqueueSVMMap`, `CL_DEVICE_SVM_CAPABILITIES`
- landed in:
  - [native/opencl/src/tq_opencl_context.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_context.cpp)
  - [native/opencl/src/tq_opencl_kernel_manager.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_kernel_manager.cpp)
  - [native/opencl/src/tq_buffer.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_buffer.cpp)
  - [scripts/verify-spirv-parity.sh](/data/data/com.termux/files/home/tmp_turboquant/scripts/verify-spirv-parity.sh)
- rationale:
  - spec-backed IL loading
  - spec-backed SVM capability detection and binding
  - direct CLI parity path with sourced driver env

`Mesa Rusticl / LLVM SPIR-V toolchain`
- landed in:
  - [native/opencl/CMakeLists.txt](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/CMakeLists.txt)
  - [native/opencl/tools/compile_spirv.sh](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/tools/compile_spirv.sh)
  - [native/opencl/driver-pack/build_mesa.sh](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/driver-pack/build_mesa.sh)
- rationale:
  - offline SPIR-V modules
  - Mesa-facing toolchain contract tightened around `llvm-spirv`, validation, and reproducible kernel build inputs

`PyOpenCL runtime program cache`
- landed in:
  - [native/opencl/include/tq_kernel_cache.h](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/include/tq_kernel_cache.h)
  - [native/opencl/src/tq_kernel_cache.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_kernel_cache.cpp)
  - [native/opencl/src/tq_opencl_kernel_manager.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_kernel_manager.cpp)
- rationale:
  - persistent binary cache keyed by device/runtime identity
  - cold-start reduction without changing math path

`MIOpen perf-db / CLTune style tuning persistence`
- landed in:
  - [native/opencl/include/tq_autotuner.h](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/include/tq_autotuner.h)
  - [native/opencl/src/tq_autotuner.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_autotuner.cpp)
  - [native/opencl/include/tq_kernel_tune.h](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/include/tq_kernel_tune.h)
  - [native/opencl/src/tq_kernel_tune.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_kernel_tune.cpp)
  - [native/opencl/src/tq_opencl_benchmark.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_benchmark.cpp)
- rationale:
  - static tune DB remains initial guess
  - runtime sweep persists tuned values plus timing evidence

`OpenCV UMat / bounded memory abstraction direction`
- landed in:
  - [native/opencl/include/tq_buffer.h](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/include/tq_buffer.h)
  - [native/opencl/src/tq_buffer.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_buffer.cpp)
  - [native/opencl/src/tq_opencl_mse_score.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_mse_score.cpp)
  - [native/opencl/src/tq_opencl_qjl_score.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_qjl_score.cpp)
  - [native/opencl/src/tq_opencl_value_dequant.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_value_dequant.cpp)
  - [native/opencl/src/tq_opencl_fused_attention_tiled.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_fused_attention_tiled.cpp)
- rationale:
  - unified host/device buffer surface
  - optional SVM binding without changing public math APIs

`ProjectPhysX/OpenCL-Wrapper host discipline`
- landed in:
  - [native/opencl/src/tq_opencl_program.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_program.cpp)
  - [native/opencl/src/tq_opencl_kernel_manager.cpp](/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_kernel_manager.cpp)
- rationale:
  - bounded wrapper ideas only
  - no external wrapper dependency imported

## Explicitly Rejected

`Generic topic pages / GitHub topic listings`
- rejected as implementation truth
- kept only as corpus discovery aids

`Full framework transplants`
- rejected:
  - ArrayFire JIT graph
  - OpenCV UMat full runtime surface
  - PlaidML / compiler-stack replacement
- reason:
  - too invasive for `native/opencl`
  - violates TQ math and packaging boundaries

`Vendor-native OpenCL truth`
- rejected for runtime classification
- TQ runtime truth remains:
  - custom `mesa_rusticl_kgsl`
  - repo-local driver pack / driver root contract

## Remaining Boundary

The accepted donor logic is infrastructure-only:
- compile/load path
- caching
- tuning persistence
- memory abstraction

It does not change:
- TurboQuant math
- context-pack format
- corpus compression format
- TypeScript algorithm semantics
