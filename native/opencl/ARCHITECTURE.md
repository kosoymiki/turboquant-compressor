# Native OpenCL Architecture

`native/opencl/` is the executable acceleration surface for TurboQuant local kernels. The shipped contract is:

1. `scripts/build-opencl-native.sh` builds `tq_opencl_cli`, parity tests, and `build/spirv/*.spv`.
2. Kernel loading is deterministic:
   - compiled binary cache from `forensics/kernel-cache/`
   - SPIR-V IL from `native/opencl/build/spirv/`
   - source fallback from `native/opencl/kernels/`
3. Runtime capability truth comes from real OpenCL probe/context:
   - `has_fp16`
   - `has_subgroups`
   - `has_il_program`
   - `has_svm_*`
4. Tuning truth comes from:
   - static kernel profile database in `tq_kernel_tune.cpp`
   - optional runtime overrides from `forensics/autotune-cache.json`
5. Driver/runtime execution truth comes from repo-local driver contract:
   - `native/opencl/driver-pack/tq-driver-pack-adreno-a7xx-a8xx.tar.zst`
   - `native/opencl/driver-root/`
   - `scripts/safe-runtime-pack-run.sh`

## Build Graph

- `tq_opencl_core`: shared native runtime/object library
- `tq_opencl_cli`: probe, self-tests, benchmark
- `test_*_parity`: CPU-reference parity executables
- `tq_spirv_kernels`: offline OpenCL C -> SPIR-V compilation target

## Kernel Compilation

Each kernel is identified by:

- kernel source path
- kernel name
- build options
- device identity hash

Compiled binaries are cached under `forensics/kernel-cache/`. The cache key is derived from:

- device name
- driver version
- compute units
- max work-group size
- kernel source path
- build options

## Capability Guards

All shipped kernels must compile when:

- `TQ_HAS_FP16=0`
- `USE_SUBGROUPS=0`

That is the baseline portability contract for CPU OpenCL / reduced-capability devices.

## Autotune Contract

`benchmark --autotune` explores bounded work-group candidates, records the best configuration, and writes:

- `forensics/autotune-cache.json`

Runtime tune lookup loads this cache on demand and overlays it onto the static profile database.

## SVM Contract

`OpenClContext` probes:

- `CL_DEVICE_SVM_CAPABILITIES`
- `CL_DEVICE_IL_VERSION`

`TqBuffer<T>` chooses:

- SVM for coarse/fine-grain capable devices and large buffers
- explicit `cl_mem` fallback otherwise

Current benchmark/runtime behavior remains correct when SVM is unavailable.
