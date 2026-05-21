# OpenCL Donor Integration 2026-05-21

This note records which donor patterns were materialized into TurboQuant and which were rejected after corpus and runtime review.

## Corpus Intake

Primary intake artifacts:

- `/data/data/com.termux/files/home/corpus-control-plane/artifacts/opencl_ecosystem_inventory.json`
- `/data/data/com.termux/files/home/corpus-control-plane/artifacts/opencl_ecosystem_group_summary.json`
- `/data/data/com.termux/files/home/corpus-control-plane/reports/opencl_ecosystem_import_report.md`
- `/data/data/com.termux/files/home/corpus-control-plane/artifacts/rusticl_opencl_donor_map.json`
- `/data/data/com.termux/files/home/corpus-control-plane/raw/forensics/android_evidence_pack_20260515`

Imported surface:

- `44` sources total
- `41` succeeded
- failures were mostly upstream `403` surfaces, not local import defects

## Patterns Accepted

- `Rusticl + Khronos + LLVM SPIR-V`
  - accepted as authority for `SPIR-V` toolchain and runtime contract
  - reflected in `native/opencl/tools/compile_spirv.sh`
  - reflected in `native/opencl/driver-pack/build_mesa.sh`

- `PyOpenCL + darktable`
  - accepted for persistent on-disk program binary caching
  - reflected in `native/opencl/include/tq_kernel_cache.h`
  - reflected in `native/opencl/src/tq_kernel_cache.cpp`
  - reflected in `forensics/kernel-cache/`

- `MIOpen`
  - accepted for perf-db style persistence of tuned parameters
  - reflected in `forensics/autotune-cache.json`
  - reflected in `native/opencl/src/tq_opencl_benchmark.cpp`
  - reflected in `native/opencl/src/tq_kernel_tune.cpp`

- `ArrayFire`
  - accepted only at the infrastructure layer
  - the useful transfer is reducing repeated compile overhead, not rewriting TurboQuant around a new AST/JIT stack
  - reflected in `SPIR-V -> binary cache -> source fallback`

- `OpenCV UMat`
  - accepted as a memory-abstraction pattern
  - reflected in `native/opencl/include/tq_buffer.h`
  - reflected in `native/opencl/src/tq_buffer.cpp`

- `ProjectPhysX/OpenCL-Wrapper`
  - accepted as a reminder to keep host-side OpenCL entrypoints clean and bounded
  - rejected as a direct architectural base because TurboQuant already needs repo-aware paths, forensics, parity evidence, and custom driver truth

## Patterns Rejected

- generic GitHub topic pages
  - useful for discovery only
  - not valid runtime truth

- packaging-only donor repos
  - useful for build flags and Android stubs
  - not valid compute correctness truth

- heavyweight donor framework architecture
  - `ArrayFire`, `OpenVINO`, `PlaidML`, `GROMACS`, `LuxCoreRender` contain strong ideas but are not directly transplantable into TurboQuant without replacing the product itself

## Resulting Changes In TurboQuant

- `native/opencl` now has a formal `SPIR-V` build path
- kernel loading now prefers `binary cache -> IL -> source`
- runtime capability truth now exposes `hasIlProgram` and `hasSvm*`
- per-kernel tuning now uses runtime kernel metadata when available
- Mesa build metadata now records `llvm-spirv`, `spirv-link`, `spirv-val`, and Rusticl runtime env assumptions
- file-based infrastructure test added for kernel cache roundtrip

## Resulting Changes In Mesa Lane

- `native/opencl/driver-pack/build_mesa.sh` now treats `SPIR-V` translator tools as explicit build prerequisites
- build metadata now captures the Rusticl toolchain contract instead of leaving it implicit in the host shell

## Non-Claims

- donor intake does not prove those donor implementations are correct for Adreno/Rusticl/KGSL
- donor intake does not override the old working evidence pack
- donor intake does not justify enabling subgroup or fp16 routes when live runtime truth says otherwise
