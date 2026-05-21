# OpenCL SPIR-V Workflow

This document defines the offline-to-runtime kernel flow for TurboQuant native OpenCL.

## Offline Stage

Run:

```sh
bash scripts/build-opencl-native.sh
```

This produces:

- `native/opencl/build/spirv/*.spv`
- `tq_opencl_cli`
- `test_*_parity`

The SPIR-V target is built by `native/opencl/tools/compile_spirv.sh`.

## Runtime Resolution Order

For every kernel load, TurboQuant uses:

1. `forensics/kernel-cache/*.bin`
2. `native/opencl/build/spirv/*.spv` via `clCreateProgramWithIL`
3. `native/opencl/kernels/*.cl` via `clCreateProgramWithSource`

This keeps cold-start low without losing portability.

## Controls

- `TQ_OPENCL_FORCE_SOURCE=1`
  Forces source-only compilation path.
- `TQ_OPENCL_DISABLE_BINARY_CACHE=1`
  Disables compiled binary reuse.
- `TQ_OPENCL_SPIRV_DIR=/custom/path`
  Overrides SPIR-V module directory.

The CLI exposes:

- `--source`
- `--spirv`

on self-test commands to compare source and SPIR-V behavior explicitly.

## Verification

Run:

```sh
scripts/verify-spirv-parity.sh
```

This writes:

- `forensics/spirv-parity-report.json`

and compares:

- `mse-score`
- `qjl-score`
- `value-dequant`
- `fused-attention`

between source and SPIR-V runtime routes.

## Autotune Coupling

`benchmark --autotune` writes:

- `forensics/autotune-cache.json`

That cache is separate from SPIR-V binaries:

- SPIR-V / binary cache = compilation reuse
- autotune cache = dispatch-parameter reuse

Both are release-facing evidence and should be regenerated intentionally.
