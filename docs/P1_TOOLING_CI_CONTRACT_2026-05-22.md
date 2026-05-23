## P1 Tooling/CI Contract

This lane is the release-grade profiling and regression surface for the live `mesa_rusticl_kgsl` route.

Artifacts:
- `forensics/adreno/opencl-event-profile.json`
- `forensics/adreno/opencl-profiler-contract.json`
- `forensics/adreno/opencl-performance-baseline.json`
- `forensics/adreno/opencl-performance-regression.json`

Execution:
- `npm run bench:opencl:event-profile`
- `npm run forensics:opencl-profiler-contract`
- `npm run verify:opencl-perf-regression`

Contract:
- Event profiling comes from `tq_opencl_cli benchmark` and its per-kernel event timings.
- Regression gate fails when `mean_ms` degrades by more than `3%` against baseline unless baseline is intentionally updated.
- Intercept/profiler lane is formalized as:
  - OpenCL event profiling in the native benchmark harness
  - optional external OpenCL Intercept Layer integration
  - Mesa/Freedreno/Turnip trace env contract

Truth:
- This lane is valid only when parity remains `true` on the benchmark kernels.
- Baseline updates must be explicit via `verify:opencl-perf-regression --update-baseline`.
