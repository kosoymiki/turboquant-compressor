# Runtime Evidence Chain

## Goal

This document defines the canonical evidence chain for release preparation of the repo-local custom stack.

## Chain

### 1. Runtime-Pack Manifest

Files:
- `native/opencl/runtime-pack/meta/manifest.json`
- `native/opencl/runtime-pack/meta/dependencies.txt`

Purpose:
- describe the assembled runtime-pack
- define dependency closure and packed stack composition

### 2. Safe Probe

Files:
- `artifacts/seamless_probe_post_boundary.json`
- `forensics/opencl-adreno-report.json`
- `forensics/adreno/loader-report.json`

Purpose:
- prove that the repo-local stack is discoverable and classified correctly
- prove the selected backend and loader path

### 3. Self-Tests

Execution surface:
- `native/opencl/src/tq_opencl_cli.cpp`

Coverage:
- `mse-score --self-test`
- `qjl-score --self-test`
- `value-dequant --self-test`
- `fused-attention --self-test`

Purpose:
- prove minimal OpenCL context, kernel load, dispatch, and parity paths on the custom stack

### 4. Safe Single-Run Benchmark

Files:
- `artifacts/safe_benchmark_2026-05-20.json`

Purpose:
- prove end-to-end execution and non-zero timing path without full stress loops

### 5. Release Evidence

Files:
- `forensics/RELEASE_EVIDENCE_MANIFEST.json`
- `forensics/mesa/driver-mesh-recovery-ready.json`
- `forensics/mcp-stdio-transcript.jsonl`

Purpose:
- preserve release-grade truth claims
- anchor runtime and packaging state in committed evidence

## Claim Hierarchy

Allowed claims:
- `custom stack discoverable`
- `custom stack self-test pass`
- `custom stack safe single-run pass`
- `custom stack export-ready`

Forbidden claim shortcuts:
- `vendor OpenCL ready` when the route is `mesa_rusticl_kgsl`
- `benchmark-safe` without safe single-run evidence
- `export-ready` without manifest, dependency, and sync proof

## Export Readiness Rule

A build is ready for export only when:
1. runtime-pack manifest exists
2. dependency list exists
3. safe probe succeeds
4. self-tests are real and pass
5. safe single-run benchmark exists
6. mirror parity is proven by `SYNC_MANIFEST`

