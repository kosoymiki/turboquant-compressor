# P2 Hybrid CPU GPU Training Runtime 2026-05-22

This document defines the next bounded execution module inside `p2_precision_training_runtime`: `hybrid_cpu_gpu_training_runtime`.

The honest claim here is not a full heterogeneous training engine. It is only that repo evidence now contains a real admission and phase-partition contract across three lanes:

- TypeScript CPU control plane
- WASM-assisted CPU math lane
- already-closed OpenCL inference/cache evidence lane

## Scope

- phase partitioning for calibration/control, teacher-student logit math, and inference-style attention execution
- artifact-first route admission using already closed OpenCL runtime contracts
- explicit fallback behavior when WASM or GPU evidence is missing
- canonical hybrid runtime artifact and contract

## Non-Goals

- live GPU backward pass
- multi-device gradient synchronization
- full QAT training loop
- dataset-backed distillation

## Current Closure Evidence

- deterministic artifact written at `forensics/hybrid-cpu-gpu-training-runtime-state.json`
- canonical contract written at `forensics/precision/hybrid-cpu-gpu-training-runtime-contract.json`
- explicit phase routing between CPU, WASM, and GPU evidence planes
- dependence on existing `module_ready` inference and paged-KV contracts instead of a fresh live OpenCL run

## Closure Boundary

This module is considered honestly closed because repo evidence now shows:

- a CPU control plane for training-side logic
- a WASM-assisted math lane for bounded student-side work
- a GPU execution lane admitted from already-closed live OpenCL evidence
- an explicit routing contract between those planes

What is still not claimed:

- live GPU backward/runtime training kernels
- full QAT training loop closure
- dataset-backed distillation
- distributed or multi-device training runtime
