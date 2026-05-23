# Frontier Tracing Contract 2026-05-21

Primary corpus anchors:

- `/storage/emulated/0/wrapper/tz.txt`
- `docs/tz-frontier-capability-matrix-2026-05-21.md`
- `docs/mesa-rusticl-vm-audit-2026-05-21.md`
- `forensics/opencl-capability-evidence.json`
- `docs/MESA_STACK_FORENSICS_MODULE_2026-05-21.md`

## Rule

No frontier lane is treated as closed from repo surface alone.

No frontier lane is pursued with "minimal", "minimum", or reduced-scope framing.
For this corpus, the standing policy is maximum implementation effort plus full forensic closure.

For `P0-P3` driver/runtime work, each claim must map to all three:

- source gate
- trace scope
- live artifact

The canonical no-root stack forensic profile is now:

- `src/native/mesa_stack_forensics.ts`
- `scripts/collect-mesa-stack-forensics.mjs`
- `scripts/collect-offline-mesa-forensics.mjs`
- `scripts/collect-adb-mesa-forensics.mjs`
- `forensics/mesa-stack-forensics-profile.json`
- `forensics/offline-mesa-forensics.json`

Canonical structured lane classifiers inside TQ:

- `src/native/external_semaphore_contract.ts`
- `src/native/frontier_contracts.ts`
- `scripts/lib/external-semaphore-contract.mjs`
- `scripts/lib/frontier-contracts.mjs`

For fresh device replay, the canonical adb orchestrator is:

- `scripts/collect-adb-mesa-forensics.mjs`

## Mandatory trace scopes

### TurboQuant native OpenCL

Environment:

- `TQ_OPENCL_TRACE=1`

Source:

- `native/opencl/include/tq_trace.h`
- `native/opencl/src/tq_opencl_context.cpp`

Required signals:

- extension gates for `initialize_memory`, `command_buffer`, `external_memory`, `external_semaphore`
- context-create retry path when `CL_CONTEXT_MEMORY_INITIALIZE_KHR` is requested
- final capability snapshot for the active OpenCL route

### Rusticl memory / interop

Environment:

- `RUSTICL_DEBUG=memory`

Source:

- `src/gallium/frontends/rusticl/api/memory.rs`
- `src/gallium/frontends/rusticl/core/context.rs`
- `src/gallium/frontends/rusticl/core/device.rs`
- `src/gallium/frontends/rusticl/api/device.rs`
- `src/gallium/frontends/rusticl/api/platform.rs`

Required signals:

- external-memory property parse begin
- accepted device handle list
- selected external handle type
- imported buffer/image path selection
- acquire/release command acceptance

### Rusticl semaphore / sync

Environment:

- `RUSTICL_DEBUG=memory`

Source:

- `src/gallium/frontends/rusticl/api/semaphore.rs`
- `src/gallium/frontends/rusticl/core/device.rs`
- `src/gallium/frontends/rusticl/core/platform.rs`

Required signals:

- semaphore create begin / accepted
- accepted device handle list
- selected import/export sync-fd handle path
- signal / wait acceptance
- get-handle / reimport acceptance
- platform/device gate verdict for external semaphore exposure

### Gallium pipe screen / resource / fence / context

Environment:

- `TQ_MESA_FORENSICS=1`
- `RUSTICL_DEBUG=memory,forensics`

Source:

- `src/gallium/frontends/rusticl/mesa/pipe/screen.rs`
- `src/gallium/frontends/rusticl/mesa/pipe/resource.rs`
- `src/gallium/frontends/rusticl/mesa/pipe/fence.rs`
- `src/gallium/frontends/rusticl/mesa/pipe/context.rs`

Required signals:

- context-create begin/result
- VM alloc begin/result
- `resource_assign_vma` begin/result
- DMA-BUF import begin/result
- resource GPU address query begin/result
- fence export/wait/signal path
- fence fd import/create gate verdict

### Rusticl SVM / VM gate

Environment:

- `RUSTICL_DEBUG=memory`

Source:

- `src/gallium/frontends/rusticl/core/platform.rs`
- `src/gallium/frontends/rusticl/core/device.rs`

Required signals:

- `system_svm_supported()`
- VM hook triad presence: `alloc_vm`, `free_vm`, `resource_assign_vma`
- VMA window

## Frontier-specific closure conditions

### `initialize_memory`

Must have:

- source path for `CL_CONTEXT_MEMORY_INITIALIZE_KHR`
- trace showing requested properties
- trace showing either successful create or explicit fallback
- live artifact proving exposure on the active stack

Current classified verdict on the active route:

- `runtime_surface_missing`

### `external_memory`

Must have:

- source path for `clCreateBufferWithProperties` / `clCreateImageWithProperties`
- device/platform import-handle queries
- acquire/release commands
- trace of parsed properties and import path
- live artifact proving exposure on the active stack

### `external_semaphore`

Must have:

- source path for create / wait / signal / import / export
- gate truth for `create_fence_fd` and `fence_get_fd`
- trace of the exact gate verdict for the active device
- traceable live exposure on the active stack

### `system/fine SVM`

Must have:

- source truth for `system_svm_supported()`
- traceable VM-hook and platform-VM state
- live artifact proving fine/system flags, not just coarse-grain SVM

Current classified verdict on the active route:

- `coarse_only`

### `command_buffer`

Must have:

- real Rusticl subsystem, not helper surface
- command creation / record / finalize / enqueue path in source
- live capability query and runtime artifact
- native probe trace showing the lane is absent on the active route until such subsystem exists

Current classified verdict on the active route:

- `runtime_surface_missing`

## Current frontier map

### Already source-backed

- `external_memory`
  - Rusticl source now has generic external-memory property parsing, import constructors, platform/device queries, and acquire/release entrypoints
  - live proof still pending
- `external_semaphore`
  - Rusticl already has real semaphore create/wait/signal/import/export source paths
  - current unresolved gate is runtime exposure through `create_fence_fd` and `fence_get_fd`
- coarse-grain SVM
  - source and live proof already agree

### Still blocked at source/runtime gap

- `initialize_memory`
  - repo/native side traces the request and fallback
  - Rusticl-side extension exposure and proof path are still missing
- `system/fine SVM`
  - source gate remains `system_svm_supported()`
  - current live route is coarse-only
- `command_buffer`
  - no real Rusticl subsystem yet
  - TQ classifier now records this as `runtime_surface_missing` rather than raw `false`

## Hard forensic policy

- No item moves to `implemented_and_evidenced` without a live artifact on the active route.
- No experimental planner, helper, or stub is counted toward frontier closure.
- No benchmark matrix is required before source-gate tracing is in place.
