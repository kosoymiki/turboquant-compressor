# Mesa Stack Forensics Module 2026-05-21

This module defines a full-stack no-root forensic profile for the TQ+Mesa frontier.

Corpus rules:

- no minimal/minimum framing
- no repo-surface closure without live artifact
- no vendor-route truth reused as Rusticl-route proof

Forensic layers:

- `tq_loader`
- `tq_probe`
- `rusticl_platform`
- `rusticl_device`
- `rusticl_context`
- `rusticl_memory`
- `rusticl_semaphore`
- `pipe_screen`
- `pipe_resource`
- `pipe_fence`
- `pipe_context`
- `freedreno_import`
- `freedreno_rd`
- `turnip_vk`
- `android_loader`

Activation surface:

- `TQ_OPENCL_TRACE=1`
- `TQ_DRIVER_TRACE=1`
- `TQ_MESA_FORENSICS=1`
- `RUSTICL_DEBUG=memory,forensics`
- `FD_MESA_DEBUG=flush`
- `FD_RD_DUMP=enable,full`
- `FD_RD_DUMP_PATH=/data/local/tmp`
- `TU_DEBUG=flushall,rd`
- `TU_PERFETTO=1`

Module sources:

- `src/native/mesa_stack_forensics.ts`
- `scripts/collect-mesa-stack-forensics.mjs`
- `scripts/collect-offline-mesa-forensics.mjs`
- `scripts/collect-adb-mesa-forensics.mjs`
- `forensics/mesa-stack-forensics-profile.json`
- `forensics/offline-mesa-forensics.json`

Canonical no-adb anchor:

- `forensics/offline-mesa-forensics.json`

Closure targets:

- `initialize_memory`
- `external_memory`
- `external_semaphore`
- `system_svm`
- `command_buffer`
- `loader_namespace`
- `staged_dependency_chain`
- `turnip_submission_trace`
- `freedreno_rd_dump`

Current engineering meaning:

- the module is designed to correlate Android loader namespace failure, ICD routing, Rusticl platform/device gate verdicts, Gallium pipe-context semaphore/fence capabilities, and external-memory import semantics in one profile
- the module now also binds `pipe_screen`, `pipe_resource`, and `pipe_fence` into the same forensic mesh, so offline analysis can trace VM allocation, DMA-BUF import, fence export/wait, and resource-address queries without device access
- the module also reserves canonical env surfaces for Turnip/Freedreno command-stream capture and flush tracing, matching Mesa documentation for `FD_RD_DUMP*`, `FD_MESA_DEBUG`, and `TU_DEBUG`
- this profile is the canonical replay point for future adb passes on fresh devices
- the offline collector closes the no-adb gap by auditing driver-root completeness, local dependency closure, trace coverage, named frontier-lane classifiers, and contract presence without requiring device access
- the adb collector closes the operator gap by staging the full custom userspace, replaying vendor/custom/linker paths, capturing render-node and libclc blockers, and emitting a single endpoint-specific forensic artifact
- the semaphore lane now has a dedicated TQ contract module instead of string-grep-only logic:
  - `src/native/external_semaphore_contract.ts`
  - `scripts/lib/external-semaphore-contract.mjs`
  - `scripts/collect-adb-mesa-forensics.mjs` now emits structured `externalSemaphoreContract { inputs, verdict }`
- route-selection and loader/namespace drift are now first-class modules too:
  - `src/native/route_selection_contract.ts`
  - `scripts/lib/route-selection-contract.mjs`
  - `scripts/collect-adb-mesa-forensics.mjs` now emits structured `loaderNamespaceContract` and `routeSelectionContract`
- kernel-facing residual blockers are now also first-class modules:
  - `src/native/kernel_blocker_contracts.ts`
  - `scripts/lib/kernel-blocker-contracts.mjs`
  - `scripts/collect-adb-mesa-forensics.mjs` now emits structured `renderNodePolicyContract`, `stagedDependencyChainContract`, and `spirvCapabilityContract`
