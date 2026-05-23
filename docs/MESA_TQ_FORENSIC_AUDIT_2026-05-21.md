# Mesa + TQ Forensic Audit 2026-05-21

Scope:

- `/data/data/com.termux/files/home/tmp_turboquant`
- `/data/data/com.termux/files/home/mesa-upstream-26.2-devel`

Primary corpus anchors:

- `docs/frontier-tracing-contract-2026-05-21.md`
- `docs/MESA_STACK_FORENSICS_MODULE_2026-05-21.md`
- `docs/RENDERDOC_GPU_DEBUG_CORPUS_2026-05-22.md`
- `forensics/opencl-capability-evidence.json`
- `forensics/opencl-adb-192.168.50.213-42609-2026-05-21.json`
- `forensics/opencl-adb-192.168.50.54-37921-2026-05-21.json`
- `/storage/emulated/0/wrapper/tz.txt`

## Donor Corpus Expansion

A new donor bundle is now ingested into the local TQ+Mesa corpus:

- archive: `/storage/emulated/0/Download/renderdoc-gpu-debug-portable.tar.gz`
- canonical ingest artifact: `forensics/renderdoc-gpu-debug-corpus-ingest.json`
- canonical memo: `docs/RENDERDOC_GPU_DEBUG_CORPUS_2026-05-22.md`

Hard facts from the ingest:

- archive sha256: `d541ee3f952efeb7c26ab70ef39e75dde03a69735070212083fffa2ef3ca2ac2`
- extracted files: `140`
- extracted bytes: `68,604,458`
- bundled MCP tool entries: `21`
- bundled recipe entries: `7`
- curated source-index entries: `53`
- extracted book corpus: `91` total / `53` GPU-relevant

Engineering meaning:

- this bundle becomes a formal donor layer for `mesa_command_buffer`, `mesa_system_svm`, GPU sync, KGSL, Vulkan synchronization, and OpenCL command/queue semantics
- it does not outrank Khronos specs, kernel docs, Mesa source, or live device evidence

## Findings

### P0. Forensic coverage was fragmented across route boundaries

Before this pass:

- TQ loader had local tracing
- Rusticl memory/semaphore had local tracing
- vendor/custom loader split existed in evidence
- Turnip/Freedreno env surfaces were not part of one canonical replay profile

Engineering meaning:

- route ambiguity was still too easy
- fresh-device replay still depended on manual operator memory

Closure in this pass:

- `src/native/mesa_stack_forensics.ts`
- `scripts/collect-mesa-stack-forensics.mjs`
- `forensics/mesa-stack-forensics-profile.json`
- `docs/MESA_STACK_FORENSICS_MODULE_2026-05-21.md`

### P0. Rusticl platform/device/context traces were not normalized as a single forensic layer

Before this pass:

- memory/semaphore traces existed
- platform VM/SVM and device capability gate traces were partly ad hoc

Closure in this pass:

- `src/gallium/frontends/rusticl/core/forensics.rs`
- `src/gallium/frontends/rusticl/core/platform.rs`
- `src/gallium/frontends/rusticl/core/device.rs`
- `src/gallium/frontends/rusticl/core/context.rs`
- `src/gallium/frontends/rusticl/api/memory.rs`
- `src/gallium/frontends/rusticl/api/semaphore.rs`
- `src/gallium/frontends/rusticl/mesa/pipe/context.rs`

### P1. Turnip/Freedreno capture existed in Mesa, but was not contract-bound in the TQ forensic profile

Current closure:

- `FD_MESA_DEBUG`
- `FD_RD_DUMP`
- `FD_RD_DUMP_PATH`
- `TU_DEBUG`
- `TU_PERFETTO`

These are now explicitly encoded in the canonical stack forensic profile and contract.

### P1. Fresh-device custom Rusticl bring-up failure was previously too easy to misclassify as generic OpenCL unavailability

Current closure:

- fresh adb forensic now distinguishes:
  - vendor Qualcomm OpenCL route
  - staged custom loader route
  - staged Rusticl userspace dependency-chain failure on one endpoint
  - staged Rusticl live bring-up reaching `platformCount=1` but failing on render-node access plus compile-time libclc path mismatch on a second endpoint

This is anchored in:

- `forensics/opencl-adb-192.168.50.213-42609-2026-05-21.json`
- `forensics/opencl-adb-192.168.50.54-37921-2026-05-21.json`
- `scripts/stage-adb-rusticl-runtime.sh`

## Coverage status after this pass

Covered layers:

- TQ loader/probe/runtime env
- Android loader namespace / staged userspace boundary
- Rusticl platform/device/context/memory/semaphore
- Gallium `pipe_screen` / `pipe_resource` / `pipe_fence` / `pipe_context` layer
- canonical Turnip/Freedreno debug env profile
- offline frontier-lane classifier with named next-fix surfaces

Still requiring live replay:

- actual `FD_RD_DUMP` artifact on device
- actual `TU_DEBUG/TU_PERFETTO` artifact on device
- post-staging Rusticl device enumeration after resolving render-node access and libclc path closure
- runtime evidence that ties Freedreno/Turnip traces back to the same replay session as TQ/Rusticl traces

Fresh live closure from the current `192.168.50.54:37921` replay:

- the replay now runs on the fresh staged `libRusticlOpenCL.so.1.0.0` by checksum, so current verdicts are not deployment drift
- `external_memory` is fully live on the generic DMA-BUF lane
- `external_semaphore` is now closed on the active route: reusable sync-fd semaphore create, signal, export, import, and wait all succeed on fresh staged runtime
- engineering meaning: the earlier lower syncobj blocker is no longer active on the canonical route; the semaphore lane moved from kernel-class residue to implemented live surface
- TQ now materializes that verdict through a dedicated contract classifier instead of ad-hoc stderr matching:
  - `src/native/external_semaphore_contract.ts`
  - `scripts/lib/external-semaphore-contract.mjs`
  - `forensics/opencl-adb-192-168-50-54-37921-2026-05-21.json` carries `customRouteAttempt.externalSemaphoreContract`
- the same modular path now exists for the other active frontier tails:
  - `initializeMemoryContract.classification=runtime_surface_missing`
  - `commandBufferContract.classification=runtime_surface_missing`
- `svmContract.classification=ready`
  - source module: `src/native/frontier_contracts.ts`
  - script mirror: `scripts/lib/frontier-contracts.mjs`
- route selection and loader drift are now also materialized as explicit TQ contracts:
  - `loaderNamespaceContract.classification=vendor_abi_mismatch`
  - `routeSelectionContract.classification=custom_preferred_vendor_abi_blocked`
  - source module: `src/native/route_selection_contract.ts`
  - script mirror: `scripts/lib/route-selection-contract.mjs`
- the kernel-facing residual layer is now classified too:
  - `renderNodePolicyContract.classification=kgsl_route_operational`
  - `stagedDependencyChainContract.classification=ready`
  - `spirvCapabilityContract.classification=clean`
  - source module: `src/native/kernel_blocker_contracts.ts`
  - script mirror: `scripts/lib/kernel-blocker-contracts.mjs`
- the important correction from the latest live replay is that the old `GenericPointer` mismatch has now been closed in Rusticl device exposure:
  - `CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT` now follows the same device contract as SPIR-V/OpenCL C exposure
  - `__opencl_c_generic_address_space` and `SpvCapabilityGenericPointer` are now surfaced together from `core/device.rs`
  - the active replay no longer emits a live `GenericPointer` capability warning, so this lane is no longer the primary frontier blocker
- the next correction from the same replay is that `/dev/dri/renderD128` denial is no longer treated as the primary route blocker:
  - Android Gallium already prefers KGSL first via `pipe_loader_prefer_kgsl()`
  - the active `run-as` custom route still reaches `platformCount=1` and `deviceCount=1`
  - therefore render-node denial remains a higher-route policy fact, but not an active-route blocker for the current Rusticl/KGSL stack

## Hard conclusion

The forensic system is now materially closer to a stack-wide module than to a collection of lane-specific debug prints.

What is closed:

- source-level forensic coverage model
- canonical env profile
- corpus-bound replay contract
- separation of vendor route and custom route truth

What is not yet closed:

- live end-to-end capture from a fresh device across every Mesa layer in one session

That remaining step is operational, not conceptual: it depends on adb availability and replay, not on missing forensic scaffolding in the repo.

Updated campaign ordering after the `GenericPointer` closure:

- primary: moved beyond `mesa_external_semaphore_syncobj`; current remaining frontier lives outside the previously open OpenCL-core semaphore/SVM lanes
- secondary: none
