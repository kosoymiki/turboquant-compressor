# TZ Frontier Capability Matrix 2026-05-21

Primary corpus anchor:

- `/storage/emulated/0/wrapper/tz.txt`

Cross-check sources used for this matrix:

- TurboQuant repo source under `native/opencl/`, `src/`, `scripts/`, `docs/`
- local Mesa upstream source under `/data/data/com.termux/files/home/mesa-upstream-26.2-devel`
- local runtime forensics under `forensics/`
- fresh vendor-route adb forensic under `forensics/opencl-adb-192.168.50.213-42609-2026-05-21.json`
- official Khronos extension/spec pages

Status vocabulary:

- `implemented_and_evidenced`: code exists and current repo/runtime evidence supports a release claim
- `implemented_partial`: code exists, but evidence is incomplete or runtime capability is narrower than the TZ target
- `spec_only_or_unproven`: extension/spec exists, but repo does not yet justify a shipped claim
- `not_present`: no meaningful implementation found in current repo

Standing corpus rule for this matrix:

- no frontier item is worked under minimal/minimum framing
- closure work must aim at maximum implementation plus full forensic confirmation

## P0 snapshot

| # | TZ task | Status | Engineering verdict |
|---|---|---|---|
| 1 | SPIR-V toolchain in build process | implemented_and_evidenced | `compile_spirv.sh`, CMake target, driver-pack build metadata, and SPIR-V workflow docs are present |
| 2 | Binary cache directory for compiled kernels | implemented_and_evidenced | kernel cache implementation and architecture docs exist |
| 3 | KernelManager with SPIR-V IL support | implemented_and_evidenced | `clCreateProgramWithIL` path is live in kernel manager |
| 4 | Binary Cache -> SPIR-V -> Source fallback | implemented_and_evidenced | documented and reflected in kernel manager/load path |
| 5 | Offline SPIR-V parity validation | implemented_and_evidenced | `scripts/verify-spirv-parity.sh` exists and release evidence expects parity artifact |
| 6 | Async compilation via `cl_khr_async_program_compilation` | spec_only_or_unproven | bounded `async-build-smoke` now proves the wrapper path is safe, but the live route still reports `available=false`, so extension-level async compilation cannot be claimed honestly yet |
| 7 | `#ifdef cl_khr_fp16` guards in kernels | implemented_and_evidenced | `forensics/opencl-capability-evidence.json` now scans all shipped kernels and proves the `TQ_HAS_FP16` gated extension-enable pattern across the kernel set |
| 8 | `#ifdef cl_khr_subgroups` guards in kernels | implemented_and_evidenced | `forensics/opencl-capability-evidence.json` now scans all shipped kernels and proves the `USE_SUBGROUPS` gated subgroup-enable pattern across the kernel set |
| 9 | AutoTuner + persistent PerfDb | implemented_and_evidenced | `forensics/autotune-cache.json` plus `forensics/opencl-capability-evidence.json` now prove a persisted per-kernel tuning cache on live hardware |
| 10 | Half-wave vs full-wave subgroup selection | implemented_partial | repo has subgroup tuning/profile logic; live per-device proof remains limited |
| 11 | suggested local work-group size query | implemented_and_evidenced | runtime probe now exposes `hasSuggestedLocalWorkSize`, native dispatchers query `clGetKernelSuggestedLocalWorkSizeKHR`, and `forensics/opencl-capability-evidence.json` proves the live Rusticl route advertises the surface |
| 12 | `TqBuffer` with auto SVM selection | implemented_and_evidenced | `tq_buffer.cpp` exercises auto-SVM routing and live runtime evidence now proves coarse-grain SVM availability through `opencl-adreno-report.json` and `opencl-capability-evidence.json` |
| 13 | AoS -> SoA memory coalescing | implemented_and_evidenced | the runtime-memory wave now ships explicit layout/vector-width/tile compile macros plus multi-item-per-thread fused value kernels, with fresh parity artifacts on the active route |
| 14 | bank-conflict padding | implemented_and_evidenced | active fused-attention value kernels now use explicit local tiling and tile-shaping macros, and parity stayed clean after the GMEM/tiling wave on live hardware |
| 15 | subgroup shuffle/reduce/scan/broadcast | implemented_partial | subgroup tune enums and kernels exist, but not all operations are release-proven |
| 16 | cooperative matrix operations | spec_only_or_unproven | Khronos surface is still draft/working-track; no shipped repo proof |
| 17 | integer dot product OpenCL 3.1 | implemented_and_evidenced | live probe now reports `hasIntegerDotProduct=true`, shipped kernels carry an actual integer-dot fast path, and current QJL/fused-attention parity artifacts prove the route works on the active Rusticl stack |
| 18 | bypass FMA degradation in SPIR-V optimizer | implemented_partial | TZ flag appears in toolchain guidance, but needs explicit evidence path |
| 19 | atomic cache swap for kernel binaries | implemented_partial | TZ requirement is referenced in docs; code/evidence needs explicit closure audit |
| 20 | GPU IOMMU/SMMU context isolation | spec_only_or_unproven | currently forensic/docs-level only, not a shipped measurable repo feature |
| 21 | `cl_khr_command_buffer` multi-queue recording | implemented_and_evidenced | the active Rusticl route now advertises `cl_khr_command_buffer`, live probe/smoke report `hasCommandBuffer=true`, and the base command-buffer route is proven end-to-end on fresh staged runtime; optional layered capability bits remain zero, but the base Khronos command-buffer lane is honestly live |
| 22 | `cl_khr_external_memory_android_hardware_buffer` zero-copy | implemented_partial | live `run-as` Rusticl route now proves the generic DMA-BUF lane end-to-end: `dmabufAllocated=true`, `importPath=dma_buf+device_list`, `imported=true`, `acquireOk=true`, `releaseOk=true`; remaining gap is specifically Android `AHardwareBuffer`, not the generic external-memory import path |
| 23 | `cl_khr_external_semaphore` OpenCL-Vulkan sync | implemented_and_evidenced | the active Rusticl/KGSL route now completes the reusable sync-fd semaphore cycle end-to-end: create, signal, export, import, and wait all succeed on fresh staged runtime, and TQ evidence marks both surface and live export as true |
| 24 | `cl_arm_protected_memory_allocation` | not_present | no meaningful shipped path found |
| 25 | `cl_qcom_large_buffer` | spec_only_or_unproven | TZ target exists, but repo proof is absent |
| 26 | `cl_qcom_perf_hint` DVFS switching | spec_only_or_unproven | thermal/perf scripts exist, but no confirmed extension-backed implementation claim |
| 27 | `cl_qcom_ml_ops` full integration | spec_only_or_unproven | no release-grade integration found |
| 28 | IFPC backport | spec_only_or_unproven | docs mention it, but shipped implementation/proof is absent |
| 29 | `cl_khr_async_program_compilation` activation | spec_only_or_unproven | same as #6: the wrapper path is present, but the live route still does not advertise the extension as available |
| 30 | `cl_khr_integer_dot_product` full support | implemented_and_evidenced | the live route advertises `cl_khr_integer_dot_product`, probe JSON surfaces `hasIntegerDotProduct=true`, and the current kernels/self-tests exercise the real path instead of a capability-only declaration |
| 31 | `cl_khr_cooperative_matrix` final support | spec_only_or_unproven | official status is not mature enough for current shipped claim |
| 32 | precompiled binary kernel library for Adreno families | implemented_and_evidenced | the runtime-memory wave exports binaries into `forensics/precompiled-kernel-library`, and fresh fused-attention stderr traces show repeated `precompiled_binary_hit` reuse on the live route |
| 33 | CI/CD performance regression suite | implemented_and_evidenced | canonical runtime-pack benchmark artifacts now exist under `forensics/adreno/opencl-performance-report.json`, release evidence validation consumes them, and `opencl-capability-evidence.json` no longer overcounts probe-only artifacts as perf proof |
| 34 | kernel MD5/SHA256 cache validation | implemented_and_evidenced | kernel-cache artifacts now carry SHA256 evidence in `forensics/opencl-capability-evidence.json`, and native `verify_sha256()` is no longer a stub |
| 35 | device fission for priority isolation | not_present | no meaningful implementation found |
| 36 | temperature monitoring + adaptive throttling | implemented_and_evidenced | the controlled sustained lane now formalizes thermal policy and bounded cooldown orchestration as a shipped host-side runtime policy; this is intentionally not a driver DVFS claim |
| 37 | `cl_khr_spirv_queries` runtime validation | spec_only_or_unproven | official target exists; repo proof absent |
| 38 | SPIR-V 1.6 extended capabilities | spec_only_or_unproven | no release-grade proof for cooperative matrix / integer dot product capability path |
| 39 | `cl_khr_priority_hints` | implemented_and_evidenced | queue-creation code consumes the property on the active route and live probe/evidence report `hasPriorityHints: true`; the shipped claim is surface support, not guaranteed scheduler behavior |
| 40 | `cl_khr_priority_hint` multi-task isolation | spec_only_or_unproven | same category as #39 |

## P1 expanded matrix

| Range | TZ tasks covered | Status | Engineering verdict |
|---|---|---|---|
| 41-43 | `cl_khr_subgroup_ballot`, `cl_khr_subgroup_shuffle`, `cl_khr_subgroup_clustered_reduce` | implemented_and_evidenced | live probe now surfaces each subgroup family explicitly, shipped kernels are subgroup-gated, and current parity/self-test lanes run on the active Rusticl route with these subgroup surfaces enabled |
| 44 | `cl_khr_il_program` | implemented_and_evidenced | current OpenCL path already uses IL ingestion via `clCreateProgramWithIL`; this is one of the strongest closed items in the whole TZ |
| 45-48 | `cl_khr_create_command_queue`, `cl_khr_initialize_memory`, `cl_khr_suggested_local_work_size`, `cl_khr_device_uuid` | implemented_and_evidenced | live evidence now shows `create_command_queue=true`, `initialize_memory=true`, `suggested_lws=true`, and `device_uuid=true`; the active Rusticl context path preserves `CL_CONTEXT_MEMORY_INITIALIZE_KHR` and TQ runtime smoke confirms the initialize-memory context remains live |
| 49-55 | `cl_qcom_*` image/subgroup/dot/fp16/tiled/recordable queues | spec_only_or_unproven | no honest shipped claim is available here; current repo contains driver-pack aspirations and docs, but no release-grade Qualcomm-extension proof on the current Mesa/Rusticl stack |
| 56 | `cl_khr_command_buffer_mutable_dispatch` | spec_only_or_unproven | official Khronos extension exists, but there is still no live release-proven mutable-dispatch path on the current Rusticl route |
| 57-60 | external memory, shared semaphores, fine-grained SVM, zero-copy `AHardwareBuffer` interop | implemented_partial | the live route now proves generic external-memory import plus acquire/release on DMA-BUF, reusable external semaphores through sync-fd, and full fine-grain/system SVM with atomics through isolated semantic smoke; zero-copy `AHardwareBuffer` remains the honest unresolved slice |
| 61-79 | constant memory, texture memory, vectorization, async copy, tiling, register pressure, mixed precision, unroll, capability guards, alignment, prefetch, ILP | implemented_partial | repo clearly contains kernel-tuning, subgroup/profile logic, vectorized layouts, tuning abstractions, and Adreno-specific heuristics, but most of these remain engineering patterns rather than individually evidenced release promises |
| 80-83 | subgroup layernorm / RMSNorm / GELU / SiLU primitives | implemented_partial | kernels and subgroup plumbing are present, but these should still be described as conditional kernel-path work, not as fully release-proven hardware primitives |
| 84-90 | pipeline parallelism, multithreaded queue processing, adaptive flush, profiling/intercept/debug layers | implemented_and_evidenced | the safe tooling wave now ships bounded event profiling, regression gating, profiler contracts, controlled intercept lanes, and a preload-safe intercept smoke artifact as consolidated release evidence |
| 91-93 | OpenCL ICD support, `cl_khr_pci_bus_info`, `cl_khr_extended_versioning` | spec_only_or_unproven | current product is a single-route Termux-first MCP/runtime package, not a demonstrated multi-ICD distribution with portable bus/version query claims |
| 94 | QNN / Qualcomm AI Engine Direct integration | spec_only_or_unproven | no release-grade QNN bridge is present |
| 95-115 | tiled GEMM, conv variants, top-k sort, scan/reduce/histogram/gather, RNG, dropout, decoding/sampling, sub-byte packing, constant memory weights, warp tiling | implemented_partial | some pieces overlap existing kernels and retrieval/inference primitives, but most of this range remains roadmap-level optimization rather than individually closed shipped functionality |
| 116-119 | out-of-order queues, Intel USM, terminate context, SPIR-V linker | spec_only_or_unproven | official/spec targets exist, but current repo/runtime does not justify these as release truths on Adreno Rusticl |

## P2 expanded matrix

| Range | TZ tasks covered | Status | Engineering verdict |
|---|---|---|---|
| 120 | `cl_khr_spirv_vulkan` experimental support | spec_only_or_unproven | explicitly experimental in TZ and not present as a release-backed route in repo |
| 121-128 | joint-wavefront vectorization, compile-time SPIR-V validation, model-specialized/AOT/JIT compilation, constant dedup, op-level profiling | implemented_partial | repo already has offline SPIR-V, parity validation, and some profiling surfaces, but this entire block remains only partially materialized against TZ ambition |
| 129-134 | auto-scheduler, evolutionary search, Bayesian optimization, bandits, transfer learning, RL DVFS | spec_only_or_unproven | current autotuner is materially simpler than this research frontier; no honest claim beyond baseline autotuning |
| 135-157 | adaptive precision, mixed-precision search, on-device distillation, zero-shot/data-free quantization, LSQ/DQ/QAT noise, low-precision training, USM model, hybrid CPU+GPU | implemented_partial | bounded precision_calibration_runtime, gradient_verified_lsq_optimizer, distillation_for_quantization_contract, hybrid_cpu_gpu_training_runtime, and full_qat_training_loop artifacts now exist with observer/range-capture, explicit step-size contract, numerically checked LSQ-style update behavior, a deterministic teacher-student quantized distillation objective, artifact-first CPU/WASM/GPU phase partitioning, and a composed improving QAT loop, but the broader training lane remains unresolved |
| 158-179 | multi-queue pipeline, user/direct submission, speculative compilation, persistent cache, preemption, memory defrag/allocators/accounting/compression, per-kernel register mgmt, sparse allocation | implemented_partial | some cache and allocator ideas exist, and retry/fallback logic appears in product shape, but the majority of this range is not individually shipped or release-proven |
| 180-195 | watchdog, suspend/resume, checkpointing, OOM retry, background compilation, Android delivery, A/B testing, deferred context init, cache invalidation, graph analysis, profiling, docs | implemented_partial | docs and some supporting scripts exist, especially architecture/profiling/release evidence; however watchdog, checkpointing, app-bundle delivery, and distributed profiling are not fully implemented claims |
| 196-198 | `ARCHITECTURE.md`, roofline docs, SPIR-V compile guide | implemented_and_evidenced | this documentation slice is real and present in repo today |
| 199-206 | dynamic batching, continuous batching, chunked prefill, prefill/decode split, speculative decoding, flash attention, paged attention | implemented_partial | this broad serving row is no longer one blocking claim in the shipped TurboQuant/OpenCL path: batching, chunked prefill, prefill/decode split, and paged-KV/prefix-cache closure now ship as bounded `TQ` modules with evidence, while speculative decoding remains the honest residual frontier |

## P3 expanded matrix

| Range | TZ tasks covered | Status | Engineering verdict |
|---|---|---|---|
| 207-208 | TrustZone protected memory, AES fallback for key protection | not_present | there is still no shipped TrustZone-backed implementation or corpus-backed secure-key subsystem in the current repo/runtime |
| 209-212 | `cl_khr_spirv_no_integer_wrap_decoration`, debug info, reflection, `cl_khr_spirv_vulkan` | spec_only_or_unproven | this is SPIR-V ecosystem frontier, not current release truth |
| 213 | device fission for legacy GPU sharing | not_present | same verdict class as P0 #35 |
| 214-229 | federated learning, DP/privacy, FP8/MX training, ZeRO/FSDP, device clusters, secure aggregation, HE | not_present | no shipped distributed-training runtime, secure-aggregation engine, or HE subsystem is present in the current repo/runtime |

## Cross-range engineering evidence

Current repo evidence that justifies `implemented_partial` rather than `not_present` in many P1/P2 rows:

- `native/opencl/src/tq_opencl_probe.cpp` and `src/native/opencl_probe.ts` now detect subgroup, IL-program, queue-property, UUID, external-memory, external-semaphore, and SVM capability tiers, while keeping command-buffer truth strictly probe-driven
- historical adb capture `forensics/opencl-adb-192.168.50.213-42609-2026-05-21.json` proved the probe could distinguish a vendor Qualcomm OpenCL route from the custom Rusticl route; the current canonical custom route has since advanced materially beyond that older snapshot
- the same fresh adb capture also proves that the staged custom Rusticl route on this endpoint currently fails before platform enumeration with `CL_PLATFORM_NOT_FOUND_KHR (-1001)` once the custom ICD loader is actually forced; root cause was narrowed from generic unavailability to a concrete staged dependency chain (`libz.so.1 -> libffi.so -> libxml2.so.16 -> libicuuc.so.78`)
- `src/native/opencl_probe.ts` now exports a canonical trace contract for unresolved lanes: `initialize_memory`, `command_buffer`, `external_memory`, `external_semaphore`, `system_svm`
- TQ now also exports structured lane contracts instead of raw booleans alone:
  - `initializeMemoryContract=runtime_surface_missing`
  - `commandBufferContract=runtime_surface_missing`
  - `svmContract=ready`
  - `externalSemaphoreContract=kernel_syncobj_missing`
- TQ now also exports route/drift contracts instead of hand-written replay interpretation:
  - `loaderNamespaceContract=vendor_abi_mismatch`
  - `routeSelectionContract=custom_preferred_vendor_abi_blocked`
- TQ now also exports lower kernel/runtime residue contracts:
  - `renderNodePolicyContract=kgsl_route_operational`
  - `stagedDependencyChainContract=ready`
  - `spirvCapabilityContract=clean`
- this materially changes the live lane assessment again: the earlier `GenericPointer` mismatch is closed, and the higher DRM render-node denial is no longer treated as the primary active-route blocker because the custom Rusticl route already enumerates `platformCount=1` and `deviceCount=1` through KGSL; the primary remaining live frontier therefore sits at the syncobj-backed semaphore substrate rather than SPIR-V drift or route selection
- `native/opencl/src/tq_buffer.cpp` already exercises coarse-grain SVM allocation and kernel-arg wiring
- `native/opencl/src/tq_kernel_tune.cpp`, `native/opencl/include/tq_kernel_tune.h`, and driver-pack manifests encode subgroup/tuning intent
- `native/opencl/src/tq_cl_vk_interop.cpp` and `native/opencl/include/tq_cl_vk_interop.h` provide concrete DMA-BUF and sync-FD import surfaces, and local Rusticl source now mirrors that with real generic external-memory import and acquire/release entrypoints even though release-grade runtime proof is still absent
- `docs/frontier-tracing-contract-2026-05-21.md` now pins every unresolved frontier lane to concrete trace scopes and closure conditions
- local Mesa Rusticl source under `src/gallium/frontends/rusticl/` was also patched to expose `cl_khr_throttle_hints` in the same capability family as queue priorities and to parse throttle queue properties
- local Mesa Rusticl source now emits dedicated `external_memory` and `external_semaphore` forensic traces under `RUSTICL_DEBUG=memory`
- fresh adb capture `forensics/opencl-adb-192.168.50.54-37921-2026-05-21.json` now additionally proves that the unresolved semaphore blocker on the custom route is not a staging mismatch: the staged `libRusticlOpenCL.so.1.0.0` checksum matches local build output, and Freedreno forensic lines show `has_syncobj=0`, `fence_signal=0`, `semaphore_create=0`
- `native/opencl/ARCHITECTURE.md`, `docs/mesa-rusticl-vm-audit-2026-05-21.md`, `docs/runtime-pack-drift-forensic-2026-05-21.md`, and the retrieval/open-test artifacts prove this repo is beyond concept stage

## Khronos / official corroboration used

- OpenCL extension registry: https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_Ext.html
- `cl_khr_command_buffer`: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_command_buffer.html
- `cl_khr_command_buffer_mutable_dispatch`: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_command_buffer_mutable_dispatch.html
- `cl_khr_external_memory`: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_external_memory.html
- `cl_khr_external_semaphore`: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_external_semaphore.html
- `clGetKernelSuggestedLocalWorkSizeKHR`: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetKernelSuggestedLocalWorkSizeKHR.html
- `cl_khr_integer_dot_product`: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_integer_dot_product.html
- Rusticl docs: https://docs.mesa3d.org/rusticl.html
- Freedreno docs: https://docs.mesa3d.org/drivers/freedreno.html

## P1-P3 hard conclusion

The full frontier expansion confirms the same structural rule seen in P0:

- ratified Khronos extension != shipped runtime capability on our live route
- repo scaffolding != release-grade evidence
- Vulkan/Freedreno capability != OpenCL/Rusticl release claim
- research roadmap != honest current product surface

Therefore the only safe patch policy for `v4.5.0` is:

1. upgrade items from `implemented_partial` to `implemented_and_evidenced` only where repo plus runtime proof already exist or can be added concretely
2. narrow or remove public claims for `spec_only_or_unproven`
3. leave `not_present` outside release claims entirely

## Hard engineering conclusion

The current repo is already materially beyond a bring-up prototype on:

- SPIR-V toolchain
- kernel cache
- IL ingestion
- source/SPIR-V parity
- runtime-pack custom stack probing
- coarse-grain SVM detection

But the TZ document overstates what can be marked zero today.

The main reasons are structural:

- several requested items are official specs without current runtime proof on the live Adreno/Rusticl stack
- several items exist only as docs/scaffolding, not release-grade evidence
- several Qualcomm- or Khronos-extension targets depend on device/runtime support that is not yet demonstrated by current forensics

## Immediate next step

Do not mark all P0-P3 as closed.

Instead:

1. close the local release packaging path
2. expand this matrix from P0-only snapshot to the full TZ set
3. convert every `implemented_partial` item into either `implemented_and_evidenced` or a narrowed honest release claim
4. explicitly separate `repo has code` from `live runtime proves capability`
