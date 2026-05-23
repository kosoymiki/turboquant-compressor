# Repo Technical Audit 2026-05-21

## Scope

This audit covers the shipped `v4.5.0` surface after kernel/runtime cleanup, the fresh open local benchmark, and the current custom driver runtime contract.

Evidence anchors:
- `bench/results/open-test-local-20260521-160651.json`
- `forensics/opencl-adreno-report.json`
- `forensics/opencl-adb-192.168.50.213-42609-2026-05-21.json`
- `forensics/opencl-adb-192.168.50.54-37921-2026-05-21.json`
- `forensics/offline-mesa-forensics.json`
- `forensics/adreno/loader-report.json`
- `forensics/mesa/driver-mesh-recovery-ready.json`
- `native/opencl/driver-pack/tq-driver-pack-adreno-a7xx-a8xx.tar.zst`
- `docs/frontier-tracing-contract-2026-05-21.md`
- `docs/mesa-single-process-build-forensics-2026-05-21.md`
- `docs/MESA_STACK_FORENSICS_MODULE_2026-05-21.md`
- `docs/MESA_TQ_FORENSIC_AUDIT_2026-05-21.md`
- `docs/RENDERDOC_GPU_DEBUG_CORPUS_2026-05-22.md`
- `forensics/renderdoc-gpu-debug-corpus-ingest.json`

Standing corpus rule:
- no minimal/minimum framing for the TQ+Mesa frontier
- implementation campaigns must prefer maximum closure effort and full forensics before any narrowing of claims
- every active lane must carry an explicit donor model in the corpus, not only the current `system_svm` campaign
- after `P0` closure, remaining `P1-P3` work must also enter through the canonical queue contract in `docs/P1_P3_FRONTIER_QUEUE_2026-05-22.md` and `forensics/p1-p3-frontier-backlog-2026-05-22.json`

New corpus expansion:

- `/storage/emulated/0/Download/renderdoc-gpu-debug-portable.tar.gz` is now ingested as a donor corpus bundle
- archive sha256: `d541ee3f952efeb7c26ab70ef39e75dde03a69735070212083fffa2ef3ca2ac2`
- extracted donor surface: `140` files / `68,604,458` bytes
- exact legacy GPU-debug tool surface recovered: `21` MCP tools + `7` recipes
- donor relevance is now formalized for `mesa_command_buffer` and `mesa_system_svm`, not left as an ad hoc side archive

## P0

None currently open on the shipped path.

The previous `P0` capability split was closed by deriving kernel specialization from the real OpenCL route rather than inferred Vulkan/GPU profile hints.

## P1

### P1. Retrieval quality is now materially stronger on the shipped corpus, but the historical artifact table still spans non-identical corpora and should not be over-compared.

Current shipped benchmark:
- `Recall@1 = 0.88`
- `Recall@5 = 1.00`
- `MRR = 0.94`

This is stronger than both the earlier weak public path and the older narrower-corpus `20260514-233707` artifact on raw retrieval metrics, but those artifacts are not directly identical-corpus comparisons.

Engineering meaning:
- the shipped public path is now truthful and coherent
- the shipped build surface now makes the contextual `hashed_tfidf` route explicit and keeps `plain_hashed_tfidf` as an opt-in weaker control
- the remaining bottleneck is not basic retrieval correctness, but long-horizon retrieval generalization and future QJL-backed public quality claims
- QJL remains research-only until a reproducible estimator/search path exists on the committed corpus

### P1. OpenCL benchmark policy and helper scripts still carry a dual-route world.

This is intentional operationally, but it means the repo still has:
- a primary custom stack contract
- a secondary diagnostic vendor route

That split is now documented honestly and no longer leaks into the main runtime truth.

## P2

### P2. Release evidence is still stronger for safe single-run than for sustained performance.

Current runtime evidence proves:
- safe probe
- self-tests
- safe single-run benchmark

It does not prove:
- sustained thermal behavior
- long-loop throughput stability
- fine-grain/system SVM on this stack

So performance language must stay scoped to committed artifacts.

Current runtime evidence does prove:
- active custom route `mesa_rusticl_kgsl`
- subgroup exposure in the live OpenCL probe
- coarse-grain SVM exposure in the live OpenCL probe

It does not prove:
- fine-grain buffer SVM
- fine-grain system SVM
- SVM atomics

### P2. Some historical evidence lanes still contain stale vendor wording.

Imported artifacts under `forensics/imported/` preserve old donor/runtime states such as `ADRENO_VENDOR_OPENCL_READY`.
They are historical evidence, not current product truth.

## P3

### P3. Repo still contains historical audit and export docs that describe earlier cleanup phases.

These are no longer product blockers, but they remain archival context and should not be treated as the primary release narrative.

## Frontier Forensics

Frontier closure is now explicitly bound to a tracing contract:

- TurboQuant native OpenCL trace scope via `TQ_OPENCL_TRACE=1`
- Rusticl memory/SVM trace scope via `RUSTICL_DEBUG=memory`
- Rusticl semaphore trace scope via `RUSTICL_DEBUG=memory`
- Gallium `pipe_screen` / `pipe_resource` / `pipe_fence` / `pipe_context` trace mesh via `TQ_MESA_FORENSICS=1` plus `RUSTICL_DEBUG=memory,forensics`
- canonical no-adb classifier in `forensics/offline-mesa-forensics.json` for unresolved frontier lanes and next-fix surfaces
- mandatory source-gate to live-artifact mapping for `initialize_memory`, `external_memory`, `external_semaphore`, `system/fine SVM`, and `command_buffer`
- single-process Mesa build cleanup is now bound to a separate forensic contract so build-graph warnings are classified before any new runtime claim or source edit

Fresh cross-device adb evidence now also proves that vendor Qualcomm OpenCL and custom Mesa/Rusticl must stay separated in all release claims:

- on `192.168.50.213:42609` (`taro`, Adreno 730, Android 16), vendor `libOpenCL.so` exposes `svmFine/system/atomics=true`
- the fresh custom `run-as` Rusticl route now proves generic DMA-BUF external-memory import and acquire/release, live initialize-memory contexts, base command-buffer routing, reusable sync-fd external semaphores, and semantic system-SVM smoke with atomics
- TQ now lifts those remaining `false` values into explicit structured verdicts:
  - `initializeMemoryContract=runtime_surface_missing`
  - `commandBufferContract=runtime_surface_missing`
  - `svmContract=ready`
- the previously lower `external_semaphore` blocker is also closed on the active route; the reusable sync-fd semaphore cycle now completes end-to-end and no longer sits behind `kernel_syncobj_missing`
- therefore vendor-route positives cannot be used to zero out unresolved Rusticl frontier tails
- an explicit custom ICD attempt on the same device now proves a separate failure class: once `tq_opencl_cli` is forced onto the staged custom `libOpenCL.so` loader plus Rusticl ICD, platform enumeration fails with `CL_PLATFORM_NOT_FOUND_KHR (-1001)` because the staged Rusticl userspace dependency chain is incomplete; the concrete blockers already observed are `libz.so.1`, `libffi.so`, `libxml2.so.16`, and `libicuuc.so.78`
- a second fresh endpoint on the same `taro` / Adreno 730 device family now proves the next stage after dependency closure: custom staging reaches `platformCount=1`, but device enumeration still fails because:
  - shell context cannot open `/dev/dri/renderD128` despite mode `666`, so the remaining gate is policy/runtime access rather than a simple filesystem mode issue
  - the staged `libRusticlOpenCL.so.1.0.0` was built with a compile-time dynamic libclc lookup path rooted at `/data/data/com.termux/files/usr/share/clc/`, so pushing `spirv-mesa3d-.spv` and `spirv64-mesa3d-.spv` into `/data/local/tmp/tq-rusticl` does not satisfy the active lookup contract
- on the current canonical endpoint this route layer is now normalized inside TQ itself:
  - `loaderNamespaceContract=vendor_abi_mismatch`
  - `routeSelectionContract=custom_preferred_vendor_abi_blocked`
  - engineering meaning: the staged custom route is the only honest executable path for the current CLI/runtime shape, while the vendor path is ABI-broken and must remain diagnostic-only
- the remaining lower kernel/runtime residue is now also explicit inside TQ:
  - `renderNodePolicyContract=kgsl_route_operational`
  - `stagedDependencyChainContract=ready`
  - `spirvCapabilityContract=clean`
  - `commandBufferContract=ready`
  - `initializeMemoryContract=ready`
  - `externalSemaphoreContract=ready`
  - `systemSvmSubstrateContract=ready`
  - engineering meaning:
    - the staged userspace dependency closure is no longer the blocker on the canonical endpoint
    - DRM render-node denial remains visible, but the active Rusticl route is already operational through KGSL and should not be treated as blocked at route-selection level
    - the previous `SpvCapabilityGenericPointer` mismatch has now been closed by aligning Rusticl generic-address-space device exposure with SPIR-V/OpenCL C capability exposure
- source and replay now agree on the remaining lower-route shape:
  - Gallium/Freedreno still probes the DRM render-node route (`pipe_loader_drm_probe`, `drmOpenWithType("msm", ..., DRM_NODE_RENDER)`), but Android Gallium also prefers KGSL first (`pipe_loader_prefer_kgsl()`), and the active custom replay already enumerates one platform and one device through that route
  - the next real implementation campaigns are therefore:
    - Freedreno/KGSL syncobj-backed semaphore substrate
    - only after that, higher-route DRM render-node policy if a direct DRM path is still required
- that new endpoint is anchored in `forensics/opencl-adb-192.168.50.54-37921-2026-05-21.json`
- repo now contains a dedicated staging script for this lane: `scripts/stage-adb-rusticl-runtime.sh`; it assembles the currently known dependency set for replay on a fresh adb endpoint instead of reconstructing the bundle manually

Canonical contract:

- `docs/frontier-tracing-contract-2026-05-21.md`
- `docs/mesa-single-process-build-forensics-2026-05-21.md`

## Closed Items

- subgroup/fp16 specialization truth now comes from the actual OpenCL probe path
- dead fused-attention fp16 lanes were removed from the shipped kernel surface
- public retrieval path now ships `hashed_tfidf + turboquant_beta`
- new regression tests cover public LEVEL_1 path defaults and context-pack build provenance
- tracked driver contract is `tar.zst` plus installed `native/opencl/driver-root`

## Current Verdict

- custom driver runtime: ready
- shipped retrieval path: coherent and evidenced
- release documentation: aligned to the fresh artifact
- main remaining engineering frontier: public QJL-quality uplift and sustained runtime/perf evidence beyond the current committed `Recall@5 = 1.00` contextual lexical path
