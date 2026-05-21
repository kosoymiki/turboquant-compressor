# Repo Technical Audit 2026-05-21

## Scope

This audit covers the shipped `v4.1.2` surface after kernel/runtime cleanup, the fresh open local benchmark, and the current custom driver runtime contract.

Evidence anchors:
- `bench/results/open-test-local-20260521-095918.json`
- `forensics/opencl-adreno-report.json`
- `forensics/adreno/loader-report.json`
- `forensics/mesa/driver-mesh-recovery-ready.json`
- `native/opencl/driver-pack/tq-driver-pack-adreno-a7xx-a8xx.tar.zst`

## P0

None currently open on the shipped path.

The previous `P0` capability split was closed by deriving kernel specialization from the real OpenCL route rather than inferred Vulkan/GPU profile hints.

## P1

### P1. Retrieval quality remains below the best historical narrower-corpus artifact.

Current shipped benchmark:
- `Recall@1 = 0.34`
- `Recall@5 = 0.66`
- `MRR = 0.4600`

This is materially better than the earlier weak public path, but still below the older narrow-corpus `20260514-233707` artifact (`Recall@5 = 0.92`, `MRR = 0.7047`).

Engineering meaning:
- the shipped public path is now truthful and coherent
- the remaining bottleneck is search quality, not driver bring-up
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
- subgroup-enabled OpenCL route on this stack

So performance language must stay scoped to committed artifacts.

### P2. Some historical evidence lanes still contain stale vendor wording.

Imported artifacts under `forensics/imported/` preserve old donor/runtime states such as `ADRENO_VENDOR_OPENCL_READY`.
They are historical evidence, not current product truth.

## P3

### P3. Repo still contains historical audit and export docs that describe earlier cleanup phases.

These are no longer product blockers, but they remain archival context and should not be treated as the primary release narrative.

## Closed Items

- subgroup/fp16 specialization truth now comes from the actual OpenCL probe path
- dead fused-attention fp16 lanes were removed from the shipped kernel surface
- public retrieval path now ships `hashed_tfidf + turboquant_beta`
- new regression tests cover public beta path defaults and context-pack build provenance
- tracked driver contract is `tar.zst` plus installed `native/opencl/driver-root`

## Current Verdict

- custom driver runtime: ready
- shipped retrieval path: coherent and evidenced
- release documentation: aligned to the fresh artifact
- main remaining engineering frontier: retrieval quality beyond the current committed `Recall@5 = 0.66` path
