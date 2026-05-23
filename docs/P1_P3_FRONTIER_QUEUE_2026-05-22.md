# P1-P3 Frontier Queue 2026-05-22

This document is the canonical backlog contract for the post-`P0` frontier.

Generator:

```sh
cd /data/data/com.termux/files/home/tmp_turboquant
node scripts/generate-p1-p3-frontier-backlog.mjs
```

Artifact:

- [p1-p3-frontier-backlog-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/p1-p3-frontier-backlog-2026-05-22.json:1)

## Rule

No `P1-P3` campaign starts as free-form exploration.

Every cluster must carry:

- unresolved matrix mapping
- donor set
- execution order
- separation between `implemented_partial`, `spec_only_or_unproven`, and `not_present`

Composite kernel rows are not blocking monoliths in this repo.

We assemble kernel frontier closure as `TQ` modules. When a broad matrix range covers multiple kernel primitives or optimization families, the blocking unit is the concrete shipped module plus evidence, not the aggregate range row by itself.

That means:

- composite kernel rows stay useful as corpus coverage anchors
- closure is evaluated against shipped `TQ` kernel modules and current evidence
- once the covered module family ships with evidence, the aggregate row moves to `accepted residual` instead of holding `P1` open as a fake monolith
- hardware-gated or draft-gated kernel surfaces remain explicit residuals and are not silently counted as closed

## Queue

1. `p1_kernel_primitives`
   Kernel acceleration lane: integer dot product, subgroup op families, mixed precision, fusion, and hardware-shaped math primitives.
2. `p1_runtime_memory`
   Runtime/memory lane: GMEM-aware tiling, AoS->SoA closure, precompiled binaries, queue/runtime surfaces, and driver-facing memory layout wins.
3. `p1_tooling_ci`
   Tooling/regression lane: intercept/profiling surfaces, roofline/perf capture, sustained benchmark discipline, and performance CI gates.
4. `p2_scheduler_allocator`
   Scheduler/allocator lane: speculative compilation, persistent caching, memory accounting/defrag/compression, and runtime management systems.
5. `p2_model_inference_runtime`
   Inference-runtime lane: dynamic/continuous batching, chunked prefill, and prefill/decode split on live TQ kernels.
6. `p2_paged_kv_prefix_cache`
   KV/prefix-cache lane: paged-KV allocation, block tables, prefix-cache block sharing, and cache-aware admission/eviction.
7. `p2_precision_training_runtime`
   Precision/training lane: adaptive precision, mixed-precision search, QAT/fake-quant/LSQ calibration policy, on-device distillation, low-precision training, and USM/hybrid runtime research.
8. `p3_distributed_privacy`
   Distributed/privacy/training lane: federated learning, secure aggregation, differential privacy, ZeRO/FSDP, and HE/privacy systems.

## Engineering Meaning

This queue is intentionally front-loaded toward `P1` because:

- it offers the highest immediate acceleration return
- it is closest to the already-closed OpenCL-core frontier
- it has the strongest donor overlap with local Mesa/Freedreno/TQ code

`P2` and `P3` remain mandatory, but they should not preempt unresolved `P1` kernel/runtime wins.

The split between `p2_model_inference_runtime`, `p2_paged_kv_prefix_cache`, and `p2_precision_training_runtime` is intentional:

- `199-206` is a repo-controllable inference scheduler/runtime frontier, but it is still too broad to treat as one honest closure claim.
- batching/chunking/prefill-decode orchestration is already isolated as one bounded `TQ` module.
- paged-KV allocation plus prefix-cache block sharing is the next bounded module on the same inference path.
- `135-157` remains a separate precision/training research lane and must not be laundered back into the immediate inference-runtime queue.
- the first honest execution module inside that lane is `precision_calibration_runtime`, not full low-precision training or distillation.

Current corpus truth for the `199-206` serving row is also intentional:

- `p2_model_inference_runtime` is closed as its own bounded module.
- `p2_paged_kv_prefix_cache` is closed as its own bounded module.
- the aggregate `199-206` row therefore no longer keeps either module open.
- the honest residual on this frontier is `speculative_decoding`, not batching/chunking/paged-KV again.

Current corpus truth for the `135-157` precision/training row is also intentional:

- `p2_precision_training_runtime` is now closed as a cluster for progression purposes.
- the first bounded module inside that row, `precision_calibration_runtime`, is now closed as its own artifact/runtime contract.
- the next bounded modules inside that row are `gradient_verified_lsq_optimizer` and `distillation_for_quantization_contract`.
- the next bounded runtime modules after those are `hybrid_cpu_gpu_training_runtime` and `full_qat_training_loop`.
- the aggregate `135-157` row therefore no longer keeps `p2_precision_training_runtime` open.
- honest residual frontier remains: `dataset_backed_distillation`, `multi_epoch_qat_optimization`, `live_gpu_backward_path`, and broader distributed training/runtime carry.

Current corpus truth for the remaining `P2/P3` clusters is also intentional:

- `p2_scheduler_allocator` is now closed for progression purposes because the shipped `runtime_scheduler` module already covers the repo-controlled scheduler/allocator frontier.
- the remaining `121-134` and `158-179` residue is broader optimization and research ambition, not a blocking repo-controlled implementation gap.
- `p3_distributed_privacy` is also closed for progression purposes.
- its remaining scope is explicit accepted residual frontier: TrustZone custody, federated/privacy/distributed-training systems, HE, and adjacent SPIR-V/toolchain carry outside the current shipped TQ runtime surface.

`P1` is considered closed for progression purposes when no repo-controlled blocking rows remain after:

- subtracting accepted residuals
- applying composite-kernel `TQ module` closure semantics
- separating hardware-gated, draft-gated, and broader research/runtime carry from actionable repo-controlled implementation gaps

After the latest canonical backlog regeneration, that condition is met: all `p1_*` clusters have `unresolved=0`, and the current active execution queue is empty.
