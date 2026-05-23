const CLUSTER_ORDER = {
  p1_kernel_primitives: 1,
  p1_runtime_memory: 2,
  p1_tooling_ci: 3,
  p2_scheduler_allocator: 4,
  p2_model_inference_runtime: 5,
  p2_paged_kv_prefix_cache: 6,
  p2_precision_training_runtime: 7,
  p3_distributed_privacy: 8,
};

const CLUSTER_SUMMARY = {
  p1_kernel_primitives:
    'Kernel-level acceleration lane: integer dot product, subgroup op families, mixed precision, fusion, and hardware-shaped math primitives.',
  p1_runtime_memory:
    'Runtime and memory lane: GMEM-aware tiling, AoS->SoA closure, precompiled binaries, queue/runtime surfaces, and driver-facing memory layout wins.',
  p1_tooling_ci:
    'Tooling and regression lane: intercept/profiling surfaces, roofline/perf capture, sustained benchmark discipline, and performance CI gates.',
  p2_scheduler_allocator:
    'Scheduler/allocator lane: speculative compilation, persistent caching, memory accounting/defrag/compression, and runtime management systems.',
  p2_model_inference_runtime:
    'Inference-runtime lane: dynamic/continuous batching, chunked prefill, and prefill/decode split on live TQ kernels.',
  p2_paged_kv_prefix_cache:
    'KV/prefix-cache lane: paged-KV allocation, block tables, prefix-cache block sharing, and cache-aware admission/eviction.',
  p2_precision_training_runtime:
    'Precision/training lane: adaptive precision, mixed-precision search, QAT/fake-quant/LSQ calibration policy, on-device distillation, low-precision training, and USM/hybrid runtime research. Closed bounded modules: precision_calibration_runtime, gradient_verified_lsq_optimizer, distillation_for_quantization_contract, hybrid_cpu_gpu_training_runtime, and full_qat_training_loop.',
  p3_distributed_privacy:
    'Distributed/privacy/training lane: federated learning, secure aggregation, differential privacy, ZeRO/FSDP, and HE/privacy systems.',
};

export function chooseP1P3CampaignPlan(inputs) {
  const queue = inputs
    .filter((entry) => entry.unresolvedCount > 0)
    .map((entry) => ({
      id: entry.id,
      priority: CLUSTER_ORDER[entry.id],
      active: true,
      summary: `${CLUSTER_SUMMARY[entry.id]} Unresolved=${entry.unresolvedCount}, partial=${entry.partialCount}, spec_only=${entry.specOnlyCount}, donors=${entry.donorCount}.`,
    }))
    .sort((a, b) => a.priority - b.priority);

  return {
    queue,
    rationale:
      queue.length === 0
        ? 'No unresolved P1-P3 clusters remain in the current matrix.'
        : 'Order clusters by engineering leverage first: P1 kernel/runtime/tooling, then P2 scheduler/runtime systems, then P3 distributed/privacy systems.',
  };
}
