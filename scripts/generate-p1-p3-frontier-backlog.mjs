#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chooseP1P3CampaignPlan } from './lib/p1-p3-campaign-contract.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, '..');
const docsDir = path.join(rootDir, 'docs');
const forensicsDir = path.join(rootDir, 'forensics');
const matrixPath = path.join(docsDir, 'tz-frontier-capability-matrix-2026-05-21.md');

mkdirSync(forensicsDir, { recursive: true });

const matrix = readFileSync(matrixPath, 'utf8');

const COMPOSITE_KERNEL_MODULE_RULE =
  'kernel frontier composite rows are evaluated as TQ module families; once the covered modules ship with evidence, the aggregate row becomes accepted residual instead of a blocking monolith';

const clusters = [
  {
    id: 'p1_kernel_primitives',
    ranges: ['10', '15', '17', '30', '41-43', '61-83', '95-115'],
    donors: [
      ['khr-integer-dot', 'web-spec', 'https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_integer_dot_product.html'],
      ['khr-subgroup-shuffle', 'web-spec', 'https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_khr_subgroup_shuffle.html'],
      ['khr-subgroup-ballot', 'web-spec', 'https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_khr_subgroup_ballot.html'],
      ['khr-subgroup-nonuniform', 'web-spec', 'https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_khr_subgroup_non_uniform_arithmetic.html'],
      ['khr-coop-matrix', 'web-spec', 'https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_khr_cooperative_matrix.html'],
      ['clblast-github', 'web-github', 'https://github.com/CNugteren/CLBlast'],
      ['flashattention-github', 'web-github', 'https://github.com/Dao-AILab/flash-attention'],
      ['mesa-ir3', 'local-source', '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/drivers/freedreno/ir3'],
      ['tq-kernel-tune', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_kernel_tune.cpp'],
      ['tq-fused-attention', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_fused_attention.cpp'],
      ['tq-fused-attention-tiled', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_fused_attention_tiled.cpp'],
      ['tq-attention-kernels', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/kernels/tq_attention_logits.cl'],
    ],
    acceptedResidualRows: {
      '10': 'subgroup-width tuning heuristic is implemented, but remains a shaping heuristic rather than a standalone release gate',
      '15': 'scan/broadcast are not individually release-evidenced; current shipped subgroup closure already covers the active shuffle/ballot/clustered-reduce route',
      '16': 'cooperative matrix remains spec/draft-gated and is not repo-controlled on the active Rusticl route',
      '18': 'SPIR-V optimizer micro-policy is a lower-level toolchain shaping concern, not a remaining P1 product blocker',
      '30': 'integer-dot support is now evidenced directly and should not keep P1 open',
      '41-43': 'specific subgroup family surfaces are now evidenced directly and should not keep P1 open',
      '61-79': `${COMPOSITE_KERNEL_MODULE_RULE}; this broad optimization bucket is already represented by shipped TQ kernel modules rather than one remaining lane`,
      '80-83': `${COMPOSITE_KERNEL_MODULE_RULE}; subgroup math primitive aspirations extend beyond the currently shipped module set and stay as non-blocking optimization residue`,
      '95-115': `${COMPOSITE_KERNEL_MODULE_RULE}; this roadmap-scale primitive bucket is tracked through concrete TQ modules, not as one blocking aggregate gate`,
    },
  },
  {
    id: 'p1_runtime_memory',
    ranges: ['13', '14', '22', '32', '36', '39', '49-60', '84-90', '158-179'],
    donors: [
      ['khr-async-compilation', 'web-spec', 'https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_Ext.html'],
      ['khr-external-memory', 'web-spec', 'https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_external_memory.html'],
      ['khr-external-semaphore', 'web-spec', 'https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_external_semaphore.html'],
      ['mesa-freedreno-gmem', 'local-source', '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/drivers/freedreno/freedreno_gmem.c'],
      ['mesa-freedreno-autotune', 'local-source', '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/drivers/freedreno/freedreno_autotune.c'],
      ['mesa-freedreno-program', 'local-source', '/data/data/com.termux/files/home/mesa-upstream-26.2-devel/src/gallium/drivers/freedreno/freedreno_program.c'],
      ['tq-buffer', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_buffer.cpp'],
      ['tq-memory', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_memory.cpp'],
      ['tq-driver-pack', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/driver-pack/pack_driver.sh'],
      ['renderdoc-github', 'web-github', 'https://github.com/baldurk/renderdoc'],
    ],
    acceptedResidualRows: {
      '22': 'the remaining gap is Android AHardwareBuffer specifically; generic external-memory zero-copy is already closed and this surface is hardware/runtime-gated',
      '49-55': 'Qualcomm-specific extension family is not honestly closable on the current Mesa/Rusticl route and is treated as hardware-gated residual',
      '56': 'mutable dispatch remains a separate Khronos/runtime frontier outside the closed base command-buffer lane',
      '57-60': 'the only honest unresolved slice here is AHardwareBuffer; the rest of the external-memory/semaphore/SVM family is already evidenced and should not keep P1 open',
      '158-179': 'this broad runtime-management bucket is now carried forward as the first P2 scheduler/allocator frontier, not a remaining P1 gate',
    },
  },
  {
    id: 'p1_tooling_ci',
    ranges: ['33', '36', '84-90', '180-195'],
    donors: [
      ['mesa-envvars', 'web-doc', 'https://docs.mesa3d.org/envvars.html'],
      ['mesa-freedreno-docs', 'web-doc', 'https://docs.mesa3d.org/drivers/freedreno.html'],
      ['perfetto-docs', 'web-doc', 'https://perfetto.dev/docs/'],
      ['renderdoc-android', 'web-doc', 'https://documentation.ubuntu.com/anbox-cloud/howto/android/debug-graphics-renderdoc/'],
      ['tq-benchmark', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_benchmark.cpp'],
      ['tq-gpu-profile', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_gpu_profile.cpp'],
      ['tq-opencl-evidence', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/scripts/generate-opencl-capability-evidence.mjs'],
      ['tq-adb-forensics', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/scripts/collect-adb-mesa-forensics.mjs'],
    ],
    acceptedResidualRows: {
      '180-195': 'this block is mostly watchdog/background-delivery/doc infrastructure and now belongs to the broader P2 runtime-management carry, not a blocking P1 tooling lane',
    },
  },
  {
    id: 'p2_scheduler_allocator',
    ranges: ['121-134', '158-179'],
    donors: [
      ['vllm-github', 'web-github', 'https://github.com/vllm-project/vllm'],
      ['xformers-github', 'web-github', 'https://github.com/facebookresearch/xformers'],
      ['flashattention-github', 'web-github', 'https://github.com/Dao-AILab/flash-attention'],
      ['tq-autotuner', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_autotuner.cpp'],
      ['tq-kernel-cache', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_kernel_cache.cpp'],
      ['tq-kv-analyze', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/src/tools/turboquant_kv_analyze.ts'],
    ],
    acceptedResidualRows: {
      '121-128': 'compile-time SPIR-V validation, model specialization, AOT/JIT selection, constant dedupe, and op-level profiling are already covered by the shipped scheduler/allocator module plus existing SPIR-V/parity/profiling evidence; the residual here is broader optimization ambition, not a blocking repo-controlled gap',
      '129-134': 'research-grade scheduler search families such as Bayesian, bandit, transfer, and RL-DVFS remain accepted residual frontier beyond the shipped runtime_scheduler lane',
      '158-179': 'multi-queue runtime management, cache residency, compile-orchestrator policy, memory budgeting, and queue-pressure control are now closed by the dedicated runtime_scheduler module; remaining ideas in this band are broader systems research residue rather than a blocking repo-controlled frontier',
    },
  },
  {
    id: 'p2_model_inference_runtime',
    ranges: ['199-206'],
    donors: [
      ['huggingface-continuous-batching-arch', 'web-doc', 'https://huggingface.co/docs/transformers/continuous_batching_architecture'],
      ['huggingface-continuous-batching-guide', 'web-doc', 'https://huggingface.co/docs/transformers/main/continuous_batching'],
      ['flashattention-repo', 'web-github', 'https://github.com/Dao-AILab/flash-attention'],
      ['sarathi-paper', 'web-paper', 'https://arxiv.org/abs/2308.16369'],
      ['tq-qjl-score', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_qjl_score.cpp'],
      ['tq-value-dequant', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_opencl_value_dequant.cpp'],
    ],
    acceptedResidualRows: {
      '199-206': 'the broad inference-serving row is decomposed into bounded TQ modules; prefill/decode split, chunked prefill, and continuous batching are now closed by the dedicated p2_model_inference_runtime contract, while speculative decoding remains separate residual frontier',
    },
  },
  {
    id: 'p2_paged_kv_prefix_cache',
    ranges: ['199-206'],
    donors: [
      ['vllm-paged-attention', 'web-github', 'https://github.com/vllm-project/vllm'],
      ['pagedattention-paper', 'web-paper', 'https://arxiv.org/abs/2309.06180'],
      ['hf-cache-explanation', 'web-doc', 'https://huggingface.co/docs/transformers/main/cache_explanation'],
      ['tq-inference-runtime', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_inference_runtime.cpp'],
      ['tq-runtime-scheduler', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/native/opencl/src/tq_runtime_scheduler.cpp'],
    ],
    acceptedResidualRows: {
      '199-206': 'the broad inference-serving row is decomposed into bounded TQ modules; paged-KV allocation, block tables, prefix-cache block sharing, and eviction are now closed by the dedicated p2_paged_kv_prefix_cache contract, while speculative decoding remains separate residual frontier',
    },
  },
  {
    id: 'p2_precision_training_runtime',
    ranges: ['135-157'],
    donors: [
      ['pytorch-quantization-support', 'web-doc', 'https://docs.pytorch.org/docs/stable/quantization-support'],
      ['pytorch-qat-llm-blog', 'web-doc', 'https://docs.pytorch.org/blog/quantization-aware-training/'],
      ['lsq-paper', 'web-paper', 'https://arxiv.org/abs/1902.08153'],
      ['intel-neural-compressor-distillation', 'web-doc', 'https://intel.github.io/neural-compressor/latest/docs/source/distillation_quantization.html'],
      ['tq-precision-calibration', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/src/core/precision_calibration.ts'],
      ['tq-lsq-optimizer', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/src/core/lsq_optimizer.ts'],
      ['tq-distillation-contract', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/src/core/distillation_for_quantization.ts'],
      ['tq-hybrid-training-runtime', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/src/runtime/hybrid_training_runtime.ts'],
      ['tq-full-qat-loop', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/src/core/full_qat_training_loop.ts'],
      ['tq-quantize-tool', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/src/tools/turboquant_quantize.ts'],
    ],
    acceptedResidualRows: {
      '135-157': 'the broad precision/training row is now decomposed into bounded TQ modules; calibration, LSQ step-size optimization, quantization distillation, hybrid CPU/WASM/GPU runtime routing, and a bounded improving QAT loop are closed, while dataset-backed distillation, multi-epoch optimization, live GPU backward, and broader distributed training remain accepted residual frontier rather than a blocking repo-controlled gap',
    },
  },
  {
    id: 'p3_distributed_privacy',
    ranges: ['207-229'],
    donors: [
      ['pysyft-github', 'web-github', 'https://github.com/OpenMined/PySyft'],
      ['flower-github', 'web-github', 'https://github.com/adap/flower'],
      ['opacus-github', 'web-github', 'https://github.com/pytorch/opacus'],
      ['tenseal-github', 'web-github', 'https://github.com/OpenMined/TenSEAL'],
      ['tq-research-index', 'local-source', '/data/data/com.termux/files/home/tmp_turboquant/src/research/index.ts'],
    ],
    acceptedResidualRows: {
      '207-208': 'TrustZone-protected memory and AES-backed key custody are outside the current repo-controlled TurboQuant product surface and remain accepted secure-runtime residual frontier',
      '209-212': 'SPIR-V no-wrap/debug/reflection/vulkan cross-ecosystem surfaces are broader toolchain frontier, not blocking repo-controlled product gaps for the shipped TQ stack',
      '213': 'device fission for legacy GPU sharing is not a practical or repo-controlled frontier on the current mobile KGSL route and remains accepted residual',
      '214-229': 'federated/privacy/distributed-training systems are broader research and product-line frontier outside the current Termux-first TQ runtime; they remain explicit accepted residuals rather than blocking implementation gaps in this repo',
    },
  },
];

const unresolvedLines = matrix
  .split('\n')
  .filter((line) => line.startsWith('|') && /(implemented_partial|spec_only_or_unproven|not_present)/.test(line));

function parseRowId(line) {
  const match = line.match(/^\|\s*([^|]+?)\s*\|/);
  return match ? match[1].trim() : '';
}

function parseNumericSpan(value) {
  if (!value) return null;
  if (value.includes('-')) {
    const [startRaw, endRaw] = value.split('-');
    const start = Number.parseInt(startRaw, 10);
    const end = Number.parseInt(endRaw, 10);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
    return { start: Math.min(start, end), end: Math.max(start, end) };
  }
  const row = Number.parseInt(value, 10);
  if (!Number.isFinite(row)) return null;
  return { start: row, end: row };
}

function spansOverlap(left, right) {
  return left.start <= right.end && right.start <= left.end;
}

function lineMatchesRangeToken(line, token) {
  const rowId = parseRowId(line);
  if (!rowId) return false;
  if (rowId === token) return true;
  const rowSpan = parseNumericSpan(rowId);
  const tokenSpan = parseNumericSpan(token);
  if (!rowSpan || !tokenSpan) return false;
  return spansOverlap(rowSpan, tokenSpan);
}

for (const cluster of clusters) {
  const matching = unresolvedLines.filter((line) =>
    cluster.ranges.some((token) => lineMatchesRangeToken(line, token)),
  );
  const acceptedResidualRows = cluster.acceptedResidualRows || {};
  const blocking = [];
  const acceptedResiduals = [];
  for (const line of matching) {
    const rowId = parseRowId(line);
    const residualEntry = Object.entries(acceptedResidualRows).find(([token]) => lineMatchesRangeToken(line, token));
    if (residualEntry) {
      const [residualToken, residualRationale] = residualEntry;
      acceptedResiduals.push({
        rowId,
        matchedResidualToken: residualToken,
        status:
          line.includes('implemented_partial')
            ? 'implemented_partial'
            : line.includes('spec_only_or_unproven')
              ? 'spec_only_or_unproven'
              : 'not_present',
        rationale: residualRationale,
      });
      continue;
    }
    blocking.push(line);
  }
  cluster.unresolvedCount = blocking.length;
  cluster.specOnlyCount = blocking.filter((line) => line.includes('spec_only_or_unproven')).length;
  cluster.partialCount = blocking.filter((line) => line.includes('implemented_partial')).length;
  cluster.notPresentCount = blocking.filter((line) => line.includes('not_present')).length;
  cluster.acceptedResidualCount = acceptedResiduals.length;
  cluster.acceptedResiduals = acceptedResiduals;
  cluster.moduleClosureRule = cluster.id === 'p1_kernel_primitives' ? COMPOSITE_KERNEL_MODULE_RULE : null;
}

const plan = chooseP1P3CampaignPlan(
  clusters.map((cluster) => ({
    id: cluster.id,
    unresolvedCount: cluster.unresolvedCount,
    specOnlyCount: cluster.specOnlyCount,
    partialCount: cluster.partialCount,
    donorCount: cluster.donors.length,
  })),
);

const payload = {
  generatedAt: new Date().toISOString(),
  sourceMatrix: matrixPath,
  totalDonors: clusters.reduce((sum, cluster) => sum + cluster.donors.length, 0),
  clusters,
  plan,
};

const outPath = path.join(forensicsDir, 'p1-p3-frontier-backlog-2026-05-22.json');
writeFileSync(outPath, JSON.stringify(payload, null, 2));
console.log(JSON.stringify({ outPath, clusters: clusters.length, totalDonors: payload.totalDonors }, null, 2));
