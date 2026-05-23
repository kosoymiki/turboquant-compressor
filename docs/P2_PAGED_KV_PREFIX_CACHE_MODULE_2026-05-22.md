# P2 Paged-KV / Prefix-Cache Module 2026-05-22

This document defines the dedicated bounded `P2` module after `p2_model_inference_runtime`: `paged_kv_allocator + prefix_cache_block_sharing`.

It is intentionally separated from the broader precision/training lane. No part of this module is justified by returning to `training residue`.

## Scope

- paged-KV allocation for decode-time cache growth
- fixed-size KV block table and logical-token to physical-block mapping
- prefix-cache block sharing across requests with identical validated prefixes
- cache-aware admission, pinning, refcounting, and eviction rules
- integration boundary with the already-shipped:
  - `prefill/decode split`
  - `chunked prefill`
  - `continuous batching`

## Closure Rules

This module is only considered honestly closed when repo evidence shows all of the following:

- `pagedKvAllocatorReady`
  - true only if live state records block-granular KV allocation rather than monolithic per-request growth
- `blockTableReady`
  - true only if logical token positions resolve through a stable block-table structure
- `prefixCacheSharingReady`
  - true only if at least two requests reuse one or more validated prefix blocks without duplicate materialization
- `cacheEvictionReady`
  - true only if bounded pressure forces an explicit eviction or compaction decision that preserves live-request correctness

## Current Closure State

- native state artifact: `forensics/paged-kv-prefix-cache-state.json`
- canonical collector artifact: `forensics/adreno/opencl-paged-kv-prefix-cache-contract.json`
- bounded CLI entrypoint: `paged-kv-prefix-cache-smoke`
- execution model: `process_scoped_isolated_smoke`
- collector policy: `artifact-first`; live smoke is only refreshed explicitly via `--refresh`

The current bounded implementation is intentionally tied to the already-validated live OpenCL anchors:
- prefill anchor: `tq_attention_logits + tq_attention_apply_values`
- decode anchor: `tq_qjl_score + tq_value_dequant`

The paged-KV plane is therefore not a disconnected planner. It proves block-granular allocation, prefix-block reuse, refcounted residency, and explicit eviction decisions while the same live TQ kernels execute.

## Canonical Evidence Shape

The closure artifact for this module should be a dedicated runtime contract, separate from the current mixed-wave runtime smoke:

- native state artifact: `forensics/paged-kv-prefix-cache-state.json`
- canonical collector artifact: `forensics/adreno/opencl-paged-kv-prefix-cache-contract.json`
- bounded CLI entrypoint: `paged-kv-prefix-cache-smoke`
- execution model: `process_scoped_isolated_smoke`

## Why This Is The Next Module

- `p2_model_inference_runtime` already closed the scheduling side of the inference lane without overstating cache topology.
- `paged_kv_allocator` and `prefix_cache_block_sharing` are the next honest repo-controlled increment on the same inference path.
- They remain inference-serving primitives, not precision/training work.
- Treating them as a separate bounded module prevents `199-206` from collapsing into one vague overclaim.

## External Donor Basis

- vLLM PagedAttention design and implementation
  - https://github.com/vllm-project/vllm
- PagedAttention paper
  - https://arxiv.org/abs/2309.06180
- Hugging Face caching concepts for LLM inference
  - https://huggingface.co/docs/transformers/main/cache_explanation

## Non-Goals

- speculative decoding
- distributed serving
- training-time KV or optimizer-state systems
- broad `135-157` precision/training residue
