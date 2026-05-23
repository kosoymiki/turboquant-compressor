**P2 Model Runtime**
This module closes the first high-end `p2_model_inference_runtime` lane for the current TQ OpenCL stack with bounded live evidence, not with a speculative serving shell.

**Scope**
- `prefill/decode split` over real TQ kernels
- `chunked prefill` with fixed token budgeting
- `continuous batching` with decode-priority mixed waves
- live execution anchors on:
  - `tq_attention_logits + tq_attention_apply_values`
  - `tq_qjl_score + tq_value_dequant`

**Contract**
- native state artifact: [inference-runtime-state.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/inference-runtime-state.json:1)
- canonical collector artifact: [opencl-inference-runtime-contract.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/adreno/opencl-inference-runtime-contract.json:1)
- bounded CLI entrypoint: `inference-runtime-smoke`
- execution model: `process_scoped_isolated_smoke`
- collector policy: `artifact-first`; live smoke is only refreshed explicitly via `--refresh`

**Readiness Rules**
- `prefillDecodeSplitReady`
  - true only if both prefill and decode waves executed
- `chunkedPrefillReady`
  - true only if total prefill chunks exceed request count and chunk size hits the configured ceiling
- `continuousBatchingReady`
  - true only if more than one request coexisted in live waves and decode batching exceeded single-request mode
- `mixedPrefillDecodeReady`
  - true only if at least one mixed wave executed with decode priority enabled

**Current TQ Policy**
- `batchingPolicy = decode_priority_mixed_chunk_continuous_batching`
- `chunkingPolicy = fixed_chunk_prefill_with_decode_interleave`
- `prefillBackend = fused_attention_tiled`
- `decodeBackend = qjl_plus_value_dequant`

**External Donor Basis**
- Hugging Face continuous batching architecture
  - https://huggingface.co/docs/transformers/continuous_batching_architecture
- Hugging Face continuous batching guide
  - https://huggingface.co/docs/transformers/main/continuous_batching
- vLLM performance and chunked prefill docs
  - https://docs.vllm.ai/en/v0.17.1/configuration/optimization/
- PagedAttention paper
  - https://arxiv.org/abs/2309.06180
- SARATHI chunked-prefill scheduling
  - https://arxiv.org/abs/2308.16369

**Engineering Reflection**
- This module is deliberately bounded to the current local kernel set, not to a hypothetical full LLM server.
- It proves that our stack can schedule split prefill/decode work with chunked admission and mixed waves on live OpenCL anchors.
- It is intentionally process-scoped because the current Mesa/Rusticl route can assert during teardown after mixed-wave smoke even when the live artifact is already valid.
- It does not claim speculative decoding or distributed serving.
- `paged_kv_allocator + prefix_cache_block_sharing` no longer live here as an unresolved tail; they are carried by the dedicated `p2_paged_kv_prefix_cache` module and contract.
