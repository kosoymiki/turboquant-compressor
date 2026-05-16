# Donor File-to-File Scan — turboquant-compressor

## Donor: 0xSero/turboquant

Source: `/storage/emulated/0/Download/turboquant-main.zip`

## File Mapping

| Donor File | Local Target | Action | Reason |
|---|---|---|---|
| turboquant/codebook.py | src/core/beta_sphere.ts, src/core/codebook.ts | Port | Beta PDF, Lloyd-Max codebooks |
| turboquant/rotation.py | src/core/rotation.ts | Port | Random orthogonal matrices, QJL |
| turboquant/quantizer.py | src/core/turboquant_mse.ts, src/core/turboquant_prod.ts | Port | LEVEL_1 MSE, LEVEL_2 QJL |
| turboquant/kv_cache.py | src/kv/value_quant.ts, src/kv/compressed_kv_cache.ts | Port | KV value quantization |
| turboquant/store.py | src/kv/compressed_store.ts | Port | Compressed append store |
| turboquant/score.py | src/kv/hybrid_score.ts | Port | Hybrid attention scoring |
| turboquant/triton_kernels.py | kernels/triton/* | Rewrite | GPU kernels |
| turboquant/integration/vllm.py | python/turboquant_ext/integration/vllm.py | Port | vLLM integration |
| turboquant/capture.py | python/turboquant_ext/integration/capture.py | Port | Capture hooks, rollback |
| benchmark.py | test/scientific/benchmark.py | Port | Scientific validation |
| proof.py | test/scientific/proof.py | Port | Scientific proof |

## Codebooks

Donor precomputed codebooks (GPL):
- turboquant/codebooks/codebook_d64_b*.json
- turboquant/codebooks/codebook_d128_b*.json
- turboquant/codebooks/codebook_d576_b*.json

These are included in runtime for LEVEL_1 MSE.

## License

Donor is GPL-3.0-or-later. This project is now GPL-3.0-or-later.

## Attribution

See docs/LICENSE_STRATEGY.md and NOTICE file.
