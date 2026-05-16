# TurboQuant Compressor v3.3.0

KV cache compression engine. FWHT rotation + Lloyd-Max Beta codebook quantization.
MCP server for Claude Code / Black Diamond ecosystem.

## Architecture

```
Input KV vectors
    │
    ├─ Keys: normalize → FWHT sign-flip rotation → Beta Lloyd-Max quantize (1-6 bit)
    │        Store: packed indices + norms
    │
    └─ Values: group quantization (2/3/4 bit, min-max per group)
               Store: packed data + scales + zeros

Attention: online softmax + sparse V threshold + tiled processing
```

## Theory

After random rotation (FWHT + sign flip), each coordinate of a unit-norm vector follows:

```
f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
```

This is a scaled Beta distribution on [-1, 1]. Lloyd-Max algorithm finds optimal
quantization levels minimizing MSE under this PDF. Convergence guaranteed (convex cost).

References:
- QJL: 1-Bit Quantized JL Transform (Zandieh et al., 2024, arXiv:2406.03482)
- KIVI: Tuning-Free Asymmetric 2bit Quantization (Liu et al., 2024, arXiv:2402.02750)
- KVLinC: Hadamard Rotation + Linear Correction (Saxena & Roy, 2025, arXiv:2510.05373)
- PolyKV: Shared Asymmetrically-Compressed KV Cache (Patel & Joshi, 2026, arXiv:2604.24971)
- Online Normalizer Calculation for Softmax (Milakov & Gimelshein, 2018, arXiv:1805.02867)

## Performance

| Config | Baseline | Compressed | Savings | Ratio |
|--------|----------|------------|---------|-------|
| k4v2 seq4k (8B model) | 512 MB | 144 MB | 71.9% | 3.56x |
| k2v2 seq4k | 512 MB | 113 MB | 78.0% | 4.54x |
| k4v2 seq16k | 2048 MB | 540 MB | 73.6% | 3.79x |
| k3v2 seq8k | 1024 MB | 244 MB | 76.1% | 4.19x |

MSE per coordinate (d=128, 4-bit): 7.28e-05 (<0.01% error).

Validated by PolyKV (2026): 2.91x compression, +0.57% PPL degradation on Llama-3-8B.

## Donor Optimizations (from llama-cpp-turboquant)

1. **Precomputed scaled centroids** — eliminates per-element multiply in scoring
2. **Sparse V threshold** — skip value dequant for low-contribution tokens (default 0.001)
3. **q8 query quantization** — reduced bandwidth in dot product scoring
4. **Tiled token processing** — process in blocks of 64 for cache locality
5. **Block size 128** — donor default, better than 32 for modern hardware
6. **Mixed KV types** — different bits for keys and values

## MCP Tools (13)

| Tool | Function |
|------|----------|
| turboquant_compress | Compress vectors (FWHT + codebook) |
| turboquant_vector_search | Nearest neighbor search on compressed data |
| turboquant_quantize | Codebook computation, vector quantization, MSE benchmark |
| turboquant_kv_analyze | KV cache savings estimation and config recommendation |
| turboquant_cost_analyze | Token usage and cache efficiency analysis |
| turboquant_cache_plan | Context block volatility classification |
| turboquant_prompt_cache_lint | Cache-busting pattern detection |
| turboquant_context_pack_build | Compressed context pack builder |
| turboquant_context_pack_search | Search within context packs |
| turboquant_cli_mcp_profile | MCP client profiles for hosts |
| turboquant_backend_probe | Backend detection (Termux-safe) |
| turboquant_opencl_probe | OpenCL/Adreno device detection |
| turboquant_adreno_loader_probe | Android linker namespace diagnostics |

## Installation

```bash
npm install
npm run build
npm run smoke:stdio  # verify
```

## Usage

### MCP Server (Claude Code)

In `.claude/settings.json`:
```json
{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["path/to/dist/server.js"]
    }
  }
}
```

### Programmatic

```typescript
import { compressVectors, searchVectors } from 'turboquant-compressor';

const result = compressVectors({
  vectors: [[1.0, 0.5, 0.3, 0.1], [0.2, 0.8, 0.1, 0.4]],
  bitsPerValue: 4,
  rotationSeed: 42
});

const matches = searchVectors({
  compressed_database_b64: result.compressed_database_b64,
  query_vector: [1.0, 0.5, 0.3, 0.1],
  top_k: 5,
  metric: 'cosine'
});
```

### KV Cache (CompressedKVCache)

```typescript
import { CompressedKVCache } from 'turboquant-compressor/core';

const cache = new CompressedKVCache({
  headDim: 128,
  keyBits: 4,
  valueBits: 2,
  sparseThreshold: 0.001,
  useQ8Query: true
});

cache.prefill(keys, values, numTokens);
cache.append(newKey, newValue);
const output = cache.computeAttention(query, cache.getKeyQuantized()!, cache.getValueQuantized()!);
```

## Native Kernels (OpenCL / Adreno)

```
native/opencl/kernels/
├── tq_fused_attention_v2.cl    — Full fused attention (LUT + sparse V + q8)
└── tq_fused_attention_tiled.cl — Tiled variant (TILE_K=32)
```

Target: Adreno 730v3 via Rusticl (Mesa/KGSL). Requires OpenCL 3.0+.

Limitations:
- fp32 only (fp16 path planned for 2x throughput)
- dim ≤ 256, key_bits ≤ 6
- No vectorized loads yet (vload4/vload8 planned)

## Known Limitations

1. **Outlier sensitivity**: min-max group quantization for values. Single outlier stretches range.
   Mitigation: use larger groupSize or 4-bit values for outlier-heavy layers.
2. **Sparse V positional bias**: early tokens less likely to be skipped than late tokens with same score.
3. **Beta PDF assumption**: codebook optimized for uniform-on-sphere. Real KV vectors have structure.
   Empirically validated by PolyKV — degradation minimal (+0.57% PPL).
4. **QJL module**: research-only, not paper-faithful. Not wired into public tools.

## Testing

```bash
npm test                    # 237 tests, 33 suites
npm run smoke:stdio         # API + MCP + numeric smoke
npm run smoke:api           # Public API contract
npm run verify:release      # Release readiness check
```

## Black Diamond Integration

Bidirectional bridge via `bd-turboquant-bridge` MCP server:
- BD → TQ: compress, kv_analyze, quantize, context_pack, cost_analyze
- TQ → BD: ecosystem_status, skill_query, guard_check

Projects remain separate. Bridge = protocol, not merge.

## License

GPL-3.0-or-later
