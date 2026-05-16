# TurboQuant Compressor v3.4.0

KV cache compression engine. FWHT rotation + Lloyd-Max Beta codebook + QJL residual sketch.
MCP server for Claude Code / Black Diamond ecosystem. WASM SIMD128 accelerated.

## Architecture

```
Input KV vectors
    │
    ├─ Keys: normalize → FWHT sign-flip rotation → Beta Lloyd-Max quantize (1-6 bit)
    │        → QJL residual sketch (1-bit sign projection)
    │        Store: packed indices + norms + qjl_sketches
    │
    └─ Values: group quantization (2/3/4 bit, min-max per group)
               Store: packed data + scales + zeros

Attention: online softmax + sparse V threshold + tiled processing
WASM SIMD128: f32x4 vectorized FWHT/dot/quantize (2-10x vs pure TS)
OpenCL fp16: vload8/vstore8 fused attention (Adreno 730)
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

## WASM SIMD128 Benchmark

| Operation | dim=256 | dim=1024 | dim=2048 |
|-----------|---------|----------|----------|
| Dot product | **10.1x** | **2.2x** | **2.0x** |
| FWHT | 1.04x | **2.1x** | 1.4x |

Measured on aarch64 (Snapdragon 8 Gen 1), Node.js 22, WASM SIMD128 vs pure TS.
At real KV-cache dimensions (1024+), WASM dominates dot-heavy paths (QJL sign projection).

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

## MCP Tools (14)

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
├── tq_fused_attention_v2.cl        — Fused attention (LUT + sparse V + q8)
├── tq_fused_attention_v3_fp16.cl   — fp16 + vload4 (Adreno 730 ALU 2x)
├── tq_fused_attention_v4_fp16.cl   — Full vectorized: vload8/vstore8 all paths
├── tq_fused_attention_tiled.cl     — Tiled variant (TILE_K=32)
└── tq_fwht_fp16                    — Vectorized FWHT butterfly (in v4)
```

Target: Adreno 730v3 via Rusticl (Mesa/KGSL). OpenCL 3.0+ / cl_khr_fp16.

v4 improvements:
- vload8/vstore8: query load, key dot, output rescale, final norm (256-bit burst)
- vload4/vstore4: value dequant+accum
- 2x bandwidth vs v3 on Adreno L2 (128B cache line)
- Separate tq_fwht_fp16 kernel — vectorized butterfly

## WASM SIMD128 Core (Rust)

```
native/wasm/
├── Cargo.toml         — cdylib, LTO, strip, opt-level=3
├── src/lib.rs         — FWHT, Lloyd-Max, QJL, pack/unpack, simd_dot
├── build.sh           — cargo + wasm-bindgen + wasm-opt pipeline
└── pkg/               — Output: 37KB .wasm + JS bindings
```

Functions (all SIMD128 f32x4 where applicable):
- `fwht` — Fast Walsh-Hadamard Transform (SIMD normalize)
- `fwht_batch` — Batch FWHT (N vectors)
- `lloyd_max_quantize` — Nearest centroid search + bit-pack
- `lloyd_max_dequantize` — Unpack + centroid lookup
- `qjl_sign_project` — 1-bit sign dot projection
- `simd_dot` — f32x4 dot product
- `pack_2bit` / `unpack_2bit` — 2-bit packing utilities

Build:
```bash
cd native/wasm
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --release
```

Integration: `src/native/wasm_backend.ts` — auto-init, fallback to TS if unavailable.

## Portable Installation

### Requirements
- Node.js 18+ (or Bun)
- Optional: Rust 1.70+ with wasm32-unknown-unknown target (for WASM rebuild)
- Optional: OpenCL 3.0 runtime (for GPU kernels)

### Quick Start (any platform)

```bash
git clone https://github.com/kosoymiki/turboquant-compressor.git
cd turboquant-compressor
npm install
npm run build
npm run smoke:stdio
```

### Termux (Android/aarch64)

```bash
pkg install nodejs-lts rust rust-std-wasm32-unknown-unknown binaryen
git clone https://github.com/kosoymiki/turboquant-compressor.git
cd turboquant-compressor
npm install
npm run build
# Optional: rebuild WASM
cd native/wasm && bash build.sh && cd ../..
```

### Claude Code MCP

Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["/path/to/turboquant-compressor/dist/server.js"]
    }
  }
}
```

Verify: `echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | node dist/server.js`

## Known Limitations

1. **Outlier sensitivity**: min-max group quantization for values. Mitigation: RotateKV channel reorder (implemented).
2. **Sparse V positional bias**: early tokens favored. Mitigation: attention-sink protection (first N tokens never skipped).
3. **Beta PDF assumption**: codebook optimized for uniform-on-sphere. Empirically validated by PolyKV (+0.57% PPL).
4. **QJL residual sketch**: wired into compress tool (`includeQJL: true`). WASM-accelerated sign projection.
5. **WASM fallback**: if wasm32 binary unavailable, pure TS path (2-10x slower on hot paths).

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
