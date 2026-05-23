# TurboQuant Compressor v4.5.0

**MCP server for compressed local vector search, context-pack retrieval, KV/cache analysis, and Adreno GPU readiness forensics.**

---

## Status

| | |
|---|---|
| **Version** | 4.5.0 |
| **Transport** | MCP stdio |
| **License** | GPL-3.0-or-later |
| **Benchmarks** | `bench/results/` |

---

## Quick Start

```bash
git clone https://github.com/kosoymiki/turboquant-compressor.git turboquant
cd turboquant
npm install
npm run build
npm run smoke:stdio
```

---

## Benchmarks (committed evidence in `bench/results/`)

| File | Dims | Ratio | R@1 | R@5 | MRR | Notes |
|------|------|-------|-------|------|------|-----|------|
| `turboquant_local_2026-05-23.json` | 1024 | **7.93x** | — | — | — | Latest build artifact |
| `open-test-local-20260521-160651.json` | 1024 | **7.93x** | 0.88 | 1.00 | 0.94 | Full test |
| `open-test-local-20260514-220247.json` | 384 | **5.88x** | 0.60 | 0.90 | — | Early baseline |

**Public path**: `LEVEL_1_PUBLIC` — TurboQuant Beta Lloyd-Max scalar quantization (no QJL correction)

---

## What It Does

| Tool | Purpose |
|------|---------|
| `turboquant_compress` | Compress vectors into compact binary payload |
| `turboquant_vector_search` | Search nearest neighbors in compressed database |
| `turboquant_context_pack_build` | Build context packs from files |
| `turboquant_context_pack_search` | Search built packs |
| `turboquant_cache_plan` | Classify context blocks by volatility |
| `turput_quant_kv_analyze` | KV cache analysis |
| `turboquant_backend_probe` | Probe available runtimes (Termux-safe) |
| `turboquant_opencl_probe` | OpenCL availability |
| `turboquant_adreno_loader_probe` | Adreno namespace state |
| `turboquant_cost_analyze` | Claude Code usage snapshots |
| `turboquant_quantize` | Codebook quantization |
| `turboquant_prompt_cache_lint` | Cache busting patterns |
| `turboquant_cli_mcp_profile` | MCP client profiles |

Corpus-first web search (always search corpus before web):

| Tool | Purpose |
|------|---------|
| `corpus_web_search` | FTS index + registry + web corroboration |

---

## Architecture

```
turboquant-compressor/
├── src/                      # TypeScript source
│   ├── tools/                # MCP tool implementations
│   ├── core/                 # Rotation, quantization, format
│   └── runtime/              # Backend probes
├── dist/                     # Compiled JS
├── native/
│   └── opencl/              # C++ headers, driver-pack, kernels
├── forensics/                 # Device evidence
├── docs/                    # Policy, contracts
├── scripts/                  # Build/verify scripts
└── bench/results/            # Benchmark artifacts
```

---

## Driver Stack (Adreno A7xx/A8xx)

Install driver pack:

```bash
# Via npm
npm run install:driver-root

# Or manually
ls native/opencl/driver-pack/
```

Verify:

```bash
npm run probe:opencl
```

---

## MCP Configuration

```json
{
  "mcpServers": {
    "turboquant": {
      "command": "node",
      "args": ["/path/to/turboquant/dist/server.js"]
    }
  }
}
```

---

## Verification

```bash
npm run build
npm run smoke:stdio
npm run verify:release
npm run verify:mcp
```

---

## References

- TurboQuant theory: arXiv:2504.19874
- Lloyd-Max quantization: committed benchmark evidence
- QJL correction: research path (not in public artifacts)

---

## License

GPL-3.0-or-later
