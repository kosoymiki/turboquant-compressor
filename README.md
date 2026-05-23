# TurboQuant Compressor v4.5.0

**MCP server for compressed local vector search, context-pack retrieval, KV/cache analysis, and custom Adreno GPU readiness forensics.**

| | |
|---|---|
| **Version** | 4.5.0 |
| **Transport** | MCP stdio |
| **License** | GPL-3.0-or-later |
| **Algorithm** | LEVEL_1_PUBLIC — TurboQuant Beta Lloyd-Max + QJL residual correction |
| **Device** | Adreno A7xx / A8xx (Qualcomm SM8475+) |
| **Android** | 16+ |
| **Driver** | Mesa Rusticl 26.2 + Turnip Vulkan (KGSL) |

---

## Production Status

```
mesa_rusticl_kgsl      — VERIFIED, production-ready
turnip_vulkan_kgsl     — VERIFIED, Vulkan compute path active
opencl_adreno          — FORBIDDEN
vendor_opencl_qualcomm — FORBIDDEN
typescript_cpu         — fallback only
```

**Device-verified on**: Realme GT 2 Pro (RMX3709 / SM8475 / Adreno 730v3)

---

## Quick Start

```bash
git clone https://github.com/kosoymiki/turboquant-compressor.git
cd turboquant-compressor
npm install
npm run build
npm run smoke:stdio
```

### MCP Configuration

```json
{
  "mcpServers": {
    "turboquant": {
      "command": "node",
      "args": ["/absolute/path/to/turboquant-compressor/dist/server.js"]
    }
  }
}
```

---

## Benchmarks — Committed Evidence in `bench/results/`

All benchmarks are committed artifacts, not promises.

| File | Dims | Ratio | R@1 | R@5 | MRR | MSE | Status |
|------|------|-------|-----|-----|-----|-----|--------|
| `turboquant_local_2026-05-23.json` | 1024 | **7.93x** | — | — | — | — | Latest build |
| `open-test-local-20260521-160651.json` | 1024 | **7.93x** | 0.88 | 1.00 | 0.94 | 1.63e-05 | Full test, 162 chunks |

**Algorithm**: `LEVEL_1_PUBLIC` — TurboQuant Beta Lloyd-Max + QJL residual correction.

### TurboQuant Pipeline (arXiv:2504.19874)

1. **Random Hadamard rotation** (FWHT-based sign flip, mobile-friendly)
2. **Beta(d/2, d/2) coordinate PDF** — after rotation, coordinates follow Beta distribution on [-1, 1]
3. **Lloyd-Max codebook** — optimal quantization levels trained for Beta distribution
4. **QJL residual sketch** — Johnson-Lindenstrauss projection for unbiased dot-product estimation (Zandieh et al., 2024)
5. **Structured JL via FWHT** — O(d log d) projection, not O(d²)

---

## MCP Tools (17 tools)

### Compression
| Tool | Purpose |
|------|---------|
| `turboquant_compress` | Compress vectors into compact binary payload (2/3/4/8 bits per coordinate) |
| `turboquant_quantize` | Codebook computation and Lloyd-Max quantization |
| `turboquant_context_pack_build` | Build compressed context packs from files |
| `turboquant_context_pack_search` | Search built context packs |

### Cache & KV
| Tool | Purpose |
|------|---------|
| `turboquant_cache_plan` | Classify context blocks by volatility, recommend cache tiers |
| `turboquant_prompt_cache_lint` | Detect cache-busting patterns in context blocks |
| `turboquant_kv_analyze` | KV cache compression analysis (headDim, numLayers, seqLen) |

### Search
| Tool | Purpose |
|------|---------|
| `turboquant_vector_search` | Nearest neighbor search in compressed vector database |

### Probes
| Tool | Purpose |
|------|---------|
| `turboquant_backend_probe` | Probe available runtimes (Termux-safe, never imports torch/triton) |
| `turboquant_opencl_probe` | OpenCL availability and Adreno GPU detection |
| `turboquant_adreno_loader_probe` | Android linker namespace state for vendor OpenCL |
| `turboquant_cli_mcp_profile` | MCP client profiles (Claude Code, Codex, Gemini CLI, etc.) |

### Analysis
| Tool | Purpose |
|------|---------|
| `turboquant_cost_analyze` | Claude Code usage snapshots, cache efficiency metrics |

### Corpus Web Search
| Tool | Purpose |
|------|---------|
| `corpus_web_search` | FTS index + registry + web corroboration (always search corpus first) |

---

## Driver Stack (Adreno A7xx/A8xx)

The custom driver stack is the production path. Vendor OpenCL and generic OpenCL are forbidden.

### Install
```bash
# Via npm
npm run install:driver-root

# Verify
npm run probe:opencl
```

### What gets installed
```
/data/data/com.termux/files/tq-rusticl/
├── libRusticlOpenCL.so.1.0.0   (90MB, Mesa 26.2+ Rusticl)
├── libvulkan_freedreno.so       (69MB, Turnip Vulkan 1.4.335)
├── clc/                         (SPIR-V kernels)
├── kernels/                     (5 TurboQuant OpenCL kernels)
└── rusticl.icd                  → libRusticlOpenCL.so.1
```

### Production Policy
```json
{
  "productionPolicy": "custom_driver_stack_only",
  "productionReady": true,
  "productionRoute": "mesa_rusticl_kgsl",
  "allowedProductionRoutes": ["mesa_rusticl_kgsl", "turnip_vulkan_kgsl"],
  "forbiddenBackends": [
    "typescript_cpu", "python_cpu", "python_gpu", "triton_gpu",
    "opencl_generic", "opencl_adreno", "vendor_opencl_qualcomm"
  ]
}
```

---

## Architecture

```
turboquant-compressor/
├── src/
│   ├── tools/          # MCP tool implementations
│   ├── core/           # Rotation, quantization, format, compression
│   ├── runtime/       # Backend probes, hybrid training runtime
│   └── native/         # OpenCL probe, production policy assessment
├── dist/               # Compiled TypeScript (Node.js)
├── native/
│   ├── opencl/         # C++ native runtime, kernel sources
│   │   ├── src/       # Loader, runtime, kernel execution
│   │   ├── include/   # 33 C++ headers (tq_opencl_*.h)
│   │   ├── kernels/   # OpenCL kernel sources
│   │   ├── driver-pack/ # Device driver binaries
│   │   └── bench/     # Native CLI benchmarks
│   └── opencl/         # Staged Rusticl/Turnip driver pack
├── forensics/          # Evidence artifacts
│   ├── precision/     # Precision contracts
│   ├── imported/       # Historical evidence packs
│   └── precompiled-kernel-library/ # Kernel binaries
├── docs/               # Policy, contracts, technical reflections
├── scripts/            # Build, verify, benchmark scripts
├── bench/results/      # Committed benchmark artifacts
└── registry/           # TurboQuant registry (131 entries)
```

---

## Verification

```bash
npm run build           # TypeScript compile
npm run smoke:stdio    # Smoke test (stdio transport)
npm run verify:release  # Full release verification
npm run verify:mcp      # MCP conformance
npm run verify:termux   # Termux readiness check
```

---

## References

- **TurboQuant theory**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **Lloyd-Max quantization**: committed benchmark evidence in `bench/results/`
- **QJL correction**: research path, not in public artifacts
- **Mesa Rusticl**: [mesa3d.org](https://docs.mesa3d.org/)
- **Turnip Vulkan**: Qualcomm Freedreno / Turnip driver

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for version history.

| Version | Date | Key Change |
|---------|------|-----------|
| 4.5.0 | 2026-05-23 | Staged Rusticl/Vulkan detection, clean forensics, corpus triad |
| 4.5.0-beta | 2026-05-21 | LEVEL_1_PUBLIC algorithm, R@1=0.88, MRR=0.94 |
| 4.1.4 | 2026-05-14 | Early baseline, 384-dim, R@5=0.90 |

---

## License

GPL-3.0-or-later