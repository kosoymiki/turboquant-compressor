# TurboQuant Compressor

**Vector compression and nearest-neighbor search for production ML systems.**

| | |
|---|---|
| Version | 4.5.2 |
| License | GPL-3.0-or-later |
| Algorithm | Beta Lloyd-Max + QJL residual correction |
| Target | Adreno A7xx/A8xx (Qualcomm SM8475+) |
| Status | Production-ready |

---

## What is this?

TurboQuant compresses high-dimensional vectors — the kind used in AI embeddings and ML models — into compact binary representations. It achieves **7.93x compression** with near-perfect nearest-neighbor search accuracy.

The compression uses:

1. **Random Hadamard rotation** — a fast, mobile-friendly transformation that spreads information uniformly across dimensions
2. **Beta Lloyd-Max quantization** — optimal compression levels for the rotated data
3. **QJL residual correction** — unbiased dot-product estimation for accurate search results

### Quick Start

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

## Benchmarks

Committed benchmark artifacts in `bench/results/`:

```yaml
---
version: 4.5.2
date: 2026-05-23
algorithm: LEVEL_1_PUBLIC
dimensions: 1024
compression_ratio: 7.93x
results:
  - id: turboquant_local_2026-05-23
    status: latest
  - id: open-test-local-20260521-160651
    status: verified
    metrics:
      recall@1:  0.88
      recall@5:  1.00
      mrr:       0.94
      mse:       1.63e-05
    dataset:
      chunks: 162
      dimensions: 1024
```

**Algorithm**: TurboQuant Beta Lloyd-Max + QJL residual correction  
**References**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874), Zandieh et al. ICML 2024

---

## Available Tools

### Compression

| Tool | Purpose |
|------|---------|
| `turboquant_compress` | Compress vectors into compact binary (2/3/4/8 bits per coordinate) |
| `turboquant_quantize` | Codebook computation and Lloyd-Max quantization |
| `turboquant_context_pack_build` | Build compressed context packs from files |
| `turboquant_context_pack_search` | Search built context packs |

### Search

| Tool | Purpose |
|------|---------|
| `turboquant_vector_search` | Nearest neighbor search in compressed database |

### Cache & KV Analysis

| Tool | Purpose |
|------|---------|
| `turboquant_cache_plan` | Classify context blocks by volatility |
| `turboquant_prompt_cache_lint` | Detect cache-busting patterns |
| `turboquant_kv_analyze` | KV cache compression analysis |

### Probes

| Tool | Purpose |
|------|---------|
| `turboquant_backend_probe` | Detect available runtimes (Termux-safe) |
| `turboquant_opencl_probe` | OpenCL availability and Adreno GPU |
| `turboquant_adreno_loader_probe` | Android linker namespace state |
| `turboquant_cli_mcp_profile` | MCP client profiles |

### Analysis

| Tool | Purpose |
|------|---------|
| `turboquant_cost_analyze` | Claude Code usage and cache efficiency |

---

## Architecture

```
turboquant-compressor/
├── src/
│   ├── tools/          # MCP tool implementations
│   ├── core/           # Rotation, quantization, format, compression
│   ├── runtime/       # Backend probes, hybrid runtime
│   └── native/        # OpenCL probe, production policy
├── dist/               # Compiled TypeScript
├── native/
│   ├── opencl/           # C++ native runtime
│   │   ├── src/          # Loader, runtime, kernel execution
│   │   ├── include/      # C++ headers
│   │   └── kernels/      # OpenCL kernel sources
│   ├── opencl-intercept/ # OpenCL API interception layer
│   │   ├── include/      # tq_intercept.h
│   │   ├── src/          # Core, API, memory, kernel profiling
│   │   └── test/         # Test suite
│   └── opencl/           # Staged Rusticl/Turnip driver
├── forensics/          # Evidence artifacts
├── docs/               # Technical documentation
├── scripts/            # Build and benchmark scripts
├── bench/results/      # Committed benchmarks
└── registry/           # TurboQuant registry
```

---

## Driver Stack

Production path for Adreno A7xx/A8xx devices.

### Verified Backends

```
✓ mesa_rusticl_kgsl      — Production-ready
✓ turnip_vulkan_kgsl     — Vulkan compute active
✗ opencl_adreno          — Forbidden
✗ vendor_opencl_qualcomm  — Forbidden
△ typescript_cpu         — Fallback only
```

### Installation

```bash
# Install custom driver stack
npm run install:driver-root

# Verify
npm run probe:opencl
```

### Installed Components

```
/data/data/com.termux/files/tq-rusticl/
├── libRusticlOpenCL.so.1.0.0   (90MB, Mesa 26.2+)
├── libvulkan_freedreno.so      (69MB, Turnip Vulkan 1.4.335)
├── clc/                         # SPIR-V kernels
├── kernels/                      # TurboQuant kernels
└── rusticl.icd                  # ICD loader
```

**Verified on**: Realme GT 2 Pro (RMX3709 / SM8475 / Adreno 730v3)

---

## Verification

```bash
npm run build           # TypeScript compile
npm run smoke:stdio    # Stdio transport smoke test
npm run verify:release  # Full release verification
npm run verify:mcp      # MCP conformance
npm run verify:termux   # Termux readiness
```

---

## Theory

TurboQuant is based on two papers:

1. **TurboQuant** ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874))
   - Random Hadamard rotation (FWHT-based)
   - Beta(d/2, d/2) coordinate distribution after rotation
   - Lloyd-Max codebook optimization

2. **QJL Correction** (Zandieh et al., ICML 2024)
   - Johnson-Lindenstrauss projection for unbiased dot-products
   - Structured JL via FWHT — O(d log d) projection

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.5.0 | 2026-05-23 | Rusticl/Vulkan detection, clean forensics, corpus triad |
| 4.5.0-beta | 2026-05-21 | LEVEL_1_PUBLIC, R@1=0.88, MRR=0.94 |
| 4.1.4 | 2026-05-14 | Early baseline, 384-dim, R@5=0.90 |

Full changelog: [`CHANGELOG.md`](CHANGELOG.md)

---

## License

GPL-3.0-or-later