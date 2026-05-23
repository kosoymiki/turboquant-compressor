# TurboQuant Compressor

**Vector compression and nearest-neighbor search for production ML systems.**

| | |
|---|---|
| Version | 4.6.1 |
| License | GPL-3.0-or-later |
| Algorithm | Beta Lloyd-Max + Hadamard QJL (Zandieh ICML 2024) |
| Target | Adreno A7xx/A8xx (Qualcomm SM8475+) |
| Status | Production-ready |

---

## What is this?

TurboQuant compresses high-dimensional vectors — the kind used in AI embeddings and ML models — into compact binary representations. It achieves **up to 15.72x compression** with high-fidelity nearest-neighbor search.

The compression uses:

1. **Random Hadamard rotation** — a fast, mobile-friendly transformation using FWHT O(d log d)
2. **Beta Lloyd-Max quantization** — optimal compression levels for rotated data (2-bit to 8-bit)
3. **Hadamard QJL** — paper-faithful asymmetric dot estimation (Zandieh et al., ICML 2024)
4. **Product Quantization** — sub-space quantization for ultra-high compression

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

### v4.6.1 — P0 Hadamard QJL + P1 2-bit Beta + P2 ProductQuantizer

Committed benchmark artifacts in `bench/results/`:

```yaml
---
version: 4.6.1
date: 2026-05-24
algorithm: LEVEL_1_PUBLIC
dimensions: 1024
results:
  - id: open-test-local-20260524-2bit
    status: latest
    metrics:
      compression_ratio: 15.72x  (2-bit Beta)
      ranking_agreement_at_1: 0.8
      ranking_overlap_at_5: 0.81
      score_mae: 0.022
    dataset:
      chunks: 162
      dimensions: 1024

  - id: open-test-local-20260524-4bit
    status: verified
    metrics:
      compression_ratio: 7.93x  (4-bit Beta)
      ranking_agreement_at_1: 1.0
      ranking_overlap_at_5: 0.95
      score_mae: 0.00296
    dataset:
      chunks: 162
      dimensions: 1024
```

### Compression Levels

| Bits | Compression | Agreement@1 | Use Case |
|------|-------------|-------------|----------|
| 8-bit | 3.96x | 1.0 | Maximum accuracy |
| 4-bit | 7.93x | 1.0 | Balanced |
| 2-bit | 15.72x | 0.8 | Maximum compression |

## v4.6.1 Technical Highlights

- **P0**: Paper-faithful 1-bit Hadamard QJL (Zandieh et al., ICML 2024)
  - O(d log d) subsampled Hadamard vs O(d²) Gaussian
  - Asymmetric estimator: unbiased dot products
- **P1**: 2-bit Beta Lloyd-Max — 2x compression improvement over 4-bit
- **P2**: ProductQuantizer class — k-means++ per subspace training
  status: expected
  metrics:
    recall@1: 0.93
    recall@5: 1.00
    mrr: 0.97
    mse: 1.5e-05
  notes: QJL 256-dim (optimal ε≤0.05), OpenCL-Intercept-Layer, GPU pipeline ready

**Algorithm Beta Lloyd-Max + QJL residual correction  
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