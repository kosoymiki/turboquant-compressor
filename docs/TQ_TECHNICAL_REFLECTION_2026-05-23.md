# TQ Technical Reflection — 2026-05-23

## Benchmarks (Latest)

### open-test-local-20260521-160651.json
```json
{
  "test_name": "turboquant_open_test_local",
  "files": 110,
  "chunks": 162,
  "approximate_tokens": 100723,
  "dimensions": 1024,
  "vector_count": 162,
  "compression_ratio": 7.930394875227077,
  "compressed_bytes": 83672,
  "original_bytes_estimate": 663552,
  "compress_ms": 206.06666666699914,
  "recall_at_1": 0.88,
  "recall_at_5": 1.0,
  "mrr": 0.94,
  "score_mae_on_exact_topk": 0.003285903916265309,
  "score_mse_on_exact_topk": 1.6264191248008223e-05,
  "algorithm_level": "LEVEL_1_PUBLIC"
}
```

### Summary
| Metric | Value |
|--------|-------|
| Compression ratio | 7.93x |
| Recall@1 | 0.88 |
| Recall@5 | 1.0 |
| MRR | 0.94 |
| MSE | 1.63e-05 |
| Algorithm | LEVEL_1_PUBLIC (TurboQuant Beta) |

---

## C++ Migration Baseline

### Current Reality

**Already Native:**
- `native/opencl/src/` — loader, runtime selection
- `native/opencl/kernels/` — kernel execution
- `native/opencl/include/` — headers
- `tq_opencl_cli` — CLI probe/smoke/benchmark

**Still JS-First:**
- `src/core/` — compression/search hot paths
- `src/runtime/` — hybrid training runtime
- `src/server.ts` — MCP server

**Must Move to C++:**
1. Runtime probes and classification
2. Compression/search hot paths
3. Kernel cache and autotune
4. Device/route admission

**Can Stay JS-Shell:**
1. MCP transport
2. JSON schema validation
3. Request/response shaping
4. Fallback adapters

**Must Leave Product Truth:**
1. ADB collector orchestration
2. Long-lived corpus ingestion
3. Evidence backfilling
4. Historical recovery intelligence

---

## Driver Stack (RMX3709 / SM8475)

### Staged on Device
```
/data/data/com.termux/files/tq-rusticl/
├── libRusticlOpenCL.so.1.0.0  (90MB, Mesa 26.2+ Rusticl)
├── libvulkan_freedreno.so (69MB, Turnip Vulkan 1.4.335)
├── clc/  (SPIR-V kernels)
├── kernels/ (5 TQ kernels: tq_attention_logits, tq_mse_score, etc.)
└── rusticl.icd → libRusticlOpenCL.so.1
```

### Probe Result
```json
{
  "recommendedBackend": "mesa_rusticl_kgsl",
  "productionRoute": "mesa_rusticl_kgsl",
  "productionReady": true,
  "platformCount": 1,
  "deviceCount": 1,
  "deviceNames": ["Qualcomm Adreno 730v3 (Mesa Rusticl/KGSL + Turnip Vulkan)"],
  "vendorNames": ["Qualcomm"],
  "versionStrings": ["OpenCL 3.1 (Mesa Rusticl)", "Vulkan 1.4.335 (Turnip/KGSL)"]
}
```

---

## Frontend State

### 4 TQ Mirrors Synced
1. `tmp_turboquant/` (canonical)
2. `turboquant/` (mirror)
3. `aebuilder-release/vendor/turboquant-compressor/`
4. `corpus-control-plane/mcp/turboquant/`

### Files Synced
- `dist/native/opencl_probe.js` (dual stack detection)
- `dist/tools/turboquant_backend_probe.js`
- `dist/runtime/backend_probe.js`

### MD5: `1b679063f16618b9fec9501c1b4a7e48`

---

## Corpus Control Plane

### Skills (32 total)
- Critical: 9 (turboquant, opencl_ecosystem, driver-debug, etc.)
- High: 4
- Medium: 19

### Corpus Triad
- `corpus_armor.py` — validation, language detection
- `corpus_sword.py` — code plan, patterns
- `corpus_shield.py` — code validation
- `corpus_triad.py` — master router
- `corpus_master_router.py` — pattern matrix detection

### Knowledge Lines
- Master skill: 188
- Domain skills: 1871 (31 skills)
- Total: 2059 lines

---

## Open Issues

1. **C++ Migration** — evidence still reads forensics/* directly
2. **WASM** — treated as peer runtime, not fallback
3. **Release truth** — depends on repo-local evidence files
4. **ADB** — external validation lane only (not hidden truth source)

---

## Next Steps

1. Replace forensics/* reads with explicit native contract inputs
2. Define one canonical native interface for all operations
3. Move JS hybrid runtime to orchestration wrapper role
4. Split product truth from corpus evidence truth
5. Run fresh benchmark on device with Rusticl + Turnip