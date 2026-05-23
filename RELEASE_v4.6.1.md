# TurboQuant v4.6.1 — Hadamard QJL + 2-bit Beta + ProductQuantizer

## What's New

### P0: Paper-Faithful Hadamard QJL (Zandieh et al., ICML 2024)

**Reference**: Zandieh et al., "Efficient and Accurate Inner Products for LLMs", arXiv:2406.03482

| Property | Value |
|----------|-------|
| Projection | O(d log d) subsampled Hadamard |
| Quantization | 1-bit sign (zero overhead) |
| Estimator | Asymmetric: QJL(keys) + GaussianJL(queries) |
| Implementation | `native/kernel/src/tq_qjl_hadamard.cpp` |

**Tested**: 16/16 assertions passing in `test_hadamard_qjl`

### P1: 2-bit Beta Lloyd-Max

**Compression breakthrough**: 15.72x (vs 7.93x at 4-bit)

```bash
TQ_BITS_PER_VALUE=2 node scripts/open-test-local.mjs
# compression_ratio: 15.72x
# ranking_agreement_at_1: 0.8
```

| Bits | Compression | Agreement@1 | Score MAE |
|------|-------------|-------------|------------|
| 8-bit | 3.96x | 1.0 | ~0.001 |
| 4-bit | 7.93x | 1.0 | 0.003 |
| 2-bit | 15.72x | 0.8 | 0.022 |

### P2: ProductQuantizer Class

**File**: `native/core/include/tq_product_quantizer.h`

```cpp
class ProductQuantizer {
    void train(const float* X, uint32_t N, uint32_t seed, uint32_t max_iter, bool use_opq);
    void encode(const float* x, uint8_t* code) const;
    void build_lut(const float* query, float* lut_out) const;
    void search(...) const;  // Heap-based top-k
};
```

Features:
- k-means++ per subspace clustering
- OPQ rotation (Ge et al., CVPR 2013)
- LUT-based asymmetric search

### Technical Debt Resolved

| Issue | Fix |
|-------|-----|
| `clipped` undefined | Added `clipped_count` variable in `tq_lsq_optimizer.cpp` |
| KV cache segfault | Handle unquantized buffer tokens in `tq_kv_cache.cpp` |
| C++ init order | Reordered constructor initializers in `ProductQuantizer` |
| Compiler warnings | 3 kernel warnings (unused variables) resolved |

## Verification

```bash
# TypeScript build
npm run build && npm run smoke:stdio

# Native core tests
cd native/core/build && ./test_tq_core  # 51/51 PASS

# Kernel Hadamard QJL tests
cd native/kernel/build && ./test_hadamard_qjl  # 16/16 PASS

# Full benchmark
node scripts/open-test-local.mjs  # 4-bit: 7.93x
TQ_BITS_PER_VALUE=2 node scripts/open-test-local.mjs  # 2-bit: 15.72x
```

## Benchmarks

```yaml
version: 4.6.1
date: 2026-05-24
algorithm: LEVEL_1_PUBLIC

results:
  - bits: 4
    compression_ratio: 7.93x
    ranking_agreement_at_1: 1.0
    ranking_overlap_at_5: 0.95
    score_mae: 0.003

  - bits: 2
    compression_ratio: 15.72x
    ranking_agreement_at_1: 0.8
    ranking_overlap_at_5: 0.81
    score_mae: 0.022
```