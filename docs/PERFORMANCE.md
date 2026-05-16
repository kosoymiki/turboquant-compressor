# Performance Characteristics

## Overview

This document describes the performance characteristics of TurboQuant Compressor and provides guidance for optimization.

## Compression Performance

### Time Complexity

| Operation | Complexity | Description |
|-----------|------------|-------------|
| Rotation (FWHT) | O(n log n) | Per vector, n = padded dimensions |
| Normalization | O(n) | Per vector |
| Quantization | O(n) | Per vector |
| Bit Packing | O(n) | Per vector |
| **Total per vector** | **O(n log n)** | Dominated by FWHT |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Input vectors | n × d × 8 bytes | Float64 |
| Rotated vectors | n × d × 8 bytes | Float64 |
| Quantized | n × ceil(d × b / 8) bytes | Packed bits |
| Norms | n × 4 bytes | Float32 |
| Overhead | ~1.5x | Header, alignment |

Where:
- n = number of vectors
- d = dimensions
- b = bits per value (2, 3, 4, or 8)

### Compression Ratio

| Bits per Value | Theoretical Ratio | Typical Observed |
|----------------|-------------------|------------------|
| 2 | 32x | 20-30x |
| 3 | 21.3x | 15-20x |
| 4 | 16x | 10-15x |
| 8 | 8x | 5-8x |

*Actual ratios depend on data distribution and padding.*

## Search Performance

### Time Complexity

| Operation | Complexity | Description |
|-----------|------------|-------------|
| Query rotation | O(d log d) | Single query |
| Decode vectors | O(n × d) | All database vectors |
| Similarity | O(n × d) | Dot/cosine computation |
| Ranking | O(n log k) | Top-k selection |
| **Total** | **O(n × d)** | Dominated by decode + similarity |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Compressed database | ~n × d × b / 8 | Packed vectors |
| Norms | n × 4 bytes | Float32 |
| Decoded (temporary) | n × d × 4 bytes | Float32 |
| **Total working set** | **~2 × compressed size** | During search |

## Mobile Performance (Termux)

### Benchmark Policy

Device-specific benchmark numbers must not be claimed without logs.

Every benchmark entry must include:

- device model
- Android version
- Termux version
- Node version
- thermal/battery state
- vector dimensions
- vector count
- bits per value
- median/p95 timings
- git/package version
- exact command used

Unmeasured examples must be labeled `illustrative only`.

### Benchmarks (Illustrative Only)

| Device | Dimensions | Vectors | Compression | Search |
|--------|------------|---------|-------------|--------|
| Pixel 6 | 1536 | 1000 | ~50ms/vector | ~100ms |
| Pixel 6 | 1536 | 10000 | ~50ms/vector | ~800ms |
| Pixel 6 | 3072 | 1000 | ~80ms/vector | ~150ms |

*These numbers are illustrative only. Actual results depend on device, thermal state, and system load.*

### Optimization Tips

1. **Use lower bits per value** (2-4) for better performance
2. **Process in batches** for large datasets
3. **Limit dimensions** if possible (512-1024)
4. **Use cosine metric** (faster than dot product)

## Scaling Considerations

### Maximum Supported

| Parameter | Limit | Reason |
|-----------|-------|--------|
| Dimensions | 8192 | Memory constraints |
| Vectors | 4096 | Default limit |
| Bits per value | 8 | Codebook size |
| Archive size | 256MB | Termux default |

### Memory Estimation

```typescript
import { estimateCompressionMemory } from './core/limits.js';

const memoryMB = estimateCompressionMemory(
  dimensions,    // e.g., 1536
  vectorCount,   // e.g., 1000
  bitsPerValue   // e.g., 4
);
```

## Comparison with Other Methods

| Method | Compression | Speed | Accuracy |
|--------|-------------|-------|----------|
| TurboQuant (ours) | 10-20x | Fast | Approximate |
| Faiss SQ | 4x | Very fast | Good |
| Faiss PQ | 8-16x | Fast | Good |
| Product Quantization | 16-64x | Medium | Excellent |

*Comparisons are approximate and depend on implementation details.*

## Performance Testing

### Run Benchmarks

```bash
# Termux benchmark
npm run bench:termux

# Smoke test with timing
npm run smoke:stdio
```

### Custom Benchmarks

```typescript
import { compressVectors, searchVectors } from './tools/compress.js';

const start = performance.now();
const result = compressVectors({ vectors, dimensions, seed, bitsPerValue });
const compressionTime = performance.now() - start;

const searchStart = performance.now();
const searchResult = searchVectors(compressedData, queryVector, { k, metric });
const searchTime = performance.now() - searchStart;

console.log(`Compression: ${compressionTime}ms`);
console.log(`Search: ${searchTime}ms`);
```

## Future Optimizations

### Planned Improvements

1. **SIMD acceleration** (ARM NEON)
2. **WebAssembly backend** for better performance
3. **Parallel compression** for multi-core
4. **Index-based search** (HNSW integration)

### Known Bottlenecks

1. **FWHT** - O(n log n) dominates compression
2. **Full decode** - Required for accurate similarity
3. **Float32 operations** - No integer path yet

## Recommendations

### For Best Performance

1. Use 2-4 bits per value
2. Limit dimensions to 1024-1536
3. Use cosine metric (faster than dot)
4. Process vectors in batches of 100-1000

### For Best Quality

1. Use 8 bits per value
2. Use higher dimensions (2048+)
3. Use dot metric for precise ranking
4. Validate on representative data