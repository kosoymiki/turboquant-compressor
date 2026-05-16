# TurboQuant Compressor Architecture

## Overview

TurboQuant Compressor is an MCP server that implements TurboQuant-inspired vector compression for mobile-first deployment. The system provides compression and search capabilities for high-dimensional vectors.

## Architecture Components

### Core Modules

1. **Rotation Engine** (`src/core/rotation.ts`)
   - Uses Fast Walsh-Hadamard Transform (FWHT) for rotation
   - Auto-pads non-power-of-two dimensions to next power of two
   - Deterministic sign-flip pattern using Mulberry32 PRNG

2. **Quantizer** (`src/core/quantizer.ts`)
   - Uniform symmetric scalar quantization
   - Fixed codebook over [-1, 1] range
   - Supports 2, 3, 4, and 8 bits per value

3. **Binary Format** (`src/core/format.ts`)
   - Versioned header with magic number 'TQMC'
   - Float32 norms storage using DataView
   - CRC32 validation over full payload
   - Strict offset and length validation

4. **Bit Packing** (`src/core/pack.ts`)
   - Cross-byte boundary packing
   - Efficient storage for quantized values

### Tool Layer

1. **Compress** (`src/tools/compress.ts`)
   - Canonical API: `turboquant_compress`
   - Input: vectors, dimensions, seed, bitsPerValue
   - Output: compressed_database_b64, metadata

2. **Search** (`src/tools/search.ts`)
   - Canonical API: `turboquant_vector_search`
   - Input: query_vector, compressed_database_b64, top_k, metric
   - Output: results with scores

## Data Flow

### Compression Pipeline

```
Input Vector
    ↓
Rotation (FWHT with sign-flip)
    ↓
L2 Normalization
    ↓
Uniform Quantization (2-8 bits)
    ↓
Bit Packing
    ↓
Binary Format + CRC32
    ↓
Base64 Output
```

### Search Pipeline

```
Compressed Database (Base64)
    ↓
Binary Format Decode + CRC Validation
    ↓
Bit Unpacking
    ↓
Quantizer Decode (fixed codebook)
    ↓
Query Rotation
    ↓
Cosine/Dot Similarity
    ↓
Top-K Ranking
    ↓
Results Output
```

## Binary Format v3

The encoder writes v3 by default. The decoder supports v2 for backward compatibility.

### Header Structure v3 (80 bytes)

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 4 | Magic | 'TQMC' |
| 4 | 4 | Version | Format version (3) |
| 8 | 4 | Dimensions | Original vector dimensions |
| 12 | 4 | PaddedDimensions | Padded to power of 2 |
| 16 | 4 | VectorCount | Number of vectors |
| 20 | 1 | BitsPerValue | Quantization bits (2,3,4,8) |
| 21 | 3 | Reserved | Reserved |
| 24 | 4 | RotationSeed | PRNG seed |
| 28 | 1 | Flags | Reserved |
| 29 | 3 | Reserved | Reserved |
| 32 | 4 | HeaderLength | Header size (80) |
| 36 | 4 | Reserved | Reserved |
| 40 | 4 | NormsOffset | Offset to norms |
| 44 | 4 | NormsLength | Norms size in bytes |
| 48 | 4 | QuantizedOffset | Offset to quantized data |
| 52 | 4 | QuantizedLength | Quantized data size |
| 56 | 4 | QjlOffset | Offset to QJL sketch |
| 60 | 4 | QjlLength | QJL sketch size |
| 64 | 4 | PayloadCrc32 | CRC32 of payload |
| 68 | 12 | Reserved | Reserved |

### Header Structure v2 (72 bytes) - backward compatibility

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 4 | Magic | 'TQMC' |
| 4 | 4 | Version | Format version (2) |
| 8 | 4 | Dimensions | Original vector dimensions |
| 12 | 4 | PaddedDimensions | Padded to power of 2 |
| 16 | 4 | VectorCount | Number of vectors |
| 20 | 1 | BitsPerValue | Quantization bits (2,3,4,8) |
| 21 | 4 | RotationSeed | PRNG seed |
| 25 | 1 | Flags | Reserved |
| 28 | 4 | HeaderLength | Header size (72) |
| 32 | 4 | NormsOffset | Offset to norms |
| 36 | 4 | NormsLength | Norms size in bytes |
| 40 | 4 | QuantizedOffset | Offset to quantized data |
| 44 | 4 | QuantizedLength | Quantized data size |
| 48 | 4 | QjlOffset | Offset to QJL sketch |
| 52 | 4 | QjlLength | QJL sketch size |
| 56 | 4 | PayloadCrc32 | CRC32 of payload |

### Payload Structure

```
[Norms (Float32Array)] → [Quantized Vectors] → [QJL Sketch]
```

## Algorithm Levels

- **LEVEL_0_TURBOQUANT_INSPIRED_MVP**: Basic rotation + uniform quantization
- Future: QJL residual, Lloyd-Max codebooks

## Limitations

- Fixed codebook over [-1, 1] range
- No adaptive quantization
- No QJL residual stage
- Single-precision Float32 norms

## Performance Characteristics

- Compression: O(n log n) per vector (FWHT)
- Search: O(m log n) for m queries, n vectors
- Memory: ~1.5x overhead for metadata
- Mobile-friendly: No native addons, pure TypeScript