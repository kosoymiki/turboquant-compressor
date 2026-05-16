# Binary Format Specification v3

## Current encoder

v3.2.0 encodes new payloads as binary format v3.

- Magic: `TQMC`
- Version: `3`
- Header length: `80`
- Endianness: little-endian
- Payload CRC32 covers norms + quantized vectors + optional QJL section
- v2 decoder remains for backward compatibility only

## v3 Header (80 bytes)

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 4 | Magic | `TQMC` (0x54, 0x51, 0x4D, 0x43) |
| 4 | 4 | Version | Format version (3) |
| 8 | 4 | Dimensions | Original vector dimensionality |
| 12 | 4 | PaddedDimensions | Next power of 2 >= dimensions |
| 16 | 4 | VectorCount | Number of vectors |
| 20 | 1 | BitsPerValue | Quantization bits (2, 3, 4, 8) |
| 21 | 3 | Reserved | Reserved for future use |
| 24 | 4 | RotationSeed | PRNG seed for rotation |
| 28 | 1 | Flags | Reserved for future use |
| 29 | 3 | Reserved | Reserved for future use |
| 32 | 4 | HeaderLength | Header size (80) |
| 36 | 4 | Reserved | Reserved for future use |
| 40 | 4 | NormsOffset | Offset to norms array |
| 44 | 4 | NormsLength | Norms size in bytes |
| 48 | 4 | QuantizedOffset | Offset to quantized data |
| 52 | 4 | QuantizedLength | Quantized data size |
| 56 | 4 | QjlOffset | Offset to QJL sketch |
| 60 | 4 | QjlLength | QJL sketch size (0 if none) |
| 64 | 4 | PayloadCRC32 | CRC32 of norms + quantized + QJL |
| 68 | 12 | Reserved | Reserved for future use |

## v2 Header (72 bytes) - backward compatibility

The decoder supports v2 for backward compatibility:

| Offset | Size | Field | Description |
|---:|---:|---|---|
| 0 | 4 | Magic | `TQMC` |
| 4 | 4 | Version | Format version (2) |
| 8 | 4 | Dimensions | Original vector dimensionality |
| 12 | 4 | PaddedDimensions | Next power of 2 >= dimensions |
| 16 | 4 | VectorCount | Number of vectors |
| 20 | 1 | BitsPerValue | Quantization bits |
| 21 | 4 | RotationSeed | PRNG seed for rotation |
| 25 | 1 | Flags | Reserved |
| 28 | 4 | HeaderLength | Header size (72) |
| 32 | 4 | NormsOffset | Offset to norms array |
| 36 | 4 | NormsLength | Norms size in bytes |
| 40 | 4 | QuantizedOffset | Offset to quantized data |
| 44 | 4 | QuantizedLength | Quantized data size |
| 48 | 4 | QjlOffset | Offset to QJL sketch |
| 52 | 4 | QjlLength | QJL sketch size |
| 56 | 4 | PayloadCRC32 | CRC32 of payload |
| 60 | 12 | Reserved | Reserved |

## Data Layout

```
[Header: 80 bytes for v3, 72 bytes for v2]
[Norms: VectorCount * 4 bytes (Float32)]
[Quantized Data: VectorCount * ceil(PaddedDimensions * BitsPerValue / 8) bytes]
[QJL Sketch: optional]
```

## Magic Number

The magic constant `TQMC` identifies TurboQuant Magic Constant files.

## CRC32

Uses standard polynomial 0xEDB88320 (same as gzip, PNG). CRC covers full payload from NormsOffset to end of QJL sketch.

## Norms Storage

Norms are stored as Float32Array using DataView serialization (little-endian).

## Example

```
Header: TQMC v3, 1024 dims, 128 vectors, 4-bit, seed=42
PaddedDimensions: 1024
Norms: 128 * 4 = 512 bytes
Quantized: 128 * 512 bytes = 65536 bytes
Total: 80 + 512 + 65536 = 66128 bytes
```

## ABI Note

Format v3 is 8-byte aligned for Rust/Zig/C/WASM compatibility. All fields are naturally aligned:
- 4-byte fields at 4-byte offsets
- 8-byte reserved section at offset 68

Format v2 has an unaligned `RotationSeed` uint32 at offset 21. JavaScript `DataView` supports unaligned reads/writes, but native readers should use byte-addressed access.

## Validation Requirements

A conforming decoder MUST reject:
- wrong magic
- unsupported version (not 2 or 3)
- headerLength != 72 (v2) or 80 (v3)
- vectorCount < 1
- dimensions outside supported range
- paddedDimensions not equal to nextPowerOfTwo(dimensions)
- bitsPerValue not in {2,3,4,8}
- normsOffset != headerLength
- quantizedOffset != normsOffset + normsLength
- qjlOffset != quantizedOffset + quantizedLength
- qjlOffset + qjlLength != file length
- payload CRC mismatch