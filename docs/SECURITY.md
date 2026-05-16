# Security Considerations

## Input Validation

All inputs are validated using Zod schemas:
- Vectors must be rectangular (all rows same length)
- All values must be finite numbers
- Base64 strings must match expected pattern

## Memory Limits

The compressor enforces limits to prevent resource exhaustion:
- Maximum dimensions: 8192
- Maximum vectors: 4096
- Memory estimation before compression

## Termux Constraints

On Android Termux:
- Maximum memory: 256 MB
- Default dimensions: 1024
- Default vectors: 128

## CRC32 Validation

All compressed data includes CRC32 checksums for integrity verification.

## No Network Access

The MCP server operates over stdio only with no network capabilities.

## Deterministic PRNG

Uses Mulberry32 for reproducible rotation patterns - no security-sensitive randomness.