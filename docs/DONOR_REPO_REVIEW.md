# Donor Repository Review

## Overview

This document reviews the donor repositories that influenced TurboQuant Compressor implementation.

## MCP/Agent Tooling

### modelcontextprotocol/typescript-sdk

**Purpose:** Official MCP TypeScript SDK

**Key Learnings:**
- StandardSchema format for tool definitions
- Proper transport abstraction (StdioServerTransport)
- Handler pattern for tool execution

**Implementation Used:**
- `McpServer` class for server creation
- `StdioServerTransport` for stdio communication
- `fromJsonSchema()` for schema conversion

**License:** MIT

### anthropics/skills

**Purpose:** Reference implementation for Claude skills

**Key Learnings:**
- Project structure and organization
- Tool input validation patterns
- Error handling approach

**License:** MIT

## TurboQuant/PolarQuant/QJL Implementations

### amirzandieh/QJL

**Purpose:** 1-bit Quantized Johnson-Lindenstrauss for KV cache

**Key Learnings:**
- QJL projection concept
- Residual estimation approach
- 1-bit quantization strategy

**Status:** Not implemented in current version

**License:** MIT

### ericshwu/PolarQuant

**Purpose:** Official PolarQuant implementation for KV cache

**Key Learnings:**
- Polar transformation approach
- KV-cache specific optimizations
- Paper-faithful implementation

**Status:** Reviewed for reference

**License:** MIT

### scos-lab/turboquant

**Purpose:** Python research implementation

**Key Learnings:**
- Python code structure
- QJL residual code
- Fidelity methodology

**Status:** Reviewed for reference

**License:** MIT

### botirk38/turboquant

**Purpose:** Zig implementation with NEON support

**Key Learnings:**
- Engine-based API design
- SIMD/NEON considerations
- Performance optimization patterns

**Status:** Reviewed for reference

**License:** MIT

### zlaabsi/turboquant-wasm

**Purpose:** JavaScript/WASM packaging

**Key Learnings:**
- WASM bundling approach
- Direct compressed-vector search
- Portable runtime design

**Status:** Reviewed for reference

**License:** MIT

## Vector Search/Quantization Systems

### facebookresearch/faiss

**Purpose:** Vector search library

**Key Learnings:**
- Separation of encoding/index/search
- PQ/SQ/HNSW naming conventions
- Binary serialization discipline

**License:** MIT

### nmslib/hnswlib

**Purpose:** HNSW implementation

**Key Learnings:**
- Simple HNSW API design
- In-memory ANN patterns
- Update/insert semantics

**License:** Apache 2.0

### qdrant/qdrant

**Purpose:** Vector database

**Key Learnings:**
- Quantization as optional layer
- Payload limits and policies
- Persistence approach

**License:** Apache 2.0

### unum-cloud/usearch

**Purpose:** Compact vector index

**Key Learnings:**
- Downcasting/quantization tradeoffs
- Cross-language bindings
- Memory efficiency patterns

**License:** Apache 2.0

### sqliteai/sqlite-vector

**Purpose:** Embedded vector storage

**Key Learnings:**
- Mobile/embedded design
- Low memory defaults
- Multiple data type support

**License:** MIT

## LLM Inference Systems

### ggml-org/llama.cpp

**Purpose:** LLM inference in C++

**Key Learnings:**
- Quantization approaches (GPTQ, AWQ)
- KV cache management
- Performance optimization

**License:** MIT

### vllm-project/vllm

**Purpose:** High-throughput LLM serving

**Key Learnings:**
- PagedAttention
- Continuous batching
- Quantization integration

**License:** Apache 2.0

## License Summary

| Repository | License | Used |
|------------|---------|------|
| modelcontextprotocol/typescript-sdk | MIT | Yes |
| anthropics/skills | MIT | Yes |
| amirzandieh/QJL | MIT | Reference |
| ericshwu/PolarQuant | MIT | Reference |
| scos-lab/turboquant | MIT | Reference |
| botirk38/turboquant | MIT | Reference |
| zlaabsi/turboquant-wasm | MIT | Reference |
| facebookresearch/faiss | MIT | Reference |
| nmslib/hnswlib | Apache 2.0 | Reference |
| qdrant/qdrant | Apache 2.0 | Reference |
| unum-cloud/usearch | Apache 2.0 | Reference |
| sqliteai/sqlite-vector | MIT | Reference |
| ggml-org/llama.cpp | MIT | Reference |
| vllm-project/vllm | Apache 2.0 | Reference |

## Code Reuse Notes

- No direct code copied from donor repositories
- Concepts and patterns adapted
- All implementations are original
- All dependencies are npm packages