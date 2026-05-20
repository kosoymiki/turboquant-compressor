
# TurboQuant Compressor v4.1.2

**Termux-first MCP server for compressed local vector search, context-pack retrieval, KV/cache analysis, and Adreno/OpenCL readiness forensics.**

TurboQuant Compressor is a local **Model Context Protocol (MCP)** server designed for Claude Code / stdio-style agent hosts on Android Termux and Linux. It provides compressed vector storage/search, context-pack construction, cache planning, prompt-cache linting, KV analysis, backend probing, and OpenCL/Adreno diagnostics.

The current public benchmark evidence supports **5.5x+ local corpus compression** on the committed open-test corpus, with retrieval quality recorded in `bench/results/`.

---

## Status

**Current release:** `v4.1.2`  
**Primary target:** Termux + local MCP hosts  
**Transport:** MCP stdio  
**License:** GPL-3.0-or-later

**Canonical custom driver artifact:** `native/opencl/driver-pack/tq-driver-pack-adreno-a7xx-a8xx.tar.zst`
It installs into `native/opencl/driver-root/` through `npm install`, `npm run install:driver-root`, or the first safe runtime call.

This repository currently ships:

- A TypeScript MCP server over stdio
- 13 MCP tools verified by the conformance transcript
- Compressed vector database format with deterministic encode/decode path
- Local corpus compression/search benchmarks
- Context-pack build/search tools for local retrieval
- KV/cache analysis utilities
- Termux/OpenCL/Adreno probe and release-evidence gates
- Release verification scripts for package, MCP, schema, scientific claims, Termux readiness, OpenCL claims, archives, and artifact parity

Verification depends on installed npm dependencies. A fresh clone must run `npm install` before `npm test` or `npm run verify:release`.

For a clean export artifact that excludes generated build residue and dead implementation lanes, use:

```bash
npm run package:release-slice
```

---

## Measured Results

The public benchmark artifacts in `bench/results/` show **5.5x+ compression** on local corpus tests.

| Artifact | Files | Chunks | Dimensions | Compression ratio | Recall@1 | Recall@5 | MRR | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `bench/results/open-test-local-20260514-220247.json` | 38 | 57 | 384 | **5.8759x** | 0.60 | 0.90 | — | Early open local test |
| `bench/results/open-test-local-20260514-224444.json` | 38 | 57 | 384 | **5.8759x** | 0.60 | 0.90 | 0.708 | 50-query evaluation |
| `bench/results/open-test-local-20260514-233707.json` | 57 | 78 | 384 | **5.8844x** | 0.58 | 0.92 | 0.7047 | Larger corpus evaluation |
| `bench/results/open-test-local-20260521-022922.json` | 103 | 151 | 384 | **5.8957x** | 0.34 | 0.66 | 0.4633 | Current truthful broader repo sweep on the shipped Beta Lloyd-Max public path |

### Practical headline

```text
Measured local corpus compression: 5.5x+
Best public artifact:              5.8957x
Recall@5 range:                    0.66–0.92
Format version:                    3
Algorithm level:                   LEVEL_0_TURBOQUANT_INSPIRED_MVP
````

The benchmark artifacts intentionally include warnings when a feature is not part of the public LEVEL_0 path. In particular, current committed public local-corpus artifacts are generated with:

```text
include_qjl: false
Public path uses TurboQuant Beta Lloyd-Max scalar quantization without QJL correction
```

This keeps the README honest: **5.5x+ compression is measured**, while higher-quality QJL-assisted retrieval remains gated until it is committed as public benchmark evidence.

---

## What TurboQuant Compressor Does

TurboQuant Compressor gives an MCP host local tools for:

* Compressing vectors into compact binary payloads
* Searching compressed vector databases
* Building compressed context packs from files
* Searching context packs by query
* Estimating cache/cost behavior
* Finding prompt-cache busting patterns
* Analyzing KV/cache compression scenarios
* Probing backend availability
* Diagnosing Termux/OpenCL/Adreno readiness
* Returning host-specific MCP configuration profiles

It is designed for local agent workflows where context, retrieval, and compression need to be inspectable and reproducible.

---

## MCP Tools

The public conformance transcript verifies **13 tools**.

| Tool                             | Purpose                                                         |
| -------------------------------- | --------------------------------------------------------------- |
| `turboquant_compress`            | Compress vectors into a compact database payload                |
| `turboquant_vector_search`       | Search nearest neighbors inside a compressed vector database    |
| `turboquant_cost_analyze`        | Analyze Claude Code usage/cache cost snapshots                  |
| `turboquant_cache_plan`          | Classify context blocks by volatility and recommend cache tiers |
| `turboquant_prompt_cache_lint`   | Detect prompt-cache busting patterns                            |
| `turboquant_context_pack_build`  | Build compressed retrieval packs from local files               |
| `turboquant_context_pack_search` | Search a previously built context pack                          |
| `turboquant_cli_mcp_profile`     | Emit MCP host profiles/config snippets                          |
| `turboquant_quantize`            | Run quantization experiments/tooling                            |
| `turboquant_kv_analyze`          | Analyze KV/cache compression scenarios                          |
| `turboquant_backend_probe`       | Probe available runtime backends                                |
| `turboquant_opencl_probe`        | Probe OpenCL state from Termux/Linux                            |
| `turboquant_adreno_loader_probe` | Diagnose Adreno/OpenCL loader state                             |

---

## Architecture

```text
TurboQuant Compressor
├── src/
│   ├── server.ts                 # MCP stdio server
│   ├── tools/                    # MCP tool implementations
│   ├── core/                     # rotation, quantizer, format, limits
│   ├── native/                   # backend probes / native readiness
│   └── research/                 # experimental research paths
├── scripts/
│   ├── verify-release.mjs
│   ├── verify-release-evidence.mjs
│   ├── verify-opencl-claims.mjs
│   ├── verify-mcp-conformance.mjs
│   ├── verify-termux-ready.mjs
│   ├── benchmark-opencl-adreno.mjs
│   ├── smoke-stdio.mjs
│   └── mcp-transcript.mjs
├── bench/results/
│   └── open-test-local-*.json    # public local compression/search evidence
├── forensics/
│   ├── mcp-conformance-transcript.json
│   └── device/release evidence
├── docs/
│   └── Termux/OpenCL/release documentation
└── package.json
```

---

## Compression Path

The public compressed-vector path uses:

1. Deterministic vector validation
2. Rotation/normalization path
3. Uniform symmetric scalar quantization
4. Compact binary format
5. Base64 MCP-safe transport payload
6. Decode/search path over compressed database contents

The measured local-corpus artifacts currently report:

```text
algorithm_level: LEVEL_0_TURBOQUANT_INSPIRED_MVP
format_version: 3
include_qjl: false
```

This means the current public proof is strongest for the shipped LEVEL_0 compression/search path. Experimental QJL/Lloyd-Max/OpenCL components should be treated as separately gated until their evidence artifacts are present and wired into the public search path.

---

## GPU / OpenCL / Adreno Status

TurboQuant Compressor includes Adreno/OpenCL readiness and benchmark tooling, but strong GPU acceleration claims are intentionally gated by forensic evidence.

Relevant commands:

```bash
npm run probe:opencl
npm run build:opencl
npm run bench:opencl
npm run verify:adreno-opencl
npm run verify:opencl-claims
npm run verify:release-evidence
```

The OpenCL claim gate requires evidence such as:

```text
forensics/opencl-adreno-report.json
forensics/adreno/loader-report.json
```

Optional performance evidence may include:

```text
forensics/adreno/opencl-performance-report.json
forensics/adreno/opencl-sustained-report.json
```

If these artifacts are missing or failing, README/Docs must not claim a verified Adreno production acceleration path. The project can still expose probes and diagnostics; it just must not overclaim runtime acceleration without device evidence.

---

## Installation

### Termux / Android

```bash
pkg update
pkg install -y nodejs git clang make cmake python

git clone https://github.com/kosoymiki/turboquant-compressor.git
cd turboquant-compressor

npm install
npm run build
npm run smoke:stdio
```

### Linux

```bash
git clone https://github.com/kosoymiki/turboquant-compressor.git
cd turboquant-compressor

npm install
npm run build
npm test -- --runInBand
npm run smoke:stdio
```

---

## MCP Host Configuration

Use the built server entrypoint after build:

```bash
npm run build
node dist/server.js
```

Example generic stdio MCP configuration:

```json
{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["/absolute/path/to/turboquant-compressor/dist/server.js"]
    }
  }
}
```

For host-specific profile generation, use:

```text
turboquant_cli_mcp_profile
```

Supported profile targets include:

```text
generic_stdio
claude_code
codex_cli
cursor
gemini_cli
opencode
crush
```

---

## Verification

Run the standard local release gate:

```bash
npm run build
npm test -- --runInBand
npm run smoke:stdio
npm run verify:release
```

Additional gates:

```bash
npm run verify:termux
npm run verify:mcp
npm run verify:opencl-claims
npm run verify:release-evidence
npm run verify:archives
npm run verify:artifact-parity
```

Package scripts include release paths:

```bash
npm run release:source
npm run release:portable
npm run release:all
```

---

## Benchmarking

Run local open-test benchmark and save results:

```bash
npm run open:test:local:save
```

Expected output location:

```text
bench/results/open-test-local-YYYYMMDD-HHMMSS.json
```

The current committed public local-corpus evidence shows:

```text
compression_ratio: 5.8759x–5.8955x
recall_at_5:       0.66–0.92
dimensions:        384
format_version:    3
```

---

## Evidence Discipline

This project treats claims as release artifacts, not marketing text.

A claim is acceptable in README only when it has one of the following:

* A committed benchmark artifact under `bench/results/`
* A committed forensic report under `forensics/`
* A conformance transcript
* A deterministic test/smoke command
* A release gate that fails when the claim is not backed

Current supported headline:

```text
5.5x+ measured local corpus compression
13 MCP tools verified by conformance transcript
Termux-first MCP stdio server
OpenCL/Adreno diagnostics and evidence gates
```

Claims requiring device evidence:

```text
Adreno production acceleration
OpenCL inference/compression claim
Snapdragon performance claim
Custom driver performance uplift
Sustained GPU throughput numbers
```

Do not publish those as verified unless the matching forensic artifacts exist and pass release gates.

---

## Known Limits

* Current public local benchmark path is LEVEL_0.
* Public local benchmark artifacts do not include QJL correction.
* `useQjl` in search does not apply QJL correction unless the database format stores the required payload.
* OpenCL/Adreno acceleration must be verified per device.
* MCP compatibility depends on host stdio behavior and SDK compatibility.
* Benchmark token counts may be approximate when derived from bytes.

These limits are intentional and should remain visible until closed by code and evidence.

---

## Development

```bash
npm install
npm run build
npm test -- --runInBand
npm run smoke:stdio
```

Useful checks:

```bash
npm run smoke:api
npm run smoke:mcp
npm run smoke:numeric
npm run verify:release
npm run verify:tests-determinism
npm run verify:no-placeholders
npm run verify:no-shell-executor
```

---

## Release Checklist

Before tagging a release:

```bash
npm run build
npm test -- --runInBand
npm run smoke:stdio
npm run verify:release
npm run verify:termux
npm run verify:opencl-claims
npm run verify:release-evidence
npm run release:all
```

For GPU/Adreno claims, also include:

```bash
npm run build:opencl
npm run bench:opencl
npm run verify:adreno-opencl
npm run verify:release-evidence
```

Release evidence should include:

```text
commit SHA
Node.js version
npm version
device model / SoC when applicable
Termux or Linux environment
MCP conformance transcript
benchmark result JSON
OpenCL probe report when claiming OpenCL
Adreno loader report when claiming Adreno
```

---

## Version History

### v4.1.2

* Custom Adreno `Rusticl/Freedreno/KGSL/Turnip` stack moved into a canonical release slice
* Real OpenCL self-tests for `mse_score`, `qjl_score`, `value_dequant`, and `fused_attention`
* Safe single-run benchmark evidence committed for the repo-local custom stack
* Canonical stack contracts added: stack map, hygiene baseline, runtime evidence chain, export checklist
* Mirror parity contract added via sync manifest
* Clean export path added via `npm run package:release-slice`
* Dead release lanes removed from export truth: legacy `native/adreno*`, generated build residue, stale fused kernel variants

### v4.0.1

* Termux-first MCP server packaging
* 13-tool MCP surface
* Local compressed vector database build/search
* Context-pack build/search
* Cache/cost analysis tools
* KV/cache analysis tool
* Backend/OpenCL/Adreno probe tools
* Release verification and evidence gates
* Public local benchmark artifacts showing 5.5x+ compression

---

## References

* Fast Walsh-Hadamard Transform / randomized rotation literature
* Lloyd-Max quantization literature
* Johnson-Lindenstrauss / quantized projection literature
* MCP stdio transport model
* Termux Android runtime constraints
* Mesa/Freedreno/Turnip/OpenCL driver ecosystem
* QJL (arXiv:2406.03482) and TurboQuant (arXiv:2504.19874) theory papers


 https://raw.githubusercontent.com/kosoymiki/turboquant-compressor/main/bench/results/open-test-local-20260514-220247.json
 https://raw.githubusercontent.com/kosoymiki/turboquant-compressor/main/bench/results/open-test-local-20260521-022922.json
 https://raw.githubusercontent.com/kosoymiki/turboquant-compressor/main/forensics/mcp-conformance-transcript.json 
 https://raw.githubusercontent.com/kosoymiki/turboquant-compressor/main/scripts/verify-release-evidence.mjs


---

## License

GPL-3.0-or-later
