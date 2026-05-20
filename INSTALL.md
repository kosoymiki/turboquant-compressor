# TurboQuant v4.1.0 — Installation Guide

## Scope

This package is a local MCP server for:

- compressed vector search
- context-pack build/search
- KV/cache analysis
- Termux/OpenCL/Adreno diagnostics

It is **not** a general-purpose installer for Claude hooks or external Python wrappers.

For a clean export artifact that excludes generated residue and dead release lanes, use:

```bash
npm run package:release-slice
```

## Verified prerequisites

- Node.js 22+
- npm 10+
- Termux or Linux shell
- optional: Adreno 7xx/8xx device for loader/OpenCL diagnostics

## Install

```bash
git clone https://github.com/kosoymiki/turboquant-compressor.git
cd turboquant-compressor

npm install
npm run build
```

## Core verification

```bash
npm run smoke:api
npm run smoke:mcp
npm run smoke:numeric
npm run smoke:stdio
npm run verify:release
```

## Termux / Adreno verification

```bash
npm run verify:adreno-loader
npm run verify:adreno-opencl
npm run verify:opencl-claims
npm run verify:termux
```

If a device-specific OpenCL route is not proven, the product must be treated as:

- diagnostics-ready
- not performance-claim-ready

## Benchmarks

Run and save the local open-test benchmark:

```bash
npm run open:test:local:save
```

Artifacts are written to:

```text
bench/results/open-test-local-YYYYMMDD-HHMMSS.json
```

Current committed public artifacts support:

- `5.5x+` measured local corpus compression
- best committed compression artifact: `5.8949x`
- committed recall@5 range: `0.06–0.92`

Do not claim stronger retrieval quality or throughput without committed artifacts.

## GPU / driver evidence

Relevant forensic artifacts live under:

```text
forensics/
forensics/adreno/
```

Important examples:

- `forensics/opencl-adreno-report.json`
- `forensics/adreno/loader-report.json`
- `forensics/RELEASE_EVIDENCE_MANIFEST.json`
- `forensics/mesa/driver-mesh-recovery-ready.json`
- `artifacts/safe_benchmark_2026-05-20.json`

These support readiness and claim-gating. They do **not** by themselves prove universal production acceleration.

## MCP usage

Build and run:

```bash
npm run build
node dist/server.js
```

Example generic stdio MCP config:

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

## Troubleshooting

### Missing dependencies

If `verify:release` or `npm test` fail with missing packages such as `zod` or `jest`, reinstall deps:

```bash
rm -rf node_modules
npm install
```

### OpenCL not available

Run:

```bash
npm run verify:adreno-loader
npm run verify:adreno-opencl
npm run probe:opencl
```

Android linker namespace isolation may block vendor `libOpenCL.so`; this product includes diagnostics for that case and must not overclaim production OpenCL when the route is not proven.

## Hard documentation rules

- Do not reference non-existent helpers like `forensics_final.py`, `tests_final.py`, or `bd_mcp_final.py`.
- Do not reference `~/.claude/hooks` as part of the supported install path.
- Do not publish unsupported throughput claims.
- Keep README / INSTALL / package metadata aligned with committed benchmark and forensic artifacts.
