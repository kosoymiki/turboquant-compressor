# Release Summary — turboquant-compressor v4.5.0

Generated: 2026-05-22T17:58:00Z
Node: v24.15.0 | npm: 11.14.1

## Version

**4.5.0**

## Build

dist/server.js present. Build OK.

## MCP Tools

```
1. turboquant_adreno_loader_probe
2. turboquant_backend_probe
3. turboquant_cache_plan
4. turboquant_cli_mcp_profile
5. turboquant_compress
6. turboquant_context_pack_build
7. turboquant_context_pack_search
8. turboquant_cost_analyze
9. turboquant_kv_analyze
10. turboquant_opencl_probe
11. turboquant_prompt_cache_lint
12. turboquant_quantize
13. turboquant_vector_search
```

## Test Suite

```
Test Suites: 50 passed, 50 total
Tests:       279 passed, 279 total
Snapshots:   0 total
Time:        312.129 s
```

Status: **PASS**

## verify:release

```text
[OK] package.json verification passed
[OK] format contract verified
[OK] scientific claims verified
[OK] tests are deterministic
[OK] verify-host-matrix: 1 verified (generic_stdio), 6 documented, 0 unverified
[OK] verify-tool-metadata: 13 tools scanned, 0 forbidden patterns
[OK] verify-no-shell-executor: 281 files scanned, no shell executor
[OK] verify-forensics-redaction: all secret patterns redacted
[OK] verify-termux-ready: quick probe 15ms, backend=diagnostic_only, no forbidden imports
[OK] verify-opencl-claims: 57 files scanned, no unsupported claims
[OK] verify-release-evidence: 2 required + 5 optional evidence files verified
[OK] live mcp-conformance passed
[OK] transcript verified: 18 entries, 0 isError, 0 timeouts
[OK] release verification suite passed
```

Status: **PASS**

## verify-host-matrix

```text
[OK] verify-host-matrix: 1 verified (generic_stdio), 6 documented, 0 unverified
```

Status: **PASS**

## MCP Conformance

```text
Total: 19 passed, 0 failed
Transcript written to forensics/mcp-conformance-transcript.json
```

Status: **PASS**

## MCP Transcript

```text
Transcript written to /data/data/com.termux/files/home/tmp_turboquant/forensics/mcp-stdio-transcript.jsonl
Total entries: 18, isError entries: 0
```

## Smoke Tests

- stdio: **PASS**
- mcp: **PASS**
- numeric: **PASS**
- api: **PASS**

## Archives

- Source (`/data/data/com.termux/files/home/turboquant-compressor-4.5.0-source.tar.gz`): **present** (SHA256: 3e830948f4d2c5f4b165cf8b87c32a89ef9a31bf164b7f7c48d68b6ee9b7f7f6)
- Portable (`/data/data/com.termux/files/home/turboquant-compressor-4.5.0-portable.tar.gz`): **present** (SHA256: de1707b6041cb6124db57194ee43e3ca76381096e53da7d53a9ff22e8eadfca7)

## Archive Verification

- verify:archives: **PASS**
- verify:artifact-parity: **PASS**

## Overall Verdict

**READY** — All gates passed for v4.5.0.
