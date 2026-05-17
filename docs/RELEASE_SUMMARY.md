# Release Summary — turboquant-compressor v4.0.0

Generated: 2026-05-15T11:11:36.168Z
Node: v25.8.2 | npm: 11.14.1

## Version

**3.2.0**

## Build

dist/server.js present. Build OK.

## MCP Tools (tools/list)

```
1. turboquant_cache_plan
2. turboquant_cli_mcp_profile
3. turboquant_compress
4. turboquant_context_pack_build
5. turboquant_context_pack_search
6. turboquant_cost_analyze
7. turboquant_prompt_cache_lint
8. turboquant_vector_search
```

## Test Suite

```
Test Suites: 26 passed, 26 total
Tests:       188 passed, 188 total
```

Status: **PASS**

## verify:release

```
[OK] package.json verification passed
[OK] format contract verified
[OK] scientific claims verified
[OK] tests are deterministic
[MCP] Starting MCP contract smoke
[OK] MCP smoke passed
[OK] cost/science claims verified
[OK] verify-mcp-conformance: pure JS, server built, all 8 tools referenced
[OK] verify-no-placeholders: 26 test files clean
[OK] verify-host-matrix: 1 verified (generic_stdio), 6 documented, 0 unverified
[OK] verify-tool-metadata: 8 tools scanned, 0 forbidden patterns
[OK] verify-no-shell-executor: 159 files scanned, no shell executor
[OK] verify-forensics-redaction: all secret patterns redacted
[GATE] running live mcp-conformance.mjs...

=== MCP Conformance Results ===

✓ initialize: PASS — serverInfo.name=turboquant-compressor
✓ initialized_notification: PASS — sent, no response expected
✓ tools/list: PASS — 8 tools
✓ tools/call:turboquant_compress: PASS — ok, 624b
✓ tools/call:turboquant_cost_analyze: PASS — ok, 710b
✓ tools/call:turboquant_cache_plan: PASS — ok, 1336b
✓ tools/call:turboquant_prompt_cache_lint: PASS — ok, 209b
✓ tools/call:turboquant_cli_mcp_profile: PASS — ok, 747b
✓ tools/call:turboquant_vector_search: PASS — two-step compress→search ok, 186b
✓ tools/call:turboquant_context_pack_build: PASS — manifest built
✓ tools/call:turboquant_context_pack_search: PASS — two-step build→search ok, 487b
✓ invalid_method: PASS — code=-32601
✓ malformed_json_does_not_crash: PASS — server still responds after malformed input
✓ stdout_purity: PASS — stderr lines=0, no secrets in stdout

Total: 14 passed, 0 failed
Transcript written to mcp-conformance-transcript.json
[OK] live mcp-conformance passed
[GATE] running live mcp-transcript.mjs...
Transcript written to /data/data/com.termux/files/home/.claude/tools/turboquant-compressor/forensics/mcp-stdio-transcript.jsonl
Total entries: 13, isError entries: 0
[OK] transcript verified: 13 entries, 0 isError, 0 timeouts
[OK] release verification suite passed
```

Status: **PASS**

## verify-host-matrix

```
[OK] verify-host-matrix: 1 verified (generic_stdio), 6 documented, 0 unverified
```

Status: **PASS**

## MCP Conformance

```
=== MCP Conformance Results ===

✓ initialize: PASS — serverInfo.name=turboquant-compressor
✓ initialized_notification: PASS — sent, no response expected
✓ tools/list: PASS — 8 tools
✓ tools/call:turboquant_compress: PASS — ok, 624b
✓ tools/call:turboquant_cost_analyze: PASS — ok, 710b
✓ tools/call:turboquant_cache_plan: PASS — ok, 1336b
✓ tools/call:turboquant_prompt_cache_lint: PASS — ok, 209b
✓ tools/call:turboquant_cli_mcp_profile: PASS — ok, 747b
✓ tools/call:turboquant_vector_search: PASS — two-step compress→search ok, 186b
✓ tools/call:turboquant_context_pack_build: PASS — manifest built
✓ tools/call:turboquant_context_pack_search: PASS — two-step build→search ok, 487b
✓ invalid_method: PASS — code=-32601
✓ malformed_json_does_not_crash: PASS — server still responds after malformed input
✓ stdout_purity: PASS — stderr lines=0, no secrets in stdout

Total: 14 passed, 0 failed
Transcript written to mcp-conformance-transcript.json

```

Status: **PASS**

## MCP Transcript

```
Transcript written to /data/data/com.termux/files/home/.claude/tools/turboquant-compressor/forensics/mcp-stdio-transcript.jsonl
Total entries: 13, isError entries: 0
```

## Smoke Tests

- stdio: **PASS**
- mcp: **PASS**
- numeric: **PASS**
- api: **PASS**

## Archives

- Source (`/data/data/com.termux/files/home/.claude/tools/turboquant-compressor/../turboquant-compressor-3.2.0-source.tar.gz`): **present** (SHA256: 2d2877c31291a164856ad7da6dfd31ef79e5d5de8cc4ef48b254bcb03f7d3d56)
- Portable (`/data/data/com.termux/files/home/.claude/tools/turboquant-compressor/../turboquant-compressor-3.2.0-portable.tar.gz`): **present** (SHA256: da837b0f413aa69da393e82410f42fc6a867a15f0772ef46a873661f401e9ce2)

## Archive Verification

- verify:archives: **PASS**
- verify:artifact-parity: **PASS**

## Overall Verdict

**READY** — All gates passed for v4.0.0.
