# Security — MCP Threat Model

## Threat surface

turboquant-compressor is a stdio MCP server. It receives JSON-RPC calls from a local host process.

## STRIDE analysis

| Threat | Vector | Mitigation |
|--------|--------|------------|
| Spoofing | Malicious tool name collision | Tool names are prefixed `turboquant_` — unique namespace |
| Tampering | Malformed JSON-RPC payload | Zod schema validation on all inputs; rejects unknown fields |
| Repudiation | No audit trail | stderr logging; forensic harness in `scripts/collect-forensics.sh` |
| Info disclosure | Secrets in stdout | No env vars read; no file system access; redaction tests in `test/security/` |
| DoS | Oversized vectors | `validateCompressionParams` enforces dimension/count limits |
| Elevation | Shell execution | No shell executor in server; no `exec`/`spawn` calls |

## Tool description hardening

Tool descriptions are short and factual. No hidden instructions, no prompt injection vectors, no client-specific bypass logic.

Rules enforced:
- No `YOLO` or `bypass-permissions` in server code
- No secrets printed to stdout or stderr
- No file-system write tools
- No shell executor
- Payload limits enforced at schema level (`top_k ≤ 100`, `dimensions ≤ 8192`, `maxItems` on arrays)

## Prompt injection risk

Tool metadata (name, description, inputSchema) is static and does not include user-controlled content. Risk: low.

## Tests

- `test/security/no-secret-output.test.ts` — compress/search/context-pack output contains no secret patterns
- `test/security/payload-limits.test.ts` — invalid/oversized inputs are rejected
- `test/security/redaction.test.ts` — forensic redaction patterns cover known secret formats
