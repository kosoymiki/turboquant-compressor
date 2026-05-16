# Generic stdio — MCP Setup

Any MCP-capable host that supports stdio transport can connect to turboquant-compressor.

## Minimal invocation

```bash
node /path/to/turboquant-compressor/dist/server.js
```

The server speaks JSON-RPC 2.0 over stdin/stdout. stderr is reserved for logging only.

## Protocol

- `initialize` → capabilities negotiation
- `tools/list` → returns all 8 tools with JSON Schema
- `tools/call` → invoke any tool by name

## Support status

`verified` — raw stdio smoke tested via `scripts/mcp-conformance.mjs`.
