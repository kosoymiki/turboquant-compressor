# Codex CLI — MCP Setup

turboquant-compressor works as a stdio MCP server with Codex CLI via `~/.codex/config.toml`.

## Status: documented

This profile is documented compatibility only.
Mark verified only after live Codex CLI MCP discovery transcript is recorded.

## Configuration

```toml
[mcp_servers."turboquant-compressor"]
command = "node"
args = ["/path/to/turboquant-compressor/dist/server.js"]
```

Add to `~/.codex/config.toml` or `.codex/config.toml` in project root.

## Support status

`documented` — Codex CLI supports MCP stdio servers via config.toml. Not live-tested.

## Notes

- All 13 tools are available once connected.
- Use `turboquant_cli_mcp_profile` with `host: "codex_cli"` for the exact config snippet.
