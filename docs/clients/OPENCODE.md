# OpenCode — MCP Setup

turboquant-compressor works as a stdio MCP server with OpenCode via its top-level `mcp` config.

## Status: documented

This profile is documented compatibility only.
Mark verified only after live OpenCode MCP discovery transcript is recorded.

## Configuration

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "turboquant-compressor": {
      "type": "local",
      "command": ["node", "/path/to/turboquant-compressor/dist/server.js"],
      "enabled": true
    }
  }
}
```

Note: OpenCode uses top-level `"mcp"` config, not generic `"mcpServers"`.

## Notes

- All 13 tools are available once connected.
- Use `turboquant_cli_mcp_profile` with `host: "opencode"` for the exact config snippet.
