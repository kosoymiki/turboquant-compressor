# Gemini CLI — MCP Setup

turboquant-compressor works as a stdio MCP server with Gemini CLI via `~/.gemini/settings.json`.

## Configuration

```json
{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["/path/to/turboquant-compressor/dist/server.js"]
    }
  }
}
```

## Support status

`documented` — Gemini CLI supports MCP stdio servers via settings.json. Not live-tested with turboquant-compressor.

## Notes

- All 13 tools are available once connected.
- Use `turboquant_cli_mcp_profile` with `host: "gemini_cli"` for the exact config snippet.
