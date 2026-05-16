# Cursor — MCP Setup

turboquant-compressor works as a stdio MCP server in Cursor via `.cursor/mcp.json`.

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

`documented` — Cursor supports MCP stdio servers. Not live-tested with turboquant-compressor.

## Notes

- Use `turboquant_cli_mcp_profile` with `host: "cursor"` for the exact config snippet.
