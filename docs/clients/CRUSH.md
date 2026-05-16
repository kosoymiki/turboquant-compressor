# Crush — MCP Setup

turboquant-compressor works as a stdio MCP server with Crush via its JSON config.

## Status: documented

This profile is documented compatibility only and is version-sensitive.
Do not mark verified until:
- installed Crush version is recorded
- config path is confirmed (`.crush.json`, `crush.json`, or `$HOME/.config/crush/crush.json`)
- server appears as connected in Crush session
- all 8 tools are discoverable

## Configuration

Config location varies by Crush version. Try one of:
- `.crush.json` (project root)
- `crush.json` (project root)
- `$HOME/.config/crush/crush.json` (user config)

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

## Notes

- All 8 tools are available once connected.
- Use `turboquant_cli_mcp_profile` with `host: "crush"` for the exact config snippet.
