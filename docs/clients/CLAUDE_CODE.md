# Claude Code — MCP Setup

turboquant-compressor works as a stdio MCP server in Claude Code.

## Configuration

Add to `~/.claude/settings.json` under `mcpServers`:

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

## Available tools

- `turboquant_compress` — compress vectors
- `turboquant_vector_search` — nearest-neighbor search
- `turboquant_cost_analyze` — analyze Claude Code usage JSONL
- `turboquant_cache_plan` — classify context blocks by volatility
- `turboquant_prompt_cache_lint` — detect cache-busting patterns
- `turboquant_context_pack_build` — build compressed context pack
- `turboquant_context_pack_search` — search context pack
- `turboquant_cli_mcp_profile` — get config for any supported host

## Support status

`documented` — Claude Code supports MCP stdio servers. Mark verified only after live `claude mcp list` transcript is recorded in this archive.

## Notes

- Claude Code is one client. The server is not Claude-specific.
- Cost analysis tools accept `claude_code_usage_jsonl` format.
- See `docs/COST_MODEL_BOUNDARY.md` for billing scope.
