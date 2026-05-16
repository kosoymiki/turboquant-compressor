# Claude Code Setup for TurboQuant Compressor

## Overview

This guide explains how to configure TurboQuant Compressor as an MCP server in Claude Code.

## MCP Configuration

Claude Code supports MCP servers through JSON configuration. Add the following to your MCP config:

### Option 1: Local Development

```json
{
  "mcpServers": {
    "turboquant": {
      "command": "node",
      "args": ["/path/to/turboquant-compressor/dist/server.js"],
      "env": {}
    }
  }
}
```

### Option 2: Using claude mcp add-json

```bash
claude mcp add-json turboquant --node -- /path/to/turboquant-compressor/dist/server.js
```

## Tool Names

After successful configuration, Claude Code will discover these tools:

- `turboquant_compress`: Compress vectors
- `turboquant_vector_search`: Search compressed database

## Usage Examples

### Compress Vectors

```json
{
  "vectors": [[0.1, 0.2, 0.3, 0.4]],
  "dimensions": 4,
  "seed": 42,
  "bitsPerValue": 4
}
```

### Search

```json
{
  "query_vector": [0.1, 0.2, 0.3, 0.4],
  "compressed_database_b64": "VFFNQwEAAAAA...",
  "dimensions": 4,
  "top_k": 5,
  "metric": "cosine"
}
```

## Output Limits

Compressed databases can become large. Be aware of Claude Code's output limits when working with high-dimensional vectors (1536+ dimensions).

## Troubleshooting

### Tools Not Visible

If tools don't appear after configuration:
1. Check server startup: `node dist/server.js`
2. Verify MCP handshake: Server should respond to `initialize`
3. Check logs for schema errors

### Schema Errors

The server uses StandardSchema format. If you see schema-related errors:
1. Ensure you're using the canonical field names
2. Check that dimensions match between compression and search

## Security Notes

- The server runs locally with no network access
- Input validation is performed on all parameters
- CRC32 validation catches data corruption