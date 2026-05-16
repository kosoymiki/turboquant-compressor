# MCP Host Matrix

turboquant-compressor supports any MCP-capable host via stdio transport.

| Host | ID | Transport | Status |
|------|----|-----------|--------|
| Generic stdio | `generic_stdio` | stdio | verified |
| Claude Code | `claude_code` | stdio | documented |
| Codex CLI | `codex_cli` | stdio | documented |
| Cursor | `cursor` | stdio | documented |
| Gemini CLI | `gemini_cli` | stdio | documented |
| OpenCode | `opencode` | stdio | documented |
| Crush | `crush` | stdio | documented |

Use `turboquant_cli_mcp_profile` tool to get config snippet and smoke command for any host.

## Status definitions

- **verified**: live local smoke transcript exists in this repo/archive
- **documented**: official docs confirm MCP stdio support and config shape; local host smoke not run
- **unverified**: config inferred from MCP spec only; no official docs or local smoke

## Evidence

- `generic_stdio`: verified via `scripts/mcp-conformance.mjs` and `scripts/mcp-transcript.mjs` in this archive
- `claude_code`: documented — Claude Code supports MCP stdio servers via `~/.claude/settings.json`; no live `claude mcp` transcript in this archive
- `codex_cli`: documented — Codex CLI supports MCP via `~/.codex/config.toml` or `.codex/config.toml`; no live codex transcript
- `cursor`: documented — Cursor supports MCP stdio servers via `.cursor/mcp.json`; no live cursor transcript
- `gemini_cli`: documented — Gemini CLI supports MCP servers via `~/.gemini/settings.json` mcpServers; no live gemini transcript
- `opencode`: documented — OpenCode supports MCP via top-level `mcp` config; no live opencode transcript
- `crush`: documented — Crush supports MCP via JSON config (`.crush.json`, `crush.json`, `$HOME/.config/crush/crush.json`); version-sensitive; no live crush transcript

## Core invariant

The server is transport-agnostic. All host-specific config lives in `src/profiles/clients/`.
Claude Code is one client, not the primary target.
