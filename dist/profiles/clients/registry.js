/**
 * MCP client profile registry.
 */
const ALL_TOOLS = [
    'turboquant_compress',
    'turboquant_vector_search',
    'turboquant_cost_analyze',
    'turboquant_cache_plan',
    'turboquant_prompt_cache_lint',
    'turboquant_context_pack_build',
    'turboquant_context_pack_search',
    'turboquant_cli_mcp_profile',
];
export const genericStdioProfile = {
    id: 'generic_stdio',
    name: 'Generic stdio MCP client',
    transport: 'stdio',
    configSnippet: `{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["/path/to/turboquant-compressor/dist/server.js"]
    }
  }
}`,
    smokeCommand: 'printf \'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}\\n\' | node dist/server.js',
    expectedTools: ALL_TOOLS,
    warnings: [],
    supportStatus: 'verified',
};
export const claudeCodeProfile = {
    id: 'claude_code',
    name: 'Claude Code',
    transport: 'stdio',
    configSnippet: `{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["/path/to/turboquant-compressor/dist/server.js"]
    }
  }
}`,
    smokeCommand: 'claude mcp add turboquant-compressor node /path/to/turboquant-compressor/dist/server.js',
    expectedTools: ALL_TOOLS,
    warnings: [
        'Add to ~/.claude/settings.json under mcpServers',
        'Ensure server is built (npm run build) before first use',
    ],
    supportStatus: 'documented',
};
export const codexCliProfile = {
    id: 'codex_cli',
    name: 'OpenAI Codex CLI',
    transport: 'stdio',
    configSnippet: `[mcp_servers."turboquant-compressor"]
command = "node"
args = ["/path/to/turboquant-compressor/dist/server.js"]`,
    smokeCommand: 'codex --mcp turboquant-compressor',
    expectedTools: ALL_TOOLS,
    warnings: [
        'Add to ~/.codex/config.toml as [mcp_servers."turboquant-compressor"]',
        'Ensure server is built before first use',
    ],
    supportStatus: 'documented',
};
export const cursorProfile = {
    id: 'cursor',
    name: 'Cursor',
    transport: 'stdio',
    configSnippet: `{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["/path/to/turboquant-compressor/dist/server.js"]
    }
  }
}`,
    smokeCommand: 'cursor --mcp turboquant-compressor',
    expectedTools: ALL_TOOLS,
    warnings: [
        'Add to .cursor/mcp.json in project root or user settings',
        'Ensure server is built before first use',
    ],
    supportStatus: 'documented',
};
export const geminiCliProfile = {
    id: 'gemini_cli',
    name: 'Gemini CLI',
    transport: 'stdio',
    configSnippet: `{
  "mcpServers": {
    "turboquant-compressor": {
      "command": "node",
      "args": ["/path/to/turboquant-compressor/dist/server.js"]
    }
  }
}`,
    smokeCommand: 'gemini --mcp turboquant-compressor',
    expectedTools: ALL_TOOLS,
    warnings: [
        'Add to ~/.gemini/settings.json under mcpServers',
        'Ensure server is built before first use',
    ],
    supportStatus: 'documented',
};
export const opencodeProfile = {
    id: 'opencode',
    name: 'OpenCode',
    transport: 'stdio',
    configSnippet: `{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "turboquant-compressor": {
      "type": "local",
      "command": ["node", "/path/to/turboquant-compressor/dist/server.js"],
      "enabled": true
    }
  }
}`,
    smokeCommand: 'opencode --help # verify MCP tools appear in OpenCode session',
    expectedTools: ALL_TOOLS,
    warnings: [
        'OpenCode uses top-level "mcp" config, not generic "mcpServers".',
        'Documented compatibility only — mark verified after live OpenCode MCP discovery transcript.',
        'Ensure server is built before first use.',
    ],
    supportStatus: 'documented',
};
export const crushProfile = {
    id: 'crush',
    name: 'Crush',
    transport: 'stdio',
    configSnippet: `# Crush MCP config is version-sensitive.
# Do not copy a generic mcpServers snippet.
# Run: crush --help, crush mcp --help, crush config --help
# Inspect active config path: .crush.json, crush.json, or $HOME/.config/crush/crush.json
# Only mark verified after:
#   1. Installed Crush version recorded
#   2. Exact config file recorded
#   3. turboquant server starts
#   4. All 8 tools discoverable or host limitation documented`,
    smokeCommand: 'crush --help # verify MCP tools appear in Crush session',
    expectedTools: ALL_TOOLS,
    warnings: [
        'Documented compatibility only — exact Crush MCP config schema is version-sensitive.',
        'Run `crush mcp` or check installed Crush version docs before writing config.',
        'Config may be .crush.json, crush.json, or $HOME/.config/crush/crush.json.',
        'Mark verified only after live Crush config path + tool discovery transcript.',
        'Ensure server is built before first use.',
    ],
    supportStatus: 'documented',
};
export const clientRegistry = {
    generic_stdio: genericStdioProfile,
    claude_code: claudeCodeProfile,
    codex_cli: codexCliProfile,
    cursor: cursorProfile,
    gemini_cli: geminiCliProfile,
    opencode: opencodeProfile,
    crush: crushProfile,
};
export function getClientProfile(id) {
    return clientRegistry[id];
}
export function getClientIds() {
    return Object.keys(clientRegistry);
}
