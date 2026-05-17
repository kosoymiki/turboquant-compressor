# Release Policy

Release artifacts:

- `*-source.tar.gz`: source only; no node_modules, no dist.
- `*-portable.tar.gz`: includes node_modules and dist for offline Termux transfer.
- Never include `.git`, coverage, nested tarballs, secrets, or local caches.

Required gates:

```bash
rm -rf dist
npm run build
npm test -- --runInBand
npm run smoke:mcp
npm run smoke:numeric
npm run smoke:stdio
npm run package:source
npm run package:portable
```

Versioning:

- package version: semantic release version.
- binary format version: independent integer.
- server version must match package version.

## MCP SDK policy

v3.2.x pins `@modelcontextprotocol/server@2.0.0-alpha.2`.

Any MCP SDK dependency change requires:

```bash
npm run build
npm run smoke:mcp
npm run smoke:stdio
```

A migration spike to a stable MCP SDK package is planned for v4.0.0.

## Dependency notes

`@cfworker/json-schema` is supplied transitively by `@modelcontextprotocol/server` if needed; it is not kept as a direct dependency.