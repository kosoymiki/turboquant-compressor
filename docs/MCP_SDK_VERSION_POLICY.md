# MCP SDK Version Policy

## Current Status

This release targets `@modelcontextprotocol/server@2.0.0-alpha.2`.

## Official SDK Guidance

The official TypeScript SDK repository states:
- v2 is currently pre-alpha
- v1.x remains the recommended production version until stable v2

## Implications

This project is:
- Release-forensic ready
- Open-beta ready
- Not SDK-stability production final

## Protection Measures

We pin the exact alpha version and protect behavior with runtime MCP smoke tests:
- initialize
- tools/list
- tools/call
- schema assertions
- negative invalid input calls

## Future Roadmap

- **v3.2.x**: Keep v2 alpha with clear documentation
- **v3.3.0**: Consider migration to v1.x production SDK if v2 is still not stable
- **v4.0.0**: Target stable v2 SDK when officially released

## Recommendation

Use this project for:
- Local development and testing
- Cost-aware context compression workflows
- MCP integration experiments

For production Claude Code deployments requiring SDK stability guarantees, consider:
- Using v1.x MCP SDK directly
- Waiting for official v2 stable release
- Implementing additional isolation layers