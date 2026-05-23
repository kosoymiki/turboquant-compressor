# Agent Donor Corpus 2026-05-22

This corpus layer pins two external donor systems into the local TurboQuant evidence base:

- `HKUDS/CLI-Anything`
- `colbymchenry/codegraph`

They are not treated as generic inspiration. They are treated as donor modules for two distinct capability families:

- `CLI-Anything`: agent-native CLI harness generation, registry/skill/plugin surfaces, preview and workflow contracts
- `codegraph`: semantic code indexing, MCP transport/tooling, installer/config surfaces, graph/query/search/sync runtime

## Canonical Intake

The canonical ingest script is:

```sh
cd /data/data/com.termux/files/home/tmp_turboquant
node scripts/ingest-agent-donor-corpus.mjs
```

It writes:

- [agent-donor-corpus-ingest-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/agent-donor-corpus-ingest-2026-05-22.json)
- [cli-anything-corpus-manifest-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/cli-anything-corpus-manifest-2026-05-22.json)
- [codegraph-corpus-manifest-2026-05-22.json](/data/data/com.termux/files/home/tmp_turboquant/forensics/codegraph-corpus-manifest-2026-05-22.json)

## Placement In Our Stack

`CLI-Anything` belongs in the corpus as the donor for:

- registry-driven agent CLI discovery
- skill/plugin/harness packaging layouts
- preview/workflow and install-first command surfacing
- codex/claude/opencode skill entry conventions

`codegraph` belongs in the corpus as the donor for:

- semantic code-intelligence indexing
- MCP server and tool contracts
- installer/config writer surfaces for agent integration
- graph/query/search/sync/watch pipelines

## Engineering Rule

These donor repos are first-class corpus inputs, not ad hoc bookmarks.
Any future integration must preserve:

- exact source URL
- local clone path
- entry files
- focus directory statistics
- file inventory manifest

This rule is not limited to agent-facing lanes.

It applies to every future frontier lane:

- OpenCL runtime lanes
- Mesa/Rusticl/Freedreno/KGSL driver lanes
- forensic/debug lanes
- release/packaging lanes

No new lane should start as a one-off fix surface without an explicit donor model in the corpus.

## Source URLs

- https://github.com/HKUDS/CLI-Anything
- https://github.com/colbymchenry/codegraph
