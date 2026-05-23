#!/usr/bin/env node

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, '..');
const forensicsDir = path.join(rootDir, 'forensics');

const donors = [
  {
    id: 'cli-anything',
    label: 'HKUDS CLI-Anything',
    sourceUrl: 'https://github.com/HKUDS/CLI-Anything',
    localRoot: '/data/data/com.termux/files/home/.cache/tq-corpus-donors/CLI-Anything',
    entryFiles: [
      'README.md',
      'registry.json',
      'public_registry.json',
      'cli-anything-plugin/.claude-plugin/plugin.json',
      'cli-anything-plugin/HARNESS.md',
      'cli-anything-plugin/README.md',
      'cli-hub/README.md',
      'cli-hub/cli_hub/cli.py',
      'cli-hub/cli_hub/installer.py',
      'cli-hub/cli_hub/registry.py',
      'codex-skill/SKILL.md',
      'macrocli/SKILL.md',
      'skills/README.md',
    ],
    focusDirs: [
      'cli-anything-plugin',
      'cli-hub',
      'cli-hub-meta-skill',
      'codex-skill',
      'macrocli',
      'skills',
      'opencode-commands',
      'docs',
    ],
  },
  {
    id: 'codegraph',
    label: 'colbymchenry codegraph',
    sourceUrl: 'https://github.com/colbymchenry/codegraph',
    localRoot: '/data/data/com.termux/files/home/.cache/tq-corpus-donors/codegraph',
    entryFiles: [
      'README.md',
      'BUNDLING.md',
      'CLAUDE.md',
      'package.json',
      'src/index.ts',
      'src/bin/codegraph.ts',
      'src/context/index.ts',
      'src/graph/index.ts',
      'src/graph/queries.ts',
      'src/mcp/index.ts',
      'src/mcp/tools.ts',
      'src/installer/index.ts',
      'src/search/query-parser.ts',
      'docs/SEARCH_QUALITY_LOOP.md',
    ],
    focusDirs: [
      'src/bin',
      'src/context',
      'src/db',
      'src/extraction',
      'src/graph',
      'src/installer',
      'src/mcp',
      'src/resolution',
      'src/search',
      'src/sync',
      '__tests__',
      'docs',
      'scripts/agent-eval',
    ],
  },
];

mkdirSync(forensicsDir, { recursive: true });

function walkFiles(root, maxDepth = 4) {
  const out = [];
  function walk(dir, depth) {
    for (const ent of readdirSync(dir, { withFileTypes: true })) {
      if (ent.name === '.git' || ent.name === 'node_modules' || ent.name === '.next') continue;
      const abs = path.join(dir, ent.name);
      const rel = path.relative(root, abs);
      if (ent.isDirectory()) {
        if (depth < maxDepth) walk(abs, depth + 1);
        continue;
      }
      const stat = statSync(abs);
      out.push({ path: rel, bytes: stat.size });
    }
  }
  walk(root, 0);
  return out.sort((a, b) => a.path.localeCompare(b.path));
}

function collectEntryCards(root, entryFiles) {
  return entryFiles
    .filter((rel) => existsSync(path.join(root, rel)))
    .map((rel) => {
      const abs = path.join(root, rel);
      const text = readFileSync(abs, 'utf8');
      const preview = text.split('\n').slice(0, 12).join('\n');
      return {
        path: rel,
        bytes: statSync(abs).size,
        preview,
      };
    });
}

function collectFocusDirStats(root, focusDirs, files) {
  return focusDirs.map((dir) => {
    const prefix = `${dir}${dir.endsWith('/') ? '' : '/'}`;
    const scoped = files.filter((f) => f.path === dir || f.path.startsWith(prefix));
    return {
      path: dir,
      files: scoped.length,
      bytes: scoped.reduce((sum, f) => sum + f.bytes, 0),
      sample: scoped.slice(0, 12).map((f) => f.path),
    };
  });
}

const generatedAt = new Date().toISOString();
const summary = {
  generatedAt,
  donors: [],
};

for (const donor of donors) {
  if (!existsSync(donor.localRoot)) {
    summary.donors.push({
      id: donor.id,
      label: donor.label,
      sourceUrl: donor.sourceUrl,
      localRoot: donor.localRoot,
      status: 'missing',
    });
    continue;
  }

  const files = walkFiles(donor.localRoot, 4);
  const entryCards = collectEntryCards(donor.localRoot, donor.entryFiles);
  const focusDirStats = collectFocusDirStats(donor.localRoot, donor.focusDirs, files);
  const manifest = {
    generatedAt,
    id: donor.id,
    label: donor.label,
    sourceUrl: donor.sourceUrl,
    localRoot: donor.localRoot,
    status: 'ready',
    totals: {
      files: files.length,
      bytes: files.reduce((sum, f) => sum + f.bytes, 0),
    },
    entryCards,
    focusDirStats,
    files,
  };

  const manifestPath = path.join(forensicsDir, `${donor.id}-corpus-manifest-2026-05-22.json`);
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  summary.donors.push({
    id: donor.id,
    label: donor.label,
    sourceUrl: donor.sourceUrl,
    localRoot: donor.localRoot,
    status: 'ready',
    manifestPath,
    totals: manifest.totals,
  });
}

writeFileSync(
  path.join(forensicsDir, 'agent-donor-corpus-ingest-2026-05-22.json'),
  JSON.stringify(summary, null, 2),
);

console.log(JSON.stringify(summary, null, 2));
