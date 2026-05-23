#!/usr/bin/env node

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, '..');
const forensicsDir = path.join(rootDir, 'forensics');

const donors = [
  {
    id: 'mesa-local-system-svm',
    label: 'Local Mesa Rusticl/Freedreno/KGSL system SVM lane',
    sourceUrl: 'local:/data/data/com.termux/files/home/mesa-upstream-26.2-devel',
    localRoot: '/data/data/com.termux/files/home/mesa-upstream-26.2-devel',
    entryFiles: [
      'src/gallium/frontends/rusticl/api/kernel.rs',
      'src/gallium/frontends/rusticl/core/context.rs',
      'src/gallium/frontends/rusticl/core/device.rs',
      'src/gallium/frontends/rusticl/core/kernel.rs',
      'src/gallium/frontends/rusticl/core/platform.rs',
      'src/gallium/frontends/rusticl/mesa/pipe/resource.rs',
      'src/gallium/frontends/rusticl/mesa/pipe/screen.rs',
      'src/gallium/drivers/freedreno/freedreno_resource.c',
      'src/gallium/drivers/freedreno/freedreno_screen.c',
      'src/freedreno/drm/kgsl/kgsl_bo.c',
      'src/freedreno/drm/kgsl/kgsl_priv.h',
    ],
    focusDirs: [
      'src/gallium/frontends/rusticl/api',
      'src/gallium/frontends/rusticl/core',
      'src/gallium/frontends/rusticl/mesa/pipe',
      'src/gallium/drivers/freedreno',
      'src/freedreno/drm/kgsl',
    ],
  },
  {
    id: 'pocl-system-svm',
    label: 'PoCL system SVM donor',
    sourceUrl: 'https://github.com/pocl/pocl',
    localRoot: '/data/data/com.termux/files/home/.cache/tq-svm-donors/pocl',
    entryFiles: [
      'README.md',
      'lib/CL/clSetKernelExecInfo.c',
      'lib/CL/pocl_ndrange_kernel.c',
      'lib/CL/pocl_mem_management.c',
      'lib/CL/pocl_cl.h',
      'lib/llvmopencl/SVMOffset.cc',
      'lib/llvmopencl/SVMOffset.hh',
      'lib/CL/devices/common_utils.c',
      'lib/CL/devices/level0/pocl-level0.cc',
      'lib/CL/devices/hsa/pocl-hsa.c',
      'lib/CL/devices/cuda/pocl-cuda.c',
      'lib/CL/devices/basic/basic.c',
    ],
    focusDirs: [
      'lib/CL',
      'lib/CL/devices',
      'lib/kernel',
      'lib/llvmopencl',
    ],
  },
  {
    id: 'clvk-svm-reference',
    label: 'clvk SVM reference',
    sourceUrl: 'https://github.com/kpet/clvk',
    localRoot: '/data/data/com.termux/files/home/.cache/tq-svm-donors/clvk',
    entryFiles: [
      'README.md',
      'src/api.cpp',
      'src/exports.map',
    ],
    focusDirs: [
      'src',
      'tests',
      'docs',
    ],
  },
];

mkdirSync(forensicsDir, { recursive: true });

function gitHead(root) {
  try {
    return execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return null;
  }
}

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
      const preview = text.split('\n').slice(0, 16).join('\n');
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
      sample: scoped.slice(0, 16).map((f) => f.path),
    };
  });
}

const generatedAt = new Date().toISOString();
const summary = {
  generatedAt,
  lane: 'mesa_system_svm',
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
    lane: 'mesa_system_svm',
    id: donor.id,
    label: donor.label,
    sourceUrl: donor.sourceUrl,
    localRoot: donor.localRoot,
    gitHead: gitHead(donor.localRoot),
    status: 'ready',
    totals: {
      files: files.length,
      bytes: files.reduce((sum, f) => sum + f.bytes, 0),
    },
    entryCards,
    focusDirStats,
    files,
  };

  const manifestPath = path.join(forensicsDir, `${donor.id}-system-svm-corpus-manifest-2026-05-22.json`);
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  summary.donors.push({
    id: donor.id,
    label: donor.label,
    sourceUrl: donor.sourceUrl,
    localRoot: donor.localRoot,
    gitHead: manifest.gitHead,
    status: 'ready',
    manifestPath,
    totals: manifest.totals,
  });
}

writeFileSync(
  path.join(forensicsDir, 'system-svm-donor-corpus-ingest-2026-05-22.json'),
  JSON.stringify(summary, null, 2),
);

console.log(JSON.stringify(summary, null, 2));
