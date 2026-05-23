import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const archivePath = '/storage/emulated/0/Download/renderdoc-gpu-debug-portable.tar.gz';
const extractedRoot =
  '/data/data/com.termux/files/home/.cache/turboquant/renderdoc-gpu-debug-portable/renderdoc-gpu-debug';
const outPath =
  '/data/data/com.termux/files/home/tmp_turboquant/forensics/renderdoc-gpu-debug-corpus-ingest.json';

function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  const fd = fs.openSync(filePath, 'r');
  try {
    const buf = Buffer.allocUnsafe(1024 * 1024);
    while (true) {
      const read = fs.readSync(fd, buf, 0, buf.length, null);
      if (read === 0) break;
      hash.update(buf.subarray(0, read));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest('hex');
}

function walk(dir) {
  const results = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      results.push(...walk(full));
    } else if (ent.isFile()) {
      results.push(full);
    }
  }
  return results;
}

function rel(filePath) {
  return path.relative(extractedRoot, filePath);
}

function size(filePath) {
  return fs.statSync(filePath).size;
}

function parseMcpTools(serverText) {
  const lines = serverText.split('\n');
  const tools = [];
  for (let i = 0; i < lines.length; i += 1) {
    if (!lines[i].includes('@mcp.tool()')) continue;
    for (let j = i + 1; j < Math.min(lines.length, i + 6); j += 1) {
      const m = lines[j].match(/async def ([a-zA-Z0-9_]+)\(/);
      if (m) {
        tools.push(m[1]);
        break;
      }
    }
  }
  return tools;
}

function parseRecipeNames(recipesText) {
  return [...recipesText.matchAll(/^RECIPE_([A-Z0-9_]+)\s*=/gm)].map((m) => m[1]);
}

function parseSourceIndex(sourceIndex) {
  const sources = sourceIndex.sources || [];
  const byPriority = {};
  const byCategory = {};
  for (const source of sources) {
    byPriority[source.priority] = (byPriority[source.priority] || 0) + 1;
    byCategory[source.category] = (byCategory[source.category] || 0) + 1;
  }
  return {
    total: sources.length,
    byPriority,
    byCategory,
  };
}

function pickRelevantSources(sourceIndex) {
  const ids = new Set([
    'opencl-spec',
    'vulkan-spec',
    'linux-drm-docs',
    'android-sync-framework',
    'kgsl-uapi',
    'mesa-rusticl-docs',
    'mesa-freedreno-docs',
    'xdc-freedreno-android',
    'qualcomm-adreno-opencl',
    'egl-android-fence',
  ]);
  return (sourceIndex.sources || [])
    .filter((src) => ids.has(src.id))
    .map((src) => ({
      id: src.id,
      priority: src.priority,
      category: src.category,
      title: src.title,
      source_url: src.source_url,
      used_for: src.used_for || [],
    }));
}

const files = walk(extractedRoot).sort();
const extCounts = {};
for (const file of files) {
  const ext = path.extname(file).toLowerCase() || '<none>';
  extCounts[ext] = (extCounts[ext] || 0) + 1;
}

const serverPath = path.join(extractedRoot, 'mcp_server/server.py');
const recipesPath = path.join(extractedRoot, 'mcp_server/recipes.py');
const sourceIndexPath = path.join(extractedRoot, 'docs/research/source-index.json');
const corpusMasterPath = path.join(extractedRoot, 'docs/research/CORPUS_MASTER.md');
const bookIndexPath = path.join(extractedRoot, 'docs/research/book-corpus-index.json');
const commandsQuickRefPath = path.join(extractedRoot, 'references/commands-quick-ref.md');

const serverText = fs.readFileSync(serverPath, 'utf8');
const recipesText = fs.readFileSync(recipesPath, 'utf8');
const sourceIndex = JSON.parse(fs.readFileSync(sourceIndexPath, 'utf8'));
const bookIndex = JSON.parse(fs.readFileSync(bookIndexPath, 'utf8'));

const topLevelEntries = fs
  .readdirSync(extractedRoot, { withFileTypes: true })
  .map((ent) => ({
    name: ent.name,
    kind: ent.isDirectory() ? 'dir' : ent.isFile() ? 'file' : 'other',
  }));

const corpusFiles = files.filter((file) => file.includes('/docs/research/corpus/'));
const extractedCorpusRoots = files.filter((file) => file.includes('/docs/research/corpus/extracted/'));

const artifact = {
  generatedAt: new Date().toISOString(),
  archive: {
    path: archivePath,
    sizeBytes: size(archivePath),
    sha256: sha256File(archivePath),
  },
  extracted: {
    root: extractedRoot,
    totalFiles: files.length,
    totalBytes: files.reduce((acc, file) => acc + size(file), 0),
    topLevelEntries,
    extensionCounts: extCounts,
  },
  authorityModel: {
    corpusMasterPath: rel(corpusMasterPath),
    sourceAuthorityOrder:
      'Official specs > kernel source > Mesa source > device evidence > academic papers > community reports',
  },
  mcpServer: {
    root: 'mcp_server',
    tools: parseMcpTools(serverText),
    recipes: parseRecipeNames(recipesText),
    files: [serverPath, recipesPath, path.join(extractedRoot, 'mcp_server/rdc_runner.py')].map(rel),
  },
  corpus: {
    sourceIndexPath: rel(sourceIndexPath),
    sourceIndexSummary: parseSourceIndex(sourceIndex),
    bookCorpusMeta: bookIndex.meta,
    bookCorpusCount: Array.isArray(bookIndex.books) ? bookIndex.books.length : 0,
    corpusFilesCount: corpusFiles.length,
    extractedCorpusFilesCount: extractedCorpusRoots.length,
    relevantSources: pickRelevantSources(sourceIndex),
  },
  activeFrontierDonorMap: {
    mesa_command_buffer: {
      rationale:
        'RenderDoc portable corpus provides command-buffer, synchronization, and API/layer debugging donor material for validating OpenCL/Vulkan command recording and replay semantics.',
      sources: [
        'opencl-spec',
        'vulkan-spec',
        'linux-drm-docs',
        'mesa-rusticl-docs',
        'mesa-freedreno-docs',
        'xdc-freedreno-android',
      ],
      localEntries: [
        rel(commandsQuickRefPath),
        rel(sourceIndexPath),
        rel(corpusMasterPath),
        'docs/research/corpus/html/vkCmdDispatch.html',
        'docs/research/corpus/html/vk_synchronization.html',
        'docs/research/corpus/pdf/OpenCL_API_3.0.pdf',
        'docs/research/corpus/pdf/vkspec_1.4.pdf',
      ],
    },
    mesa_system_svm: {
      rationale:
        'Bundle contains OpenCL, Android sync, KGSL, kernel fence, and Mesa/Freedreno donor references needed to reason about deeper system/fine SVM capability boundaries.',
      sources: [
        'opencl-spec',
        'linux-drm-docs',
        'android-sync-framework',
        'kgsl-uapi',
        'mesa-rusticl-docs',
        'qualcomm-adreno-opencl',
      ],
      localEntries: [
        'docs/research/corpus/c/msm_kgsl.h',
        'docs/research/corpus/c/kgsl_sync.c',
        'docs/research/corpus/html/kernel_dmabuf.html',
        'docs/research/corpus/html/kernel_syncfile.html',
        'docs/research/corpus/html/android_sync_framework.html',
        'docs/research/corpus/pdf/OpenCL_API_3.0.pdf',
      ],
    },
  },
  ingestionAdvice: {
    nextLocalActions: [
      'Treat renderdoc-gpu-debug bundle as P0/P1 donor corpus for command-buffer, synchronization, KGSL, and Vulkan/OpenCL call-surface analysis.',
      'Use MCP server tool list and recipes as exact legacy entry map, not as production claims.',
      'Use source-index and corpus-master as authority filter before borrowing old implementation patterns.',
    ],
  },
};

fs.writeFileSync(outPath, `${JSON.stringify(artifact, null, 2)}\n`);
console.log(outPath);
