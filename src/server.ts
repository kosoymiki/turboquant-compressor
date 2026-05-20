import { McpServer, StdioServerTransport, fromJsonSchema } from '@modelcontextprotocol/server';
import { readFileSync } from 'node:fs';
import { initWasmBackend, isWasmReady } from './native/wasm_backend.js';
import { compressVectors } from './tools/compress.js';
import { searchVectors } from './tools/search.js';
import { parseCompressInput, parseSearchInput } from './tools/validation.js';
import { turboquantCostAnalyze } from './tools/cost_analyze.js';
import { turboquantCachePlan } from './tools/cache_plan.js';
import { turboquantPromptCacheLint } from './tools/prompt_cache_lint.js';
import { turboquantContextPackBuild } from './tools/context_pack_build.js';
import { turboquantContextPackSearch } from './tools/context_pack_search.js';
import { getCliProfile, listSupportedHosts } from './tools/cli_profile.js';
import { turboquantQuantize, turboquantQuantizeSchema } from './tools/turboquant_quantize.js';
import { turboquantKvAnalyze, turboquantKvAnalyzeSchema } from './tools/turboquant_kv_analyze.js';
import { turboquantBackendProbe, turboquantBackendProbeSchema } from './tools/turboquant_backend_probe.js';
import { turboquantOpenclProbe, turboquantOpenclProbeSchema } from './tools/turboquant_opencl_probe.js';
import { turboquantAdrenoLoaderProbe, turboquantAdrenoLoaderProbeSchema } from './tools/turboquant_adreno_loader_probe.js';

function readServerVersion(): string {
  try {
    const packageJsonPath = new URL('../package.json', import.meta.url);
    const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8')) as { version?: string };
    return packageJson.version ?? '4.1.0';
  } catch {
    return '4.1.0';
  }
}

type CompressToolArgs = {
  vectors: number[][];
  dimensions?: number;
  seed?: number;
  rotationSeed?: number;
  bitsPerValue?: 2 | 3 | 4 | 8;
  includeQJL?: boolean;
};

type SearchToolArgs = {
  query_vector?: number[];
  compressed_database_b64?: string;
  dimensions?: number;
  top_k?: number;
  metric?: 'cosine' | 'dot';
  compressedData?: string;
  queryVector?: number[];
  k?: number;
  useQjl?: boolean;
};

const server = new McpServer(
  { name: 'turboquant-compressor', version: readServerVersion() },
  { capabilities: {} }
);

const CompressInputSchema = fromJsonSchema({
  type: 'object',
  properties: {
    vectors: {
      type: 'array',
      items: {
        type: 'array',
        items: { type: 'number' },
        minItems: 1
      }
    },
    dimensions: { type: 'number', minimum: 1, maximum: 8192 },
    seed: { type: 'number' },
    bitsPerValue: { type: 'number', enum: [2, 3, 4, 8] },
    includeQJL: { type: 'boolean' }
  },
  required: ['vectors']
});

const SearchInputSchema = fromJsonSchema({
  type: 'object',
  properties: {
    query_vector: {
      type: 'array',
      items: { type: 'number' },
      minItems: 1
    },
    compressed_database_b64: { type: 'string' },
    dimensions: { type: 'number', minimum: 1, maximum: 8192 },
    top_k: { type: 'number', minimum: 1, maximum: 100 },
    metric: { type: 'string', enum: ['cosine', 'dot'] },
    compressedData: { type: 'string' },
    queryVector: {
      type: 'array',
      items: { type: 'number' }
    },
    k: { type: 'number', minimum: 1, maximum: 100 },
    useQjl: { type: 'boolean' }
  },
  required: []
});

server.registerTool(
  'turboquant_compress',
  {
    title: 'TurboQuant Compress',
    description: 'Compress vectors using TurboQuant-inspired compression',
    inputSchema: CompressInputSchema
  },
  async (args: unknown) => {
    const typed = parseCompressInput(args);
    const seed = typed.seed ?? typed.rotationSeed ?? 0;

    const result = compressVectors({
      vectors: typed.vectors,
      dimensions: typed.dimensions,
      bitsPerValue: typed.bitsPerValue,
      rotationSeed: seed,
      includeQJL: typed.includeQJL,
    });
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(result)
      }]
    };
  }
);

server.registerTool(
  'turboquant_vector_search',
  {
    title: 'TurboQuant Vector Search',
    description: 'Search for nearest neighbors in compressed vector database',
    inputSchema: SearchInputSchema
  },
  async (args: unknown) => {
    const typed = parseSearchInput(args);
    const compressedDataFinal = typed.compressed_database_b64 ?? typed.compressedData;
    const queryVectorFinal = typed.query_vector ?? typed.queryVector;
    const kFinal = typed.top_k ?? typed.k ?? 5;
    const metricFinal = typed.metric ?? 'cosine';

    if (!compressedDataFinal || !queryVectorFinal) {
      throw new Error('Missing required fields: query_vector and compressed_database_b64');
    }

    const result = searchVectors({
      compressed_database_b64: compressedDataFinal,
      query_vector: queryVectorFinal,
      dimensions: typed.dimensions,
      top_k: kFinal,
      metric: metricFinal,
      useQjl: typed.useQjl,
    });

    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(result)
      }]
    };
  }
);

const CostAnalyzeInputSchema = fromJsonSchema({
  type: 'object',
  properties: {
    usage: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          inputTokens: { type: 'number', minimum: 0 },
          outputTokens: { type: 'number', minimum: 0 },
          cacheCreationInputTokens: { type: 'number', minimum: 0 },
          cacheReadInputTokens: { type: 'number', minimum: 0 },
          totalCostUsdEstimate: { type: 'number' },
          sessionId: { type: 'string' },
          model: { type: 'string' },
          timestamp: { type: 'string' }
        },
        required: ['inputTokens', 'outputTokens', 'cacheCreationInputTokens', 'cacheReadInputTokens']
      },
      minItems: 1,
      maxItems: 10000
    }
  },
  required: ['usage']
});

server.registerTool(
  'turboquant_cost_analyze',
  {
    title: 'TurboQuant Cost Analyze',
    description: 'Analyze Claude Code usage snapshots and compute cache efficiency metrics',
    inputSchema: CostAnalyzeInputSchema
  },
  async (args: unknown) => {
    const result = turboquantCostAnalyze(args);
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(result)
      }]
    };
  }
);

const CachePlanInputSchema = fromJsonSchema({
  type: 'object',
  properties: {
    blocks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          label: { type: 'string' },
          text: { type: 'string' },
          volatilityHint: { type: 'string' }
        },
        required: ['label', 'text']
      },
      minItems: 1,
      maxItems: 512
    },
    target: { type: 'string' }
  },
  required: ['blocks']
});

server.registerTool(
  'turboquant_cache_plan',
  {
    title: 'TurboQuant Cache Plan',
    description: 'Classify context blocks by volatility and recommend cache tiers',
    inputSchema: CachePlanInputSchema
  },
  async (args: unknown) => {
    const result = turboquantCachePlan(args);
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(result)
      }]
    };
  }
);

const PromptCacheLintInputSchema = fromJsonSchema({
  type: 'object',
  properties: {
    blocks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          label: { type: 'string' },
          text: { type: 'string' }
        },
        required: ['label', 'text']
      },
      minItems: 1,
      maxItems: 512
    }
  },
  required: ['blocks']
});

server.registerTool(
  'turboquant_prompt_cache_lint',
  {
    title: 'TurboQuant Prompt Cache Lint',
    description: 'Detect cache-busting patterns in context blocks',
    inputSchema: PromptCacheLintInputSchema
  },
  async (args: unknown) => {
    const result = turboquantPromptCacheLint(args);
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(result)
      }]
    };
  }
);

const ContextPackBuildInputSchema = fromJsonSchema({
  type: 'object',
  properties: {
    files: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          text: { type: 'string' }
        },
        required: ['path', 'text']
      },
      minItems: 1,
      maxItems: 512
    },
    dimensions: { type: 'number', minimum: 1, maximum: 8192 },
    chunkBytes: { type: 'number', minimum: 256, maximum: 65536 },
    bitsPerValue: { type: 'number', enum: [2, 3, 4, 8] },
    storageMode: { type: 'string', enum: ['inline_text', 'preview_only', 'external_store'] },
    maxInlineChunkBytes: { type: 'number', minimum: 0, maximum: 65536 }
  },
  required: ['files']
});

server.registerTool(
  'turboquant_context_pack_build',
  {
    title: 'TurboQuant Context Pack Build',
    description: 'Build a compressed context pack from files for local retrieval',
    inputSchema: ContextPackBuildInputSchema
  },
  async (args: unknown) => {
    const result = turboquantContextPackBuild(args);
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(result)
      }]
    };
  }
);

const ContextPackSearchInputSchema = fromJsonSchema({
  type: 'object',
  properties: {
    manifest: { type: 'object' },
    query: { type: 'string' },
    top_k: { type: 'number', minimum: 1, maximum: 100 },
    externalStore: {
      type: 'object',
      properties: {
        kind: { type: 'string', enum: ['inline_entries'] },
        entries: {
          type: 'array',
          maxItems: 1000,
          items: {
            type: 'object',
            properties: {
              chunk_id: { type: 'string' },
              fingerprint: { type: 'string' },
              text: { type: 'string', maxLength: 1000000 }
            },
            required: ['chunk_id', 'fingerprint', 'text']
          }
        }
      },
      required: ['kind', 'entries']
    }
  },
  required: ['manifest', 'query']
});

server.registerTool(
  'turboquant_context_pack_search',
  {
    title: 'TurboQuant Context Pack Search',
    description: 'Search a context pack for relevant chunks',
    inputSchema: ContextPackSearchInputSchema
  },
  async (args: unknown) => {
    const result = turboquantContextPackSearch(args);
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(result)
      }]
    };
  }
);

const CliProfileInputSchema = fromJsonSchema({
  type: 'object',
  properties: {
    host: {
      type: 'string',
      enum: ['generic_stdio', 'claude_code', 'codex_cli', 'cursor', 'gemini_cli', 'opencode', 'crush']
    },
    includeSmokeTest: { type: 'boolean' }
  },
  required: ['host']
});

server.registerTool(
  'turboquant_cli_mcp_profile',
  {
    title: 'TurboQuant CLI MCP Profile',
    description: 'Return MCP client profile for a given host: config snippet, transport, smoke command, support status. Hosts: generic_stdio, claude_code, codex_cli, cursor, gemini_cli, opencode, crush.',
    inputSchema: CliProfileInputSchema
  },
  async (args: unknown) => {
    const input = args as { host: string; includeSmokeTest?: boolean };
    const result = getCliProfile(input);
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify(result)
      }]
    };
  }
);

server.registerTool(
  'turboquant_quantize',
  {
    title: 'TurboQuant Quantize',
    description: turboquantQuantizeSchema.description,
    inputSchema: fromJsonSchema(turboquantQuantizeSchema.inputSchema)
  },
  async (args: unknown) => {
    const result = turboquantQuantize(args as any);
    return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
  }
);

server.registerTool(
  'turboquant_kv_analyze',
  {
    title: 'TurboQuant KV Analyze',
    description: turboquantKvAnalyzeSchema.description,
    inputSchema: fromJsonSchema(turboquantKvAnalyzeSchema.inputSchema)
  },
  async (args: unknown) => {
    const result = turboquantKvAnalyze(args as any);
    return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
  }
);

server.registerTool(
  'turboquant_backend_probe',
  {
    title: 'TurboQuant Backend Probe',
    description: turboquantBackendProbeSchema.description,
    inputSchema: fromJsonSchema(turboquantBackendProbeSchema.inputSchema)
  },
  async (args: unknown) => {
    const result = turboquantBackendProbe(args as any);
    return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
  }
);

server.registerTool(
  'turboquant_opencl_probe',
  {
    title: 'TurboQuant OpenCL Probe',
    description: turboquantOpenclProbeSchema.description,
    inputSchema: fromJsonSchema(turboquantOpenclProbeSchema.inputSchema)
  },
  async (args: unknown) => {
    const result = turboquantOpenclProbe(args as any);
    return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
  }
);

server.registerTool(
  'turboquant_adreno_loader_probe',
  {
    title: 'TurboQuant Adreno Loader Probe',
    description: turboquantAdrenoLoaderProbeSchema.description,
    inputSchema: fromJsonSchema(turboquantAdrenoLoaderProbeSchema.inputSchema)
  },
  async (args: unknown) => {
    const result = turboquantAdrenoLoaderProbe(args as any);
    return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
  }
);

// Auto-init WASM backend (non-blocking, falls back to TS)
initWasmBackend().then(ok => {
  if (process.env.TURBOQUANT_MCP_QUIET === '1') return;
  if (ok) process.stderr.write('[turboquant] WASM SIMD128 backend: ready (17KB)\n');
  else process.stderr.write('[turboquant] WASM unavailable, using TS fallback\n');
}).catch(() => {});

const transport = new StdioServerTransport();
server.connect(transport).catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
