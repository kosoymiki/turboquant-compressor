export type CacheTier =
  | 'static_1h_candidate'
  | 'session_5m_candidate'
  | 'dynamic_uncached'
  | 'context_pack_candidate';

export type CostLayer =
  | 'claude_code_prompt_cache'
  | 'local_mcp_vector_compression'
  | 'self_hosted_kv_cache';

export type WasteSignalType =
  | 'write_heavy'
  | 'low_cache_reuse'
  | 'context_bloat'
  | 'output_heavy';

export type WasteSignalSeverity = 'low' | 'medium' | 'high';

export interface UsageSnapshot {
  source: 'ccusage_json' | 'claude_jsonl' | 'manual';
  inputTokens: number;
  outputTokens: number;
  cacheCreationInputTokens: number;
  cacheReadInputTokens: number;
  totalCostUsdEstimate?: number;
  sessionId?: string;
  model?: string;
  timestamp?: string;
}

export interface WasteSignal {
  type: WasteSignalType;
  severity: WasteSignalSeverity;
  evidence: Record<string, number>;
}

export interface CacheEfficiencyMetrics {
  totalInputTokens: number;
  cacheWriteTokens: number;
  cacheReadTokens: number;
  outputTokens: number;
  cacheReadToWriteRatio: number;
  cacheWriteShareOfInput: number;
  cacheReadShareOfInput: number;
  estimatedWasteClass:
    | 'unknown'
    | 'healthy_cache_reuse'
    | 'write_heavy'
    | 'context_bloat'
    | 'output_heavy';
  wasteSignals: WasteSignal[];
}

export interface ContextBlock {
  id: string;
  label: string;
  text: string;
  byteLength: number;
  approximateTokens: number;
  fingerprint: string;
  volatility: 'static' | 'low' | 'medium' | 'high';
  recommendedTier: CacheTier;
  reason: string;
}

export interface CachePlan {
  algorithm: 'COST_AWARE_BRAIN_V1';
  generatedAt: string;
  blocks: ContextBlock[];
  recommendations: string[];
  warnings: string[];
  nonGoals: string[];
}

export type ContextPackStorageMode = 'inline_text' | 'preview_only' | 'external_store';

export interface ContextPackChunk {
  id: string;
  path: string;
  startLine?: number;
  endLine?: number;
  fingerprint: string;
  approximateTokens: number;
  textPreview?: string;
  text?: string;
}

export interface ContextPackManifest {
  version: 1;
  createdAt: string;
  rootFingerprint: string;
  dimensions: number;
  vectorizer: {
    id: string;
    kind: 'token_hash' | 'hashed_tfidf' | 'semantic_embedding';
    dimensions: number;
    config?: {
      documentCount?: number;
      maxFeaturesPerDocument?: number;
      idfWeights?: number[];
    };
  };
  provenance: import('../context/provenance.js').ContextPackProvenance;
  replay: import('../context/replay_manifest.js').ReplayManifest;
  chunkCount: number;
  approximateTokens: number;
  compressedDatabaseB64: string;
  storageMode: ContextPackStorageMode;
  maxInlineChunkBytes: number;
  chunks: ContextPackChunk[];
}

export interface CostAnalysisResult {
  algorithm_level: string;
  metrics: CacheEfficiencyMetrics;
  recommendations: string[];
  warnings: string[];
  non_goals: string[];
}

export interface CacheLintResult {
  algorithm_level: string;
  issues: Array<{
    severity: 'P1' | 'P2' | 'P3';
    block: string;
    problem: string;
    recommendation: string;
  }>;
  cacheStabilityScore: number;
  warnings: string[];
}

export interface ContextPackBuildResult {
  algorithm_level: string;
  manifest: ContextPackManifest;
  warnings: string[];
}

export interface ContextPackSearchResult {
  algorithm_level: string;
  results: Array<{
    path: string;
    chunk_id: string;
    score: number;
    text: string;
    text_available: boolean;
    source: 'inline' | 'preview' | 'external' | 'missing';
    verified: boolean;
    warning?: string;
  }>;
  storage_mode: ContextPackStorageMode;
  warnings: string[];
}
