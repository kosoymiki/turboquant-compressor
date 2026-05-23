/**
 * ContextVectorizer interface and built-in deterministic local vectorizers.
 *
 * These vectorizers are reproducible lexical baselines. They are not semantic
 * embeddings and should not be described as embedding-quality retrieval.
 */

export type VectorizerKind = 'token_hash' | 'hashed_tfidf' | 'semantic_embedding';

export interface VectorizerSpec {
  id: string;
  kind: VectorizerKind;
  dimensions: number;
  config?: {
    documentCount?: number;
    maxFeaturesPerDocument?: number;
    idfWeights?: number[];
    contextualMode?: 'none' | 'path_chunk_prefix_v1';
  };
}

export interface ContextVectorizer {
  readonly spec: VectorizerSpec;
  readonly id: string;
  readonly kind: VectorizerKind;
  readonly dimensions: number;
  embed(text: string): number[];
}

function splitIdentifierLikeToken(token: string): string[] {
  const parts = token
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z0-9]+)/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/i)
    .filter(Boolean);
  return parts.length > 0 ? parts : [token.toLowerCase()];
}

function normalizeToken(text: string): string[] {
  const raw = text
    .replaceAll('/', ' ')
    .replaceAll('\\', ' ')
    .replaceAll('-', ' ')
    .split(/[^A-Za-z0-9_]+/)
    .filter(Boolean);
  const out: string[] = [];
  for (const token of raw) {
    out.push(token.toLowerCase());
    for (const part of splitIdentifierLikeToken(token)) {
      out.push(part);
    }
  }
  return out;
}

function hash32(str: string, seed = 2166136261): number {
  let h = seed >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function normalizeL2(values: number[]): number[] {
  let norm = Math.sqrt(values.reduce((s, x) => s + x * x, 0));
  if (norm === 0) {
    norm = 1;
  }
  return values.map((x) => x / norm);
}

/**
 * FNV-1a token-hash vectorizer.
 * Deterministic, local, zero-dependency. Not semantic.
 */
export class TokenHashVectorizer implements ContextVectorizer {
  readonly spec: VectorizerSpec;
  readonly id = 'token_hash_fnv1a';
  readonly kind: VectorizerKind = 'token_hash';
  readonly dimensions: number;

  constructor(dimensions: number) {
    if (dimensions < 1 || dimensions > 8192) {
      throw new Error(`dimensions must be 1–8192, got ${dimensions}`);
    }
    this.dimensions = dimensions;
    this.spec = {
      id: this.id,
      kind: this.kind,
      dimensions: this.dimensions,
    };
  }

  embed(text: string): number[] {
    const out = new Array<number>(this.dimensions).fill(0);
    const tokens = text.split(/\s+/).filter(Boolean);

    for (const token of tokens) {
      const h = hash32(token);
      const idx = (h >>> 0) % this.dimensions;
      const sign = (h & 1) === 0 ? 1 : -1;
      out[idx] = (out[idx] ?? 0) + sign;
    }

    return normalizeL2(out);
  }
}

export function createTokenHashVectorizer(dimensions: number): ContextVectorizer {
  return new TokenHashVectorizer(dimensions);
}

type FeatureCountMap = Map<string, number>;

export interface ContextualDocument {
  path: string;
  text: string;
  chunkIndex?: number;
}

function normalizePathTerms(path: string): string {
  return path
    .replace(/\.[^.]+$/u, '')
    .replaceAll('\\', '/')
    .split('/')
    .flatMap((segment) => splitIdentifierLikeToken(segment))
    .join(' ')
    .trim();
}

export function contextualizeDocumentText(document: ContextualDocument): string {
  const pathTerms = normalizePathTerms(document.path);
  const chunkTag = document.chunkIndex === undefined ? '' : ` chunk ${document.chunkIndex}`;
  const stablePrefix = `path ${pathTerms}${chunkTag} ${pathTerms}`.trim();
  return `${stablePrefix}\n${document.text}`;
}

function buildFeatureCounts(text: string, maxFeaturesPerDocument: number): FeatureCountMap {
  const tokens = normalizeToken(text);
  const counts: FeatureCountMap = new Map();
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]!;
    counts.set(token, (counts.get(token) ?? 0) + 1);
    if (i + 1 < tokens.length) {
      const bigram = `${token}::${tokens[i + 1]!}`;
      counts.set(bigram, (counts.get(bigram) ?? 0) + 1);
    }
    if (counts.size >= maxFeaturesPerDocument) {
      break;
    }
  }
  return counts;
}

export class HashedTfidfVectorizer implements ContextVectorizer {
  readonly spec: VectorizerSpec;
  readonly id: string;
  readonly kind: VectorizerKind = 'hashed_tfidf';
  readonly dimensions: number;
  private readonly idfWeights: Float32Array;
  private readonly maxFeaturesPerDocument: number;
  private readonly documentCount: number;
  private readonly contextualMode: 'none' | 'path_chunk_prefix_v1';

  constructor(
    dimensions: number,
    idfWeights: number[],
    documentCount: number,
    maxFeaturesPerDocument = 4096,
    contextualMode: 'none' | 'path_chunk_prefix_v1' = 'none'
  ) {
    if (dimensions < 1 || dimensions > 8192) {
      throw new Error(`dimensions must be 1–8192, got ${dimensions}`);
    }
    if (idfWeights.length !== dimensions) {
      throw new Error(`idfWeights length ${idfWeights.length} must match dimensions ${dimensions}`);
    }
    this.dimensions = dimensions;
    this.id = contextualMode === 'path_chunk_prefix_v1'
      ? 'contextual_identifier_tfidf_hash_v1'
      : 'hashed_tfidf_fnv1a_v1';
    this.idfWeights = new Float32Array(idfWeights);
    this.maxFeaturesPerDocument = maxFeaturesPerDocument;
    this.documentCount = documentCount;
    this.contextualMode = contextualMode;
    this.spec = {
      id: this.id,
      kind: this.kind,
      dimensions: this.dimensions,
      config: {
        documentCount: this.documentCount,
        maxFeaturesPerDocument: this.maxFeaturesPerDocument,
        idfWeights: Array.from(this.idfWeights),
        contextualMode: this.contextualMode,
      },
    };
  }

  embed(text: string): number[] {
    const counts = buildFeatureCounts(text, this.maxFeaturesPerDocument);
    const out = new Array<number>(this.dimensions).fill(0);
    for (const [feature, count] of counts.entries()) {
      const h = hash32(feature);
      const idx = h % this.dimensions;
      const sign = (hash32(feature, 0x9e3779b9) & 1) === 0 ? 1 : -1;
      const tf = 1 + Math.log(count);
      out[idx] = (out[idx] ?? 0) + sign * tf * (this.idfWeights[idx] ?? 1);
    }
    return normalizeL2(out);
  }
}

export function buildHashedTfidfVectorizer(
  dimensions: number,
  documents: string[],
  maxFeaturesPerDocument = 4096,
  contextualMode: 'none' | 'path_chunk_prefix_v1' = 'none'
): ContextVectorizer {
  if (documents.length === 0) {
    throw new Error('documents must not be empty');
  }
  const documentFrequencies = new Uint32Array(dimensions);
  for (const document of documents) {
    const seen = new Set<number>();
    for (const feature of buildFeatureCounts(document, maxFeaturesPerDocument).keys()) {
      seen.add(hash32(feature) % dimensions);
    }
    for (const idx of seen) {
      documentFrequencies[idx]! += 1;
    }
  }

  const n = documents.length;
  const idfWeights = new Array<number>(dimensions);
  for (let i = 0; i < dimensions; i++) {
    idfWeights[i] = Math.log((1 + n) / (1 + documentFrequencies[i]!)) + 1;
  }

  return new HashedTfidfVectorizer(dimensions, idfWeights, n, maxFeaturesPerDocument, contextualMode);
}

export function buildContextualHashedTfidfVectorizer(
  dimensions: number,
  documents: ContextualDocument[],
  maxFeaturesPerDocument = 4096
): ContextVectorizer {
  return buildHashedTfidfVectorizer(
    dimensions,
    documents.map((document) => contextualizeDocumentText(document)),
    maxFeaturesPerDocument,
    'path_chunk_prefix_v1'
  );
}

export function createVectorizerFromSpec(spec: VectorizerSpec): ContextVectorizer {
  if (spec.kind === 'token_hash') {
    return createTokenHashVectorizer(spec.dimensions);
  }
  if (spec.kind === 'hashed_tfidf') {
    const idfWeights = spec.config?.idfWeights;
    const documentCount = spec.config?.documentCount;
    if (!idfWeights || !documentCount) {
      throw new Error('hashed_tfidf vectorizer requires config.idfWeights and config.documentCount');
    }
    return new HashedTfidfVectorizer(
      spec.dimensions,
      idfWeights,
      documentCount,
      spec.config?.maxFeaturesPerDocument ?? 4096,
      spec.config?.contextualMode ?? 'none'
    );
  }
  throw new Error(`Unsupported vectorizer kind: ${spec.kind}`);
}
