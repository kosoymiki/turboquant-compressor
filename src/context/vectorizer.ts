/**
 * ContextVectorizer interface and built-in token-hash implementation.
 *
 * token_hash is the only built-in vectorizer. It is deterministic and local.
 * It is NOT semantic embedding quality. Semantic providers are future adapters.
 */

export type VectorizerKind = 'token_hash' | 'semantic_embedding';

export interface ContextVectorizer {
  readonly id: string;
  readonly kind: VectorizerKind;
  readonly dimensions: number;
  embed(text: string): number[];
}

/**
 * FNV-1a token-hash vectorizer.
 * Deterministic, local, zero-dependency. Not semantic.
 */
export class TokenHashVectorizer implements ContextVectorizer {
  readonly id = 'token_hash_fnv1a';
  readonly kind: VectorizerKind = 'token_hash';
  readonly dimensions: number;

  constructor(dimensions: number) {
    if (dimensions < 1 || dimensions > 8192) {
      throw new Error(`dimensions must be 1–8192, got ${dimensions}`);
    }
    this.dimensions = dimensions;
  }

  embed(text: string): number[] {
    const out = new Array<number>(this.dimensions).fill(0);
    const tokens = text.split(/\s+/).filter(Boolean);

    for (const token of tokens) {
      let h = 2166136261 >>> 0;
      for (let i = 0; i < token.length; i++) {
        h ^= token.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      const idx = (h >>> 0) % this.dimensions;
      const sign = (h & 1) === 0 ? 1 : -1;
      out[idx] = (out[idx] ?? 0) + sign;
    }

    let norm = Math.sqrt(out.reduce((s, x) => s + x * x, 0));
    if (norm === 0) norm = 1;
    return out.map((x) => x / norm);
  }
}

export function createTokenHashVectorizer(dimensions: number): ContextVectorizer {
  return new TokenHashVectorizer(dimensions);
}
