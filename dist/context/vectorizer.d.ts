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
export declare class TokenHashVectorizer implements ContextVectorizer {
    readonly id = "token_hash_fnv1a";
    readonly kind: VectorizerKind;
    readonly dimensions: number;
    constructor(dimensions: number);
    embed(text: string): number[];
}
export declare function createTokenHashVectorizer(dimensions: number): ContextVectorizer;
