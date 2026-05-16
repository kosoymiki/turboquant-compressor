import type { CacheLintResult } from './types.js';
export declare function lintCacheStability(blocks: Array<{
    label: string;
    text: string;
}>): CacheLintResult;
