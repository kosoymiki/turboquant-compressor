import type { CacheTier } from './types.js';
export declare function classifyVolatility(label: string, text: string, volatilityHint?: string): 'static' | 'low' | 'medium' | 'high';
export declare function recommendTier(volatility: 'static' | 'low' | 'medium' | 'high', byteLength?: number): CacheTier;
export declare function tierReason(volatility: 'static' | 'low' | 'medium' | 'high', byteLength?: number): string;
